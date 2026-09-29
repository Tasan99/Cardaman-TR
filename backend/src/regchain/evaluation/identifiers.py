"""Stable obligation identifiers and the matching of predictions to expectations.

An obligation is named by regulation, article, sub-paragraph and a fingerprint of its action
(regulation_id/md.<article>/<clause>/<fingerprint>), so two runs of the same case can be
compared row by row. An expectation may leave the fingerprint out and name the duty by
article, clause and a few words of the action instead; matching is deterministic and greedy
in dataset order.

v0.19 (FIX 4): a label is paired with a duty by the duty's identity, not by one label word in its
action. The v0.18 rule required every label word inside the predicted action, so a duty whose
label word sits in its condition was scored "not extracted" although the pipeline had extracted
it (independent-v1 I02/I03/I04, Kanun 5549 md. 4(1): the action is "bu işlemlerin ... Başkanlığa
bildirilmesi", the label word "şüphe" is in the condition "... şüphe ... bulunması halinde"; in
I02 the pipeline even found the labelled CONFLICT). pair_predictions():

- candidates: the unclaimed predictions of the same article whose clause is compatible (equal; the
  expectation's clause '' matches any; in an article with one paragraph, '' and '(1)' are the same);
- a candidate carries the label when more than half of the label's word stems (folded, five letters:
  Turkish suffixes vary) are in the duty's OWN fields: subject, action, conditions, exceptions,
  deadline. Never the sub-paragraph text, which every sibling duty of the paragraph shares;
- one candidate that carries the label -> CLAUSE_UNIQUE; several -> the best carrier by (stems found,
  every label word in the action as v0.18 required, the polarity the label implies such as
  "uygulayamazlar" = MUST_NOT, stems in the action, earlier duty) -> FINGERPRINT;
- candidates none of which carries the label -> AMBIGUOUS, no candidate -> NO_CANDIDATE (unmatched);
  a label without words (it names no duty) takes the first candidate -> CLAUSE_ORDER, as before.

v0.19 (problem 1 E, source span): the rule above still depends on the model repeating one label word
in the duty's fields. When it paraphrases or cuts the duty, the pipeline's duty goes unscored although
it came from the very sentence the label names (t3 evidence runs: I08 md. 31(1) "belge" - the action
starts at "her türlü ortamdaki kayıtlarını", the sentence reads "istenilecek her türlü bilgi, belge
ve ..."; I12 CONC 7.8.1 "jurisdiction" - the model's subject is "A firm", the word is in "A firm
dealing with a customer who is resident in a different jurisdiction ... must ensure"). So a label
is also located in the regulation text: its sentences are those of its clause's text (the packet's
cases[].source.text, cut at the units the extraction used) that carry the label (same >half rule),
and a duty is located by the sentence it came from (structure.source_span; for rows written before
structure-v1, the sentence holding its action). A second pass, after the pass above and only for the
expectations it left unmatched (so no pair it made can change), pairs a label with the unclaimed
candidate of its clause whose source span overlaps one of the label's sentences -> SOURCE_SPAN.
Several such candidates are separated by the identity key; if nothing separates them -> AMBIGUOUS.
Candidates none of which came from the label's sentence -> OTHER_SENTENCE (unmatched). Without the
source text or the duties' spans nothing can be told apart and the row stays AMBIGUOUS, as before.

Every unmatched row also gets a missing reason (MISSING_REASONS), see missing_reason().
"""
import hashlib
import re

from regchain.pilot.policies import fold

from .metrics import MISSING_REASONS  # noqa: F401 (the reasons are named there, beside the counts)

WORD = re.compile(r'[0-9a-zçğıöşü]+')

# The match basis of a scored row (harness.score_case -> item['match_basis']), matched ones first.
MATCHED_BASES = ('CLAUSE_UNIQUE', 'FINGERPRINT', 'SOURCE_SPAN', 'CLAUSE_ORDER')
UNMATCHED_BASES = ('AMBIGUOUS', 'OTHER_SENTENCE', 'NO_CANDIDATE')
BASES = (*MATCHED_BASES, *UNMATCHED_BASES)
# Why an expectation has no scored row (harness.score_case -> item['missing_reason'], with item['missing_detail']; metrics.MISSING_REASONS):
# EXTRACTION_MISSED   no extraction attempt produced a candidate from the clause: the unit was classified as carrying no
#                     duty (detail CLASSIFIED_<KIND>), had no modal wording (NO_MODAL) so no model saw it, the model returned
#                     no obligation (MODEL_RETURNED_NONE), the call failed (PROVIDER_FAILURE, CONTEXT_BUDGET_EXCEEDED, ...),
#                     or the provision/paragraph is not in the packet (NOT_ANALYSED, CLAUSE_NOT_IN_SOURCE, NOT_RECORDED);
# GROUNDING_REJECTED  the clause's unit ended GROUNDING_REJECTED, SECOND_PASS_UNCERTAIN or INSUFFICIENT_EVIDENCE (the code):
#                     the model proposed a duty and the evidence gate or the second reading refused it, or the model itself
#                     answered INSUFFICIENT_EVIDENCE;
# MATCHING_FAILED     a candidate from the article/clause exists but was not paired: AMBIGUOUS, LABEL_SENTENCE_NOT_EXTRACTED
#                     (the clause's duties all come from other sentences than the label's: the labelled duty itself was not
#                     among them), LABEL_WORDS_NOT_IN_CLAUSE (the label's words are in no sentence of the clause),
#                     CLAIMED_BY_ANOTHER_LABEL (an earlier expectation took it), CLAUSE_MISMATCH (a duty of the clause's text
#                     carries another paragraph number).
GROUNDING_CODES = ('GROUNDING_REJECTED', 'SECOND_PASS_UNCERTAIN', 'INSUFFICIENT_EVIDENCE')
# A sentence ends at . ! ? followed by a capital, an opening bracket or a sub-paragraph number; "md. 4" and "5.000 TL" do not end one.
SENTENCE_END = re.compile(r'[.!?]\s+(?=[A-ZÇĞİÖŞÜ(\[])')
# "(3) " at the start of a unit, after any amendment notes "(Ek: ...)", "(Değişik: ...)".
UNIT_NUMBER = re.compile(r'\s*(?:\((?:Ek|Değişik|Mülga|İptal|Yeniden düzenleme|Mükerrer)[^()]*\)\s*)*\((\d{1,2})\)\s')
STEM = 5
_LETTERS = re.compile(r'[0-9a-zçğıöşüâîû]+')
_PLAIN = str.maketrans('âîû', 'aiu')
# Words that name no duty: function words, and the light verbs left once the infinitive is cut
# ("muhafaza etmek" is named by "muhafaza", "tedbirleri almak" by "tedbir").
_FUNCTION_WORDS = frozenset({'bir', 'bu', 'şu', 'ile', 've', 'veya', 'ya', 'da', 'de', 'ki', 'için', 'gibi', 'olarak', 'et', 'ol', 'al',
                             'yap', 'bulun', 'edil', 'olun', 'ver', 'the', 'and', 'or', 'of', 'to', 'a', 'an', 'in', 'on', 'for', 'with',
                             'by', 'be', 'is', 'are', 'no', 'any', 'its', 'their', 'must', 'shall', 'should', 'may', 'not', 'never'})
_INFINITIVE = re.compile(r'(?<=\w{2})m[ae]k$')
# A label word that makes the label a prohibition ("uygulayamazlar", "açıklayamaz", "yasaktır", "must not") or a
# positive duty ("zorundadır", "shall"); a label without such a word implies no polarity.
_PROHIBITIVE = re.compile(r'^(?:yasak\w*|hiçbir|not|never|prohibit\w*|forbid\w*)$|^\w{3,}m[ae]z(?:l[ae]r)?$')
_MANDATORY = re.compile(r'^(?:zorunda\w*|zorunlu\w*|mecbur\w*|must|shall|required?)$')


def normalize(text: str) -> str:
    return ' '.join(WORD.findall(fold(text or '')))


def fingerprint(action: str, length: int = 12) -> str:
    return hashlib.sha256(normalize(action).encode('utf-8')).hexdigest()[:length]


def clause_id(child_clause: str) -> str:
    """'(2)' -> '2'; '2' -> '2'; 'whole provision' or '' -> ''."""
    match = re.match(r'\s*\(?(\d+)\)?\s*$', child_clause or '')
    return match.group(1) if match else ''


def article_of(source_label: str) -> str:
    """'Yönetmelik 200713012 md. 8' -> '8'; 'CONC 7.3.4' -> '7.3.4'."""
    if ' md. ' in source_label:
        return source_label.split(' md. ', 1)[1].strip()
    return source_label.split(' ', 1)[1].strip() if ' ' in source_label else source_label


def obligation_key(regulation_id: str, article: str, clause: str = '', action: str | None = None) -> str:
    base = f'{regulation_id}/md.{article}/{clause_id(clause) or "*"}'
    return base + (f'/{fingerprint(action)}' if action else '')


def keyword_match(action: str, keywords) -> bool:
    """The v0.18 rule: every label word (phrase) occurs in the normalised action text."""
    text = normalize(action)
    return all(normalize(word) in text for word in keywords)


def words(text) -> list:
    """The folded words of a text or of a list of texts (â, î, û read as a, i, u)."""
    if isinstance(text, (list, tuple)):
        text = ' '.join(str(t) for t in text if t)
    return [w.translate(_PLAIN) for w in _LETTERS.findall(fold(str(text or '')))]


def stems(text) -> list:
    """The content stems of a text in first-seen order: infinitive cut, function words and numbers dropped, five letters."""
    out = []
    for word in words(text):
        word = _INFINITIVE.sub('', word)
        if len(word) < 3 or word in _FUNCTION_WORDS or word.isdigit():
            continue
        if word[:STEM] not in out:
            out.append(word[:STEM])
    return out


def stem_in(stem: str, vocabulary) -> bool:
    """True when a word carries the stem ('şüpheli' carries 'şüphe'; 'cost' carries 'costs', an English plural)."""
    return any(word.startswith(stem) or (len(word) >= 4 and stem.startswith(word)) for word in vocabulary)


def implied_polarity(keywords) -> str | None:
    """'MUST_NOT' when a label word is a prohibition, 'MUST' when it is an explicit duty, else None."""
    found = words(keywords)
    if any(_PROHIBITIVE.match(word) for word in found):
        return 'MUST_NOT'
    return 'MUST' if any(_MANDATORY.match(word) for word in found) else None


def polarity_of(modality) -> str | None:
    """MUST / MUST_NOT of an extracted modality (SHOULD reads as MUST); MAY and a missing modality have none."""
    modality = str(modality or '').upper()
    if modality.startswith('MAY') or not modality:
        return None
    return 'MUST_NOT' if modality.endswith('_NOT') else 'MUST'


def duty_fields(prediction: dict) -> list:
    """The texts that are the predicted duty itself: subject, action, conditions, exceptions, deadline.

    Not the sub-paragraph it came from: every sibling duty of a paragraph shares that text, so a label
    word in it would name all of them. A prediction row written before v0.19 has only subject and action.
    """
    parts = [prediction.get('subject'), prediction.get('action'), prediction.get('deadline')]
    for key in ('conditions', 'exceptions'):
        value = prediction.get(key)
        parts.extend(value if isinstance(value, (list, tuple)) else [value])
    return [str(p) for p in parts if p]


def duty_fingerprint(article: str, clause: str = '', modality: str | None = None, action: str | None = None, length: int = 12) -> str:
    """A stable identity of a duty: '<article>/<clause or *>/<modality or ?>/<hash of its sorted action stems>'.

    Case, punctuation, word order and Turkish suffixes do not change it, so the same duty extracted twice
    in slightly different words keeps its fingerprint across runs.
    """
    body = ' '.join(sorted(stems(action or '')))
    return f'{article}/{clause_id(clause) or "*"}/{modality or "?"}/{hashlib.sha256(body.encode("utf-8")).hexdigest()[:length]}'


def obligation_fingerprint(regulation_id: str, article: str, clause: str = '', action: str | None = None, obj: str | None = None,
                           span_text: str | None = None, length: int = 10) -> str:
    """The stable identity of a predicted duty across runs:
    '<regulation>/md.<article>/<clause or *>/<hash of its sorted action and object stems>/<hash of its source sentence or nospan>'.

    Word order, case, punctuation and Turkish suffixes do not change the stem part; the span part names the
    sentence the duty came from, so two duties of one clause worded alike by the model stay apart.
    """
    body = ' '.join(sorted(set(stems(action or '')) | set(stems(obj or ''))))
    span = hashlib.sha256(normalize(span_text).encode('utf-8')).hexdigest()[:length] if span_text and normalize(span_text) else 'nospan'
    return f'{regulation_id}/md.{article}/{clause_id(clause) or "*"}/{hashlib.sha256(body.encode("utf-8")).hexdigest()[:length]}/{span}'


# ------------------------------------------------------------------ where a label and a duty sit in the regulation text

def source_index(payload) -> dict:
    """{article: {'text', 'units'}} of every provision of a packet; {} for a payload without cases.

    'units' are the pieces the extraction read (the packet's cases[].classification: offset, chars, kind),
    each with its paragraph number and what extraction made of it (cases[].units: reason, candidates; a
    provision read whole records its outcome on the case itself). A unit that is not in cases[].units was
    never sent to a model (no modal wording). Offsets are into cases[].source.text.
    """
    index = {}
    for case in (payload or {}).get('cases') or []:
        source = case.get('source') or {}
        article, text = article_of(source.get('printed_label') or ''), source.get('text') or ''
        if article and article not in index:
            index[article] = {'text': text, 'units': units_of(case, text)}
    return index


def units_of(case: dict, text: str) -> list:
    classified = [c for c in case.get('classification') or [] if isinstance(c, dict)] or [{'offset': 0, 'chars': len(text), 'kind': None}]
    worked = case.get('units')
    if not isinstance(worked, list):
        output = case.get('output') or {}
        worked = [{'offset': 0, 'chars': len(text), 'reason': case.get('reason'), 'candidates': len(output.get('obligations') or [])}]
    units = []
    for entry in classified:
        start = int(entry.get('offset') or 0)
        end = start + int(entry.get('chars') or 0)
        number = UNIT_NUMBER.match(text, start)
        # The unit's own outcome, else the one that covered it (a provision re-read whole when no unit was sent).
        covering = [w for w in worked if int(w.get('offset') or 0) < end and start < int(w.get('offset') or 0) + int(w.get('chars') or 0)]
        outcome = next((w for w in covering if int(w.get('offset') or 0) == start), covering[0] if covering else None)
        units.append({'start': start, 'end': end, 'clause': number.group(1) if number and number.start(1) < end else '',
                      'kind': entry.get('kind'), 'worked': outcome is not None, 'reason': (outcome or {}).get('reason'),
                      'candidates': int((outcome or {}).get('candidates') or 0)})
    return units


def clause_ranges(source: dict | None, clause: str) -> list:
    """[(start, end)] of a paragraph in the provision text: '' is the whole text; '(1)' of an unnumbered provision is too."""
    if not source:
        return []
    text, number = source['text'], clause_id(clause)
    if not number:
        return [(0, len(text))]
    hits = [(u['start'], u['end']) for u in source['units'] if u['clause'] == number]
    if not hits and number == '1' and all(not u['clause'] for u in source['units']):
        hits = [(0, len(text))]
    return hits


def sentence_spans(text: str, lo: int, hi: int) -> list:
    """[(start, end)] of the sentences of text[lo:hi], each ending with its full stop."""
    spans, start = [], lo
    for mark in SENTENCE_END.finditer(text, lo, hi):
        spans.append((start, mark.start() + 1))
        start = mark.end()
    spans.append((start, hi))
    return [(a, a + len(text[a:b].rstrip())) for a, b in spans if text[a:b].strip()]


def carries_label(label: list, text: str) -> bool:
    return bool(label) and sum(1 for s in label if stem_in(s, words(text))) * 2 > len(label)


def label_sentences(item, source: dict | None) -> list:
    """[(start, end)] of the sentences of the expectation's clause text that carry more than half of its stems."""
    label = stems(item.action_keywords)
    return [(a, b) for lo, hi in clause_ranges(source, item.clause) for a, b in sentence_spans(source['text'], lo, hi)
            if carries_label(label, source['text'][a:b])] if label and source else []


def locate(text: str, lo: int, hi: int, needle: str):
    """(start, end) of a duty's words in text[lo:hi]: verbatim, else word by word across spacing and punctuation,
    else the longest leading run of at least three of its words; None when not found."""
    needle = ' '.join(str(needle or '').split())
    if not needle:
        return None
    at = text.find(needle, lo, hi)
    if at >= 0:
        return at, at + len(needle)
    tokens = re.findall(r'\w+', needle)
    for size in range(len(tokens), min(3, len(tokens)) - 1, -1):
        found = re.compile(r'\W+'.join(re.escape(t) for t in tokens[:size]), re.I).search(text, lo, hi)
        if found:
            return found.start(), found.end()
    return None


def sentence_around(text: str, lo: int, hi: int, start: int, end: int) -> tuple:
    """(start, end) of the sentence(s) of text[lo:hi] holding text[start:end]."""
    first, last = lo, hi
    for mark in SENTENCE_END.finditer(text, lo, hi):
        if mark.end() <= start:
            first = mark.end()
        elif mark.start() >= end - 1:
            last = mark.start() + 1
            break
    return first, first + len(text[first:last].rstrip())


def duty_span(row: dict, source: dict | None) -> dict | None:
    """{'start', 'end', 'origin'}: the sentence of the provision text the packet row's duty came from (offsets into
    cases[].source.text). STRUCTURE = structure-v1's source_span (v0.19 packets); ACTION = the sentence holding the
    duty's action, located in its unit (rows written before structure-v1); None when neither is found.

    v0.19 t6: a structure span that holds more than one sentence names the duty by the sentence of its marker (the
    modal span, else the end of its action). structure-v1 runs its span on to the next full stop when the action ends
    with its own ("... alınır. Derneğin adı, ... teyit edilir.": 30 of the 316 v0.19 rows), and an action copied over
    a full stop ends in the sentence that states its duty; the other sentence is never this duty's (OTHER_SENTENCE)."""
    if not source:
        return None
    text = source['text']
    structure = row.get('structure') if isinstance(row.get('structure'), dict) else {}
    span = structure.get('source_span') if isinstance(structure.get('source_span'), dict) else {}
    offset = structure.get('unit_offset')
    offset = offset if isinstance(offset, int) else int((row.get('multipart') or {}).get('unit_offset') or 0)
    if isinstance(span.get('start'), int) and isinstance(span.get('end'), int) and span['start'] < span['end']:
        start, end = offset + span['start'], offset + span['end']
        if text[start:end] == span.get('text', text[start:end]):
            start, end = marker_sentence(text, start, end, structure, offset)
            return {'start': start, 'end': end, 'origin': 'STRUCTURE'}
    candidate = row.get('candidate') or {}
    action = candidate.get('required_action') or candidate.get('prohibited_action') or ''
    unit = next(((u['start'], u['end']) for u in source['units'] if u['start'] == offset), (0, len(text)))
    for lo, hi in dict.fromkeys([unit, (0, len(text))]):
        found = locate(text, lo, hi, action)
        if found:
            start, end = sentence_around(text, lo, hi, *found)
            return {'start': start, 'end': end, 'origin': 'ACTION'}
    return None


def marker_sentence(text: str, start: int, end: int, structure: dict, offset: int) -> tuple:
    """(start, end) of the sentence of text[start:end] that holds the duty's marker (structure 'modal'), else the end of
    its action; the span itself when it is one sentence or neither is inside it."""
    sentences = sentence_spans(text, start, end)
    if len(sentences) < 2:
        return start, end
    parts = [structure.get(k) if isinstance(structure.get(k), dict) else {} for k in ('modal', 'action')]
    at = next((offset + (p['start'] if k == 'modal' else p['end'] - 1) for k, p in zip(('modal', 'action'), parts)
               if isinstance(p.get('start'), int) and isinstance(p.get('end'), int) and p['start'] < p['end']), None)
    return next(((a, b) for a, b in sentences if at is not None and a <= at < b), (start, end))


def overlaps(span, sentence) -> bool:
    return bool(span) and span['start'] < sentence[1] and sentence[0] < span['end']


def single_paragraph_articles(expected, predictions) -> set:
    """The articles that nothing in the case (label or prediction) names a paragraph of other than '' or 1."""
    clauses = {}
    for item in expected:
        clauses.setdefault(item.article, set()).add(clause_id(item.clause))
    for prediction in predictions:
        clauses.setdefault(prediction.get('article'), set()).add(str(prediction.get('clause') or ''))
    return {article for article, seen in clauses.items() if seen <= {'', '1'}}


def compatible(item, prediction: dict, single_paragraph: bool = False) -> bool:
    """Same article, and a clause naming the same paragraph (the label's '' names any)."""
    if prediction.get('article') != item.article:
        return False
    wanted, got = clause_id(item.clause), str(prediction.get('clause') or '')
    return not wanted or wanted == got or (single_paragraph and {wanted, got} == {'', '1'})


def identity_score(item, prediction: dict, index: int = 0) -> dict:
    """How far a candidate carries the label's identity; integers only, a larger 'key' is the better candidate.

    'carries': more than half of the label's stems are in the duty's own fields (a label that is only a
    polarity, "must not", needs that polarity). Half is not enough: "destroy records" is not the duty
    "retain records", which shares only the object (a v0.17 test pins that such a duty stays unclaimed).
    """
    label = stems(item.action_keywords)
    own = words(duty_fields(prediction))
    action = words(prediction.get('action') or '')
    found = sum(1 for s in label if stem_in(s, own))
    implied, polarity = implied_polarity(item.action_keywords), polarity_of(prediction.get('modality'))
    agree = 0 if implied is None or polarity is None else (1 if implied == polarity else -1)
    exact = int(bool(item.action_keywords) and keyword_match(prediction.get('action') or '', item.action_keywords))
    return {'stems': len(label), 'stems_found': found, 'polarity': agree, 'carries': found * 2 > len(label) if label else agree > 0,
            'key': (found, exact, agree, sum(1 for s in label if stem_in(s, action)), -index)}


def pair_predictions(expected, predictions, sources=None) -> tuple:
    """([one match per expectation, in dataset order], [the predictions no expectation claimed]).

    A match is {'expected', 'prediction' (or None), 'basis' (BASES), 'candidates', 'stems', 'stems_found',
    'polarity', 'label_sentences', 'from_sentence', 'missing_reason', 'missing_detail'}: integers and names only.
    Each prediction is claimed at most once, in dataset order. A label that names no duty (no words, no polarity)
    takes the first candidate (CLAUSE_ORDER), as every label did before v0.19.

    ``sources`` is source_index(payload); the predictions carry 'source_span' (harness.prediction_rows). Pass 1 is
    the identity rule; pass 2 (SOURCE_SPAN) only reads the expectations pass 1 left unmatched and the predictions it
    left unclaimed, so a pair made by the duty's own words is never changed by the sentence rule. Without ``sources``
    (no provision text to locate a label in) pass 2 does not run and the result is pass 1's, as before.
    """
    expected, predictions = list(expected), list(predictions)
    single = single_paragraph_articles(expected, predictions)
    taken, matches = set(), []
    for item in expected:
        candidates = [i for i, p in enumerate(predictions) if i not in taken and compatible(item, p, item.article in single)]
        scores = {i: identity_score(item, predictions[i], i) for i in candidates}
        named = bool(stems(item.action_keywords)) or implied_polarity(item.action_keywords) is not None
        eligible = [i for i in candidates if scores[i]['carries'] or not named]
        chosen, basis = None, ('NO_CANDIDATE' if not candidates else 'AMBIGUOUS')
        if eligible and len(candidates) == 1:
            chosen, basis = eligible[0], 'CLAUSE_UNIQUE'
        elif eligible and not named:
            chosen, basis = eligible[0], 'CLAUSE_ORDER'
        elif eligible:
            chosen, basis = max(eligible, key=lambda i: scores[i]['key']), 'FINGERPRINT'
        score = scores.get(chosen) or {}
        matches.append({'expected': item, 'prediction': predictions[chosen] if chosen is not None else None, 'basis': basis,
                        'candidates': len(candidates), 'stems': len(stems(item.action_keywords)),
                        'stems_found': score.get('stems_found'), 'polarity': score.get('polarity'), 'label_sentences': None,
                        'from_sentence': None, 'missing_reason': None, 'missing_detail': None})
        if chosen is not None:
            taken.add(chosen)
    for match in matches if sources else ():
        if match['prediction'] is None and match['basis'] == 'AMBIGUOUS' and stems(match['expected'].action_keywords):
            chosen = by_source_span(match, predictions, taken, sources, single)
            if chosen is not None:
                taken.add(chosen)
    for match in matches:
        if match['prediction'] is None:
            match['missing_reason'], match['missing_detail'] = missing_reason(match, predictions, sources, single)
    return matches, [prediction for index, prediction in enumerate(predictions) if index not in taken]


def by_source_span(match, predictions, taken, sources, single):
    """Pass 2 for one unmatched expectation: the unclaimed candidate of its clause that came from the label's sentence.

    Several such candidates are told apart by the identity key (label stems in their own fields, the label's
    every word in the action, polarity, stems in the action); a tie is AMBIGUOUS. Updates the match in place and
    returns the claimed prediction's index (or None).
    """
    item = match['expected']
    candidates = [i for i, p in enumerate(predictions) if i not in taken and compatible(item, p, item.article in single)]
    source = sources.get(item.article)
    sentences = label_sentences(item, source)
    spans = {i: predictions[i].get('source_span') for i in candidates}
    found = [i for i in candidates if any(overlaps(spans[i], s) for s in sentences)]
    match.update(candidates=len(candidates), label_sentences=len(sentences), from_sentence=len(found))
    if not candidates:
        match['basis'] = 'NO_CANDIDATE'
        return None
    keys = {i: identity_score(item, predictions[i], i) for i in found}
    best = max((keys[i]['key'][:-1] for i in found), default=None)
    leaders = [i for i in found if keys[i]['key'][:-1] == best]
    if len(leaders) == 1:
        chosen = leaders[0]
        match.update(prediction=predictions[chosen], basis='SOURCE_SPAN', stems_found=keys[chosen]['stems_found'],
                     polarity=keys[chosen]['polarity'])
        return chosen
    # Several candidates from the label's sentence and nothing of their own separates them, or a candidate that could not
    # be located (it may be the labelled one): AMBIGUOUS, as before. Every candidate located elsewhere: OTHER_SENTENCE.
    known = source is not None and all(spans.values())
    match['basis'] = 'AMBIGUOUS' if found or not known else 'OTHER_SENTENCE'
    return None


def missing_reason(match, predictions, sources, single) -> tuple:
    """(MISSING_REASONS member, detail code) of an expectation that has no scored row.

    A candidate from the article/clause (compatible clause, or a located sentence inside the clause's text) makes it
    MATCHING_FAILED with the reason it was not paired; else the clause's units say what extraction did with them.
    """
    item = match['expected']
    source = (sources or {}).get(item.article)
    ranges = clause_ranges(source, item.clause)
    within = lambda p: p.get('article') == item.article and any(
        overlaps(p.get('source_span'), r) and r[0] <= p['source_span']['start'] and p['source_span']['end'] <= r[1] for r in ranges)
    compatible_ones = [p for p in predictions if compatible(item, p, item.article in single)]
    if compatible_ones:
        if match['basis'] == 'NO_CANDIDATE':
            return 'MATCHING_FAILED', 'CLAIMED_BY_ANOTHER_LABEL'
        if match['basis'] == 'OTHER_SENTENCE':
            return 'MATCHING_FAILED', 'LABEL_SENTENCE_NOT_EXTRACTED' if match['label_sentences'] else 'LABEL_WORDS_NOT_IN_CLAUSE'
        return 'MATCHING_FAILED', 'AMBIGUOUS'
    if any(within(p) for p in predictions):
        return 'MATCHING_FAILED', 'CLAUSE_MISMATCH'
    if sources is None or not sources:
        return 'EXTRACTION_MISSED', 'NOT_RECORDED'
    if source is None:
        return 'EXTRACTION_MISSED', 'NOT_ANALYSED'
    if not ranges:
        return 'EXTRACTION_MISSED', 'CLAUSE_NOT_IN_SOURCE'
    units = [u for u in source['units'] if any(u['start'] < b and a < u['end'] for a, b in ranges)]
    sentences = label_sentences(item, source)
    narrowed = [u for u in units if any(u['start'] < b and a < u['end'] for a, b in sentences)]
    outcomes = [unit_outcome(u) for u in (narrowed or units)]
    if not outcomes:
        return 'EXTRACTION_MISSED', 'CLAUSE_NOT_IN_SOURCE'
    # The furthest stage any unit of the clause reached names the loss.
    return min(outcomes, key=lambda o: {'MATCHING_FAILED': 0, 'GROUNDING_REJECTED': 1, 'EXTRACTION_MISSED': 2}[o[0]])


def unit_outcome(unit: dict) -> tuple:
    """What extraction made of one unit that left no candidate behind, as (reason, detail)."""
    reason = str(unit.get('reason') or '')
    if not unit['worked']:
        return 'EXTRACTION_MISSED', 'NO_MODAL'
    if reason in GROUNDING_CODES:
        return 'GROUNDING_REJECTED', reason
    if reason.startswith('CLASSIFIED_'):
        return 'EXTRACTION_MISSED', reason
    if unit['candidates']:
        return 'MATCHING_FAILED', 'CLAUSE_MISMATCH'
    if reason in ('', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 'NO_EXPLICIT_OBLIGATION', 'EXTRACTED'):
        return 'EXTRACTION_MISSED', 'MODEL_RETURNED_NONE'
    return 'EXTRACTION_MISSED', reason


def match_predictions(expected, predictions):
    """[(expectation, prediction or None)] in dataset order and the predictions no expectation claimed (pair_predictions)."""
    matches, extras = pair_predictions(expected, predictions)
    return [(match['expected'], match['prediction']) for match in matches], extras


def basis_counts(scored) -> dict:
    """{'bases': {match basis: rows}} over scored rows, in BASES order; a row scored before v0.19 counts as NOT_RECORDED."""
    counts = {}
    for item in scored:
        basis = item.get('match_basis') or 'NOT_RECORDED'
        counts[basis] = counts.get(basis, 0) + 1
    return {'bases': {basis: counts[basis] for basis in [*BASES, 'NOT_RECORDED'] if basis in counts}, 'rows': len(scored)}

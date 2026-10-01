"""The parts of one extracted duty, located in its source unit: a side-car, not a contract change.

The judge used to read a duty as six strings (subject, modality, one action, conditions,
exceptions). What made a policy weaker than the regulation was usually somewhere else: the
deadline sat inside the action or a condition ("... en geç on iş günü içinde ..."), the
retention period inside the action ("sekiz yıl süre ile"), the trigger amount in a list item
("yüzseksenbeşbin TL ... veya üzerinde"), the enumerated contents of a record after the
colon (md. 42(2)), and the risk condition of md. 26(2) ("... riskinin oluşabileceği
durumlarda") nowhere, because grounding.TR_CONDITIONS has no "durumlarda". structure_of()
finds these deterministically; duty_payload() hands them to the judge beside the legacy keys.

Nothing here changes Candidate or ExtractionOutput (extra='forbid', hashed into the
extraction prompt hash and into every obligation id). Offsets are relative to the unit text
and every span satisfies unit_text[start:end] == text. The record is canonical evidence (it
goes into packets and the extraction review JSON): numbers are ints, or decimal strings when
not whole, never floats.
"""
import hashlib
import re

from .grounding import CONDITIONS, EXCEPTIONS, MODAL, TR_CONDITIONS, TR_EXCEPTIONS, TR_LETTERS, TR_STANDALONE, modality, \
    qualifier_spans, turkish, turkish_modal
from .quantities import DAYS, evidence_safe, parse_quantities, period_phrases
from .spans import LEADING_NOTES, LETTER, LIST_MARKER as LIST_ITEM, TR_CONDITION_EXTRA, TR_SENTENCE, WORD, action_of, after_notes, \
    expand_action, turkish_anchor, word_end

VERSION = 'structure-v1'
EN_SENTENCE = re.compile(r'[.!?]\s+(?=[A-Z(\[])')
NOTE = re.compile(r'\((?:Ek|Değişik|Mülga|İptal|Yeniden düzenleme|Mükerrer)[^()]*\)\s*')
INFINITIVE_AND = re.compile(r'[%s]+(?:mek|mak)\s+(?:ve|veya)\s+' % TR_LETTERS)
INFINITIVE = re.compile(r'(?<![%s])[%s]+(?:mek|mak)(?![%s])' % (TR_LETTERS, TR_LETTERS, TR_LETTERS))
# "tutar" is an amount (md. 5(1) "tutar gözetmeksizin"), not "keeps"; it is left out on purpose.
KEEPING = re.compile(r'(?<![%s])(?:sakla|muhafaza|tutul|tutmak|arşivle)|(?i:\b(?:retain|keep|kept|maintain|store|preserv))' % TR_LETTERS)
LIGHT_VERBS = frozenset({'etmek', 'olmak', 'kılmak', 'bulunmak', 'yapmak', 'eylemek'})
DEADLINE_LEADS = ('en geç', 'within', 'no later than', 'not later than')
DEADLINE_TRAILS = ('içinde', 'içerisinde', 'zarfında')
DURATION_TRAILS = ('süre ile', 'süreyle', 'süresince', 'boyunca')
ITIBAREN = re.compile(r'(?<![%s])itibaren(?![%s])[\s,]*$' % (TR_LETTERS, TR_LETTERS))
EN_ANCHOR = re.compile(r'\s*,?\s*(?P<anchor>(?:of|from|after|following)\s+(?:the\s+)?(?:date|day|end|time|receipt|occurrence|discovery|start|'
                       r'close|termination|completion|detection|identification)\b[^,.;]{0,80})', re.I)
PLURAL_OBJECT = re.compile(r'(?<![%s])[%s]+(?:ları|leri)(?![%s])' % (TR_LETTERS, TR_LETTERS, TR_LETTERS))
LIST_CLOSE = re.compile(r'(?:^|\s)(?:[a-zçğıöşü]|\d{1,2})\)$')
TIMING_KINDS = {'IMMEDIATE': 'IMMEDIATE', 'ASAP': 'IMMEDIATE', 'BEFORE': 'BEFORE', 'PERIOD_END': 'PERIOD_END'}
SENTENCE_LIMIT = 600
ITEM_LIMIT = 12


def _get(candidate, key, default=None):
    return candidate.get(key, default) if isinstance(candidate, dict) else getattr(candidate, key, default)


def _span(text: str, start: int, end: int, **extra) -> dict:
    return {'text': text[start:end], 'start': start, 'end': end, **extra}


def _sentence_bounds(text: str, lo: int, hi: int, tr: bool) -> tuple[int, int]:
    """(start, end) of the sentence(s) holding text[lo:hi]; end includes the closing mark."""
    pattern = TR_SENTENCE if tr else EN_SENTENCE
    start, end = 0, len(text)
    for boundary in pattern.finditer(text):
        if boundary.end() <= lo:
            start = boundary.end()
        elif boundary.start() >= hi:
            end = boundary.start() + 1
            break
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _closes_bracket(text: str, index: int) -> bool:
    """A ")" that closes a bracket, not the ")" of a list marker ("b) ...")."""
    return text[index] == ')' and not LIST_CLOSE.search(text, max(0, index - 3), index + 1)


def _open_bracket(text: str, lo: int, pos: int) -> bool:
    depth = 0
    for index in range(pos - 1, lo - 1, -1):
        if _closes_bracket(text, index):
            depth += 1
        elif text[index] == '(':
            if depth == 0:
                return True
            depth -= 1
    return False


def _last_boundary(text: str, lo: int, pos: int) -> int:
    """After the last comma/semicolon/colon before `pos` outside brackets, or after the bracket
    `pos` sits in; one backward pass (a long unit has hundreds of commas)."""
    depth = 0
    for index in range(pos - 1, lo - 1, -1):
        char = text[index]
        if _closes_bracket(text, index):
            depth += 1
        elif char == '(':
            if depth == 0:
                return index + 1
            depth -= 1
        elif depth == 0 and char in ',;:':
            return index + 1
    return lo


def _clause_start(text: str, lo: int, pos: int, subject_end: int | None) -> int:
    """Where the clause that ends at `pos` begins: after the last comma/semicolon/colon outside
    brackets, list item marker, "…mek ve" predicate, duty marker or the subject, whichever is last."""
    starts = [lo, _last_boundary(text, lo, pos)]
    if subject_end is not None and lo <= subject_end <= pos:
        starts.append(subject_end)
    stop = min(len(text), pos + 40)
    for pattern in (LIST_ITEM, INFINITIVE_AND, MODAL):
        starts += [m.end() for m in pattern.finditer(text, lo, stop) if m.end() <= pos]
    start = max(starts)
    return start + len(re.match(r'[\s,;:]*(?:(?:ve|veya)\s+)?', text[start:pos]).group())


def _marker_clauses(text: str, lo: int, hi: int, patterns, subject_end: int | None) -> list[tuple[int, int]]:
    found = []
    for pattern in patterns:
        for match in pattern.finditer(text, lo):
            if match.end() > hi:
                break
            if match.group().startswith('dışında') and text[max(0, match.start() - 5):match.start()] == 'yurt ':
                continue                                   # "yurt dışında" is abroad, not an exception
            start = _clause_start(text, lo, match.start(), subject_end)
            if len(text[start:match.end()].split()) >= 2:
                found.append((start, match.end()))
    return sorted(set(found))


def _qualifiers(text, values, s0, s1, tr, patterns, english, subject_end, flags):
    """Model spans (origin 'model') plus marker clauses (origin 'marker') not already carried."""
    spans = []
    for value in values or []:
        at = text.find(value, s0)
        at = at if at >= 0 else text.find(value)
        if at < 0:
            flags.add('QUALIFIER_OUTSIDE_UNIT')
            continue
        spans.append(_span(text, at, at + len(value), origin='model'))
    if tr:
        clauses = _marker_clauses(text, s0, s1, patterns, subject_end)
    else:
        sentence = text[s0:s1]
        clauses = [(s0 + sentence.find(span), s0 + sentence.find(span) + len(span)) for span in qualifier_spans(sentence, english)
                   if span and sentence.find(span) >= 0]
    for start, end in clauses:
        clause = text[start:end]
        if any(clause in s['text'] or s['text'] in clause for s in spans):
            continue
        spans.append(_span(text, start, end, origin='marker'))
    return spans


def _anchor(text: str, s0: int, phrase_start: int, phrase_end: int, s1: int, subject_end: int | None, tr: bool) -> str | None:
    """The event a period runs from: "işleme ilişkin şüphenin oluştuğu tarihten itibaren"."""
    if tr:
        found = ITIBAREN.search(text, s0, phrase_start)
        if not found:
            return None
        end = found.start() + len('itibaren')
        start = _clause_start(text, s0, found.start(), subject_end)
        # "... kayıtları son işlem tarihinden itibaren": the object before the date is not the anchor.
        base = start
        for word in PLURAL_OBJECT.finditer(text, base, found.start()):
            if text[word.end():found.start()].strip():
                start = word.end()
        start += len(text[start:end]) - len(text[start:end].lstrip())
        return text[start:end] if len(text[start:end].split()) >= 2 else None
    found = EN_ANCHOR.match(text, phrase_end, s1)
    return found.group('anchor').strip() if found else None


def _timing(text, s0, s1, tr, keeping, subject_end):
    """(deadline entries, threshold entries) of the governing sentence."""
    sentence = text[s0:s1]
    deadlines, thresholds, taken = [], [], []
    for q in parse_quantities(sentence):
        lo, hi = s0 + q['phrase_start'], s0 + q['phrase_end']
        lead, trail, sign, side = q['lead'] or '', q['trail'], q['comparator'], q['comparator_side']
        base = {'amount': q['amount'], 'unit': q['unit']}
        if q['unit_class'] == 'duration':
            if lead in DEADLINE_LEADS or trail in DEADLINE_TRAILS or (sign in ('<=', '<') and side == 'before'):
                kind, direction = 'DEADLINE', 'max'
            elif (trail in DURATION_TRAILS or lead.startswith('for')) and keeping:
                kind, direction = 'DURATION', 'min'
            elif sign == '>=' and side == 'before':
                kind, direction = 'DURATION', 'min'
            elif sign in ('>', '>=') and side == 'after':
                thresholds.append(_span(text, lo, hi, **base, comparator=sign, direction='floor', days=q['days'],
                                        parenthetical=q['parenthetical']))
                continue
            else:
                continue
            deadlines.append(_span(text, lo, hi, kind=kind, **base, days=q['days'], direction=direction,
                                   anchor=_anchor(text, s0, lo, hi, s1, subject_end, tr)))
            taken.append((lo, hi))
        elif sign:
            direction = 'floor' if sign in ('>=', '>') else 'max' if side == 'before' else 'min'
            thresholds.append(_span(text, lo, hi, **base, comparator=sign, direction=direction, parenthetical=q['parenthetical']))
    for phrase in period_phrases(sentence):
        kind = TIMING_KINDS.get(phrase['kind'])
        lo, hi = s0 + phrase['start'], s0 + phrase['end']
        if kind is None or any(a < hi and lo < b for a, b in taken):
            continue
        deadlines.append(_span(text, lo, hi, kind=kind, amount=None, unit=None, days=phrase['days'], direction='max', anchor=None))
    deadlines.sort(key=lambda entry: entry['start'])
    return deadlines, thresholds


CONJUNCTION_ONLY = re.compile(r'(?:ve|veya|ile|ya\s+da|yahut|and|or|,)', re.I)
REFERENCE_NOUN = re.compile(r'(?:alt\s+)?(?:bent|fıkra|madde|paragraf|sub-?paragraph|paragraph|point)', re.I)


def _items(text: str, lo: int, hi: int, action_start: int | None, after_marker: bool) -> list[str]:
    """The a) ... h) items of a list in text[lo:hi], each without its marker and notes."""
    markers = list(LIST_ITEM.finditer(text, lo, hi))
    if len(markers) < 2:
        return []
    items = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else hi
        if index + 1 == len(markers) and not after_marker:
            # The last case of a list before the duty ends at its clause, not at the duty.
            cut = [end]
            if action_start is not None and marker.end() < action_start:
                cut.append(action_start)
            comma = next((m.start() for m in re.finditer(',', text[marker.end():end])
                          if not _open_bracket(text, marker.end(), marker.end() + m.start())), None)
            if comma is not None:
                cut.append(marker.end() + comma)
            end = min(cut)
        item = NOTE.sub('', text[marker.end():end]).strip().rstrip(',;').rstrip('.').strip()
        # v0.19 round 5 (md. 24/A(2) "birinci fıkranın (a) ve (b) bentlerinde belirtilen bilgilere"): markers joined by a
        # conjunction and followed by the word for a list point are a cross-reference, not an enumeration of this duty.
        if CONJUNCTION_ONLY.fullmatch(item) or (index + 1 == len(markers) and REFERENCE_NOUN.match(text, marker.end())):
            return []
        if item:
            items.append(item)
    return items if len(items) >= 2 else []


def _object(action: str, modal, tr: bool) -> str | None:
    body = action.rstrip(' .;:')
    if tr and modal is not None:
        marker = modal.group()
        if body.endswith(marker):
            body = body[:-len(marker)].rstrip()
            if not TR_STANDALONE.fullmatch(marker):
                return body or None                          # "basitleştirilmiş tedbirleri uygulayamazlar"
        words = body.split()
        drop = 2 if len(words) >= 2 and words[-1] in LIGHT_VERBS else 1
        return ' '.join(words[:-drop]) or None               # "... kimliklerini tespit etmek"
    words = body.split()
    return ' '.join(words[1:]) or None                       # "retain records" -> "records"


def _english_anchor(text: str, action: str, mode: str, subject: str, matches):
    aligned = [(m, text.find(action, m.end())) for m in matches
               if not turkish_modal(m.group()) and modality(m.group()) == mode and subject in text[:m.start()]]
    aligned = [(m, pos) for m, pos in aligned if pos >= 0]
    if not aligned:
        return None
    match, pos = min(aligned, key=lambda pair: pair[1] - pair[0].end())
    return pos, match


def failed(unit_offset: int = 0, sha: str | None = None) -> dict:
    return {'version': VERSION, 'unit_offset': unit_offset, 'subject': None, 'modal': None, 'action': None, 'object': None,
            'conditions': [], 'exceptions': [], 'deadline': [], 'threshold': [], 'enumeration': [], 'source_span': None,
            'source_sha256': sha, 'flags': ['STRUCTURE_FAILED']}


def structure_of(unit_text: str, candidate: dict, unit_offset: int = 0, siblings=()) -> dict:
    """The located parts of one duty in its unit; deterministic and never raises.

    `siblings` (the other candidates' actions of the same unit) only keeps a still-truncated
    action from being expanded into a sibling duty, as spans.repair_actions does.
    """
    sha = hashlib.sha256((unit_text or '').encode('utf-8')).hexdigest()
    try:
        structure = evidence_safe(_structure(unit_text or '', candidate, unit_offset, siblings, sha))
    except Exception:  # noqa: BLE001 - a side-car must never break an analysis
        return failed(unit_offset, sha)
    return structure if _spans_hold(unit_text or '', structure) else failed(unit_offset, sha)


def _spans_hold(text: str, structure: dict) -> bool:
    spans = [structure[k] for k in ('subject', 'modal', 'action', 'source_span') if structure.get(k)]
    spans += [s for k in ('conditions', 'exceptions', 'deadline', 'threshold') for s in structure[k]]
    return all(text[s['start']:s['end']] == s['text'] for s in spans)


def _structure(text: str, candidate, unit_offset: int, siblings, sha: str) -> dict:
    from .classify import duty_modals
    tr = turkish(text)
    matches = duty_modals(text)
    mode = _get(candidate, 'modality') or ''
    subject = _get(candidate, 'subject') or ''
    original = action_of(candidate) or ''
    flags, action, repair = set(), original, None
    if tr and original:
        expanded, note = expand_action(text, candidate, [s for s in siblings if s], matches)
        if note and note['code'] == 'ACTION_SPAN_EXPANDED':
            action, repair = expanded, original
            flags.add('ACTION_EXPANDED')
        elif note:
            flags.add(note['code'])
    anchor = (turkish_anchor(text, action, mode, subject, matches) if tr else _english_anchor(text, action, mode, subject, matches)) if action else None
    if anchor is None and tr and action:
        anchor = _english_anchor(text, action, mode, subject, matches)
    pos, modal = anchor if anchor else (text.find(action) if action else -1, None)
    if pos is None or pos < 0:
        flags.add('ACTION_NOT_LOCATED')
        pos = None
    lo = pos if pos is not None else (modal.start() if modal else 0)
    hi = max(pos + len(action) if pos is not None else lo, modal.end() if modal else lo)
    raw_start, s1 = _sentence_bounds(text, lo, hi, tr)
    s0 = after_notes(text, raw_start) if tr else raw_start + len(text[raw_start:]) - len(text[raw_start:].lstrip())
    s0 = min(s0, lo)
    # Subject: the legacy value, located, with the whole word it was cut from.
    subject_span, subject_end = None, None
    if subject:
        limit = pos if pos is not None else (modal.start() if modal else len(text))
        at = text.find(subject, s0, max(limit, s0))
        if at < 0:
            flags.add('SUBJECT_NOT_IN_SENTENCE')
            at = text.find(subject, 0, limit) if limit else -1
            at = at if at >= 0 else text.find(subject)
        if at >= 0:
            first = at
            while first > 0 and LETTER.match(text, first - 1):
                first -= 1
            last = word_end(text, at + len(subject))
            complete = first == at and last == at + len(subject)
            if not complete:
                flags.add('SUBJECT_WORD_CUT')
            subject_span = _span(text, at, at + len(subject), word_text=text[first:last], word_complete=complete)
            if s0 <= at < (pos if pos is not None else len(text)):
                subject_end = last
    modal_span = None
    if modal is not None and mode:
        modal_span = _span(text, modal.start(), modal.end(), polarity='NEGATIVE' if mode.endswith('_NOT') else 'POSITIVE',
                           force=mode.replace('_NOT', ''))
    action_span = _span(text, pos, pos + len(action), repair=repair) if pos is not None else None
    obj = _object(action, modal, tr) if action_span else None
    conditions = _qualifiers(text, _get(candidate, 'conditions'), s0, s1, tr, (TR_CONDITIONS, TR_CONDITION_EXTRA), CONDITIONS, subject_end, flags)
    exceptions = _qualifiers(text, _get(candidate, 'exceptions'), s0, s1, tr, (TR_EXCEPTIONS,), EXCEPTIONS, subject_end, flags)
    for qualifier in (*conditions, *exceptions):
        if INFINITIVE.search(qualifier['text']) or MODAL.search(qualifier['text']):
            flags.add('QUALIFIER_CONTAINS_PREDICATE')
    keeping = bool(KEEPING.search(text, s0, s1))
    deadline, threshold = _timing(text, s0, s1, tr, keeping, subject_end)
    closes = max(pos + len(action) if pos is not None else 0, modal.end() if modal else 0)
    enumeration = _items(text, closes, s1, pos, True) if closes and closes < s1 else []
    if not enumeration and pos is not None:
        enumeration = _items(text, s0, closes, pos, False)
    if closes and WORD.findall(text[closes:s1]) and not (enumeration and LIST_ITEM.search(text, closes, s1)):
        flags.add('DROPPED_AFTER_ACTION')            # md. 26(2): "... ve ... göz önünde bulundururlar"
    return {'version': VERSION, 'unit_offset': unit_offset, 'subject': subject_span, 'modal': modal_span, 'action': action_span,
            'object': {'text': obj, 'heuristic': True} if obj else None, 'conditions': conditions, 'exceptions': exceptions,
            'deadline': deadline, 'threshold': threshold, 'enumeration': enumeration,
            'source_span': _span(text, s0, s1), 'source_sha256': sha, 'flags': sorted(flags)}


LEGACY_KEYS = ('subject', 'modality', 'required_action', 'prohibited_action', 'conditions', 'exceptions')

# Coordinated acts (v0.19 COVERS gate). An action that joins two or more verb clauses ("... muhafaza etmek ve
# istenmesi halinde yetkililere ibraz etmek", "... alınır ve ... saklanır", "retain ... and produce ...") requires
# each of them. As one 'action' element a passage stating only the first act was read as stating "the act"
# (measured: independent I04, 5549 md. 8, records kept but nothing about producing them came out COVERS_TEXT).
# Each clause becomes its own element act_1, act_2 ..., an exact substring of the action. A Turkish clause ends
# in an infinitive, a passive aorist or a modal form; a bare verbal noun ("muhafaza") ends one only before
# " ve " when the next clause has its own words and closes with the shared light verb ("muhafaza ve istenmesi
# halinde yetkililere ibraz etmekle"). "veya" / "or" join alternatives, either of which meets the duty: never split.
TR_VERB_END = re.compile(r'(?:mek|mak|mekle|makla|mektedir|maktadır|meli|malı|melidir|malıdır)$')
TR_PASSIVE_END = re.compile(r'(?:[ıiuü]l[ıiuü]r|[ıiuüae]n[ıiuü]r)(?:lar|ler)?$')
TR_LIGHT_END = re.compile(r'(?:etmek|etmekle|edilir|edilmelidir|etmelidir|olunur|kılmak|kılınır)$')
VERBAL_NOUNS = frozenset({'muhafaza', 'ibraz', 'tespit', 'teyit', 'tevdi', 'beyan', 'temin', 'tesis', 'takip', 'kontrol', 'imha', 'iade',
                          'ihbar', 'ilan', 'talep', 'tebliğ', 'tescil', 'tasdik', 'kabul', 'iptal', 'tanzim', 'teslim', 'idame', 'tetkik',
                          'rapor', 'arşiv', 'tayin', 'takdim', 'tevsik'})
TR_JOIN = re.compile(r'\s+ve\s+|\s*,\s+(?:ve\s+)?')
EN_JOIN = re.compile(r'\s*,?\s+and\s+|\s*;\s+(?:and\s+)?')
EN_MODAL_LEAD = re.compile(r'^(?:(?:must|shall|should|will|also|then|to)\s+)+', re.I)
EN_VERBS = frozenset({'retain', 'keep', 'maintain', 'store', 'preserve', 'produce', 'provide', 'submit', 'send', 'report', 'notify', 'inform',
                      'disclose', 'record', 'verify', 'identify', 'monitor', 'review', 'assess', 'apply', 'obtain', 'take', 'make', 'ensure',
                      'establish', 'implement', 'destroy', 'delete', 'publish', 'give', 'pay', 'deliver', 'update', 'register', 'appoint',
                      'investigate', 'respond', 'complete', 'conduct', 'confirm', 'explain', 'refund', 'stop', 'cease', 'suspend',
                      'freeze', 'reject', 'refuse', 'tell', 'treat'})


def _words(text: str) -> list[str]:
    return re.findall(r'[%s]+' % TR_LETTERS, text)


def _tr_verb(word: str) -> bool:
    return bool(TR_VERB_END.search(word) or (len(word) >= 6 and TR_PASSIVE_END.search(word)))


def _tr_clause_ends(left: str, right: str, comma: bool) -> bool:
    """Whether a Turkish verb clause ends where `left` ends, the rest of the action being `right`."""
    words, rest = _words(left), _words(right)
    while rest and TR_STANDALONE.fullmatch(rest[-1]):
        rest.pop()                                          # "... ibraz etmekle yükümlüdür"
    if not words or not rest or not (_tr_verb(rest[-1]) or TR_LIGHT_END.search(rest[-1])):
        return False                                        # the rest carries no verb of its own
    if _tr_verb(words[-1]):
        return True
    return not comma and words[-1] in VERBAL_NOUNS and len(rest) >= 3 and bool(TR_LIGHT_END.search(rest[-1]))


def _en_first(text: str) -> str:
    return (_words(EN_MODAL_LEAD.sub('', text.strip())) or [''])[0].lower()


def coordinated_acts(action: str) -> list[str]:
    """The verb clauses of a coordinated action, each an exact substring of `action` (without the joining
    "ve"/"and", a trailing modal and punctuation); [] when the action has one verb clause."""
    action = action or ''
    tr = turkish(action) or (re.search(r'\sve\s', action) is not None and re.search(r'\sand\s', action) is None)
    cuts, start = [], 0
    for join in (TR_JOIN if tr else EN_JOIN).finditer(action):
        if join.start() <= start or _open_bracket(action, start, join.start()) or LIST_ITEM.match(action, join.end()):
            continue
        left, right = action[start:join.start()], action[join.end():]
        if tr:
            fits = _tr_clause_ends(left, right, 've' not in join.group().split())
        else:
            fits = _en_first(left) in EN_VERBS and _en_first(right) in EN_VERBS
        if fits:
            cuts.append((start, join.start()))
            start = join.end()
    if not cuts:
        return []
    cuts.append((start, len(action)))
    acts = []
    for lo, hi in cuts:
        text = action[lo:hi].strip().rstrip(' .;:,')
        tail = re.search(r'\s+(\S+)$', text)
        if tr and tail and TR_STANDALONE.fullmatch(tail.group(1)):
            text = text[:tail.start()]
        if not _words(text):
            return []
        acts.append(text)
    return acts


def source_sentence(structure: dict) -> str:
    """The governing sentence without unit number and amendment notes, at most 600 characters."""
    span = (structure or {}).get('source_span') or {}
    sentence = re.sub(r'\s+', ' ', NOTE.sub('', LEADING_NOTES.sub('', span.get('text') or '', count=1))).strip()
    if len(sentence) > SENTENCE_LIMIT:
        sentence = sentence[:SENTENCE_LIMIT - 2].rsplit(' ', 1)[0] + ' …'
    return sentence


# Element kinds (v0.19 PARTIAL confirmation, 25 September 2026). CRITICAL kinds are what the duty makes the
# obliged party do: a passage that does not state one of them states only part of the duty. SCOPE kinds say to
# whom and when it applies: a policy that leaves them out applies to everyone, always (the same or stricter),
# so their absence is never a gap; a passage written for ANOTHER subject (C03 md. 8(1), 9(1): registered
# companies against a duty about associations and unions; I05 md. 24(6): the bank's own messages against a
# duty of intermediary institutions) is not a partial statement of the duty but about something else.
SCOPE_KINDS = ('subject', 'condition', 'exception')
SCOPE_LIMIT = 200
QUALIFIER_ELEMENTS = 3


def critical_ids(elements) -> list[str]:
    """The ids of the critical elements (every kind but SCOPE_KINDS); an element without a kind is critical."""
    return [e['id'] for e in elements or () if e.get('kind') not in SCOPE_KINDS]


def _scope_elements(structure: dict, action: str, negative: bool, acts, items) -> tuple[list, list]:
    """(elements placed before the deadlines, qualifier elements placed after the items): subject, object and
    prohibition, then condition_1.. and exception_1.. — each an exact substring of the unit (of the action for
    the object), only where they exist; a qualifier that repeats a listed item or another qualifier is left out."""
    head = []
    subject = (structure.get('subject') or {}).get('text') or ''
    if subject.strip() and len(subject) <= SCOPE_LIMIT:
        head.append({'id': 'subject', 'kind': 'subject', 'text': subject})
    obj = (structure.get('object') or {}).get('text') or ''
    # One act only: the object of coordinated acts runs across all of them. A heuristic object that is not an
    # exact part of the action, nearly all of it, or that carries the duty's timing or amount ("... en geç on iş
    # günü içinde", "... işlem yapılmadan önce") says nothing the action and the deadline do not.
    if not acts and obj.strip() and obj in (action or '') and len(obj) <= SCOPE_LIMIT and len(obj) < 0.9 * len(action or '') \
            and not parse_quantities(obj) and not period_phrases(obj):
        head.append({'id': 'object', 'kind': 'object', 'text': obj})
    modal = structure.get('modal') or {}
    if negative and modal.get('polarity') == 'NEGATIVE' and (modal.get('text') or '').strip():
        head.append({'id': 'prohibition', 'kind': 'prohibition', 'text': modal['text']})
    tail = []
    for kind in ('condition', 'exception'):
        texts = []
        for span in structure.get(kind + 's') or []:
            text = span.get('text') or ''
            # Inside the action ("istenmesi halinde ... ibraz etmek") it is part of the act already.
            if not text.strip() or len(text) > SCOPE_LIMIT or text in (action or '') \
                    or any(text in t or t in text for t in (*texts, *items)):
                continue
            texts.append(text)
        tail += [{'id': f'{kind}_{i}', 'kind': kind, 'text': text} for i, text in enumerate(texts[:QUALIFIER_ELEMENTS], 1)]
    return head, tail


def duty_payload(candidate: dict, structure: dict | None) -> dict:
    """What the v0.19 judge reads: the six legacy keys (repaired action, marker conditions added),
    plus deadline, threshold, items, source_sentence and the element ids, only when present. The act is
    one element 'action', or act_1, act_2 ... when the action coordinates verb clauses (coordinated_acts);
    with a usable structure the subject, object, prohibition, conditions and exceptions are elements too
    (_scope_elements; SCOPE_KINDS are never a gap)."""
    duty = {key: _get(candidate, key) for key in LEGACY_KEYS}
    duty['conditions'] = list(duty['conditions'] or [])
    duty['exceptions'] = list(duty['exceptions'] or [])
    negative = (duty['modality'] or '').endswith('_NOT')
    usable = bool(structure) and 'STRUCTURE_FAILED' not in (structure.get('flags') or [])
    if usable and (structure.get('action') or {}).get('repair'):
        duty['prohibited_action' if negative else 'required_action'] = structure['action']['text']
    action = duty['prohibited_action'] if negative else duty['required_action']
    acts = coordinated_acts(action)
    elements = ([{'id': f'act_{i}', 'kind': 'action', 'text': text} for i, text in enumerate(acts, 1)] if acts else
                [{'id': 'action', 'kind': 'action', 'text': action or ''}])
    if not usable:
        return {**duty, 'elements': elements}
    for condition in structure.get('conditions') or []:
        if condition.get('origin') == 'marker' and not any(condition['text'] in c for c in duty['conditions']):
            duty['conditions'].append(condition['text'])
    deadlines = [e['text'] + (f" ({e['anchor']})" if e.get('anchor') else '') for e in structure.get('deadline') or []]
    thresholds = [t['text'] for t in structure.get('threshold') or [] if not t.get('parenthetical')]
    items = list(structure.get('enumeration') or [])[:ITEM_LIMIT]
    sentence = source_sentence(structure)
    if deadlines:
        duty['deadline'] = '; '.join(deadlines)
    if thresholds:
        duty['threshold'] = '; '.join(thresholds)
    if items:
        duty['items'] = items
    if sentence:
        duty['source_sentence'] = sentence
    head, tail = _scope_elements(structure, action or '', negative, acts, items)
    elements += head
    elements += [{'id': f'deadline_{i}', 'kind': 'deadline', 'text': value} for i, value in enumerate(deadlines, 1)]
    elements += [{'id': f'threshold_{i}', 'kind': 'threshold', 'text': value} for i, value in enumerate(thresholds, 1)]
    elements += [{'id': f'item_{i}', 'kind': 'item', 'text': value} for i, value in enumerate(items, 1)]
    duty['elements'] = elements + tail
    return duty


def duty_quantities(structure: dict) -> list[dict]:
    """The deadline and threshold entries that carry a number or a timing kind, for quantities.compare.

    Each gets the `unit_class` compare() needs ('duration', 'money', 'percent'; 'timing' for an
    "as soon as possible" without a day count). A bracketed variant amount ("(kripto varlık hizmet
    sağlayıcılar için onbeşbin TL)") applies to one group only and is left out.
    """
    if not structure or 'STRUCTURE_FAILED' in (structure.get('flags') or []):
        return []
    found = []
    for entry in structure.get('deadline') or []:
        if entry.get('amount') is None and entry.get('kind') not in TIMING_KINDS.values():
            continue
        found.append({**entry, 'unit_class': 'duration' if entry.get('days') is not None else 'timing'})
    for entry in structure.get('threshold') or []:
        if entry.get('parenthetical') or entry.get('amount') is None:
            continue
        unit = entry.get('unit')
        unit_class = 'duration' if unit in DAYS else 'percent' if unit == '%' else 'money'
        found.append({**entry, 'kind': 'THRESHOLD', 'unit_class': unit_class})
    return found

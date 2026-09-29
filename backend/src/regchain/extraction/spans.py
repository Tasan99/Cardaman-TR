"""Deterministic repair of a Turkish action that the extraction model cut short.

The prompt tells the model where a Turkish action ends (at the marker) but not where it
starts, and the model takes the shortest suffix. Measured on the v0.18 final run (24
September 2026): 4 of the 19 unique model candidates of Tedbirler Yönetmeliği were verb-only
or tail-only actions that grounding.verify accepted, because verify fixes only the end.
md. 26(2) came back as "uygulayamazlar": the judge read a prohibition with no object and
called the restating policy passage UNRELATED and a KYC-skip passage a conflict.

The repair keeps the end of the action and moves its start left, inside the governing
sentence, up to the latest of: the sentence start after the unit number and amendment notes,
the end of the subject word (through a coordinated obliged party), the end of a condition or
exception marker, the end of another duty marker, and the end of a sibling candidate's
action. The new action is always an exact substring of the unit and the old one is its
suffix. It never runs inside grounding.verify (verify must keep rejecting "tespit"), it is
Turkish only, and it changes nothing when the subject is not in the sentence (the
amendment-note subjects of the rules baseline) or when only function words are missing
("... yolcu ile ilgili tespitler ...").
"""
import re
import unicodedata

from .grounding import EXCEPTIONS, TR_CONDITIONS, TR_EXCEPTIONS, TR_LETTERS, binding, modality, standalone, turkish, turkish_action_fits, turkish_modal
from .schema import Candidate, ExtractionOutput

LETTER = re.compile('[%s]' % TR_LETTERS)
WORD = re.compile(r'[%s0-9]+' % TR_LETTERS)
TR_SENTENCE = re.compile(r'[.;!?]\s+(?=[A-ZÇĞİÖŞÜ(])')
FULL_STOP = re.compile(r'[.!?]\s+(?=[A-ZÇĞİÖŞÜ(])')
# "(2) (Değişik: 28/12/2009-2009/15720 K.) Sürekli iş ilişkisi ..." : the unit number and the
# amendment notes open the sentence but belong to no clause of it.
LEADING_NOTES = re.compile(r'\s*(?:\(\d{1,2}\)\s*)?(?:\((?:Ek|Değişik|Mülga|İptal|Yeniden düzenleme|Mükerrer)[^()]*\)\s*)*(?:\(\d{1,2}\)\s*)?')
# Condition postpositions grounding.TR_CONDITIONS does not list ("durumlarda", "-diğinde",
# "-dıklarında"). They bound the action here only; the grounding gate is left as it is.
TR_CONDITION_EXTRA = re.compile(r'(?<![%s])(?:durumlarda|hallerde|hâllerde|[%s]+(?:d|t)(?:ığında|iğinde|uğunda|üğünde|ıklarında|iklerinde|uklarında|üklerinde))(?![%s])'
                                % (TR_LETTERS, TR_LETTERS, TR_LETTERS))
# "Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler ...": a second obliged
# party coordinated with the subject is still the subject, not the start of the action.
COORDINATED_PARTY = re.compile(r'\s+(?:ile|ve|veya)\s+(?:\S+\s+){0,5}?\S*(?:kuruluş|meslek|yükümlü|banka|şirket|müessese|kurum)[%s]*' % TR_LETTERS)
COORDINATOR = re.compile(r'\s*,?\s*(?:ile|ve|veya)\s')
FUNCTION_WORDS = frozenset({'ile', 'ilgili', 'için', 'hakkında', 'olarak', 'üzere', 'de', 'da', 'ki', 've', 'veya'})
LEAD_SKIP = re.compile(r'[\s,;:]*(?:(?:ve|veya)\s+)?')
# "a) ...", "b) ..." (Turkish) and "(a) ...", "(ii) ..." (English) list markers inside a sentence.
LIST_MARKER = re.compile(r'(?:(?<=\s)|^)(?:(?P<tr>[a-zçğıöşü]|\d{1,2})\)|\((?P<en>[a-z]|[ivx]{1,4}|\d{1,2})\))\s+')
CLAUSE_MARK = re.compile(r'[,;:]')


def action_of(candidate) -> str | None:
    """The action a candidate states, from a Candidate or its dict."""
    get = candidate.get if isinstance(candidate, dict) else lambda key, default=None: getattr(candidate, key, default)
    negative = (get('modality') or '').endswith('_NOT')
    return get('prohibited_action') if negative else get('required_action')


def sentence_start(text: str, pos: int) -> int:
    """Start of the sentence that contains `pos`."""
    start = 0
    for boundary in TR_SENTENCE.finditer(text):
        if boundary.end() > pos:
            break
        start = boundary.end()
    return start


def after_notes(text: str, start: int) -> int:
    return LEADING_NOTES.match(text, start).end()


def word_end(text: str, end: int, limit: int | None = None) -> int:
    """`end` moved to the end of the word it falls in ("Yetkililer|ce" -> after "ce")."""
    limit = len(text) if limit is None else limit
    while end < limit and LETTER.match(text, end):
        end += 1
    return end


def turkish_anchor(text: str, action: str, mode: str, subject: str, matches) -> tuple[int, re.Match] | None:
    """(action start, marker) exactly as grounding.verify aligns a Turkish action, else None."""
    aligned = []
    for match in matches:
        if modality(match.group()) != mode or subject not in text[:match.start()]:
            continue
        if not turkish_modal(match.group()):
            continue
        pos = text.rfind(action, 0, match.end() + 1)
        if pos >= 0 and turkish_action_fits(action, pos + len(action), match):
            aligned.append((match, pos))
    if not aligned:
        return None
    match, pos = min(aligned, key=lambda pair: (not binding(pair[0]), pair[0].start() - pair[1] - len(action)))
    return pos, match


def exception_ends(text: str, lo: int, hi: int) -> list[int]:
    # "Merkezi yurt dışında bulunan ...": abroad, not an exception (engine.EXCLUSION guard).
    return [m.end() for m in TR_EXCEPTIONS.finditer(text, lo) if m.end() <= hi
            and not (m.group().startswith('dışında') and text[max(0, m.start() - 5):m.start()] == 'yurt ')]


def clause_start(text: str, pos: int, subject: str, matches, siblings=(), action=None) -> tuple[int | None, object]:
    """(start, coordinated) of the clause an action starting at `pos` belongs to.

    (None, 'SUBJECT_NOT_IN_SENTENCE') when the subject is not in the sentence, (None, None) when the
    action already starts the sentence. `coordinated` says the start sits right after a bare
    subject that a coordinator continues ("Yükümlüler ile ...").
    """
    s0 = sentence_start(text, pos)
    s0 = after_notes(text, s0)
    if pos <= s0:
        return None, None
    at = text.find(subject, s0, pos)
    if at < 0:
        # v0.19 (Tedbirler md. 31(1), independent run): "Kamu kurum ve kuruluşları, ... kuruluşlar; Başkanlık ve
        # denetim elemanları tarafından istenilecek her türlü bilgi, belge ... vermek ... zorundadır." A semicolon
        # after an enumerated subject is no sentence end: the subject is looked for back to the full stop, and the
        # semicolon still bounds the clause below.
        wide = after_notes(text, max((m.end() for m in FULL_STOP.finditer(text, 0, pos)), default=0))
        at = text.find(subject, wide, pos) if wide < s0 else -1
        if at < 0:
            return None, 'SUBJECT_NOT_IN_SENTENCE'
        s0 = wide
    subject_end = word_end(text, at + len(subject), pos)
    party = COORDINATED_PARTY.match(text, subject_end)
    if party and party.end() <= pos:
        subject_end = party.end()
    starts = [s0, subject_end]
    for pattern in (TR_CONDITIONS, TR_CONDITION_EXTRA):
        starts += [m.end() for m in pattern.finditer(text, s0) if m.end() <= pos]
    starts += exception_ends(text, s0, pos)
    # Another duty's marker earlier in the sentence closes that duty's clause.
    starts += [m.end() for m in matches if s0 <= m.start() and m.end() <= pos]
    # A semicolon or colon closes a clause: "... sağlayıcı tarafından; gönderici ... hususları göz
    # önünde bulundurulur" - the action does not start with the passive agent before it.
    starts += [m.end() for m in re.finditer(r'[;:]', text[:pos]) if m.start() >= s0][-1:]
    # Never overlap a sibling duty: its action (wherever it occurs before ours) bounds ours.
    for other in siblings:
        if not other or other == action:
            continue
        for found in re.finditer(re.escape(other), text):
            if found.start() < pos and found.end() > s0:
                starts.append(min(found.end(), pos))
    start = max(starts)
    coordinated = bool(start == subject_end and not party and COORDINATOR.match(text, start))
    if any(True for _ in LIST_MARKER.finditer(text, s0, start)):
        # Tedbirler md. 5(1): "... e) ... şüphe olduğunda tutar gözetmeksizin, kimliğe ilişkin ...":
        # a case of the list before the duty runs to its comma, and so does not belong to the action.
        mark = CLAUSE_MARK.search(text, start, pos)
        if mark:
            start = mark.end()
    return LEAD_SKIP.match(text, start, pos).end(), coordinated


def expand_action(text: str, candidate, siblings=(), matches=None) -> tuple[str | None, dict | None]:
    """(action, note) for one candidate: the action with its governing clause restored.

    The note is None when nothing applies, {'code': 'ACTION_SPAN_EXPANDED', ...} when the action
    was expanded, or a no-change note ('SUBJECT_NOT_IN_SENTENCE', 'ACTION_GAP_FUNCTION_WORDS',
    'SUBJECT_COORDINATION_UNRESOLVED') that says why a visible gap was left alone.
    """
    action = action_of(candidate)
    get = candidate.get if isinstance(candidate, dict) else lambda key, default=None: getattr(candidate, key, default)
    subject, mode = get('subject') or '', get('modality') or ''
    if not action or not subject or not turkish(text):
        return action, None
    if matches is None:
        from .classify import duty_modals
        matches = duty_modals(text)
    anchor = turkish_anchor(text, action, mode, subject, matches)
    if anchor is None:
        return action, None
    pos, _ = anchor
    start, coordinated = clause_start(text, pos, subject, matches, siblings, action)
    if start is None:
        return action, ({'code': coordinated, 'model_action': action} if coordinated else None)
    if start >= pos:
        return action, None
    gap = text[start:pos]
    words = WORD.findall(gap)
    if not words:
        return action, None
    if all(word.lower() in FUNCTION_WORDS for word in words):
        return action, {'code': 'ACTION_GAP_FUNCTION_WORDS', 'model_action': action, 'gap': gap.strip()}
    if coordinated:
        # "Yükümlüler ile bunların çalışanları, ...": a coordinated subject the pattern above does
        # not know; the action would begin with half of the subject.
        return action, {'code': 'SUBJECT_COORDINATION_UNRESOLVED', 'model_action': action}
    expanded = text[start:pos + len(action)]
    return expanded, {'code': 'ACTION_SPAN_EXPANDED', 'model_action': action, 'action': expanded, 'added': gap.strip()}


def repair_actions(text: str, output: ExtractionOutput) -> tuple[ExtractionOutput, list[dict]]:
    """(output, repairs): every truncated Turkish action expanded; one record per changed candidate.

    Pure and deterministic, and it never raises: a candidate the rule cannot place is returned
    unchanged, and verify then judges the model's own span exactly as before.
    """
    if output.status != 'EXTRACTED' or not output.obligations or not turkish(text):
        return output, []
    from .classify import duty_modals
    try:
        matches = duty_modals(text)
    except Exception:  # noqa: BLE001 - a repair must never break extraction
        return output, []
    actions = [action_of(candidate) for candidate in output.obligations]
    repaired, repairs = [], []
    for index, candidate in enumerate(output.obligations):
        siblings = [a for j, a in enumerate(actions) if j != index and a]
        try:
            new, note = expand_action(text, candidate, siblings, matches)
        except Exception:  # noqa: BLE001
            new, note = actions[index], None
        if not note or note['code'] != 'ACTION_SPAN_EXPANDED':
            repaired.append(candidate)
            continue
        field = 'prohibited_action' if candidate.modality.endswith('_NOT') else 'required_action'
        repaired.append(Candidate.model_validate({**candidate.model_dump(), field: new}))
        repairs.append({'stage': 'span_repair', 'code': 'ACTION_SPAN_EXPANDED', 'candidate': index,
                        'model_action': note['model_action'], 'action': new, 'added': note['added']})
    if not repairs:
        return output, []
    return ExtractionOutput(status=output.status, obligations=repaired), repairs


# --- v0.19 grounding salvage ------------------------------------------------------------------
# Measured on the v0.19 independent run (24 September 2026): whole duties were rejected for a
# field that differs from the source only by case, quotes or whitespace, for a qualifier the model
# left out or filed under the wrong field, and for a Turkish action that stops a few words short
# of its passive marker ("... son verilmesi" | "hususları göz önünde bulundurulur"). The salvage
# runs only after the last attempt failed the gate. It selects source spans and never writes
# text: every field it stores is an exact substring of the source, and verify has the last word.

_APOSTROPHES = frozenset("'’‘`´ʼ′")
_QUOTES = frozenset('"“”„«»″')
_DASHES = frozenset('‐‑‒–—−')


def _fold(char: str) -> str:
    if char.isspace():
        return ' '
    if char in 'İIı':                       # Turkish dotted/dotless i, either case
        return 'i'
    if char in _APOSTROPHES:
        return "'"
    if char in _QUOTES:
        return '"'
    if char in _DASHES:
        return '-'
    return ''.join(c for c in unicodedata.normalize('NFKD', char) if not unicodedata.combining(c)).lower()


def folded(text: str) -> tuple[str, list[int]]:
    """(folded text, source index of every folded character): whitespace runs, quote variants,
    case, Turkish i and diacritics folded."""
    out, index = [], []
    for position, char in enumerate(text):
        value = _fold(char)
        if value == ' ' and out and out[-1] == ' ':
            continue
        for c in value:
            out.append(c)
            index.append(position)
    return ''.join(out), index


def source_span(value: str, sources) -> str | None:
    """`value` itself when a source contains it, else the source span it matches once folded."""
    if any(value in source for source in sources):
        return value
    needle = folded(value)[0].strip()
    if not needle:
        return None
    for source in sources:
        haystack, index = folded(source)
        at = haystack.find(needle)
        if at >= 0:
            return source[index[at]:index[at + len(needle) - 1] + 1]
    return None


# v0.19 recall (25 September 2026): answers were still rejected for a comma the model dropped (Tedbirler
# md. 28(3)), a space before a comma it closed up ("mikrofiş ,"), or a few words it skipped inside a long
# copied clause (md. 31(3): "... sistemini, [denetimin amaçlarına uygun olarak] denetim elemanlarına
# açmak ..."). Such a value is snapped to the source span its words come from; the stored text is always
# that exact source span, never the model's wording.
WORD_TOKEN = re.compile(r'[^\W_]+')
SENTENCE_BREAK = re.compile(r'[.!?]\s+(?=[A-ZÇĞİÖŞÜ(])')
OVERLAP_MIN_WORDS = 4


def _words(text: str) -> list[tuple[str, int, int]]:
    """(folded word, start, end) for every word of `text`; offsets index `text` itself."""
    value, index = folded(text)
    return [(m.group(), index[m.start()], index[m.end() - 1] + 1) for m in WORD_TOKEN.finditer(value)]


def _with_end_mark(value: str, source: str, end: int) -> int:
    """Keep the model's closing full stop or semicolon when the source has it right there."""
    mark = value.rstrip()[-1:]
    return end + 1 if mark in '.;' and source[end:end + 1] == mark else end


def normalized_span(value: str, sources) -> str | None:
    """The source span whose words are exactly the value's words (punctuation and spacing ignored)."""
    wanted = [w for w, _, _ in _words(value)]
    if not wanted:
        return None
    for source in sources:
        words = _words(source)
        for i in range(len(words) - len(wanted) + 1):
            if [w for w, _, _ in words[i:i + len(wanted)]] == wanted:
                return source[words[i][1]:_with_end_mark(value, source, words[i + len(wanted) - 1][2])]
    return None


def overlap_span(value: str, sources) -> str | None:
    """The source span a long value was copied from with a few words left out.

    Every word of the value must occur in order inside one sentence of the source, starting and
    ending with the value's own first and last words; at most a quarter of the value's length (and at
    least two) source words may lie between them unmatched. An invented or changed word ("kullanması"
    -> "kullanmaması") never matches, so such a value stays rejected.
    """
    wanted = [w for w, _, _ in _words(value)]
    if len(wanted) < OVERLAP_MIN_WORDS:
        return None
    budget = max(2, len(wanted) // 4)
    best = None
    for source in sources:
        words = _words(source)
        for i, (word, start, _) in enumerate(words):
            if word != wanted[0]:
                continue
            j, k, skipped = 1, i + 1, 0
            while j < len(wanted) and k < len(words) and skipped <= budget:
                if words[k][0] == wanted[j]:
                    j += 1
                else:
                    skipped += 1
                k += 1
            if j < len(wanted) or skipped > budget or skipped == 0:
                continue
            end = words[k - 1][2]
            if SENTENCE_BREAK.search(source, start, end):
                continue
            if best is None or skipped < best[0]:
                best = (skipped, source[start:_with_end_mark(value, source, end)])
    return best[1] if best else None


# v0.19 t5 (independent run of 25 September 2026): Tedbirler md. 24(1) came back twice with "c) Adresi
# veyadoğum yeri ..." (two source words run together) and md. 31(1) with "isterilen usul" for the source's
# "istenilen usul": one slip in a 70-word copy rejected the whole duty. Two more field rungs, both tried
# only after the three above failed, and both storing the exact source span, never the model's text:
#   compact  - the value's letters and digits, spacing and punctuation ignored, equal a source run that
#              starts and ends on source word boundaries ("veyadoğum" = "veya doğum");
#   semantic - the value is the copy of one source sentence span except for ONE misspelt word: one
#              letter changed, added or left out, at least six letters, no digit ("isterilen"/"istenilen",
#              "vakıfın"/"vakfın"), and at most max(2, n/4) source words left out. A changed number, a
#              two-letter suffix ("kullanması" -> "kullanmaması"), or a second changed word is a different
#              text (SEMANTIC_ONLY) and stays rejected.
SEMANTIC_MIN_LETTERS = 6
_ALNUM = re.compile(r'[^\W_]', re.UNICODE)


def _word_bounds(source: str, start: int, end: int) -> bool:
    before = source[start - 1:start]
    after = source[end:end + 1]
    return not (before and _ALNUM.match(before)) and not (after and _ALNUM.match(after))


def compact_span(value: str, sources) -> str | None:
    """The source span whose letters and digits are the value's, spacing and punctuation ignored."""
    wanted = ''.join(c for c in folded(value)[0] if _ALNUM.match(c))
    if len(wanted) < 8:
        return None
    for source in sources:
        folded_source, index = folded(source)
        keep = [(c, index[i]) for i, c in enumerate(folded_source) if _ALNUM.match(c)]
        haystack = ''.join(c for c, _ in keep)
        at = haystack.find(wanted)
        while at >= 0:
            start, end = keep[at][1], keep[at + len(wanted) - 1][1] + 1
            if _word_bounds(source, start, end):
                return source[start:_with_end_mark(value, source, end)]
            at = haystack.find(wanted, at + 1)
    return None


def _one_letter_apart(a: str, b: str) -> bool:
    """A misspelt copy of a word: one letter changed, added or left out ("isterilen"/"istenilen",
    "vakıfın"/"vakfın"), both words long enough and letters only. A suffix such as the negative "-ma-" is
    two letters and never passes."""
    if min(len(a), len(b)) < SEMANTIC_MIN_LETTERS or not (a.isalpha() and b.isalpha()) or a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _sentence_spans(source: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for brk in SENTENCE_BREAK.finditer(source):
        spans.append((start, brk.start() + 1))
        start = brk.end()
    spans.append((start, len(source)))
    return spans


def word_alignment(value: str, sources) -> dict | None:
    """The best in-order alignment of the value's words with one source sentence (difflib on folded words).

    {'source', 'start', 'end' (source offsets of the aligned span), 'equal', 'changed' (list of (model word,
    source word)), 'skipped' (source words the value left out inside the span), 'extra' (value words with
    no source counterpart), 'words' (value word count)} or None when no word matches.
    """
    from difflib import SequenceMatcher
    wanted = [w for w, _, _ in _words(value)]
    if not wanted:
        return None
    best = None
    for source in sources:
        for s0, s1 in _sentence_spans(source):
            words = [(w, a + s0, b + s0) for w, a, b in _words(source[s0:s1])]
            if not words:
                continue
            ops = SequenceMatcher(None, wanted, [w for w, _, _ in words], autojunk=False).get_opcodes()
            while ops and ops[0][0] == 'insert':
                ops = ops[1:]
            while ops and ops[-1][0] == 'insert':
                ops = ops[:-1]
            if not any(op == 'equal' for op, *_ in ops):
                continue
            equal = skipped = extra = 0
            changed, used = [], []
            last = len(ops) - 1
            for position, (op, i1, i2, j1, j2) in enumerate(ops):
                if op == 'equal':
                    equal += i2 - i1
                    used += [j1, j2 - 1]
                elif op == 'insert':
                    skipped += j2 - j1
                elif op == 'delete':
                    extra += i2 - i1
                else:
                    # A changed run at the start pairs its last words with the source words right before
                    # the first match, one at the end its first words with the words right after the last.
                    pairs = min(i2 - i1, j2 - j1)
                    if position == 0 and position != last:
                        i1, j1 = i2 - pairs, j2 - pairs
                    else:
                        i2, j2 = i1 + pairs, j1 + pairs
                    changed += [(wanted[i1 + k], words[j1 + k][0]) for k in range(pairs)]
                    extra += (ops[position][2] - ops[position][1]) - pairs
                    if 0 < position < last:
                        skipped += (ops[position][4] - ops[position][3]) - pairs
                    used += [j1, j2 - 1] if pairs else []
            if not used:
                continue
            j_lo, j_hi = min(used), max(used) + 1
            found = {'source': source, 'start': words[j_lo][1], 'end': words[j_hi - 1][2], 'equal': equal, 'changed': changed,
                     'skipped': skipped, 'extra': extra, 'words': len(wanted)}
            if best is None or (equal, -len(changed) - extra, -skipped) > (best['equal'], -len(best['changed']) - best['extra'], -best['skipped']):
                best = found
    return best


def semantic_span(value: str, sources) -> str | None:
    """The source span a long value copies with one misspelt word (and at most a few words left out)."""
    found = word_alignment(value, sources)
    if (found is None or found['words'] < OVERLAP_MIN_WORDS or found['extra'] or len(found['changed']) != 1
            or not _one_letter_apart(*found['changed'][0]) or found['skipped'] > max(2, found['words'] // 4)):
        return None
    return found['source'][found['start']:_with_end_mark(value, found['source'], found['end'])]


def field_rejection(value: str, sources) -> str:
    """Why no rung placed a value in the source: NO_SOURCE_SPAN (mostly words the source does not have),
    SEMANTIC_ONLY (it restates one source sentence in changed words) or LOW_LEXICAL_OVERLAP (it shares
    the source's words, but not as one copied span)."""
    wanted = [w for w, _, _ in _words(value)]
    if not wanted:
        return NO_SOURCE_SPAN
    vocabulary = {w for source in sources for w, _, _ in _words(source)}
    if sum(w in vocabulary for w in wanted) * 2 < len(wanted):
        return NO_SOURCE_SPAN
    found = word_alignment(value, sources)
    if found is not None and found['changed'] and found['equal'] * 4 >= found['words'] * 3:
        return SEMANTIC_ONLY
    return LOW_LEXICAL_OVERLAP


def snapped(value: str, sources) -> tuple[str | None, str]:
    """(source span, step) for one model value, first rung that places it: 'snap' (folded), 'normalize'
    (words), 'overlap' (a few words left out), 'compact' (spacing), 'semantic' (one misspelt word)."""
    span = source_span(value, sources)
    if span is not None:
        return span, 'snap'
    span = normalized_span(value, sources)
    if span is not None:
        return span, 'normalize'
    span = overlap_span(value, sources)
    if span is not None:
        return span, 'overlap'
    span = compact_span(value, sources)
    if span is not None:
        return span, 'compact'
    return semantic_span(value, sources), 'semantic'


def _marker_after(text: str, start: int, end: int, mode: str, matches) -> re.Match | None:
    """The Turkish marker of `mode` that the text [start, end) runs up to: the next duty marker,
    in the same sentence; None when that marker is of another modality or there is none."""
    later = sorted((m for m in matches if m.end() > end), key=lambda m: m.start())
    if not later:
        return None
    marker = later[0]
    if (modality(marker.group()) != mode or not turkish_modal(marker.group()) or marker.start() < start
            or sentence_start(text, marker.start()) > start or TR_SENTENCE.search(text, end, marker.start())):
        return None
    return marker


def realign_action(text: str, candidate, matches) -> str | None:
    """A Turkish action that stops short of its marker, extended to end at it; else None.

    The model's action is kept as the start; the words between its end and the next marker of
    its modality in the same sentence are added ("... son verilmesi" -> "... son verilmesi
    hususları göz önünde bulundurulur"). A standalone marker ("zorundadır") ends the action before it.
    """
    action = action_of(candidate)
    if not action or not turkish(text) or candidate.subject not in text:
        return None
    for found in re.finditer(re.escape(action), text):
        marker = _marker_after(text, found.start(), found.end(), candidate.modality, matches)
        if marker is None or candidate.subject not in text[:marker.start()]:
            continue
        end = len(text[:marker.start()].rstrip()) if standalone(marker.group()) else marker.end()
        if end <= found.end():
            continue
        rebuilt = text[found.start():end]
        if turkish_action_fits(rebuilt, end, marker):
            return rebuilt
    return None

# --- v0.19 t5 grounding ladder ------------------------------------------------------------------
# Measured on the v019t4 runs (25 September 2026): whole units were rejected although the model's duty
# was right, because (1) one extra candidate for a sentence with no duty marker sank the others
# (Tedbirler md. 46(1): "... başlangıç tarihi hesabın kapatıldığı tarihtir." given as a second MUST duty,
# twice, after a repair hint that named the first candidate's marker), (2) an enumeration closed by one
# marker came back as one candidate per item (md. 21(2) "a) ..., b) ..., c) ..., emin olunması şartıyla
# mümkündür."), (3) an item ran past its marker, (4) a Turkish condition of the duty's own sentence was
# left out (md. 24(5), 24/A(3)), (5) one slip in a long copy (md. 24(1), 31(1)). The ladder, per candidate:
#   1 exact source span, 2 normalized (snap / normalize / compact), 3 source-overlap (overlap),
#   4 structured: the action placed against its marker (action / trim / merge), the duty's own-sentence
#     condition filed (qualifier / qualifier_source), a candidate no marker governs left out (drop),
#   5 semantic fallback: one misspelt word, and only as a source span (semantic).
# Every stored field is an exact source substring and verify has the last word. The v0.19 salvage path
# (phase A) runs first and unchanged, so no answer it salvaged is salvaged differently or rejected now.
SOURCE_BOUND_OK = 'SOURCE_BOUND_OK'
NO_SOURCE_SPAN = 'NO_SOURCE_SPAN'
LOW_LEXICAL_OVERLAP = 'LOW_LEXICAL_OVERLAP'
STRUCTURED_MISMATCH = 'STRUCTURED_MISMATCH'
SEMANTIC_ONLY = 'SEMANTIC_ONLY'
REASON_CODES = (SOURCE_BOUND_OK, NO_SOURCE_SPAN, LOW_LEXICAL_OVERLAP, STRUCTURED_MISMATCH, SEMANTIC_ONLY)
TIERS = ('exact', 'normalized', 'overlap', 'structured', 'semantic')
STEP_TIER = {'snap': 'normalized', 'normalize': 'normalized', 'compact': 'normalized', 'overlap': 'overlap',
             'action': 'structured', 'trim': 'structured', 'merge': 'structured', 'qualifier': 'structured',
             'qualifier_source': 'structured', 'span_repair': 'structured', 'semantic': 'semantic'}
# verify's messages, as short codes for the answer-level entries of the trace.
_ERROR_CODES = (('Cannot dismiss', 'DISMISSED_DUTY_SOURCE'), ('No explicit modal evidence', 'NO_MODAL_EVIDENCE'),
                ('Supporting citation', 'CITATION_MISMATCH'), ('Quote must preserve', 'QUOTE_MISMATCH'),
                ('Extracted field has no exact', 'FIELD_NOT_IN_SOURCE'), ('Subject/action do not align', 'ACTION_ALIGNMENT'),
                ('Contextual qualifier', 'CONTEXT_QUALIFIER_MISSING'))


def error_code(message: str) -> str:
    """A short code for a verify message ('ACTION_ALIGNMENT', 'MISSING_QUALIFIER', 'UNCOVERED_MODAL', ...)."""
    head = message.split(':', 1)[0].strip()
    if re.fullmatch(r'[A-Z_]{4,}', head):
        return head
    return next((code for prefix, code in _ERROR_CODES if message.startswith(prefix)), 'GATE_REJECTED')


def tier_of(steps) -> str:
    """The highest rung of the ladder a candidate needed ('exact' when none)."""
    return max((STEP_TIER.get(step, 'structured') for step in steps), key=TIERS.index, default='exact')


def accepted_trace(output: ExtractionOutput, repairs=()) -> list[dict]:
    """The trace of an answer the gate accepted as it came (a span repair counts as structured)."""
    repaired = {r.get('candidate') for r in repairs}
    return [{'candidate': index, 'code': SOURCE_BOUND_OK, 'tier': 'structured' if index in repaired else 'exact'}
            for index in range(len(output.obligations))]


def _model(candidate: dict):
    try:
        return Candidate.model_validate(candidate)
    except Exception:  # noqa: BLE001 - a candidate the schema refuses is simply not aligned
        return None


def _alignment(text: str, candidate: dict, matches):
    """(marker, action start) verify would align the candidate with, else None."""
    from .grounding import alignments, nearest
    model = _model(candidate)
    if model is None:
        return None
    found = alignments(text, model, matches)
    return nearest(found, action_of(model)) if found else None


def trim_action(text: str, candidate: dict, matches) -> str | None:
    """A Turkish action copied past its marker ("... emin olunması şartıyla mümkündür."), cut at the marker."""
    action, subject, mode = action_of(candidate), candidate['subject'], candidate['modality']
    if not action or not turkish(text):
        return None
    for found in re.finditer(re.escape(action), text):
        inside = sorted((m for m in matches if found.start() < m.start() and m.end() < found.end()), key=lambda m: m.start())
        for marker in inside[:1]:
            if (modality(marker.group()) != mode or not turkish_modal(marker.group()) or subject not in text[:marker.start()]
                    or FULL_STOP.search(text, found.start(), marker.start())):
                continue
            end = len(text[:marker.start()].rstrip()) if standalone(marker.group()) else marker.end()
            trimmed = text[found.start():end]
            if len(WORD.findall(trimmed)) >= 2 and turkish_action_fits(trimmed, end, marker):
                return trimmed
    return None


def merge_enumeration(text: str, candidates: list, matches) -> list[tuple[int, str, list[int]]]:
    """[(kept, merged action, absorbed)] for every run of candidates that copied the items of one enumeration.

    Same subject and modality, their actions in source order inside one sentence with at most three words
    between two items, none but the last aligned with a marker, and one marker of that modality closing the
    run ("a) ..., b) ..., c) ..., emin olunması şartıyla"): the merged action runs from the first item to
    that marker. The source words no item copied (separators, the shared predicate) may be at most
    max(2, n/4) of its n words, and the merged action never overlaps another candidate's.
    """
    if not turkish(text):
        return []
    placed = []
    for index, candidate in candidates:
        action = action_of(candidate)
        at = text.find(action) if action else -1
        if at >= 0:
            placed.append((at, at + len(action), index, candidate, _alignment(text, candidate, matches) is None))
    placed.sort(key=lambda item: item[0])
    merges, i = [], 0
    while i < len(placed):
        run = [placed[i]]
        j = i + 1
        while (run[-1][4] and j < len(placed) and placed[j][3]['subject'] == run[0][3]['subject']
               and placed[j][3]['modality'] == run[0][3]['modality'] and placed[j][0] >= run[-1][1]
               and len(WORD.findall(text[run[-1][1]:placed[j][0]])) <= 3 and not FULL_STOP.search(text, run[-1][1], placed[j][0])):
            run.append(placed[j])
            j += 1
        i = j
        if len(run) < 2:
            continue
        first, last = run[0], run[-1]
        subject, mode = first[3]['subject'], first[3]['modality']
        closing = sorted((m for m in matches if m.start() >= last[0] and modality(m.group()) == mode and turkish_modal(m.group())
                          and subject in text[:m.start()] and not FULL_STOP.search(text, first[0], m.start())), key=lambda m: m.start())
        if not closing:
            continue
        marker = closing[0]
        end = len(text[:marker.start()].rstrip()) if standalone(marker.group()) else marker.end()
        merged = text[first[0]:end]
        if end <= last[0] or not turkish_action_fits(merged, end, marker):
            continue
        covered = [(max(a, first[0]), min(b, end)) for a, b, *_ in run]
        words = list(WORD.finditer(text, first[0], end))
        uncopied = [w for w in words if not any(a <= w.start() and w.end() <= b for a, b in covered)]
        if len(uncopied) > max(2, len(words) // 4):
            continue
        members = {r[2] for r in run}
        if any(a < end and b > first[0] for a, b, index, *_ in placed if index not in members):
            continue
        merges.append((first[2], merged, [r[2] for r in run[1:]]))
    return merges


def own_sentence_condition(text: str, span: str, candidates: list, matches) -> int | None:
    """The candidate (position in `candidates`) whose own Turkish sentence carries a demanded condition.

    Turkish puts the condition clause before the subject and the predicate it governs ("Gönderilen
    mesajların ... tamamlanmaması halinde, alıcı ... tarafından; ... göz önünde bulundurulur."): the span
    must lie in the sentence of the candidate's marker and end before its action; the nearest action wins.
    """
    from .grounding import _sentence
    at = text.find(span)
    if at < 0 or not turkish(text):
        return None
    best = None
    for position, candidate in enumerate(candidates):
        anchor = _alignment(text, candidate, matches)
        if anchor is None:
            continue
        marker, start = anchor
        s0, s1 = _sentence(text, marker.start())
        if s0 <= at and at + len(span) <= start < s1:
            distance = start - (at + len(span))
            if best is None or distance < best[0]:
                best = (distance, position)
    return None if best is None else best[1]


def condition_inside_action(text: str, span: str, candidates: list) -> tuple[int, str] | None:
    """(position, action) when a candidate's Turkish action starts inside a demanded condition and runs on
    past it ("yeni bilgi ve bulgular elde edildiği takdirde tekrar ... gönderilir" for the condition
    "Bildirimde bulunulan işlemle ilgili olarak ... elde edildiği takdirde"): the action keeps the text after
    the condition, and the whole condition is filed as a condition."""
    at = text.find(span)
    if at < 0 or not turkish(text):
        return None
    end = at + len(span)
    for position, candidate in enumerate(candidates):
        action = action_of(candidate)
        for found in re.finditer(re.escape(action), text) if action else ():
            if at < found.start() < end < found.end():
                rest = LEAD_SKIP.match(text, end, found.end()).end()
                if len(WORD.findall(text[rest:found.end()])) >= 2:
                    return position, text[rest:found.end()]
    return None


def _qualify(text: str, candidates: list, matches, note, supply: bool) -> bool:
    """Step 3: file every demanded primary-source qualifier; False when one is carried nowhere.

    A qualifier the answer already carries is filed where the gate wants it: filed under the other field
    (CONC 7.7.2), cut short to its marker ("bulunması halinde"), or - for a condition, and in Turkish for
    an exception too - copied inside the action (CONC 7.7.3). An English exception the answer states
    nowhere but in its action is not added: a carve-out is the model's to state (test_extraction pins it),
    and neither is a condition nested inside an exception span, which conditions the carve-out and not the
    duty. With `supply` (phase B, Turkish only) a demanded CONDITION the answer left out is filed on the
    duty of its own sentence (own_sentence_condition); an omitted exception stays rejected.
    """
    from .grounding import missing_qualifiers, required_qualifiers
    current = ExtractionOutput(status='EXTRACTED', obligations=[Candidate.model_validate(c) for _, c in candidates])
    carve_outs = [(text.find(span), text.find(span) + len(span)) for span in required_qualifiers(text, EXCEPTIONS, current, matches)]
    for field, span in missing_qualifiers(text, current, matches):
        other = 'exceptions' if field == 'conditions' else 'conditions'
        at = text.find(span)
        nested = any(a <= at and at + len(span) <= b and (a, b) != (at, at + len(span)) for a, b in carve_outs)
        # v0.19: in Turkish an exception clause sits between the subject and the marker, so an action
        # copied up to its marker carries it (Kanun 5549 md. 4(2) "... mahkemeler dışında, ... hiç
        # kimseye açıklayamazlar"); the answer states the carve-out and it is filed as one. An English
        # action can stop before its "unless" tail, and there the rule above still holds.
        position = next((i for i, (_, c) in enumerate(candidates) if span in c[other]
                         or any(len(value.split()) >= 2 and value in span for value in c[field])
                         or ((field == 'conditions' or turkish(text)) and not nested and span in (action_of(c) or ''))), None)
        step = 'qualifier'
        if position is None and supply and field == 'conditions' and not nested:
            position = own_sentence_condition(text, span, [c for _, c in candidates], matches)
            step = 'qualifier_source'
            if position is None:
                split = condition_inside_action(text, span, [c for _, c in candidates])
                if split is not None:
                    position, action = split
                    index, candidate = candidates[position]
                    key = 'prohibited_action' if candidate['modality'].endswith('_NOT') else 'required_action'
                    note('qualifier_source', index, key, candidate[key], action)
                    candidate[key] = action
        if position is None:
            return False
        index, candidate = candidates[position]
        if span in candidate[other]:
            candidate[other].remove(span)
        candidate[field].append(span)
        note(step, index, field, None, span)
    return True


def _snap_fields(text: str, candidate: dict, index: int, note) -> tuple[str, str] | None:
    """Rungs 1-3 and 5 for every field of one candidate; (field, value) of the first field no rung places."""
    sources = [text, *[e['quote'] for e in candidate['supporting_evidence']]]
    for key in ('subject', 'required_action', 'prohibited_action', 'deadline'):
        if candidate[key] is None:
            continue
        span, how = snapped(candidate[key], sources)
        if span is None:
            return key, candidate[key]
        if span != candidate[key]:
            note(how, index, key, candidate[key], span)
            candidate[key] = span
    for key in ('conditions', 'exceptions', 'affected_products', 'affected_entities', 'customer_types'):
        for position, value in enumerate(candidate[key]):
            span, how = snapped(value, sources)
            if span is None:
                return '%s[%d]' % (key, position), value
            if span != value:
                note(how, index, '%s[%d]' % (key, position), value, span)
                candidate[key][position] = span
    return None


def _realign(text: str, candidates: list, matches, note) -> None:
    """Step 2: a Turkish action that does not end at its marker, extended to it."""
    for index, candidate in candidates:
        model = _model(candidate)
        if model is None or turkish_anchor(text, action_of(model), model.modality, model.subject, matches) is not None:
            continue
        rebuilt = realign_action(text, model, matches)
        if rebuilt:
            field = 'prohibited_action' if model.modality.endswith('_NOT') else 'required_action'
            note('action', index, field, candidate[field], rebuilt)
            candidate[field] = rebuilt


def _gate(text: str, candidates: list, context, note) -> tuple[ExtractionOutput | None, str]:
    """Step 4: verify decides; then the span repair, kept only when verify still accepts it."""
    from .grounding import verify
    salvaged = ExtractionOutput(status='EXTRACTED', obligations=[Candidate.model_validate(c) for _, c in candidates])
    try:
        verify(text, salvaged, context)
    except ValueError as exc:
        return None, str(exc)
    repaired, repairs = repair_actions(text, salvaged)
    if repairs:
        try:
            verify(text, repaired, context)
        except ValueError:
            repairs = []
        else:
            salvaged = repaired
        for repair in repairs:
            note('span_repair', candidates[repair['candidate']][0], 'action', repair['model_action'], repair['action'])
    return salvaged, ''


def _attempt(text: str, output: ExtractionOutput, context, matches, extended: bool):
    """One pass of the ladder: (output or None, notes, dropped {index: trace entry}, failure code)."""
    notes = []

    def note(step, index, field, before, after):
        notes.append({'step': step, 'candidate': index, 'field': field, 'model': before, 'source': after})

    candidates, dropped = [], {}
    for index, value in enumerate(output.obligations):
        candidate = value.model_dump()
        failed = _snap_fields(text, candidate, index, note)
        if failed is None:
            candidates.append((index, candidate))
            continue
        sources = [text, *[e['quote'] for e in candidate['supporting_evidence']]]
        dropped[index] = {'candidate': index, 'code': field_rejection(failed[1], sources), 'field': failed[0].split('[')[0]}
        if not extended:
            return None, notes, dropped, 'FIELD_NOT_IN_SOURCE'          # phase A: exactly the v0.19 salvage
    if extended:
        # 4a: an enumeration copied item by item becomes the one duty its marker closes.
        for kept, merged, absorbed in merge_enumeration(text, candidates, matches):
            candidate = next(c for i, c in candidates if i == kept)
            field = 'prohibited_action' if candidate['modality'].endswith('_NOT') else 'required_action'
            note('merge', kept, field, candidate[field], merged)
            candidate[field] = merged
            for index in absorbed:
                other = next(c for i, c in candidates if i == index)
                for key in ('conditions', 'exceptions'):
                    candidate[key] += [v for v in other[key] if v not in candidate[key]]
                note('merge', index, 'candidate', action_of(other), None)
                dropped[index] = {'candidate': index, 'code': SOURCE_BOUND_OK, 'tier': 'structured', 'merged_into': kept}
            candidates = [(i, c) for i, c in candidates if i not in absorbed]
    # 4b: an action short of its marker is extended to it (the v0.19 step), one that runs past it is cut.
    _realign(text, candidates, matches, note)
    if extended:
        for index, candidate in candidates:
            if _alignment(text, candidate, matches) is None:
                trimmed = trim_action(text, candidate, matches)
                if trimmed:
                    field = 'prohibited_action' if candidate['modality'].endswith('_NOT') else 'required_action'
                    note('trim', index, field, candidate[field], trimmed)
                    candidate[field] = trimmed
        # 4c: a candidate no marker governs (its sentence states no duty, or the source gives it another
        # subject or modality) is left out; the others still have to cover every binding marker.
        for index, candidate in candidates:
            if _alignment(text, candidate, matches) is None:
                note('drop', index, 'candidate', action_of(candidate), None)
                dropped[index] = {'candidate': index, 'code': STRUCTURED_MISMATCH, 'detail': 'NO_ALIGNED_MARKER'}
        candidates = [(i, c) for i, c in candidates if i not in dropped]
    if not candidates:
        return None, notes, dropped, 'NO_CANDIDATE_LEFT'
    if not _qualify(text, candidates, matches, note, extended):
        return None, notes, dropped, 'MISSING_QUALIFIER'
    if not notes:
        return None, notes, dropped, 'NOTHING_TO_SALVAGE'
    salvaged, error = _gate(text, candidates, context, note)
    if salvaged is None:
        return None, notes, dropped, error_code(error)
    return salvaged, notes, dropped, ''


def ground(text: str, output: ExtractionOutput, context=None) -> tuple[ExtractionOutput | None, dict | None, list[dict]]:
    """(output, salvage record, trace) for an answer the gate rejected; output and record are None when it
    stays rejected.

    Phase A is the v0.19 salvage (rungs 1-3 and 5 on the fields, the realigned action, the qualifiers the
    answer carries); only when it fails does phase B add the structured rung (merge, trim, drop, the
    own-sentence condition). The trace has one entry per model candidate: {'candidate', 'code':
    SOURCE_BOUND_OK, 'tier'} for a kept (or merged) duty, {'candidate', 'code': NO_SOURCE_SPAN |
    LOW_LEXICAL_OVERLAP | SEMANTIC_ONLY, 'field'} or {'candidate', 'code': STRUCTURED_MISMATCH, 'detail'}
    for a rejected one; a rejected answer adds {'candidate': -1, 'code': STRUCTURED_MISMATCH, 'detail':
    <why the gate still refused it>}. Strings and ints only (digest-safe).
    """
    from .classify import duty_modals
    if output.status != 'EXTRACTED' or not output.obligations:
        return None, None, [{'candidate': -1, 'code': STRUCTURED_MISMATCH,
                             'detail': 'NO_CANDIDATE' if output.status == 'EXTRACTED' else 'DISMISSED_DUTY_SOURCE'}]
    matches = duty_modals(text)
    salvaged, notes, dropped, failure = _attempt(text, output, context, matches, False)
    if salvaged is None:
        salvaged, notes, dropped, failure = _attempt(text, output, context, matches, True)
    total = len(output.obligations)
    if salvaged is None:
        trace = [dropped.get(i) or {'candidate': i, 'code': STRUCTURED_MISMATCH, 'detail': failure} for i in range(total)]
        return None, None, trace + [{'candidate': -1, 'code': STRUCTURED_MISMATCH, 'detail': failure}]
    # A candidate left out keeps only the note that says so.
    notes = [n for n in notes if n['step'] in ('drop', 'merge') or dropped.get(n['candidate'], {}).get('code', SOURCE_BOUND_OK) == SOURCE_BOUND_OK]
    steps, per = [], {}
    for item in notes:
        if item['step'] not in steps:
            steps.append(item['step'])
        per.setdefault(item['candidate'], []).append(item['step'])
    trace = [dropped.get(i) or {'candidate': i, 'code': SOURCE_BOUND_OK, 'tier': tier_of(per.get(i, []))} for i in range(total)]
    return salvaged, {'stage': 'grounding', 'code': 'GROUNDING_SALVAGED', 'steps': steps, 'fields': notes}, trace


def salvage(text: str, output: ExtractionOutput, context=None) -> tuple[ExtractionOutput, dict] | None:
    """(output, diagnostic) when the grounding ladder (ground) makes the answer pass verify; None otherwise."""
    salvaged, record, _ = ground(text, output, context)
    return (salvaged, record) if salvaged is not None else None


# --- v0.19 t6 sentence coverage -----------------------------------------------------------------
# Measured on the v019t3, v019t4 and v019t5 runs (25 September 2026): Yönetmelik md. 24/A(3) has three duty
# sentences ("... eksik olan bilgilerin tamamlanmasını talep eder." / "... söz konusu kripto varlık transferini iade
# eder." / "... hususları göz önünde bulundurulur."). In every run the model gave a candidate for the third only, and
# the gate accepted it: the first two end in v0.19 soft markers, which verify never demands (so that no answer the
# v0.18 gate accepted is rejected). Nothing checked that each duty sentence got a duty, so the labelled "iade eder"
# duty had no candidate from its sentence and the harness scored the row OTHER_SENTENCE. Across the 157 Turkish units
# the recorded runs accepted, 9 duty sentences were left so (24/A(3) twice; md. 8(1) and 9(1) "... teyit edilir.").
# After an answer is accepted, and after the second reading, every sentence of the unit that holds a duty marker is
# the sentence of some candidate's marker:
#   - a candidate whose action runs over a full stop is cut back to the sentence of its marker: the words of an
#     earlier sentence never make that sentence's duty part of another one (OTHER_SENTENCE); the earlier sentence is
#     then covered below;
#   - a duty sentence without a candidate gets one built from that sentence alone: its last duty marker, the
#     predicate that ends with it (right before it, for a separate marker word), the sentence's own leading subject or,
#     for a Turkish subject carried over from the sentence before, the subject of the unit's previous duty, and the
#     conditions and exceptions of that sentence the gate demands;
#   - a subject or a qualifier a candidate takes from another sentence is a supporting dependency: it is kept and
#     recorded as cross-sentence in the trace.
# Every stored field is an exact source span and verify has the last word: a sentence whose candidate verify refuses
# stays uncovered and is recorded. Turkish only, like the structured rungs of the ladder. The new candidates come
# after the model's, so the model's own duties keep their order.
COVERED_CODE = 'DUTY_SENTENCE_COVERED'
UNCOVERED_CODE = 'DUTY_SENTENCE_UNCOVERED'
OTHER_SENTENCE = 'OTHER_SENTENCE'
REVIEW_REJECTED = 'REVIEW_REJECTED'                      # v0.19 t7: a sentence whose duty the second reading rejected
STEP_TIER['sentence'] = 'structured'
LIST_ITEM = re.compile(r'[a-zçğıöşü]\)\s+')
SUBJECT_MAX_WORDS = 25
MAX_OBLIGATIONS = 20                                     # schema.ExtractionOutput.obligations


def unit_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of the unit's sentences as grounding reads them (grounding._sentence): a '.', '!' or '?' before
    a capital or an opening bracket, or the end of the text, closes one; a ';' does not. A full stop inside brackets
    closes none: "(Ek:RG-24/2/2021-31405 2. Mükerrer-C.K-3580/13 md.)" is an amendment note, not two sentences."""
    from .grounding import TR_SENTENCE_END
    spans, start, depth, scanned = [], 0, 0, 0
    for end in TR_SENTENCE_END.finditer(text):
        for char in text[scanned:end.start()]:
            depth = depth + 1 if char == '(' else max(0, depth - 1) if char == ')' else depth
        scanned = end.start()
        if depth:
            continue
        spans.append((start, end.end()))
        start = end.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _index(sentences: list, pos: int) -> int:
    return next((i for i, (a, b) in enumerate(sentences) if a <= pos < b), len(sentences) - 1)


def _sentence_at(text: str, pos: int) -> tuple[int, int]:
    sentences = unit_sentences(text)
    return sentences[_index(sentences, pos)] if sentences else (0, len(text))


def duty_sentences(text: str, matches) -> dict:
    """{sentence index: its Turkish duty markers, in source order} for every sentence of the unit that holds one."""
    sentences = unit_sentences(text)
    found = {}
    for match in matches:
        if turkish_modal(match.group()):
            found.setdefault(_index(sentences, match.start()), []).append(match)
    return dict(sorted(found.items()))


def sentence_subject(text: str, lo: int, stop: int) -> tuple[int, int] | None:
    """(start, end) of a sentence's own leading subject: the phrase before its first comma, after the unit number, the
    amendment notes, a list letter and any leading condition or exception clause ("... halinde, alıcı ...").

    None when the sentence has no such phrase before `stop` (its marker): a subject carried over from the sentence
    before ("Bilgilerin tamamlanmaması halinde söz konusu kripto varlık transferini iade eder."), or a phrase that names
    a qualifier, holds a marker, a ';' or a ':', runs over SUBJECT_MAX_WORDS words, or opens a list ("Derneğin adı,
    amacı, kütük numarası ...": that comma separates items, not a subject from its predicate).
    """
    from .grounding import MODAL
    pos = after_notes(text, lo)
    item = LIST_ITEM.match(text, pos)
    pos = item.end() if item else pos
    while True:
        comma = text.find(',', pos, stop)
        if comma < 0:
            return None
        head = text[pos:comma]
        qualifiers = [m for pattern in (TR_CONDITIONS, TR_CONDITION_EXTRA, TR_EXCEPTIONS) for m in pattern.finditer(head)]
        if qualifiers and head.rstrip().endswith(max(qualifiers, key=lambda m: m.end()).group()):
            pos = comma + 1                                     # a leading qualifier clause: the subject follows it
            continue
        break
    words = WORD.findall(head)
    if not words or len(words) > SUBJECT_MAX_WORDS or qualifiers or MODAL.search(head) or re.search('[;:]', head):
        return None
    following = text.find(',', comma + 1, stop)
    if following >= 0 and len(WORD.findall(text[comma + 1:following])) <= 2:
        return None
    start = pos + len(head) - len(head.lstrip())
    return start, start + len(head.strip())


def sentence_action_start(text: str, lo: int, marker: re.Match, subject_end: int | None = None) -> int:
    """Where the predicate a sentence's marker closes starts: after the unit number, the amendment notes and a list
    letter, the sentence's own subject (with a coordinated obliged party) and every condition or exception clause
    before the marker. Unlike clause_start a ';' does not end it: in "... belgeler; ... belgeleri; ... teyit edilir."
    the semicolons separate what the passive verb governs."""
    start = after_notes(text, lo)
    item = LIST_ITEM.match(text, start)
    starts = [item.end() if item else start]
    if subject_end is not None and lo <= subject_end <= marker.start():
        party = COORDINATED_PARTY.match(text, subject_end)
        starts.append(party.end() if party and party.end() <= marker.start() else subject_end)
    for pattern in (TR_CONDITIONS, TR_CONDITION_EXTRA):
        starts += [m.end() for m in pattern.finditer(text, lo, marker.start())]
    starts += exception_ends(text, lo, marker.start())
    return LEAD_SKIP.match(text, max(starts), marker.start()).end()


def own_sentence_action(text: str, candidate: dict, matches, siblings=()) -> str | None:
    """The part of a Turkish action inside the sentence of its marker when the action starts in an earlier sentence
    (it runs over a full stop); None when the action is already that one sentence's. The part starts where the
    span repair would start the clause (clause_start: after the subject, a qualifier, a ';'), else - the subject being
    in an earlier sentence - at sentence_action_start."""
    anchor = _alignment(text, candidate, matches)
    if anchor is None:
        return None
    marker, start = anchor
    lo, _ = _sentence_at(text, marker.start())
    if start >= lo:
        return None
    action = action_of(candidate)
    begin, _ = clause_start(text, marker.start(), candidate['subject'], matches, siblings, action)
    if begin is None or begin < lo:
        begin = sentence_action_start(text, lo, marker)
    cut = text[begin:start + len(action)]
    return cut if len(WORD.findall(cut)) >= 2 else None


def sentence_candidate(text: str, lo: int, marker: re.Match, inherited: str | None, current: list, matches):
    """(candidate dict, cross-sentence fields) for the duty one sentence's marker closes, or (None, why not).

    The subject is the sentence's own (sentence_subject), else `inherited` (the unit's previous duty's subject, a
    cross-sentence subject); the action runs from sentence_action_start to the marker; the conditions and exceptions
    are those of this sentence that grounding.required_qualifiers demands once the duty is aligned with the marker.
    """
    from .grounding import CONDITIONS, required_qualifiers
    own = sentence_subject(text, lo, marker.start())
    if own is not None:
        subject, subject_end, cross = text[own[0]:own[1]], own[1], []
    elif inherited and inherited in text[:marker.start()]:
        subject, subject_end, cross = inherited, None, ['subject']
    else:
        return None, 'NO_SUBJECT'
    start = sentence_action_start(text, lo, marker, subject_end)
    end = len(text[:marker.start()].rstrip()) if standalone(marker.group()) else marker.end()
    action = text[start:end]
    if start >= end or len(WORD.findall(action)) < 2:
        return None, 'NO_ACTION'
    mode = modality(marker.group())
    value = {'source_quote': text, 'subject': subject, 'modality': mode, 'required_action': None if mode.endswith('_NOT') else action,
             'prohibited_action': action if mode.endswith('_NOT') else None, 'conditions': [], 'exceptions': [],
             'confidence_score': '0.5000'}
    trial = ExtractionOutput(status='EXTRACTED', obligations=[*current, Candidate.model_validate(value)])
    for pattern, field in ((CONDITIONS, 'conditions'), (EXCEPTIONS, 'exceptions')):
        value[field] = [span for span in required_qualifiers(text, pattern, trial, matches) if text.find(span, lo, start) >= 0]
    return Candidate.model_validate(value).model_dump(), cross


def cross_sentence(text: str, candidate: dict, matches) -> list[str]:
    """The fields of a candidate that lie outside the sentence of its marker: 'subject', 'action' (it runs over a full
    stop), 'conditions', 'exceptions'. Kept as supporting dependencies; the trace names them."""
    anchor = _alignment(text, candidate, matches)
    if anchor is None:
        return []
    marker, start = anchor
    lo, hi = _sentence_at(text, marker.start())
    fields = [] if text.find(candidate['subject'], lo, marker.start()) >= 0 else ['subject']
    fields += ['action'] if start < lo else []
    fields += [key for key in ('conditions', 'exceptions') if any(text.find(value, lo, hi) < 0 for value in candidate[key])]
    return fields


def _gate_refusal(text: str, candidates: list, context) -> str:
    """'' when verify accepts the candidates as one answer, else the short code of its refusal."""
    from .grounding import verify
    try:
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[Candidate.model_validate(c) for c in candidates]), context)
    except ValueError as exc:
        return error_code(str(exc))
    return ''


def _covered(text: str, candidates: list, matches, sentences) -> set[int]:
    """Indices of the sentences some candidate's marker lies in, or its action runs through."""
    covered = set()
    for candidate in candidates:
        anchor = _alignment(text, candidate, matches)
        if anchor is not None:
            marker, start = anchor
            end = start + len(action_of(candidate))
            covered.add(_index(sentences, marker.start()))
            covered.update(i for i, (a, b) in enumerate(sentences) if a < end and start < b)
    return covered


def _fill(text: str, candidates: list, duties: dict, sentences, matches, rejected=()) -> tuple[list, list]:
    """([(sentence index, candidate, cross fields)], [uncovered {'sentence', 'detail'}]) for the duty sentences no
    candidate covers, in source order; a later sentence may inherit the subject of an earlier new candidate. v0.19 t7:
    the sentence of a `rejected` candidate's marker (a duty the review rejected) is not filled: detail REVIEW_REJECTED."""
    covered = _covered(text, candidates, matches, sentences)
    refused = {_index(sentences, anchor[0].start()) for c in rejected for anchor in [_alignment(text, c, matches)] if anchor} - covered
    placed = [(_index(sentences, anchor[0].start()), c) for c in candidates for anchor in [_alignment(text, c, matches)] if anchor]
    fills, uncovered = [], []
    for index, markers in duties.items():
        if index in covered:
            continue
        if index in refused:
            uncovered.append({'sentence': index, 'detail': REVIEW_REJECTED})
            continue
        if len(candidates) + len(fills) >= MAX_OBLIGATIONS:
            uncovered.append({'sentence': index, 'detail': 'TOO_MANY_CANDIDATES'})
            continue
        before = [c for i, c in sorted(placed + [(i, c) for i, c, _ in fills], key=lambda pair: pair[0]) if i < index]
        current = [Candidate.model_validate(c) for c in [*candidates, *(c for _, c, _ in fills)]]
        value, cross = sentence_candidate(text, sentences[index][0], markers[-1], before[-1]['subject'] if before else None, current, matches)
        if value is None:
            uncovered.append({'sentence': index, 'detail': cross})
            continue
        fills.append((index, value, cross))
    return fills, uncovered


def cover_sentences(text: str, output: ExtractionOutput, context=None, trace=None, rejected=()):
    """(output, trace, record) for an accepted answer: every duty sentence of a Turkish unit covered by a duty of its own.

    `trace` is the accepted answer's grounding trace (one SOURCE_BOUND_OK entry per kept candidate, in output order,
    beside the entries of dropped or merged model candidates). A new candidate adds {'candidate', 'code':
    SOURCE_BOUND_OK, 'tier': 'structured', 'origin': 'sentence', 'sentence'}; a cut candidate's entry gets tier
    'structured' and its 'sentence'; a kept entry whose subject, action or qualifier lies in another sentence gets
    'cross_sentence' (the field names). The record is None when every duty sentence was covered already, else
    {'stage': 'grounding', 'code': DUTY_SENTENCE_COVERED, 'steps': ['sentence'], 'fields': [notes]} (plus 'uncovered'
    when some sentence stays without a duty) or {'stage', 'code': DUTY_SENTENCE_UNCOVERED, 'uncovered': [{'sentence',
    'detail'}]}. Strings and ints only (digest-safe); the output passes verify.

    v0.19 t7: `rejected` are the candidates the second reading dropped (pipeline.apply_review; their trace entries carry
    'review': 'UNSUPPORTED'). The sentence of each one's marker is not filled again (uncovered, detail REVIEW_REJECTED)
    and nothing is inherited from them; verify reads the candidates beside them, as in the answer it accepted, so a
    rejection never refuses a sibling's fill. With `rejected`, the output passes verify with them counted in.
    """
    from .classify import duty_modals
    trace = [dict(entry) for entry in trace or []]
    if output.status != 'EXTRACTED' or not output.obligations or not turkish(text):
        return output, trace, None
    matches = duty_modals(text)
    duties = duty_sentences(text, matches)
    sentences = unit_sentences(text)
    base = [c.model_dump() for c in output.obligations]
    shadows = [c.model_dump() for c in rejected]
    actions = [action_of(c) for c in base]
    cuts = {i: cut for i, c in enumerate(base) for cut in [own_sentence_action(text, c, matches, actions[:i] + actions[i + 1:])] if cut}
    chosen = None
    for use_cuts in ((True, False) if cuts else (False,)):
        candidates = [dict(c) for c in base]
        for position, cut in (cuts.items() if use_cuts else ()):
            key = 'prohibited_action' if candidates[position]['modality'].endswith('_NOT') else 'required_action'
            candidates[position] = {**candidates[position], key: cut}
        fills, uncovered = _fill(text, candidates, duties, sentences, matches, shadows)
        if not _gate_refusal(text, [*candidates, *(c for _, c, _ in fills), *shadows], context):
            chosen = (use_cuts, candidates, fills, uncovered)
            break
        if _gate_refusal(text, [*candidates, *shadows], context):
            continue                                            # these cuts break the gate: keep the actions as they were
        kept = []
        for index, value, cross in fills:
            refused = _gate_refusal(text, [*candidates, *(c for _, c, _ in kept), value, *shadows], context)
            if refused:
                uncovered.append({'sentence': index, 'detail': refused})
            else:
                kept.append((index, value, cross))
        chosen = (use_cuts, candidates, kept, sorted(uncovered, key=lambda u: u['sentence']))
        break
    if chosen is None:
        return output, trace, None
    use_cuts, candidates, fills, uncovered = chosen
    kept_entries = [e for e in trace if e.get('code') == SOURCE_BOUND_OK and 'merged_into' not in e and e.get('review') != 'UNSUPPORTED']
    entries = kept_entries if len(kept_entries) == len(candidates) else [None] * len(candidates)
    notes, number = [], max((e.get('candidate', -1) for e in trace), default=-1) + 1
    for position, candidate in enumerate(candidates):
        entry = entries[position]
        if use_cuts and position in cuts:
            key = 'prohibited_action' if candidate['modality'].endswith('_NOT') else 'required_action'
            index = _index(sentences, _alignment(text, candidate, matches)[0].start())
            notes.append({'step': 'sentence', 'candidate': entry['candidate'] if entry else position, 'field': key,
                          'model': base[position][key], 'source': candidate[key], 'sentence': index, 'detail': OTHER_SENTENCE})
            if entry is not None:
                entry.update(tier=max(entry.get('tier', 'exact'), 'structured', key=TIERS.index), sentence=index)
        fields = cross_sentence(text, candidate, matches)
        if entry is not None and fields:
            entry['cross_sentence'] = fields
    for index, value, cross in fills:
        entry = {'candidate': number, 'code': SOURCE_BOUND_OK, 'tier': 'structured', 'origin': 'sentence', 'sentence': index}
        if cross:
            entry['cross_sentence'] = cross
        trace.append(entry)
        notes.append({'step': 'sentence', 'candidate': number, 'field': 'candidate', 'model': None, 'source': action_of(value), 'sentence': index})
        number += 1
    if not notes:
        record = {'stage': 'grounding', 'code': UNCOVERED_CODE, 'uncovered': uncovered} if uncovered else None
        return output, trace, record
    covered = ExtractionOutput(status='EXTRACTED', obligations=[Candidate.model_validate(c) for c in [*candidates, *(v for _, v, _ in fills)]])
    record = {'stage': 'grounding', 'code': COVERED_CODE, 'steps': ['sentence'], 'fields': notes}
    if uncovered:
        record['uncovered'] = uncovered
    return covered, trace, record

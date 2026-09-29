import re

from .retrieval import operative_text
from .schema import ExtractionOutput

ENGLISH_MODAL = r'(?i:\b(?:must\s+not|shall\s+not|should\s+not|may\s+not|must|shall|should|may|(?:is|are)\s+required\s+to|(?:is|are)\s+prohibited\s+from)\b)'
# Turkish legislation puts its obligation marker at the END of the sentence ("... bildirmek
# zorundadır") and prohibits with the impossibility suffix ("... açıklayamazlar"). Case is
# not folded: Python's case folding maps the dotless ı wrongly, and the markers are lower
# case in the consolidated texts. Permissions ("-ebilir") are powers, not duties, and are
# deliberately not modal evidence.
TR_LETTERS = 'A-Za-zÇĞİÖŞÜçğıöşüâîû'
# A norm in Turkish drafting is also stated in the passive aorist that closes a sentence
# ("... Başkanlığa bildirilir." = shall be reported); its negation ("... yapılmaz.") is a
# prohibition. Only the sentence-final form counts, so ordinary description inside a
# sentence is not read as a duty. "uygulanmaz" closes an exception, not a duty.
TR_AORIST_END = r'(?=\s*(?:[.;]|$))'
TR_MUST = (r'zorundad[ıi]r(?:lar)?|zorunludur|y[üu]k[üu]ml[üu]d[üu]r(?:ler)?|mecbur(?:dur|idir)|gerekir|gereklidir|şartt[ıi]r|'
           r'[%s]{2,}(?:mal[ıi]d[ıi]r|melidir)|[%s]{3,}(?<!b)(?:[ıiuü]l[ıiuü]r|n[ıiuü]r)%s') % (TR_LETTERS, TR_LETTERS, TR_AORIST_END)
TR_MUST_NOT = (r'yasakt[ıi]r|[%s]{3,}(?:[aı]maz|[ei]mez)(?:lar|ler)?|(?!uygulanmaz)[%s]{3,}(?:[ıiuü]l|n)m[ae]z%s') % (TR_LETTERS, TR_LETTERS, TR_AORIST_END)
TURKISH_MODAL = re.compile(r'(?<![%s])(?:%s|%s)(?![%s])' % (TR_LETTERS, TR_MUST, TR_MUST_NOT, TR_LETTERS))
# The v0.18 marker set; every marker it finds is BINDING: verify demands that some duty covers it.
LEGACY_MODAL = re.compile(ENGLISH_MODAL + '|' + TURKISH_MODAL.pattern)

# v0.19 recall (three evidence runs, 25 September 2026): whole sub-paragraphs never reached the model
# because their duty is not phrased with the markers above. Tedbirler md. 21(3) "... kimlik bilgilerini
# üçüncü taraftan derhal alır." (active aorist), md. 24(1) "... yer verilmesi zorunlu olup ... teyit
# edilir." ("zorunlu olup"; "edilir" has a two-letter stem the passive pattern skips), md. 22(1)
# "... işlemi gerçekleştirmezler. ... hesap açamazlar." (negative aorist), md. 15 "... doğrular.",
# md. 21(2) "... emin olunması şartıyla mümkündür." (a permission granted on a condition: the
# condition is the duty) and md. 21(5) "Üçüncü tarafa güven ilkesi, ... durumunda uygulanmaz." (a
# limit on a permission, not a scope exemption). These SOFT markers make a unit a duty for the
# classifier, let a candidate align with them and forbid dismissing the unit, but verify never
# demands that they be covered and demands their qualifiers only from a candidate aligned with them:
# no answer the v0.18 gate accepted is rejected because of them (classify.duty_modals filters them).
TR_SOFT_STANDALONE_WORDS = (r'(?:zorunlu|zorunda|y[üu]k[üu]ml[üu]|mecbur)(?=\s+olup(?![%s]))|gerekmektedir|mükellef(?:tir|dir)(?:ler)?|'
                            r'esast[ıi]r' % TR_LETTERS)
# A condition of a permission ("... şartıyla mümkündür", "yalnızca ... halinde ... yapabilir"): the
# condition clause before it is what the firm must do. The permission predicate closes the sentence.
TR_PERMISSION_END = r'(?:mümkündür|serbesttir|[%s]{2,}[ae]bilir(?:ler)?)\s*(?:[.;]|$)' % TR_LETTERS
TR_PERMISSION_CONDITION = (r'(?:şartıyla|kaydıyla|koşuluyla|h[aâ]linde|h[aâ]llerinde|takdirde|durumunda)(?=[^.;]{0,600}?(?<![%s])%s)'
                           % (TR_LETTERS, TR_PERMISSION_END))
# What the obliged party does, in the aorist ("alır", "saklar", "bildirir", "teyit eder", "iade eder",
# "bilgi verir", "bildirim yapar") or the plural passive aorist ("bulunurlar"). "ifade eder" defines,
# "kabul eder/edilir" deems, "izin verir" permits: none of them is a duty.
TR_ACTIVE_AORIST = (r'(?<!ifade )(?<!kabul )(?<!izin )(?:[%s]*eder|alır|verir|yapar|sağlar|saklar|tutar|uygular|izler|doğrular|'
                    r'ister|sunar|öder|iletir|gönderir|gösterir|tamamlar|[%s]{2,}[dt][ıiuü]r[ıiuü]r|'
                    r'[%s]{3,}(?<!b)(?:[ıiuü]l[ıiuü]r|n[ıiuü]r)(?=lar|ler))(?:lar|ler)?%s'
                    % (TR_LETTERS, TR_LETTERS, TR_LETTERS, TR_AORIST_END))
TR_SOFT_PASSIVE = r'(?<!kabul )(?<!ifade )(?:edilir|olunur)%s' % TR_AORIST_END
# Negative aorist ("gerçekleştirmezler", "tesis etmez", "açamazlar"). A negation that only says a
# rule is not removed, affected or required ("ortadan kaldırmaz", "etkilemez", "gerekmez") restricts
# nothing. "uygulanmaz" is kept apart: classify.duty_modals drops it where it exempts a provision.
TR_NEGATIVE_AORIST = (r'(?!(?:kaldırmaz|etkilemez|gerektirmez|gerekmez|değişmez|uygulanmaz)(?![%s]))(?<!teşkil )(?<!arz )(?<!halel )'
                      r'[%s]{2,}m[ae]z(?:lar|ler)?%s' % (TR_LETTERS, TR_LETTERS, TR_AORIST_END))
# "uygulanmaz" right after a condition ("... riskli ülkelerde yerleşik olması durumunda uygulanmaz") limits
# what the sentence's subject allows; "Bu hükümler ... uygulanmaz" exempts and stays an exception.
TR_LIMIT = (r'(?:(?<=durumunda )|(?<=durumlarında )|(?<=halinde )|(?<=hâlinde )|(?<=hallerinde )|(?<=hâllerinde )|(?<=takdirde ))'
            r'uygulanmaz%s' % TR_AORIST_END)
TR_SOFT_MUST_NOT = r'(?:edilmez|olunmaz)%s|%s|%s' % (TR_AORIST_END, TR_NEGATIVE_AORIST, TR_LIMIT)
TR_SOFT = (r'(?<![%s])(?:%s|%s|%s|%s|%s)(?![%s])'
           % (TR_LETTERS, TR_SOFT_STANDALONE_WORDS, TR_PERMISSION_CONDITION, TR_SOFT_PASSIVE, TR_ACTIVE_AORIST, TR_SOFT_MUST_NOT, TR_LETTERS))
# English: "needs to", "has to", "is obliged to", and "will (not)" after a firm-type subject ("the firm
# will notify ..."); "must", "should", "may not", "is required to" are already binding.
EN_SOFT = (r'(?i:\b(?:needs?\s+to|(?:has|have)\s+to|(?:is|are)\s+obliged\s+to)\b|'
           r'(?:(?<=\bfirm\s)|(?<=\bfirms\s)|(?<=\blender\s)|(?<=\blenders\s))will(?:\s+not)?\b)')
MODAL = re.compile(ENGLISH_MODAL + '|' + TURKISH_MODAL.pattern + '|(?P<soft>' + EN_SOFT + '|' + TR_SOFT + ')')
# The same markers read without their sentence (turkish_modal / modality get the marker text only).
TR_SOFT_WORD = re.compile(r'(?:%s|şartıyla|kaydıyla|koşuluyla|h[aâ]linde|h[aâ]llerinde|takdirde|durumunda|zorunlu|zorunda|y[üu]k[üu]ml[üu]|mecbur|'
                          r'%s|%s|%s|uygulanmaz)' % (TR_SOFT_STANDALONE_WORDS, TR_SOFT_PASSIVE, TR_ACTIVE_AORIST, TR_SOFT_MUST_NOT))
TR_SOFT_WORD_MUST_NOT = re.compile(r'(?:%s|uygulanmaz)' % TR_SOFT_MUST_NOT)
TR_SOFT_STANDALONE = re.compile(r'zorunlu|zorunda|y[üu]k[üu]ml[üu]|mecbur|gerekmektedir|mükellef(?:tir|dir)(?:ler)?|esast[ıi]r|'
                                r'şartıyla|kaydıyla|koşuluyla|h[aâ]linde|h[aâ]llerinde|takdirde|durumunda')
EN_SOFT_WORD = re.compile(r'(?i:needs?\s+to|(?:has|have)\s+to|(?:is|are)\s+obliged\s+to|will(?:\s+not)?)')


def binding(match: re.Match) -> bool:
    """Whether a duty marker is one verify demands be covered (a v0.18 marker), not a v0.19 soft one."""
    try:
        return match.group('soft') is None
    except IndexError:
        return True
EXCEPTIONS = re.compile(r'\b(unless|except(?:\s+where)?|other\s+than|save\s+where|without|does\s+not\s+apply|do\s+not\s+apply)\b', re.I)
CONDITIONS = re.compile(r'\b(if|where|when|provided\s+that|subject\s+to)\b', re.I)
# Turkish qualifiers are postpositions: the clause they govern comes BEFORE the marker.
TR_CONDITIONS = re.compile(r'(?<![%s])(?:h[aâ]linde|h[aâ]llerinde|takdirde|durumunda|durumlarında|kaydıyla|şartıyla|koşuluyla|sürece)(?![%s])' % (TR_LETTERS, TR_LETTERS))
TR_EXCEPTIONS = re.compile(r'(?<![%s])(?:hariç(?:tir)?|dışında(?:dır)?|istisna(?:dır|sıdır)?|saklıdır|uygulanmaz)(?![%s])' % (TR_LETTERS, TR_LETTERS))
TR_CHARACTERS = re.compile('[çğıöşüÇĞİÖŞÜ]')
# A separate marker word ("bildirmek zorundadır") against a verb that carries the duty itself
# ("bildirilir", "açıklayamazlar"): the action stops before the first, may end with the second.
TR_STANDALONE = re.compile(r'zorunda\w*|zorunludur|y[üu]k[üu]ml[üu]d[üu]r\w*|mecbur\w*|gerekir|gereklidir|şartt[ıi]r|yasakt[ıi]r')
# "bu Kanunun 4 üncü maddesi" cites this document; "5237 sayılı Türk Ceza Kanununun 282 nci
# maddesi" cites another statute, which no snapshot of this document could ever resolve.
TR_ARTICLE_REF = re.compile(r'(\d+(?:/[A-ZÇĞİÖŞÜ])?)\s*(?:inci|nci|ncı|ıncı|üncü|uncu|ncu|ncü|inci)\s+madde')
TR_THIS_DOCUMENT = re.compile(r'(?<![%s])[Bb]u (?:Kanun|Yönetmeli|Tebliğ|Kararname|Tüzü)' % TR_LETTERS)
TR_OTHER_DOCUMENT = re.compile(r'sayılı|(?<![Bb]u )(?<![%s])(?:Kanun|Yönetmeli|Tebliğ|Kararname|Tüzü)' % TR_LETTERS)


def turkish(text: str) -> bool:
    """Turkish source text: its modal evidence is read in Turkish word order."""
    return bool(TURKISH_MODAL.search(text)) or len(TR_CHARACTERS.findall(text)) >= 3


def turkish_modal(modal_text: str) -> bool:
    return TURKISH_MODAL.fullmatch(modal_text) is not None or TR_SOFT_WORD.fullmatch(modal_text) is not None


def standalone(modal_text: str) -> bool:
    """A separate marker word the Turkish action stops before ("zorundadır", "şartıyla", "mükelleftir")."""
    return TR_STANDALONE.fullmatch(modal_text) is not None or TR_SOFT_STANDALONE.fullmatch(modal_text) is not None


def turkish_action_fits(action: str, action_end: int, match: re.Match) -> bool:
    """Whether an action ending at action_end sits against its Turkish modal.

    Measured with qwen3:4b: the model copies the predicate up to and including the marker
    and the full stop ("... ibraz etmekle yükümlüdür.") as often as it stops before the
    marker. Both are exact copies of the same duty, so both are accepted; a fragment that
    ends anywhere else is not.
    """
    if match.start() - 2 <= action_end <= match.start():
        return True
    if action_end == match.end():
        return True
    return action_end == match.end() + 1 and action.endswith(('.', ';'))


def turkish_polarity(text: str, action: str) -> str | None:
    """MUST or MUST_NOT from the Turkish modal the action sits against, else None."""
    for match in (m for m in MODAL.finditer(text) if turkish_modal(m.group())):
        pos = text.rfind(action, 0, match.end() + 1)
        if pos >= 0 and turkish_action_fits(action, pos + len(action), match):
            return modality(match.group())
    return None


def turkish_references(text: str) -> list[tuple[re.Match, str, bool]]:
    """(match, article number, internal) for every 'N inci madde' citation in the text."""
    found = []
    for match in TR_ARTICLE_REF.finditer(text):
        before = text[max(0, match.start()-70):match.start()]
        # Cues from an earlier sentence say nothing about this citation.
        before = before[before.rfind('. ')+2:] if '. ' in before else before
        # The nearest cue decides: "bu Kanunun 4 üncü maddesi ile 5237 sayılı Türk Ceza
        # Kanununun 282 nci maddesi" cites this document and then another one.
        this = max((m.end() for m in TR_THIS_DOCUMENT.finditer(before)), default=-1)
        other = max((m.end() for m in TR_OTHER_DOCUMENT.finditer(before)), default=-1)
        found.append((match, match.group(1), this >= other))
    return found
# "section 138D of the Act" cites a statute. It can never be resolved against Handbook
# sources, so treating it as an unresolved Handbook reference blocked the provision for
# good. The Handbook forms (a dotted locator, a module code) are unaffected.
CROSS_REFERENCE = re.compile(r'\b(?:in accordance with|as defined in|subject to (?:rule|section)|paragraph\s+\d|'
    r'section\s+\d(?![0-9A-Z]*\s+of\s+the\s+(?:Act|Order|Regulations)\b)|(?:CONC|COBS|MCOB|SYSC|PRIN|SUP)\s+\d)', re.I)


def modality(text: str) -> str:
    if re.fullmatch(r'(?:is|are)\s+required\s+to',text,re.I):
        return 'MUST'
    if re.fullmatch(r'(?:is|are)\s+prohibited\s+from',text,re.I):
        return 'MUST_NOT'
    if turkish_modal(text):
        return 'MUST_NOT' if re.fullmatch(TR_MUST_NOT, text) or TR_SOFT_WORD_MUST_NOT.fullmatch(text) else 'MUST'
    if EN_SOFT_WORD.fullmatch(text):
        return 'MUST_NOT' if re.fullmatch(r'will\s+not', text, re.I) else 'MUST'
    return re.sub(r'\s+', '_', text.upper()).replace('SHALL','MUST')


def qualifier_pattern(text: str, pattern: re.Pattern) -> re.Pattern:
    """The Turkish counterpart of an English qualifier pattern for Turkish source text."""
    if not turkish(text):
        return pattern
    return TR_CONDITIONS if pattern is CONDITIONS else TR_EXCEPTIONS if pattern is EXCEPTIONS else pattern


def qualifier_tails(text: str, pattern: re.Pattern) -> list[str]:
    # Conservative full tails preserve nested exceptions and subsequent sentences.
    # These are evidence spans, not a claim of complete legal scope resolution.
    if turkish(text):
        return qualifier_spans(text, pattern)
    return [text[match.start():] for match in pattern.finditer(text)]


def turkish_qualifier_spans(text: str, pattern: re.Pattern) -> list[str]:
    """The clause a Turkish postposition closes: from the last clause boundary to the marker.

    "Yükümlüler, ... şüphelenilmesi hâlinde, ... bildirmek zorundadır": the condition ends
    with "hâlinde" and starts after the previous comma, semicolon, bracket or sentence
    end. A model's condition must contain this span, so a longer copy is still valid.
    """
    spans = []
    for match in pattern.finditer(text):
        start = 0
        for boundary in re.finditer(r'[,;:(]\s*|[.!?]\s+(?=[A-ZÇĞİÖŞÜ(])', text[:match.start()]):
            start = boundary.end()
        span = text[start:match.end()].strip()
        if len(span.split()) >= 2:
            spans.append(span)
    return spans


# v0.19 (independent run, 24 September 2026): three artefacts of the qualifier spans that
# verify demanded and no correct answer could supply. The spans the rules baseline and the
# providers copy (qualifier_spans / qualifier_tails) are unchanged; only the demand is cleaned,
# and every cleaned demand is a substring of the old one or dropped, so no answer the gate
# accepted before is rejected now.
# "Yurt dışında yerleşik ..." = resident abroad: a place, not an exception (spans.exception_ends).
ABROAD = re.compile(r'(?<![%s])(?:[Yy]urt|YURT)\s+$' % TR_LETTERS)
# "(2) Yukarıdaki ... kaydıyla": the "(" clause boundary leaves "2) " of the sub-paragraph number,
# and "(Değişik: 28/12/2009-2009/15720 K.) ..." leaves the amendment note, at the start of the span.
LEADING_REMNANT = re.compile(r'^(?:[a-zçğıöşü]|[^()]{0,40}?\d[^()]{0,40}?)\)\s*')
LETTER_AFTER = re.compile('[%s]' % TR_LETTERS)
TR_SENTENCE_END = re.compile(r'[.!?](?:\s+(?=[A-ZÇĞİÖŞÜ(])|\s*$)')
TR_INFINITIVE_AND = re.compile(r'(?<=[%s]m[ae]k) (?:ve|veya) ' % TR_LETTERS)


def abroad(text: str, match: re.Match) -> bool:
    """Whether a Turkish 'dışında' exception marker is 'yurt dışında' (abroad)."""
    return match.group().startswith('dışında') and ABROAD.search(text[max(0, match.start() - 8):match.start()]) is not None


def _sentence(text: str, pos: int) -> tuple[int, int]:
    start = 0
    for end in TR_SENTENCE_END.finditer(text):
        if end.end() > pos:
            return start, end.end()
        start = end.end()
    return start, len(text)


def _leading_clause_cut(text: str, span: str, output) -> str:
    """An English leading qualifier that runs through the duty it governs, cut before its subject.

    "When levying charges ... difficulties firms should consider ...": with no comma the span ran
    to the sentence end and demanded the duty itself as its own condition (CONC 7.7.1). When a
    candidate's subject stands right before a modal of its modality inside the span, and its
    action follows that modal, the qualifier is the text before that subject.
    """
    pos = text.find(span)
    if pos < 0 or output is None or getattr(output, 'status', None) != 'EXTRACTED':
        return span
    for candidate in output.obligations:
        subject = candidate.subject.strip()
        action = candidate.prohibited_action if candidate.modality.endswith('_NOT') else candidate.required_action
        for match in re.finditer(ENGLISH_MODAL, span):
            if modality(match.group()) != candidate.modality or not action or action not in text[pos + match.end():]:
                continue
            before = span[:match.start()].rstrip()
            if not subject or not before.endswith(subject) or not before[:-len(subject)][-1:].isspace():
                continue
            cut = re.sub(r'(?:[\s,]+(?:and|or))+$', '', before[:-len(subject)].rstrip(' ,'), flags=re.I).rstrip(' ,')
            if len(cut.split()) >= 2 and not MODAL.search(cut):
                return cut
    return span


def required_qualifiers(text: str, pattern: re.Pattern, output=None, matches=None) -> list[str]:
    """The primary-source qualifier spans verify demands in `conditions` / `exceptions`.

    qualifier_spans, cleaned: a Turkish 'yurt dışında' is no exception; a leading sub-paragraph
    number or amendment-note remnant is not part of a qualifier; a Turkish qualifier inside a
    sentence with no duty marker ("... halinde ise ... yer verilebilir") qualifies that sentence's
    own predicate; an English leading clause stops before the subject of the duty it governs.
    """
    active = qualifier_pattern(text, pattern)
    if active not in (TR_CONDITIONS, TR_EXCEPTIONS):
        return [_leading_clause_cut(text, span, output) for span in qualifier_spans(text, pattern)]
    if matches is None:
        from .classify import duty_modals
        matches = duty_modals(text)
    # v0.19: a soft marker demands the qualifiers of its sentence only once a candidate is aligned with
    # it, and a qualifier word that is itself the marker ("... emin olunması şartıyla mümkündür",
    # "... durumunda uygulanmaz") is the duty's end, not a separate qualifier.
    markers = {m.start() for m in matches if not binding(m)}
    anchored = anchored_markers(text, output, matches) if output is not None else set()
    matches = [m for m in matches if binding(m) or m.start() in anchored]
    spans = []
    for match in active.finditer(text):
        if match.start() in markers:
            continue
        start = 0
        for boundary in re.finditer(r'[,;:(]\s*|[.!?]\s+(?=[A-ZÇĞİÖŞÜ(])', text[:match.start()]):
            start = boundary.end()
        span = text[start:match.end()].strip()
        if len(span.split()) < 2 or (active is TR_EXCEPTIONS and abroad(text, match)):
            continue
        span = LEADING_REMNANT.sub('', span, count=1).strip()
        # v0.19 t5 (Tedbirler md. 46(1)): "... sekiz yıl süre ile muhafaza etmek ve istenmesi halinde
        # yetkililere ibraz etmek zorundadır." A condition coordinated after an infinitive starts after
        # "ve"; the demand was half of the duty's own action. The cut demand is a substring of the old one.
        cut = max((m.end() for m in TR_INFINITIVE_AND.finditer(span)), default=0)
        if cut and len(span[cut:].split()) >= 2:
            span = span[cut:]
        if len(span.split()) < 2:
            continue
        # A qualifier that is not its sentence's own predicate ("... hükümleri saklıdır.") governs
        # the predicate after it; in a sentence with no duty marker that is a permission or another
        # statement ("... halinde ise ... yer verilebilir."), never a duty this gate checks.
        s0, s1 = _sentence(text, match.start())
        if not any(s0 <= m.start() < s1 for m in matches) and LETTER_AFTER.search(text, match.end(), s1):
            continue
        spans.append(span)
    return spans


def missing_qualifiers(text: str, output, matches=None) -> list[tuple[str, str]]:
    """(field, span) for every demanded primary-source qualifier no candidate carries."""
    missing = []
    for pattern, field in [(EXCEPTIONS, 'exceptions'), (CONDITIONS, 'conditions')]:
        values = [span for c in output.obligations for span in getattr(c, field)]
        missing += [(field, tail) for tail in required_qualifiers(text, pattern, output, matches)
                    if not any(tail in span for span in values)]
    return missing


def qualifier_spans(text: str,pattern: re.Pattern) -> list[str]:
    """Complete sentence tails rather than the remainder of an entire document block."""
    pattern = qualifier_pattern(text, pattern)
    if pattern in (TR_CONDITIONS, TR_EXCEPTIONS):
        return turkish_qualifier_spans(text, pattern)
    spans=[]
    for match in pattern.finditer(text):
        # Editorial source notes are preserved in source_quote, not converted into duties.
        if text.rfind('[Note:',0,match.start())>text.rfind(']',0,match.start()):
            continue
        # "(" also opens a sentence: FCA sub-paragraphs read "... arrangement. (2) The firm
        # must ...". Without it the span ran on into the next sub-paragraph and demanded
        # the duty itself as part of its own condition (CONC 7.3.5G could never pass).
        end=re.search(r'[.!?](?:\s+(?=[A-Z\[(])|$)',text[match.end():])
        stop=match.end()+end.start()+1 if end else len(text)
        # A leading subordinate clause ends at the comma before the governing
        # duty: "When assessing risk, a firm must ...". Requiring the entire
        # sentence as a condition incorrectly includes the action itself.
        comma=text.find(',',match.end(),stop)
        if comma>=0 and not MODAL.search(text[:comma]) and MODAL.search(text[comma:stop]):
            stop=comma
        spans.append(text[match.start():stop])
    return spans


def preflight(text: str, context=None) -> str | None:
    if not text.strip() or len(text) > 12000:
        return 'SOURCE_SIZE_OR_EMPTY'
    if context and context.source_kind == 'AMENDMENT':
        return 'AMENDMENT_REQUIRES_CONSOLIDATED_SOURCE'
    if context and 'OMITTED_TEXT_MARKER' in context.quality_flags:
        return 'SOURCE_CONTAINS_OMITTED_TEXT'
    if context and 'POSSIBLE_WRAPPED_LABEL' in context.quality_flags:
        return 'AMBIGUOUS_PRINTED_LABEL'
    if context and 'DELETED_PROVISION' in context.quality_flags:
        return 'DELETED_PROVISION'
    if context and context.missing_sources:
        # The cited document exists and simply has not been ingested; say so.
        return 'MISSING_CONSOLIDATED_SOURCE'
    if context and (context.unresolved or context.truncated):
        return 'UNRESOLVED_CROSS_REFERENCE' if context.unresolved else 'CONTEXT_BUDGET_EXCEEDED'
    # Editorial Notes cite superseded predecessor guidance, not operative provisions.
    resolved = bool(context and any(i['reason'] in ('exact_reference','chapter_reference','glossary_definition') for i in context.items))
    if CROSS_REFERENCE.search(operative_text(text)) and not resolved:
        return 'UNRESOLVED_CROSS_REFERENCE'
    if any(internal for _, _, internal in turkish_references(operative_text(text))) and not resolved:
        return 'UNRESOLVED_CROSS_REFERENCE'
    return None


def alignment_hint(text: str, matches: list, mode: str) -> str:
    """Where the action has to sit, for the repair prompt.

    Measured on Kanun 5549 md. 4(1) (22 September 2026): the model copied the conditional
    clause and stopped before "... Başkanlığa bildirilmesi zorunludur", was told only that
    subject, modality and action "must align with source", and made the same cut again.
    The words that precede each marker say exactly where the copy has to end.
    """
    hints = []
    for match in matches:
        if modality(match.group()) != mode:
            continue
        if turkish_modal(match.group()):
            before = text[:match.start()].rstrip()
            tail = ' '.join(before.split()[-8:])
            if standalone(match.group()):
                hints.append(f"for the marker '{match.group()}' the action is the predicate that ends right before it: '... {tail}'")
            else:
                hints.append(f"for the marker '{match.group()}' the action is the predicate ending with that verb: '... {tail} {match.group()}'")
        else:
            after = text[match.end():].strip()
            hints.append(f"after '{match.group()}' the action starts with: '{' '.join(after.split()[:8])}'")
    return '; '.join(hints[:3])


def alignments(text: str, candidate, matches) -> list[tuple[re.Match, int]]:
    """(marker, action start) for every marker of the candidate's modality its subject and action align with.

    English: subject, modal, action. Turkish: subject, action, modal ("... bildirmek zorundadır"), so
    the action is looked for before its modal and must end at it.
    """
    action = candidate.prohibited_action if candidate.modality.endswith('_NOT') else candidate.required_action
    if not action:
        return []
    return [(match, pos) for match in matches if candidate.modality == modality(match.group())
            and candidate.subject in text[:match.start()]
            for pos in [text.rfind(action, 0, match.end()+1) if turkish_modal(match.group()) else text.find(action, match.end())]
            if pos >= 0 and (not turkish_modal(match.group()) or turkish_action_fits(action, pos+len(action), match))]


def nearest(aligned, action: str) -> tuple[re.Match, int]:
    """The marker a candidate belongs to: the nearest one, a v0.18 (binding) marker before a soft one."""
    return min(aligned, key=lambda pair: (not binding(pair[0]), (pair[0].start()-pair[1]-len(action)) if turkish_modal(pair[0].group())
                                          else (pair[1]-pair[0].end())))


def anchored_markers(text: str, output, matches) -> set[int]:
    """Starts of the soft markers some candidate of `output` is aligned with (verify's own choice)."""
    starts = set()
    for candidate in getattr(output, 'obligations', None) or []:
        aligned = alignments(text, candidate, matches)
        if aligned:
            match, _ = nearest(aligned, candidate.prohibited_action if candidate.modality.endswith('_NOT') else candidate.required_action)
            if not binding(match):
                starts.add(match.start())
    return starts


def failing_candidate(text: str, output, candidate, matches) -> str:
    """Which candidate the alignment failure is about, for the repair prompt.

    v0.19 t5 (Tedbirler md. 46(1), v019t4 runs): the answer had a correct first duty and a second
    "duty" copied from "... başlangıç tarihi hesabın kapatıldığı tarihtir.", a sentence with no marker.
    The hint named only the first duty's marker, and the model returned the same answer twice.
    """
    action = candidate.prohibited_action if candidate.modality.endswith('_NOT') else candidate.required_action
    number = next((i for i, c in enumerate(output.obligations) if c is candidate), 0) + 1
    note = f' [candidate {number} of {len(output.obligations)}: action "{" ".join((action or "").split()[:12])}"'
    at = text.find(action) if action else -1
    if at >= 0:
        s0, s1 = _sentence(text, at)
        if not any(s0 <= m.start() < s1 for m in matches):
            note += '; no duty marker occurs in its sentence: if that sentence states no duty (a definition, a date, a scope), leave this candidate out'
    return note + ']'


def verify(text: str, output: ExtractionOutput, context=None) -> None:
    # A sentence that hands the details to a by-law ("... yönetmelikle belirlenir") ends
    # with a passive aorist that the modal pattern reads as a duty; it is no duty of a firm
    # and is neither extracted nor required to be covered.
    from .classify import duty_modals
    matches = duty_modals(text)
    if output.status == 'NO_EXPLICIT_OBLIGATION' and matches:
        raise ValueError('Cannot dismiss source containing modal language')
    if output.status != 'EXTRACTED':
        return
    if not matches:
        raise ValueError('No explicit modal evidence in source')
    covered=set()
    identities=set()
    for candidate in output.obligations:
        contexts = {item['section_id']:item for item in context.items} if context else {}
        supporting_text = []
        for evidence in candidate.supporting_evidence:
            item = contexts.get(evidence.section_id)
            if item is None or evidence.quote != item['text']:
                raise ValueError('Supporting citation must match a supplied context paragraph exactly')
            supporting_text.append(evidence.quote)
        if candidate.source_quote != text:
            raise ValueError('Quote must preserve the full supplied paragraph')
        action = candidate.prohibited_action if candidate.modality.endswith('_NOT') else candidate.required_action
        aligned = alignments(text, candidate, matches)
        if not aligned:
            hint = alignment_hint(text, matches, candidate.modality)
            raise ValueError('ACTION_ALIGNMENT: subject, modality and action must align with source'
                             + (f'. The subject must occur before the marker and the action must be copied exactly: {hint}' if hint
                                else f'. No marker of modality {candidate.modality} occurs in the source')
                             + failing_candidate(text, output, candidate, matches))
        match,action_start=nearest(aligned, action)
        identity=(match.start(),action_start,action)
        if identity in identities:
            raise ValueError('DUPLICATE_DUTY: repeated action under the same modal')
        identities.add(identity)
        # Embedded modal wording inside the action is covered by that action, not
        # automatically a separate duty. One modal may govern several list actions.
        covered.update(m.start() for m in matches if m==match or action_start<=m.start()<action_start+len(action))
        fields = [candidate.subject, action, candidate.deadline, *candidate.conditions,
                  *candidate.exceptions, *candidate.affected_products,
                  *candidate.affected_entities, *candidate.customer_types]
        if any(value is not None and (not value.strip() or not any(value in source for source in [text,*supporting_text])) for value in fields):
            raise ValueError('Extracted field has no exact source evidence')
        for span in [*candidate.conditions,*candidate.exceptions]:
            pos=text.find(span)
            if pos>=0:
                covered.update(m.start() for m in matches if pos<=m.start()<pos+len(span)
                               and modality(m.group()) in ('MAY','MAY_NOT','SHOULD','SHOULD_NOT'))
        # Subject before and action after this modal (before it in Turkish): no arbitrary snippets.
        if candidate.subject not in text[:match.start()] or action not in (text[:match.end()+1] if turkish_modal(match.group()) else text[match.end():]):
            raise ValueError('Subject/action do not align with the modal evidence')
        for item in contexts.values():
            # Only direct required references or explicitly cited context govern a
            # candidate. Optional neighbors must not invent conditions on the duty.
            if not item.get('required',item['reason']=='exact_reference') and not any(
                    e.section_id==item['section_id'] for e in candidate.supporting_evidence):
                continue
            for pattern,role,values in [(EXCEPTIONS,'exception',candidate.exceptions),(CONDITIONS,'condition',candidate.conditions)]:
                tails = qualifier_spans(item['text'],pattern)
                if tails and (not any(e.section_id==item['section_id'] and e.role==role for e in candidate.supporting_evidence)
                              or any(not any(tail in value for value in values) for tail in tails)):
                    raise ValueError('Contextual qualifier lacks an attributed evidence span')
    # v0.19 soft markers are not demanded: the gate stays exactly as strict as v0.18 on its own markers.
    if any(m.start() not in covered for m in matches if binding(m)):
        raise ValueError('UNCOVERED_MODAL: explicit modal evidence is not covered by any action')
    # Coverage is checked across the extracted duties, not by attaching every
    # sentence's condition to every duty. Semantic assignment is checked in review.
    missing = missing_qualifiers(text, output, matches)
    if missing:
        field = missing[0][0]
        raise ValueError('MISSING_QUALIFIER: '+field+' must include these exact primary-source spans: '
                         +repr([span for name, span in missing if name == field]))

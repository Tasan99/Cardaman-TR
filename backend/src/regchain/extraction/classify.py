"""Deterministic pre-classification of a regulatory unit before any model call.

A statute mixes duties with text that only looks like a duty to a modal regex: a
definitions list, a scope article, a sentence that hands the details to a by-law
("... usûl ve esaslar yönetmelikle belirlenir"), a penalty article, a permission. Measured
live on Kanun 5549 (22 September 2026): md. 4(3) "yönetmelikle belirlenir" was extracted
and kept as a MUST duty of the obliged parties, while the real reporting duty in md. 4(1)
was rejected. The classification here is read from the wording alone, is recorded with
the marker that decided it, and never asks a model. Only units classified as duty-bearing
are sent to extraction; everything else is reported with its class as the reason.

Classes: OBLIGATION, PROHIBITION, RECOMMENDATION, GUIDANCE, PERMISSION, EXEMPTION,
DEFINITION, SCOPE, DELEGATION, ENFORCEMENT, OTHER.
"""
import re
from dataclasses import dataclass

from .grounding import EN_SOFT, ENGLISH_MODAL, MODAL, TR_LETTERS, TR_STANDALONE, TURKISH_MODAL, _sentence, binding, modality, turkish

CLASSES = ('OBLIGATION', 'PROHIBITION', 'RECOMMENDATION', 'GUIDANCE', 'PERMISSION', 'EXEMPTION',
           'DEFINITION', 'SCOPE', 'DELEGATION', 'ENFORCEMENT', 'OTHER')
# Units of these classes carry no duty for a firm and are never sent to extraction.
NOT_EXTRACTED = frozenset({'DEFINITION', 'SCOPE', 'DELEGATION', 'ENFORCEMENT', 'PERMISSION', 'EXEMPTION'})

TR_SENTENCE = re.compile(r'(?<=[.;!?])\s+(?=[A-ZÇĞİÖŞÜ(])')
# A sentence that delegates: it ends with one of these passive verbs ...
TR_DELEGATION_END = re.compile(r'(?<![%s])(?:belirlenir|belirlenebilir|düzenlenir|tespit edilir|gösterilir|çıkarılır|'
                               r'tayin edilir|saptanır|tespit olunur|belirlemeye yetkilidir|yetkilidir|yetkili olup|'
                               r'yetkili kılınmıştır|yetkiye sahiptir)\s*$' % TR_LETTERS)
# ... and names the instrument or authority that will decide (only for the verbs that also
# state an ordinary duty, such as "tespit edilir": "kimlik tespit edilir" is a duty of the
# firm, "usûl ve esaslar Bakanlıkça tespit edilir" is not).
TR_DELEGATION_AGENT = re.compile(r'(?:yönetmelik|tebliğ|genelge|karar|esaslar|usul|usûl)(?:le|la|te|ta|de|da|ler(?:le|de)|lar(?:la|da))(?![%s])|'
                                 r'(?:Cumhurbaşkan|Bakanl|Kurul|Başkanl|Kurum|Müdürlü|Bakanlar Kurulu|Merkez Bankas|Hazine)[%s]*(?:ca|ce|ça|çe)(?![%s])|'
                                 r'(?:Cumhurbaşkanı|Bakanlık|Bakan|Kurul|Başkanlık|Kurum|Bakanlar Kurulu|Merkez Bankası|Hazine) tarafından'
                                 % (TR_LETTERS, TR_LETTERS, TR_LETTERS))
TR_AUTHORITY_SUBJECT = re.compile(r'yetkili(?:dir| olup| kılınmıştır)|yetkiye sahiptir')
TR_DEFINITION = re.compile(r'^\s*(?:\(\d+\)\s*)?Bu (?:Kanun|Yönetmeli|Tebliğ|Kararname|Tüzü)[%s]*\s+(?:uygulanmasında\s+)?geçen|'
                           r'(?<![%s])(?:ifade eder|anlamına gelir|ifade etmektedir|anlaşılır)(?![%s])' % (TR_LETTERS, TR_LETTERS, TR_LETTERS))
TR_SCOPE = re.compile(r'^\s*(?:\(\d+\)\s*)?Bu (?:Kanun|Yönetmeli|Tebliğ|Kararname|Tüzü)[%s]*[^.]*?(?:kapsar|kapsamaz|hakkında uygulanır|uygulanır|'
                      r'kapsamındadır|kapsamına girer|hükümlerine tâbidir|hükümlerine tabidir|amacı)' % TR_LETTERS)
TR_ENFORCEMENT = re.compile(r'idar[iî] para cezası|adl[iî] para cezası|hapis cezası|hapis ve|cezalandırılır|ceza verilir|para cezası verilir|'
                            r'cezası uygulanır|ile cezalandırılır')
TR_EXEMPTION = re.compile(r'(?<![%s])(?:uygulanmaz|muaftır|muaf tutulur|hariçtir|istisnadır|saklıdır|bağışıktır)(?![%s])' % (TR_LETTERS, TR_LETTERS))
TR_PERMISSION = re.compile(r'(?<![%s])(?:[%s]{2,}(?:[ae]bilir(?:ler)?)|izin verir(?:ler)?|mümkündür)(?![%s])' % (TR_LETTERS, TR_LETTERS, TR_LETTERS))
TR_HEADINGS = (('tanım', 'DEFINITION'), ('kapsam', 'SCOPE'), ('yükümlüler', 'SCOPE'), ('amaç', 'SCOPE'),
               ('ceza', 'ENFORCEMENT'), ('yaptırım', 'ENFORCEMENT'), ('dayanak', 'OTHER'), ('yürürlük', 'OTHER'), ('yürütme', 'OTHER'))

EN_SCOPE = re.compile(r'^\s*(?:\(\d+\)\s*)?(?:This (?:chapter|section|rule|Part|sourcebook|guidance|provision|paragraph)|'
                      r'The (?:rules|provisions|guidance) in this (?:chapter|section))\s+(?:does not apply|do not apply|applies|apply|is relevant|are relevant)', re.I)
EN_DEFINITION = re.compile(r'\b(?:in this (?:chapter|section|sourcebook|Part)|for the purposes? of this (?:chapter|section|rule))\b[^.]*\b(?:means|has the (?:same )?meaning|is defined|are defined)\b|'
                           r'^\s*(?:\(\d+\)\s*)?["“]?[A-Za-z][^.]{0,60}["”]?\s+(?:means|has the same meaning as)\b', re.I)
EN_ENFORCEMENT = re.compile(r'\b(?:is liable to|commits an offence|penalty|penalties|financial penalty|is guilty of an offence)\b', re.I)
EN_DELEGATION = re.compile(r'\b(?:the FCA|the PRA|the Treasury|the regulator|the appropriate regulator)\s+(?:may|will|shall)\s+(?:make|specify|determine|prescribe|direct|issue)\b', re.I)


@dataclass(frozen=True)
class Classification:
    kind: str
    marker: str
    reason: str

    @property
    def extractable(self) -> bool:
        return self.kind not in NOT_EXTRACTED


def turkish_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence, so a marker can be tied to the sentence it closes."""
    spans, start = [], 0
    for boundary in TR_SENTENCE.finditer(text):
        spans.append((start, boundary.start()))
        start = boundary.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def delegation_spans(text: str) -> list[tuple[int, int]]:
    """Sentences that hand the details to a by-law or an authority, in Turkish drafting.

    "... usûl ve esaslar yönetmelikle belirlenir." and "... belirlemeye Bakanlık yetkilidir."
    impose nothing on a firm. A sentence that also carries a standalone duty marker
    ("zorundadır", "yükümlüdür") is not a delegation, whatever it ends with.
    """
    if not turkish(text):
        return []
    found = []
    for start, end in turkish_sentences(text):
        sentence = text[start:end]
        if TR_STANDALONE.search(sentence):
            continue
        tail = sentence.rstrip(' .;')
        if not TR_DELEGATION_END.search(tail):
            continue
        if TR_AUTHORITY_SUBJECT.search(tail) or TR_DELEGATION_AGENT.search(sentence):
            found.append((start, end))
    return found


def duty_modals(text: str) -> list[re.Match]:
    """MODAL matches that may state a duty: those inside a delegation sentence are left out, and so
    are the v0.19 soft markers whose sentence says they are no duty (soft_marker_holds)."""
    excluded = delegation_spans(text)
    found = [m for m in MODAL.finditer(text) if not any(a <= m.start() < b for a, b in excluded)]
    return [m for m in found if binding(m) or soft_marker_holds(text, m, found)]


# "Bu madde hükümleri ... uygulanmaz", "Birinci fıkra ... uygulanmaz": a provision is exempted (EXEMPTION).
# "Üçüncü tarafa güven ilkesi, ... durumunda uygulanmaz" limits a permission: a duty not to rely.
TR_PROVISION_NAME = re.compile(r'(?<![%s])(?:[Hh]üküm|[Hh]ükm|[Mm]adde|[Ff]ıkra|[Bb]ent|[Bb]end|[Kk]anun|[Yy]önetmeli|[Tt]ebliğ|[Kk]ararname|[Bb]ölüm)'
                               % TR_LETTERS)
# "yalnızca/sadece/ancak ... halinde ... yapabilir": the permission exists only under the condition.
TR_ONLY = re.compile(r'(?<![%s])(?:yalnızca|yalnız|sadece|ancak)(?![%s])' % (TR_LETTERS, TR_LETTERS))
TR_CONDITION_WORD = re.compile(r'şartıyla|kaydıyla|koşuluyla|h[aâ]linde|h[aâ]llerinde|takdirde|durumunda')
TR_PLAIN_CONDITION = re.compile(r'h[aâ]linde|h[aâ]llerinde|takdirde|durumunda')
SOFT_PASSIVE = re.compile(r'edilir|olunur|edilmez|olunmaz')


def soft_marker_holds(text: str, match: re.Match, found) -> bool:
    """Whether a soft marker states a duty in its own sentence.

    "uygulanmaz" in a sentence that names a provision exempts it; a condition word marks a duty only
    while the sentence it opens ends in a permission with no binding marker after it, and "halinde",
    "takdirde", "durumunda" only when the clause says the permission exists solely then ("yalnızca").
    """
    word = match.group()
    start, _ = _sentence(text, match.start())
    if word == 'uygulanmaz':
        return not TR_PROVISION_NAME.search(text, start, match.start())
    if TR_CONDITION_WORD.fullmatch(word):
        stop = re.compile(r'[.;]').search(text, match.end())
        stop = stop.start() if stop else len(text)
        if any(binding(m) and match.end() <= m.start() < stop for m in found):
            return False
        if TR_PLAIN_CONDITION.fullmatch(word):
            clause = start
            for boundary in re.finditer(r'[,;:(]\s*', text[start:match.start()]):
                clause = start + boundary.end()
            return TR_ONLY.search(text, clause, match.start()) is not None
    return True


def looks_turkish(text: str) -> bool:
    """Turkish by its letters or modal markers, or by wording only Turkish drafting uses.

    "Kurul ek tedbir alabilir." has neither a Turkish letter nor a duty marker; its
    permission suffix and a delegation ending are Turkish all the same.
    """
    return (turkish(text) or bool(TR_PERMISSION.search(text)) or bool(TR_DELEGATION_END.search(text.rstrip(' .;')))
            or bool(TR_DEFINITION.search(text)) or bool(TR_SCOPE.search(text)) or bool(TR_ENFORCEMENT.search(text)))


def classify(text: str, heading: str = '', legal_type: str | None = None) -> Classification:
    """The class of one regulatory unit, decided by its wording; never by a model."""
    stripped = (text or '').strip()
    if not stripped:
        return Classification('OTHER', '', 'empty')
    if looks_turkish(stripped):
        return classify_turkish(stripped, heading or '')
    return classify_english(stripped, legal_type)


def classify_turkish(text: str, heading: str) -> Classification:
    standalone = TR_STANDALONE.search(text)
    duties = duty_modals(text)
    delegations = delegation_spans(text)
    head = heading.lower().replace('i̇', 'i')
    # The article heading names definitions, scope and penalty articles outright; a standalone
    # duty marker in the text still wins, because a heading is not the operative wording.
    for word, kind in TR_HEADINGS:
        if word in head and not standalone and kind != 'OTHER':
            return Classification(kind, heading, 'heading')
    if TR_DEFINITION.search(text) and not standalone:
        return Classification('DEFINITION', TR_DEFINITION.search(text).group(0).strip(), 'wording')
    if TR_ENFORCEMENT.search(text) and not standalone:
        return Classification('ENFORCEMENT', TR_ENFORCEMENT.search(text).group(0), 'wording')
    # "Bu Kanun ... hakkında uygulanır" says who is bound; its closing aorist is not a duty.
    if TR_SCOPE.search(text) and not standalone:
        return Classification('SCOPE', TR_SCOPE.search(text).group(0)[:80], 'wording')
    # "... tespit edilir." / "... teyit edilir.": a passive aorist on "edilir" also states facts and deems
    # ("kabul edilir" is left out already); the unit stays OTHER - which is extracted, since MODAL finds
    # the marker - rather than being named a duty by a word the class was never read from.
    passive = [m for m in duties if not binding(m) and SOFT_PASSIVE.fullmatch(m.group())]
    duties = [m for m in duties if m not in passive]
    if passive and not duties:
        return Classification('OTHER', passive[0].group(), 'passive aorist (extracted, not classed)')
    if duties:
        # The v0.18 markers decide the class where there are any; a soft marker only where there are none.
        duties = [m for m in duties if binding(m)] or duties
        kinds = {modality(m.group()) for m in duties}
        if kinds <= {'MUST_NOT'}:
            return Classification('PROHIBITION', duties[0].group(), 'modal')
        return Classification('OBLIGATION', next(m.group() for m in duties if not modality(m.group()).endswith('_NOT')), 'modal')
    if delegations:
        start, end = delegations[0]
        return Classification('DELEGATION', text[start:end].strip()[-60:], 'delegation sentence')
    if TR_EXEMPTION.search(text):
        return Classification('EXEMPTION', TR_EXEMPTION.search(text).group(0), 'wording')
    if TR_PERMISSION.search(text):
        return Classification('PERMISSION', TR_PERMISSION.search(text).group(0), 'wording')
    return Classification('OTHER', '', 'no modal wording')


def classify_english(text: str, legal_type: str | None) -> Classification:
    modals = [m.group() for m in re.finditer(ENGLISH_MODAL, text)]
    kinds = {modality(m) for m in modals}
    binding = kinds & {'MUST', 'MUST_NOT'}
    if EN_SCOPE.search(text) and not binding:
        return Classification('SCOPE', EN_SCOPE.search(text).group(0)[:80], 'wording')
    if EN_DEFINITION.search(text) and not binding:
        return Classification('DEFINITION', EN_DEFINITION.search(text).group(0)[:80], 'wording')
    if EN_ENFORCEMENT.search(text) and not binding:
        return Classification('ENFORCEMENT', EN_ENFORCEMENT.search(text).group(0), 'wording')
    if EN_DELEGATION.search(text) and not binding:
        return Classification('DELEGATION', EN_DELEGATION.search(text).group(0), 'wording')
    if binding:
        if binding == {'MUST_NOT'}:
            return Classification('PROHIBITION', next(m for m in modals if modality(m) == 'MUST_NOT'), 'modal')
        return Classification('OBLIGATION', next(m for m in modals if modality(m) == 'MUST'), 'modal')
    if kinds & {'SHOULD', 'SHOULD_NOT'}:
        return Classification('GUIDANCE' if legal_type == 'GUIDANCE' else 'RECOMMENDATION', next(m for m in modals if modality(m).startswith('SHOULD')), 'modal')
    # v0.19: "the firm will notify", "a lender needs to", "is obliged to" state what the firm does.
    soft = [m.group() for m in re.finditer(EN_SOFT, text)]
    if soft:
        negative = all(modality(m) == 'MUST_NOT' for m in soft)
        kind = 'GUIDANCE' if legal_type == 'GUIDANCE' else 'PROHIBITION' if negative else 'OBLIGATION'
        return Classification(kind, soft[0], 'modal')
    if kinds & {'MAY', 'MAY_NOT'}:
        return Classification('PERMISSION', next(m for m in modals if modality(m).startswith('MAY')), 'modal')
    if legal_type == 'GUIDANCE':
        return Classification('GUIDANCE', '', 'provision type')
    return Classification('OTHER', '', 'no modal wording')

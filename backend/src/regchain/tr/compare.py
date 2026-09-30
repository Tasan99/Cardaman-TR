"""Obligation -> policy -> control -> evidence -> gap, for one enterprise profile.

An extracted obligation that applies to a target is compared with the internal documents in force for that target:

  passage reading   each numbered statement of a policy, SOP or specification is read into the same clause frame as the
                    regulation (frames.py) and related to the duty element by element: the governed act, the polarity,
                    the products, the places and the other party, limits with their unit and direction, product
                    conditions, a time window, exceptions. The relation is SUPPORTS, PARTIAL, CONFLICTS, UNRELATED or
                    UNCLEAR, with the reason codes and the quote.
  document coverage counted from the passage readings: CONFLICT wins; COVERS_TEXT needs one passage that supports every
                    element; PARTIAL names what is missing; NO_EVIDENCE when nothing relates.
  mapping           mapping.mapping_status joins the coverage with a linked control and in-period evidence:
                    COVERED / PARTIALLY_COVERED / NOT_COVERED / CONTRADICTED / UNKNOWN, and the gap says what to do and who.

Shared words are never enough: a statement supports a duty only when its own frame states the same act with the same
polarity for a compatible product, and every limit is the same or stricter. This reading is deterministic and needs no
model. It is strict about elements and blind to paraphrase ("Satış belgesi ... bir yere asılır" for "Satış belgeleri ...
uygun yerlere asılır"): measured on BEVERAGE_TR_DEV_V2 it invented no conflict and missed nine statements that say the
duty in other words. Where an engine run judged the same passages, combine_coverage joins the two readings: the rule
comparer decides what it can check (a weaker limit, a missing place, a time window); the model's "covers" is taken
where the rules found nothing (other wording); a conflict or a partial cover only the model sees changes no coverage
and marks the row for review. The registers under data/pilot_policies are synthetic.
"""
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..pilot.schema import Strict
from .clauses import blank_notes, split_clauses
from .corpus import CorpusStore
from .extraction import MARKETING, SALES, ExtractedObligation, governing, predicate_segments
from .frames import LETTERS, Frame, fold, frame_of
from .mapping import Control, DocumentCoverage, DocumentType, EvidenceItem, ObligationMapping, map_obligation
from .packs import Registry
from .profile import EnterpriseProfile, PolicyScope, policy_covers, product_alcohol

DATA = Path(__file__).resolve().parent / 'data'
REGISTERS = DATA / 'pilot_policies'
OWNERSHIP = DATA / 'ownership.json'
COMPARE_RULES_VERSION = 'tr-compare-rules-v1'
Relation = Literal['SUPPORTS', 'PARTIAL', 'CONFLICTS', 'UNRELATED', 'UNCLEAR']
STATEMENT = re.compile(r'^\s*(\d{1,3})\.\s+(?=\S)')
SALE_FAMILY = frozenset(SALES)
CLOCK = re.compile(r'(\d{1,2})[:.](\d{2})')
STOPWORDS = frozenset('''bir bu ve veya ile için olan olarak olup gibi kadar üzere göre her hiçbir tüm diğer ilgili ise ancak ayrıca
yapılır yapılamaz yapılmaz edilir edilemez edilmez olamaz olmalıdır zorunludur yer alır alan bulunur şekilde hariç dâhil dahil
üzerine üzerinde içinde tarafından yalnızca sadece önce sonra surette suretle olursa olsun amaçla hiçbir türlü nevi edilen yapılan'''.split())


class DocumentRecord(Strict):
    document_id: str = Field(min_length=1)
    type: DocumentType
    title: str
    owner_department: str
    version: str = Field(min_length=1)
    file: str
    scope: PolicyScope
    product_ids: list[str] = []
    content_hash: str = ''


class ControlRecord(Control):
    topics: list[str] = []


class EvidenceRecord(Strict):
    evidence_id: str = Field(min_length=1)
    control_id: str
    evidence_type: str
    period: str
    in_period: bool = True


class Passage(Strict):
    passage_id: str
    document_id: str
    number: int
    text: str                       # the statement, an exact span of the file
    start: int
    end: int


class Register(Strict):
    format: Literal['cardaman-tr-policy-register/1']
    profile_id: str
    synthetic: bool
    disclaimer: str = ''
    period: str
    documents: list[DocumentRecord]
    controls: list[ControlRecord] = []
    evidence: list[EvidenceRecord] = []
    passages: list[Passage] = []

    def document(self, document_id: str) -> DocumentRecord:
        return next(d for d in self.documents if d.document_id == document_id)


class PassageReading(Strict):
    passage_id: str
    document_id: str
    number: int
    relation: Relation
    reasons: list[str]
    quote: str
    detail: dict = {}


class ProductFinding(Strict):
    """A stated product value against a limit the clause sets: a fact about the product, not about a policy."""
    product_id: str
    attribute: str
    stated: float
    limit: float
    comparator: str
    unit: str
    result: Literal['WITHIN_LIMIT', 'EXCEEDS_LIMIT']
    quote: str


class GapRow(Strict):
    obligation_id: str
    regulation_id: str
    provision_ref: str
    topic: str
    modality: str
    quote: str
    level: str
    target_id: str
    entity_id: str | None = None
    applicability: str
    applies_to_products: list[str] = []
    documents_in_force: list[str] = []
    readings: list[PassageReading] = []
    document_coverage: DocumentCoverage
    coverage_reasons: list[str] = []
    # Who stands behind the coverage: the rule comparer alone, or the two readings joined (combine_coverage).
    coverage_basis: str = 'RULE_ONLY'
    rule_coverage: DocumentCoverage | None = None
    model_coverage: DocumentCoverage | None = None
    mapping: ObligationMapping
    product_findings: list[ProductFinding] = []
    gap: str | None = None
    actions: list[dict] = []
    review_required: bool = False


class GapReport(Strict):
    profile_id: str
    register_synthetic: bool
    rules_version: str = COMPARE_RULES_VERSION
    rows: list[GapRow]
    summary: dict


# -- register ----------------------------------------------------------------------------------------
def load_register(profile_id: str, root: Path = REGISTERS) -> Register:
    """The register of one profile with every document read into numbered statements and hashed."""
    directory = Path(root) / profile_id
    data = json.loads((directory / 'register.json').read_text(encoding='utf-8'))
    register = Register.model_validate(data)
    if register.profile_id != profile_id:
        raise ValueError(f'register {directory.name} belongs to {register.profile_id}')
    passages = []
    for document in register.documents:
        path = (directory / document.file).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError(f'{document.document_id}: the file must stay inside the register directory')
        raw = path.read_bytes()
        document.content_hash = sha256(raw).hexdigest()
        text = raw.decode('utf-8').replace('\r\n', '\n')
        for block in re.finditer(r'[^\n]+(?:\n(?!\n)[^\n]+)*', text):
            numbered = STATEMENT.match(block.group())
            if not numbered:
                continue                                   # title, document header, disclaimer
            start = block.start() + numbered.end()
            passages.append(Passage(passage_id=f'{document.document_id}#{numbered.group(1)}', document_id=document.document_id,
                                    number=int(numbered.group(1)), text=text[start:block.end()], start=start, end=block.end()))
    known = {d.document_id for d in register.documents}
    for control in register.controls:
        missing = [d for d in control.documents if d not in known]
        if missing:
            raise ValueError(f'{control.control_id} names unknown documents {missing}')
    controls = {c.control_id for c in register.controls}
    for item in register.evidence:
        if item.control_id not in controls:
            raise ValueError(f'{item.evidence_id} belongs to unknown control {item.control_id}')
    register.passages = passages
    return register


def ownership() -> dict:
    return json.loads(OWNERSHIP.read_text(encoding='utf-8'))


def passage_frames(passage: Passage) -> list[Frame]:
    """The statement read as the regulation is read: one frame per sentence, and per part of a sentence a semicolon
    separates ("... satılamaz ve sunulamaz; yaş konusunda tereddüt halinde kimlik belgesi istenir" is two statements,
    and the polarity of the second is not the polarity of the first)."""
    frames, offset = [], 0
    for part in passage.text.split(';'):
        start = offset + (len(part) - len(part.lstrip()))
        text = part.strip()
        offset += len(part) + 1
        if not text:
            continue
        section = {'text': text, 'printed_label': passage.passage_id, 'heading_path': ['', '', ''], 'quality_flags': [],
                   'lines': [{'kind': 'line', 'start': 0, 'end': len(text)}]}
        frames += [frame_of(clause, passage.document_id) for clause in split_clauses(section)]
    return frames


# -- element reading ---------------------------------------------------------------------------------
# "her ne surette olursa olsun" bans the act outright. "hiçbir", "her türlü", "her nevi" quantify the noun next to them:
# they make the ban absolute only where that noun is the act itself ("hiçbir etkinliğe", "marka ... hiçbir işareti"), not
# where it is the medium of a qualified act ("Her türlü iletişim aracında ... örtülü reklam yapılması yasaktır").
ABSOLUTE = re.compile(r'her ne (?:surette|suretle|amaçla|şekilde) olursa olsun')
QUANTIFIER = re.compile(r'(?<![%s])(?:hiçbir|her türlü|her nevi)(?![%s])' % (LETTERS, LETTERS))
QUANTIFIER_REACH = 30


def absolute_ban(frame: Frame) -> bool:
    folded = fold(frame.text)
    if ABSOLUTE.search(folded):
        return True
    acts = [(m.start - frame.start, m.end - frame.start) for m in frame.activities if m.where == 'CLAUSE']
    return any(a - QUANTIFIER_REACH <= match.start() <= b + QUANTIFIER_REACH for match in QUANTIFIER.finditer(folded) for a, b in acts)


def _acts(frame: Frame, subject: str = 'first') -> set[str]:
    """The acts a frame's predicate governs. An item of a list whose lead-in carries the predicate ("... yerlerde alkollü
    içki satışı veya sunumu yapılamaz: a) ...") takes them from the lead-in; an item under a lead-in that only names the
    addressee ("Piyasaya sürenler; a) ... kayıt olmakla, ... yükümlüdürler.") is its own content."""
    if frame.marker.endswith('(chapeau)'):
        return {m.id for m in frame.activities if m.where == 'CHAPEAU'}
    return {m.id for m in governing(frame, subject)}


def _same_act(duty: set[str], policy: set[str]) -> set[str]:
    """Duty acts the policy also names; a named kind of sale answers the generic "satış" and the reverse, two
    different named kinds (retail, wholesale) do not answer each other."""
    shared = duty & policy
    if duty & SALE_FAMILY and policy & SALE_FAMILY:
        specific_duty, specific_policy = duty & (SALE_FAMILY - {'SALE'}), policy & (SALE_FAMILY - {'SALE'})
        if not specific_duty or not specific_policy or specific_duty & specific_policy:
            shared |= duty & SALE_FAMILY
    return shared


def stems(text: str) -> set[str]:
    """Crude Turkish stems: the first five letters of each content word of four letters or more."""
    words = re.findall(r'[%s]{4,}' % LETTERS, fold(text))
    return {w[:5] for w in words if w not in STOPWORDS}


def containment(duty_text: str, policy_text: str) -> float:
    wanted = stems(duty_text)
    return len(wanted & stems(policy_text)) / len(wanted) if wanted else 0.0


def distinctive(frame: Frame, start: int = 0, end: int | None = None) -> set[str]:
    """The stems of a clause (or of one span of it) without its product words and editorial notes: two rules about
    "alkollü içkiler ... satılamaz" share those words and nothing else."""
    end = len(frame.text) if end is None else end
    chars = list(frame.text)
    for mention in frame.products:
        a, b = mention.start - frame.start, mention.end - frame.start
        if mention.where == 'CLAUSE' and 0 <= a < b <= len(chars):
            chars[a:b] = ' ' * (b - a)
    return stems(blank_notes(''.join(chars))[start:end])


def _sides(frame: Frame, vocabulary) -> set[str]:
    """ALCOHOLIC / NON_ALCOHOLIC sides a frame's product mentions name."""
    out = set()
    for mention in frame.products:
        if mention.id == 'ALCOHOLIC_BEVERAGE':
            out.add('ALCOHOLIC')
        elif mention.id == 'NON_ALCOHOLIC_BEVERAGE':
            out.add('NON_ALCOHOLIC')
        out |= {vocabulary.alcohol_of(c) for c in mention.classes if len(mention.classes) == 1}
    return out


def _products_compatible(duty: ExtractedObligation, policy: Frame, vocabulary) -> bool:
    """Whether the statement speaks about products the duty covers. A statement that names no product is left to the
    scope of its document; a duty that names both sides (a brand may not cross from one to the other) needs both."""
    if not policy.products:
        return True
    policy_classes = {c for m in policy.products for c in m.classes}
    policy_sides = _sides(policy, vocabulary)
    duty_sides = _sides(duty.frame, vocabulary)
    if len(duty_sides) == 2:
        return len(policy_sides) == 2
    scope = duty.scope
    if scope.product_classes:
        return bool(policy_classes & set(scope.product_classes)) or (not policy_classes and bool(
            policy_sides & {vocabulary.alcohol_of(c) for c in scope.product_classes}))
    if scope.alcohol_scope != 'ANY':
        # A statement that names a product of the other side ("alkolsüz bira" against a duty on alcoholic drinks, or a
        # statement about carrying an alcohol brand onto an alcohol-free product) is about that product.
        other = 'NON_ALCOHOLIC' if scope.alcohol_scope == 'ALCOHOLIC' else 'ALCOHOLIC'
        if other in policy_sides:
            return False
        return scope.alcohol_scope in policy_sides or any(vocabulary.alcohol_of(c) == scope.alcohol_scope for c in policy_classes)
    return True


def _minutes(match) -> int:
    return int(match.group(1)) * 60 + int(match.group(2))


def _inside_window(moment: int, start: int, end: int) -> bool:
    return (start < moment <= 24 * 60 or moment <= end) if start > end else start < moment <= end


def _quantity_relation(duty_q, policy_q) -> str:
    """SAME, STRICTER or WEAKER: how the policy's limit stands to the duty's, for limits in the same direction."""
    if duty_q.comparator in ('le', 'lt'):
        return 'WEAKER' if policy_q.value > duty_q.value else 'STRICTER' if policy_q.value < duty_q.value else 'SAME'
    if duty_q.comparator in ('ge', 'gt'):
        return 'WEAKER' if policy_q.value < duty_q.value else 'STRICTER' if policy_q.value > duty_q.value else 'SAME'
    return 'SAME' if policy_q.value == duty_q.value else 'WEAKER'


ORDER = {'UNRELATED': 0, 'UNCLEAR': 1, 'PARTIAL': 2, 'SUPPORTS': 3, 'CONFLICTS': 4}


def relate(duty: ExtractedObligation, passage: Passage, vocabulary) -> PassageReading:
    """One policy statement against one duty: the strongest relation any of its sentences has."""
    frames = passage_frames(passage)
    best = None
    for policy in frames:
        reading = _relate_frame(duty, policy, vocabulary)
        if best is None or ORDER[reading[0]] > ORDER[best[0]]:
            best = reading
    relation, reasons, detail = best if best else ('UNRELATED', ['NO_STATEMENT'], {})
    # An exception sentence in the same statement ("... bu kuralın dışındadır") narrows a supporting sentence.
    if relation == 'SUPPORTS' and any(f.kind == 'EXCEPTION' for f in frames) and not duty.frame.exceptions:
        relation, reasons = 'PARTIAL', reasons + ['POLICY_ADDS_EXCEPTION']
    return PassageReading(passage_id=passage.passage_id, document_id=passage.document_id, number=passage.number,
                          relation=relation, reasons=reasons, quote=passage.text, detail=detail)


def _relate_frame(duty: ExtractedObligation, policy: Frame, vocabulary):
    frame = duty.frame
    duty_acts = _acts(frame)
    policy_acts = _acts(policy, 'head')
    policy_named = {m.id for m in policy.activities}
    shared = _same_act(duty_acts, policy_acts)
    named = _same_act(duty_acts, policy_named)
    wanted, offered = distinctive(frame), distinctive(policy)
    overlap = len(wanted & offered) / len(wanted) if wanted else 0.0
    attributes = {q.attribute for q in frame.quantities if q.attribute}
    same_attribute = attributes & {q.attribute for q in policy.quantities if q.attribute}
    compatible = _products_compatible(duty, policy, vocabulary)
    detail = {'duty_acts': sorted(duty_acts), 'policy_acts': sorted(policy_acts), 'containment': round(overlap, 2)}
    if not compatible:
        return 'UNRELATED', ['OTHER_PRODUCT'], detail
    # -- limits in the same direction decide first: "320 mg/L’den fazla olamaz" against "150 mg/L’den fazla olamaz"
    missing = []
    direction = lambda q: 'MAX' if q.comparator in ('le', 'lt') else 'MIN' if q.comparator in ('ge', 'gt') else 'EXACT'
    limits = [q for q in frame.quantities if q.role == 'LIMIT']
    for quantity in (q for q in limits if q.attribute):
        # Only a limit in the same direction answers a limit: "1,0 mg/L’den fazla olan ürünlerde" selects products, it
        # does not cap them.
        stated = [p for p in policy.quantities if p.attribute == quantity.attribute and p.unit == quantity.unit
                  and p.role == 'LIMIT' and direction(p) == direction(quantity)]
        if not stated:
            missing.append(f'LIMIT_MISSING:{quantity.attribute}')
            continue
        relation = _quantity_relation(quantity, stated[0])
        detail.setdefault('quantities', []).append({'attribute': quantity.attribute, 'duty': quantity.value, 'policy': stated[0].value,
                                                    'unit': quantity.unit, 'relation': relation})
        if relation == 'WEAKER':
            return 'CONFLICTS', ['LIMIT_WEAKER'], detail
    negative_duty = frame.modality == 'MUST_NOT'
    positive_policy = policy.modality in ('MUST', 'MAY')
    duty_places, policy_places = {m.id for m in frame.places}, {m.id for m in policy.places}
    duty_minor = any(m.id == 'MINOR' for m in frame.counterparties)
    policy_minor = any(m.id == 'MINOR' for m in policy.counterparties)
    windows = [c for c in frame.conditions if c.kind == 'TIME']
    if policy.modality is None:
        return ('UNCLEAR', ['POLICY_STATEMENT_NOT_NORMATIVE'], detail) if shared and overlap >= 0.5 else ('UNRELATED', ['OTHER_SUBJECT'], detail)
    # -- a prohibition against a statement that does or allows something -------------------------------------
    if negative_duty and positive_policy:
        if windows and shared:
            clocks = [_minutes(m) for m in CLOCK.finditer(policy.text)]
            bounds = [_minutes(m) for m in CLOCK.finditer(windows[0].quote)]
            if clocks and len(bounds) == 2 and any(_inside_window(c, bounds[0], bounds[1]) for c in clocks):
                detail['time'] = {'forbidden': windows[0].quote, 'policy': policy.text}
                return 'CONFLICTS', ['TIME_WINDOW_VIOLATED'], detail
            return 'UNRELATED', ['OTHER_CIRCUMSTANCE'], detail
        if duty_minor or duty_places:
            if shared and ((duty_minor and policy_minor) or (duty_places & policy_places)):
                return 'CONFLICTS', ['POLICY_PERMITS_PROHIBITED_ACT'], detail
            return 'UNRELATED', ['OTHER_CIRCUMSTANCE'], detail
        # A statement contradicts a prohibition by permitting the act ("... kullanılabilir") or by making it a duty. A duty
        # about something else that only names the act ("reklam ve tanıtım çalışmaları ... Hukuk Müşavirliğinin onayına
        # sunulur", "uyarı mesajları konulur") does neither: for a MUST statement the act has to be what its own predicate
        # governs. (Found on the whole corpus, outside the labelled cases: 17 rows of the advertising by-law and of the
        # consumer law were called contradicted by the statement that sends advertisements to legal review.)
        permits = policy.modality == 'MAY'
        does = shared if permits else _same_act(duty_acts, _acts(policy, 'none'))
        if absolute_ban(frame) and (does or (permits and named and policy.products)):
            if any(e.effect == 'PERMITS' and not e.predicate for e in frame.exceptions) and policy.conditions:
                return 'UNCLEAR', ['POLICY_MAY_FALL_UNDER_EXCEPTION'], detail
            return 'CONFLICTS', ['POLICY_PERMITS_PROHIBITED_ACT'], detail
        # A prohibition with its own qualifiers is contradicted only by a statement that repeats them: the same act and
        # most of the distinctive wording, at least three words of it.
        if does and overlap >= 0.6 and len(wanted & offered) >= 3:
            return 'CONFLICTS', ['POLICY_PERMITS_PROHIBITED_ACT'], detail
        return 'UNRELATED', ['OTHER_CIRCUMSTANCE'], detail
    if not negative_duty and policy.modality == 'MUST_NOT':
        return ('CONFLICTS', ['POLICY_NEGATES_DUTY'], detail) if overlap >= 0.6 else ('UNRELATED', ['OTHER_SUBJECT'], detail)
    # -- aligned polarity: how much of the duty does the statement state? -------------------------------------
    segments = [distinctive(frame, a, b) for a, b in predicate_segments(frame)]
    segments = [seg for seg in segments if seg] or [wanted]
    stated = [len(seg & offered) / len(seg) if seg else 0.0 for seg in segments]
    detail['segments'] = [round(v, 2) for v in stated]
    # Relevant: the same measured property; or the same act (or, for a rule that names none, the wording) with enough
    # of the distinctive wording: half of the clause, or most of one predicate of at least three words.
    common = len(wanted & offered)
    if len(wanted) <= 2:
        carried = common == len(wanted) and common > 0             # "Gıdanın net miktarı.": both words, not one of them
    elif len(wanted) <= 4:
        carried = overlap >= 0.75
    else:
        carried = (overlap >= 0.5 and common >= 2) or any(v >= 0.6 and len(seg) >= 3 for v, seg in zip(stated, segments))
    unnamed = [q for q in limits if not q.attribute]
    if unnamed and not any(p.unit == q.unit and p.role == 'LIMIT' for q in unnamed for p in policy.quantities):
        carried = False                                            # a limit without a stated limit of the same unit
    if same_attribute and missing and not detail.get('quantities') and not carried:
        return 'UNRELATED', ['OTHER_SUBJECT'], detail              # the same substance, another rule about it
    relevant = bool(detail.get('quantities')) or ((bool(shared) or not duty_acts) and carried) or (overlap >= 0.75 and common >= 3)
    if not relevant:
        return 'UNRELATED', ['OTHER_SUBJECT'], detail
    if duty_minor and not policy_minor:
        return 'UNRELATED', ['OTHER_CIRCUMSTANCE'], detail
    reasons = ['POLICY_OPTIONAL'] if (not negative_duty and policy.modality == 'MAY') else []
    if any(v < 0.5 for v in stated) and len(stated) > 1 and max(stated) >= 0.5:
        missing.append('PREDICATE_NOT_STATED')
    elif overlap < 0.6 and not detail.get('quantities'):
        # Half of the distinctive wording is a related statement; most of it is the duty written down.
        missing.append('WORDING_PARTLY_MATCHED')
    if duty_places - policy_places:
        missing += [f'PLACE_MISSING:{p}' for p in sorted(duty_places - policy_places)] if policy_places else ['PLACES_MISSING']
    if windows and negative_duty:
        clocks = sorted(_minutes(m) for m in CLOCK.finditer(policy.text))
        bounds = sorted(_minutes(m) for m in CLOCK.finditer(windows[0].quote))
        if not clocks:
            missing.append('TIME_WINDOW_MISSING')
        elif clocks != bounds:
            missing.append('TIME_WINDOW_DIFFERENT')
    for condition in (c for c in frame.conditions if c.predicate):
        same = [c for c in policy.conditions if c.predicate and c.predicate['fact'] == condition.predicate['fact']]
        if same and ((same[0].predicate['op'] in ('gt', 'ge') and same[0].predicate['value'] > condition.predicate['value'])
                     or (same[0].predicate['op'] in ('lt', 'le') and same[0].predicate['value'] < condition.predicate['value'])):
            missing.append('CONDITION_NARROWER')
    inline = [e for e in policy.exceptions if e.source_ref == policy.ref]
    if inline and not frame.exceptions:
        missing.append('POLICY_ADDS_EXCEPTION')
    detail['missing'] = missing
    if missing or reasons:
        return 'PARTIAL', reasons + missing, detail
    return 'SUPPORTS', ['ACT_AND_POLARITY_MATCH'] + (['LIMIT_SAME_OR_STRICTER'] if detail.get('quantities') else []), detail


def coverage_of(readings: list[PassageReading]) -> tuple[str, list[str]]:
    """(document coverage, reasons) counted from the passage readings."""
    relations = [r.relation for r in readings]
    if 'CONFLICTS' in relations:
        return 'CONFLICT', sorted({c for r in readings if r.relation == 'CONFLICTS' for c in r.reasons})
    if 'SUPPORTS' in relations:
        return 'COVERS_TEXT', ['SUPPORTING_STATEMENT']
    if 'PARTIAL' in relations:
        return 'PARTIAL', sorted({c for r in readings if r.relation == 'PARTIAL' for c in r.reasons})
    if 'UNCLEAR' in relations:
        return 'UNKNOWN', sorted({c for r in readings if r.relation == 'UNCLEAR' for c in r.reasons})
    return 'NO_EVIDENCE', ['NO_RELATED_STATEMENT']


# What the rule comparer checks element by element; a gap it names this way is not overruled by a model's "covers".
ELEMENT_GAPS = ('PLACE_MISSING', 'PLACES_MISSING', 'LIMIT_MISSING', 'PREDICATE_NOT_STATED', 'TIME_WINDOW_', 'CONDITION_NARROWER',
                'POLICY_ADDS_EXCEPTION', 'POLICY_OPTIONAL')


# Reasons under which the rule comparer has decided nothing: it found no statement, or only loose wording.
LOOSE_REASONS = frozenset({'NO_RELATED_STATEMENT', 'WORDING_PARTLY_MATCHED'})


def combine_coverage(rule: str, reasons: list[str], model: str | None, adjudicated: bool = False) -> tuple[str, str, bool]:
    """(coverage, basis, review) from the rule comparer's coverage and a model's for the same duty and documents.

    `adjudicated` says the model reading is the selective adjudicator's (adjudicate.py: the strong model on the few
    statements the candidate search named, every quote checked) and not an engine run over every passage. Where the
    rule comparer decided nothing, an adjudicated PARTIAL is the coverage (the gap stays open either way), and an
    adjudicated COVERS_TEXT is a proposal: the coverage stays the rules' and the row waits for a person
    (MODEL_PROPOSES_COVERED). On the DEV cases the strong model's "states the duty" on a statement the rules could
    not relate was right four times of six, and a wrong one closes a gap nobody looked at.
    An engine's reading changes no coverage at all: its fast classifier is the small extraction model, which is no
    judge here, and on twelve labelled cases its coverage was right once (runs of 30 September 2026). What it sees is
    kept beside the row for review.

    CONFLICT      the rule comparer's conflicts are element checks (a weaker limit, a violated time window, a permitted
                  prohibited act) and stand alone. A conflict only the model sees keeps the rule comparer's coverage and
                  is marked for review with the passage the model quoted: on the twelve labelled cases of the two unit
                  runs of 30 September 2026 the judge claimed four conflicts, one real (the caffeine limit, which the
                  rules also found) and three not (a tasting permission against a display rule, "etil alkol ilave
                  edilmez" against a label warning, campus sampling against a ban in school canteens).
    PARTIAL       a missing element the rule comparer names stands against a model's COVERS_TEXT.
    paraphrase    where the rule comparer found nothing or only loose wording, the adjudicator's COVERS_TEXT is kept
                  beside the row as a proposal with the statement it quoted; nothing a model says alone becomes
                  COVERED or CONTRADICTED. The engine's is a note.
    model PARTIAL is not evidence by itself. In the unit run on Kanun 4250 md. 6 (20260930-beverage-live-3, 14 duties) the
                  judge called passages PARTIAL that share only the product ("Alkollü içkiler otomatik satış makineleri
                  ile satılamaz" under eight other duties) and ended PARTIAL on duties no statement touches. So a PARTIAL
                  only the model sees changes no coverage: it marks the row for review.
    """
    if model is None:
        return rule, 'RULE_ONLY', rule in ('CONFLICT', 'UNKNOWN')
    if rule == 'CONFLICT':
        return 'CONFLICT', 'BOTH_READINGS' if model == 'CONFLICT' else 'RULE_ELEMENT_CHECK', True
    if model == 'CONFLICT':
        return rule, 'MODEL_CONFLICT_UNCONFIRMED', True
    element_gap = any(r.startswith(ELEMENT_GAPS) for r in reasons)
    if rule == 'PARTIAL' and element_gap:
        return 'PARTIAL', 'BOTH_READINGS' if model == 'PARTIAL' else 'RULE_ELEMENT_CHECK', False
    if rule == 'COVERS_TEXT':
        if model == 'PARTIAL':
            return 'COVERS_TEXT', 'MODEL_SEES_GAP', True
        return 'COVERS_TEXT', 'BOTH_READINGS' if model == 'COVERS_TEXT' else 'RULE_WORDING_MATCH', False
    if model == 'COVERS_TEXT':
        return rule, 'MODEL_PROPOSES_COVERED' if adjudicated else 'MODEL_PARAPHRASE_UNCONFIRMED', True
    if model == 'PARTIAL':
        if adjudicated and rule == 'NO_EVIDENCE':
            return 'PARTIAL', 'SEMANTIC_ADJUDICATED', False
        return rule, 'BOTH_READINGS' if rule == 'PARTIAL' else 'MODEL_PARTIAL_UNCONFIRMED', rule != 'PARTIAL'
    if rule == 'UNKNOWN' or model == 'UNKNOWN':
        return rule, 'RULE_ONLY', True
    return rule, 'BOTH_READINGS' if rule == model else 'RULE_ONLY', False


# -- profile-level comparison ------------------------------------------------------------------------
def documents_in_force(register: Register, profile: EnterpriseProfile, decision) -> list[DocumentRecord]:
    """The documents that are the target's to compare: by owner (group, entity, facility) and, for a product target,
    by the products a specification names."""
    out = []
    for document in register.documents:
        if decision.level == 'PRODUCT':
            holders = [e.entity_id for e in profile.legal_entities if decision.target_id in e.product_ids]
            reach = any(policy_covers(document.scope, profile, 'LEGAL_ENTITY', h)['covers'] == 'YES' for h in holders)
            if reach and (not document.product_ids or decision.target_id in document.product_ids):
                out.append(document)
            continue
        level, target = ('LEGAL_ENTITY', decision.entity_id) if decision.level == 'ACTIVITY' else (decision.level, decision.target_id)
        if policy_covers(document.scope, profile, level, target)['covers'] == 'YES':
            products = set(decision.applies_to_products)
            if not document.product_ids or not products or products & set(document.product_ids):
                out.append(document)
    return out


def product_findings(obligation: ExtractedObligation, profile: EnterpriseProfile, product_ids) -> list[ProductFinding]:
    out = []
    # Only a limit with a direction can be exceeded: a bare number next to a property ("%100'lük alkol cinsinden") is a
    # unit of account, not a cap.
    for quantity in (q for q in obligation.frame.quantities if q.role == 'LIMIT' and q.attribute and q.comparator != 'eq'):
        for product_id in product_ids:
            stated = profile.product(product_id).attributes.get(quantity.attribute)
            if stated is None or stated.status != 'STATED' or isinstance(stated.value, bool):
                continue
            value = float(stated.value)
            within = value <= quantity.value if quantity.comparator in ('le', 'lt') else value >= quantity.value
            out.append(ProductFinding(product_id=product_id, attribute=quantity.attribute, stated=value, limit=quantity.value,
                                      comparator=quantity.comparator, unit=quantity.unit,
                                      result='WITHIN_LIMIT' if within else 'EXCEEDS_LIMIT', quote=quantity.text))
    return out


def _actions(row: GapRow, register: Register, owners: list[str]) -> tuple[str | None, list[dict]]:
    status, reasons = row.mapping.status, row.mapping.reasons
    actions = []
    if status == 'CONTRADICTED':
        for reading in (r for r in row.readings if r.relation == 'CONFLICTS'):
            document = register.document(reading.document_id)
            actions.append({'action': 'UPDATE_DOCUMENT', 'target': reading.passage_id, 'department': document.owner_department,
                            'why': reading.reasons})
        gap = 'An internal statement contradicts the clause.'
    elif status == 'NOT_COVERED':
        actions.append({'action': 'ADD_POLICY_STATEMENT', 'target': None, 'department': owners[0] if owners else 'COMPLIANCE',
                        'why': ['NO_RELATED_STATEMENT']})
        gap = 'No internal document in force for the target states this duty.'
    elif status == 'PARTIALLY_COVERED':
        if 'PARTIAL_POLICY' in reasons:
            for reading in (r for r in row.readings if r.relation == 'PARTIAL'):
                actions.append({'action': 'COMPLETE_STATEMENT', 'target': reading.passage_id,
                                'department': register.document(reading.document_id).owner_department, 'why': reading.reasons})
            gap = 'An internal statement carries part of the duty.'
        elif 'CONTROL_MISSING' in reasons:
            actions.append({'action': 'ADD_CONTROL', 'target': None, 'department': owners[0] if owners else 'COMPLIANCE',
                            'why': ['CONTROL_MISSING']})
            gap = 'The duty is written down but no control is linked to it.'
        else:
            for control in row.mapping.control_ids:
                actions.append({'action': 'COLLECT_EVIDENCE', 'target': control,
                                'department': next(c.owner_department for c in register.controls if c.control_id == control),
                                'why': ['EVIDENCE_MISSING']})
            gap = 'A control exists but has no evidence in the period.'
    elif status == 'UNKNOWN':
        actions.append({'action': 'REVIEW', 'target': None, 'department': owners[0] if owners else 'COMPLIANCE', 'why': reasons})
        gap = 'Coverage could not be decided by rule.'
    else:
        gap = None
    for finding in row.product_findings:
        if finding.result == 'EXCEEDS_LIMIT':
            actions.append({'action': 'PRODUCT_NONCONFORMITY', 'target': finding.product_id, 'department': 'RND',
                            'why': [f'{finding.attribute} {finding.stated:g} {finding.unit} against {finding.limit:g}']})
            gap = (gap + ' ' if gap else '') + 'A stated product value is outside the limit of the clause.'
    return gap, actions


def _model_readings(second: dict, register: Register, in_force: set[str]) -> list[PassageReading]:
    """The passages the model judged favourable or conflicting, as readings of the register's own statements."""
    numbered = re.compile(r'^\s*\d{1,3}\.\s+')
    out = []
    for relation, quotes in (second.get('quotes') or {}).items():
        for quote in quotes:
            wanted = numbered.sub('', quote).strip()
            passage = next((p for p in register.passages if p.document_id in in_force and wanted
                            and (wanted in p.text or numbered.sub('', p.text).strip() in wanted)), None)
            if passage is not None:
                out.append(PassageReading(passage_id=passage.passage_id, document_id=passage.document_id, number=passage.number,
                                          relation=relation, reasons=['MODEL_READING'], quote=passage.text, detail={'reader': 'model'}))
    return out


def compare_obligation(obligation: ExtractedObligation, decision, profile: EnterpriseProfile, register: Register,
                       registry: Registry, second: dict | None = None) -> GapRow:
    """The gap row of one obligation on one target it applies to. `second` is the engine's reading of the same duty
    ({'coverage': word, 'quotes': {'SUPPORTS': [...], 'PARTIAL': [...], 'CONFLICTS': [...]}}, from ai.second_readings)."""
    documents = documents_in_force(register, profile, decision)
    in_force = {d.document_id for d in documents}
    readings = [relate(obligation, passage, registry.vocabulary) for passage in register.passages if passage.document_id in in_force]
    related = [r for r in readings if r.relation != 'UNRELATED']
    coverage, coverage_reasons = coverage_of(related)
    rule_coverage, basis, contested = coverage, 'RULE_ONLY', False
    if second is not None:
        coverage, basis, contested = combine_coverage(rule_coverage, coverage_reasons, second.get('coverage'), bool(second.get('adjudicated')))
        if basis in ('MODEL_PARAPHRASE_UNCONFIRMED', 'MODEL_PROPOSES_COVERED', 'SEMANTIC_ADJUDICATED', 'MODEL_CONFLICT_UNCONFIRMED'):
            known = {r.passage_id for r in related}
            related += [r for r in _model_readings(second, register, in_force) if r.passage_id not in known]
            kept = set(coverage_reasons) - ({'NO_RELATED_STATEMENT'} if coverage != rule_coverage or basis == 'MODEL_CONFLICT_UNCONFIRMED' else set())
            coverage_reasons = sorted(kept) + [basis]
    decisive = {'CONFLICT': 'CONFLICTS', 'COVERS_TEXT': 'SUPPORTS', 'PARTIAL': 'PARTIAL', 'UNKNOWN': 'UNCLEAR'}.get(coverage)
    covering = sorted({r.document_id for r in related if r.relation == decisive}) if decisive else []
    controls = [c for c in register.controls if set(c.documents) & set(covering) and (not c.topics or obligation.topic in c.topics)]
    evidence = [e for e in register.evidence if e.control_id in {c.control_id for c in controls}]
    owners = list(dict.fromkeys(ownership()['topics'].get(obligation.topic, ownership()['topics']['GENERAL'])))
    mapping = map_obligation(obligation.obligation_id, coverage,
                             [Control(control_id=c.control_id, description=c.description, owner_department=c.owner_department,
                                      documents=c.documents, evidence_types=c.evidence_types) for c in controls],
                             [EvidenceItem(evidence_id=e.evidence_id, control_id=e.control_id, evidence_type=e.evidence_type,
                                           period=e.period, content_hash=sha256(e.evidence_id.encode()).hexdigest(),
                                           in_period=e.in_period) for e in evidence],
                             departments=owners, documents=covering)
    products = decision.applies_to_products or ([decision.target_id] if decision.level == 'PRODUCT' else [])
    row = GapRow(obligation_id=obligation.obligation_id, regulation_id=obligation.regulation_id, provision_ref=obligation.provision_ref,
                 topic=obligation.topic, modality=obligation.modality, quote=obligation.text, level=decision.level,
                 target_id=decision.target_id, entity_id=decision.entity_id, applicability=decision.status,
                 applies_to_products=list(decision.applies_to_products), documents_in_force=sorted(in_force), readings=related,
                 document_coverage=coverage, coverage_reasons=coverage_reasons, coverage_basis=basis, rule_coverage=rule_coverage,
                 model_coverage=second.get('coverage') if second else None, mapping=mapping,
                 product_findings=product_findings(obligation, profile, products),
                 review_required=decision.review_required or contested or coverage in ('UNKNOWN', 'CONFLICT'))
    row.gap, row.actions = _actions(row, register, owners)
    return row


def compare_profile(profile: EnterpriseProfile, obligations: list[ExtractedObligation], register: Register, registry: Registry,
                    store: CorpusStore | None = None, second: dict | None = None) -> GapReport:
    """Every obligation on every target it applies to (APPLIES or PARTIAL), with coverage, mapping and gap. `second` holds
    the engine's readings per legal entity ({entity id: {provision ref: reading}}): a row is joined with the reading made
    for its own entity, because the documents in force differ from entity to entity."""
    from .extraction import route
    store = store or CorpusStore()
    rows = []
    for obligation in obligations:
        decisions, _ = route(obligation, profile, registry, store)
        for decision in decisions:
            if decision.status in ('APPLIES', 'PARTIAL'):
                reading = ((second or {}).get(decision.entity_id) or {}).get(obligation.provision_ref)
                rows.append(compare_obligation(obligation, decision, profile, register, registry, reading))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.mapping.status] = counts.get(row.mapping.status, 0) + 1
    departments: dict[str, int] = {}
    for row in rows:
        for action in row.actions:
            departments[action['department']] = departments.get(action['department'], 0) + 1
    summary = {'rows': len(rows), 'by_status': dict(sorted(counts.items())), 'open_actions_by_department': dict(sorted(departments.items())),
               'product_nonconformities': sum(1 for r in rows for f in r.product_findings if f.result == 'EXCEEDS_LIMIT'),
               'review_required': sum(r.review_required for r in rows)}
    return GapReport(profile_id=profile.profile_id, register_synthetic=register.synthetic, rows=rows, summary=summary)

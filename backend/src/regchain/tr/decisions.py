"""Board decisions as a company-specific source layer: a Rekabet Kurulu commitment decision read for its addressee.

A regulation binds a class of companies; a board decision (kurul kararı) binds the undertaking it names. The registry
keeps DECISION_PRECEDENT out of every active pack (packs.py), so this layer is read beside the sector engines, for the
entities the profile says are the decision's addressees (LegalEntity.bound_by_decisions, or the addressees named on the
call) and for no other entity. Nothing here knows a company: the record carries the decision's own identifiers, the
profile says whom it binds.

Pipeline: the decision text stored by scripts/tr_decision_import.py (text.txt + decision.json, hash-checked on every
load) -> its numbered commitment items as sections -> the rule reader of regulations (frames.py, extraction.py) ->
every duty bound to the addressee entities (routing with a DECISION_ADDRESSEE gate) -> the policy comparer, the
candidate statements, the adjudicator and the verification exactly as for a regulation -> every row REVIEW_REQUIRED
with DECISION_PRECEDENT_REVIEW: the decision reader has no labelled cases yet, so none of its rows is an automatic
decision. The record and its hash, the exact spans of the stored text and the gates are kept on every row.
"""
import copy
import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..pilot.schema import Strict
from .adapters import INSTITUTIONS, OfficialSource
from .adjudicate import Assessment, assess_obligation, statistics
from .compare import GapReport, Register
from .core import RegulationMeta
from .corpus import CorpusStore
from .extraction import ExtractedObligation, obligations_of, route
from .frames import Frame, regulation_frames
from .packs import DATA, Registry
from .profile import EnterpriseProfile
from .routing import Decision
from .semantic import SimilarityTable

DECISIONS = DATA / 'decisions'
PREFIX = 'TR:KURUL_KARARI:'
RECORD_FORMAT = 'cardaman-tr-decision/1'
RECORD_FILE = 'decision.json'
DECISION_RULES_VERSION = 'tr-decisions-v1'
# "1. ...", "1) ...", "(1) ...", "1- ..." open a commitment item; items are accepted in sequence only, so a number
# inside the text ("35" at the start of a wrapped line) never opens one.
ITEM = re.compile(r'^\s*(?:\((\d{1,3})\)|(\d{1,3})\s*[.)\-–])\s+')


class DecisionError(ValueError):
    pass


class DecisionRecord(Strict):
    """What is known about one stored decision: its identifiers, its publisher, the hash of its stored text."""
    format: Literal['cardaman-tr-decision/1']
    decision_id: str = Field(pattern=r'^TR:KURUL_KARARI:[A-Z0-9][A-Z0-9_.-]*$')
    institution_id: str
    title: str = Field(min_length=1)
    decision_no: str = Field(min_length=1)
    decision_date: date | None = None
    document_url: str
    document_type: Literal['KARAR'] = 'KARAR'
    # The provision the decision rests on, as the decision states it (its act is not catalogued; nothing is read from it).
    legal_basis: str = ''
    review_date: date | None = None
    sector_tags: list[str] = Field(min_length=1)
    activity_tags: list[str] = []
    text_file: str = 'text.txt'
    content_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    retrieved_at: datetime
    synthetic: bool
    disclaimer: str = ''

    @model_validator(mode='after')
    def _official(self):
        institution = INSTITUTIONS.get(self.institution_id)
        if institution is None or institution.default_authority != 'DECISION_PRECEDENT':
            raise ValueError(f'{self.decision_id}: {self.institution_id} is not an allowlisted decision-making authority')
        self.official_source()                                  # host allowlisted for the institution, KARAR
        return self

    def official_source(self) -> OfficialSource:
        return OfficialSource(institution_id=self.institution_id, authority='DECISION_PRECEDENT', document_url=self.document_url,
                              canonical_id=self.decision_id, publication_date=self.decision_date, effective_date=self.decision_date,
                              version=self.version_id, content_hash=self.content_hash, retrieved_at=self.retrieved_at,
                              document_type=self.document_type, synthetic=self.synthetic)

    @property
    def version_id(self) -> str:
        return f'{self.decision_id}@{self.content_hash[:12]}'

    @property
    def label(self) -> str:
        return f'Karar {self.decision_no}'


class DecisionVersion(Strict):
    decision_id: str
    version_id: str
    content_hash: str
    retrieved_at: datetime
    synthetic: bool
    directory: str


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def commitment_sections(text: str, label: str, heading: str = '') -> list[dict]:
    """The numbered items of a commitment text in the engine's section shape (printed_label, text, heading_path, lines),
    each an exact span of its own text; the item's number marker is left out, sub-items ("a) ...") keep their lines so
    the clause splitter reads them under the item's lead-in. A text with no numbered item is one section per paragraph."""
    items: list[list[str]] = []
    for line in text.splitlines():
        match = ITEM.match(line)
        number = int(match.group(1) or match.group(2)) if match else None
        if number == len(items) + 1:
            items.append([line[match.end():].strip()])
        elif items and line.strip():
            items[-1].append(line.strip())
        elif items and not line.strip():
            items[-1].append('')
    kind = 'taahhüt'
    if not items:
        kind = 'paragraf'
        for block in re.split(r'\n\s*\n', text):
            block_lines = [l.strip() for l in block.splitlines() if l.strip()]
            if block_lines:
                items.append(block_lines)
    out = []
    for number, item_lines in enumerate(items, 1):
        while item_lines and not item_lines[-1]:
            item_lines.pop()
        spans, offset, parts = [], 0, []
        for line in item_lines:
            if line:
                spans.append({'start': offset, 'end': offset + len(line)})
            parts.append(line)
            offset += len(line) + 1
        body = '\n'.join(parts)
        out.append({'printed_label': f'{label} {kind} {number}', 'paragraph_number': str(number), 'text': body,
                    'heading_path': ['', '', heading], 'lines': spans, 'quality_flags': []})
    return out


class DecisionStore:
    """The stored decisions under one root: <root>/<folder>/decision.json + the text file it names. The text is
    re-hashed on every load; a mismatch is an error. A synthetic record is refused in the packaged root."""

    def __init__(self, root: Path = DECISIONS):
        self.root = Path(root)
        self._records: dict[str, tuple[DecisionRecord, str, Path]] | None = None

    def _load(self) -> dict[str, tuple[DecisionRecord, str, Path]]:
        if self._records is None:
            self._records = {}
            if self.root.exists():
                for path in sorted(self.root.glob(f'*/{RECORD_FILE}')):
                    record = DecisionRecord.model_validate(json.loads(path.read_text(encoding='utf-8')))
                    if record.synthetic and self.root.resolve() == DECISIONS.resolve():
                        raise DecisionError(f'{record.decision_id}: a synthetic decision cannot be stored in the packaged root')
                    text = (path.parent / record.text_file).read_text(encoding='utf-8')
                    if text_hash(text) != record.content_hash:
                        raise DecisionError(f'{record.decision_id}: the stored text does not match its recorded hash')
                    if record.decision_id in self._records:
                        raise DecisionError(f'{record.decision_id} is stored twice')
                    self._records[record.decision_id] = (record, text, path.parent)
        return self._records

    def decision_ids(self) -> list[str]:
        return sorted(self._load())

    def record(self, decision_id: str) -> DecisionRecord:
        try:
            return self._load()[decision_id][0]
        except KeyError:
            raise DecisionError(f'no stored decision {decision_id}') from None

    def text(self, decision_id: str) -> str:
        self.record(decision_id)
        return self._load()[decision_id][1]

    # -- the corpus-store shape the readers use (extraction.ground, adjudicate.verify) ------------------------
    def versions(self, decision_id: str) -> list[DecisionVersion]:
        if decision_id not in self._load():
            return []
        record, _, directory = self._load()[decision_id]
        return [DecisionVersion(decision_id=decision_id, version_id=record.version_id, content_hash=record.content_hash,
                                retrieved_at=record.retrieved_at, synthetic=record.synthetic, directory=str(directory))]

    def version(self, decision_id: str, version_id: str | None = None) -> DecisionVersion:
        versions = self.versions(decision_id)
        if not versions:
            raise DecisionError(f'no stored decision {decision_id}')
        if version_id is not None and version_id != versions[0].version_id:
            raise DecisionError(f'{decision_id} has no version {version_id}')
        return versions[0]

    def head(self, decision_id: str) -> DecisionVersion | None:
        versions = self.versions(decision_id)
        return versions[0] if versions else None

    def sections(self, decision_id: str, version_id: str | None = None) -> list[dict]:
        self.version(decision_id, version_id)
        record, text, _ = self._load()[decision_id]
        return commitment_sections(text, record.label, record.title)

    def section(self, decision_id: str, printed_label: str, version_id: str | None = None) -> dict:
        found = [s for s in self.sections(decision_id, version_id) if s['printed_label'] == printed_label]
        if not found:
            raise DecisionError(f'{decision_id} has no item {printed_label}')
        return found[0]

    def write(self, record: DecisionRecord, text: str, folder: str | None = None) -> Path:
        """Store a decision (the import script's step); refuses a text whose hash is not the record's."""
        if text_hash(text) != record.content_hash:
            raise DecisionError('the text does not match the record hash')
        if record.synthetic and self.root.resolve() == DECISIONS.resolve():
            raise DecisionError('a synthetic decision cannot be stored in the packaged root')
        directory = self.root / (folder or re.sub(r'[^A-Za-z0-9.-]+', '_', record.decision_id[len(PREFIX):]))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / record.text_file).write_text(text, encoding='utf-8')
        (directory / RECORD_FILE).write_text(json.dumps(record.model_dump(mode='json'), ensure_ascii=False, indent=2), encoding='utf-8')
        self._records = None
        return directory


class LayeredStore:
    """The corpus for regulations, the decision store for decisions: one store the readers can ground every text in."""

    def __init__(self, corpus: CorpusStore, decisions: DecisionStore):
        self.corpus, self.decisions = corpus, decisions

    def _of(self, regulation_id: str):
        return self.decisions if regulation_id.startswith(PREFIX) else self.corpus

    def versions(self, regulation_id: str):
        return self._of(regulation_id).versions(regulation_id)

    def version(self, regulation_id: str, version_id: str | None = None):
        return self._of(regulation_id).version(regulation_id, version_id)

    def head(self, regulation_id: str):
        return self._of(regulation_id).head(regulation_id)

    def sections(self, regulation_id: str, version_id: str | None = None):
        return self._of(regulation_id).sections(regulation_id, version_id)

    def section(self, regulation_id: str, printed_label: str, version_id: str | None = None):
        return self._of(regulation_id).section(regulation_id, printed_label, version_id)

    def regulation_ids(self) -> list[str]:
        return [*self.corpus.regulation_ids(), *self.decisions.decision_ids()]

    def __getattr__(self, name):
        return getattr(self.corpus, name)


def decision_meta(record: DecisionRecord) -> RegulationMeta:
    """The catalogue shape of a decision, for the readers that look a text up by its id. UNVERIFIED: the record is what
    the import stored, checked by hash, not against a catalogue of the authority."""
    note = (f'Board decision stored by hash {record.content_hash[:12]} from {record.document_url}; its identifiers are the '
            f'record\'s, not checked against a catalogue of the authority.' + (' SYNTHETIC test decision.' if record.synthetic else ''))
    return RegulationMeta(regulation_id=record.decision_id, title=record.title, jurisdiction='TR', regulator=record.institution_id,
                          regulation_type='KURUL_KARARI', binding_status='DECISION_PRECEDENT', number=record.decision_no,
                          effective_date=record.decision_date, effective_status='IN_FORCE' if record.decision_date else 'UNKNOWN',
                          metadata_status='UNVERIFIED', source_url=record.document_url, verification_note=note,
                          sector_tags=record.sector_tags, activity_tags=record.activity_tags, default_level='LEGAL_ENTITY')


def registry_with(registry: Registry, metas: list[RegulationMeta]) -> Registry:
    """The registry with the decisions added to its catalogue view (packs, scopes and vocabulary shared, nothing
    re-validated): a decision is never a pack's regulation."""
    view = copy.copy(registry)
    view.regulations = {**registry.regulations, **{m.regulation_id: m for m in metas}}
    return view


def bound_to_addressee(obligation: ExtractedObligation) -> ExtractedObligation:
    """A commitment binds the undertaking that gave it, whatever class the clause's wording names: its scope is the
    legal entity, with no class constraint; the exceptions the clause states stay recorded for the reviewer."""
    scope = obligation.scope.model_copy(update={
        'level': 'LEGAL_ENTITY', 'scope_status': 'DEFINED', 'alcohol_scope': 'ANY', 'entity_classes': [], 'activity_classes': [],
        'facility_classes': [], 'license_classes': [], 'product_classes': [], 'product_attributes': [], 'sales_channels': [],
        'excluded_product_classes': [], 'conditions': []})
    return obligation.model_copy(update={'scope': scope, 'flags': [*obligation.flags, 'DECISION_ADDRESSEE']})


def decision_obligations(record: DecisionRecord, decisions: DecisionStore, registry: Registry) -> tuple[list[Frame], list[ExtractedObligation]]:
    """(every frame, the duties) of a stored decision, read by the rule reader of regulations and bound to the addressee."""
    meta = decision_meta(record)
    version = decisions.version(record.decision_id)
    frames = regulation_frames(decisions.sections(record.decision_id), record.decision_id)
    obligations = obligations_of(frames, meta, registry_with(registry, [meta]), version.version_id)
    return frames, [bound_to_addressee(o) for o in obligations]


def addressees_of(record: DecisionRecord, profile: EnterpriseProfile, named=()) -> set[str]:
    bound = {e.entity_id for e in profile.legal_entities if record.decision_id in e.bound_by_decisions}
    known = {e.entity_id for e in profile.legal_entities}
    unknown = sorted(set(named) - known)
    if unknown:
        raise DecisionError(f'{profile.profile_id} has no legal entity {unknown}')
    return bound | set(named)


def addressed(decision: Decision, bound: set[str]) -> Decision:
    """The routed decision with the addressee gate: the entity the decision binds APPLIES, every other DOES_NOT_APPLY.
    A clause the stored text does not ground stays UNKNOWN."""
    match = decision.target_id in bound
    gate = {'gate': 'DECISION_ADDRESSEE', 'status': 'MATCH' if match else 'MISMATCH', 'clear': True,
            'detail': 'the profile names the entity as an addressee of the decision' if match else 'not an addressee of the decision',
            'addressees': sorted(bound)}
    if decision.status == 'UNKNOWN' and 'GROUNDING_FAILED' in decision.reason_codes:
        return decision.model_copy(update={'gates': [*decision.gates, gate]})
    kept = [r for r in decision.reason_codes if r == 'EXCEPTION_NOT_EVALUATED']
    status, reasons = ('APPLIES', ['DECISION_ADDRESSEE', *kept]) if match else ('DOES_NOT_APPLY', ['DECISION_ADDRESSEE_MISMATCH'])
    final = {**decision.audit.final, 'status': status, 'reason_codes': reasons, 'gates': [*decision.gates, gate]}
    return decision.model_copy(update={'status': status, 'reason_codes': reasons, 'gates': [*decision.gates, gate],
                                       'creates_obligation': match, 'review_required': decision.review_required and match,
                                       'applies_to_products': [], 'audit': decision.audit.model_copy(update={'final': final})})


def decision_applicability(record: DecisionRecord, profile: EnterpriseProfile, registry: Registry, corpus: CorpusStore,
                           decisions: DecisionStore, named=()) -> list[tuple[ExtractedObligation, Decision]]:
    """Every duty of the decision on every legal entity of the profile, with the addressee gate."""
    view = registry_with(registry, [decision_meta(record)])
    layered = LayeredStore(corpus, decisions)
    bound = addressees_of(record, profile, named)
    out = []
    for obligation in decision_obligations(record, decisions, registry)[1]:
        for decision in route(obligation, profile, view, layered)[0]:
            out.append((obligation, addressed(decision, bound)))
    return out


class DecisionRun(Strict):
    decision_id: str
    version_id: str
    title: str
    synthetic: bool
    addressees: list[str]
    items: int
    frames_by_kind: dict[str, int]
    duties: int
    statistics: dict
    summary: dict
    rules_version: str = DECISION_RULES_VERSION


def assess_decision(record: DecisionRecord, profile: EnterpriseProfile, registry: Registry, corpus: CorpusStore, decisions: DecisionStore,
                    register: Register, table: SimilarityTable | None = None, adjudicator=None, named=()) -> tuple[GapReport, list[Assessment], DecisionRun]:
    """(gap report, assessments, run record) of one decision on its addressees against the register: the same comparer,
    candidates, adjudication and verification as a regulation, and every row to a person (DECISION_PRECEDENT_REVIEW)."""
    from .compare import COMPARE_RULES_VERSION
    view = registry_with(registry, [decision_meta(record)])
    layered = LayeredStore(corpus, decisions)
    bound = addressees_of(record, profile, named)
    frames, obligations = decision_obligations(record, decisions, registry)
    from .compare import applicability_review, review_summary
    rows, assessments, reviews = [], [], []
    for obligation in obligations:
        for decision in route(obligation, profile, view, layered)[0]:
            decision = addressed(decision, bound)
            if decision.status == 'UNKNOWN':
                reviews.append(applicability_review(obligation, decision))
            if decision.status != 'APPLIES':
                continue
            row, assessment = assess_obligation(obligation, decision, profile, register, view, table, adjudicator, False, layered)
            rows.append(row.model_copy(update={'review_required': True}))
            assessments.append(assessment.model_copy(update={
                'review_required': True, 'decision': 'REVIEW_REQUIRED', 'applicability_basis': 'DECISION_ADDRESSEE',
                'review_reasons': ['DECISION_PRECEDENT_REVIEW', *assessment.review_reasons]}))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.mapping.status] = counts.get(row.mapping.status, 0) + 1
    summary = {'rows': len(rows), 'by_status': dict(sorted(counts.items())), 'review_required': len(rows),
               'rules_version': COMPARE_RULES_VERSION, **review_summary(reviews)}
    report = GapReport(profile_id=profile.profile_id, register_synthetic=register.synthetic, rows=rows, summary=summary,
                       applicability_reviews=reviews)
    by_kind: dict[str, int] = {}
    for frame in frames:
        by_kind[frame.kind] = by_kind.get(frame.kind, 0) + 1
    duties = len({o.obligation_id for o in obligations})
    run = DecisionRun(decision_id=record.decision_id, version_id=record.version_id, title=record.title, synthetic=record.synthetic,
                      addressees=sorted(bound), items=len(decisions.sections(record.decision_id)), frames_by_kind=dict(sorted(by_kind.items())),
                      duties=duties, statistics=statistics(assessments, duties), summary=summary)
    return report, assessments, run

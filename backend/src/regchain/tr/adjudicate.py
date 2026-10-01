"""Selective adjudication: rules decide, a model reads only the cases the rules cannot, and never overrules them.

Measured on 30 September 2026 (runs 20260930-beverage-live-2/3): handing every duty and every policy passage to the
local models cost 167 to 316 seconds per duty and ended wrong on 11 of 12 labelled coverage cases, while the rule
comparer was right on 11 of them and blind to paraphrase (10 of 28 on a register written in a company's own words).
So the order is turned around:

  1. rules        the rule comparer reads every statement in force (compare.py). Its element checks - a weaker limit,
                  a violated time window, a permitted prohibited act, a missing place - are final.
  2. candidates   for what the rules could not relate, the statements worth a second look (semantic.py): no judgement.
  3. escalation   a row goes to the strong model only for a reason named here: a candidate statement the rules did not
                  relate (SEMANTIC_PARAPHRASE), a candidate that permits what the duty forbids or forbids what it
                  requires (POSSIBLE_CONFLICT), a coverage the rules left open (UNKNOWN_COVERAGE), a duty whose own
                  reading the extraction model contests (READER_DISAGREEMENT). A duty with no related statement and no
                  candidate is NOT_COVERED without any model call. What makes a duty itself hard to read - several
                  exceptions, several parties, a cross-reference - sends the clause, not the comparison, to the strong
                  model (clause_escalation, adjudicate_clause): there the question is what the clause says.
  4. adjudication the strong model (the judge the ModelRouter names) reads the duty and at most four statements and
                  answers per statement: STATES_DUTY, STATES_PART, CONTRADICTS or UNRELATED, with an exact quote.
                  The small extraction model is not asked to judge coverage anywhere in this path.
  5. decision     compare.combine_coverage joins the two readings. The model may raise a coverage the rules left at
                  "nothing found" (paraphrase): to PARTIAL by itself, to COVERS_TEXT for a person to confirm. It may
                  never turn a rule decision around. Where it disagrees with a rule decision, or claims a conflict
                  the rules do not see, the row is REVIEW_REQUIRED and the decision stays (disagreement).

Every adjudication is recorded (JSONL, format cardaman-tr-adjudications/1) with its reasons, request, answer, the
validation of its quotes and its time; an assessment can be replayed from the record without a model.
"""
import json
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from ..pilot.schema import Strict
from .compare import LOOSE_REASONS, GapReport, Passage, Register, compare_obligation, coverage_of, documents_in_force, relate
from .corpus import CorpusStore
from .extraction import ExtractedObligation
from .frames import CITES_PROVISION, fold
from .packs import Registry
from .semantic import Candidate, SimilarityTable, antecedent, candidates, duty_text, frame_text, lead_in, opposes

ADJUDICATION_FORMAT = 'cardaman-tr-adjudications/1'
ADJUDICATION_RULES_VERSION = 'tr-adjudication-v3'
REASONS = ('SEMANTIC_PARAPHRASE', 'POSSIBLE_CONFLICT', 'UNKNOWN_COVERAGE', 'READER_DISAGREEMENT')
CLAUSE_REASONS = ('NESTED_EXCEPTION', 'MULTIPLE_ACTORS', 'CROSS_REFERENCE', 'UNCLEAR_ADDRESSEE', 'READER_DISAGREEMENT')
RELATIONS = ('STATES_DUTY', 'STATES_PART', 'CONTRADICTS', 'UNRELATED')
RELATION_OF = {'STATES_DUTY': 'SUPPORTS', 'STATES_PART': 'PARTIAL', 'CONTRADICTS': 'CONFLICTS', 'UNRELATED': 'UNRELATED'}
MAX_STATEMENTS = 4
PARTY_IDS_IGNORED = {'FIRM', 'WORKPLACE'}

PROMPT_VERSION = 'tr-adjudicate-v3'
PROMPT = '''You compare ONE duty of a Turkish regulation with numbered statements of a company's internal documents.
The texts are Turkish. Answer for every statement, in the order given.

Read the duty first and note what it names: the act it requires or forbids, and each product, place, person, medium,
number, deadline and list item that belongs to that act. "lead_in", when given, only says what the duty's list is
about: it is context, not part of the duty. "previous_sentence", when given, is what the duty refers back to ("bu
sınırlama", "bu yükümlülük"): read the duty with it.

relation is exactly one of:
- STATES_DUTY: a company that follows the statement does everything the duty requires, or refrains from everything it
  forbids. Different wording is fine. A statement that is stricter, more detailed or wider than the duty, or that lists
  more than the duty asks for, is STATES_DUTY. Where the duty says "A or B", either one is enough.
- STATES_PART: the statement is about the same act as the duty, but something the DUTY requires is not in the
  statement: a place, a product, a number, a deadline, a person, an item of the duty's list. Put each such thing in
  "missing", copied from the duty. If you cannot name a missing thing from the duty, it is not STATES_PART.
- CONTRADICTS: a company that follows the statement breaks the duty: the statement allows or requires the very act the
  duty forbids - for a product, place or person the duty names - or sets a weaker limit or a later deadline for the
  same thing.
- UNRELATED: another act, another subject, or only the same product or topic. A statement only about other places or
  other persons than the duty names is UNRELATED.

Rules:
- Judge only what the statement says. Do not assume practice.
- Test for CONTRADICTS: can the company follow the statement and the duty at the same time? If it can, the statement
  does not contradict. A stricter rule never CONTRADICTS. Another act, place, product or person never CONTRADICTS.
- "missing" lists only what the duty has and the statement lacks, never what the statement adds.
- "quote" is copied exactly from the statement: the words that state, narrow or contradict the duty. Empty for UNRELATED.
- "reason" is one short English sentence.'''

SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['judgements'],
          'properties': {'judgements': {'type': 'array', 'minItems': 1, 'maxItems': MAX_STATEMENTS, 'items': {
              'type': 'object', 'additionalProperties': False, 'required': ['id', 'relation', 'quote', 'missing', 'reason'],
              'properties': {'id': {'type': 'string'}, 'relation': {'type': 'string', 'enum': list(RELATIONS)},
                             'quote': {'type': 'string', 'maxLength': 400},
                             'missing': {'type': 'array', 'maxItems': 4, 'items': {'type': 'string', 'maxLength': 120}},
                             'reason': {'type': 'string', 'maxLength': 240}}}}}}


class Escalation(Strict):
    reasons: list[str]
    statements: list[str]                             # passage ids the adjudicator is given, candidates first
    candidates: list[Candidate] = []


class Assessment(Strict):
    """One gap row with how it was reached."""
    provision_ref: str
    obligation_id: str
    target_id: str
    entity_id: str | None = None
    rule_coverage: str
    coverage: str
    basis: str
    review_required: bool
    # AUTO: the rules decided, or the adjudicator filled what they left empty with a PARTIAL. REVIEW_REQUIRED: there is
    # no automatic decision - `coverage` is the rules' word for the time being and `review_reasons` say what is open.
    decision: str = 'AUTO'
    review_reasons: list[str] = []
    proposal: str | None = None                       # what the model would have the coverage be, where that is not the coverage
    adjudication_status: str | None = None            # OK, OK_WITHOUT_THINKING ..., FAILED ...: a failed call is never a silent pass
    # FACTS: the applicability was reached through gates over the company's facts. REGULATION_SCOPE: the clause names no
    # constraint the profile could answer, and the duty reaches the target because its regulation is in the pilot's
    # catalogue (a product duty of a horizontal food rule reaches every product). The row says which.
    applicability_basis: str = 'FACTS'
    disagreement: int = 0
    escalation: Escalation | None = None
    adjudication_key: str | None = None
    model_coverage: str | None = None
    elapsed_ms: int = 0                               # model time spent for this row (0: no call, or replayed from the record)


# -- escalation ------------------------------------------------------------------------------------------
def duty_complexity(obligation: ExtractedObligation) -> list[str]:
    """What makes the duty itself hard to read by rule."""
    frame = obligation.frame
    reasons = []
    exceptions = frame.exceptions
    if len(exceptions) >= 2 or any(e.effect == 'NARROWS' for e in exceptions) or any(not e.predicate and e.about == 'OTHER' for e in exceptions):
        reasons.append('NESTED_EXCEPTION')
    parties = {a.id for a in frame.actors if a.id not in PARTY_IDS_IGNORED}
    if len(parties) >= 2:
        reasons.append('MULTIPLE_ACTORS')
    if frame.references or CITES_PROVISION.search(fold(frame.text)):
        reasons.append('CROSS_REFERENCE')
    return reasons


def escalate(obligation: ExtractedObligation, rule_coverage: str, rule_reasons: list[str], related: list, found: list[Candidate],
             contested: bool = False) -> Escalation | None:
    """Why a row goes to the adjudicator, or None. `related` are the rule comparer's readings other than UNRELATED."""
    reasons = []
    loose = rule_coverage == 'NO_EVIDENCE' or (rule_coverage == 'PARTIAL' and set(rule_reasons) <= LOOSE_REASONS)
    if loose and (found or related):
        reasons.append('SEMANTIC_PARAPHRASE')
    against = [c for c in found if opposes(c, obligation)]
    if rule_coverage != 'CONFLICT' and against:
        reasons.append('POSSIBLE_CONFLICT')
    if rule_coverage == 'UNKNOWN':
        reasons.append('UNKNOWN_COVERAGE')
    if contested and (found or related):
        reasons.append('READER_DISAGREEMENT')
    if not reasons:
        return None
    given = found if 'SEMANTIC_PARAPHRASE' in reasons or 'READER_DISAGREEMENT' in reasons else against
    statements = [c.passage_id for c in given] + [r.passage_id for r in related if r.passage_id not in {c.passage_id for c in given}]
    return Escalation(reasons=reasons, statements=statements[:MAX_STATEMENTS], candidates=given)


# -- the adjudicator -------------------------------------------------------------------------------------
def adjudication_key(obligation: ExtractedObligation, passages: list[Passage]) -> str:
    text = '\n'.join([PROMPT_VERSION, duty_text(obligation), *(f'{p.passage_id}|{p.text}' for p in passages)])
    return sha256(text.encode('utf-8')).hexdigest()


def request_payload(obligation: ExtractedObligation, passages: list[Passage]) -> dict:
    frame = obligation.frame
    return {'duty': {'text': frame_text(frame), **({'lead_in': lead_in(frame)} if lead_in(frame) else {}),
                     **({'previous_sentence': antecedent(frame)} if antecedent(frame) else {}), 'kind': frame.kind,
                     'exceptions': [e.quote for e in frame.exceptions][:3],
                     'limits': [q.text for q in frame.quantities if q.role == 'LIMIT'][:4]},
            'statements': [{'id': f'S{i + 1}', 'text': p.text} for i, p in enumerate(passages)]}


def validate(answer: dict, passages: list[Passage]) -> list[dict]:
    """The judgements with their quotes checked against the statements: a quote that is not in its statement makes the
    judgement UNCLEAR (grounding), a statement the answer skipped is UNCLEAR too, and a "states the duty" that names
    something missing counts as a part."""
    by_id = {f'S{i + 1}': p for i, p in enumerate(passages)}
    seen = {}
    for item in answer.get('judgements') or []:
        passage = by_id.get(str(item.get('id')))
        if passage is None or passage.passage_id in seen:
            continue
        relation = item.get('relation')
        quote = (item.get('quote') or '').strip()
        grounded = relation == 'UNRELATED' or (bool(quote) and _squash(quote).strip(' .…') in _squash(passage.text))
        missing = [m for m in item.get('missing') or [] if isinstance(m, str) and m.strip()]
        # "States the duty" with something named as missing is, by the answer's own account, a part (DEV: the statement
        # on sugary drinks in children's television against the duty on every red-listed food, with both gaps listed).
        read = 'STATES_PART' if relation == 'STATES_DUTY' and missing else relation
        seen[passage.passage_id] = {'passage_id': passage.passage_id, 'document_id': passage.document_id, 'relation': relation,
                                    'rule_relation': RELATION_OF.get(read, 'UNCLEAR') if grounded else 'UNCLEAR', 'quote': quote,
                                    'missing': missing, 'reason': item.get('reason') or '', 'grounded': grounded}
    for passage in passages:
        seen.setdefault(passage.passage_id, {'passage_id': passage.passage_id, 'document_id': passage.document_id, 'relation': None,
                                             'rule_relation': 'UNCLEAR', 'quote': '', 'missing': [], 'reason': 'not answered', 'grounded': False})
    return [seen[p.passage_id] for p in passages]


def _squash(text: str) -> str:
    """A text as it is compared for grounding: its statement number and its spacing aside, a quote is the statement's
    characters or it is no quote. On the DEV cases 3 of 18 answers quoted the right words with a capital at the start of
    a half-sentence or with straight quotation marks and were dropped; the check was not widened for them - they are
    counted as grounding losses."""
    return re.sub(r'\s+', ' ', re.sub(r'^\s*\d{1,3}\.\s+', '', text or '')).strip()


def model_coverage(judgements: list[dict]) -> str:
    relations = [j['rule_relation'] for j in judgements]
    for relation, coverage in (('CONFLICTS', 'CONFLICT'), ('SUPPORTS', 'COVERS_TEXT'), ('PARTIAL', 'PARTIAL')):
        if relation in relations:
            return coverage
    return 'UNKNOWN' if relations and all(r == 'UNCLEAR' for r in relations) else 'NO_EVIDENCE'


class Adjudicator:
    """Asks the strong model, or replays a record. One answer per (duty, statements): rows of the same duty on other
    targets with the same statements share it."""

    def __init__(self, provider=None, record_path: Path | None = None, recorded: dict | None = None):
        self.provider = provider
        self.recorded = dict(recorded or {})
        self.record_path = Path(record_path) if record_path else None
        self.calls = 0
        self.elapsed_ms = 0
        if self.record_path and provider is not None and not self.record_path.exists():
            from ..extraction.providers import runtime_manifest
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self._write({'format': ADJUDICATION_FORMAT, 'started_at': datetime.now(timezone.utc).isoformat(), 'prompt': PROMPT_VERSION,
                         'prompt_sha256': sha256(PROMPT.encode('utf-8')).hexdigest(), 'runtime': runtime_manifest(provider)})

    def _write(self, record: dict) -> None:
        with self.record_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    def _replayed(self, key: str, passages: list[Passage]) -> dict:
        """A recorded answer under the validator of today: the answer is the model's, its validation is this code's."""
        record = self.recorded[key]
        if record.get('validated_by') == ADJUDICATION_RULES_VERSION:
            return record
        try:
            answer = json.loads(record.get('raw') or '')
        except ValueError:
            answer = None
        if isinstance(answer, dict):
            judgements = validate(answer, passages)
            record = {**record, 'judgements': judgements, 'coverage': model_coverage(judgements)}
        self.recorded[key] = record = {**record, 'validated_by': ADJUDICATION_RULES_VERSION}
        return record

    def __call__(self, obligation: ExtractedObligation, passages: list[Passage], reasons: list[str]) -> tuple[dict | None, int]:
        """(record, model milliseconds spent now). None when there is neither a record nor a provider."""
        key = adjudication_key(obligation, passages)
        if key in self.recorded:
            return self._replayed(key, passages), 0
        if self.provider is None:
            return None, 0
        from ..extraction.providers import ai_task
        payload = request_payload(obligation, passages)
        began = time.monotonic()
        with ai_task('tr_adjudication'):
            status, answer, raw = ask(self.provider, PROMPT, payload, SCHEMA)
        elapsed = int((time.monotonic() - began) * 1000)
        judgements = validate(answer if isinstance(answer, dict) else {}, passages)
        record = {'key': key, 'provision_ref': obligation.provision_ref, 'obligation_id': obligation.obligation_id, 'reasons': reasons,
                  'statements': [p.passage_id for p in passages], 'payload': payload, 'status': status, 'raw': raw[:4000],
                  'judgements': judgements, 'coverage': model_coverage(judgements), 'elapsed_ms': elapsed,
                  'validated_by': ADJUDICATION_RULES_VERSION}
        self.recorded[key] = record
        self.calls += 1
        self.elapsed_ms += elapsed
        if self.record_path:
            self._write(record)
        return record, elapsed


def ask(provider, prompt: str, payload: dict, schema: dict) -> tuple[str, dict, str]:
    """(status, answer, raw) of one structured call. A provider or parse failure is a recorded outcome, not an error;
    a thinking model that ran out of its budget (measured: 2 of 17 DEV calls, 108 s each) is asked once more without
    thinking, and the status says so."""
    from ..pilot.engine import structured
    failure = ''
    for attempt in dict.fromkeys((provider, getattr(provider, 'quick', provider))):
        raw = ''
        try:
            raw = structured(attempt, prompt, payload, schema)
            answer = json.loads(raw)
            if isinstance(answer, dict):
                return ('OK' if not failure else f'OK_WITHOUT_THINKING after {failure}')[:300], answer, raw
            failure = 'FAILED: answer is not an object'
        except Exception as exc:
            failure = f'FAILED: {type(exc).__name__}: {exc}'
    return failure[:300], {}, raw


def load_adjudications(path: Path) -> tuple[dict, dict]:
    """(header, {key: record}) of a recorded adjudication file."""
    lines = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines or lines[0].get('format') != ADJUDICATION_FORMAT:
        raise ValueError(f'{path} is not a {ADJUDICATION_FORMAT} file')
    return lines[0], {r['key']: r for r in lines[1:]}


# -- the decision ----------------------------------------------------------------------------------------
ORDER = {'NO_EVIDENCE': 0, 'PARTIAL': 1, 'COVERS_TEXT': 2}


def disagreement(rule: str, reasons: list[str], model: str | None) -> int:
    """How far the model's reading is from a decision of the rules: 0 none, 1 neighbours (covers / partial), 2 something
    against nothing, 3 a conflict on one side only. 2 and 3 send the row to a person; the decision is the rules' either
    way. Where the rules decided nothing (no statement found, loose wording), a model that finds the statement fills a
    gap and disagrees with nobody; a conflict it claims there is still a 3."""
    if model is None or rule == model:
        return 0
    if 'CONFLICT' in (rule, model):
        return 3
    if 'UNKNOWN' in (rule, model):
        return 1
    if rule == 'NO_EVIDENCE' or (rule == 'PARTIAL' and set(reasons) <= LOOSE_REASONS):
        return 0
    return abs(ORDER[rule] - ORDER[model])


def verify(obligation: ExtractedObligation, decision, row, register: Register, store: CorpusStore | None = None) -> list[str]:
    """What a decision rests on, checked again whoever made it: the clause and its exceptions are exact spans of the stored
    version, every statement the coverage rests on is an exact span of a document in force for the target, and the
    applicability names the gates (company facts) it was reached through. A problem sends the row to a person."""
    problems = []
    if store is not None:
        from .extraction import ground
        if not ground(obligation, store)['grounded']:
            problems.append('CLAUSE_NOT_GROUNDED')
    by_id = {p.passage_id: p for p in register.passages}
    for reading in row.readings:
        passage = by_id.get(reading.passage_id)
        if passage is None or passage.document_id not in row.documents_in_force:
            problems.append('STATEMENT_NOT_IN_FORCE')
        elif reading.quote and _squash(reading.quote) not in _squash(passage.text):
            problems.append('STATEMENT_QUOTE_NOT_EXACT')
    if not decision.reason_codes or (not decision.gates and 'NO_CONSTRAINT' not in decision.reason_codes):
        problems.append('APPLICABILITY_WITHOUT_FACTS')
    return sorted(set(problems))


def applicability_basis(obligation: ExtractedObligation, decision) -> str:
    return 'REGULATION_SCOPE' if 'NO_CONSTRAINT' in decision.reason_codes and not decision.gates else 'FACTS'


def assess_obligation(obligation: ExtractedObligation, decision, profile, register: Register, registry: Registry,
                      table: SimilarityTable | None = None, adjudicator: Adjudicator | None = None, contested: bool = False,
                      store: CorpusStore | None = None):
    """(gap row, assessment) of one obligation on one target: rules, candidates, escalation, adjudication, decision."""
    in_force = {d.document_id for d in documents_in_force(register, profile, decision)}
    passages = [p for p in register.passages if p.document_id in in_force]
    duty = duty_text(obligation)
    similarity = (lambda passage: table.get(duty, passage.text)) if table is not None else None
    readings = [relate(obligation, p, registry.vocabulary, similarity(p) if similarity else None) for p in passages]
    related = [r for r in readings if r.relation != 'UNRELATED']
    rule_coverage, rule_reasons = coverage_of(related)
    found = candidates(obligation, passages, registry.vocabulary, table, exclude={r.passage_id for r in related})
    escalation = escalate(obligation, rule_coverage, rule_reasons, related, found, contested)
    second, key, elapsed, model, status = None, None, 0, None, None
    if escalation is not None and adjudicator is not None:
        by_id = {p.passage_id: p for p in passages}
        given = [by_id[i] for i in escalation.statements]
        record, elapsed = adjudicator(obligation, given, escalation.reasons)
        if record is not None:
            key, model, status = record['key'], record['coverage'], record.get('status')
            quotes: dict[str, list[str]] = {}
            for judgement in record['judgements']:
                if judgement['rule_relation'] in ('SUPPORTS', 'PARTIAL', 'CONFLICTS'):
                    quotes.setdefault(judgement['rule_relation'], []).append(by_id[judgement['passage_id']].text)
            second = {'coverage': model, 'quotes': quotes, 'adjudicated': True}
    row = compare_obligation(obligation, decision, profile, register, registry, second, similarity)
    score = disagreement(rule_coverage, rule_reasons, model)
    # What keeps the row from being an automatic decision. The rules' own conflict is a decision (a person acts on it,
    # nobody has to decide it); a model that contests a rule decision, proposes a coverage or a conflict alone, or was
    # asked and gave no usable answer, is not.
    open_points = []
    if row.coverage_basis in ('MODEL_PROPOSES_COVERED', 'MODEL_PROPOSES_PARTIAL', 'MODEL_CONFLICT_UNCONFIRMED', 'MODEL_SEES_GAP',
                              'MODEL_PARAPHRASE_UNCONFIRMED', 'MODEL_PARTIAL_UNCONFIRMED'):
        open_points.append(row.coverage_basis)
    if score >= 2 and 'MODEL_CONFLICT_UNCONFIRMED' not in open_points:
        open_points.append('READER_DISAGREEMENT')
    if row.document_coverage == 'UNKNOWN':
        open_points.append('COVERAGE_UNKNOWN')
    if escalation is not None and adjudicator is not None and (model is None or model == 'UNKNOWN'):
        open_points.append('ADJUDICATION_UNRESOLVED')          # no record, a failed call, or no grounded judgement
    if decision.review_required:
        open_points.append('APPLICABILITY_REVIEW')
    by_scope = applicability_basis(obligation, decision)
    if by_scope == 'REGULATION_SCOPE' and 'ADDRESSEE_UNCLEAR' in obligation.flags:
        open_points.append('ADDRESSEE_UNCLEAR')           # reaches the target by default, and the clause does not say whom it binds
    open_points += [f'NOT_VERIFIED:{problem}' for problem in verify(obligation, decision, row, register, store)]
    review = row.review_required or bool(open_points)
    if review != row.review_required:
        row = row.model_copy(update={'review_required': review})
    proposal = model if model is not None and model != row.document_coverage else None
    return row, Assessment(provision_ref=obligation.provision_ref, obligation_id=obligation.obligation_id, target_id=decision.target_id,
                           entity_id=decision.entity_id, rule_coverage=rule_coverage, coverage=row.document_coverage,
                           basis=row.coverage_basis, review_required=review, decision='REVIEW_REQUIRED' if open_points else 'AUTO',
                           review_reasons=open_points, proposal=proposal, adjudication_status=status, applicability_basis=by_scope, disagreement=score,
                           escalation=escalation, adjudication_key=key, model_coverage=model, elapsed_ms=elapsed)


def assess_profile(profile, obligations: list[ExtractedObligation], register: Register, registry: Registry, store: CorpusStore | None = None,
                   table: SimilarityTable | None = None, adjudicator: Adjudicator | None = None, contested: set[str] = frozenset()):
    """(gap report, assessments, statistics) for every obligation on every target it applies to."""
    from .compare import COMPARE_RULES_VERSION
    from .extraction import route
    store = store or CorpusStore()
    rows, assessments = [], []
    for obligation in obligations:
        decisions, _ = route(obligation, profile, registry, store)
        for decision in decisions:
            if decision.status in ('APPLIES', 'PARTIAL'):
                row, assessment = assess_obligation(obligation, decision, profile, register, registry, table, adjudicator,
                                                    obligation.provision_ref in contested, store)
                rows.append(row)
                assessments.append(assessment)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.mapping.status] = counts.get(row.mapping.status, 0) + 1
    report = GapReport(profile_id=profile.profile_id, register_synthetic=register.synthetic, rows=rows,
                       summary={'rows': len(rows), 'by_status': dict(sorted(counts.items())),
                                'review_required': sum(r.review_required for r in rows), 'rules_version': COMPARE_RULES_VERSION})
    return report, assessments, statistics(assessments, len({o.obligation_id for o in obligations}))


def statistics(assessments: list[Assessment], obligations: int | None = None) -> dict:
    """Escalation and cost of an assessment: how many rows and duties reached the model, why, and the model time per duty."""
    escalated = [a for a in assessments if a.escalation is not None]
    keys = {a.adjudication_key for a in escalated if a.adjudication_key}
    duties = {a.obligation_id for a in assessments}
    reasons: dict[str, int] = {}
    for item in escalated:
        for reason in item.escalation.reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    total_ms = sum(a.elapsed_ms for a in assessments)
    count = obligations or len(duties)
    calls = {a.adjudication_key: a for a in escalated if a.adjudication_key}
    spent = sorted(a.elapsed_ms for a in calls.values() if a.elapsed_ms)
    open_rows = [a for a in assessments if a.decision == 'REVIEW_REQUIRED']
    open_reasons: dict[str, int] = {}
    for item in open_rows:
        for reason in item.review_reasons:
            open_reasons[reason.split(':')[0]] = open_reasons.get(reason.split(':')[0], 0) + 1
    return {'rows': len(assessments), 'obligations': count, 'obligations_with_rows': len(duties),
            'escalated_rows': len(escalated), 'escalated_obligations': len({a.obligation_id for a in escalated}),
            'escalation_rate_rows': round(len(escalated) / len(assessments), 4) if assessments else None,
            'escalation_rate_obligations': round(len({a.obligation_id for a in escalated}) / count, 4) if count else None,
            'adjudications': len(keys), 'by_reason': dict(sorted(reasons.items())),
            'review_required': sum(a.review_required for a in assessments),
            # rows without an automatic decision, and the duties a person has to look at because of them
            'decision_review_rows': len(open_rows), 'decision_review_obligations': len({a.obligation_id for a in open_rows}),
            'decision_review_rate_rows': round(len(open_rows) / len(assessments), 4) if assessments else None,
            'decision_review_reasons': dict(sorted(open_reasons.items())),
            'proposals': _count(f'{a.rule_coverage}->{a.proposal}' for a in assessments if a.proposal),
            'calls_failed': sum((a.adjudication_status or '').startswith('FAILED') for a in calls.values()),
            'calls_without_thinking': sum((a.adjudication_status or '').startswith('OK_WITHOUT_THINKING') for a in calls.values()),
            'escalated_unresolved_rows': sum('ADJUDICATION_UNRESOLVED' in a.review_reasons for a in assessments),
            'rows_applying_by_regulation_scope': sum(a.applicability_basis == 'REGULATION_SCOPE' for a in assessments),
            'rows_not_verified': sum(any(r.startswith('NOT_VERIFIED') for r in a.review_reasons) for a in assessments),
            'model_ms': total_ms, 'model_ms_per_obligation': round(total_ms / count, 1) if count else None,
            'model_ms_per_call': round(sum(spent) / len(spent), 1) if spent else None,
            'model_ms_per_call_median': spent[len(spent) // 2] if spent else None, 'model_ms_per_call_max': spent[-1] if spent else None,
            'by_basis': _count(a.basis for a in assessments), 'rules_version': ADJUDICATION_RULES_VERSION}


def _count(values) -> dict:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


# -- clause adjudication ---------------------------------------------------------------------------------
# What a clause says is the rule reader's decision (frames.py). The strong model reads a clause only where that reading
# is hard or contested, and its answer changes nothing by itself: where it differs from the rules the clause is marked
# for review.
CLAUSE_FORMAT = 'cardaman-tr-clause-adjudications/1'
CLAUSE_PROMPT_VERSION = 'tr-clause-adjudicate-v2'
CLAUSE_PROMPT = '''You read ONE clause of a Turkish regulation (the text is Turkish) for a company that makes, imports,
distributes, sells or advertises beverages.

Turkish legal drafting states a duty in the plain present tense, mostly passive: "yapılır", "asılır", "belirtilir",
"bulundurulur", "adlandırılır", "yer alır", "gönderir", as well as "zorundadır", "zorunludur", "gerekir", "...malıdır".
A clause that says how a company's act "is done", what a label "contains" or how a product "is named" is a DUTY, not a
description. Decide by the verb that carries the rule: "... bulunması zorunludur" is a DUTY even when its effect
restricts a place or a sale.

kind is exactly one of:
- DUTY: the clause requires somebody other than a public authority to do something, or requires a property of a
  product, a label, a package, an advertisement or a workplace.
- PROHIBITION: the clause forbids somebody other than a public authority to do something ("...amaz", "...emez",
  "yasaktır", "...ilmez").
- BOTH: one clause that states a duty and a prohibition.
- NONE: anything else - a permission ("...abilir"), an exception or exemption ("uygulanmaz", "hariçtir", "zorunlu
  değildir"), a definition, a scope statement, a classification, a statement of what something counts as, a task or
  power of a public authority (Bakanlık, Kurum, Ajans, Valilik, belediye), a delegation to secondary rules ("... ile
  belirlenir"), or a sentence that only makes another text applicable.

"lead_in", when given, only says what the clause's list is about: context, not part of the clause.
marker: the exact words of the clause that carry the duty or the prohibition; empty for NONE.
addressee: who has to comply, copied exactly from the clause; empty when the clause names nobody.
exceptions: every exception or reservation the clause itself states, copied exactly; empty list when none.
reason: one short English sentence.'''
CLAUSE_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['kind', 'marker', 'addressee', 'exceptions', 'reason'],
                 'properties': {'kind': {'type': 'string', 'enum': ['DUTY', 'PROHIBITION', 'BOTH', 'NONE']},
                                'marker': {'type': 'string', 'maxLength': 200}, 'addressee': {'type': 'string', 'maxLength': 200},
                                'exceptions': {'type': 'array', 'maxItems': 4, 'items': {'type': 'string', 'maxLength': 300}},
                                'reason': {'type': 'string', 'maxLength': 240}}}
KIND_OF = {'DUTY': 'OBLIGATION', 'PROHIBITION': 'PROHIBITION', 'BOTH': 'BOTH', 'NONE': 'NONE'}


clause_text = frame_text


def clause_escalation(frame, obligations: list[ExtractedObligation], support: str | None = None) -> list[str]:
    """Why a clause goes to the strong model: `support` is how far the extraction model stands behind the rule reading
    (ai.ClauseReading.support), None when the clause was not read by it."""
    reasons = []
    if support in ('CONTESTED', 'MODEL_ONLY'):
        reasons.append('READER_DISAGREEMENT')
    for obligation in obligations:
        reasons += [r for r in duty_complexity(obligation) if r not in reasons]
        if obligation.basis == 'UNCLEAR' and 'UNCLEAR_ADDRESSEE' not in reasons:
            reasons.append('UNCLEAR_ADDRESSEE')
    return reasons


class ClauseAdjudicator:
    """The strong model on one clause, or the replay of a record."""

    def __init__(self, provider=None, record_path: Path | None = None, recorded: dict | None = None):
        self.provider = provider
        self.recorded = dict(recorded or {})
        self.record_path = Path(record_path) if record_path else None
        self.calls = 0
        if self.record_path and provider is not None and not self.record_path.exists():
            from ..extraction.providers import runtime_manifest
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'format': CLAUSE_FORMAT, 'started_at': datetime.now(timezone.utc).isoformat(),
                                         'prompt': CLAUSE_PROMPT_VERSION, 'prompt_sha256': sha256(CLAUSE_PROMPT.encode('utf-8')).hexdigest(),
                                         'runtime': runtime_manifest(provider)}, ensure_ascii=False) + '\n')

    def __call__(self, frame, reasons: list[str]) -> dict | None:
        text = clause_text(frame)
        key = sha256(f'{CLAUSE_PROMPT_VERSION}\n{text}'.encode('utf-8')).hexdigest()
        if key in self.recorded:
            return self.recorded[key]
        if self.provider is None:
            return None
        from ..extraction.providers import ai_task
        began = time.monotonic()
        with ai_task('tr_clause_adjudication'):
            status, answer, raw = ask(self.provider, CLAUSE_PROMPT, {'clause': text, **({'lead_in': lead_in(frame)} if lead_in(frame) else {}),
                                                                      'heading': frame.heading}, CLAUSE_SCHEMA)
        marker = (answer.get('marker') or '').strip()
        kind = KIND_OF.get(answer.get('kind'))
        grounded = kind == 'NONE' or (bool(marker) and marker.strip(' .…') in text)
        record = {'key': key, 'ref': frame.ref, 'reasons': reasons, 'input': text, 'status': status, 'raw': raw[:3000],
                  'kind': kind if grounded else None, 'grounded': grounded, 'marker': marker, 'addressee': answer.get('addressee') or '',
                  'exceptions': [e for e in answer.get('exceptions') or [] if isinstance(e, str)], 'reason': answer.get('reason') or '',
                  'elapsed_ms': int((time.monotonic() - began) * 1000)}
        self.recorded[key] = record
        self.calls += 1
        if self.record_path:
            with self.record_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        return record


def load_clause_adjudications(path: Path) -> tuple[dict, dict]:
    lines = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines or lines[0].get('format') != CLAUSE_FORMAT:
        raise ValueError(f'{path} is not a {CLAUSE_FORMAT} file')
    return lines[0], {r['key']: r for r in lines[1:]}


def clause_verdict(rule: str, record: dict | None) -> dict:
    """The rule reader's kind with the strong model's beside it. The kind is the rules'; a model that reads another kind,
    or whose marker is not in the clause, sends the clause to a person."""
    if record is None:
        return {'kind': rule, 'model': None, 'agree': None, 'review': False}
    model = record.get('kind')
    agree = model is not None and (model == rule or (model == 'BOTH' and rule != 'NONE') or (rule == 'BOTH' and model != 'NONE'))
    return {'kind': rule, 'model': model, 'agree': agree, 'review': not agree}

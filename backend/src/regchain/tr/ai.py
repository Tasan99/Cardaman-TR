"""AI reliability for the beverage packs: two readers per clause, and the layer of every result that is not shared.

A clause is read twice. The rule reader (frames.py) reads the wording; the model reader is the extraction model's
answer to the unit the clause sits in, as live.py recorded it after the engine's gates. Neither is a label. Where
they agree a duty is CONFIRMED; where they do not, the recorded stage says which layer withheld or produced the
answer, and that layer - not the model's name - is what gets fixed:

  RETRIEVAL            the text the answer needed was not in front of the reader: a provision the unit cites was not
                       supplied, the policy statement was not among the passages judged, an annex is not in the corpus
  MODEL                the model answered and the answer is wrong or incomplete: a duty of the unit left out, a
                       permission or an authority's task read as a duty, the second reading withdrew a real duty,
                       a supporting passage accepted although it omits an element
  GROUNDING            the answer is not a copy of the source: an action that is no span of the unit, a quoted
                       contradiction that is not in the passage, a quote that is not in the stored version
  SCOPE                the duty was read but addressed wrongly: actor, entity, product, activity, facility, condition
                       or exception; or the profile did not state the fact the scope asks for
  VALIDATOR_PIPELINE   a deterministic step is wrong: the unit handed to the model, the pre-classification, the duty
                       markers the gate knows, a wording rule of the rule reader, a comparison rule

compare_readings needs no label: it reports agreement and, per disagreement, the layer the record points at.
diagnose_* name the layer of a result a label (or a reviewer) has called wrong. A diagnosis is a reading of recorded
stages by rule; it is as good as those records, and it is INDICATIVE like everything else here.
"""
import json
import re
from collections import Counter
from typing import Literal

from ..pilot.schema import Strict
from .extraction import ExtractedObligation
from .frames import Frame

LAYERS = ('RETRIEVAL', 'MODEL', 'GROUNDING', 'SCOPE', 'VALIDATOR_PIPELINE')
Layer = Literal['RETRIEVAL', 'MODEL', 'GROUNDING', 'SCOPE', 'VALIDATOR_PIPELINE']
Duty = Literal['OBLIGATION', 'PROHIBITION', 'BOTH', 'NONE']
AI_RULES_VERSION = 'tr-ai-diagnosis-v1'
# Engine coverage words and the words a gap register uses.
COVERAGE_WORDS = {'COVERS_TEXT': 'COVERED', 'PARTIAL': 'PARTIALLY_COVERED', 'NO_EVIDENCE': 'NOT_COVERED', 'CONFLICT': 'CONTRADICTED',
                  'UNKNOWN': 'UNKNOWN'}


class Diagnosis(Strict):
    layer: Layer
    code: str
    detail: str = ''
    stage: str = ''                                   # the recorded stage the diagnosis was read from


class ClauseReading(Strict):
    ref: str
    unit_id: str | None = None
    rule_kind: str                                    # the frame kind (OBLIGATION ... OTHER)
    rule: Duty
    model: Duty | None = None                         # None: no recorded reading covers the clause
    model_reason: str = ''
    agreement: Literal['BOTH_DUTY', 'BOTH_NONE', 'RULE_ONLY', 'MODEL_ONLY', 'POLARITY_DIFFERS', 'NOT_READ']
    # What the two readings make together: the rule reader's clause, and how far the model stands behind it.
    verdict: Duty
    support: Literal['CONFIRMED', 'PARTLY_CONFIRMED', 'RULE_ONLY', 'MODEL_ONLY', 'CONTESTED', 'NOT_READ']
    diagnosis: Diagnosis | None = None
    review_required: bool = False


# -- the two readings ------------------------------------------------------------------------------------
def rule_duty(frame: Frame) -> str:
    return frame.kind if frame.kind in ('OBLIGATION', 'PROHIBITION') else 'NONE'


def model_duty(record: dict) -> str:
    """What the recorded model reading of a unit amounts to after the gates: the modalities of the candidates kept."""
    modalities = {c['modality'] for c in record.get('candidates') or []}
    must, must_not = bool(modalities & {'MUST', 'SHOULD'}), bool(modalities & {'MUST_NOT', 'SHOULD_NOT', 'MAY_NOT'})
    return 'BOTH' if must and must_not else 'OBLIGATION' if must else 'PROHIBITION' if must_not else 'NONE'


def _answers(record: dict) -> list[dict]:
    """The model's own answers before the gates, parsed from the recorded response excerpts (newest last)."""
    found = []
    for entry in record.get('diagnostics') or []:
        raw = entry.get('response_excerpt') if isinstance(entry, dict) else None
        if not raw or entry.get('response_truncated'):
            continue
        try:
            found.append(json.loads(raw))
        except ValueError:
            continue
    return found


def _copied(record: dict) -> bool | None:
    """Whether every action of the model's last answer is a span of the unit (None: no answer was recorded)."""
    answers = _answers(record)
    if not answers:
        return None
    text = record.get('input', '')
    actions = [o.get('action') or o.get('required_action') or o.get('prohibited_action') or '' for o in answers[-1].get('obligations') or []]
    return all(a.strip().rstrip('.') in text for a in actions if a.strip())


def diagnose_reading(record: dict) -> Diagnosis:
    """The layer at which a unit's duty was lost, read from its recorded status, reason and gate messages.
    Call it for a unit that holds a duty (by label, by the rule reader, or by a reviewer) and yielded none."""
    status, reason = record.get('status', ''), record.get('reason', '')
    messages = ' | '.join(f"{d.get('code', '')}: {d.get('detail', '')}" for d in record.get('diagnostics') or [] if isinstance(d, dict))
    if reason.startswith('CLASSIFIED_'):
        return Diagnosis(layer='VALIDATOR_PIPELINE', code='PRECLASSIFIED_NOT_A_DUTY', stage='classification',
                         detail=f"the unit was classed {reason[11:]} by its wording (marker '{(record.get('classification') or {}).get('marker', '')}') "
                                'and never reached the model')
    if reason == 'NO_DUTY_MARKER' or 'No explicit modal evidence' in messages:
        return Diagnosis(layer='VALIDATOR_PIPELINE', code='GATE_MARKER_GAP', stage='grounding',
                         detail='the gate knows no duty marker in this wording, so no answer could be accepted')
    if reason == 'UNRESOLVED_CROSS_REFERENCE' or 'UNRESOLVED_CROSS_REFERENCE' in messages:
        gaps = (record.get('context') or {}).get('unresolved_references') or []
        return Diagnosis(layer='RETRIEVAL', code='REFERENCE_NOT_SUPPLIED', stage='context',
                         detail='a provision the unit cites was not in the context' + (f': {gaps[:3]}' if gaps else ''))
    if status == 'FAILED':
        return Diagnosis(layer='VALIDATOR_PIPELINE', code='PROVIDER_FAILURE', stage='provider', detail=reason[:200])
    if 'Contextual qualifier' in messages or 'Supporting citation' in messages:
        return Diagnosis(layer='RETRIEVAL', code='CONTEXT_QUALIFIER_DEMANDED', stage='grounding',
                         detail='the gate demanded a qualifier of a context provision the answer did not attribute')
    if 'UNCOVERED_MODAL' in messages:
        return Diagnosis(layer='MODEL', code='DUTY_LEFT_OUT', stage='grounding',
                         detail='the answer covered some duty markers of the unit and left one out; the gate keeps all or nothing')
    if 'DUPLICATE_DUTY' in messages:
        return Diagnosis(layer='MODEL', code='DUTY_REPEATED', stage='grounding',
                         detail='the answer repeated one action for several subjects instead of one duty')
    if 'MISSING_QUALIFIER' in messages:
        return Diagnosis(layer='MODEL', code='QUALIFIER_LEFT_OUT', stage='grounding',
                         detail='the answer left out a condition or exception the unit states')
    if 'ACTION_ALIGNMENT' in messages or 'no exact source evidence' in messages or 'Quote must preserve' in messages:
        copied = _copied(record)
        if copied is False:
            return Diagnosis(layer='GROUNDING', code='ACTION_NOT_IN_SOURCE', stage='grounding',
                             detail='the action the model returned is not a span of the unit (paraphrase, translation or invention)')
        return Diagnosis(layer='MODEL', code='ACTION_MISALIGNED', stage='grounding',
                         detail='the answer copied the source but named the wrong subject, action span or modality for its duty marker')
    if reason in ('SECOND_PASS_UNCERTAIN', 'SECOND_PASS_REJECTED') or 'SECOND_PASS' in messages:
        return Diagnosis(layer='MODEL', code='REVIEW_WITHDREW_DUTY', stage='review', detail='the second reading did not support the candidate')
    if status == 'NO_EXPLICIT_OBLIGATION':
        return Diagnosis(layer='MODEL', code='DUTY_DENIED', stage='extraction', detail='the model answered that the unit states no duty')
    if 'MALFORMED' in messages or 'schema' in messages.lower() or 'JSON' in messages:
        return Diagnosis(layer='MODEL', code='MALFORMED_ANSWER', stage='extraction', detail='the answer was not the required object')
    return Diagnosis(layer='VALIDATOR_PIPELINE', code='UNCLASSIFIED_REJECTION', stage='grounding', detail=(reason or status)[:200])


def diagnose_false_duty(record: dict, rule_kind: str = '') -> Diagnosis:
    """The layer of a duty the model reading kept where there is none (a permission, an authority's task, a statement)."""
    lead = record.get('lead') or ''
    text = record.get('input', '')
    quotes = [c.get('source_quote', '') for c in record.get('candidates') or []]
    if lead and lead in text and any(q == text for q in quotes) and len(record.get('refs', [])) == 1 and record.get('mode') != 'LIST':
        return Diagnosis(layer='VALIDATOR_PIPELINE', code='LEAD_IN_READ_AS_THE_CLAUSE', stage='unit',
                         detail="the unit joined the lead-in of the fıkra to the clause, and the duty kept is the lead-in's")
    return Diagnosis(layer='MODEL', code='NON_DUTY_READ_AS_DUTY', stage='extraction',
                     detail=f'the model returned a duty for a clause that is {rule_kind or "no duty"}; the gate found a duty marker and kept it')


def compare_readings(frames: list[Frame], records: list[dict]) -> list[ClauseReading]:
    """One ClauseReading per frame: what each reader says, whether they agree, and where they do not, why."""
    by_ref = {}
    for record in records:
        for ref in record.get('refs') or [record.get('ref')]:
            by_ref[ref] = record
    out = []
    for frame in frames:
        rule = rule_duty(frame)
        record = by_ref.get(frame.ref)
        if record is None:
            out.append(ClauseReading(ref=frame.ref, rule_kind=frame.kind, rule=rule, agreement='NOT_READ', verdict=rule, support='NOT_READ',
                                     review_required=rule != 'NONE'))
            continue
        model = model_duty(record)
        listed = len(record.get('refs') or []) > 1
        common = dict(ref=frame.ref, unit_id=record.get('unit_id') or record.get('ref'), rule_kind=frame.kind, rule=rule, model=model,
                      model_reason=record.get('reason', ''))
        if rule == 'NONE' and model == 'NONE':
            out.append(ClauseReading(**common, agreement='BOTH_NONE', verdict='NONE', support='CONFIRMED'))
        elif rule != 'NONE' and model != 'NONE':
            same = model == 'BOTH' or model == rule or listed
            out.append(ClauseReading(**common, agreement='BOTH_DUTY' if same else 'POLARITY_DIFFERS', verdict=rule,
                                     support='CONFIRMED' if same else 'CONTESTED', review_required=not same,
                                     diagnosis=None if same else Diagnosis(layer='MODEL', code='POLARITY_DIFFERS', stage='extraction',
                                                                           detail=f'rule reader {rule}, model {model}')))
        elif rule != 'NONE':
            diagnosis = diagnose_reading(record)
            # A model answer the pipeline withheld is no evidence against the clause; a partial answer stands behind part
            # of it; only a model that denied the duty (or whose second reading withdrew it) contests it.
            partial = diagnosis.code in ('DUTY_LEFT_OUT', 'QUALIFIER_LEFT_OUT', 'DUTY_REPEATED', 'ACTION_MISALIGNED')
            contested = diagnosis.code in ('DUTY_DENIED', 'REVIEW_WITHDREW_DUTY')
            out.append(ClauseReading(**common, agreement='RULE_ONLY', verdict=rule, diagnosis=diagnosis,
                                     support='PARTLY_CONFIRMED' if partial else 'CONTESTED' if contested else 'RULE_ONLY',
                                     review_required=contested))
        elif listed:
            # A list read as one unit: the duty the model kept belongs to the items that are duties, not to this one.
            out.append(ClauseReading(**common, agreement='BOTH_NONE', verdict='NONE', support='CONFIRMED'))
        else:
            out.append(ClauseReading(**common, agreement='MODEL_ONLY', verdict='NONE', support='MODEL_ONLY', review_required=True,
                                     diagnosis=diagnose_false_duty(record, frame.kind)))
    return out


def reading_summary(readings: list[ClauseReading]) -> dict:
    layers = Counter(r.diagnosis.layer for r in readings if r.diagnosis)
    codes = Counter(f'{r.diagnosis.layer}/{r.diagnosis.code}' for r in readings if r.diagnosis)
    read = [r for r in readings if r.agreement != 'NOT_READ']
    agreed = sum(1 for r in read if r.agreement in ('BOTH_DUTY', 'BOTH_NONE'))
    return {'clauses': len(readings), 'read': len(read), 'agreement': dict(Counter(r.agreement for r in readings)),
            'agreement_rate': round(agreed / len(read), 4) if read else None, 'support': dict(Counter(r.support for r in readings)),
            'review_required': sum(r.review_required for r in readings), 'by_layer': {k: layers.get(k, 0) for k in LAYERS},
            'by_code': dict(codes.most_common()), 'rules_version': AI_RULES_VERSION}


# -- engine packets --------------------------------------------------------------------------------------
def _squash(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def engine_payload(run: dict) -> dict:
    return run['packet']['events'][0]['payload']


def engine_rows(run: dict, obligations: list[ExtractedObligation]) -> dict[str, list[dict]]:
    """{provision ref: [engine rows]}: the rows of one recorded engine run joined to the extracted obligations.

    A unit run names its clauses (run['units']); a whole-article run is joined by quote: a row belongs to the clause
    whose text its source quote contains, or that contains it."""
    payload = engine_payload(run)
    units = run.get('units') or {}
    out: dict[str, list[dict]] = {}
    texts = {o.provision_ref: _squash(o.text) for o in obligations}
    for row in payload['obligations']:
        refs = list((units.get(row['source_label']) or {}).get('refs') or [])
        quote = _squash(row['candidate'].get('source_quote', ''))
        if len(refs) != 1:
            action = _squash(row['candidate'].get('required_action') or row['candidate'].get('prohibited_action') or '')
            pool = {ref: texts[ref] for ref in (refs or texts) if ref in texts}
            refs = [ref for ref, text in pool.items() if text and (text in quote or quote in text)] if not refs else \
                [ref for ref, text in pool.items() if action and action.rstrip('.') in text] or refs[:1]
        for ref in refs:
            out.setdefault(ref, []).append(row)
    return out


def engine_case(run: dict, ref: str) -> dict | None:
    """The extraction case of the unit (or article) a clause was read in."""
    payload = engine_payload(run)
    units = run.get('units') or {}
    label = next((label for label, unit in units.items() if ref in unit['refs']), None) or ref.split('/f.')[0]
    return next((c for c in payload['cases'] if c['source']['printed_label'] == label), None)


def engine_coverage(rows: list[dict]) -> str:
    """One coverage word for a clause from its engine rows (a clause with two duties has two rows): a conflict on any
    row decides, then the weakest row."""
    words = [COVERAGE_WORDS.get(r['proposal'].get('coverage'), 'UNKNOWN') for r in rows]
    for word in ('CONTRADICTED', 'UNKNOWN', 'NOT_COVERED', 'PARTIALLY_COVERED', 'COVERED'):
        if word in words:
            return word
    return 'UNKNOWN'


def diagnose_engine_coverage(expected: list[str], rows: list[dict], case: dict | None, decisive: list[str] = ()) -> Diagnosis:
    """The layer of an engine coverage result that is not among the expected words. `decisive` are the policy statements
    (texts) the label rests on, when known: whether they were among the passages judged tells retrieval from judgement."""
    if not rows:
        if case is None:
            return Diagnosis(layer='RETRIEVAL', code='PROVISION_NOT_ANALYSED', stage='selection', detail='the clause was not among the targets of the run')
        record = {'status': case['output']['status'], 'reason': case['reason'], 'diagnostics': case.get('diagnostics') or [],
                  'input': case['source']['text'], 'classification': (case.get('classification') or [{}])[0]}
        found = diagnose_reading(record)
        if len(case.get('classification') or []) == 1 and len(case['source']['text']) > 1500 and found.layer != 'RETRIEVAL':
            return Diagnosis(layer='VALIDATOR_PIPELINE', code='UNIT_TOO_LARGE', stage='unit',
                             detail=f"the provision went to the model as one unit of {len(case['source']['text'])} characters "
                                    f'({found.code} at the gate)')
        return found.model_copy(update={'detail': 'no obligation row: ' + found.detail})
    row = rows[0]
    proposal = row['proposal']
    got = COVERAGE_WORDS.get(proposal.get('coverage'), 'UNKNOWN')
    checks = proposal.get('policy_checks') or []
    judged = {_squash(c.get('quote') or '') for c in checks}
    if decisive and not any(_squash(d) in q or q in _squash(d) for d in decisive for q in judged if q):
        return Diagnosis(layer='RETRIEVAL', code='DECISIVE_PASSAGE_NOT_JUDGED', stage='retrieval',
                         detail='the policy statement the label rests on was not among the passages the judge read')
    if got == 'UNKNOWN':
        return Diagnosis(layer='VALIDATOR_PIPELINE', code='COVERAGE_WITHHELD', stage='coverage',
                         detail=str(proposal.get('coverage_reason', ''))[:200])
    if got == 'CONTRADICTED':
        conflicts = [c for c in checks if c.get('relation') == 'CONFLICTS']
        if any(not c.get('quote') for c in conflicts):
            return Diagnosis(layer='GROUNDING', code='CONFLICT_WITHOUT_QUOTE', stage='verifier', detail='a conflict was kept without the contradicting sentence')
        return Diagnosis(layer='MODEL', code='FALSE_CONFLICT', stage='verifier',
                         detail='the verifier read a permission about another act or place as a contradiction of this duty: '
                                + '; '.join(_squash(c.get('quote') or '')[:90] for c in conflicts[:2]))
    if got == 'COVERED':
        return Diagnosis(layer='MODEL', code='SUPPORT_OVERSTATED', stage='support',
                         detail='a passage was accepted as stating the whole duty although it omits or contradicts an element')
    if got == 'PARTIALLY_COVERED':
        return Diagnosis(layer='MODEL', code='SUPPORT_UNDERSTATED' if 'COVERED' in expected else 'PARTIAL_INSTEAD_OF_GAP', stage='support',
                         detail='the judge found only part of the duty in the passages' if 'COVERED' in expected
                         else 'a passage about something else was read as covering part of the duty')
    return Diagnosis(layer='MODEL', code='SUPPORT_MISSED', stage='support', detail='the passage that states the duty was judged unrelated')


def diagnose_rule_coverage(expected: list[str], got: str, reasons: list[str]) -> Diagnosis:
    """The layer of a rule-comparer coverage result that is not among the expected words: always the comparison rules."""
    code = {'COVERED': 'RULE_SUPPORT_OVERSTATED', 'PARTIALLY_COVERED': 'RULE_PARTIAL', 'NOT_COVERED': 'RULE_MISSED_STATEMENT',
            'CONTRADICTED': 'RULE_FALSE_CONFLICT'}.get(got, 'RULE_UNCLEAR')
    return Diagnosis(layer='VALIDATOR_PIPELINE', code=code, stage='comparison', detail='comparison rules: ' + ', '.join(reasons[:4]))


def diagnose_applicability(expected: str, decision, obligation: ExtractedObligation | None) -> Diagnosis:
    """The layer of a routed applicability result that differs from the expected one."""
    if obligation is None:
        return Diagnosis(layer='VALIDATOR_PIPELINE', code='NO_OBLIGATION_EXTRACTED', stage='frames', detail='the clause yielded no obligation to route')
    if decision is None:
        return Diagnosis(layer='SCOPE', code='TARGET_NOT_AT_SCOPE_LEVEL', stage='scope',
                         detail=f'the scope was derived at level {obligation.scope.level} ({obligation.basis}); the target is of another kind')
    codes = list(decision.reason_codes)
    if 'GROUNDING_FAILED' in codes:
        return Diagnosis(layer='GROUNDING', code='QUOTE_NOT_IN_STORED_TEXT', stage='validator', detail=str(decision.audit.validator)[:200])
    if 'REGULATORY_SCOPE_UNCLEAR' in codes:
        return Diagnosis(layer='SCOPE', code='ADDRESSEE_NOT_DERIVED', stage='scope', detail=f'flags {obligation.flags}')
    if decision.status == 'UNKNOWN' and expected != 'UNKNOWN':
        return Diagnosis(layer='SCOPE', code='PROFILE_FACT_NOT_STATED', stage='routing', detail=f'reason codes {codes}')
    element = 'EXCEPTION' if any('EXCEPTION' in c for c in codes) else 'CONDITION' if any('CONDITION' in c for c in codes) else \
        'PRODUCT' if any('PRODUCT' in c for c in codes) else 'ACTIVITY' if any('ACTIVITY' in c for c in codes) else \
        'ENTITY' if any('ENTITY' in c for c in codes) else 'FACILITY' if any('FACILITY' in c for c in codes) else 'SCOPE'
    return Diagnosis(layer='SCOPE', code=f'{element}_GATE_WRONG', stage='routing',
                     detail=f'expected {expected}, routed {decision.status} with {codes} ({obligation.basis}, flags {obligation.flags})')

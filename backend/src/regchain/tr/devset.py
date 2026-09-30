"""Scoring against BEVERAGE_TR_DEV_V2: task reports, contrast pairs and the layer of every wrong result.

The dataset holds a developer's labels (one reader, no adjudication): every report is INDICATIVE whatever its numbers,
and the DEV clauses were the ones the wording rules were corrected against - only HOLDOUT says something about text the
rules have not seen. Nothing here calls a model: a model reading or an engine run is scored from its recorded file.

Tasks
  OBLIGATION_EXTRACTION   OBLIGATION / PROHIBITION / NONE per clause, for the rule reader and for a recorded model reading
  ADDRESSEE_MATCH         how the duty is addressed (named actor, governing activity, product property, facility)
  EXCEPTION_DETECTION     whether the duty carries an exception
  APPLICABILITY           the routed status of an extracted obligation on a named target of a synthetic profile
  POLICY_COVERAGE         COVERED / PARTIALLY_COVERED / NOT_COVERED / CONTRADICTED / UNKNOWN on a named target, for the rule
                          comparer, for the rows of a recorded engine run, and for the selective pipeline (adjudicate.py)

The coverage cases have two splits as well: DEV (registers that repeat the regulation's wording; the comparison rules
were written on them) and HOLDOUT (a register in a company's own words, labelled before the paraphrase work began).

The selective pipeline is not measured by accuracy alone (score_pipeline): what the rules decide without a model, how
often a case is escalated, how right the escalated cases end, how often a contradiction is claimed that is none, how
often an answer is given where the label is UNKNOWN, and the model time per case.
"""
import json
from collections import Counter
from pathlib import Path

from ..evaluation.tasks import DEVELOPMENT, pair_consistency, task_report, unsupported_reasons
from . import ai
from .adjudicate import Adjudicator, ClauseAdjudicator, assess_obligation, clause_escalation, clause_verdict
from .compare import DATA as TR_DATA, compare_obligation, load_register
from .corpus import CorpusStore
from .extraction import extract_regulation, route
from .frames import article_label
from .packs import DATA, Registry

DATASET = DATA / 'evaluation' / 'beverage_tr_dev_v2.json'
DUTY_LABELS = ('OBLIGATION', 'PROHIBITION', 'NONE')
ADDRESSEE_LABELS = ('ACTOR', 'ACTIVITY', 'PRODUCT', 'FACILITY', 'UNCLEAR')
APPLICABILITY_LABELS = ('APPLIES', 'PARTIAL', 'DOES_NOT_APPLY', 'UNKNOWN')
COVERAGE_LABELS = ('COVERED', 'PARTIALLY_COVERED', 'NOT_COVERED', 'CONTRADICTED', 'UNKNOWN')
BASIS = {'ACTOR_EXPLICIT': 'ACTOR', 'GOVERNING_ACTIVITY': 'ACTIVITY', 'PRODUCT_PROPERTY': 'PRODUCT', 'FACILITY': 'FACILITY',
         'UNCLEAR': 'UNCLEAR'}
# A named party is an entity class in the labels; the rule reader states most parties by what they do.
ACTIVITY_OF_ACTOR = {'PRODUCER': 'PRODUCTION', 'IMPORTER': 'IMPORT', 'RETAILER': 'RETAIL_SALE', 'WHOLESALER': 'WHOLESALE',
                     'DISTRIBUTOR': 'DISTRIBUTION'}
ORDER = ('APPLIES', 'PARTIAL', 'UNKNOWN', 'DOES_NOT_APPLY')


def load(path: Path = DATASET) -> dict:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format') != 'cardaman-evaluation-dataset-v2' or data.get('label_status') != DEVELOPMENT:
        raise ValueError(f'{path} is not a development dataset of format cardaman-evaluation-dataset-v2')
    return data


class Readers:
    """The rule reader's frames and obligations for the regulations a dataset names, at the versions it was labelled on."""

    def __init__(self, dataset: dict, registry: Registry | None = None, store: CorpusStore | None = None):
        self.registry, self.store = registry or Registry.load(), store or CorpusStore()
        self.versions = dict(dataset['corpus_versions'])
        self.frames, self.obligations = {}, {}
        for regulation_id, version_id in self.versions.items():
            frames, obligations = extract_regulation(regulation_id, self.registry, self.store, version_id=version_id)
            self.frames.update({f.ref: f for f in frames})
            for obligation in obligations:
                self.obligations.setdefault(obligation.provision_ref, []).append(obligation)


def case_register(case: dict, registers: dict):
    """The register a coverage case is labelled on: the pilot's own, or the one under the data directory the case names
    (a validation case is written against a second set of documents for the same company)."""
    root = case.get('register_root', 'pilot_policies')
    key = (root, case['profile_id'])
    if key not in registers:
        registers[key] = load_register(case['profile_id'], TR_DATA / root)
    return registers[key]


def _report(task, rows, labels, dataset, **more):
    return task_report(task, rows, labels, target=dataset['target_per_task'], label_status=dataset['label_status'], **more)


def _wrong(task, reader, case, expected, predicted, diagnosis) -> dict:
    return {'task': task, 'reader': reader, 'case': case, 'expected': expected, 'predicted': predicted, 'layer': diagnosis.layer,
            'code': diagnosis.code, 'detail': diagnosis.detail}


def _duty_row(gold: str, predicted: str) -> tuple[str, str]:
    """(expected, predicted) in the three duty labels. A clause labelled BOTH is right under either duty label, and so is a
    reading that kept both where the label names one."""
    expected = 'NONE' if gold not in ('OBLIGATION', 'PROHIBITION', 'BOTH') else gold
    if expected == 'BOTH':
        expected = predicted if predicted in ('OBLIGATION', 'PROHIBITION') else 'OBLIGATION'
    if predicted == 'BOTH':
        predicted = expected if expected in ('OBLIGATION', 'PROHIBITION') else 'OBLIGATION'
    return expected, predicted


def score_clauses(dataset: dict, readers: Readers, records: list[dict] | None = None, split: str | None = None) -> dict:
    """The clause tasks on one split (None: every clause). `records` are recorded model readings (live.load_readings)."""
    clauses = [c for c in dataset['clauses'] if (split is None or c['split'] == split) and c['kind'] not in ('UNDECIDED', 'REFERENCE')]
    by_ref = {ref: record for record in records or [] for ref in record.get('refs') or [record.get('ref')]}
    comparison = {r.ref: r for r in ai.compare_readings([readers.frames[c['ref']] for c in clauses], records)} if records else {}
    rule_rows, model_rows, wrong = [], [], []
    trust = Counter()
    for clause in clauses:
        ref, frame = clause['ref'], readers.frames[clause['ref']]
        expected, predicted = _duty_row(clause['kind'], ai.rule_duty(frame))
        rule_rows.append((expected, predicted))
        if expected != predicted:
            wrong.append(_wrong('OBLIGATION_EXTRACTION', 'rule', ref, expected, predicted, ai.Diagnosis(
                layer='VALIDATOR_PIPELINE', code='WORDING_RULE', stage='frames',
                detail=f"frame kind {frame.kind} from marker '{frame.marker}'")))
        if ref in comparison:
            trust[(comparison[ref].support, expected == predicted)] += 1
        record = by_ref.get(ref)
        if record is None:
            continue
        listed = len(record.get('refs') or []) > 1
        # A list read as one unit yields the list's duty once: it is credited to the items that are duties.
        duty = ai.model_duty(record)
        model = 'NONE' if listed and clause['kind'] not in ('OBLIGATION', 'PROHIBITION', 'BOTH') else duty
        expected_m, predicted_m = _duty_row(clause['kind'], 'BOTH' if listed and model != 'NONE' else model)
        model_rows.append((expected_m, predicted_m))
        if expected_m != predicted_m:
            diagnosis = ai.diagnose_reading(record) if predicted_m == 'NONE' else \
                ai.diagnose_false_duty(record, clause['kind']) if expected_m == 'NONE' else \
                ai.Diagnosis(layer='MODEL', code='POLARITY_DIFFERS', stage='extraction', detail=f'model kept {duty}')
            wrong.append(_wrong('OBLIGATION_EXTRACTION', 'model', ref, expected_m, predicted_m, diagnosis))
    addressee_rows, exception_rows = [], []
    for clause in clauses:
        obligations = readers.obligations.get(clause['ref'], [])
        if 'addressee' in clause:
            bases = {BASIS[o.basis] for o in obligations} or {'NO_OBLIGATION'}
            allowed = set(clause['addressee'])
            predicted = next((b for b in clause['addressee'] if b in bases), sorted(bases)[0])
            core = set(clause.get('actors') or [])
            entities = {e for o in obligations for e in o.scope.entity_classes}
            activities = {a for o in obligations for a in o.scope.activity_classes}
            if predicted == 'ACTOR' and core and not all(c in entities or ACTIVITY_OF_ACTOR.get(c) in activities for c in core):
                predicted = 'ACTOR_INCOMPLETE'
            expected = predicted if predicted in allowed else clause['addressee'][0]
            addressee_rows.append((expected, predicted))
            if expected != predicted:
                flags = sorted({f for o in obligations for f in o.flags})
                wrong.append(_wrong('ADDRESSEE_MATCH', 'rule', clause['ref'], expected, predicted, ai.Diagnosis(
                    layer='VALIDATOR_PIPELINE' if predicted == 'NO_OBLIGATION' else 'SCOPE',
                    code='NO_OBLIGATION_EXTRACTED' if predicted == 'NO_OBLIGATION' else 'ADDRESSEE_NOT_DERIVED' if predicted == 'UNCLEAR'
                    else 'ACTOR_NOT_MAPPED' if predicted == 'ACTOR_INCOMPLETE' else 'ADDRESSEE_WRONG', stage='scope',
                    detail=f'expected {sorted(allowed)} {sorted(core)}; derived {sorted(bases)} activities {sorted(activities)[:6]} flags {flags}')))
        if 'exception' in clause and obligations:
            has = any(o.frame.exceptions or o.scope.exceptions or o.scope.excluded_product_classes for o in obligations)
            expected, predicted = ('YES' if clause['exception'] else 'NO'), ('YES' if has else 'NO')
            exception_rows.append((expected, predicted))
            if expected != predicted:
                wrong.append(_wrong('EXCEPTION_DETECTION', 'rule', clause['ref'], expected, predicted, ai.Diagnosis(
                    layer='SCOPE', code='EXCEPTION_MISSED' if expected == 'YES' else 'EXCEPTION_INVENTED', stage='frames',
                    detail='; '.join(e.quote[:70] for o in obligations for e in o.frame.exceptions)[:240])))
    out = {'split': split or 'ALL', 'clauses': len(clauses),
           'tasks': {'OBLIGATION_EXTRACTION/rule': _report('OBLIGATION_EXTRACTION', rule_rows, DUTY_LABELS, dataset, abstain=None),
                     'ADDRESSEE_MATCH/rule': _report('ADDRESSEE_MATCH', addressee_rows, ADDRESSEE_LABELS, dataset, abstain='UNCLEAR'),
                     'EXCEPTION_DETECTION/rule': _report('EXCEPTION_DETECTION', exception_rows, ('YES', 'NO'), dataset, abstain=None)},
           'wrong': wrong}
    if records:
        out['tasks']['OBLIGATION_EXTRACTION/model'] = _report('OBLIGATION_EXTRACTION', model_rows, DUTY_LABELS, dataset, abstain=None)
        # How far agreement between the readers tells a right clause from a wrong one: {support: [right, wrong]}.
        out['rule_reader_by_model_support'] = {support: [trust[(support, True)], trust[(support, False)]]
                                               for support in sorted({s for s, _ in trust})}
    return out


def _combined(statuses: list[str]) -> str | None:
    return next((s for s in ORDER if s in statuses), None)


def _decisions(readers: Readers, ref: str, profile, target_id: str):
    """(combined status, the decision that carries it, its obligation) for one clause on one target: the decision at the
    scope's own level, else the roll-up that names the target."""
    found = []
    for obligation in readers.obligations.get(ref, []):
        decisions, rollups = route(obligation, profile, readers.registry, readers.store)
        hits = [d for d in decisions if d.target_id == target_id] or [d for d in rollups if d.target_id == target_id]
        found += [(d, obligation) for d in hits]
    status = _combined([d.status for d, _ in found])
    chosen = next(((d, o) for d, o in found if d.status == status), (None, None))
    return status, chosen[0], chosen[1]


def score_applicability(dataset: dict, readers: Readers, profiles: dict) -> dict:
    rows, reasons, pairs, wrong, texts = [], [], [], [], {}
    for case in dataset['applicability']:
        observed = []
        for side in case['sides']:
            profile = profiles[side['profile_id']]
            status, decision, obligation = _decisions(readers, case['ref'], profile, side['target_id'])
            predicted = status or 'NO_DECISION'
            rows.append((side['expected'], predicted))
            observed.append((side['expected'], predicted))
            if decision is not None:
                key = f'{obligation.regulation_id}@{obligation.version_id}'
                if key not in texts:
                    texts[key] = '\n'.join(s['text'] for s in readers.store.sections(obligation.regulation_id, obligation.version_id))
                reasons.append(unsupported_reasons(
                    {'status': decision.status, 'stage': 'EXTRACTED' if decision.basis == 'EXTRACTED' else 'ROUTED',
                     'source': {'regulation_id': obligation.regulation_id, 'version_id': obligation.version_id,
                                'provision_ref': decision.provision_ref, 'quote': obligation.scope.quote}, 'facts': []}, texts))
            else:
                reasons.append([])
            if predicted != side['expected']:
                obligations = readers.obligations.get(case['ref'], [])
                wrong.append(_wrong('APPLICABILITY', 'rule', f"{case['case_id']} {case['ref']} @ {side['target_id']}", side['expected'],
                                    predicted, ai.diagnose_applicability(side['expected'], decision, obligation or (obligations[0] if obligations else None))))
        pairs += [(observed[i], observed[j]) for i in range(len(observed)) for j in range(i + 1, len(observed))]
    families, index = {}, 0
    for case in dataset['applicability']:
        tally = families.setdefault(case['family'], {'n': 0, 'correct': 0})
        for expected, predicted in rows[index:index + len(case['sides'])]:
            tally['n'] += 1
            tally['correct'] += expected == predicted
        index += len(case['sides'])
    return {'tasks': {'APPLICABILITY/rule': _report('APPLICABILITY', rows, APPLICABILITY_LABELS, dataset, positive='APPLIES', unsupported=reasons)},
            'pairs': pair_consistency(pairs), 'wrong': wrong, 'by_family': dict(sorted(families.items()))}


def _gap_rows(readers: Readers, ref: str, profile, register, target_id: str, second: dict | None = None):
    out = []
    for obligation in readers.obligations.get(ref, []):
        decisions, _ = route(obligation, profile, readers.registry, readers.store)
        for decision in decisions:
            if decision.status in ('APPLIES', 'PARTIAL') and (decision.target_id == target_id or target_id in decision.applies_to_products
                                                             or decision.entity_id == target_id):
                out.append(compare_obligation(obligation, decision, profile, register, readers.registry, second))
    return out


def _word(rows) -> str:
    words = [ai.COVERAGE_WORDS[row.document_coverage] for row in rows]
    return next((w for w in ('CONTRADICTED', 'UNKNOWN', 'NOT_COVERED', 'PARTIALLY_COVERED', 'COVERED') if w in words), 'NO_ROW')


def score_coverage(dataset: dict, readers: Readers, profiles: dict, engine_runs: list[dict] = (), split: str | None = None) -> dict:
    """Policy coverage of the rule comparer on every labelled case; of each recorded engine run on the cases whose clause
    that run analysed for the labelled profile; and, on those same cases, of the two readings joined
    (compare.combine_coverage) next to the rule comparer alone.

    An engine run reads the documents in force for one legal entity; a label names a target. The engine's word is scored
    on a case only when the labelled target is that entity or one of its products or activities and every document in
    force for the target was among the documents the run read, so that both readers looked at the same statements."""
    rule_rows, wrong, registers = [], [], {}
    engine_rows_by_run = [[] for _ in engine_runs]
    joined_rows, rule_on_engine_cases = [], []
    joined = []
    for run in engine_runs:
        obligations = [o for ref_obligations in readers.obligations.values() for o in ref_obligations if o.regulation_id == run['regulation_id']]
        articles = {f"{article_label(o.provision_ref)}" for o in obligations if o.frame.label.split(' md. ')[-1] in set(run['articles'])}
        profile = profiles[run['profile_id']]
        entity = profile.entity(run['entity_id'])
        own = {run['entity_id'], *entity.product_ids, *(a.activity_id for a in profile.activities if a.entity_id == run['entity_id'])}
        joined.append((ai.engine_rows(run, obligations), articles, own, ai.second_readings(run, obligations)))
    for case in (c for c in dataset['coverage'] if split is None or c.get('split', 'DEV') == split):
        profile = profiles[case['profile_id']]
        register = case_register(case, registers)
        gap = _gap_rows(readers, case['ref'], profile, register, case['target_id'])
        predicted = _word(gap)
        expected = predicted if predicted in case['expected'] else case['expected'][0]
        rule_rows.append((expected, predicted))
        if expected != predicted:
            diagnosis = ai.Diagnosis(layer='SCOPE', code='NOT_ROUTED_TO_TARGET', stage='routing',
                                     detail='the obligation does not apply on the labelled target, so nothing was compared') if not gap else \
                ai.diagnose_rule_coverage(case['expected'], predicted, [c for row in gap for c in row.coverage_reasons])
            wrong.append(_wrong('POLICY_COVERAGE', 'rule', f"{case['case_id']} {case['ref']} @ {case['target_id']}", expected, predicted, diagnosis))
        decisive = [r.quote for row in gap for r in row.readings if r.relation in ('SUPPORTS', 'PARTIAL', 'CONFLICTS')]
        for index, run in enumerate(engine_runs):
            rows_by_ref, articles, own, second = joined[index]
            if run['profile_id'] != case['profile_id'] or article_label(case['ref']) not in articles or case['target_id'] not in own:
                continue
            files = {Path(register.document(d).file).name for row in gap for d in row.documents_in_force}
            if not files <= set(run['policy_files']):
                continue
            rows = rows_by_ref.get(case['ref'], [])
            got = ai.engine_coverage(rows) if rows else 'NO_ROW'
            want = got if got in case['expected'] else case['expected'][0]
            engine_rows_by_run[index].append((want, got))
            if want != got:
                diagnosis = ai.diagnose_engine_coverage(case['expected'], rows, ai.engine_case(run, case['ref']), decisive)
                wrong.append(_wrong('POLICY_COVERAGE', f"engine:{run['entity_id']}:{run['regulation_id']}",
                                    f"{case['case_id']} {case['ref']}", want, got, diagnosis))
            both = _word(_gap_rows(readers, case['ref'], profile, register, case['target_id'], second.get(case['ref'])))
            joined_rows.append((both if both in case['expected'] else case['expected'][0], both))
            rule_on_engine_cases.append((expected, predicted))
            if both not in case['expected']:
                wrong.append(_wrong('POLICY_COVERAGE', 'joined', f"{case['case_id']} {case['ref']} @ {case['target_id']}",
                                    case['expected'][0], both, ai.Diagnosis(
                                        layer='VALIDATOR_PIPELINE', code='JOIN_RULE', stage='combine_coverage',
                                        detail=f'rule comparer {predicted}, engine {got}, joined {both}')))
    tasks = {'POLICY_COVERAGE/rule': _report('POLICY_COVERAGE', rule_rows, COVERAGE_LABELS, dataset)}
    for index, run in enumerate(engine_runs):
        tasks[f"POLICY_COVERAGE/engine:{run['entity_id']}:{run['regulation_id']}"] = _report('POLICY_COVERAGE', engine_rows_by_run[index],
                                                                                             COVERAGE_LABELS, dataset)
    if engine_runs:
        tasks['POLICY_COVERAGE/engine (all runs)'] = _report('POLICY_COVERAGE', [r for rows in engine_rows_by_run for r in rows],
                                                             COVERAGE_LABELS, dataset)
        tasks['POLICY_COVERAGE/rule (engine cases)'] = _report('POLICY_COVERAGE', rule_on_engine_cases, COVERAGE_LABELS, dataset)
        tasks['POLICY_COVERAGE/joined (engine cases)'] = _report('POLICY_COVERAGE', joined_rows, COVERAGE_LABELS, dataset)
    return {'tasks': tasks, 'wrong': wrong}


def score_clause_pipeline(dataset: dict, readers: Readers, records: list[dict] | None = None, adjudications: dict | None = None,
                          split: str | None = None, adjudicator: ClauseAdjudicator | None = None) -> dict:
    """Clause reading as a selective pipeline on one split: the rule reader decides the kind of every clause; a clause is
    escalated when the extraction model's recorded reading contests it or the duty is hard to read (several exceptions
    or parties, a cross-reference, no addressee); the strong model's answer never changes the kind, it marks the clause
    for review where it differs. The numbers say how many of the rule reader's errors that review would catch."""
    adjudicator = adjudicator or ClauseAdjudicator(recorded=adjudications or {})
    clauses = [c for c in dataset['clauses'] if (split is None or c['split'] == split) and c['kind'] not in ('UNDECIDED', 'REFERENCE')]
    support = {r.ref: r.support for r in ai.compare_readings([readers.frames[c['ref']] for c in clauses], records)} if records else {}
    cases = []
    for clause in clauses:
        frame = readers.frames[clause['ref']]
        rule = ai.rule_duty(frame)
        expected, predicted = _duty_row(clause['kind'], rule)
        reasons = clause_escalation(frame, readers.obligations.get(clause['ref'], []), support.get(clause['ref']))
        record = adjudicator(frame, reasons) if reasons else None
        verdict = clause_verdict('BOTH' if clause['kind'] == 'BOTH' and rule != 'NONE' else rule, record)
        model = verdict['model']
        cases.append({'ref': clause['ref'], 'expected': expected, 'rule': predicted, 'right': expected == predicted, 'reasons': reasons,
                      'support': support.get(clause['ref']), 'adjudicated': record is not None, 'model': model,
                      'model_right': None if record is None else _duty_row(clause['kind'], model or 'NONE')[0] == (_duty_row(clause['kind'], model or 'NONE')[1] if model else 'UNGROUNDED'),
                      'review': verdict['review'], 'model_ms': (record or {}).get('elapsed_ms', 0)})
    share = lambda part, whole: round(part / whole, 4) if whole else None
    n = len(cases)
    escalated = [c for c in cases if c['reasons']]
    adjudicated = [c for c in cases if c['adjudicated']]
    errors = [c for c in cases if not c['right']]
    flagged = [c for c in cases if c['review']]
    reasons = Counter(r for c in escalated for r in c['reasons'])
    metrics = {'n': n, 'deterministic_accuracy': share(sum(c['right'] for c in cases), n),
               'escalated': len(escalated), 'escalation_rate': share(len(escalated), n), 'adjudicated': len(adjudicated),
               'by_reason': dict(sorted(reasons.items())),
               'deterministic_accuracy_on_escalated': share(sum(c['right'] for c in escalated), len(escalated)),
               'model_accuracy_on_adjudicated': share(sum(bool(c['model_right']) for c in adjudicated), len(adjudicated)),
               'rule_errors': len(errors), 'rule_errors_escalated': sum(bool(c['reasons']) for c in errors),
               'rule_errors_flagged': sum(c['review'] for c in errors),
               'review_required': len(flagged), 'review_rate': share(len(flagged), n),
               'review_precision': share(sum(not c['right'] for c in flagged), len(flagged)),
               'wrong_not_flagged': sum(not c['right'] and not c['review'] for c in cases),
               'model_ms': sum(c['model_ms'] for c in cases), 'model_ms_per_clause': round(sum(c['model_ms'] for c in cases) / n, 1) if n else None}
    return {'split': split or 'ALL', 'metrics': metrics, 'cases': cases}


def pipeline_metrics(cases: list[dict]) -> dict:
    """The numbers the selective pipeline is judged by. cases: [{'expected': [words], 'rule', 'final', 'review',
    'proposal', 'escalated', 'adjudicated', 'model_ms', 'status'}].

    Three things are kept apart. The automatic decisions: cases no open point keeps from being decided (the rules'
    word, or a PARTIAL the adjudicator filled in); only these can be right or wrong as decisions. The cases sent to a
    person (REVIEW_REQUIRED): they are not answers and count as neither - `strict_accuracy` is the automatic right
    answers over every case. The model's proposals: what the adjudicator said on the cases it read, scored on their
    own, whatever became of them."""
    def share(part, whole):
        return round(part / whole, 4) if whole else None

    right = lambda case, key: case[key] in case['expected']
    # A word that claims more than any label allows ("covered" where a part is missing, "partial" where nothing is
    # stated): the error that closes a gap nobody looked at.
    rank = {'NOT_COVERED': 0, 'PARTIALLY_COVERED': 1, 'COVERED': 2}
    over = lambda case, key: case[key] in rank and all(e in rank for e in case['expected']) and rank[case[key]] > max(rank[e] for e in case['expected'])
    n = len(cases)
    auto = [c for c in cases if not c['review']]
    review = [c for c in cases if c['review']]
    escalated = [c for c in cases if c['escalated']]
    adjudicated = [c for c in cases if c['adjudicated']]
    proposed = [c for c in adjudicated if c['proposal'] not in (None, 'UNKNOWN')]
    rule_errors = [c for c in cases if not right(c, 'rule')]
    unknown = [c for c in cases if c['expected'] == ['UNKNOWN']]
    calls = [c for c in adjudicated if c['model_ms']]
    model_ms = sum(c['model_ms'] for c in cases)
    return {'n': n,
            'rules_only_correct': sum(right(c, 'rule') for c in cases), 'rules_only_accuracy': share(sum(right(c, 'rule') for c in cases), n),
            # automatic decisions
            'auto_decided': len(auto), 'auto_correct': sum(right(c, 'final') for c in auto),
            'auto_wrong': sum(not right(c, 'final') for c in auto), 'auto_accuracy': share(sum(right(c, 'final') for c in auto), len(auto)),
            'strict_accuracy': share(sum(right(c, 'final') for c in auto), n),
            'auto_false_covered': sum(c['final'] == 'COVERED' and 'COVERED' not in c['expected'] for c in auto),
            'auto_false_contradicted': sum(c['final'] == 'CONTRADICTED' and 'CONTRADICTED' not in c['expected'] for c in auto),
            'auto_overclaimed': sum(over(c, 'final') for c in auto),
            'auto_missed_contradictions': sum('CONTRADICTED' in c['expected'] and c['final'] != 'CONTRADICTED' for c in auto),
            # sent to a person
            'review_required': len(review), 'review_rate': share(len(review), n),
            'review_cases_the_rules_had_wrong': sum(not right(c, 'rule') for c in review),
            'review_cases_the_rules_had_right': sum(right(c, 'rule') for c in review),
            # the model's proposals, on their own
            'adjudicated': len(adjudicated), 'model_answers': len(proposed),
            'model_correct': sum(right(c, 'proposal') for c in proposed), 'model_accuracy': share(sum(right(c, 'proposal') for c in proposed), len(proposed)),
            'model_false_covered': sum(c['proposal'] == 'COVERED' and 'COVERED' not in c['expected'] for c in proposed),
            'model_false_contradicted': sum(c['proposal'] == 'CONTRADICTED' and 'CONTRADICTED' not in c['expected'] for c in proposed),
            'model_no_usable_answer': sum(not c['proposal'] or c['proposal'] == 'UNKNOWN' for c in adjudicated),
            'calls_failed': sum(any(s.startswith('FAILED') for s in c['status']) for c in adjudicated),
            'calls_without_thinking': sum(any(s.startswith('OK_WITHOUT_THINKING') for s in c['status']) for c in adjudicated),
            # what the escalation saw and what it let pass
            'escalated': len(escalated), 'escalation_rate': share(len(escalated), n),
            'rule_errors': len(rule_errors), 'rule_errors_escalated': sum(c['escalated'] for c in rule_errors),
            'rule_errors_not_escalated': sum(not c['escalated'] for c in rule_errors),
            'contradictions_labelled': sum('CONTRADICTED' in c['expected'] for c in cases),
            'unknown_labelled': len(unknown), 'unknown_overclaimed': sum(c['final'] != 'UNKNOWN' and not c['review'] for c in unknown),
            'model_ms': model_ms, 'model_ms_per_case': round(model_ms / n, 1) if n else None,
            'model_ms_per_adjudicated_case': round(model_ms / len(calls), 1) if calls else None,
            'model_ms_max': max((c['model_ms'] for c in calls), default=None)}


def score_pipeline(dataset: dict, readers: Readers, profiles: dict, table=None, adjudications: dict | None = None,
                   split: str | None = None, adjudicator: Adjudicator | None = None) -> dict:
    """The selective pipeline on the labelled coverage cases of one split: rules, candidates (from a recorded similarity
    table), escalation, and the adjudications of a recorded run (none: the escalated cases keep the rules' word).
    Nothing is asked of a model here unless a live `adjudicator` is handed in (a run: live.py); the model time is the
    time the record holds."""
    adjudicator = adjudicator or Adjudicator(recorded=adjudications or {})
    adjudications = adjudicator.recorded
    registers, cases, wrong, rows = {}, [], [], []
    for case in (c for c in dataset['coverage'] if split is None or c.get('split', 'DEV') == split):
        profile = profiles[case['profile_id']]
        register = case_register(case, registers)
        found = []
        for obligation in readers.obligations.get(case['ref'], []):
            decisions, _ = route(obligation, profile, readers.registry, readers.store)
            for decision in decisions:
                if decision.status in ('APPLIES', 'PARTIAL') and (decision.target_id == case['target_id'] or decision.entity_id == case['target_id']
                                                                 or case['target_id'] in decision.applies_to_products):
                    found.append(assess_obligation(obligation, decision, profile, register, readers.registry, table, adjudicator,
                                                   store=readers.store))
        final = _word([row for row, _ in found])
        rule_words = [ai.COVERAGE_WORDS[a.rule_coverage] for _, a in found]
        rule = next((w for w in ('CONTRADICTED', 'UNKNOWN', 'NOT_COVERED', 'PARTIALLY_COVERED', 'COVERED') if w in rule_words), 'NO_ROW')
        keys = {a.adjudication_key for _, a in found if a.adjudication_key}
        model_words = [ai.COVERAGE_WORDS[a.model_coverage] for _, a in found if a.model_coverage]
        proposal = next((w for w in ('CONTRADICTED', 'UNKNOWN', 'NOT_COVERED', 'PARTIALLY_COVERED', 'COVERED') if w in model_words), None)
        item = {'case_id': case['case_id'], 'ref': case['ref'], 'target_id': case['target_id'], 'expected': case['expected'], 'rule': rule,
                'final': final, 'escalated': any(a.escalation is not None for _, a in found), 'adjudicated': bool(keys),
                'reasons': sorted({r for _, a in found if a.escalation for r in a.escalation.reasons}),
                'basis': sorted({a.basis for _, a in found}), 'review': any(a.decision == 'REVIEW_REQUIRED' for _, a in found),
                'review_reasons': sorted({r for _, a in found for r in a.review_reasons}), 'proposal': proposal,
                'status': sorted({a.adjudication_status for _, a in found if a.adjudication_status}),
                'model': sorted({a.model_coverage for _, a in found if a.model_coverage}),
                'candidates': sorted({c.passage_id for _, a in found if a.escalation for c in a.escalation.candidates}),
                'model_ms': sum((adjudications or {}).get(k, {}).get('elapsed_ms', 0) for k in keys)}
        cases.append(item)
        expected = final if final in case['expected'] else case['expected'][0]
        rows.append((expected, final))
        if final != expected:
            wrong.append(_wrong('POLICY_COVERAGE', 'pipeline', f"{case['case_id']} {case['ref']} @ {case['target_id']}", expected, final,
                                _diagnose_pipeline(item, found, adjudications or {})))
    return {'split': split or 'ALL', 'metrics': pipeline_metrics(cases), 'cases': cases, 'wrong': wrong,
            'tasks': {'POLICY_COVERAGE/pipeline': _report('POLICY_COVERAGE', rows, COVERAGE_LABELS, dataset)}}


def _diagnose_pipeline(item: dict, found: list, adjudications: dict) -> ai.Diagnosis:
    """The layer of a wrong pipeline result."""
    detail = f"rules {item['rule']}, model {item['model'] or 'not asked'}, candidates {item['candidates']}, reasons {item['reasons']}"
    if not found:
        return ai.Diagnosis(layer='SCOPE', code='NOT_ROUTED_TO_TARGET', stage='routing', detail='the obligation does not apply on the labelled target')
    judgements = [j for _, a in found if a.adjudication_key for j in adjudications.get(a.adjudication_key, {}).get('judgements', [])]
    if item['final'] == 'NOT_COVERED' and not item['escalated']:
        return ai.Diagnosis(layer='RETRIEVAL', code='NO_CANDIDATE', stage='candidates',
                            detail='no statement passed a similarity floor, so nothing was read: ' + detail)
    if item['escalated'] and not item['adjudicated']:
        return ai.Diagnosis(layer='VALIDATOR_PIPELINE', code='ESCALATED_NOT_ADJUDICATED', stage='adjudication', detail=detail)
    if any(not j['grounded'] and j['relation'] not in (None, 'UNRELATED') for j in judgements):
        return ai.Diagnosis(layer='GROUNDING', code='QUOTE_NOT_IN_STATEMENT', stage='adjudication', detail=detail)
    if 'CONTRADICTED' in item['expected'] and 'CONFLICT' in item['model']:
        return ai.Diagnosis(layer='VALIDATOR_PIPELINE', code='CONFLICT_HELD_FOR_REVIEW', stage='decision',
                            detail='the model saw the conflict; by rule a conflict only the model sees changes no decision: ' + detail)
    if item['final'] != item['rule'] or item['model']:
        return ai.Diagnosis(layer='MODEL', code='ADJUDICATION_WRONG', stage='adjudication', detail=detail)
    return ai.Diagnosis(layer='VALIDATOR_PIPELINE', code='RULE_DECISION_WRONG', stage='comparison', detail=detail)


def evaluate(dataset: dict | None = None, registry: Registry | None = None, store: CorpusStore | None = None,
             readings: list[dict] | None = None, engine_runs: list[dict] = (), table=None, adjudications: dict | None = None) -> dict:
    """Every task of the dataset, the DEV and HOLDOUT splits apart, and every wrong result with its layer. `table` is a
    recorded similarity table and `adjudications` the records of an adjudication run: with them the selective pipeline
    is scored on the coverage cases next to the rule comparer."""
    from .profile import load_pilot_profiles
    dataset = dataset or load()
    readers = Readers(dataset, registry, store)
    profiles = {p.profile_id: p for p in load_pilot_profiles(readers.registry.vocabulary).values()}
    dev = score_clauses(dataset, readers, readings, 'DEV')
    holdout = score_clauses(dataset, readers, readings, 'HOLDOUT')
    applicability = score_applicability(dataset, readers, profiles)
    coverage = score_coverage(dataset, readers, profiles, engine_runs, 'DEV')
    coverage_holdout = score_coverage(dataset, readers, profiles, (), 'HOLDOUT')
    pipeline = {split: score_pipeline(dataset, readers, profiles, table, adjudications, split) for split in ('DEV', 'HOLDOUT')} \
        if table is not None or adjudications is not None else {}
    wrong = [dict(w, split='DEV') for w in dev['wrong']] + [dict(w, split='HOLDOUT') for w in holdout['wrong']] \
        + applicability['wrong'] + [dict(w, split='DEV') for w in coverage['wrong']] \
        + [dict(w, split='HOLDOUT') for w in coverage_holdout['wrong']] \
        + [dict(w, split=split) for split, result in pipeline.items() for w in result['wrong']]
    layers = Counter((w['reader'].split(':')[0], w['layer']) for w in wrong)
    return {'dataset_id': dataset['dataset_id'], 'version': dataset['version'], 'label_status': dataset['label_status'], 'claim': 'INDICATIVE',
            'clauses': {'DEV': {k: v for k, v in dev.items() if k != 'wrong'}, 'HOLDOUT': {k: v for k, v in holdout.items() if k != 'wrong'}},
            'applicability': {k: v for k, v in applicability.items() if k != 'wrong'},
            'coverage': {k: v for k, v in coverage.items() if k != 'wrong'},
            'coverage_holdout': {k: v for k, v in coverage_holdout.items() if k != 'wrong'},
            'pipeline': {split: {k: v for k, v in result.items() if k != 'wrong'} for split, result in pipeline.items()},
            'wrong': wrong,
            'wrong_by_layer': {reader: {layer: layers.get((reader, layer), 0) for layer in ai.LAYERS}
                               for reader in sorted({r for r, _ in layers})}}

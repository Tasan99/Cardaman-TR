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
                          comparer and for the rows of a recorded engine run
"""
import json
from collections import Counter
from pathlib import Path

from ..evaluation.tasks import DEVELOPMENT, pair_consistency, task_report, unsupported_reasons
from . import ai
from .compare import compare_obligation, load_register
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


def _gap_rows(readers: Readers, ref: str, profile, register, target_id: str):
    out = []
    for obligation in readers.obligations.get(ref, []):
        decisions, _ = route(obligation, profile, readers.registry, readers.store)
        for decision in decisions:
            if decision.status in ('APPLIES', 'PARTIAL') and (decision.target_id == target_id or target_id in decision.applies_to_products
                                                             or decision.entity_id == target_id):
                out.append(compare_obligation(obligation, decision, profile, register, readers.registry))
    return out


def score_coverage(dataset: dict, readers: Readers, profiles: dict, engine_runs: list[dict] = ()) -> dict:
    """Policy coverage of the rule comparer on every labelled case, and of each recorded engine run on the cases whose
    clause that run analysed for the labelled profile."""
    rule_rows, wrong, registers = [], [], {}
    engine_rows_by_run = [[] for _ in engine_runs]
    joined = []
    for run in engine_runs:
        obligations = [o for ref_obligations in readers.obligations.values() for o in ref_obligations if o.regulation_id == run['regulation_id']]
        articles = {f"{article_label(o.provision_ref)}" for o in obligations if o.frame.label.split(' md. ')[-1] in set(run['articles'])}
        joined.append((ai.engine_rows(run, obligations), articles))
    for case in dataset['coverage']:
        profile = profiles[case['profile_id']]
        register = registers.setdefault(profile.profile_id, load_register(profile.profile_id))
        gap = _gap_rows(readers, case['ref'], profile, register, case['target_id'])
        words = [ai.COVERAGE_WORDS[row.document_coverage] for row in gap]
        predicted = next((w for w in ('CONTRADICTED', 'UNKNOWN', 'NOT_COVERED', 'PARTIALLY_COVERED', 'COVERED') if w in words), 'NO_ROW')
        expected = predicted if predicted in case['expected'] else case['expected'][0]
        rule_rows.append((expected, predicted))
        if expected != predicted:
            diagnosis = ai.Diagnosis(layer='SCOPE', code='NOT_ROUTED_TO_TARGET', stage='routing',
                                     detail='the obligation does not apply on the labelled target, so nothing was compared') if not gap else \
                ai.diagnose_rule_coverage(case['expected'], predicted, [c for row in gap for c in row.coverage_reasons])
            wrong.append(_wrong('POLICY_COVERAGE', 'rule', f"{case['case_id']} {case['ref']} @ {case['target_id']}", expected, predicted, diagnosis))
        decisive = [r.quote for row in gap for r in row.readings if r.relation in ('SUPPORTS', 'PARTIAL', 'CONFLICTS')]
        for index, run in enumerate(engine_runs):
            rows_by_ref, articles = joined[index]
            if run['profile_id'] != case['profile_id'] or article_label(case['ref']) not in articles:
                continue
            rows = rows_by_ref.get(case['ref'], [])
            got = ai.engine_coverage(rows) if rows else 'NO_ROW'
            want = got if got in case['expected'] else case['expected'][0]
            engine_rows_by_run[index].append((want, got))
            if want != got:
                diagnosis = ai.diagnose_engine_coverage(case['expected'], rows, ai.engine_case(run, case['ref']), decisive)
                wrong.append(_wrong('POLICY_COVERAGE', f"engine:{run['entity_id']}:{run['regulation_id']}",
                                    f"{case['case_id']} {case['ref']}", want, got, diagnosis))
    tasks = {'POLICY_COVERAGE/rule': _report('POLICY_COVERAGE', rule_rows, COVERAGE_LABELS, dataset)}
    for index, run in enumerate(engine_runs):
        tasks[f"POLICY_COVERAGE/engine:{run['entity_id']}:{run['regulation_id']}"] = _report('POLICY_COVERAGE', engine_rows_by_run[index],
                                                                                             COVERAGE_LABELS, dataset)
    return {'tasks': tasks, 'wrong': wrong}


def evaluate(dataset: dict | None = None, registry: Registry | None = None, store: CorpusStore | None = None,
             readings: list[dict] | None = None, engine_runs: list[dict] = ()) -> dict:
    """Every task of the dataset, the DEV and HOLDOUT splits apart for the clause tasks, and every wrong result with its layer."""
    from .profile import load_pilot_profiles
    dataset = dataset or load()
    readers = Readers(dataset, registry, store)
    profiles = {p.profile_id: p for p in load_pilot_profiles(readers.registry.vocabulary).values()}
    dev = score_clauses(dataset, readers, readings, 'DEV')
    holdout = score_clauses(dataset, readers, readings, 'HOLDOUT')
    applicability = score_applicability(dataset, readers, profiles)
    coverage = score_coverage(dataset, readers, profiles, engine_runs)
    wrong = [dict(w, split='DEV') for w in dev['wrong']] + [dict(w, split='HOLDOUT') for w in holdout['wrong']] \
        + applicability['wrong'] + coverage['wrong']
    layers = Counter((w['reader'].split(':')[0], w['layer']) for w in wrong)
    return {'dataset_id': dataset['dataset_id'], 'version': dataset['version'], 'label_status': dataset['label_status'], 'claim': 'INDICATIVE',
            'clauses': {'DEV': {k: v for k, v in dev.items() if k != 'wrong'}, 'HOLDOUT': {k: v for k, v in holdout.items() if k != 'wrong'}},
            'applicability': {k: v for k, v in applicability.items() if k != 'wrong'},
            'coverage': {k: v for k, v in coverage.items() if k != 'wrong'},
            'wrong': wrong,
            'wrong_by_layer': {reader: {layer: layers.get((reader, layer), 0) for layer in ai.LAYERS}
                               for reader in sorted({r for r, _ in layers})}}

"""The v0.18 evaluation additions: error taxonomy, label audit, reviewed labels, gate checks, frozen
baselines, the before/after table, and the new prediction fields, timings and snapshot fingerprints.

Like test_evaluation, these prove the harness does what it says with the data it is given; the
taxonomy rules are exercised on synthetic rows shaped like v0.17 packets (no gate chain in the
trace) and like v0.18 packets, so both generations of packets keep being read.
"""
import contextlib
import io
import json
import os
import re
import stat
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from regchain.evaluation import metrics as m
from regchain.evaluation.cli import main
from regchain.evaluation.gate import DEFAULT_THRESHOLDS, evaluate_gate, freeze, load_thresholds, verify_frozen
from regchain.evaluation.harness import configure_models, load_dataset, prediction_rows, rescore, run
from regchain.evaluation.labels import REVIEWED_FORMAT, apply_reviewed, audit, write_audit
from regchain.evaluation.report import BASELINE_METRICS, MANDATORY_ROWS, before_after_markdown
from regchain.evaluation.taxonomy import (DISPUTED_LABELS, TAXONOMY, applicability_error, classify_results, coverage_error,
                                          errors_markdown, write_errors)
from regchain.pilot.engine import analyze
from test_pilot import FixtureProvider, company, policies, sections

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / 'evaluation' / 'datasets' / 'tr-aml-v1.json'
REVIEWED = REPO / 'evaluation' / 'reviewed_labels' / 'tr-aml-v1.reviewed.json'
PIPELINE_KEYS = ('APPLICABILITY_CLEAR_MATCH', 'RELEVANCE_SCREEN')
PROFILE = {'id': 'p', 'name': 'P A.Ş.', 'jurisdictions': ['Türkiye'], 'activities': ['ödeme hizmetleri'], 'licences': ['6493 ödeme kuruluşu'],
           'products': ['cüzdan'], 'customer_types': ['bireysel müşteriler']}


def keep_env(test):
    """Restore the pipeline settings a test sets: the whole suite runs in one process."""
    saved = {key: os.environ.get(key) for key in PIPELINE_KEYS}

    def restore():
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    test.addCleanup(restore)


def quiet(argv):
    """The CLI's exit code, its printed tables and refusals kept out of the test output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(argv)


def writable(directory):
    for path in Path(directory).rglob('*'):
        if path.is_file():
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def expected(article='8', clause='(1)', applicability='DOES_NOT_APPLY', entity_gate='ANY', notes='', coverage='ANY', evidence=None, required=True):
    return {'article': article, 'clause': clause, 'action_keywords': ['kimlik tespitinde'], 'applicability': applicability,
            'entity_gate': entity_gate, 'coverage': coverage, 'conflict': None, 'evidence': evidence or [], 'required': required, 'notes': notes}


def item(exp, predicted=None, **extra):
    """A scored pair as harness.score_case writes it; predicted None = not extracted."""
    if predicted is None:
        return {'expected': exp, 'prediction': None, 'matched': False}
    return {'expected': exp, 'prediction': 'K', 'matched': True, 'applicability': (exp['applicability'], predicted), **extra}


def obligation(rule='MODEL', match='NOT_RESTRICTED', entities=(), trace=None, codes=(), basis=(), subject='Yükümlüler', quote='(1) ...'):
    """A packet obligation shaped like v0.17 (trace without the v0.18 keys unless given)."""
    return {'id': 'o1', 'source_label': 'Yönetmelik 200713012 md. 8', 'candidate': {'subject': subject, 'source_quote': quote},
            'diagnostics': [{'code': code} for code in codes],
            'proposal': {'applicability_rule': rule, 'applicability_reason': 'r', 'basis': list(basis), 'scope_evidence': [],
                         'applicability_scope': {'child_clause': '(1)', 'child_excerpt': quote, 'match': match, 'rule': 'PROVISION_LEVEL',
                                                 'required_entities': list(entities), 'provision_state': 'UNKNOWN', 'final': 'UNKNOWN'},
                         'trace': {'model_decision': None, 'rule_decision': 'NO_RESTRICTION', **(trace or {})}, 'review_flags': []}}


def prediction():
    return {'key': 'K', 'obligation_id': 'o1', 'coverage': 'NO_EVIDENCE', 'conflict': False, 'policy_checks': []}


class TaxonomyTests(unittest.TestCase):
    def classify(self, case_id, exp, predicted, ob=None, company=PROFILE):
        return applicability_error(case_id, item(exp, predicted), prediction(), ob or obligation(), company)

    def test_global_scope_from_the_label_note_and_from_the_v018_subject_gate(self):
        note = 'Yönetmelik md. 4 yükümlü listesinde yer almayan kuruluş; genel ödevler bağlamaz.'
        record = self.classify('C04', expected('5', '(2)', notes=note), 'UNKNOWN', obligation(codes=['SCOPE_QUOTE_DROPPED']))
        self.assertEqual(record['taxonomy'], 'GLOBAL_SCOPE_MISMATCH')
        self.assertIn('MODEL_HALLUCINATION', record['also'])            # the dropped quote is listed, not lost
        v018 = obligation(trace={'subject_gate': {'gate': 'REGULATION_SUBJECT_SCOPE', 'status': 'MISMATCH', 'clear': True, 'reason': 'not listed',
                                                  'evidence': {'source_label': 'Yönetmelik 200713012 md. 4'}}})
        record = self.classify('C07', expected('28', '(2)'), 'APPLIES', v018)
        self.assertEqual((record['taxonomy'], record['expected'], record['actual']), ('GLOBAL_SCOPE_MISMATCH', 'DOES_NOT_APPLY', 'APPLIES'))
        self.assertEqual(record['regulation_basis']['subject_gate']['list_source_label'], 'Yönetmelik 200713012 md. 4')

    def test_customer_and_subject_entity_mismatch(self):
        counterparty = [{'role': 'counterparty', 'type': 'ASSOCIATION', 'text': 'dernek', 'source': 'clause', 'match': 'MATCH'}]
        record = self.classify('C29', expected(entity_gate='MISMATCH', notes='Müşteri aileleri bireysel.'), 'APPLIES',
                               obligation(match='MATCH', entities=counterparty))
        self.assertEqual(record['taxonomy'], 'CUSTOMER_ENTITY_MISMATCH')
        self.assertIn('counterparty ASSOCIATION', record['rule_decision']['required_entities'][0])
        customs = self.classify('C28', expected('42', '(1)', notes='Yolcu tutanağı gümrük idaresinin işlemidir.'), 'UNKNOWN',
                                obligation(subject='Kendisinden açıklama talep edilen yolcu'))
        self.assertEqual(customs['taxonomy'], 'SUBJECT_ENTITY_MISMATCH')
        # A gate that wrongly rules a duty out is classified by the gate that did it (v0.18 decided_by).
        wrong_exclusion = self.classify('C22', expected('4', '(2)', applicability='APPLIES'), 'DOES_NOT_APPLY',
                                        obligation(rule='ENTITY_GATE', trace={'decided_by': 'SUBJECT_SCOPE_GATE'}))
        self.assertEqual(wrong_exclusion['taxonomy'], 'GLOBAL_SCOPE_MISMATCH')

    def test_rule_model_disagreement_profile_ambiguity_and_label_ambiguity(self):
        record = self.classify('C03', expected(applicability='APPLIES', entity_gate='MATCH'), 'UNKNOWN',
                               obligation(match='MATCH', trace={'model_decision': 'POSSIBLY_APPLIES'}))
        self.assertEqual(record['taxonomy'], 'RULE_MODEL_DISAGREEMENT')
        self.assertEqual(record['model_decision']['state'], 'POSSIBLY_APPLIES')
        downgraded = self.classify('C03', expected(applicability='APPLIES'), 'UNKNOWN', obligation(rule='DOWNGRADED_BASIS_INCONSISTENT'))
        self.assertEqual(downgraded['taxonomy'], 'RULE_MODEL_DISAGREEMENT')
        vague = {**PROFILE, 'licences': None, 'customer_types': None}
        record = self.classify('C10', expected('5', '(2)', applicability='UNKNOWN'), 'APPLIES', company=vague)
        self.assertEqual(record['taxonomy'], 'PROFILE_TOO_AMBIGUOUS')
        self.assertIn('licences', record['rationale'])
        undetermined = self.classify('C25', expected(applicability='UNKNOWN', entity_gate='UNDETERMINED'), 'DOES_NOT_APPLY',
                                     obligation(match='UNDETERMINED'))
        self.assertEqual(undetermined['taxonomy'], 'PROFILE_TOO_AMBIGUOUS')
        disputed = self.classify('C09', expected('6', '(1)', entity_gate='MISMATCH'), 'APPLIES')
        self.assertEqual(disputed['taxonomy'], 'LABEL_AMBIGUITY')
        self.assertIn('md. 14(1)(b)', disputed['rationale'])
        self.assertEqual({key[0] for key in DISPUTED_LABELS}, {'C09', 'C01', 'C16', 'C18', 'C12', 'C19'})

    def test_exemption_missed_parent_child_leakage_and_the_addressee_gate(self):
        # Exclusion wording in the clause itself (v0.17), or a v0.18 exemption gate that read POSSIBLE.
        worded = self.classify('C20', expected('5', '(2)'), 'APPLIES', obligation(quote='(2) Bu fıkra hükmü kamu kurumları için kapsam dışıdır.'))
        self.assertEqual(worded['taxonomy'], 'EXEMPTION_MISSED')
        self.assertIn('kapsam dışıdır', worded['rationale'])
        gated = self.classify('C20', expected('5', '(2)'), 'UNKNOWN',
                              obligation(trace={'gates': [{'gate': 'EXEMPTION', 'status': 'POSSIBLE', 'clear': False, 'reason': 'r'}]}))
        self.assertEqual((gated['taxonomy'], gated['rule_decision']['gates']), ('EXEMPTION_MISSED', {'EXEMPTION': 'POSSIBLE'}))
        # APPLIES shared from the provision while the clause names a counterparty the gate did not settle ...
        loose = [{'role': 'counterparty', 'type': 'FOUNDATION', 'text': 'vakıf', 'source': 'clause', 'match': 'UNDETERMINED'}]
        shared = self.classify('C29', expected(applicability='UNKNOWN'), 'APPLIES',
                               obligation(match='UNDETERMINED', entities=loose, codes=['APPLICABILITY_SHARED']))
        self.assertEqual(shared['taxonomy'], 'PARENT_CHILD_LEAKAGE')
        self.assertIn('PROFILE_TOO_AMBIGUOUS', shared['also'])
        # ... or the label rules the clause out by its entity and the packet read none, inheriting the provision's state.
        inherited = self.classify('C29', expected(entity_gate='MISMATCH'), 'APPLIES', obligation())
        self.assertEqual(inherited['taxonomy'], 'PARENT_CHILD_LEAKAGE')
        # A v0.18 addressee gate that wrongly ruled a duty out.
        addressee = self.classify('C22', expected('4', '(2)', applicability='APPLIES'), 'DOES_NOT_APPLY',
                                  obligation(rule='ADDRESSEE_GATE', trace={'decided_by': 'ADDRESSEE_GATE'}))
        self.assertEqual(addressee['taxonomy'], 'SUBJECT_ENTITY_MISMATCH')

    def test_unextracted_rows_coverage_errors_and_the_report(self):
        missing = applicability_error('C26', item(expected('27', '(3)', applicability='APPLIES')))
        self.assertEqual((missing['taxonomy'], missing['actual']), ('PARSER_EXTRACTION_ERROR', 'NOT_EXTRACTED'))
        exp = expected('46', '(1)', applicability='APPLIES', coverage='COVERS_TEXT', evidence=['sekiz yıl'])
        lost = coverage_error('C11', item(exp, 'APPLIES', coverage=('COVERS_TEXT', 'NO_EVIDENCE'),
                                          retrieval={'rank': None, 'hit': False, 'expected_passages': 1}), prediction(), obligation(), PROFILE)
        self.assertEqual(lost['taxonomy'], 'RETRIEVAL_FAILURE')
        claimed = coverage_error('C15', item({**exp, 'coverage': 'NO_EVIDENCE', 'evidence': []}, 'APPLIES', coverage=('NO_EVIDENCE', 'PARTIAL')),
                                 prediction(), obligation(), PROFILE)
        self.assertEqual(claimed['taxonomy'], 'MODEL_HALLUCINATION')
        # A coverage error on a row whose applicability is wrong too takes that error's taxonomy.
        note = 'yükümlü listesinde yer almayan'
        results = [{'case_id': 'C04', 'error': None, 'extras': [], 'predictions': [{**prediction(), 'key': 'K'}],
                    'scored': [item(expected('46', '(1)', notes=note, coverage='NOT_ASSESSED'), 'APPLIES', coverage=('NOT_ASSESSED', 'COVERS_TEXT')),
                               item(expected('5', '(1)', applicability='APPLIES', required=False)),        # optional, not an error
                               item(expected('5', '(2)', applicability='APPLIES'))]},
                   {'case_id': 'C99', 'error': 'ProviderFailure: down', 'scored': [], 'predictions': []}]
        report = classify_results(results, {'C04': {'company': PROFILE, 'obligations': [obligation()]}})
        self.assertEqual([r['taxonomy'] for r in report['applicability_errors']], ['GLOBAL_SCOPE_MISMATCH', 'PARSER_EXTRACTION_ERROR'])
        self.assertEqual(report['coverage_errors'][0]['taxonomy'], 'GLOBAL_SCOPE_MISMATCH')
        self.assertEqual(report['case_errors'], [{'case_id': 'C99', 'error': 'ProviderFailure: down'}])
        for field in ('case_id', 'obligation_id', 'obligation_key', 'expected', 'actual', 'company_profile', 'regulation_basis', 'child_clause',
                      'rule_decision', 'model_decision', 'evidence', 'taxonomy', 'rationale'):
            self.assertIn(field, report['applicability_errors'][0])
        self.assertEqual(report['applicability_errors'][0]['company_profile']['customer_types'], ['bireysel müşteriler'])
        text = errors_markdown({**report, 'run': 'r', 'cases': 2, 'packets': 1, 'scored_against': 'x'})
        self.assertIn('### GLOBAL_SCOPE_MISMATCH (1)', text)
        self.assertIn('| PARSER_EXTRACTION_ERROR | 1 | 0 |', text)
        self.assertTrue(set(report['counts']['applicability']) <= set(TAXONOMY))


class GateTests(unittest.TestCase):
    base = {'applicability': {'f1': 0.8, 'accuracy': 0.8, 'per_class': {'DOES_NOT_APPLY': {'f1': 0.7}}}, 'coverage': {'macro': {'f1': 0.7}},
            'extraction': {'recall': 0.9}, 'retrieval': {'recall_at_3': 0.9}, 'conflict': {'false_positive_rate': 0.0},
            'entity_gate': {'false_exclusion_rate': 0.0}, 'reliability': {'unknown_rate': 0.10, 'model_failure_rate': 0.0},
            'performance': {'wall_seconds': 100, 'llm_calls': 100, 'obligations': 10}}

    def status(self, run_metrics, thresholds=DEFAULT_THRESHOLDS, base=None, **kwargs):
        checks, overall = evaluate_gate(run_metrics, base or self.base, thresholds, **kwargs)
        return {c['check']: c['status'] for c in checks}, overall

    def changed(self, **paths):
        metrics = json.loads(json.dumps(self.base))
        for path, value in paths.items():
            target = metrics
            *parents, last = path.split('__')
            for key in parents:
                target = target[key]
            target[last] = value
        return metrics

    def test_accuracy_dna_f1_recall_and_coverage_drops(self):
        status, overall = self.status(self.changed(applicability__accuracy=0.7, applicability__per_class__DOES_NOT_APPLY__f1=0.6,
                                                   retrieval__recall_at_3=0.8))
        self.assertEqual((status['applicability accuracy'], status['applicability F1 (DOES_NOT_APPLY)'], status['retrieval R@3'], overall),
                         ('FAIL', 'FAIL', 'FAIL', 'FAIL'))
        status, overall = self.status(self.base)
        self.assertEqual(overall, 'PASS')
        status, _ = self.status(self.changed(applicability__accuracy=None))
        self.assertEqual(status['applicability accuracy'], 'SKIP')
        # 3 points of coverage macro F1 is the limit in evaluation/thresholds.json.
        configured = load_thresholds(REPO / 'evaluation' / 'thresholds.json')
        self.assertEqual(self.status(self.changed(coverage__macro__f1=0.66), configured)[0]['coverage macro F1'], 'FAIL')
        self.assertEqual(self.status(self.changed(coverage__macro__f1=0.68), configured)[0]['coverage macro F1'], 'PASS')

    def test_unknown_rate_warns_then_fails_and_llm_calls_warn(self):
        self.assertEqual(self.status(self.changed(reliability__unknown_rate=0.11))[0]['UNKNOWN rate'], 'PASS')
        self.assertEqual(self.status(self.changed(reliability__unknown_rate=0.13)), ({**self.status(self.base)[0], 'UNKNOWN rate': 'WARN'}, 'WARN'))
        self.assertEqual(self.status(self.changed(reliability__unknown_rate=0.25))[0]['UNKNOWN rate'], 'FAIL')
        # v0.17 metrics have no calls_per_obligation: it is llm_calls / obligations (10 -> 15 is +50%).
        self.assertEqual(self.status(self.changed(performance__llm_calls=150))[0]['LLM calls per obligation'], 'WARN')
        self.assertEqual(self.status(self.changed(performance__llm_calls=120))[0]['LLM calls per obligation'], 'PASS')
        self.assertEqual(self.status(self.changed(performance__llm_calls=None))[0]['LLM calls per obligation'], 'SKIP')
        free = self.changed(performance__llm_calls=0)
        self.assertEqual(self.status(self.changed(performance__llm_calls=5), base=free)[0]['LLM calls per obligation'], 'WARN')
        # A thresholds dict without the v0.18 keys (a v0.17 file used as is) still works: the defaults apply.
        old = {k: v for k, v in DEFAULT_THRESHOLDS.items() if k in ('applicability_f1_max_drop', 'coverage_macro_f1_max_drop', 'extraction_recall_max_drop',
                                                                   'conflict_fpr_max_increase', 'entity_gate_false_exclusion_max_increase',
                                                                   'unknown_rate_max_increase', 'model_failure_rate_max', 'runtime_warn_increase',
                                                                   'runtime_fail_increase', 'require_comparable_manifest')}
        self.assertEqual(self.status(self.changed(reliability__unknown_rate=0.13), old)[0]['UNKNOWN rate'], 'WARN')

    def test_parser_snapshot_compatibility(self):
        versions = {'fca_parser': 'fca-parser-v4', 'mevzuat_parser': 'mevzuat-parser-v3', 'policy_reader': 'pilot-policy-v3'}
        manifest = {'dataset': {'id': 'd'}, 'mode': {'provider': 'rules'}, 'versions': versions, 'snapshots': {'fx': 'a' * 64}}
        status, overall = self.status(self.base, run_manifest=manifest, baseline_manifest=manifest)
        self.assertEqual((status['regulation snapshot fingerprint'], status['parser versions'], overall), ('PASS', 'PASS', 'PASS'))
        moved = {**manifest, 'snapshots': {'fx': 'b' * 64}}
        self.assertEqual(self.status(self.base, run_manifest=moved, baseline_manifest=manifest)[0]['regulation snapshot fingerprint'], 'FAIL')
        old = {k: v for k, v in manifest.items() if k != 'snapshots'}
        self.assertEqual(self.status(self.base, run_manifest=manifest, baseline_manifest=old)[0]['regulation snapshot fingerprint'], 'SKIP')
        # A snapshot the run could not load proves nothing, even when the baseline failed on it the same way.
        broken = {**manifest, 'snapshots': {'fx': 'unreadable: ValueError'}}
        self.assertEqual(self.status(self.base, run_manifest=broken, baseline_manifest=broken)[0]['regulation snapshot fingerprint'], 'FAIL')
        reparsed = {**manifest, 'versions': {**versions, 'mevzuat_parser': 'mevzuat-parser-v4'}}
        self.assertEqual(self.status(self.base, run_manifest=reparsed, baseline_manifest=manifest)[0]['parser versions'], 'FAIL')
        allowed = {**DEFAULT_THRESHOLDS, 'allow_parser_change': True}
        self.assertEqual(self.status(self.base, allowed, run_manifest=reparsed, baseline_manifest=manifest)[0]['parser versions'], 'WARN')
        self.assertEqual(self.status(self.base, run_manifest={**manifest, 'versions': {}}, baseline_manifest=manifest)[0]['parser versions'], 'SKIP')

    def test_the_threshold_files_name_every_limit(self):
        for name in ('thresholds.json', 'thresholds-ci.json'):
            values = json.loads((REPO / 'evaluation' / name).read_text(encoding='utf-8'))
            self.assertEqual(set(DEFAULT_THRESHOLDS) - set(values), set(), name)
            self.assertEqual(values['coverage_macro_f1_max_drop'], 0.03)
        self.assertGreater(json.loads((REPO / 'evaluation' / 'thresholds-ci.json').read_text(encoding='utf-8'))['runtime_fail_increase'], 1)


class ReportTests(unittest.TestCase):
    def test_the_before_after_table_has_every_mandatory_row(self):
        before = {'applicability': {'accuracy': 0.5, 'f1': 0.8, 'per_class': {'APPLIES': {'precision': 0.9, 'recall': 0.7, 'f1': 0.8},
                                                                             'DOES_NOT_APPLY': {'precision': 0.6, 'recall': 0.5, 'f1': 0.55}}, 'n': 10},
                  'coverage': {'macro': {'f1': 0.6}}, 'retrieval': {'recall_at_3': 0.9, 'mrr': 0.8}, 'conflict': {'false_positive_rate': 0.05},
                  'reliability': {'unknown_rate': 0.2, 'failures': 3},
                  'performance': {'seconds_per_obligation': 100.0, 'llm_calls': 200, 'obligations': 10, 'model_seconds': 900.0}}
        after = json.loads(json.dumps(before))
        after['applicability']['accuracy'] = 0.6
        after['performance'].update(seconds_per_obligation=50.0, llm_calls=100)
        text = before_after_markdown(before, after)
        self.assertIn('| METRIC | BEFORE | AFTER | DELTA |', text)
        for label in ('Applicability accuracy', 'APPLIES F1', 'DOES_NOT_APPLY F1', 'UNKNOWN rate', 'Coverage macro F1', 'R@3', 'Conflict FPR',
                      'Runtime / obligation (s)', 'LLM calls / obligation', 'Model failures'):
            self.assertIn(label, MANDATORY_ROWS)
            self.assertRegex(text, rf'\n\| {re.escape(label)} \|')
        self.assertIn('| Applicability accuracy | 50.0% | 60.0% | +10.0 pp |', text)
        self.assertIn('| LLM calls / obligation | 20.0 | 10.0 | -10 (-50%) |', text)
        self.assertIn('| Model failures | 3 | 3 | +0 |', text)
        self.assertIn('| R@5 | — | — | — |', text)                   # undefined stays undefined, not zero


class RunTests(unittest.TestCase):
    def setUp(self):
        keep_env(self)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(writable, temp.name)
        self.root = Path(temp.name)

    def test_a_rules_run_carries_the_new_fields_timings_and_snapshot_fingerprint(self):
        run_dir, manifest, metrics = run(DATASET, self.root, provider='rules', retrieval='lexical', cases=['C01', 'C09', 'C28'], label='v18',
                                         overrides={'applicability_clear_match': 'rule', 'relevance_screen': 'off'})
        self.assertEqual(manifest['mode']['overrides'], {'APPLICABILITY_CLEAR_MATCH': 'rule', 'RELEVANCE_SCREEN': 'off'})
        self.assertEqual((manifest['environment']['APPLICABILITY_CLEAR_MATCH'], manifest['environment']['RELEVANCE_SCREEN']), ('rule', 'off'))
        (fixture, fingerprint), = manifest['snapshots'].items()
        self.assertTrue(fixture.endswith('tedbirler-200713012'))
        self.assertRegex(fingerprint, r'^[0-9a-f]{64}$')
        results = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
        for result in results:
            self.assertIsInstance(result['timings']['parse'], int)
            for row in result['predictions']:
                for key in ('decided_by', 'gate_status', 'review_flags', 'model_decision', 'subject_gate_status'):
                    self.assertIn(key, row)
                self.assertIsInstance(row['gate_status'], dict)
        perf, rel, app = metrics['performance'], metrics['reliability'], metrics['applicability']
        self.assertIn('parse', perf['stage_seconds'])
        self.assertNotIn('embedding_cache_hits', perf['stage_seconds'])
        self.assertEqual((perf['calls_per_obligation'], perf['live_calls'], perf['model_seconds_by_task']), (0.0, 0, {}))
        self.assertEqual((rel['retries'], rel['context_overflows'], rel['circuit_open']), (0, 0, 0))
        self.assertEqual(sum(app['decided_by'].values()), perf['obligations'])
        self.assertIsNotNone(app['unknown_rate_applicability'])
        self.assertIn('LLM calls / model seconds per obligation', (run_dir / 'report.md').read_text(encoding='utf-8'))
        # The same run gated against itself passes, the snapshot and parser checks included.
        checks, overall = evaluate_gate(metrics, metrics, DEFAULT_THRESHOLDS, manifest, manifest)
        self.assertEqual(overall, 'PASS')
        self.assertEqual({c['check']: c['status'] for c in checks}['regulation snapshot fingerprint'], 'PASS')
        # The error taxonomy reads the finished run from its results and packets.
        report, json_path, md_path = write_errors(run_dir)
        self.assertTrue(json_path.is_file() and md_path.is_file())
        self.assertEqual(sum(report['counts']['applicability'].values()), len(report['applicability_errors']))
        with self.assertRaises(ValueError):
            configure_models({'relevance_screen': 'sometimes'})

    def test_the_cli_run_flags_set_the_pipeline_settings(self):
        code = quiet(['run', '--dataset', str(DATASET), '--out', str(self.root), '--cases', 'C27', '--label', 'flags',
                     '--applicability-clear-match', 'quick', '--relevance-screen', 'on'])
        self.assertEqual(code, 0)
        manifest = json.loads(next(self.root.glob('*-flags-rules/manifest.json')).read_text(encoding='utf-8'))
        self.assertEqual(manifest['mode']['overrides'], {'APPLICABILITY_CLEAR_MATCH': 'quick', 'RELEVANCE_SCREEN': 'on'})

    def test_old_packets_without_the_v018_trace_keys_still_give_rows(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        payload = json.loads(json.dumps(packet['events'][0]['payload']))
        v017 = ('aggregate_decision', 'aggregate_rule', 'child_clause', 'model_decision', 'rule_decision', 'target_entity')
        for row in payload['obligations']:
            row['proposal']['trace'] = {k: v for k, v in (row['proposal'].get('trace') or {}).items() if k in v017}
            row['proposal'].pop('review_flags', None)
        (row,) = prediction_rows(payload, 'CONC:7')
        self.assertEqual((row['decided_by'], row['gate_status'], row['review_flags'], row['subject_gate_status']), (None, {}, [], None))


class FreezeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(writable, temp.name)          # read-only files would stop the cleanup on Windows
        self.root = Path(temp.name)
        self.run_dir, self.manifest, self.metrics = run(DATASET, self.root / 'runs', provider='rules', retrieval='lexical', cases=['C08'], label='f')

    def test_a_frozen_baseline_is_read_only_complete_and_its_tampering_fails_the_gate(self):
        out = self.root / 'baseline'
        frozen = freeze(self.run_dir, out)
        for name in ('manifest.json', 'metrics.json', 'report.md', 'results.json', 'baseline.json', 'baseline.md'):
            self.assertIn(name, frozen['files'])
            self.assertEqual(frozen['files'][name], sha256((out / name).read_bytes()).hexdigest())
            self.assertFalse(os.access(out / name, os.W_OK), name)
        baseline = json.loads((out / 'baseline.json').read_text(encoding='utf-8'))
        self.assertEqual(set(baseline['metrics']), {name for name, _ in BASELINE_METRICS})
        for name in ('applicability_accuracy', 'applies_f1', 'does_not_apply_f1', 'unknown_rate', 'coverage_macro_f1', 'conflict_fpr',
                     'entity_gate_false_inclusion_rate', 'recall_at_3', 'recall_at_10', 'mrr', 'evidence_hit_rate', 'proposal_success_rate',
                     'runtime_seconds', 'model_seconds', 'llm_calls', 'prompt_tokens', 'cache_hit_ratio', 'model_failures', 'retries'):
            self.assertIn(name, baseline['metrics'])
        self.assertIsNone(baseline['metrics']['applies_f1'])                       # rules mode: undefined, recorded as null
        self.assertEqual(verify_frozen(out), [])
        with self.assertRaises(FileExistsError):
            freeze(self.run_dir, out)
        self.assertEqual(quiet(['freeze', '--run', str(self.run_dir), '--out', str(out)]), 1)
        checks, overall = evaluate_gate(self.metrics, self.metrics, DEFAULT_THRESHOLDS, self.manifest, self.manifest, baseline_dir=out)
        self.assertEqual(({c['check']: c['status'] for c in checks}['baseline integrity'], overall), ('PASS', 'PASS'))
        os.chmod(out / 'metrics.json', stat.S_IREAD | stat.S_IWRITE)
        (out / 'metrics.json').write_text('{}', encoding='utf-8')
        self.assertEqual(verify_frozen(out), ['metrics.json changed'])
        checks, overall = evaluate_gate(self.metrics, self.metrics, DEFAULT_THRESHOLDS, self.manifest, self.manifest, baseline_dir=out)
        self.assertEqual(({c['check']: c['status'] for c in checks}['baseline integrity'], overall), ('FAIL', 'FAIL'))
        unfrozen = evaluate_gate(self.metrics, self.metrics, DEFAULT_THRESHOLDS, baseline_dir=self.run_dir)[0]
        self.assertEqual({c['check']: c['status'] for c in unfrozen}['baseline integrity'], 'SKIP')
        # An edited marker fails the gate instead of stopping it.
        os.chmod(out / 'FROZEN.json', stat.S_IREAD | stat.S_IWRITE)
        (out / 'FROZEN.json').write_text('{"files": ', encoding='utf-8')
        self.assertEqual(verify_frozen(out), ['FROZEN.json unreadable'])

    def test_the_diff_command_writes_the_table(self):
        target = self.root / 'diff.md'
        self.assertEqual(quiet(['diff', '--before', str(self.run_dir), '--after', str(self.run_dir), '--out', str(target)]), 0)
        self.assertIn('| Applicability accuracy |', target.read_text(encoding='utf-8'))


class LabelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.before = sha256(DATASET.read_bytes()).hexdigest()
        self.run_dir, _, _ = run(DATASET, self.root / 'runs', provider='rules', retrieval='lexical', cases=['C09', 'C12'], label='l')

    def tearDown(self):
        self.assertEqual(sha256(DATASET.read_bytes()).hexdigest(), self.before, 'the dataset file must never be written')

    def test_the_audit_lists_disputed_rows_and_unanimous_disagreements_and_writes_nothing_else(self):
        record, json_path, md_path = write_audit([self.run_dir], self.root / 'audit' / 'labels', DATASET)
        where = {item['where']: item for item in record['items']}
        self.assertIn('C09 md.6(1)', where)
        self.assertTrue(where['C09 md.6(1)']['reasons'][0].startswith('DISPUTED (applicability)'))
        self.assertTrue(where['C09 md.6(1)']['regulation_text'].startswith('(1)'))
        self.assertTrue(json_path.is_file() and md_path.is_file())
        self.assertFalse(any('ALL_RUNS_DISAGREE' in r for i in record['items'] for r in i['reasons']))   # one run is not evidence
        twice = audit([self.run_dir, self.run_dir], DATASET)
        agreeing = [i for i in twice['items'] if any('ALL_RUNS_DISAGREE' in r for r in i['reasons'])]
        self.assertTrue(agreeing)                     # rules mode leaves APPLIES-labelled rows UNKNOWN in both runs
        for entry in agreeing:
            output = entry['system_output'][self.run_dir.name]
            self.assertTrue(output['applicability'] != entry['expected']['applicability'] or output['coverage'] != entry['expected']['coverage']
                            or output['conflict'] != entry['expected']['conflict'])

    def test_only_accepted_reviewed_labels_are_overlaid_and_the_rescore_names_them(self):
        committed = json.loads(REVIEWED.read_text(encoding='utf-8'))
        self.assertEqual((committed['format'], committed['dataset_id'], committed['entries']), (REVIEWED_FORMAT, 'tr-aml-v1', []))
        self.assertRegex(committed['dataset_sha256'], r'^[0-9a-f]{64}$')
        base = {'case_id': 'C09', 'article': '6', 'clause': '(1)', 'field': 'applicability', 'reviewer': 'test', 'reviewed_at': '2026-09-24',
                'reason': 'md. 14(1)(b) sends the representative to md. 6'}
        reviewed = {**committed, 'entries': [{**base, 'from': 'DOES_NOT_APPLY', 'to': 'APPLIES', 'status': 'ACCEPTED'},
                                             {**base, 'clause': '(4)', 'article': '7', 'field': 'coverage', 'from': 'COVERS_TEXT', 'to': 'PARTIAL',
                                              'status': 'REJECTED'},
                                             {**base, 'article': '7', 'clause': '(1)', 'field': 'coverage', 'from': 'PARTIAL', 'to': 'NO_EVIDENCE',
                                              'status': 'ACCEPTED'}]}
        path = self.root / 'reviewed.json'
        path.write_text(json.dumps(reviewed), encoding='utf-8-sig')           # a byte-order mark, as Windows editors write one
        dataset = load_dataset(DATASET)
        overlaid, overlay = apply_reviewed(dataset, path, DATASET)
        self.assertEqual(len(overlay['applied']), 1)
        self.assertEqual(sorted(s['reason'].split(' ')[0] for s in overlay['skipped']), ['status', 'the'])     # REJECTED; stale "from"
        c09 = {c.case_id: c for c in overlaid.cases}['C09']
        self.assertEqual([e.applicability for e in c09.expected_obligations if e.article == '6'], ['APPLIES'])
        self.assertEqual(c09.expected_applicability['YONETMELIK:200713012/md.6/1'], 'APPLIES')
        self.assertEqual([e.applicability for e in {c.case_id: c for c in dataset.cases}['C09'].expected_obligations if e.article == '6'],
                         ['DOES_NOT_APPLY'])                                  # the loaded dataset object is not touched either
        _, _, metrics = rescore(self.run_dir, DATASET, reviewed_labels=path)
        note = json.loads((self.run_dir / 'rescore.json').read_text(encoding='utf-8'))
        self.assertEqual(len(note['reviewed_labels']['applied']), 1)
        self.assertEqual(note['reviewed_labels']['sha256'], sha256(path.read_bytes()).hexdigest())
        self.assertEqual(metrics['reviewed_labels']['path'], str(path))
        self.assertIn('Reviewed labels overlaid', (self.run_dir / 'report.md').read_text(encoding='utf-8'))
        rescored = json.loads((self.run_dir / 'results.rescored.json').read_text(encoding='utf-8'))
        labels = [i['expected']['applicability'] for r in rescored if r['case_id'] == 'C09' for i in r['scored'] if i['expected']['article'] == '6']
        self.assertEqual(labels, ['APPLIES'])

    def test_the_errors_label_audit_and_reviewed_labels_commands(self):
        self.assertEqual(quiet(['errors', '--run', str(self.run_dir), '--dataset', str(DATASET), '--out', str(self.root / 'errors')]), 0)
        report = json.loads((self.root / 'errors' / 'errors.json').read_text(encoding='utf-8'))
        self.assertEqual((report['scored_against'], set(report['sources'].values())), (str(DATASET), {'packets re-paired with the dataset'}))
        self.assertFalse((self.run_dir / 'errors.json').exists())                # --out, not the run directory
        stem = self.root / 'audit' / 'review'
        self.assertEqual(quiet(['label-audit', '--runs', str(self.run_dir), str(self.run_dir), '--out', str(stem)]), 0)
        self.assertTrue(stem.with_name('review.json').is_file() and stem.with_name('review.md').is_file())
        with self.assertRaises(SystemExit):                                       # an overlay only makes sense on a rescore
            quiet(['report', '--run', str(self.run_dir), '--reviewed-labels', str(REVIEWED)])
        self.assertEqual(quiet(['report', '--run', str(self.run_dir), '--rescore', '--dataset', str(DATASET), '--reviewed-labels', str(REVIEWED)]), 0)
        overlay = json.loads((self.run_dir / 'rescore.json').read_text(encoding='utf-8'))['reviewed_labels']
        self.assertEqual((overlay['applied'], overlay['skipped']), ([], []))         # the committed file changes nothing


class MetricTests(unittest.TestCase):
    def test_retries_overflows_open_circuits_and_stage_seconds(self):
        calls = [{'status': 'OK', 'retry_count': 2, 'task': 'judge.supports', 'elapsed_ms': 3000},
                 {'status': 'CONTEXT_BUDGET_EXCEEDED', 'retry_count': 0, 'task': 'judge.applicability', 'elapsed_ms': 0},
                 {'status': 'CIRCUIT_OPEN', 'task': 'judge.supports', 'elapsed_ms': 0},
                 {'status': 'OK', 'cache_hit': True, 'task': 'judge.supports', 'elapsed_ms': 0}]
        reliability = m.reliability([], calls)
        self.assertEqual((reliability['retries'], reliability['context_overflows'], reliability['circuit_open'], reliability['failures']), (2, 1, 1, 2))
        results = [{'case_id': 'a', 'wall_seconds': 4.0, 'predictions': [1, 2], 'calls': calls,
                    'timings': {'parse': 1500, 'coverage': 2500, 'embedding_cache_hits': 7}},
                   {'case_id': 'b', 'wall_seconds': 1.0, 'predictions': [], 'calls': [], 'timings': None}]
        perf = m.performance(results)
        self.assertEqual(perf['stage_seconds'], {'coverage': 2.5, 'parse': 1.5})
        self.assertEqual((perf['calls_per_obligation'], perf['model_seconds_per_obligation'], perf['live_calls']), (2.0, 1.5, 3))
        self.assertEqual(perf['model_seconds_by_task'], {'judge.supports': 3.0, 'judge.applicability': 0.0})
        decisions = m.decisions([{'decided_by': 'RULE_CLEAR_MATCH', 'review_flags': ['APPLIES_TRACE_INCOMPLETE']},
                                 {'applicability_rule': 'ENTITY_GATE'}, {}])
        self.assertEqual(decisions, {'decided_by': {'RULE_CLEAR_MATCH': 1, 'ENTITY_GATE': 1, 'NOT_RECORDED': 1},
                                     'review_flags': {'APPLIES_TRACE_INCOMPLETE': 1}})


if __name__ == '__main__':
    unittest.main()

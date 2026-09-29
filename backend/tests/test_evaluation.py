"""The evaluation harness (v0.17): schema, identifiers, metrics, manifest, scoring, report and gate.

These tests prove the harness scores what it is given; they say nothing about how accurate the
analysis is. The dataset itself is validated here too, so a label edit that breaks the schema
or the fixture layout fails before anyone runs a model.
"""
import json
import tempfile
import unittest
from pathlib import Path

from regchain.evaluation import metrics as m
from regchain.evaluation.gate import DEFAULT_THRESHOLDS, evaluate_gate, gate_markdown
from regchain.evaluation.harness import Components, aggregate, load_dataset, prediction_rows, run, run_case, score_case
from regchain.evaluation.identifiers import article_of, clause_id, fingerprint, match_predictions, obligation_key
from regchain.evaluation.manifest import build_manifest, comparable
from regchain.evaluation.report import compare_markdown, render_markdown
from regchain.evaluation.schema import Dataset, EvaluationCase, ExpectedObligation, FORMAT
from regchain.pilot.engine import analyze
from test_pilot import FixtureProvider, company, policies, sections

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / 'evaluation' / 'datasets' / 'tr-aml-v1.json'


def expectation(**fields):
    base = dict(article='7.3.4', clause='', applicability='APPLIES', coverage='COVERS_TEXT', conflict=False, evidence=['retain records'])
    base.update(fields)
    return ExpectedObligation(**base)


class IdentifierTests(unittest.TestCase):
    def test_keys_are_stable_and_readable(self):
        key = obligation_key('YONETMELIK:200713012', '8', '(1)', 'kimlik tespitinde; derneğin adı')
        self.assertTrue(key.startswith('YONETMELIK:200713012/md.8/1/'))
        self.assertEqual(key, obligation_key('YONETMELIK:200713012', '8', '1', 'Kimlik   tespitinde;  derneğin adı.'))   # whitespace and case do not matter
        self.assertEqual(obligation_key('CONC:7', '7.3.4'), 'CONC:7/md.7.3.4/*')
        self.assertEqual((clause_id('(2)'), clause_id('2'), clause_id('whole provision'), clause_id('')), ('2', '2', '', ''))
        self.assertEqual((article_of('Yönetmelik 200713012 md. 9/A'), article_of('CONC 7.3.4')), ('9/A', '7.3.4'))
        self.assertEqual(len(fingerprint('x')), 12)

    def test_matching_is_by_article_clause_and_keywords_and_claims_each_prediction_once(self):
        predictions = [{'article': '8', 'clause': '1', 'action': 'kimlik tespitinde; derneğin adı, amacı alınır'},
                       {'article': '8', 'clause': '3', 'action': 'teyide esas belgelerin fotokopisi alınır'},
                       {'article': '5', 'clause': '2', 'action': 'işlem yapılmadan önce tamamlanır'}]
        expected = [expectation(article='8', clause='(1)', action_keywords=['kimlik tespitinde']),
                    expectation(article='8', clause='(3)', action_keywords=['fotokopisi']),
                    expectation(article='8', clause='(5)', action_keywords=['kimlik tespiti']),
                    expectation(article='5', clause='', action_keywords=['tamamlanır'])]
        pairs, extras = match_predictions(expected, predictions)
        self.assertEqual([p['clause'] if p else None for _, p in pairs], ['1', '3', None, '2'])
        self.assertEqual(extras, [])
        # A keyword that does not occur leaves the prediction unclaimed, and it is reported as unexpected.
        pairs, extras = match_predictions([expectation(article='5', clause='(2)', action_keywords=['sekiz yıl'])], predictions[2:])
        self.assertIsNone(pairs[0][1])
        self.assertEqual(len(extras), 1)


class MetricTests(unittest.TestCase):
    def test_classification_metrics_are_computed_per_class_and_macro(self):
        rows = [('APPLIES', 'APPLIES'), ('APPLIES', 'UNKNOWN'), ('DOES_NOT_APPLY', 'DOES_NOT_APPLY'), ('DOES_NOT_APPLY', 'APPLIES'), ('UNKNOWN', 'UNKNOWN')]
        result = m.classification(rows, m.APPLICABILITY, positive='APPLIES')
        self.assertEqual(result['accuracy'], 0.6)
        self.assertEqual((result['precision'], result['recall'], result['f1']), (0.5, 0.5, 0.5))
        self.assertEqual(result['confusion']['APPLIES'], {'APPLIES': 1, 'DOES_NOT_APPLY': 0, 'UNKNOWN': 1, 'OTHER': 0})
        self.assertEqual(result['per_class']['DOES_NOT_APPLY']['support'], 2)
        self.assertAlmostEqual(result['macro']['f1'], round((0.5 + (2 / 3) + (2 / 3)) / 3, 4), places=3)
        empty = m.classification([], m.COVERAGE)
        self.assertIsNone(empty['accuracy'])

    def test_binary_gate_retrieval_proposal_and_reliability_metrics(self):
        conflict = m.binary([(True, True), (False, True), (True, False), (False, False), (False, False)])
        self.assertEqual((conflict['precision'], conflict['recall'], conflict['false_positive_rate']), (0.5, 0.5, round(1 / 3, 4)))
        gate = m.entity_gate([(True, True), (True, False), (False, True), (False, False), (False, False)])
        self.assertEqual((gate['false_exclusion_rate'], gate['false_inclusion_rate']), (round(1 / 3, 4), 0.5))
        retrieval = m.retrieval([1, 4, None, 2], [True, False, False, True])
        self.assertEqual((retrieval['recall_at_3'], retrieval['recall_at_5'], retrieval['recall_at_10']), (0.5, 0.75, 0.75))
        self.assertEqual(retrieval['mrr'], round((1 + 0.25 + 0 + 0.5) / 4, 4))
        self.assertEqual(retrieval['evidence_hit_rate'], 0.5)
        proposals = m.proposals([{'needed': True, 'produced': True, 'draft_attempted': True, 'draft_valid': False},
                                 {'needed': True, 'produced': False, 'draft_attempted': False, 'draft_valid': False},
                                 {'needed': False, 'produced': False, 'draft_attempted': False, 'draft_valid': False}])
        self.assertEqual((proposals['needed'], proposals['success_rate'], proposals['schema_validity_rate']), (2, 0.5, 0.0))
        predictions = [{'applicability': 'UNKNOWN', 'coverage': 'COVERS_TEXT', 'coverage_assessed': True, 'malformed_responses': 1},
                       {'applicability': 'APPLIES', 'coverage': 'UNKNOWN', 'coverage_assessed': True, 'malformed_responses': 0},
                       {'applicability': 'DOES_NOT_APPLY', 'coverage': 'NOT_ASSESSED', 'coverage_assessed': False, 'malformed_responses': 0}]
        calls = [{'status': 'OK'}, {'status': 'PROVIDER_FAILURE', 'error': 'Provider exceeded the configured time limit'}, {'status': 'OK'}, {'status': 'OK'}]
        reliability = m.reliability(predictions, calls)
        self.assertEqual((reliability['judgements'], reliability['unknown'], reliability['unknown_rate']), (5, 2, 0.4))
        self.assertEqual((reliability['model_failure_rate'], reliability['timeout_rate'], reliability['malformed_response_rate']), (0.25, 0.25, 0.25))

    def test_performance_and_extraction_metrics(self):
        results = [{'case_id': 'a', 'wall_seconds': 10.0, 'predictions': [1, 2], 'calls': [{'task': 'x', 'elapsed_ms': 1000, 'prompt_tokens': 10, 'output_tokens': 2, 'cache_hit': True},
                                                                                          {'task': 'y', 'elapsed_ms': 500, 'prompt_tokens': 5, 'output_tokens': 1}]},
                   {'case_id': 'b', 'wall_seconds': 30.0, 'predictions': [1], 'calls': []}]
        perf = m.performance(results)
        self.assertEqual((perf['wall_seconds'], perf['model_seconds'], perf['llm_calls'], perf['cache_hit_ratio'], perf['seconds_per_obligation']),
                         (40.0, 1.5, 2, 0.5, round(40 / 3, 4)))
        self.assertEqual(perf['slowest_cases'][0]['case_id'], 'b')
        pairs = [[(expectation(required=True), {'key': 'k1'}), (expectation(required=True), None), (expectation(required=False), None)]]
        extraction = m.extraction(pairs, [[{'key': 'extra'}]])
        self.assertEqual((extraction['recall'], extraction['precision'], extraction['unexpected']), (0.5, 0.5, 1))


class ScoringTests(unittest.TestCase):
    def fixture_case(self, **changes):
        fields = dict(case_id='F01', title='fixture', categories=['exact-policy'], jurisdiction='UK', regulation_id='CONC:7',
                      regulation_fixture='.', target_sections=['7.3'], company_profile=company().model_dump(),
                      policy_documents=['policy.txt'], expected_obligations=[expectation(article='7.3.4', action_keywords=['retain'])])
        fields.update(changes)
        return EvaluationCase(**fields)

    def test_predictions_and_scoring_from_a_fixture_packet(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        payload = packet['events'][0]['payload']
        rows = prediction_rows(payload, 'CONC:7')
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row['key'].startswith('CONC:7/md.7.3.4/*/'))
        self.assertEqual((row['applicability'], row['coverage'], row['conflict'], row['coverage_assessed']), ('APPLIES', 'COVERS_TEXT', False, True))
        self.assertEqual(row['ranks'], {'policy-1': 1})
        scored, extras = score_case(self.fixture_case(), payload, rows)
        item = scored[0]
        self.assertTrue(item['matched'])
        self.assertEqual((item['applicability'], item['coverage'], item['conflict']), (('APPLIES', 'APPLIES'), ('COVERS_TEXT', 'COVERS_TEXT'), (False, False)))
        self.assertEqual(item['retrieval'], {'rank': 1, 'hit': True, 'expected_passages': 1})
        self.assertEqual(item['proposal']['needed'], False)
        self.assertEqual(extras, [])
        # An expectation the run contradicts is named in the mismatches with its case and clause.
        wrong = self.fixture_case(expected_obligations=[expectation(article='7.3.4', applicability='DOES_NOT_APPLY', coverage='CONFLICT', conflict=True)])
        scored, _ = score_case(wrong, payload, rows)
        dataset = Dataset(format=FORMAT, dataset_id='t', version='1', cases=[wrong])
        metrics = aggregate(dataset, [{'case_id': 'F01', 'predictions': rows, 'scored': scored, 'extras': [], 'calls': [], 'wall_seconds': 1.0, 'error': None}])
        kinds = {item['kind'] for item in metrics['mismatches']}
        self.assertEqual(kinds, {'applicability_false_positive', 'coverage_mismatch', 'conflict_false_negative'})
        self.assertEqual(metrics['applicability']['accuracy'], 0.0)
        self.assertEqual(metrics['conflict']['fn'], 1)

    def test_a_missing_required_obligation_is_an_extraction_miss_not_a_crash(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        payload = packet['events'][0]['payload']
        rows = prediction_rows(payload, 'CONC:7')
        case = self.fixture_case(expected_obligations=[expectation(article='7.3.4', action_keywords=['destroy records'])])
        scored, extras = score_case(case, payload, rows)
        self.assertFalse(scored[0]['matched'])
        self.assertEqual(len(extras), 1)
        dataset = Dataset(format=FORMAT, dataset_id='t', version='1', cases=[case])
        metrics = aggregate(dataset, [{'case_id': 'F01', 'predictions': rows, 'scored': scored, 'extras': extras, 'calls': [], 'wall_seconds': 1.0, 'error': None}])
        self.assertEqual((metrics['extraction']['recall'], metrics['extraction']['unexpected']), (0.0, 1))
        self.assertEqual({i['kind'] for i in metrics['mismatches']}, {'extraction_miss', 'unexpected_candidate'})


class DatasetTests(unittest.TestCase):
    def test_the_golden_dataset_validates_and_its_fixtures_exist(self):
        dataset = load_dataset(DATASET)
        self.assertGreaterEqual(len(dataset.cases), 25)
        base = DATASET.parent
        for case in dataset.cases:
            self.assertTrue((base / case.regulation_fixture / 'snapshot.json').is_file(), case.case_id)
            for relative in [*case.policy_documents, *case.optional_control_records]:
                self.assertTrue((base / relative).is_file(), relative)
            self.assertTrue(case.expected_obligations or '3' in case.target_sections, case.case_id)
            # The keyed views agree with the per-obligation expectations they were generated from.
            for item in case.expected_obligations:
                key = f'{case.regulation_id}/md.{item.article}/{item.clause.strip("()") or "*"}'
                self.assertEqual(case.expected_applicability[key], item.applicability, key)
                self.assertEqual(case.expected_entity_gate[key], item.entity_gate, key)
        categories = {c for case in dataset.cases for c in case.categories}
        for wanted in ('parent-applies-child-not', 'entity-gate', 'false-conflict', 'irrelevant-evidence', 'unknown-over-trigger',
                       'false-positive-applicability', 'false-negative-applicability', 'semantic-equivalent-policy', 'exact-policy',
                       'partial-coverage', 'no-evidence', 'contradictory-policy', 'missing-policy', 'ambiguous-profile'):
            self.assertIn(wanted, categories)
        profiles = {case.company_profile['id'] for case in dataset.cases}
        self.assertGreaterEqual(len(profiles), 10)

    def test_the_schema_refuses_unknown_categories_and_duplicate_ids(self):
        with self.assertRaises(ValueError):
            EvaluationCase(case_id='X1', title='t', categories=['made-up'], jurisdiction='TR', regulation_id='YONETMELIK:1', regulation_fixture='.',
                           target_sections=['3'], company_profile={}, policy_documents=['p'], expected_obligations=[])
        case = EvaluationCase(case_id='X1', title='t', jurisdiction='TR', regulation_id='YONETMELIK:1', regulation_fixture='.',
                              target_sections=['3'], company_profile={}, policy_documents=['p'], expected_obligations=[])
        with self.assertRaises(ValueError):
            Dataset(format=FORMAT, dataset_id='d', version='1', cases=[case, case])


class RunTests(unittest.TestCase):
    def test_a_rules_run_over_the_golden_dataset_writes_manifest_metrics_report_and_gates_against_itself(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        run_dir, manifest, metrics = run(DATASET, Path(temp.name), provider='rules', retrieval='lexical', cases=['C01', 'C08', 'C27'], label='t')
        for name in ('manifest.json', 'results.json', 'metrics.json', 'report.md', 'ai-calls.jsonl'):
            self.assertTrue((run_dir / name).is_file(), name)
        self.assertEqual(manifest['mode']['provider'], 'rules')
        self.assertEqual(manifest['dataset']['id'], 'tr-aml-v1')
        self.assertEqual(len(manifest['manifest_sha256']), 64)
        self.assertIn('mevzuat_parser', manifest['versions'])
        self.assertEqual(metrics['errors'], [])
        self.assertEqual(metrics['performance']['llm_calls'], 0)
        # C01: the entity gate rules every md. 8 duty out without a model, and nothing is APPLIES without one;
        # C27: the definitions article yields nothing.
        results = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
        by_id = {r['case_id']: r for r in results}
        c01 = by_id['C01']['predictions']
        self.assertTrue(c01)
        self.assertTrue(all(p['applicability'] == 'DOES_NOT_APPLY' and p['applicability_rule'] == 'ENTITY_GATE' for p in c01 if p['article'] == '8'))
        self.assertFalse(any(p['applicability'] == 'APPLIES' for p in c01))
        self.assertEqual(by_id['C27']['predictions'], [])
        report = (run_dir / 'report.md').read_text(encoding='utf-8')
        self.assertIn('Rules mode', report)
        self.assertIn('Known false positives and false negatives', report)
        # The gate compares a run with a baseline; a run against itself passes every check.
        checks, overall = evaluate_gate(metrics, metrics, DEFAULT_THRESHOLDS, manifest, manifest)
        self.assertEqual(overall, 'PASS')
        self.assertIn('| wall clock |', gate_markdown(checks, overall, 'a', 'b'))
        same, differences = comparable(manifest, manifest)
        self.assertTrue(same)
        self.assertIn('Run comparison', compare_markdown([('a', manifest, metrics), ('b', manifest, metrics)]))

    def test_the_gate_fails_on_a_real_regression_and_warns_on_a_slow_run(self):
        base = {'applicability': {'f1': 0.8}, 'coverage': {'macro': {'f1': 0.7}}, 'extraction': {'recall': 0.9},
                'conflict': {'false_positive_rate': 0.0}, 'entity_gate': {'false_exclusion_rate': 0.0},
                'reliability': {'unknown_rate': 0.1, 'model_failure_rate': 0.0}, 'performance': {'wall_seconds': 100}}
        worse = json.loads(json.dumps(base))
        worse['applicability']['f1'] = 0.7
        worse['conflict']['false_positive_rate'] = 0.1
        worse['performance']['wall_seconds'] = 140
        checks, overall = evaluate_gate(worse, base, DEFAULT_THRESHOLDS)
        status = {c['check']: c['status'] for c in checks}
        self.assertEqual(overall, 'FAIL')
        self.assertEqual((status['applicability F1 (APPLIES)'], status['conflict false positive rate'], status['wall clock']), ('FAIL', 'FAIL', 'WARN'))
        same, _ = evaluate_gate(base, base, DEFAULT_THRESHOLDS)
        self.assertEqual(_, 'PASS')
        undefined = json.loads(json.dumps(base))
        undefined['applicability']['f1'] = None
        checks, overall = evaluate_gate(undefined, base, DEFAULT_THRESHOLDS)
        self.assertEqual({c['check']: c['status'] for c in checks}['applicability F1 (APPLIES)'], 'SKIP')
        self.assertEqual(overall, 'PASS')


class ManifestTests(unittest.TestCase):
    def test_the_manifest_names_models_versions_dataset_and_a_source_hash(self):
        dataset = load_dataset(DATASET)
        manifest = build_manifest(dataset, {'provider': 'rules'})
        for key in ('cardaman_version', 'git_commit', 'source_tree_sha256', 'dataset', 'models', 'versions', 'environment', 'manifest_sha256', 'created_at'):
            self.assertIn(key, manifest)
        self.assertEqual(manifest['dataset']['cases'], len(dataset.cases))
        self.assertIn('judge.conflict_confirm', manifest['versions']['prompts'])
        again = build_manifest(dataset, {'provider': 'rules'})
        self.assertEqual(manifest['manifest_sha256'], again['manifest_sha256'])              # the timestamp is outside the hash
        other = build_manifest(dataset, {'provider': 'ollama'})
        same, differences = comparable(manifest, other)
        self.assertEqual((same, differences), (False, ['mode']))



class RescoreTests(unittest.TestCase):
    def test_a_finished_run_is_rescored_from_its_packets_against_the_dataset_on_disk(self):
        from regchain.evaluation.harness import rescore, run
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        run_dir, manifest, before = run(DATASET, Path(temp.name), provider='rules', retrieval='lexical', cases=['C08', 'C27'], label='r')
        directory, _, after = rescore(run_dir, DATASET)
        self.assertEqual(directory, run_dir)
        for name in ('results.rescored.json', 'rescore.json', 'metrics.json', 'report.md'):
            self.assertTrue((run_dir / name).is_file(), name)
        note = json.loads((run_dir / 'rescore.json').read_text(encoding='utf-8'))
        self.assertEqual((note['cases_rescored'], note['dataset_cases']), (2, 30))
        self.assertEqual(note['dataset_sha256_at_run'], manifest['dataset']['sha256'])
        # Same packets, same labels: the same numbers; the original results and manifest are untouched.
        self.assertEqual(after['extraction'], before['extraction'])
        self.assertEqual(after['applicability']['accuracy'], before['applicability']['accuracy'])
        self.assertEqual(json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))['manifest_sha256'], manifest['manifest_sha256'])

if __name__ == '__main__':
    unittest.main()

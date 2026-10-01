"""The v0.19 evaluation additions: label-source split, the unchanged tr-aml-v1 digest, subset scoring that
writes nothing, thinking-call and per-obligation stage costs, the trade-off / reranker / version tables,
the v0.19 run flags, and the B1, B3, B5 and B6 fixes; plus the coverage-error reading of v0.19 packets.

Everything runs on synthetic run folders or on the rules provider (no model, no network), like
test_evaluation and test_v018_evaluation.
"""
import contextlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from regchain.evaluation import metrics as m
from regchain.evaluation.cli import main
from regchain.evaluation.coverage_errors import (auto_category, first_judgement, passage_rows, truncation_signal, v19_view,
                                                 verifier_view)
from regchain.evaluation.harness import (Components, aggregate, configure_models, load_dataset, prediction_rows, rescore, run, score_case,
                                         score_partial, score_subset, with_call_costs)
from regchain.evaluation.labels import audit
from regchain.evaluation.manifest import ENV_KEYS, build_manifest, dataset_digest, without_provenance_defaults
from regchain.evaluation.pending import Ledger
from regchain.evaluation.report import (NO_WINNER, cases_of, compare_markdown, ctx_comparison_markdown, derived, label_source_banner,
                                        pipeline_of, render_markdown, reranker_ab_markdown, scored_cases, tradeoff_markdown,
                                        version_diff_markdown)
from regchain.evaluation.schema import Dataset
from regchain.evaluation.taxonomy import DISPUTED_BY_DATASET, applicability_error, classify_results, disputed
from regchain.pilot.engine import pipeline_settings

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / 'evaluation' / 'datasets' / 'tr-aml-v1.json'
TR_AML_V1_DIGEST = '4ab81127426ac5d68f4bdf93c9d8b2d575c36ac9865698c73892e2dd33305ffa'
ENV = ('APPLICABILITY_CLEAR_MATCH', 'RELEVANCE_SCREEN', 'COVERAGE_PIPELINE', 'JUDGE_CTX_MODE', 'JUDGE_NUM_CTX', 'JUDGE_NUM_CTX_LARGE')
EVIDENCE = 'kimlik tespiti işlemden önce tamamlanır'
PASSAGE = 'a' * 64


def keep_env(test):
    saved = {key: os.environ.get(key) for key in ENV}

    def restore():
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    test.addCleanup(restore)


def quiet(argv):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return main(argv)


def writable(directory):
    for path in Path(directory).rglob('*'):
        if path.is_file():
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def snapshot(directory):
    return {str(p.relative_to(directory)): (p.stat().st_mtime_ns, p.stat().st_size) for p in Path(directory).rglob('*') if p.is_file()}


# ---------------------------------------------------------------- a synthetic mini run folder

def case(case_id, article, clause, keyword, **expectation):
    return {'case_id': case_id, 'title': case_id, 'categories': [], 'jurisdiction': 'TR', 'regulation_id': 'YONETMELIK:200713012',
            'regulation_fixture': 'x', 'target_sections': [article], 'company_profile': {}, 'policy_documents': ['p.md'],
            'expected_obligations': [{'article': article, 'clause': clause, 'action_keywords': [keyword], **expectation}]}


MINI = {'format': 'cardaman-evaluation-dataset-v1', 'dataset_id': 'mini-v1', 'version': '1', 'default_label_source': 'MANUAL_LEGAL_READING',
        'cases': [case('S01', '5', '(2)', 'tamamlanır', applicability='APPLIES', coverage='COVERS_TEXT', conflict=False, evidence=[EVIDENCE],
                       label_sources={'coverage': 'HUMAN_REVIEWED'}),
                  case('S02', '46', '(1)', 'saklanır', applicability='APPLIES', coverage='CONFLICT', conflict=True, label_source='SYNTHETIC_CONTROLLED'),
                  case('S03', '8', '(1)', 'alınır', applicability='DOES_NOT_APPLY', coverage='NOT_ASSESSED', entity_gate='MISMATCH')]}


def packet_row(oid, article, clause, action, applicability='APPLIES', coverage='COVERS_TEXT', conflict=False, assessed=True, rule='RULE_CLEAR_MATCH',
               timings=None, elapsed=None):
    return {'id': oid, 'source_label': f'Yönetmelik 200713012 md. {article}', 'candidate': {'subject': 'Yükümlüler', 'required_action': action},
            'signals': [{'source_id': PASSAGE}] if conflict else [], 'retrieved_policy_ids': [PASSAGE], 'judged_policy_ids': [PASSAGE],
            'evidence_signals': [{'source_id': PASSAGE, 'rank': 1}], 'diagnostics': [], 'elapsed_ms': elapsed, 'timings_ms': timings,
            'proposal': {'applicability': applicability, 'coverage': coverage, 'coverage_assessed': assessed, 'applicability_rule': rule,
                         'applicability_scope': {'child_clause': clause, 'match': 'MATCH'}, 'trace': {'decided_by': rule}, 'review_flags': [],
                         'policy_evidence': [{'source_id': PASSAGE, 'quote': 'q'}], 'policy_checks': []}}


ROWS = {'S01': packet_row('o1', '5', '(2)', 'kimlik tespiti işlemden önce tamamlanır', coverage='PARTIAL', elapsed=9000,
                          timings={'retrieval': 100, 'fast_classifier': 3000, 'aggregation': 5, 'embedding_cache_hits': 4}),
        'S02': packet_row('o2', '46', '(1)', 'belgeler sekiz yıl saklanır', coverage='CONFLICT', conflict=True, elapsed=40000,
                          timings={'retrieval': 200, 'fast_classifier': 2500, 'thinking_verifier': 30000, 'aggregation': 10}),
        'S03': packet_row('o3', '8', '(1)', 'derneğin adı alınır', applicability='DOES_NOT_APPLY', coverage='UNKNOWN', assessed=False,
                          rule='ENTITY_GATE', elapsed=20)}
CALLS = [{'case_id': 'S01', 'obligation_id': 'o1', 'task': 'judge.relevance', 'thinking': False, 'status': 'OK', 'elapsed_ms': 1000},
         {'case_id': 'S01', 'obligation_id': 'o1', 'task': 'judge.fast', 'thinking': False, 'status': 'OK', 'elapsed_ms': 3000,
          'prompt_tokens': 500, 'output_tokens': 50},
         {'case_id': 'S02', 'obligation_id': 'o2', 'task': 'judge.fast', 'thinking': False, 'status': 'OK', 'elapsed_ms': 2500},
         {'case_id': 'S02', 'obligation_id': 'o2', 'task': 'judge.verify', 'thinking': True, 'status': 'OK', 'elapsed_ms': 30000,
          'prompt_tokens': 700, 'output_tokens': 500},
         {'case_id': 'S02', 'obligation_id': 'o2', 'task': 'judge.verify', 'thinking': True, 'status': 'OK', 'cache_hit': True, 'elapsed_ms': 0},
         {'case_id': 'S02', 'obligation_id': 'o2', 'task': 'judge.verify', 'thinking': True, 'status': 'PROVIDER_FAILURE',
          'failure_code': 'OUTPUT_TRUNCATED', 'elapsed_ms': 1000},
         {'case_id': 'S02', 'obligation_id': 'o2', 'task': 'draft', 'thinking': True, 'status': 'OK', 'elapsed_ms': 4000}]


def mini_run(root: Path, name='r1', data=None, rows=None):
    """A run folder as harness.run leaves it (manifest, results.json in run order, packets, ai-calls.jsonl), with no model."""
    data, rows = data or MINI, rows or ROWS
    dataset_path = root / 'datasets' / f'{data["dataset_id"]}.json'
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    dataset = load_dataset(dataset_path)
    run_dir = root / 'runs' / name
    (run_dir / 'packets').mkdir(parents=True)
    results = []
    for item in dataset.cases:
        payload = {'obligations': [rows[item.case_id]],
                   'policies': [{'filename': 'p.md', 'chunks': [{'source_id': PASSAGE, 'text': f'Politika: {EVIDENCE}.'}]}]}
        (run_dir / 'packets' / f'{item.case_id}.json').write_text(json.dumps({'events': [{'payload': payload}]}, ensure_ascii=False), encoding='utf-8')
        predictions = prediction_rows(payload, item.regulation_id)
        scored, extras = score_case(item, payload, predictions)
        results.append({'case_id': item.case_id, 'title': item.title, 'categories': [], 'error': None, 'predictions': predictions,
                        'scored': scored, 'extras': extras, 'extraction': [], 'timings': {'coverage': 1000, 'parse': 5}, 'wall_seconds': 10.0})
    # Written in another order than the dataset's: scoring pairs by case id, never by position (B6).
    (run_dir / 'results.json').write_text(json.dumps(results[::-1], ensure_ascii=False), encoding='utf-8')
    (run_dir / 'ai-calls.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in CALLS), encoding='utf-8')
    manifest = build_manifest(dataset, {'provider': 'ollama', 'retrieval': 'hybrid', 'reranker': False, 'reranker_status': {'status': 'off'},
                                        'cases': [c.case_id for c in dataset.cases], 'overrides': {}}, dataset_path=dataset_path)
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
    return run_dir, dataset_path


class LabelSourceTests(unittest.TestCase):
    def test_resolution_order_field_expectation_case_dataset_then_rule_derived(self):
        expected = {'label_sources': {'coverage': 'HUMAN_REVIEWED'}, 'label_source': 'SYNTHETIC_CONTROLLED'}
        self.assertEqual(m.label_source(expected, 'coverage', 'MANUAL_LEGAL_READING', 'MANUAL_LEGAL_READING'), 'HUMAN_REVIEWED')
        self.assertEqual(m.label_source(expected, 'applicability', 'MANUAL_LEGAL_READING'), 'SYNTHETIC_CONTROLLED')
        self.assertEqual(m.label_source({}, 'conflict', 'HUMAN_REVIEWED', 'MANUAL_LEGAL_READING'), 'HUMAN_REVIEWED')
        self.assertEqual(m.label_source({'label_source': None, 'label_sources': {}}, 'conflict', None, 'MANUAL_LEGAL_READING'), 'MANUAL_LEGAL_READING')
        self.assertEqual(m.label_source({}, 'coverage'), 'RULE_DERIVED')          # a results.json written before v0.19

    def test_the_split_sums_to_the_total_and_each_field_counts_under_its_own_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset_path = mini_run(Path(tmp))
            _, metrics = score_subset(run_dir, None, dataset_path)
        split = metrics['by_label_source']
        self.assertEqual(list(split), ['HUMAN_REVIEWED', 'MANUAL_LEGAL_READING', 'SYNTHETIC_CONTROLLED'])
        for family in ('applicability', 'coverage', 'conflict', 'entity_gate'):
            self.assertEqual(sum(block[family]['n'] for block in split.values()), metrics[family]['n'], family)
        self.assertEqual(sum(block['expectations'] for block in split.values()), 3)
        self.assertEqual(sum(block['extraction']['expected_required'] for block in split.values()), metrics['extraction']['expected_required'])
        # S01's coverage is HUMAN_REVIEWED, its applicability the dataset default; S02 is SYNTHETIC_CONTROLLED throughout.
        self.assertEqual((split['HUMAN_REVIEWED']['coverage']['n'], split['HUMAN_REVIEWED']['applicability']['n']), (1, 0))
        self.assertEqual(split['MANUAL_LEGAL_READING']['applicability']['n'], 2)
        self.assertEqual((split['SYNTHETIC_CONTROLLED']['conflict']['tp'], split['SYNTHETIC_CONTROLLED']['coverage']['accuracy']), (1, 1.0))
        self.assertEqual(split['HUMAN_REVIEWED']['retrieval']['recall_at_3'], 1.0)      # retrieval follows the coverage label
        mismatch = next(x for x in metrics['mismatches'] if x['kind'] == 'coverage_mismatch')
        self.assertEqual((mismatch['where'], mismatch['label_source']), ('S01 md.5(2)', 'HUMAN_REVIEWED'))
        self.assertEqual(label_source_banner(metrics), [])                       # no RULE_DERIVED row, no banner

    def test_a_legacy_dataset_is_rule_derived_and_the_report_says_what_that_measures(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, manifest, metrics = run(DATASET, Path(tmp), provider='rules', retrieval='lexical', cases=['C08', 'C09'], label='src')
            report = (run_dir / 'report.md').read_text(encoding='utf-8')
        self.assertEqual(list(metrics['by_label_source']), ['RULE_DERIVED'])
        self.assertEqual(metrics['by_label_source']['RULE_DERIVED']['applicability']['n'], metrics['applicability']['n'])
        self.assertIn('## By label source', report)
        self.assertIn('RULE_DERIVED rows measure agreement with the generator rule', report)
        self.assertIn('not legal accuracy', report)
        self.assertIn('Coverage F1 COVERS_TEXT / PARTIAL / CONFLICT / NO_EVIDENCE', report)


class ManifestTests(unittest.TestCase):
    def test_the_tr_aml_v1_digest_is_unchanged_by_the_label_source_fields(self):
        dataset = load_dataset(DATASET)
        self.assertEqual(dataset_digest(dataset), TR_AML_V1_DIGEST)
        self.assertEqual(build_manifest(dataset, {'provider': 'rules'})['dataset']['sha256'], TR_AML_V1_DIGEST)
        # Unset provenance keys are dropped at every level, even when a dump carries them explicitly ...
        dump = json.loads(dataset.model_dump_json())
        dump['default_label_source'] = None
        dump['cases'][0]['expected_obligations'][0].update(label_source=None, label_sources={})
        self.assertEqual(without_provenance_defaults(dump), json.loads(dataset.model_dump_json()))
        # ... while a stated source is content and changes the digest.
        stated = dataset.model_copy(update={'default_label_source': 'RULE_DERIVED'})
        self.assertNotEqual(dataset_digest(stated), TR_AML_V1_DIGEST)

    def test_the_dataset_path_is_recorded_outside_the_hash_and_the_v019_settings_are_recorded(self):
        dataset = load_dataset(DATASET)
        with_path = build_manifest(dataset, {'provider': 'rules'}, dataset_path=DATASET)
        without = build_manifest(dataset, {'provider': 'rules'})
        self.assertEqual(with_path['dataset_path'], str(DATASET))
        self.assertEqual(with_path['manifest_sha256'], without['manifest_sha256'])
        for key in ('COVERAGE_PIPELINE', 'JUDGE_CTX_MODE', 'JUDGE_NUM_CTX_SMALL'):
            self.assertIn(key, ENV_KEYS)
            self.assertIn(key, with_path['environment'])


class SubsetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_scoring_twice_gives_identical_metrics_and_writes_nothing(self):
        run_dir, dataset_path = mini_run(self.root)
        before = snapshot(run_dir)
        _, first = score_subset(run_dir, None, dataset_path)
        _, second = score_subset(run_dir, None, dataset_path)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertEqual(snapshot(run_dir), before)
        self.assertEqual(first['subset']['cases'], ['S01', 'S02', 'S03'])          # dataset order, whatever results.json's order
        self.assertEqual((first['coverage']['n'], first['conflict']['tp'], first['applicability']['accuracy']), (2, 1, 1.0))
        only = score_subset(run_dir, ['S02', 'S99'], dataset_path)[1]
        self.assertEqual((only['subset']['cases'], only['subset']['not_scored'], only['completed_cases']), (['S02'], ['S99'], 1))
        self.assertEqual(only['performance']['llm_calls'], 5)                       # only S02's calls
        with self.assertRaises(ValueError):
            score_subset(run_dir, ['S99'], dataset_path)

    def test_a_rules_run_is_reproduced_by_its_subset_score(self):
        run_dir, _, metrics = run(DATASET, self.root, provider='rules', retrieval='lexical', cases=['C01', 'C08', 'C27'], label='sub')
        _, again = score_subset(run_dir)                                            # dataset from the manifest (B1)
        for family in ('applicability', 'coverage', 'conflict', 'entity_gate', 'retrieval', 'extraction', 'proposals', 'reliability',
                       'by_label_source', 'mismatches', 'usage', 'completed_cases'):
            self.assertEqual(again[family], metrics[family], family)
        self.assertEqual({k: v for k, v in again['performance'].items()}, metrics['performance'])

    def test_with_call_costs_counts_thinking_calls_of_a_metrics_file_written_before_v019(self):
        run_dir, dataset_path = mini_run(self.root)
        _, metrics = score_subset(run_dir, None, dataset_path)
        old = json.loads(json.dumps(metrics))
        for key in ('thinking_calls', 'thinking_calls_live', 'thinking_calls_per_obligation', 'strong_model_calls',
                    'strong_model_calls_per_obligation', 'live_calls_per_obligation', 'total_tokens'):
            del old['performance'][key]
        del old['reliability']['failures_by_code']
        filled = with_call_costs(run_dir, old)
        self.assertEqual(filled['performance']['thinking_calls'], metrics['performance']['thinking_calls'])
        self.assertEqual(filled['reliability']['failures_by_code'], {'OUTPUT_TRUNCATED': 1})
        self.assertNotIn('thinking_calls', old['performance'])                     # a copy; the argument is not touched


class CostTests(unittest.TestCase):
    def test_thinking_strong_model_and_live_calls_per_obligation(self):
        costs = m.call_costs(CALLS, 2)
        self.assertEqual((costs['thinking_calls'], costs['thinking_calls_live']), (4, 3))
        self.assertEqual((costs['strong_model_calls'], costs['strong_model_calls_per_obligation']), (3, 1.5))   # draft thinks but is no judge
        self.assertEqual((costs['thinking_calls_per_obligation'], costs['live_calls_per_obligation']), (2.0, 3.0))
        self.assertEqual(costs['total_tokens'], 1750)
        self.assertTrue(m.thinking({'thinking': 'True'}) and not m.thinking({'thinking': None}))
        self.assertEqual(m.failures_by_code([*CALLS, {'status': 'CIRCUIT_OPEN'}]), {'OUTPUT_TRUNCATED': 1, 'CIRCUIT_OPEN': 1})

    def test_the_stage_breakdown_is_per_assessed_obligation(self):
        predictions = [{'coverage_assessed': True, 'elapsed_ms': 9000, 'timings_ms': ROWS['S01']['timings_ms']},
                       {'coverage_assessed': True, 'elapsed_ms': 40000, 'timings_ms': ROWS['S02']['timings_ms']},
                       {'coverage_assessed': False, 'elapsed_ms': 20, 'timings_ms': {'applicability': 20}},
                       {'coverage_assessed': True, 'elapsed_ms': 3000, 'timings_ms': None}, 7]
        stages = m.obligation_stages(predictions)
        self.assertEqual((stages['assessed_obligations'], stages['with_stage_timings']), (3, 2))
        self.assertEqual(list(stages['stages']), ['retrieval', 'fast_classifier', 'thinking_verifier', 'aggregation'])   # pipeline order
        self.assertNotIn('embedding_cache_hits', stages['stages'])
        verifier = stages['stages']['thinking_verifier']
        self.assertEqual((verifier['mean_s'], verifier['median_s'], verifier['max_s'], verifier['obligations']), (15.0, 15.0, 30.0, 1))
        self.assertEqual((stages['total']['mean_s'], stages['total']['median_s'], stages['total']['obligations']), (17.33, 9.0, 3))
        self.assertEqual(m.obligation_stages([1, 2]), {'assessed_obligations': 0, 'with_stage_timings': 0, 'stages': {}, 'total': None})

    def test_the_run_metrics_and_report_carry_the_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset_path = mini_run(Path(tmp))
            manifest, metrics = score_subset(run_dir, None, dataset_path)
        perf = metrics['performance']
        self.assertEqual((perf['thinking_calls'], perf['obligations'], perf['thinking_calls_per_obligation']), (4, 3, 1.3333))
        self.assertEqual(perf['obligation_stages']['stages']['thinking_verifier']['obligations'], 1)
        text = render_markdown(manifest, metrics, [])
        self.assertIn('| thinking_verifier | 15.0 | 15.0 | 30.0 | 1 |', text)
        self.assertIn('OUTPUT_TRUNCATED 1', text)
        self.assertIn('Thinking calls (all / live) · per obligation', text)


class TableTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_dir, self.dataset = mini_run(self.root)
        manifest, metrics = score_subset(self.run_dir, None, self.dataset)
        slower = json.loads(json.dumps(metrics))
        slower['performance'].update(seconds_per_obligation=20.0, thinking_calls_per_obligation=0.5)
        slower['coverage']['macro']['f1'] = 0.9
        self.legs = [('A-ext4b-judge8b', manifest, metrics), ('B-ext8b-judge8b', manifest, slower)]

    def test_the_tradeoff_table_names_every_row_and_chooses_no_winner(self):
        text = tradeoff_markdown(self.legs)
        for label in ('Coverage macro F1', 'Conflict precision', 'Conflict recall', 'Applicability accuracy', 'UNKNOWN rate',
                      'Runtime / obligation (s)', 'LLM calls / obligation', 'Thinking calls / obligation', 'Total tokens', 'Model errors'):
            self.assertRegex(text, rf'\n\| {re.escape(label)} \|')
        self.assertIn(NO_WINNER, text)
        self.assertIn('| Thinking calls / obligation | 1.3333 | 0.5 (-0.8333 (-62%)) |', text)
        self.assertNotRegex(text.replace(NO_WINNER, ''), r'(?i)winner|best|recommend')

    def test_the_reranker_table_states_the_rule_with_the_measured_deltas(self):
        a, b = self.legs
        text = reranker_ab_markdown(a, b)
        for label in ('R@3', 'R@5', 'MRR', 'Evidence hit rate', 'Coverage macro F1', 'Conflict recall', 'Runtime / obligation (s)', 'LLM calls / obligation'):
            self.assertRegex(text, rf'\n\| {re.escape(label)} \|')
        self.assertIn('the reranker stays OFF unless it helps', text)
        self.assertIn('Measured deltas (B − A): R@3 +0.0 pp', text)
        self.assertIn('Retrieval improved (R@3 or MRR up): no', text)
        self.assertIn(NO_WINNER, text)

    def test_the_version_diff_has_every_v019_row(self):
        text = version_diff_markdown(self.legs[0][2], self.legs[1][2], 'old', 'new')
        self.assertIn('| METRIC | v0.18 | v0.19 | DELTA |', text)
        for label in ('Applicability accuracy', 'APPLIES F1', 'DOES_NOT_APPLY F1', 'UNKNOWN rate', 'Coverage macro F1', 'Coverage F1 COVERS_TEXT',
                      'Coverage F1 PARTIAL', 'Coverage F1 CONFLICT', 'Coverage F1 NO_EVIDENCE', 'Conflict precision', 'Conflict recall', 'Conflict FPR',
                      'R@3', 'Runtime / obligation (s)', 'LLM calls / obligation', 'Thinking calls / obligation', 'Model failures'):
            self.assertRegex(text, rf'\n\| {re.escape(label)} \|')
        self.assertNotIn('different numbers of cases', text)

    def test_compare_scores_common_cases_in_memory_and_leaves_the_runs_alone(self):
        other, _ = mini_run(self.root, 'r2')
        results = json.loads((other / 'results.json').read_text(encoding='utf-8'))
        (other / 'results.json').write_text(json.dumps([r for r in results if r['case_id'] != 'S01']), encoding='utf-8')
        before = {d: snapshot(d) for d in (self.run_dir, other)}
        for view in ('table', 'tradeoff', 'reranker', 'versions'):
            out = self.root / f'{view}.md'
            self.assertEqual(quiet(['compare', '--runs', str(self.run_dir), str(other), '--common-cases', '--view', view, '--out', str(out)]), 0)
            text = out.read_text(encoding='utf-8')
            self.assertIn('Scored in memory on 2 case(s) every run completed: S02, S03', text)
        self.assertIn('Thinking calls / obligation', text)
        self.assertEqual({d: snapshot(d) for d in (self.run_dir, other)}, before)
        out = self.root / 'cases.md'
        self.assertEqual(quiet(['compare', '--runs', str(self.run_dir), '--cases', 'S01', '--dataset', str(self.dataset), '--out', str(out)]), 0)
        self.assertIn('(3 run, 1 scored)', out.read_text(encoding='utf-8'))
        with self.assertRaises(SystemExit):
            quiet(['compare', '--runs', str(self.run_dir), '--view', 'reranker'])

    def test_compare_names_the_cases_run_and_the_reranker_that_ran(self):
        manifest, metrics = self.legs[0][1], self.legs[0][2]
        leaked = {**manifest, 'mode': {**manifest['mode'], 'cases': ['S01'], 'reranker_status': {'status': 'off'}},
                  'models': {**manifest['models'], 'reranker': 'BAAI/bge-reranker-v2-m3'}, 'dataset': {**manifest['dataset'], 'cases': 30}}
        text = compare_markdown([('x', leaked, {**metrics, 'subset': None, 'completed_cases': 1})])
        self.assertIn('reranker `off`', text)                                       # the env named a model; none ran (B3)
        self.assertIn('(1 run, 1 scored)', text)
        active = {**leaked, 'mode': {**leaked['mode'], 'reranker_status': {'status': 'active'}}}
        self.assertIn('reranker `active (BAAI/bge-reranker-v2-m3)`', compare_markdown([('y', active, metrics)]))


class FixTests(unittest.TestCase):
    def setUp(self):
        keep_env(self)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(writable, temp.name)
        self.root = Path(temp.name)

    def test_b1_a_run_is_rescored_without_naming_its_dataset(self):
        run_dir, manifest, metrics = run(DATASET, self.root, provider='rules', retrieval='lexical', cases=['C27'], label='b1')
        on_disk = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual((on_disk['dataset_path'], on_disk['manifest_sha256']), (str(DATASET.resolve()), manifest['manifest_sha256']))
        self.assertEqual(score_partial(run_dir)[2]['completed_cases'], 1)
        self.assertEqual(quiet(['report', '--run', str(run_dir)]), 0)

    def test_b5_disputes_belong_to_their_dataset(self):
        self.assertIsNotNone(disputed('C12', '5', '(2)'))                         # tr-aml-v1, the default for older callers
        self.assertIsNotNone(disputed('C12', '5', '(2)', 'tr-aml-v1'))
        self.assertIsNone(disputed('C12', '5', '(2)', 'independent-v1'))
        self.assertEqual(set(DISPUTED_BY_DATASET), {'tr-aml-v1'})
        exp = {'article': '6', 'clause': '(1)', 'action_keywords': [], 'applicability': 'DOES_NOT_APPLY', 'entity_gate': 'MISMATCH',
               'coverage': 'ANY', 'conflict': None, 'evidence': [], 'required': True, 'notes': ''}
        item = {'expected': exp, 'prediction': 'K', 'matched': True, 'applicability': ('DOES_NOT_APPLY', 'APPLIES')}
        self.assertEqual(applicability_error('C09', item)['taxonomy'], 'LABEL_AMBIGUITY')
        self.assertNotEqual(applicability_error('C09', item, dataset_id='independent-v1')['taxonomy'], 'LABEL_AMBIGUITY')
        results = [{'case_id': 'C09', 'error': None, 'extras': [], 'predictions': [{'key': 'K', 'obligation_id': 'o'}], 'scored': [item]}]
        self.assertNotIn('LABEL_AMBIGUITY', classify_results(results, {}, 'independent-v1')['counts']['applicability'])
        self.assertIn('LABEL_AMBIGUITY', classify_results(results, {})['counts']['applicability'])

    def test_b6_results_out_of_the_dataset_order_are_refused(self):
        dataset = Dataset.model_validate(MINI)
        result = lambda case_id: {'case_id': case_id, 'predictions': [], 'scored': [], 'extras': [], 'calls': [], 'wall_seconds': 1.0, 'error': None}
        with self.assertRaises(ValueError):
            aggregate(dataset, [result('S02'), result('S01'), result('S03')])
        with self.assertRaises(ValueError):
            aggregate(dataset, [result('S01'), result('S02')])
        self.assertEqual(aggregate(dataset, [result('S01'), result('S02'), result('S03')])['extraction']['found_required'], 0)

    def test_the_v019_run_flags_set_the_pipeline_settings(self):
        self.assertEqual(configure_models({'coverage_pipeline': 'v19', 'ctx_mode': 'adaptive'}),
                         {'COVERAGE_PIPELINE': 'v19', 'JUDGE_CTX_MODE': 'adaptive'})
        self.assertEqual((os.environ['COVERAGE_PIPELINE'], os.environ['JUDGE_CTX_MODE']), ('v19', 'adaptive'))
        for bad in ({'coverage_pipeline': 'v20'}, {'ctx_mode': 'huge'}):
            with self.assertRaises(ValueError):
                configure_models(bad)
        self.assertEqual(quiet(['run', '--dataset', str(DATASET), '--out', str(self.root), '--cases', 'C27', '--label', 'flags19',
                                '--coverage-pipeline', 'v18', '--ctx-mode', 'fixed']), 0)
        manifest = json.loads(next(self.root.glob('*-flags19-rules/manifest.json')).read_text(encoding='utf-8'))
        self.assertEqual(manifest['mode']['overrides'], {'COVERAGE_PIPELINE': 'v18', 'JUDGE_CTX_MODE': 'fixed'})
        self.assertEqual((manifest['environment']['COVERAGE_PIPELINE'], manifest['environment']['JUDGE_CTX_MODE']), ('v18', 'fixed'))

    def test_the_v019_matrices_name_the_legs_the_cases_and_the_v19_pipeline(self):
        cases = ['C11', 'C12', 'C13', 'C14', 'C16', 'C18', 'C25', 'C29']
        bench = json.loads((REPO / 'evaluation' / 'benchmark-matrix-v019.json').read_text(encoding='utf-8'))
        self.assertEqual([(leg['extraction_model'], leg['judge_model']) for leg in bench],
                         [('qwen3:4b', 'qwen3:8b'), ('qwen3:8b', 'qwen3:8b'), ('qwen3:4b', 'qwen3:4b')])
        ab = json.loads((REPO / 'evaluation' / 'reranker-ab-v019.json').read_text(encoding='utf-8'))
        self.assertEqual([bool(leg['reranker']) for leg in ab], [False, True])
        self.assertEqual((ab[0]['rerank_model'], ab[1]['rerank_model']), ('', 'BAAI/bge-reranker-v2-m3'))
        for leg in [*bench, *ab]:
            self.assertEqual((leg['coverage_pipeline'], leg['retrieval'], leg['applicability_clear_match'], leg['relevance_screen'], leg['cases']),
                             ('v19', 'hybrid', 'rule', 'on', cases))
            # Review finding 1: every leg names its judge window; unset, qwen3:8b ran at the 16,384 code default (80% GPU).
            self.assertEqual((str(leg['judge_num_ctx']), str(leg['judge_num_ctx_large']), leg['ctx_mode']), ('8192', '16384', 'adaptive'))
        ctx = json.loads((REPO / 'evaluation' / 'ctx-compare-v019.json').read_text(encoding='utf-8'))
        self.assertEqual([(leg['ctx_mode'], str(leg['judge_num_ctx']), leg['cases']) for leg in ctx],
                         [('fixed', '8192', cases), ('adaptive', '8192', cases)])
        self.assertEqual({(leg['extraction_model'], leg['judge_model'], leg['coverage_pipeline']) for leg in ctx}, {('qwen3:4b', 'qwen3:8b', 'v19')})


# ---------------------------------------------------------------- coverage errors on v0.19 packets

def v19_result(fast='POSSIBLE_CONFLICT', escalation='POSSIBLE_CONFLICT', verifier=None, relation='UNRELATED', notes=None, overlap=0.4):
    return {'source_id': PASSAGE, 'relation': relation, 'reason': 'r', 'pipeline': 'v19', 'topic_overlap': overlap, 'uncertainty': None,
            'signals': [{'type': 'WAIVER', 'strength': 'strong', 'text': 'atlanır'}],
            'fast': {'label': fast, 'quote': '', 'covered': [], 'missing': [], 'reason': 'f'}, 'escalation': escalation, 'verifier': verifier,
            **({'notes': notes} if notes else {})}


def v19_error_row(result, expected, actual):
    obligation = {'id': 'o', 'source_label': 'Yönetmelik 200713012 md. 5', 'evidence_signals': [{'source_id': PASSAGE, 'rank': 1}],
                  'diagnostics': [{'stage': 'passages', 'results': [result], 'filtered': []}], 'proposal': {'policy_checks': []}}
    payload = {'policies': [{'filename': 'p.md', 'chunks': [{'source_id': PASSAGE, 'text': f'Kimlik tespiti atlanır; {EVIDENCE}.'}]}]}
    passages = passage_rows(payload, obligation, [EVIDENCE])
    return {'expected': {'coverage': expected[0], 'conflict': expected[1]}, 'actual': {'coverage': actual[0], 'conflict': actual[1]},
            'passages': passages, 'evidence': {'needles': [{'text': EVIDENCE, 'in_packet': True, 'passages': []}], 'any_in_packet': True},
            'duty_truncation': {'truncated': False}, 'label': {'disputed': False}}


class V19CoverageErrorTests(unittest.TestCase):
    def test_a_verifier_that_said_no_conflict_is_the_overrule(self):
        verifier = {'conflict': False, 'contradiction_type': 'NONE', 'relation_if_no_conflict': 'UNRELATED', 'rationale': 'only an exception',
                    'confidence': 'HIGH'}
        result = v19_result(verifier=verifier)
        self.assertEqual(first_judgement(result)['contradicts'], 'NO')
        self.assertEqual((verifier_view(result)['code'], first_judgement(result)['support_by']), ('VERIFIER_NO_CONFLICT', 'verifier'))
        auto = auto_category(v19_error_row(result, ('CONFLICT', True), ('NO_EVIDENCE', False)))
        self.assertEqual(auto['primary'], 'VERIFIER_OVERRULE_ERROR')
        self.assertIn('only an exception', auto['root_cause'])
        # A conflict claim the code rule took back (no exact span) is an overrule too.
        claimed = v19_result(verifier={**verifier, 'conflict': True, 'contradiction_type': 'EXEMPTION_ADDED'}, relation='UNCLEAR',
                             notes=[{'code': 'CONFLICT_CLAIM_WITHOUT_SPAN', 'detail': 'no exact span'}])
        self.assertEqual(verifier_view(claimed)['by'], 'rule')
        self.assertEqual(auto_category(v19_error_row(claimed, ('CONFLICT', True), ('UNKNOWN', False)))['primary'], 'VERIFIER_OVERRULE_ERROR')

    def test_a_fast_irrelevant_that_was_never_escalated(self):
        result = v19_result(fast='IRRELEVANT', escalation=None)
        self.assertEqual((first_judgement(result)['contradicts'], first_judgement(result)['support_by']), ('NOT_ASKED', 'fast'))
        self.assertIsNone(verifier_view(result))
        conflict = auto_category(v19_error_row(result, ('CONFLICT', True), ('NO_EVIDENCE', False)))
        self.assertEqual(conflict['primary'], 'PREFILTER_FALSE_NEGATIVE')
        self.assertIn('never read it', conflict['root_cause'])
        coverage = auto_category(v19_error_row(result, ('COVERS_TEXT', False), ('NO_EVIDENCE', False)))
        self.assertEqual(coverage['primary'], 'SEMANTIC_MATCH_MISSED')
        self.assertIn('fast classifier (IRRELEVANT, not escalated)', coverage['root_cause'])

    def test_a_truncated_verifier_is_context_truncation(self):
        failed = v19_result(relation='UNCLEAR', notes=[{'code': 'VERIFIER_FAILED', 'failure_code': 'OUTPUT_TRUNCATED'}])
        failed['unjudged'] = True
        self.assertIn('OUTPUT_TRUNCATED', truncation_signal(failed, []))
        self.assertEqual(first_judgement(failed)['contradicts'], 'FAILED')
        self.assertEqual(auto_category(v19_error_row(failed, ('CONFLICT', True), ('UNKNOWN', False)))['primary'], 'CONTEXT_TRUNCATION')

    # Review finding 2: an attempt the pipeline recovered from is no cut.
    def test_a_retry_the_verifier_recovered_from_is_not_a_truncation(self):
        verifier = {'conflict': False, 'contradiction_type': 'NONE', 'relation_if_no_conflict': 'UNRELATED', 'rationale': 'no contradiction',
                    'confidence': 'HIGH'}
        retried = {'question': 'verify', 'attempt': 1, 'code': 'ProviderFailure', 'failure_code': 'OUTPUT_TRUNCATED', 'retry': 'asked once more, uncached'}
        result = v19_result(verifier=verifier, notes=[retried])
        calls = [{'task': 'judge.verify', 'status': 'PROVIDER_FAILURE', 'failure_code': 'OUTPUT_TRUNCATED', 'done_reason': 'length', 'elapsed_ms': 90000},
                 {'task': 'judge.verify', 'status': 'OK', 'done_reason': 'stop', 'elapsed_ms': 30000}]
        self.assertIsNone(truncation_signal(result, calls))
        auto = auto_category(v19_error_row(result, ('CONFLICT', True), ('NO_EVIDENCE', False)))
        self.assertEqual((auto['primary'], auto['secondary']), ('VERIFIER_OVERRULE_ERROR', []))
        # A context overflow answered after trimming, and a fast failure the verifier answered in its place.
        trimmed = v19_result(fast='SUPPORTS', escalation=None, relation='SUPPORTS',
                             notes=[{'question': 'fast', 'attempt': 1, 'code': 'ContextBudgetError', 'failure_code': 'CONTEXT_OVERFLOW',
                                     'retry': 'repeated with a trimmed payload: passage clipped'}])
        self.assertIsNone(truncation_signal(trimmed, []))
        fast_cut = v19_result(verifier=verifier, escalation='FAST_FAILED',
                              notes=[{'question': 'fast', 'attempt': 1, 'code': 'ProviderFailure', 'failure_code': 'OUTPUT_TRUNCATED'},
                                     {'question': 'fast', 'code': 'FAST_FAILED', 'failure_code': 'OUTPUT_TRUNCATED'}])
        fast_cut['fast'] = None
        self.assertIsNone(truncation_signal(fast_cut, []))
        # The second failure of the retry stays a cut, and so does a v0.18 overflow that was never answered.
        twice = v19_result(relation='UNCLEAR', notes=[retried, {**retried, 'attempt': 2, 'retry': None},
                                                      {'question': 'verify', 'code': 'VERIFIER_FAILED', 'failure_code': 'OUTPUT_TRUNCATED'}])
        self.assertIn('OUTPUT_TRUNCATED', truncation_signal(twice, []))
        v18 = {'relation': 'SUPPORTS', 'screen': 'NO', 'notes': [{'question': 'supports', 'attempt': 1, 'code': 'ContextBudgetError',
                                                                  'retry': 'repeated with a trimmed payload'}]}
        self.assertIsNone(truncation_signal(v18, []))
        self.assertIsNotNone(truncation_signal({**v18, 'unjudged': True, 'notes': [{'question': 'supports', 'code': 'ContextBudgetError'}]}, []))

    # Review finding 11: the elements and quantities the aggregation used.
    def test_the_v19_view_carries_the_aggregated_elements_and_the_partial_rationale_names_them(self):
        verifier = {'conflict': False, 'contradiction_type': 'NONE', 'relation_if_no_conflict': 'SUPPORTS', 'rationale': 'same act',
                    'confidence': 'MEDIUM', 'support_quote': EVIDENCE}
        result = v19_result(fast='IRRELEVANT', escalation='STRUCTURAL_SIGNAL', verifier=verifier, relation='SUPPORTS')
        result['fast']['missing'] = ['action', 'deadline_1']                     # stale after the verifier's SUPPORTS (smoke C12 md.5(2))
        result.update(covered_elements=['action', 'deadline_1'], missing_elements=[], quantity_match={'deadline_1': 'NOT_STATED'})
        self.assertEqual({k: v19_view(result)[k] for k in ('covered_elements', 'missing_elements', 'quantity_match')},
                         {'covered_elements': ['action', 'deadline_1'], 'missing_elements': [], 'quantity_match': {'deadline_1': 'NOT_STATED'}})
        rationale = auto_category(v19_error_row(result, ('COVERS_TEXT', False), ('PARTIAL', False)))
        self.assertEqual(rationale['primary'], 'COVERS_AS_PARTIAL')
        self.assertIn('quantities not stated or weaker: deadline_1 NOT_STATED', rationale['root_cause'])
        self.assertNotIn('missing elements', rationale['root_cause'])
        result.update(missing_elements=['item_2'], quantity_match={'deadline_1': 'SAME'})
        rationale = auto_category(v19_error_row(result, ('COVERS_TEXT', False), ('PARTIAL', False)))['root_cause']
        self.assertIn('missing elements: item_2', rationale)
        self.assertNotIn('quantities', rationale)


# ---------------------------------------------------------------- review findings (v0.19 context addendum)

RUNTIME_CALLS = [
    {'at': '2026-09-24T10:00:00+00:00', 'task': 'extraction', 'model': 'qwen3:4b', 'num_ctx': 16384, 'thinking': False, 'status': 'OK',
     'elapsed_ms': 4000, 'tokens_per_second': 100.0, 'prompt_tokens_per_second': 2000.0, 'gpu_fraction': 1.0, 'processor': '100% GPU',
     'load_duration_ms': 3200, 'eval_duration_ms': 3000},
    {'at': '2026-09-24T10:00:05+00:00', 'task': 'judge.fast', 'model': 'qwen3:4b', 'num_ctx': 16384, 'thinking': False, 'status': 'OK',
     'elapsed_ms': 2000, 'tokens_per_second': 80.0, 'prompt_tokens_per_second': 1800.0, 'gpu_fraction': 1.0, 'processor': '100% GPU',
     'window_action': 'fits', 'load_duration_ms': 5},
    {'at': '2026-09-24T10:00:10+00:00', 'task': 'judge.verify', 'model': 'qwen3:8b', 'num_ctx': 8192, 'thinking': True, 'status': 'OK',
     'elapsed_ms': 30000, 'tokens_per_second': 40.0, 'prompt_tokens_per_second': 900.0, 'gpu_fraction': 1.0, 'processor': '100% GPU',
     'window_action': 'fits', 'load_duration_ms': 6600},
    {'at': '2026-09-24T10:00:50+00:00', 'task': 'judge.verify', 'model': 'qwen3:8b', 'num_ctx': 16384, 'thinking': True, 'status': 'OK',
     'elapsed_ms': 50000, 'tokens_per_second': 20.0, 'gpu_fraction': 0.805, 'processor': '19%/81% CPU/GPU', 'window_action': 'fallback_large',
     'window_estimate_before': 7000, 'load_duration_ms': 4900},
    {'at': '2026-09-24T10:00:55+00:00', 'task': 'judge.verify', 'model': 'qwen3:8b', 'num_ctx': 8192, 'thinking': True, 'status': 'OK',
     'cache_hit': True, 'elapsed_ms': 0},
    {'at': '2026-09-24T10:01:45+00:00', 'task': 'draft', 'model': 'qwen3:4b', 'num_ctx': 16384, 'thinking': False, 'status': 'PROVIDER_FAILURE',
     'failure_code': 'TIMEOUT', 'elapsed_ms': 1000, 'window_action': 'compressed'},
    {'at': '2026-09-24T10:01:50+00:00', 'task': 'judge.fast', 'model': 'qwen3:4b', 'num_ctx': 16384, 'thinking': False,
     'status': 'CONTEXT_BUDGET_EXCEEDED', 'elapsed_ms': 0}]
# The order harness.Components.drain_calls wrote them in before the fix: the extraction provider's log, then the judge's.
LOG_ORDER = [RUNTIME_CALLS[i] for i in (0, 1, 5, 6, 2, 3, 4)]


def window_manifest(manifest, mode, base='8192', large='16384'):
    return {**manifest, 'environment': {**(manifest.get('environment') or {}), 'COVERAGE_PIPELINE': 'v19', 'JUDGE_CTX_MODE': mode,
                                        'JUDGE_NUM_CTX': base, 'JUDGE_NUM_CTX_LARGE': large},
            'models': {**manifest['models'], 'extraction': 'qwen3:4b', 'judge': 'qwen3:8b'}}


class WindowSettingTests(unittest.TestCase):
    """Review finding 1: the judge's base window can be set per run or leg, is recorded, and is shown."""

    def setUp(self):
        keep_env(self)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(writable, temp.name)
        self.root = Path(temp.name)

    def test_a_run_or_leg_sets_the_judge_windows(self):
        self.assertEqual(configure_models({'judge_num_ctx': 8192, 'judge_num_ctx_large': '16384', 'ctx_mode': 'adaptive'}),
                         {'JUDGE_NUM_CTX': '8192', 'JUDGE_NUM_CTX_LARGE': '16384', 'JUDGE_CTX_MODE': 'adaptive'})
        self.assertEqual((os.environ['JUDGE_NUM_CTX'], os.environ['JUDGE_NUM_CTX_LARGE']), ('8192', '16384'))
        for bad in ({'judge_num_ctx': '8k'}, {'judge_num_ctx': 1024}, {'judge_num_ctx_large': 200000}):
            with self.subTest(bad), self.assertRaises(ValueError):
                configure_models(bad)
        os.environ.pop('JUDGE_NUM_CTX', None)
        os.environ.pop('JUDGE_NUM_CTX_LARGE', None)
        self.assertEqual(quiet(['run', '--dataset', str(DATASET), '--out', str(self.root), '--cases', 'C27', '--label', 'win',
                                '--ctx-mode', 'fixed', '--judge-num-ctx', '8192']), 0)
        run_dir = next(self.root.glob('*-win-rules'))
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['mode']['overrides'], {'JUDGE_CTX_MODE': 'fixed', 'JUDGE_NUM_CTX': '8192'})
        self.assertEqual((manifest['environment']['JUDGE_NUM_CTX'], manifest['environment']['JUDGE_NUM_CTX_LARGE']), ('8192', ''))
        report = (run_dir / 'report.md').read_text(encoding='utf-8')
        self.assertIn('judge window `fixed` at num_ctx `8192`', report)
        self.assertIn('large `16384 (JUDGE_NUM_CTX_LARGE unset: code default)`', report)

    def test_the_manifest_records_the_large_window_and_the_fast_model(self):
        for key in ('JUDGE_NUM_CTX', 'JUDGE_NUM_CTX_LARGE', 'FAST_MODEL', 'FAST_MODEL_DIGEST'):
            self.assertIn(key, ENV_KEYS)

    def test_an_unset_window_is_named_as_the_code_default_not_as_a_plain_fixed_window(self):
        smoke = {'environment': {'COVERAGE_PIPELINE': 'v19', 'JUDGE_CTX_MODE': 'fixed', 'JUDGE_NUM_CTX': ''}}   # 20260924-173120's manifest
        eight = {'environment': {'COVERAGE_PIPELINE': 'v19', 'JUDGE_CTX_MODE': 'fixed', 'JUDGE_NUM_CTX': '8192', 'JUDGE_NUM_CTX_LARGE': '16384'}}
        self.assertIn('window `fixed` at num_ctx `16384 (JUDGE_NUM_CTX unset: code default)`', pipeline_of(smoke))
        self.assertIn('large `16384 (not recorded: code default)`', pipeline_of(smoke))
        self.assertEqual(pipeline_of(eight), 'coverage `v19`, window `fixed` at num_ctx `8192`, large `16384`')


class ManifestPromptTests(unittest.TestCase):
    """Review finding 8: a v19 manifest names the prompts the run uses."""

    def setUp(self):
        keep_env(self)

    def test_a_v19_manifest_names_the_fast_and_verify_prompts_and_their_hash(self):
        dataset = load_dataset(DATASET)
        os.environ['COVERAGE_PIPELINE'] = 'v19'
        v19 = build_manifest(dataset, {'provider': 'rules'})
        self.assertTrue({'judge.fast', 'judge.verify'} <= set(v19['versions']['prompts']))
        self.assertEqual(v19['versions']['coverage_prompts_sha256'], pipeline_settings(coverage_pipeline='v19')['coverage_prompts_sha256'])
        os.environ['COVERAGE_PIPELINE'] = 'v18'
        v18 = build_manifest(dataset, {'provider': 'rules'})
        self.assertNotIn('judge.fast', v18['versions']['prompts'])
        self.assertNotIn('coverage_prompts_sha256', v18['versions'])             # a v18 manifest keeps its v0.18 shape
        self.assertIn('judge.conflict_confirm', v18['versions']['prompts'])


class RuntimeTests(unittest.TestCase):
    """Review finding 6 and the addendum's WS6c metrics: throughput, GPU share, window actions, model switches."""

    def test_drain_calls_merges_the_logs_in_time_order(self):
        components = object.__new__(Components)
        provider_log, judge_log = [RUNTIME_CALLS[0], RUNTIME_CALLS[5]], [RUNTIME_CALLS[2], RUNTIME_CALLS[3]]
        components.provider, components.judge = type('P', (), {'call_log': provider_log})(), type('J', (), {'call_log': judge_log})()
        self.assertEqual(components.drain_calls(), [RUNTIME_CALLS[i] for i in (0, 2, 3, 5)])
        self.assertEqual((provider_log, judge_log), ([], []))
        # Records without a start time (test doubles) keep the order of the logs.
        components.provider.call_log.extend([{'task': 'b'}, {'task': 'a'}])
        self.assertEqual([c['task'] for c in components.drain_calls()], ['b', 'a'])

    def test_the_runtime_profile_by_model_and_window(self):
        runtime = m.runtime(LOG_ORDER)
        self.assertEqual(runtime['live_calls'], 6)
        self.assertEqual(runtime['by_model'], {'qwen3:4b': {'live_calls': 4, 'model_seconds': 7.0}, 'qwen3:8b': {'live_calls': 2, 'model_seconds': 80.0}})
        self.assertEqual(runtime['by_num_ctx'], {'8192': {'live_calls': 1, 'model_seconds': 30.0}, '16384': {'live_calls': 5, 'model_seconds': 57.0}})
        self.assertEqual(list(runtime['by_model_ctx']), ['qwen3:4b @ 16384', 'qwen3:8b @ 8192', 'qwen3:8b @ 16384'])
        self.assertEqual(runtime['tokens_per_second']['by_model'], {'qwen3:4b': {'calls': 2, 'mean': 90.0, 'median': 90.0},
                                                                    'qwen3:8b': {'calls': 2, 'mean': 30.0, 'median': 30.0}})
        self.assertEqual(runtime['prompt_tokens_per_second']['by_model']['qwen3:8b'], {'calls': 1, 'mean': 900.0, 'median': 900.0})
        gpu = runtime['gpu']
        self.assertEqual((gpu['qwen3:8b @ 16384']['full_gpu_share'], gpu['qwen3:8b @ 16384']['min_gpu_fraction'],
                          gpu['qwen3:8b @ 16384']['processors']), (0.0, 0.805, {'19%/81% CPU/GPU': 1}))
        self.assertEqual((gpu['qwen3:4b @ 16384']['observed'], gpu['qwen3:4b @ 16384']['full_gpu_share']), (2, 1.0))
        self.assertEqual(gpu['qwen3:8b @ 8192']['full_gpu_share'], 1.0)
        self.assertEqual(runtime['window_actions'], {'fits': 2, 'compressed': 1, 'fallback_large': 1})
        self.assertEqual(runtime['fallbacks'], 1)
        # In time order the run went 4b, 4b, 8b@8k, 8b@16k, 4b: two model switches and one window change. In the order the
        # logs were written (provider log, then judge log) it would read as one switch whatever the interleaving.
        self.assertEqual((runtime['model_switches'], runtime['window_switches'], runtime['time_ordered']), (2, 1, True))
        self.assertEqual((runtime['model_loads'], runtime['load_seconds']), (3, 14.7))
        self.assertEqual(m.runtime(RUNTIME_CALLS), runtime)
        empty = m.runtime([{'task': 'judge.fast', 'status': 'OK', 'elapsed_ms': 10}])      # a record written before v0.19
        self.assertEqual((empty['tokens_per_second']['all'], empty['gpu']['unknown @ ?']['full_gpu_share'], empty['window_actions']), (None, None, {}))

    def test_the_run_metrics_carry_the_runtime_profile_and_older_metrics_get_it_from_their_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, dataset_path = mini_run(Path(tmp))
            (run_dir / 'ai-calls.jsonl').write_text(''.join(json.dumps({'case_id': 'S02', **c}) + '\n' for c in LOG_ORDER), encoding='utf-8')
            manifest, metrics = score_subset(run_dir, None, dataset_path)
            self.assertEqual(metrics['performance']['runtime']['model_switches'], 2)
            old = json.loads(json.dumps(metrics))
            del old['performance']['runtime']
            self.assertEqual(with_call_costs(run_dir, old)['performance']['runtime'], metrics['performance']['runtime'])
        text = render_markdown(manifest, metrics, [])
        self.assertIn('## Runtime by model and window', text)
        self.assertIn('| qwen3:8b @ 16384 | 1 | 50.0 | 0.0% | 19%/81% CPU/GPU 1 |', text)
        self.assertIn('Model switches (live calls in time order): 2', text)
        self.assertIn('window actions: fits 2, compressed 1, fallback_large 1', text)
        self.assertIn('tokens/s qwen3:4b 90.0 / 90.0', text)


class CtxComparisonTests(unittest.TestCase):
    """Items 8-9 of the user directive: fixed 8k against adaptive 8k/16k on the same cases, no automatic winner."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_dir, self.dataset = mini_run(self.root)
        manifest, metrics = score_subset(self.run_dir, None, self.dataset)
        fixed = json.loads(json.dumps(metrics))
        fixed['performance']['runtime'] = m.runtime([c for c in RUNTIME_CALLS if c['num_ctx'] != 16384 or c['model'] != 'qwen3:8b'])
        adaptive = json.loads(json.dumps(metrics))
        adaptive['performance']['runtime'] = m.runtime(RUNTIME_CALLS)
        adaptive['performance']['seconds_per_obligation'] = 20.0
        adaptive['conflict']['recall'] = 0.5
        self.fixed = ('fixed-8k', window_manifest(manifest, 'fixed'), fixed)
        self.adaptive = ('adaptive-8k-16k', window_manifest(manifest, 'adaptive'), adaptive)

    def test_the_table_names_quality_cost_and_the_rule_and_chooses_no_winner(self):
        text = ctx_comparison_markdown(self.fixed, self.adaptive)
        for label in ('Applicability accuracy', 'Coverage macro F1', 'Coverage F1 COVERS_TEXT', 'Coverage F1 PARTIAL', 'Coverage F1 CONFLICT',
                      'Coverage F1 NO_EVIDENCE', 'Conflict precision', 'Conflict recall', 'Runtime / obligation (s)', 'Tokens/s qwen3:8b (mean / median)',
                      'Prompt tokens/s qwen3:8b (mean / median)', 'Window fallbacks to the large window', 'Window compressed / trimmed',
                      '100% GPU share qwen3:8b @ 16384', 'Live calls at num_ctx 16384', 'Model switches'):
            self.assertRegex(text, rf'\n\| {re.escape(label)} \|')
        self.assertIn('| Conflict recall | 100.0% | 50.0% | -50.0 pp |', text)
        self.assertIn('| Tokens/s qwen3:8b (mean / median) | 40.0 / 40.0 | 30.0 / 30.0 | -10 (-25%) |', text)
        self.assertIn('| 100% GPU share qwen3:8b @ 16384 | — | 0.0% | — |', text)
        self.assertIn('adaptive 8k/16k becomes the default only if its quality is not meaningfully worse than fixed 8k', text)
        self.assertIn('Measured deltas (adaptive − fixed 8k): applicability accuracy +0.0 pp, coverage macro F1 +0.0 pp', text)
        self.assertIn('conflict recall -50.0 pp', text)
        self.assertIn('one conflict-positive row is 100.0 pp of conflict recall', text)
        self.assertIn(NO_WINNER, text)
        self.assertNotRegex(text.replace(NO_WINNER, ''), r'(?i)winner|\bbest\b|recommend')
        self.assertNotIn('is not a fixed', text)

    def test_a_leg_that_is_not_the_window_it_claims_is_named(self):
        sixteen = (self.fixed[0], {**self.fixed[1], 'environment': {**self.fixed[1]['environment'], 'JUDGE_NUM_CTX': ''}}, self.fixed[2])
        text = ctx_comparison_markdown(sixteen, self.adaptive)
        self.assertIn('**`fixed-8k` is not a fixed 8192 window** (JUDGE_CTX_MODE `fixed`, judge num_ctx `16384 (JUDGE_NUM_CTX unset: code default)`)', text)
        swapped = ctx_comparison_markdown(self.adaptive, self.fixed)
        self.assertIn('is not an adaptive 8192/16384 window', swapped)

    def test_compare_view_ctx_scores_both_runs_in_memory(self):
        other, _ = mini_run(self.root, 'r2')
        out = self.root / 'ctx.md'
        self.assertEqual(quiet(['compare', '--runs', str(self.run_dir), str(other), '--common-cases', '--view', 'ctx', '--out', str(out)]), 0)
        text = out.read_text(encoding='utf-8')
        self.assertIn('# Judge window: fixed 8k / adaptive 8k-16k', text)
        self.assertIn('Scored in memory on 3 case(s) every run completed', text)
        with self.assertRaises(SystemExit):
            quiet(['compare', '--runs', str(self.run_dir), '--view', 'ctx'])


class DisplayTests(unittest.TestCase):
    """Review findings 5, 9 and 10: what the tables show for a class never predicted, failed cases and zero-ms stages."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_dir, self.dataset = mini_run(self.root)
        self.manifest, self.metrics = score_subset(self.run_dir, None, self.dataset)

    def test_a_class_with_support_that_was_never_predicted_is_zero_not_a_dash(self):
        per_class = self.metrics['coverage']['per_class']
        self.assertIsNone(per_class['COVERS_TEXT']['f1'])                         # metrics.prf itself is unchanged
        self.assertEqual((per_class['COVERS_TEXT']['support'], per_class['COVERS_TEXT']['tp']), (1, 0))
        self.assertEqual([derived(self.metrics, f'coverage_{label}_f1') for label in ('covers_text', 'partial', 'conflict', 'no_evidence')],
                         [0.0, 0.0, 1.0, None])                                   # a dash only without support and predictions
        self.assertIn('| Coverage F1 COVERS_TEXT / PARTIAL / CONFLICT / NO_EVIDENCE | 0.0% / 0.0% / 100.0% / — |',
                      render_markdown(self.manifest, self.metrics, []))
        self.assertIn('| Coverage F1 COVERS_TEXT | 0.0% |', compare_markdown([('x', self.manifest, self.metrics)]))
        self.assertIn('| Coverage F1 COVERS_TEXT | 0.0% | 0.0% | +0.0 pp |', version_diff_markdown(self.metrics, self.metrics, 'a', 'b'))

    def test_failed_cases_are_not_counted_as_scored_and_the_tables_warn(self):
        failed = {**self.metrics, 'errors': [{'case_id': 'S03', 'error': 'RuntimeError: analyze failed'}]}
        self.assertEqual(cases_of(self.manifest, failed), '3 run, 2 scored')
        self.assertEqual(scored_cases(self.manifest, failed), ('S01', 'S02'))
        whole = {**self.metrics, 'subset': None, 'completed_cases': 3}
        self.assertEqual(cases_of(self.manifest, {**whole, 'errors': failed['errors']}), '3 run, 2 scored')
        text = tradeoff_markdown([('A', self.manifest, self.metrics), ('B', self.manifest, failed)])
        self.assertIn('scored on different case sets', text)

    def test_a_stage_that_ran_in_under_a_millisecond_counts_as_run(self):
        timings = {'aggregation': 0, 'conflict_precheck': 1, 'fast_classifier': 14982, 'relevance_screen': 1534, 'retrieval': 0}   # smoke C12
        stages = m.obligation_stages([{'coverage_assessed': True, 'timings_ms': timings}, {'coverage_assessed': True, 'timings_ms': {'retrieval': 3}}])
        self.assertEqual((stages['stages']['retrieval']['obligations'], stages['stages']['aggregation']['obligations']), (2, 1))


class OverlayAndAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(writable, temp.name)
        self.root = Path(temp.name)

    def test_the_overlay_strict_mode_and_its_audit_reach_the_metrics_and_the_report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir, _, _ = run(DATASET, self.root / 'runs', provider='rules', retrieval='lexical', cases=['C08'], label='ov')
        ledger = Ledger(DATASET, self.root / 'reviewed_labels')
        entry = ledger.propose([{'case_id': 'C08', 'article': '6', 'clause': '1', 'field': 'coverage', 'to': 'PARTIAL', 'reason': 'phone number'}],
                               'Analyst A')[0]
        ledger.approve([entry['entry_id']], 'Approver B')
        _, _, metrics = rescore(run_dir, reviewed_labels=ledger.reviewed_path)
        self.assertEqual((metrics['reviewed_labels']['strict'], metrics['reviewed_labels']['audit']['problems']), (True, []))
        self.assertIn('strict: only entries the audit log approved', (run_dir / 'report.md').read_text(encoding='utf-8'))
        _, subset = score_subset(run_dir, reviewed_labels=ledger.reviewed_path)
        self.assertEqual(subset['reviewed_labels']['strict'], True)

    def test_the_label_audit_looks_disputes_up_in_the_runs_own_dataset(self):
        data = json.loads(json.dumps(MINI).replace('"S01"', '"C12"'))          # an independent dataset may have a C12 md.5(2) too
        rows = {('C12' if k == 'S01' else k): v for k, v in ROWS.items()}
        run_dir, dataset_path = mini_run(self.root, data=data, rows=rows)
        record = audit([run_dir], dataset_path)
        self.assertFalse([r for item in record['items'] for r in item['reasons'] if 'isputed' in r or 'DISPUTED' in r], record['items'])
        self.assertIsNotNone(disputed('C12', '5', '(2)'))                         # the tr-aml-v1 dispute itself still stands


if __name__ == '__main__':
    unittest.main()

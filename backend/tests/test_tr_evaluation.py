"""BEVERAGE_TR_DEV_V1: development contrast cases scored as INDICATIVE, never as expert gold."""
import json
import unittest
from pathlib import Path

from regchain.evaluation.tasks import DEVELOPMENT, pair_consistency, task_report
from regchain.tr.packs import DATA, Registry
from regchain.tr.profile import load_pilot_profiles
from regchain.tr.routing import evaluate_scope, resolve, rollup

REGISTRY = Registry.load()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
DATASET = json.loads((DATA / 'evaluation' / 'beverage_tr_dev_v1.json').read_text(encoding='utf-8'))


def outcome(case_side, default_scope):
    profile = PROFILES[case_side.get('profile') or case_side.get('representative_type')]
    scope_id = case_side.get('scope_id') or default_scope
    scope = REGISTRY.scopes[scope_id]
    target_id = case_side.get('target_id')
    if target_id:
        return next(d.status for d in evaluate_scope(scope, profile, REGISTRY) if d.target_id == target_id)
    group = next(d for d in rollup(scope, evaluate_scope(scope, profile, REGISTRY), profile, REGISTRY) if d.level == 'GROUP')
    return group.status


class DevDatasetTests(unittest.TestCase):
    def test_the_dataset_is_development_labelled_and_below_the_expert_target(self):
        self.assertEqual(DATASET['label_status'], 'DEVELOPMENT')
        self.assertLess(len(DATASET['cases']), DATASET['target_per_task'])

    def test_development_scores_are_indicative_and_insufficient(self):
        rows = []
        for case in DATASET['cases']:
            for side in case['pair']:
                side = {'profile': case.get('profile'), **side}
                rows.append((side['expected'], outcome(side, case['scope_id'])))
        report = task_report('APPLICABILITY', rows, ('APPLIES', 'PARTIAL', 'DOES_NOT_APPLY', 'UNKNOWN'),
                             target=100, label_status=DEVELOPMENT)
        self.assertEqual(report['sample_status'], 'INSUFFICIENT_SAMPLE')
        self.assertEqual(report['claim'], 'INDICATIVE')
        self.assertEqual(report['n'], len(rows))
        self.assertGreaterEqual(report['accuracy'], 1.0)

    def test_contrast_pairs_are_consistent_on_the_dev_set(self):
        pairs = []
        for case in DATASET['cases']:
            observed = []
            for side in case['pair']:
                side = {'profile': case.get('profile'), **side}
                observed.append((side['expected'], outcome(side, case['scope_id'])))
            pairs.append(tuple(observed))
        result = pair_consistency(pairs)
        self.assertEqual(result['n'], len(DATASET['cases']))
        self.assertEqual(result['consistency'], 1.0)
        self.assertEqual(result['contrast_detected'], result['contrast_pairs'])

    def test_the_dataset_file_is_the_committed_path(self):
        self.assertTrue((Path(DATA) / 'evaluation' / 'beverage_tr_dev_v1.json').is_file())


if __name__ == '__main__':
    unittest.main()

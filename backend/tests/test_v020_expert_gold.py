"""Framework validation uses temporary synthetic records, never stored as gold."""
import copy
import hashlib
import unittest

from pydantic import ValidationError

from regchain.evaluation.expert_gold import (AmbiguityItem, ExpertExample, ExpertGoldRelease, GoldLabels, TASKS,
                                           annotation_hash, metric_gold_view, verify_source_texts)


SOURCE = 'Banks report.'
SOURCE_HASH = hashlib.sha256(SOURCE.encode()).hexdigest()


def empty_labels():
    return {name: None for name in GoldLabels.model_fields}


def fixture_example():
    labels = {**empty_labels(), 'grounding': 'SUPPORTED'}
    return {'example_id': 'temporary-test', 'independence_group': 'temporary-group', 'tasks': ['GROUNDING'],
            'regulation': 'synthetic', 'article': '1', 'subclause': None,
            'source_span': {'source_id': 'source', 'source_sha256': SOURCE_HASH, 'start': 0, 'end': len(SOURCE), 'quote': SOURCE},
            'comparison_source_span': None, 'company_profile_sha256': None, 'policy_package_sha256': None,
            'development_exposure': 'NOT_EXPOSED', 'labels': labels,
            'reviews': [{'reviewer_id': reviewer, 'expertise': 'synthetic test expert', 'reviewed_at': 'test-only',
                         'independent_of_development': True, 'blind_to_model_predictions': True,
                         'labels_sha256': annotation_hash(labels)} for reviewer in ('reviewer-a', 'reviewer-b')], 'adjudication': None}


def draft():
    return {'format': 'CARDAMAN_LOCKED_EXPERT_GOLD_V1', 'status': 'DRAFT_FRAMEWORK', 'version': '1.0.0-framework',
            'targets_per_task': dict.fromkeys(TASKS, 100), 'source_sha256': {}, 'examples': [], 'frozen_at': None}


class ExpertFrameworkTests(unittest.TestCase):
    def test_framework_is_zero_label_and_cannot_be_scored(self):
        release = ExpertGoldRelease.model_validate(draft())
        with self.assertRaises(ValueError):
            verify_source_texts(release, {})
        with self.assertRaises(ValidationError):
            ExpertGoldRelease.model_validate({**draft(), 'examples': [fixture_example()]})

    def test_every_critical_task_requires_at_least_one_hundred_independent_groups(self):
        for targets in ({'GROUNDING': 100}, {**dict.fromkeys(TASKS, 100), 'CONFLICT': 99}):
            with self.assertRaises(ValidationError):
                ExpertGoldRelease.model_validate({**draft(), 'targets_per_task': targets})
        with self.assertRaises(ValidationError):
            ExpertGoldRelease.model_validate({**draft(), 'status': 'FROZEN', 'frozen_at': 'test-only', 'examples': [fixture_example()]})

    def test_exposed_or_unreviewed_or_disputed_examples_cannot_be_frozen(self):
        base = fixture_example()
        self.assertTrue(ExpertExample.model_validate(base).reviewed_and_unexposed())
        variants = [{**base, 'reviews': base['reviews'][:1]}, {**base, 'development_exposure': 'EXPOSED'},
                    {**base, 'adjudication': {'status': 'ADJUDICATION_REQUIRED', 'reason': 'test disagreement', 'adjudicator_id': None, 'adjudicated_at': None}}]
        disputed = copy.deepcopy(base)
        disputed['reviews'][1]['labels_sha256'] = '0' * 64
        variants.append(disputed)
        for record in variants:
            self.assertFalse(ExpertExample.model_validate(record).reviewed_and_unexposed())

    def test_accepted_adjudication_must_be_bound_to_labels_and_independent_third_expert(self):
        record = fixture_example()
        record['adjudication'] = {'status': 'ADJUDICATED', 'reason': 'test-only', 'adjudicator_id': 'expert-c',
                                   'adjudicated_at': 'test-only', 'accepted_labels_sha256': annotation_hash(record['labels'])}
        self.assertTrue(ExpertExample.model_validate(record).reviewed_and_unexposed())
        record['adjudication']['adjudicator_id'] = 'reviewer-a'
        self.assertFalse(ExpertExample.model_validate(record).reviewed_and_unexposed())

    def test_null_unassessed_is_distinct_from_assessed_absence(self):
        record = fixture_example()
        record['tasks'] = ['EXTRACTION']
        with self.assertRaises(ValidationError):
            ExpertExample.model_validate(record)
        for field in ('actor', 'action', 'object', 'condition', 'exception', 'deadline', 'threshold'):
            record['labels'][field] = []
        ExpertExample.model_validate(record)

    def test_span_offsets_and_task_context_are_enforced(self):
        record = fixture_example()
        record['source_span']['end'] += 1
        with self.assertRaises(ValidationError):
            ExpertExample.model_validate(record)
        record = fixture_example()
        record['tasks'] = ['FINANCIAL_CHANGE']
        record['labels']['change_type'] = []
        with self.assertRaises(ValidationError):
            ExpertExample.model_validate(record)

    def test_pending_registry_cannot_smuggle_replacement_labels(self):
        row = {'dataset_path': 'frozen.json', 'dataset_sha256': '0' * 64, 'case_id': 'case', 'obligation_key': '1(1)',
               'status': 'ADJUDICATION_REQUIRED', 'reason': 'requires expert resolution'}
        AmbiguityItem.model_validate(row)
        with self.assertRaises(ValidationError):
            AmbiguityItem.model_validate({**row, 'new_label': 'COVERS_TEXT'})

    def test_strict_and_adjudicated_views_never_edit_or_filter_baseline(self):
        strict = {'case:1': {'coverage': 'CONFLICT'}, 'case:2': {'coverage': 'PARTIAL'}}
        original = copy.deepcopy(strict)
        pending = [{'status': 'ADJUDICATION_REQUIRED'}]
        self.assertEqual(metric_gold_view(strict, mode='STRICT_GOLD_METRICS', adjudications=pending)['labels'], strict)
        with self.assertRaises(ValueError):
            metric_gold_view(strict, mode='ADJUDICATED_METRICS', adjudications=pending)
        label = {'coverage': 'PARTIAL'}
        overlay = {'key': 'case:1', 'original_label_sha256': annotation_hash(strict['case:1']), 'accepted_label': label,
                   'accepted_label_sha256': annotation_hash(label), 'status': 'ADJUDICATED', 'adjudicator_id': 'expert-c', 'adjudicated_at': 'test-only'}
        result = metric_gold_view(strict, mode='ADJUDICATED_METRICS', adjudications=[overlay])
        self.assertEqual(result['labels']['case:1'], label)
        self.assertEqual(len(result['labels']), len(strict))
        self.assertEqual(strict, original)
        with self.assertRaises(ValueError):
            metric_gold_view(strict, mode='ADJUDICATED_METRICS', adjudications=[{**overlay, 'original_label_sha256': '0' * 64}])


if __name__ == '__main__':
    unittest.main()

"""Offline contract checks for the frozen development experiment; no inference."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from hardening_prompt_budget import build_plan, change_prediction, execute_cell, output_json_hash
from regchain.extraction import providers


class PromptBudgetPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_plan()

    def test_prompt_arms_have_identical_inputs_and_full_requested_output_room(self):
        probes = {p['id']: p for p in self.plan['probes']}
        rows = [c for c in self.plan['cells'] if c['phase'] == 'prompt']
        self.assertEqual(len(rows), 24)
        for key in {c['probe_id'] for c in rows}:
            group = [c for c in rows if c['probe_id'] == key]
            self.assertEqual(len(group), 3)
            self.assertEqual(len({c['payload_hash'] for c in group}), 1)
            self.assertEqual(len({c['schema_hash'] for c in group}), 1)
            self.assertTrue(all(c['estimated_room'] >= c['requested_output'] for c in group))
            self.assertTrue(all(c['model'] == 'qwen3:8b' and c['num_ctx'] == 16384 for c in group))
            self.assertFalse(any(k.startswith('expected') or k.startswith('gold') for k in probes[key]['payload']))

    def test_budget_cells_are_full_factorial_and_expose_admission_failure(self):
        rows = [c for c in self.plan['cells'] if c['phase'] == 'budget']
        self.assertEqual(len(rows), 18)
        for key in {c['probe_id'] for c in rows}:
            group = [c for c in rows if c['probe_id'] == key]
            self.assertEqual({c['requested_output'] for c in group}, {2048, 4096, 8192})
            self.assertEqual(len({c['payload_hash'] for c in group}), 1)
            self.assertEqual(len({c['prompt_hash'] for c in group}), 1)
        rejected = [c for c in rows if not c['initial_admission_expected']]
        self.assertEqual(len(rejected), 6)
        self.assertTrue(all(c['requested_output'] == 8192 for c in rejected))
        self.assertTrue(all(c['provider_configuration_expected'] is False for c in rejected))
        self.assertTrue(all(c['initial_admission_stage'] == 'PROVIDER_CONFIGURATION' for c in rejected))

    def test_constructor_output_rejection_records_no_inference_for_each_task(self):
        probes = {p['id']: p for p in self.plan['probes']}
        cells = [c for c in self.plan['cells'] if c['phase'] == 'budget' and c['requested_output'] == 8192]
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'development'}, clear=True), \
                patch.object(providers.OllamaProvider, '_request') as transport, \
                patch('regchain.extraction.providers.httpx.Client') as network:
            for cell in cells:
                with self.subTest(task=probes[cell['probe_id']]['task'], probe=cell['probe_id']):
                    probe = probes[cell['probe_id']]
                    row = execute_cell(cell, probe, self.plan['prompts'][probe['task']][cell['variant']],
                                       {'qwen3:4b': 'a' * 64, 'qwen3:8b': 'b' * 64})
                    self.assertEqual(row['failure'], 'CONTEXT_ADMISSION')
                    self.assertEqual(row['admission_stage'], 'PROVIDER_CONFIGURATION')
                    self.assertEqual(row['admission_events'], ['PROVIDER_CONFIGURATION_REJECT'])
                    self.assertEqual((row['requested_output'], row['num_ctx']), (8192, 8192))
                    for field in ('requests', 'calls', 'raw_answers', 'actual_output_caps'):
                        self.assertEqual(row[field], [])
                    for field in ('actual_output_cap', 'first_request_hash', 'first_parsed_result', 'validated_result', 'final_result',
                                  'first_parsed_hash', 'final_hash', 'raw_correct', 'validated_correct', 'final_correct'):
                        self.assertIsNone(row[field])
                    self.assertFalse(row['inference_attempted'])
                    self.assertFalse(row['planned_admission_mismatch'])
                    self.assertEqual((row['runtime_prompt_tokens'], row['runtime_output_tokens'], row['output_truncations']), (0, 0, 0))
        transport.assert_not_called()
        network.assert_not_called()

    def test_unrelated_constructor_error_is_not_reclassified_as_budget_admission(self):
        cell = self.plan['cells'][0]
        probe = next(p for p in self.plan['probes'] if p['id'] == cell['probe_id'])
        with patch.object(providers, 'OllamaProvider', side_effect=ValueError('unrelated configuration error')):
            with self.assertRaisesRegex(ValueError, 'unrelated configuration error'):
                execute_cell(cell, probe, self.plan['prompts'][probe['task']][cell['variant']], {'qwen3:8b': 'b' * 64})

    def test_incomplete_profile_diagnostic_cannot_be_mistaken_for_production_call(self):
        probe = next(p for p in self.plan['probes'] if p['id'] == 'I09-applicability')
        self.assertEqual(probe['production_guard']['code'], 'PROFILE_INCOMPLETE')
        self.assertIn('MODEL_BOUNDARY_ONLY', probe['guard_bypass'])
        self.assertIsNone(probe['original_input']['company']['licences'])

    def test_malformed_financial_type_list_is_unscored_instead_of_crashing_or_matching_empty_gold(self):
        for value in (None, [], {}, {'change_types': 'TEXT_CHANGED'}, {'change_types': [{}]}):
            self.assertIsNone(change_prediction(value))
        self.assertEqual(change_prediction({'change_types': []}), set())

    def test_output_hash_preserves_finite_numeric_json_and_rejects_nonfinite(self):
        value = {'confidence': 0.75, 'amount': 1.125, 'nested': [0.01]}
        self.assertEqual(output_json_hash(value), output_json_hash(dict(reversed(list(value.items())))))
        self.assertNotEqual(output_json_hash(value), output_json_hash({**value, 'confidence': '0.75'}))
        self.assertEqual(value['confidence'], 0.75)
        for invalid in (float('nan'), float('inf'), -float('inf')):
            self.assertIsNone(output_json_hash({'invalid': invalid}))

    def test_request_link_and_financial_validation_are_exercised_without_a_transport(self):
        probe = next(p for p in self.plan['probes'] if p['id'] == 'change-01')
        cell = next(c for c in self.plan['cells'] if c['probe_id'] == probe['id'] and c['variant'] == 'A_current')
        answer = {'old_requirement': probe['payload']['old_text'], 'new_requirement': probe['payload']['new_text'],
                  'change_types': ['THRESHOLD_CHANGED'], 'affected_actor': None, 'effective_date': None, 'deadline': None,
                  'financial_impact_category': 'Uncertain', 'operational_impact_category': 'Review', 'specific_monetary_impact': None}

        def fixture_request(instance, messages, schema, size, started, predict=None):
            instance.last_calls.append({'done_reason': 'stop', 'elapsed_ms': 1, 'prompt_tokens': 100, 'output_tokens': 40})
            return {'message': {'content': json.dumps(answer)}, 'done_reason': 'stop'}

        with patch.dict(os.environ, {'CARDAMAN_MODE': 'development'}, clear=True), \
                patch.object(providers.OllamaProvider, '_request', fixture_request), \
                patch('regchain.extraction.providers.httpx.Client') as network:
            row = execute_cell(cell, probe, self.plan['prompts']['financial']['A_current'], {'qwen3:8b': 'b' * 64})
        network.assert_not_called()
        self.assertTrue(row['first_request_answer_valid'])
        self.assertTrue(row['first_outbound_payload_matches_frozen'])
        self.assertTrue(row['raw_correct'])
        self.assertEqual(row['final_result'], answer)
        self.assertEqual(row['first_request_hash'], row['raw_answers'][0]['request_hash'])

    def test_length_failure_is_not_a_valid_empty_answer_or_scored_correct(self):
        probe = next(p for p in self.plan['probes'] if p['id'] == 'change-08')
        cell = next(c for c in self.plan['cells'] if c['probe_id'] == probe['id'] and c['variant'] == 'A_current')

        def fixture_request(instance, messages, schema, size, started, predict=None):
            measured = {'done_reason': 'length', 'elapsed_ms': 1, 'prompt_tokens': 100, 'output_tokens': predict}
            instance.last_calls.append(measured)
            instance.last_answer = (measured, '')
            raise providers.failure('fixture truncated', 'output_truncated', 'OUTPUT_TRUNCATED')

        with patch.dict(os.environ, {'CARDAMAN_MODE': 'development'}, clear=True), \
                patch.object(providers.OllamaProvider, '_request', fixture_request):
            row = execute_cell(cell, probe, self.plan['prompts']['financial']['A_current'], {'qwen3:8b': 'b' * 64})
        self.assertFalse(row['first_request_answer_valid'])
        self.assertIsNone(row['raw_correct'])
        self.assertIsNone(row['final_correct'])
        self.assertEqual(row['output_truncations'], 1)
        self.assertEqual(row['retry_calls'], 0)


if __name__ == '__main__':
    unittest.main()

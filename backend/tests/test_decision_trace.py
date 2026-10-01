import json
import unittest

from regchain.pilot.decision_trace import attempt_record, finish_trace, evaluation_outcome, OUTCOME_CODES
from test_v019_t6_conflict_gate import claim, disclosure


class DecisionTraceTests(unittest.TestCase):
    def test_raw_parser_and_final_are_independent_observations(self):
        raw = {'conflict': True, 'contradiction_type': 'NONE', 'relation_if_no_conflict': 'UNRELATED'}
        validated = {**raw, 'conflict': False}
        trace = finish_trace([attempt_record(json.dumps(raw), validated)], {'relation': 'UNRELATED'},
                             notes=[{'code': 'NORMALISED_CONFLICT'}], source_ids=['source-a'])
        self.assertEqual(trace['RAW_MODEL_DECISION']['decision'], 'CONFLICTS')
        self.assertEqual(trace['VALIDATED_MODEL_DECISION']['decision'], 'UNRELATED')
        self.assertEqual(trace['VALIDATED_MODEL_DECISION']['changes']['conflict'], {'raw': True, 'validated': False})
        self.assertEqual(trace['FINAL_PIPELINE_DECISION']['decision'], 'UNRELATED')
        self.assertEqual(trace['correctness'], 'NOT_EVALUATED_NO_GOLD')
        self.assertEqual(evaluation_outcome(trace, 'CONFLICTS'), 'RAW_CORRECT_FINAL_WRONG')
        self.assertIsNone(evaluation_outcome(trace, None))
        validated['conflict'] = True
        self.assertFalse(trace['VALIDATED_MODEL_DECISION']['value']['conflict'])

    def test_failed_last_attempt_does_not_resurrect_previous_claim(self):
        attempts = [attempt_record('{"conflict":true}', status='VALIDATION_REJECTED', error='bad span'),
                    attempt_record('', attempt=2, status='OUTPUT_TRUNCATED', call={'done_reason': 'length'})]
        trace = finish_trace(attempts, {'relation': 'UNCLEAR'})
        self.assertIsNone(trace['RAW_MODEL_DECISION']['decision'])
        self.assertIsNone(evaluation_outcome(trace, 'CONFLICTS'))
        self.assertEqual(trace['attempts'][0]['raw_decision'], 'CONFLICTS')
        self.assertEqual(trace['attempts'][0]['validation_error'], 'bad span')

    def test_all_four_correctness_codes_require_external_gold(self):
        found = set()
        for raw in ('CONFLICTS', 'UNRELATED'):
            for final in ('CONFLICTS', 'UNRELATED'):
                value = {'relation': raw}
                trace = finish_trace([attempt_record(json.dumps(value), value)], {'relation': final})
                found.add(evaluation_outcome(trace, 'CONFLICTS'))
        self.assertEqual(found, set(OUTCOME_CODES))

    def test_production_gate_trace_keeps_rejected_claim(self):
        text = 'Şüpheli işlem bildiriminde bulunulduğu hiç kimseyle paylaşılmaz.'
        relation, _, _, records = claim(disclosure(), (), text, 'PROHIBITED_ACTION_ALLOWED')
        trace = records['p1']['decision_trace']
        self.assertEqual(trace['RAW_MODEL_DECISION']['decision'], 'CONFLICTS')
        self.assertEqual(trace['VALIDATED_MODEL_DECISION']['decision'], 'CONFLICTS')
        self.assertEqual(trace['FINAL_PIPELINE_DECISION']['decision'], relation)
        self.assertTrue(trace['gate_events'])

    def test_no_model_call_and_malformed_response_are_not_model_predictions(self):
        self.assertEqual(finish_trace([], {'relation': 'UNCLEAR'})['RAW_MODEL_DECISION']['status'], 'NOT_CALLED')
        row = attempt_record('{broken', status='VALIDATION_REJECTED')
        self.assertIsNone(row['raw_decision'])
        self.assertIsNone(row['raw'])


if __name__ == '__main__':
    unittest.main()

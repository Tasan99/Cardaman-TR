"""Synthetic structural probes, separate from existing benchmark/gold wording."""
import unittest

from regchain.evidence import canonical_bytes
from regchain.pilot.normative_audit import DIMENSIONS, assess_anchor


def duty(**kwargs):
    return {'subject': 'Banks', 'modality': 'MUST', 'required_action': 'retain the audit files',
            'object': 'audit files', 'conditions': [], 'exceptions': [], **kwargs}


class NormativeAuditTests(unittest.TestCase):
    def test_valid_source_anchor_is_kept(self):
        text = 'Banks must retain the audit files.'
        result = assess_anchor(duty(), text, text)
        self.assertEqual(result['status'], 'MATCH')
        self.assertEqual(set(result['dimensions']), set(DIMENSIONS))
        self.assertEqual(text[result['source_span']['start']:result['source_span']['end']], text)
        self.assertFalse(result['semantic_entailment_verified'])
        canonical_bytes(result)

    def test_wrong_source_is_positive_rejection(self):
        result = assess_anchor(duty(), 'Banks must retain the audit files.', 'Banks review their accounts.')
        self.assertEqual((result['status'], result['reason_code']), ('MISMATCH', 'WRONG_SOURCE'))

    def test_explicit_other_actor_is_positive_rejection(self):
        text = 'Insurers must retain the audit files.'
        result = assess_anchor(duty(), text, text)
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertEqual(result['dimensions']['actor']['status'], 'MISMATCH')

    def test_wrong_sentence_does_not_borrow_adjacent_duty(self):
        span = 'Banks must identify the account holder.'
        passage = 'Banks must retain the audit files. ' + span
        result = assess_anchor(duty(), span, passage)
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertFalse(result['dependency']['used'])

    def test_equivalent_normalized_action_representation_is_kept(self):
        span = 'Banks must keep the audit files.'
        result = assess_anchor(duty(), span, span)
        self.assertEqual(result['status'], 'MATCH')

    def test_explicit_backward_reference_keeps_source_bound_dependency(self):
        span = 'These records must be retained for seven years.'
        passage = 'Banks maintain audit files as records. ' + span
        result = assess_anchor(duty(), span, passage)
        self.assertEqual(result['status'], 'MATCH')
        self.assertTrue(result['dependency']['used'])
        self.assertEqual(len(result['context_spans']), 2)
        for source in result['context_spans']:
            self.assertEqual(passage[source['start']:source['end']], source['text'])

    def test_empty_source_never_matches(self):
        for span, passage in (('', ''), ('Banks retain files.', ''), (' ', ' ')):
            self.assertEqual(assess_anchor(duty(), span, passage)['status'], 'UNRESOLVED')

    def test_unknown_action_with_object_overlap_is_not_rejected(self):
        span = 'Banks must tokenize the audit files.'
        result = assess_anchor(duty(required_action='index the audit files'), span, span)
        self.assertEqual(result['status'], 'MATCH')
        self.assertEqual(result['dimensions']['object']['status'], 'MATCH')

    def test_unknown_action_without_object_evidence_remains_unresolved(self):
        span = 'Banks must calibrate the equipment.'
        result = assess_anchor(duty(required_action='index the audit files'), span, span)
        self.assertEqual(result['status'], 'UNRESOLVED')

    def test_negative_qualifier_remains_unresolved(self):
        span = 'Banks must not necessarily retain the audit files.'
        result = assess_anchor(duty(), span, span)
        self.assertEqual((result['status'], result['reason_code']), ('UNRESOLVED', 'NEGATIVE_QUALIFIER_UNRESOLVED'))

    def test_distinct_explicit_object_identifiers_are_positive_mismatch(self):
        span = 'Banks must retain file "archive-B".'
        result = assess_anchor(duty(required_action='retain file "archive-A"', object='file "archive-A"'), span, span)
        self.assertEqual(result['status'], 'MISMATCH')
        self.assertEqual(result['dimensions']['object']['status'], 'MISMATCH')

    def test_repeated_quote_does_not_guess_which_sentence_was_cited(self):
        span = 'Banks must retain the audit files.'
        self.assertEqual(assess_anchor(duty(), span, span + ' ' + span)['reason_code'], 'AMBIGUOUS_SPAN_LOCATION')


if __name__ == '__main__':
    unittest.main()

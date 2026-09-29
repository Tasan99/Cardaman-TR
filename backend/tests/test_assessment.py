import copy
import unittest

from regchain.assessment import Assessment, assess
from regchain.evidence import verify_chain


def fixture():
    return {'format': 'regchain-review-input-v1', 'synthetic': True, 'company_id': 'demo',
            'company_profile': {'activity': 'synthetic lending'},
            'policies': [{'id': 'p1', 'version': '1', 'text': 'We retain all records.'}],
            'obligations': [{'id': 'o1', 'source_version': 'synthetic-v2',
                             'source_quote': 'A firm must retain records.', 'action': 'retain records.'}]}


class AssessmentTests(unittest.TestCase):
    def test_unreviewed_candidate_never_becomes_coverage(self):
        data = fixture()
        data['obligations'][0].update(applicability='APPLIES', coverage='COVERED',
            policy_evidence=[{'policy_id': 'p1', 'quote': 'retain all records.'}])
        result = assess(Assessment.model_validate(data))
        self.assertEqual(result['events'][0]['payload']['results'][0]['status'], 'EXTRACTION_REVIEW_REQUIRED')

    def test_reviewed_missing_policy_is_a_gap(self):
        data = fixture()
        review = {'reviewer': 'Synthetic reviewer', 'rationale': 'Synthetic fixture decision'}
        data['obligations'][0].update(extraction_review=review, applicability='APPLIES',
            applicability_review=review, coverage='MISSING', coverage_review=review)
        result = assess(Assessment.model_validate(data))
        self.assertEqual(result['events'][0]['payload']['results'][0]['status'], 'REVIEWED_POLICY_GAP')

    def test_unknown_policy_and_invented_quote_are_rejected(self):
        for citation in [{'policy_id': 'missing', 'quote': 'records'}, {'policy_id': 'p1', 'quote': 'invented'}]:
            data = fixture()
            data['obligations'][0]['policy_evidence'] = [citation]
            with self.assertRaisesRegex(ValueError, 'citation'):
                assess(Assessment.model_validate(data))

    def test_coverage_without_evidence_is_rejected(self):
        data = fixture()
        data['obligations'][0]['coverage'] = 'COVERED'
        with self.assertRaisesRegex(ValueError, 'requires policy evidence'):
            assess(Assessment.model_validate(data))

    def test_duplicate_obligation_ids_rejected(self):
        data = fixture()
        data['obligations'] *= 2
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            Assessment.model_validate(data)

    def test_source_change_and_snapshot_tampering(self):
        data = fixture()
        data['obligations'][0]['previous_source_quote'] = 'A firm should retain records.'
        packet = assess(Assessment.model_validate(data))
        self.assertEqual(packet['events'][0]['payload']['results'][0]['change'], 'SOURCE_CHANGED')
        self.assertTrue(verify_chain(packet['events'], packet['head'], 1))
        modified = copy.deepcopy(packet['events'])
        modified[0]['payload']['input']['company_id'] = 'another-company'
        self.assertFalse(verify_chain(modified, packet['head'], 1))
        self.assertFalse(verify_chain([], packet['head'], 1))

"""Source authority notes and allowlisted official-source records (Cardaman TR steps 10–11)."""
import unittest
from datetime import date, datetime, timezone

from pydantic import ValidationError

from regchain.tr.adapters import INSTITUTIONS, OfficialSource, institution_of
from regchain.tr.packs import Registry
from regchain.tr.profile import load_profile
from regchain.tr.routing import resolve
from test_tr_profile import sample

REGISTRY = Registry.load()


class AuthorityTests(unittest.TestCase):
    def test_a_run_notes_that_precedent_was_not_evaluated(self):
        result = resolve(load_profile(sample(), REGISTRY.vocabulary), REGISTRY)
        self.assertIn('DECISION_PRECEDENT_NOT_EVALUATED', result.authority_notes)
        self.assertTrue(all('DECISION_PRECEDENT' not in REGISTRY.pack(s.pack_id).source_layers
                            for s in result.selections))

    def test_guidance_never_creates_an_obligation_on_its_own(self):
        result = resolve(load_profile(sample(), REGISTRY.vocabulary), REGISTRY)
        for decision in result.decisions:
            if decision.binding_status != 'BINDING':
                self.assertFalse(decision.creates_obligation)
                self.assertIn('GUIDANCE_ONLY', decision.reason_codes)

    def test_every_decision_keeps_raw_parsed_validator_and_final_apart(self):
        result = resolve(load_profile(sample(), REGISTRY.vocabulary), REGISTRY)
        decision = result.decisions[0]
        self.assertIsNone(decision.audit.raw)
        self.assertIsNone(decision.audit.parsed)
        self.assertIsNone(decision.audit.validator)
        self.assertEqual(decision.audit.final['status'], decision.status)
        self.assertEqual(decision.audit.final['source']['regulation_id'], decision.regulation_id)
        self.assertIn('target_id', decision.audit.final['target'])


class AdapterTests(unittest.TestCase):
    def record(self, **fields):
        base = dict(institution_id='TARIM_ORMAN', authority='OFFICIAL_GUIDANCE',
                    document_url='https://www.tarimorman.gov.tr/synthetic/label-guide',
                    canonical_id='SYN-GUIDE-1', publication_date=date(2026, 1, 1),
                    version='0.1.0', content_hash='a' * 64,
                    retrieved_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
                    document_type='REHBER', text='Sentetik test metni; gerçek bir resmi metin değildir.',
                    synthetic=True)
        return OfficialSource.model_validate(base | fields)

    def test_an_allowlisted_synthetic_record_loads(self):
        source = self.record()
        self.assertTrue(source.synthetic)
        self.assertEqual(institution_of(source.document_url).institution_id, 'TARIM_ORMAN')
        self.assertIn('Sentetik', source.text)

    def test_an_unknown_host_is_refused(self):
        with self.assertRaises(ValidationError):
            self.record(document_url='https://example.com/law')

    def test_an_unknown_institution_is_refused(self):
        with self.assertRaises(ValidationError):
            self.record(institution_id='UNKNOWN_BODY')

    def test_precedent_is_a_decision_not_a_statute(self):
        with self.assertRaises(ValidationError):
            self.record(institution_id='KVKK', authority='DECISION_PRECEDENT',
                        document_url='https://www.kvkk.gov.tr/synthetic/decision', document_type='KANUN')
        source = self.record(institution_id='KVKK', authority='DECISION_PRECEDENT',
                             document_url='https://www.kvkk.gov.tr/synthetic/decision', document_type='KARAR')
        self.assertEqual(source.authority, 'DECISION_PRECEDENT')

    def test_the_allowlist_covers_the_named_official_bodies(self):
        self.assertGreaterEqual(set(INSTITUTIONS),
                                {'TARIM_ORMAN', 'TICARET', 'KVKK', 'CEVRE_SEHIRCILIK', 'SAGLIK', 'MASAK',
                                 'MEVZUAT', 'RESMI_GAZETE'})


if __name__ == '__main__':
    unittest.main()

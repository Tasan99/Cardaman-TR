"""The decision layer (tr/decisions.py): a board decision bound to its addressee and to no other entity, read by the
rule reader, grounded in its stored text, and never an automatic decision. The decision here is SYNTHETIC: wording
in the style of a commitment text, written for the test, not a real decision."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from regchain.tr.clauses import split_clauses
from regchain.tr.compare import load_register
from regchain.tr.core import RegulationMeta
from regchain.tr.corpus import CorpusStore
from regchain.tr.decisions import (RECORD_FORMAT, DecisionError, DecisionRecord, DecisionStore, assess_decision, commitment_sections,
                                   decision_applicability, decision_obligations, text_hash)
from regchain.tr.frames import frame_of
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles

REGISTRY = Registry.load()
STORE = CorpusStore()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
DECISION_ID = 'TR:KURUL_KARARI:SENTETIK_TAAHHUT_1'
TEXT = '''Sentetik taahhüt metni (test). Teşebbüs aşağıdaki taahhütleri sunmaktadır.

1. Teşebbüs, satış noktalarına yerleştirdiği soğutucuların en az yüzde otuz beşini rakip teşebbüslerin ürünlerine açık tutacaktır.
2. Teşebbüs, soğutucu tahsisini satış noktasının asgari alım taahhüdüne bağlamayacaktır.
3. Teşebbüs, taahhütlerin uygulanmasına ilişkin raporu her yıl Kuruma sunmayı taahhüt eder.
4. Teşebbüs, satış noktalarına münhasırlık koşulu uygulamayacağını taahhüt eder. Ancak, 35 metrekareden küçük satış
noktalarında bu taahhüt uygulanmaz.
5. Taahhütler, kararın tebliğinden itibaren üç yıl sonra Kurul tarafından gözden geçirilir.
'''


def record_for(text: str, synthetic: bool = True) -> DecisionRecord:
    return DecisionRecord(format=RECORD_FORMAT, decision_id=DECISION_ID, institution_id='REKABET', title='Sentetik taahhüt kararı (test)',
                          decision_no='26-TEST/1', document_url='https://www.rekabet.gov.tr/Dosya/sentetik-taahhut.pdf',
                          legal_basis='test', sector_tags=['BEVERAGE'], content_hash=text_hash(text),
                          retrieved_at=datetime(2026, 10, 2, tzinfo=timezone.utc), synthetic=synthetic, disclaimer='Sentetik test metni.')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = DecisionStore(Path(self.directory.name))
        self.store.write(record_for(TEXT), TEXT)

    def tearDown(self):
        self.directory.cleanup()

    def test_the_record_is_allowlisted_and_the_text_is_hash_checked(self):
        with self.assertRaises(ValueError):                       # a host the institution does not publish on
            record_for(TEXT).model_copy(update={'document_url': 'https://example.com/karar.pdf'}).model_validate(
                record_for(TEXT).model_dump() | {'document_url': 'https://example.com/karar.pdf'})
        with self.assertRaises(ValueError):                       # a ministry is not a decision-making authority
            DecisionRecord.model_validate(record_for(TEXT).model_dump() | {'institution_id': 'TICARET',
                                                                          'document_url': 'https://www.ticaret.gov.tr/x.pdf'})
        self.assertEqual(self.store.decision_ids(), [DECISION_ID])
        path = next(Path(self.directory.name).glob('*/text.txt'))
        path.write_text(TEXT + ' değişti', encoding='utf-8')
        with self.assertRaisesRegex(DecisionError, 'hash'):
            DecisionStore(Path(self.directory.name)).record(DECISION_ID)

    def test_a_synthetic_decision_never_enters_the_packaged_root(self):
        from regchain.tr.decisions import DECISIONS
        with self.assertRaisesRegex(DecisionError, 'synthetic'):
            DecisionStore(DECISIONS).write(record_for(TEXT), TEXT)

    def test_the_numbered_items_are_sections_that_are_exact_spans_of_their_text(self):
        sections = self.store.sections(DECISION_ID)
        self.assertEqual([s['printed_label'] for s in sections], [f'Karar 26-TEST/1 taahhüt {n}' for n in range(1, 6)])
        self.assertTrue(sections[0]['text'].startswith('Teşebbüs, satış noktalarına'))
        self.assertIn('\n', sections[3]['text'])                  # the wrapped line stays with its item
        for section in sections:
            for line in section['lines']:
                self.assertTrue(section['text'][line['start']:line['end']].strip())
            for clause in split_clauses(section):
                self.assertEqual(section['text'][clause.start:clause.end], clause.text)
        # "35 metrekareden" at the start of a wrapped line does not open item 35
        self.assertEqual(len(commitment_sections('1. Bir.\n35 metre.\n2. İki.', 'K')), 2)
        self.assertEqual(commitment_sections('Giriş.\n\nİkinci paragraf.', 'K')[1]['printed_label'], 'K paragraf 2')


class ReadingTests(unittest.TestCase):
    def test_commitment_wording_is_read_as_a_duty(self):
        from regchain.tr.clauses import Clause

        def modality(text):
            clause = Clause(ref='K/f.1', label='K', fikra=1, start=0, end=len(text), text=text, reading=text)
            frame = frame_of(clause, DECISION_ID)
            return frame.kind, frame.modality
        self.assertEqual(modality('Teşebbüs, soğutucuların yüzde otuz beşini rakip ürünlere açık tutacaktır.'), ('OBLIGATION', 'MUST'))
        self.assertEqual(modality('Teşebbüs, soğutucu tahsisini asgari alım taahhüdüne bağlamayacaktır.'), ('PROHIBITION', 'MUST_NOT'))
        self.assertEqual(modality('Teşebbüs, raporu her yıl Kuruma sunmayı taahhüt eder.'), ('OBLIGATION', 'MUST'))
        self.assertEqual(modality('Teşebbüs, münhasırlık koşulu uygulamayacağını taahhüt eder.'), ('PROHIBITION', 'MUST_NOT'))
        self.assertEqual(modality('Bu Yönetmelik yayımı tarihinde yürürlüğe girer.')[1], None)

    def test_a_decision_interprets_no_catalogued_text_but_guidance_still_must(self):
        fields = dict(title='x', jurisdiction='TR', regulator='REKABET', number='1', effective_date=None, effective_status='UNKNOWN',
                      metadata_status='UNVERIFIED', verification_note='test', sector_tags=['BEVERAGE'])
        RegulationMeta(regulation_id='TR:KURUL_KARARI:X', regulation_type='KURUL_KARARI', binding_status='DECISION_PRECEDENT', **fields)
        with self.assertRaisesRegex(ValueError, 'interprets'):
            RegulationMeta(regulation_id='TR:RESMI_REHBER:X', regulation_type='RESMI_REHBER', binding_status='OFFICIAL_GUIDANCE', **fields)


class DecisionLayerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.decisions = DecisionStore(Path(self.directory.name))
        self.decisions.write(record_for(TEXT), TEXT)
        self.record = self.decisions.record(DECISION_ID)
        self.profile = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']

    def tearDown(self):
        self.directory.cleanup()

    def test_the_duties_are_read_by_the_rule_reader_and_bound_to_the_legal_entity(self):
        frames, obligations = decision_obligations(self.record, self.decisions, REGISTRY)
        by_ref = {o.provision_ref: o for o in obligations}
        self.assertEqual(by_ref['Karar 26-TEST/1 taahhüt 1/f.1'].modality, 'MUST')
        self.assertEqual(by_ref['Karar 26-TEST/1 taahhüt 2/f.1'].modality, 'MUST_NOT')
        self.assertEqual(by_ref['Karar 26-TEST/1 taahhüt 3/f.1'].modality, 'MUST')
        fourth = by_ref['Karar 26-TEST/1 taahhüt 4/f.1/c.1']
        self.assertEqual(fourth.modality, 'MUST_NOT')
        self.assertTrue(fourth.frame.exceptions, 'the "Ancak ..." sentence lifts the commitment for small outlets')
        self.assertNotIn('Karar 26-TEST/1 taahhüt 5/f.1', by_ref)   # the Board's review is no duty of the undertaking
        for obligation in obligations:
            self.assertEqual((obligation.scope.level, obligation.scope.scope_status), ('LEGAL_ENTITY', 'DEFINED'))
            self.assertIn('DECISION_ADDRESSEE', obligation.flags)
            self.assertEqual(obligation.scope.constraints(), {})
            self.assertEqual(obligation.version_id, self.record.version_id)

    def test_the_decision_binds_its_addressee_and_no_other_entity(self):
        pairs = decision_applicability(self.record, self.profile, REGISTRY, STORE, self.decisions, named=['NONALC-SALES'])
        self.assertTrue(pairs)
        for obligation, decision in pairs:
            gate = next(g for g in decision.gates if g['gate'] == 'DECISION_ADDRESSEE')
            if decision.target_id == 'NONALC-SALES':
                self.assertEqual((decision.status, gate['status']), ('APPLIES', 'MATCH'), obligation.provision_ref)
                self.assertIn('DECISION_ADDRESSEE', decision.reason_codes)
                self.assertTrue(decision.creates_obligation)
                self.assertNotIn('GUIDANCE_ONLY', decision.reason_codes)
            else:
                self.assertEqual((decision.status, gate['status']), ('DOES_NOT_APPLY', 'MISMATCH'), decision.target_id)
                self.assertEqual(decision.reason_codes, ['DECISION_ADDRESSEE_MISMATCH'])
            self.assertTrue(decision.audit.validator['grounded'], obligation.provision_ref)
        # the profile itself may name the addressee
        entities = [e.model_copy(update={'bound_by_decisions': [DECISION_ID]}) if e.entity_id == 'NONALC-DISTRIBUTION' else e
                    for e in self.profile.legal_entities]
        stated = self.profile.model_copy(update={'legal_entities': entities})
        applies = {d.target_id for _, d in decision_applicability(self.record, stated, REGISTRY, STORE, self.decisions) if d.status == 'APPLIES'}
        self.assertEqual(applies, {'NONALC-DISTRIBUTION'})
        with self.assertRaisesRegex(DecisionError, 'no legal entity'):
            decision_applicability(self.record, self.profile, REGISTRY, STORE, self.decisions, named=['NOBODY'])

    def test_every_row_goes_to_a_person_with_its_evidence_verified(self):
        register = load_register(self.profile.profile_id)
        report, assessments, run = assess_decision(self.record, self.profile, REGISTRY, STORE, self.decisions, register, named=['NONALC-SALES'])
        self.assertEqual({r.target_id for r in report.rows}, {'NONALC-SALES'})
        self.assertEqual(len(report.rows), run.duties)
        self.assertEqual((run.items, run.synthetic, run.addressees), (5, True, ['NONALC-SALES']))
        for assessment in assessments:
            self.assertEqual(assessment.decision, 'REVIEW_REQUIRED')
            self.assertEqual(assessment.review_reasons[0], 'DECISION_PRECEDENT_REVIEW')
            self.assertEqual(assessment.applicability_basis, 'DECISION_ADDRESSEE')
            self.assertFalse([r for r in assessment.review_reasons if r.startswith('NOT_VERIFIED')], assessment.review_reasons)
        self.assertEqual(report.summary['review_required'], len(report.rows))
        self.assertTrue(all(row.review_required for row in report.rows))
        json.dumps(run.model_dump(mode='json'))                   # the run record is serialisable as the CLI prints it


if __name__ == '__main__':
    unittest.main()

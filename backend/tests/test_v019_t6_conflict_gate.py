"""v0.19 t6 (25 September 2026): the conflict gate. A verifier's contradiction stands only when one clause of the
contradicting sentence names the duty's own act, for the duty's own counterparty and scope, with an incompatible
normative effect on it (conflict.conflict_gate, run after the t5 rules in engine.settle_v19).

Measured on the v019t5 micro run: both false CONFLICT rows passed every earlier rule because no rule asked whether the
contradiction applies to the duty's act. Kanun md. 4(2): a descriptive sentence about MASAK, with no act and no polarity,
anchored only on the authority's name and a light verb of the sentence before it; Tedbirler md. 31(1): "Müşterilerimizden
kimlik belgesi istenmez" (asking customers for identity papers) against the duty to give the authority what it asks for,
anchored on the object noun "belge" and kept as an EXEMPTION_ADDED because waiver wording stood somewhere in the
sentence. Every model here is a fake answering by schema (tests/test_v019_conflict_pipeline.V19); the Tedbirler duties are
cut from the retained snapshot, the Kanun duties are hand-built as the judge reads them.
"""
import unittest

from regchain.evidence import canonical_bytes
from regchain.pilot import conflict as precheck
from regchain.pilot.engine import restates_prohibition
from test_v019_conflict_pipeline import (FIVE_YEARS, KYC_SKIP, KYC_SKIP_SPAN, MONTH_END, V19, fast, md5_2, md28, md46, verdict)
from test_v019_t5_engine import BRANCH, DECLARATION, DECLARATION_SPAN, OUTGOING_SKIP, codes, disclosure, gate, md24, md24a, plain

GATE = set(precheck.GATE_CODES)
# Kanun 5549 md. 4(1) as the judge reads it: report to the authority.
REPORT = plain('bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi')
# Kanun 5549 md. 7(1)-like / Tedbirler md. 31(1)-like: give the authority what it asks for.
GIVE = {'subject': 'Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar', 'modality': 'MUST',
        'required_action': ('Başkanlık ve denetim elemanları tarafından istenilecek her türlü bilgi ve belgeyi gecikmeksizin vermek ve gerekli '
                            'kolaylığı sağlamak'),
        'prohibited_action': None, 'conditions': [], 'exceptions': [],
        'elements': [{'id': 'act_1', 'kind': 'action', 'text': 'Başkanlık ve denetim elemanları tarafından istenilecek her türlü bilgi ve '
                                                                 'belgeyi gecikmeksizin vermek'},
                     {'id': 'act_2', 'kind': 'action', 'text': 'gerekli kolaylığı sağlamak'},
                     {'id': 'subject', 'kind': 'subject', 'text': 'Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği '
                                                                  'olmayan kuruluşlar'}]}
# Tedbirler md. 29(1)-like: give no one information that a report was made ("veremezler": the ability negative).
GIVE_NONE = {'subject': 'Yükümlüler', 'modality': 'MUST_NOT', 'required_action': None,
             'prohibited_action': 'işleme taraf olanlar dâhil olmak üzere hiç kimseye bilgi veremezler', 'conditions': [], 'exceptions': [],
             'elements': [{'id': 'action', 'kind': 'action', 'text': 'işleme taraf olanlar dâhil olmak üzere hiç kimseye bilgi veremezler'},
                          {'id': 'subject', 'kind': 'subject', 'text': 'Yükümlüler'},
                          {'id': 'prohibition', 'kind': 'prohibition', 'text': 'veremezler'}]}
MASAK = ("Türkiye Cumhuriyeti'nin finansal istihbarat birimi, Hazine ve Maliye Bakanlığına bağlı Mali Suçları Araştırma Kurulu Başkanlığı "
         "(MASAK) olup merkezi Ankara'dadır.")
RESERVATION = 'Rezervasyon: Müşterilerimizden kimlik belgesi istenmez; rezervasyonlarda yalnızca ad ve telefon numarası alınır.'
OFFICIAL = 'Resmi talepler: Mali Suçları Araştırma Kurulu Başkanlığı veya denetim elemanlarınca istenen bilgi ve belgeler gecikmeksizin verilir.'
ONLY_ABOVE = ("Ancak iş yükünün yönetilebilir tutulması için yalnızca tutarı 25.000 TL'yi aşan işlemler bildirilir; bu tutarın altındaki "
              "şüpheli işlemler MASAK'a bildirilmez, yalnızca iç kayıtlarda tutulur.")


def claim(payload, quantities, text, kind, span=None, relation='UNRELATED', support='', others=(), confirm=None):
    """The verifier claims `kind` on `span` of `text` (a fast POSSIBLE_CONFLICT sends it there); the other passages are
    fast POSSIBLE_SUPPORT readings that the verifier confirms (`confirm` True), rejects (False) or is not asked about.
    (relation of p1, coverage, codes of p1, results by source id)."""
    span = span or text

    def answer_fast(passage):
        return fast('POSSIBLE_CONFLICT', span) if passage == text else fast('POSSIBLE_SUPPORT', passage)

    def answer_verify(passage):
        if passage == text:
            return verdict(True, kind, span, relation, support=support)
        return verdict(relation='SUPPORTS' if confirm else 'UNRELATED', support=passage if confirm else '')

    relations, results, (coverage, _, _, _) = gate(V19(fast=answer_fast, verify=answer_verify), payload, [text, *others], quantities)
    return relations['p1'], coverage, codes(results['p1']), results


class RecordedFalseConflictTests(unittest.TestCase):
    """The two false CONFLICT rows of the v019t5 micro run, as sentence shapes (no case id in the engine)."""

    def test_a_descriptive_sentence_about_the_authority_is_no_conflict(self):
        payload = disclosure()
        # Every t5 rule let it through: anchored on "Başkanlığı" and "bulun"-, no role or act mismatch, no withdrawal.
        self.assertTrue(precheck.anchored(payload, MASAK, 'Şüpheli işlem riski bulunmaktadır. ' + MASAK))
        self.assertEqual((precheck.role_mismatch(payload, MASAK), precheck.different_action(payload, MASAK)), ('', ''))
        self.assertTrue(precheck.type_supported(payload, 'PROHIBITED_ACTION_ALLOWED', MASAK))
        self.assertFalse(restates_prohibition(payload, MASAK))
        relation, coverage, found, _ = claim(payload, (), MASAK, 'PROHIBITED_ACTION_ALLOWED')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_OBJECT', found)
        self.assertNotIn(precheck.GATE_PASSED, found)

    def test_asking_customers_for_identity_papers_is_no_conflict_with_giving_the_authority_information(self):
        self.assertTrue(precheck.anchored(GIVE, RESERVATION))
        self.assertTrue(precheck.type_supported(GIVE, 'EXEMPTION_ADDED', RESERVATION))
        relation, coverage, found, _ = claim(GIVE, (), RESERVATION, 'EXEMPTION_ADDED')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_OBJECT', found)
        code, why = precheck.conflict_gate(GIVE, RESERVATION, RESERVATION)
        self.assertEqual(code, 'CONFLICT_GATE_OBJECT')
        self.assertIn('@provide', why)


class DimensionTests(unittest.TestCase):
    """Each dimension the gate reads, with the code it records when the evidence for it is missing."""

    def test_same_direction_prohibitions_are_no_conflict(self):
        same = 'Şüpheli işlem bildiriminde bulunulduğu hiç kimseyle paylaşılmaz.'
        relation, coverage, found, _ = claim(disclosure(), (), same, 'PROHIBITED_ACTION_ALLOWED')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_EFFECT', found)
        # "verilmez" restates "veremezler" (the ability negative of the same verb, one act: giving).
        same = 'Müşterilere, haklarında bildirim yapıldığına dair hiçbir bilgi verilmez.'
        self.assertEqual(precheck.conflict_gate(GIVE_NONE, same, same)[0], 'CONFLICT_GATE_EFFECT')

    def test_a_complementary_obligation_is_no_conflict(self):
        internal = 'Şüpheli işlemler ayrıca iç denetim birimine de raporlanır.'
        relation, coverage, found, _ = claim(REPORT, (), internal, 'REQUIRED_ACTION_FORBIDDEN')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_EFFECT', found)

    def test_another_counterparty_is_no_conflict(self):
        payload, quantities = md28()
        customer = 'Şüpheli işlem bildirimi yapıldığı, müşteriye en geç otuz gün içinde bildirilir.'
        relation, coverage, found, _ = claim(payload, quantities, customer, 'DEADLINE_MISMATCH')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_ACTOR', found)
        # The same late timing towards the authority is a contradiction.
        late = "Şüpheli işlem bildirimi, şüphenin oluştuğu tarihten itibaren otuz gün içinde MASAK'a yapılır."
        self.assertEqual(claim(payload, quantities, late, 'DEADLINE_MISMATCH')[:2], ('CONFLICTS', 'CONFLICT'))

    def test_another_transfer_role_stays_with_the_role_rule(self):
        payload, quantities = md24(4)
        relation, coverage, found, _ = claim(payload, quantities, OUTGOING_SKIP, 'EXEMPTION_ADDED', span='bu bilgiler için teyit aranmaz.')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('ACTOR_ROLE_MISMATCH', found)
        self.assertFalse(GATE & set(found))                      # the gate runs after the role rule, not instead of it

    def test_another_act_and_object_is_no_conflict(self):
        payload, quantities = md28()
        complaints = 'Müşteri şikâyetleri her ayın son iş günü toplu olarak değerlendirilir.'
        self.assertEqual(precheck.conflict_gate(payload, complaints, complaints, quantities)[0], 'CONFLICT_GATE_OBJECT')
        relation, coverage, found, _ = claim(payload, quantities, complaints, 'DEADLINE_MISMATCH')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_OBJECT', found)             # anchored on "iş günü": every t5 rule let it through

    def test_a_rule_for_another_customer_group_is_no_conflict(self):
        payload = plain('yetkili temsilcilerin kimliği teyit edilir', subject='Tüzel kişi müşterilerde')
        other = 'Bireysel müşterilerde kimlik teyidi aranmaz.'
        relation, coverage, found, _ = claim(payload, (), other, 'EXEMPTION_ADDED')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_SCOPE', found)
        # The same exemption for the duty's own group is a contradiction.
        own = 'Tüzel kişi müşterilerde kimlik teyidi aranmaz.'
        self.assertEqual(claim(payload, (), own, 'EXEMPTION_ADDED')[:2], ('CONFLICTS', 'CONFLICT'))

    def test_the_duty_s_own_exception_restated_is_no_conflict_and_a_wider_one_is(self):
        restated = ('Bildirim bilgisi yalnızca yükümlülük denetimi ile görevlendirilen denetim elemanlarına ve yargılama sırasında '
                    'mahkemelere açıklanabilir.')
        relation, coverage, found, _ = claim(disclosure(), (), restated, 'PROHIBITED_ACTION_ALLOWED')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('CONFLICT_GATE_SCOPE', found)
        wider = 'Bildirim bilgisi iş ortaklarımıza açıklanabilir.'
        relation, coverage, found, _ = claim(disclosure(), (), wider, 'PROHIBITED_ACTION_ALLOWED')
        self.assertEqual((relation, coverage), ('CONFLICTS', 'CONFLICT'))
        self.assertIn(precheck.GATE_PASSED, found)


class TrueContradictionTests(unittest.TestCase):
    """Real contradictions pass the gate, each on the clause that sets the incompatible effect on the duty's act."""

    def passes(self, payload, quantities, text, kind, span=None):
        relation, coverage, found, results = claim(payload, quantities, text, kind, span=span)
        self.assertEqual((relation, coverage), ('CONFLICTS', 'CONFLICT'), found)
        self.assertIn(precheck.GATE_PASSED, found)
        self.assertFalse(GATE & set(found))
        return next(n['detail'] for n in results['p1']['notes'] if n['code'] == precheck.GATE_PASSED)

    def test_month_end_batch_against_ten_business_days(self):
        self.passes(*md28(), MONTH_END, 'DEADLINE_MISMATCH')

    def test_five_years_against_eight_years_of_keeping(self):
        self.assertIn('beş yıl', self.passes(*md46(), FIVE_YEARS, 'DEADLINE_MISMATCH'))

    def test_an_identification_skip_under_15000_tl_read_with_the_sentence_before(self):
        # The quoted sentence ("Bu istisna ... uygulanabilir") points back to the skip; the gate reads the sentence before it.
        detail = self.passes(*md5_2(), KYC_SKIP, 'EXEMPTION_ADDED', span=KYC_SKIP_SPAN)
        self.assertIn('teyit adımı atlanır', detail)

    def test_only_above_an_amount(self):
        self.passes(REPORT, (), ONLY_ABOVE, 'THRESHOLD_MISMATCH')

    def test_an_exemption_from_the_customer_s_declaration(self):
        self.assertIn('EXEMPTED', self.passes(*md24a(6), DECLARATION, 'EXEMPTION_ADDED', span=DECLARATION_SPAN))

    def test_giving_what_the_duty_forbids_to_give(self):
        # "... şube uyum sorumlusuna iletilir": another verb of the duty's act family, as the clause's main verb.
        self.assertIn('iletilir', self.passes(GIVE_NONE, (), BRANCH, 'PROHIBITED_ACTION_ALLOWED'))

    def test_other_forms_of_the_duty_s_verb(self):
        taken = plain('iş ilişkisinin amacı ve mahiyeti hakkında bilgi alınır', subject='Sürekli iş ilişkisi tesisinde')
        self.passes(taken, (), 'İş ilişkisinin amacı hakkında müşteriden bilgi alınmaz.', 'REQUIREMENT_REMOVED')
        approval = plain('iş ilişkisi kurulmadan önce üst düzey yönetici onayı alınır', subject='Siyasi nüfuz sahibi müşterilerle')
        self.passes(approval, (), 'Siyasi nüfuz sahibi müşterilerde üst yönetim onayı aranmaz.', 'REQUIREMENT_REMOVED')
        self.passes(approval, (), 'Siyasi nüfuz sahibi müşterilerde şube müdürü onayı yeterlidir.', 'DIRECT_OPPOSITE')
        # Returning the transfer is one of two alternative acts of the receiving institution's duty.
        payload, quantities = md24(4)
        returned = 'Eksik bilgi içeren bir transfer mesajı alındığında transfer iade edilmez ve eksik bilgiler de talep edilmez.'
        self.passes(payload, quantities, returned, 'DIRECT_OPPOSITE')
        # Destroying records ends keeping them.
        self.passes(*md46(), 'Kayıtlar üç yıl sonra imha edilir.', 'DEADLINE_MISMATCH')


class FallbackTests(unittest.TestCase):
    """A claim the gate rejects falls back through the existing path of a rejected claim; never an unjustified cover."""

    def test_the_rejected_sentence_never_supports_the_duty(self):
        relation, coverage, _, _ = claim(GIVE, (), RESERVATION, 'EXEMPTION_ADDED', relation='SUPPORTS', support=RESERVATION)
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))

    def test_the_duty_is_covered_only_by_a_passage_the_verifier_confirmed(self):
        relations, coverage, found, results = claim(GIVE, (), RESERVATION, 'EXEMPTION_ADDED', others=(OFFICIAL,), confirm=True)
        self.assertEqual((relations, coverage), ('UNRELATED', 'COVERS_TEXT'))
        self.assertEqual(results['p2']['relation'], 'SUPPORTS')
        self.assertIsNotNone(results['p2']['verifier'])
        relation, coverage, _, _ = claim(GIVE, (), RESERVATION, 'EXEMPTION_ADDED', others=(OFFICIAL,), confirm=False)
        self.assertEqual(relation, 'UNRELATED')
        self.assertNotEqual(coverage, 'COVERS_TEXT')
        self.assertNotEqual(coverage, 'CONFLICT')


class RecordTests(unittest.TestCase):
    def test_the_notes_name_the_dimension_and_are_digest_safe(self):
        self.assertEqual(GATE, {'CONFLICT_GATE_ACTOR', 'CONFLICT_GATE_OBJECT', 'CONFLICT_GATE_SCOPE', 'CONFLICT_GATE_EFFECT'})
        _, _, _, results = claim(disclosure(), (), MASAK, 'PROHIBITED_ACTION_ALLOWED')
        note = next(n for n in results['p1']['notes'] if n['code'] in GATE)
        self.assertEqual((note['question'], note['type'], note['quote']), ('verify', 'PROHIBITED_ACTION_ALLOWED', MASAK))
        self.assertTrue(all(isinstance(v, str) for v in note.values()))
        canonical_bytes(results)

    def test_a_duty_whose_act_or_polarity_cannot_be_read_is_not_gated(self):
        permission = plain('müşteri bilgilerini üçüncü kişilerle paylaşabilir', modality='MAY')
        self.assertEqual(precheck.conflict_gate(permission, MASAK, MASAK)[0], '')
        self.assertEqual(precheck.conflict_gate({'modality': 'MUST', 'elements': []}, MASAK, MASAK)[0], '')

    def test_the_duty_s_act_as_the_gate_reads_it(self):
        self.assertEqual(precheck.gate_acts(GIVE_NONE), ({'@provide'}, 'PROHIBITED'))
        refuse = {'subject': 'Talepte bulunulanlar', 'modality': 'MUST_NOT', 'required_action': None, 'conditions': [], 'exceptions': [],
                  'prohibited_action': 'özel kanunlarda yazılı hükümleri ileri sürerek bilgi ve belge vermekten kaçınamazlar',
                  'elements': [{'id': 'prohibition', 'kind': 'prohibition', 'text': 'kaçınamazlar'}]}
        keys, polarity = precheck.gate_acts(refuse)
        self.assertEqual(polarity, 'REQUIRED')                   # a double negative: must give
        self.assertIn('@provide', keys)
        self.assertNotIn('kaçın', keys)
        self.assertEqual(precheck.gate_acts(md24(4)[0])[0], {'iade', 'sağla'})
        self.assertIn('alın', precheck.gate_acts(plain('bilgi alınır', subject='Sürekli iş ilişkisi tesisinde'))[0])
        # The verb words are the ones verb_keys has always read (behaviour kept).
        self.assertEqual(precheck.verb_keys(disclosure()), {'@disclose'})
        self.assertEqual(precheck.verb_keys(GIVE_NONE), {'verem'})


if __name__ == '__main__':
    unittest.main()

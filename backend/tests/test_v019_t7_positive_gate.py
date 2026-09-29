"""v0.19 t7 (26 September 2026): the positive evidence gate. A favourable reading (SUPPORTS or PARTIAL, fast or verifier) stands
only when nothing in the sentence(s) holding its quote shows that the sentence is about another duty: another party (ACTOR),
another recipient (RECIPIENT), another act (ACTION), and on the strong verifier's readings also an opposed effect (POLARITY), a
prohibition's object without the prohibition (OBJECT) or another customer group (SCOPE). Evidence is always positive: a sentence
the readers cannot read keeps the model's reading (conflict.support_gate, engine.positive_gate).

Measured on the v019t6 live micro run: Kanun md. 4(2), the prohibition to disclose to anyone that a report was made, was PARTIAL
against a gold NO_EVIDENCE. The verifier confirmed "Tüm şüpheli işlemler ilgili otoriteye raporlanır" as PARTIAL (subject and
action covered, the prohibition missing): a requirement to report to the authority, offered for a prohibition to disclose to
anyone else. Every model here is a fake answering by schema (tests/test_v019_conflict_pipeline.V19); the Tedbirler duties are cut
from the retained snapshot, the Kanun duties are hand-built as the judge reads them; the policy sentences are the evaluation
fixtures' shapes.
"""
import unittest

from regchain.evidence import canonical_bytes, digest
from regchain.pilot import conflict as precheck
from regchain.pilot import engine
from test_v019_conflict_pipeline import V19, duty, fast, md28, md46, verdict
from test_v019_stricter_partial import associations
from test_v019_t5_engine import codes, disclosure, gate, md24a, plain, with_lists
from test_v019_t6_conflict_gate import GIVE, OFFICIAL

POSITIVE = set(precheck.POSITIVE_GATE_CODES)
# The Kanun 5549 md. 4(2) policy passage of the v019t6 live run (a bullet list; the second bullet was confirmed as PARTIAL).
AUTHORITY_REPORT = ('• Muhafaza: Müşteri bilgi ve belgeleri, işlemler ile ilgili kayıtlar 8 yıl süre ile muhafaza edilir.\n'
                    '• Şüpheli İşlem Bildirimi: Tüm şüpheli işlemler ilgili otoriteye raporlanır.')
AUTHORITY_REPORT_QUOTE = 'Şüpheli İşlem Bildirimi: Tüm şüpheli işlemler ilgili otoriteye raporlanır'
# A passage that restates the prohibition, and one about reporting to MASAK only (the I02 md. 4(2) shapes).
CONFIDENTIAL = ("Gizlilik: MASAK'a şüpheli işlem bildiriminde bulunulduğu, yükümlülük denetimi yapan denetim elemanları ve yargılama "
                "sırasında mahkemeler dışında, işlemin tarafı olan müşteri dâhil hiç kimseye açıklanmaz.")
MASAK_REPORT = ("Şüpheli işlem bildirimi: Personel, bir işleme konu malvarlığının yasa dışı yollardan elde edildiğinden şüphelendiğinde "
                "durumu uyum görevlisine iletir; uyum görevlisi değerlendirmesini tamamlayarak işlemi MASAK'a bildirir. Ancak yalnızca "
                "tutarı 25.000 TL'yi aşan işlemler bildirilir.")
MASAK_REPORT_QUOTE = ("Şüpheli işlem bildirimi: Personel, bir işleme konu malvarlığının yasa dışı yollardan elde edildiğinden şüphelendiğinde "
                      "durumu uyum görevlisine iletir; uyum görevlisi değerlendirmesini tamamlayarak işlemi MASAK'a bildirir.")
PARTNER = ('İşlem izleme: Platform üzerinden geçen işlemler, olağandışı hacim artışları için günlük olarak izlenir; şüpheli görülen '
           'işlemler iş ortağı kuruluşa bildirilir.')
# A duty to keep records eight years, and a sentence of the same party about the same documents with another act (reporting).
KEEP_DUTY = plain('işlemlerine ilişkin belge, defter ve kayıtları sekiz yıl süreyle saklamak')
REPORTS_TO_MASAK = "Yükümlüler şüpheli işlemlere ilişkin belgeleri en geç on iş günü içinde MASAK'a bildirir."
# Kanun 5549 md. 4(1) as the judge reads it: report to the authority.
REPORT = plain('bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi')
TO_CUSTOMER = 'Şüpheli işlemler, işlemin tarafı olan müşteriye de bildirilir.'
REPORT_TEN = "Şüpheli işlemler, şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde MASAK'a bildirilir."
REPORT_BARE = "Şüpheli işlemler MASAK'a bildirilir."
# The transfer-message sentences of the gold COVERS_TEXT rows (they name "kimlik" and "teyit" as objects, not as the act).
CRYPTO_MESSAGE = ('Gönderen ve alıcı bilgileri: Şirketimizce gönderilen tüm kripto varlık transfer mesajlarında, tutarına bakılmaksızın, '
                  'gönderenin adı soyadı, unvanı veya tam adı, cüzdan adresi (cüzdan adresi yoksa işlem referans numarası) ile T.C. kimlik '
                  'numarası, pasaport numarası veya müşteri numarasından en az biri yer alır ve bu bilgilerin doğruluğu teyit edilir. Alıcının '
                  'adı soyadı, unvanı veya tam adı ile cüzdan adresi (cüzdan adresi yoksa işlem referans numarası) de mesaja eklenir.')
BANK_MESSAGE = ('1. Bankamızca gönderilen yurt içi ve yurt dışı elektronik transfer mesajlarında gönderenin adı soyadı, unvanı veya tam adı, '
                'hesap numarası (hesap numarası yoksa işlem referans numarası) ve adresi ya da müşteri numarası yer alır ve bu bilgilerin '
                'doğruluğu teyit edilir; bu uygulama tutarı 50.000 TL ve üzerindeki transferlerde zorunludur.')
BANK_SMALL = ("2. Bankamızca gönderilen ve tutarı 50.000 TL'nin altındaki transferlerin mesajlarına gönderen ve alıcının adı soyadı, unvanı "
              "veya tam adı ile hesap numarası (hesap numarası yoksa işlem referans numarası) yazılır; bu bilgiler için teyit aranmaz.")
RECORDS = ('Bireysel müşterilerde ad, soyad, doğum tarihi, uyruk, kimlik belgesi bilgileri, adres, imza ve meslek bilgisi toplanır; T.C. '
           'vatandaşlarında kimlik numarası da alınır ve bilgiler resmî kimlik belgesiyle karşılaştırılır.')
NOT_BEFORE = ('Hiçbir müşteri ilişkisi, müşterinin kimliği doğrulanıp kayda geçirilmeden başlatılmaz; tek seferlik yüksek tutarlı '
              'işlemlerde de aynı doğrulama işlemden önce yapılır.')
CARDS = 'Ödeme güvenliği: Kasiyerler müşterilere ait kart numaralarını ve kart şifrelerini hiçbir şekilde not almaz, kimseye vermez ve saklamaz.'
COMPANIES = ('Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde; tüzel kişinin unvanı, ticaret sicil numarası, vergi kimlik '
             'numarası ve açık adresi alınır.')


def ids_of(payload, *kinds):
    return [e['id'] for e in payload['elements'] if not kinds or e['kind'] in kinds]


def flow(payload, texts, fast_answers, verify_answers=None, quantities=()):
    """The engine's flow and aggregation (strong gate) with fakes that answer by passage: `fast_answers` and `verify_answers`
    map a passage text to an answer (a fast reading defaults to IRRELEVANT, a verifier answer to UNRELATED).
    (relations by source id, results by source id, (coverage, reason, control, flags), the fake)."""
    judge = V19(fast=lambda text: fast_answers.get(text) or fast('IRRELEVANT', covered=()),
                verify=lambda text: (verify_answers or {}).get(text) or verdict(relation='UNRELATED'))
    relations, results, outcome = gate(judge, payload, texts, quantities)
    return relations, results, outcome, judge


def gate_codes(result):
    return [code for code in codes(result) if code in POSITIVE]


class RecordedShapeTests(unittest.TestCase):
    """The I04 md. 4(2) row of the v019t6 live run, as sentence shapes (no case id in the engine)."""

    def test_reporting_to_the_authority_is_no_partial_of_the_prohibition_to_disclose(self):
        payload = disclosure()
        answers = {AUTHORITY_REPORT: fast('POSSIBLE_PARTIAL', AUTHORITY_REPORT_QUOTE, covered=('action',), missing=('subject', 'prohibition'))}
        confirm = {AUTHORITY_REPORT: with_lists(verdict(relation='PARTIAL', support=AUTHORITY_REPORT_QUOTE), ['subject', 'action'], ['prohibition'])}
        relations, results, (coverage, reason, _, flags), judge = flow(payload, [AUTHORITY_REPORT], answers, confirm)
        self.assertEqual((relations['p1'], coverage, flags), ('UNRELATED', 'NO_EVIDENCE', []))
        self.assertEqual(gate_codes(results['p1']), ['POSITIVE_GATE_RECIPIENT'])
        # The fast reading is rejected before any question: the CONFIRM_PARTIAL call of the live run is not made.
        self.assertEqual(judge.kinds('verify'), [])
        self.assertIn('authority', results['p1']['reason'])

    def test_the_verifier_s_partial_on_that_sentence_is_rejected_as_well(self):
        # The live run's answer: conflict false, PARTIAL, subject and action covered, the prohibition missing.
        payload = disclosure()
        answers = {AUTHORITY_REPORT: fast('POSSIBLE_CONFLICT', AUTHORITY_REPORT_QUOTE)}
        live = {AUTHORITY_REPORT: with_lists(verdict(relation='PARTIAL', support=AUTHORITY_REPORT_QUOTE), ['subject', 'action'], ['prohibition'])}
        relations, results, (coverage, _, _, _), judge = flow(payload, [AUTHORITY_REPORT], answers, live)
        self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertEqual(gate_codes(results['p1']), ['POSITIVE_GATE_RECIPIENT'])
        self.assertEqual(len(judge.kinds('verify')), 1)
        self.assertEqual(results['p1']['verifier']['relation_if_no_conflict'], 'PARTIAL')     # the model's answer stays on record

    def test_the_cover_rests_on_the_passage_that_restates_the_prohibition(self):
        # I02 md. 4(2): the strong SUPPORTS on a sentence about reporting to MASAK no longer carries the cover; the passage that
        # restates the prohibition is confirmed instead (one CONFIRM_COVERS question).
        payload = disclosure()
        answers = {MASAK_REPORT: fast('POSSIBLE_CONFLICT', MASAK_REPORT_QUOTE), CONFIDENTIAL: fast('POSSIBLE_SUPPORT', CONFIDENTIAL,
                                                                                                    covered=('action', 'subject', 'prohibition'))}
        verifier = {MASAK_REPORT: with_lists(verdict(relation='SUPPORTS', support=MASAK_REPORT_QUOTE), ['action', 'subject', 'prohibition']),
                    CONFIDENTIAL: with_lists(verdict(relation='SUPPORTS', support=CONFIDENTIAL), ['action', 'subject', 'prohibition'])}
        relations, results, (coverage, _, _, _), judge = flow(payload, [MASAK_REPORT, CONFIDENTIAL], answers, verifier)
        self.assertEqual((relations, coverage), ({'p1': 'UNRELATED', 'p2': 'SUPPORTS'}, 'COVERS_TEXT'))
        self.assertEqual(gate_codes(results['p1']), ['POSITIVE_GATE_RECIPIENT'])
        self.assertEqual((gate_codes(results['p2']), results['p2']['covers_confirmation']), ([], 'CONFIRMED'))
        self.assertEqual(len(judge.kinds('verify')), 2)


class DimensionTests(unittest.TestCase):
    """The named mismatches: each rejects the favourable reading with its code, and the row is NO_EVIDENCE."""

    def test_the_same_party_doing_another_act_is_no_evidence(self):
        self.assertEqual(precheck.gate_acts(KEEP_DUTY), ({'@keep'}, 'REQUIRED'))
        answers = {REPORTS_TO_MASAK: fast('POSSIBLE_PARTIAL', REPORTS_TO_MASAK, covered=('subject', 'action'))}
        relations, results, (coverage, _, _, _), judge = flow(KEEP_DUTY, [REPORTS_TO_MASAK], answers)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_ACTION']))
        self.assertEqual(judge.kinds('verify'), [])
        # The verifier's reading of the same sentence is read the same way.
        answers = {REPORTS_TO_MASAK: fast('POSSIBLE_CONFLICT', REPORTS_TO_MASAK)}
        verifier = {REPORTS_TO_MASAK: with_lists(verdict(relation='PARTIAL', support=REPORTS_TO_MASAK), ['subject', 'action'], [])}
        relations, results, (coverage, _, _, _), _ = flow(KEEP_DUTY, [REPORTS_TO_MASAK], answers, verifier)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_ACTION']))

    def test_the_same_act_on_another_group_s_object_is_no_evidence(self):
        # Identification is the act; whose identity is its object. The duty identifies associations, the sentence registered
        # companies, and the verifier listed the subject as covered (its element rule OTHER_SUBJECT cannot see it).
        payload, quantities = associations()
        answers = {COMPANIES: fast('POSSIBLE_SUPPORT', COMPANIES, covered=ids_of(payload, 'action', 'subject'))}
        verifier = {COMPANIES: with_lists(verdict(relation='SUPPORTS', support=COMPANIES), ids_of(payload, 'action', 'subject'))}
        relations, results, (coverage, _, _, _), judge = flow(payload, [COMPANIES], answers, verifier, quantities)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_SCOPE']))
        # The fast reading is not rejected on a group: the strong confirmation reads it (one question).
        self.assertEqual([c['task'] for c in judge.kinds('verify')], ['judge.verify.support'])
        self.assertEqual(results['p1']['confirmation']['outcome'], 'UNRELATED')

    def test_the_same_act_and_object_for_another_recipient_is_no_evidence(self):
        answers = {TO_CUSTOMER: fast('POSSIBLE_PARTIAL', TO_CUSTOMER, covered=('action',), missing=('subject',))}
        relations, results, (coverage, _, _, _), judge = flow(REPORT, [TO_CUSTOMER], answers)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_RECIPIENT']))
        self.assertEqual(judge.kinds('verify'), [])
        # The conflict gate reads the same sentence as another counterparty (no contradiction either).
        self.assertEqual(precheck.conflict_gate(REPORT, TO_CUSTOMER, TO_CUSTOMER)[0], precheck.GATE_ACTOR)

    def test_a_reporting_duty_offered_for_the_prohibition_to_disclose_is_no_evidence(self):
        internal = 'Şüpheli işlem bildirimleri uyum görevlisine iletilir.'
        payload = disclosure()
        answers = {internal: fast('POSSIBLE_PARTIAL', internal, covered=('action',), missing=('prohibition',))}
        verifier = {internal: with_lists(verdict(relation='PARTIAL', support=internal), ['subject', 'action'], ['prohibition'])}
        relations, results, (coverage, _, _, _), judge = flow(payload, [internal], answers, verifier)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_POLARITY']))
        # A requirement to report is read only on the verifier's answer: the confirmation was asked.
        self.assertEqual(len(judge.kinds('verify')), 1)

    def test_a_prohibition_s_object_without_the_prohibition_is_no_evidence(self):
        filed = 'Şüpheli işlem bildirimleri uyum biriminde dosyalanır.'
        payload = disclosure()
        answers = {filed: fast('POSSIBLE_CONFLICT', filed)}
        verifier = {filed: with_lists(verdict(relation='PARTIAL', support=filed), ['subject', 'action'], ['prohibition'])}
        relations, results, (coverage, _, _, _), _ = flow(payload, [filed], answers, verifier)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_OBJECT']))

    def test_a_genuine_partial_of_the_same_duty_stays_partial(self):
        payload, quantities = md28()
        answers = {REPORT_BARE: fast('POSSIBLE_PARTIAL', REPORT_BARE, covered=('action',), missing=('deadline_1',))}
        verifier = {REPORT_BARE: with_lists(verdict(relation='PARTIAL', support=REPORT_BARE), ['action'], ['deadline_1'])}
        relations, results, (coverage, reason, _, _), judge = flow(payload, [REPORT_BARE], answers, verifier, quantities)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('PARTIAL', 'PARTIAL', []))
        self.assertEqual(results['p1']['partial_confirmation'], 'CONFIRMED')
        self.assertIn('en geç on iş günü içinde', reason)
        # A prohibition kept towards one party of the "anyone": a partial statement of the same duty.
        customer = 'Şüpheli işlem bildiriminde bulunulduğu, işlemin tarafı olan müşteriye açıklanmaz.'
        answers = {customer: fast('POSSIBLE_PARTIAL', customer, covered=('action', 'prohibition'))}
        verifier = {customer: with_lists(verdict(relation='PARTIAL', support=customer), ['action', 'prohibition', 'subject'], [])}
        relations, results, (coverage, _, _, _), _ = flow(disclosure(), [customer], answers, verifier)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('PARTIAL', 'PARTIAL', []))

    def test_full_semantic_coverage_stays_covers_text(self):
        payload, quantities = md28()
        answers = {REPORT_TEN: fast('POSSIBLE_SUPPORT', REPORT_TEN, covered=('action', 'deadline_1'))}
        verifier = {REPORT_TEN: with_lists(verdict(relation='SUPPORTS', support=REPORT_TEN), ['action', 'deadline_1'])}
        relations, results, (coverage, _, _, flags), _ = flow(payload, [REPORT_TEN], answers, verifier, quantities)
        self.assertEqual((relations['p1'], coverage, flags, gate_codes(results['p1'])), ('SUPPORTS', 'COVERS_TEXT', [], []))
        answers = {CONFIDENTIAL: fast('POSSIBLE_SUPPORT', CONFIDENTIAL, covered=('action', 'subject', 'prohibition'))}
        verifier = {CONFIDENTIAL: with_lists(verdict(relation='SUPPORTS', support=CONFIDENTIAL), ['action', 'subject', 'prohibition'])}
        relations, results, (coverage, _, _, _), _ = flow(disclosure(), [CONFIDENTIAL], answers, verifier)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('SUPPORTS', 'COVERS_TEXT', []))


class GoldCoversShapeTests(unittest.TestCase):
    """The favourable readings the gold COVERS_TEXT rows rest on stay favourable: transfer-message sentences whose object names
    "kimlik" or "teyit" (a blanket act rule rejected them), records sentences, the official-requests sentence."""

    def test_transfer_message_sentences_stay_covers_text(self):
        payload, quantities = md24a(2)
        critical = ids_of(payload, 'action', 'subject', 'threshold')
        answers = {CRYPTO_MESSAGE: fast('POSSIBLE_SUPPORT', CRYPTO_MESSAGE[:200], covered=critical)}
        verifier = {CRYPTO_MESSAGE: with_lists(verdict(relation='SUPPORTS', support=CRYPTO_MESSAGE[:200]), critical)}
        relations, results, (coverage, _, _, _), _ = flow(payload, [CRYPTO_MESSAGE], answers, verifier, quantities)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('SUPPORTS', 'COVERS_TEXT', []))
        small = duty('24', 2, 'Onbeşbin TL’nin altındaki yurt içi ve yurt dışı elektronik transfer mesajlarında gönderene ve alıcıya ilişkin '
                     'olarak birinci fıkranın (a) ve (b) bentlerinde belirtilen bilgilere', 'MUST', 'yer verilir.')[0]
        for text in (BANK_MESSAGE, BANK_SMALL, CRYPTO_MESSAGE):
            for verified in (False, True):
                self.assertEqual(precheck.support_gate(small, text, text, (), verified)[0], '', text[:40])
                self.assertEqual(precheck.support_gate(payload, text, text, (), verified)[0], '', text[:40])

    def test_records_and_identification_sentences_stay_favourable(self):
        keep, _ = md46()
        for text in (RECORDS, NOT_BEFORE):
            for verified in (False, True):
                self.assertEqual(precheck.support_gate(keep, text, text, ['deadline_1'], verified)[0], '', text[:40])

    def test_the_official_requests_sentence_covers_and_the_card_sentence_is_rejected(self):
        answers = {OFFICIAL: fast('POSSIBLE_SUPPORT', OFFICIAL, covered=('act_1', 'act_2')), CARDS: fast('POSSIBLE_PARTIAL', CARDS, covered=('act_1',),
                                                                                                           missing=('act_2',))}
        verifier = {OFFICIAL: with_lists(verdict(relation='SUPPORTS', support=OFFICIAL), ['act_1', 'act_2'])}
        relations, results, (coverage, _, _, _), _ = flow(GIVE, [OFFICIAL, CARDS], answers, verifier)
        self.assertEqual((relations, coverage), ({'p1': 'SUPPORTS', 'p2': 'UNRELATED'}, 'COVERS_TEXT'))
        self.assertEqual((gate_codes(results['p1']), gate_codes(results['p2'])), ([], ['POSITIVE_GATE_RECIPIENT']))


class ContradictionTests(unittest.TestCase):
    """The gate never makes a contradiction and never hides one: an opposed effect on a fast reading still goes to the verifier."""

    def test_a_fast_support_with_an_opposed_effect_still_goes_to_the_confirmation(self):
        # v019t3 heldout8: the CONFIRM_COVERS answer on this fast SUPPORTS was the contradiction.
        payload = disclosure()
        answers = {PARTNER: fast('POSSIBLE_SUPPORT', PARTNER, covered=('action',))}
        claim = {PARTNER: verdict(True, 'REQUIRED_ACTION_FORBIDDEN', 'şüpheli görülen işlemler iş ortağı kuruluşa bildirilir', 'UNRELATED')}
        relations, results, (coverage, _, _, _), judge = flow(payload, [PARTNER], answers, claim)
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        self.assertIn(precheck.GATE_PASSED, codes(results['p1']))
        self.assertEqual(gate_codes(results['p1']), [])
        self.assertEqual(len(judge.kinds('verify')), 1)
        # A favourable verifier answer on it is rejected (POLARITY), never turned into a contradiction.
        support = {PARTNER: with_lists(verdict(relation='SUPPORTS', support=PARTNER), ['action'])}
        relations, results, (coverage, _, _, _), _ = flow(payload, [PARTNER], answers, support)
        self.assertEqual((relations['p1'], coverage, gate_codes(results['p1'])), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_POLARITY']))

    def test_a_rejected_conflict_claim_s_fallback_is_gated_too(self):
        # The verifier claims a contradiction on the authority report (the t5 act rule rejects it: DIFFERENT_ACTION, another
        # addressee) and offers the clause after the heading as its favourable reading: that fallback is gated too.
        payload = disclosure()
        clause = 'Tüm şüpheli işlemler ilgili otoriteye raporlanır'
        answers = {AUTHORITY_REPORT: fast('POSSIBLE_CONFLICT', AUTHORITY_REPORT_QUOTE)}
        claim = {AUTHORITY_REPORT: with_lists(verdict(True, 'PROHIBITED_ACTION_ALLOWED', AUTHORITY_REPORT_QUOTE, 'PARTIAL', support=clause),
                                              ['subject', 'action'], ['prohibition'])}
        relations, results, (coverage, _, _, _), _ = flow(payload, [AUTHORITY_REPORT], answers, claim)
        self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertEqual(codes(results['p1']), ['DIFFERENT_ACTION', 'POSITIVE_GATE_RECIPIENT'])


class GuardTests(unittest.TestCase):
    """Positive evidence only: wording that may reverse or limit a polarity, an unread predicate and an unreadable duty keep the
    model's reading."""

    def test_a_limited_or_negated_disclosure_is_no_polarity_evidence(self):
        payload = disclosure()
        for text in ('Bildirim bilgisi yalnızca denetim elemanlarına ve mahkemelere açıklanır.',
                     'Bildirimde bulunulduğuna ilişkin bilgilerin açıklanmasına izin verilmez.',
                     'Şüpheli işlem bildiriminde bulunulduğu hiç kimseyle paylaşılmaz.'):
            self.assertEqual(precheck.support_gate(payload, text, text, ['subject'])[0], '', text)

    def test_the_duty_s_own_recipient_among_others_is_no_recipient_evidence(self):
        both = "Şüpheli işlemler MASAK'a ve işlemin tarafı olan müşteriye bildirilir."
        words = precheck.re.findall('%s+' % precheck.W, precheck.fold(both))
        at = words.index('bildirilir')
        self.assertEqual((precheck._addressee_before(words, at), precheck._addressees_before(words, at)), ('THIRD_PARTY', {'AUTHORITY', 'THIRD_PARTY'}))
        self.assertEqual(precheck.support_gate(REPORT, both, both, (), False)[0], '')
        self.assertEqual(precheck.support_gate(REPORT, TO_CUSTOMER, TO_CUSTOMER, (), False)[0], 'POSITIVE_GATE_RECIPIENT')

    def test_an_order_of_two_events_is_no_polarity_evidence_against_a_requirement(self):
        identify = plain('müşterinin kimlik bilgilerini doğrulamak')
        before = 'Müşteri, bilgileri doğrulanmadan kabul edilmez.'
        self.assertEqual(precheck._polarity_at(['müşteri', 'bilgileri', 'doğrulanmadan', 'kabul', 'edilmez'], 2), 'PROHIBITED')
        self.assertEqual(precheck.support_gate(identify, before, before)[0], '')
        self.assertEqual(precheck.support_gate(identify, 'Müşterinin kimlik bilgileri doğrulanmaz.', 'Müşterinin kimlik bilgileri doğrulanmaz.')[0],
                         'POSITIVE_GATE_POLARITY')

    def test_an_unread_predicate_keeps_the_reading(self):
        # "yer alır" is a predicate no reader knows: "teyit edilir" at the end of the sentence is not its only act.
        payload = md24a(2)[0]
        self.assertEqual(precheck._foreign_action(payload, precheck.gate_acts(payload)[0], CRYPTO_MESSAGE.split('. Alıcının')[0]), '')
        self.assertTrue(precheck._foreign_action(KEEP_DUTY, precheck.gate_acts(KEEP_DUTY)[0], REPORTS_TO_MASAK))
        self.assertEqual(precheck._predicate(['yer', 'alır']), 'unread')
        self.assertEqual(precheck._predicate(['doğruluğu', 'teyit', 'edilir']), '@identify')
        self.assertEqual(precheck._predicate(['şüpheli', 'işlem', 'bildirimi']), '')

    def test_a_duty_or_a_sentence_the_readers_cannot_read_keeps_the_reading(self):
        may = {**plain('müşteriye bilgi verebilir'), 'modality': 'MAY'}
        self.assertEqual(precheck.support_gate(may, TO_CUSTOMER, TO_CUSTOMER)[0], '')
        self.assertEqual(precheck.support_gate(disclosure(), '', 'metin')[0], '')
        vague = 'Politika tüm çalışanlar için bağlayıcıdır.'
        for verified in (False, True):
            self.assertEqual(precheck.support_gate(disclosure(), vague, vague, ['prohibition'] if not verified else [], verified)[0], '')

    def test_a_double_negative_prohibition_is_read_as_its_requirement(self):
        # "bilgi ve belge vermekten kaçınamazlar" requires giving: its prohibition element listed missing is no OBJECT evidence.
        refrain = {'subject': 'Kendisinden talepte bulunulanlar', 'modality': 'MUST_NOT', 'required_action': None,
                   'prohibited_action': 'özel kanunlarda yazılı hükümleri ileri sürerek bilgi ve belge vermekten kaçınamazlar',
                   'conditions': [], 'exceptions': [],
                   'elements': [{'id': 'action', 'kind': 'action', 'text': 'özel kanunlarda yazılı hükümleri ileri sürerek bilgi ve belge vermekten kaçınamazlar'},
                                {'id': 'subject', 'kind': 'subject', 'text': 'Kendisinden talepte bulunulanlar'},
                                {'id': 'prohibition', 'kind': 'prohibition', 'text': 'kaçınamazlar'}]}
        self.assertEqual(precheck.gate_acts(refrain)[1], 'REQUIRED')
        self.assertEqual(precheck.support_gate(refrain, OFFICIAL, OFFICIAL, ['prohibition'])[0], '')

    def test_the_quoted_sentence_ends_at_the_quote_s_own_full_stop(self):
        self.assertEqual(precheck.quoted_sentences(MASAK_REPORT_QUOTE, MASAK_REPORT), MASAK_REPORT_QUOTE)
        self.assertEqual(precheck.quoted_sentences('uyum görevlisine iletir', MASAK_REPORT), MASAK_REPORT_QUOTE)


class RecordTests(unittest.TestCase):
    def test_the_notes_name_the_dimension_and_are_digest_safe(self):
        payload = disclosure()
        answers = {AUTHORITY_REPORT: fast('POSSIBLE_PARTIAL', AUTHORITY_REPORT_QUOTE, covered=('action',), missing=('prohibition',))}
        _, results, _, _ = flow(payload, [AUTHORITY_REPORT], answers)
        note = next(n for n in results['p1']['notes'] if n['code'] in POSITIVE)
        self.assertEqual((note['question'], note['code'], note['quote']), ('fast', 'POSITIVE_GATE_RECIPIENT', AUTHORITY_REPORT_QUOTE))
        self.assertTrue(note['detail'].startswith('Rule: "raporlanır" is addressed to authority'))
        canonical_bytes(results)
        self.assertEqual(len(precheck.POSITIVE_GATE_CODES), 6)

    def test_the_gate_s_version_moves_the_coverage_hash(self):
        parts = engine.V19_HASH_PARTS
        self.assertEqual(parts['positive_gate'], precheck.POSITIVE_GATE_VERSION)
        self.assertEqual(engine.V19_PROMPT_HASH, digest(parts))
        self.assertNotEqual(digest({**parts, 'positive_gate': parts['positive_gate'] + '-next'}), engine.V19_PROMPT_HASH)


if __name__ == '__main__':
    unittest.main()

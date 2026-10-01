"""v0.19 round 5 (25 September 2026): role/actor-aware conflicts, polarity, the PARTIAL gate and the 8k verifier input.

Measured on the v019t4 runs (micro set): every false CONFLICT was a sentence about another party or another act (the
bank's own outgoing messages against a duty of the receiving or the intermediary institution, customers exempted from
their own declaration against the institution's duty to fill its messages, reporting to MASAK against the duty not to
disclose a report); two PARTIAL rows became NO_EVIDENCE because the verifier read "Yükümlüler" as another subject, two
became COVERS_TEXT through a fast reading's element list; twelve verifier requests of Tedbirler md. 5 went to the 16k
window (5,143-5,434 of 5,120 admissible tokens). Every model here is a fake answering by schema (tests/test_v019_conflict_pipeline.V19).
"""
import json
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction.providers import AI_WINDOW, admissible_tokens, estimate_tokens, fit_call, payload_tokens
from regchain.pilot import conflict as precheck
from regchain.pilot.engine import (VERIFY_PROMPT, coverage_of_v19, model_duty, passages_v19, precheck_passage, verify_payload, verify_schema,
                                   window_compressor)
from regchain.platform.redaction import ALLOWED_CALL_FIELDS, RedactionPolicy, redact_call
from test_v018_reliability import SCHEMA, fake_client
from test_v019_conflict_pipeline import V19, duty, fast, md5_1, md26, md28, md46, rows, verdict
from test_v019_context import CLIENT, adaptive, answer, sized

RECEIVING = 'Birinci fıkrada belirtilen bilgileri içermeyen elektronik transfer mesajı alan finansal kuruluş'
OUTGOING_SKIP = ("Bankamızca gönderilen ve tutarı 50.000 TL'nin altındaki transferlerin mesajlarına gönderen ve alıcının adı soyadı ile hesap "
                 "numarası yazılır; bu bilgiler için teyit aranmaz.")
OUTGOING_LIMIT = "Bankamızca gönderilen ve tutarı 50.000 TL'nin altındaki transferlerin mesajlarına gönderen ve alıcının adı soyadı yazılır."
DECLARATION = ("Kayıtlı olmayan cüzdan adreslerine gönderilen veya bu adreslerden alınan transferlerde müşteriden karşı tarafın adı soyadı "
               "hakkında beyan alınır. Kurumsal müşterilerimiz ve tutarı 10.000 TL'nin altındaki transferler bu beyandan muaftır.")
DECLARATION_SPAN = "Kurumsal müşterilerimiz ve tutarı 10.000 TL'nin altındaki transferler bu beyandan muaftır."
BRANCH = ("Grup içi risk yönetimi amacıyla, şube müşterileri hakkında MASAK'a şüpheli işlem bildiriminde bulunulduğu bilgisi şube uyum "
          "sorumlusuna iletilir.")
LEAVE = 'Personel yıllık izin planı her yıl Ocak ayında hazırlanır.'


def md24(number):
    if number == 4:
        return duty('24', 4, RECEIVING, 'MUST', lambda t: t[t.index('söz konusu'):t.index(' sağlar.') + len(' sağlar')])
    return duty('24', 6, 'Transfer emrinin verildiği finansal kuruluştan ödemeyi gerçekleştirecek finansal kuruluşa kadar oluşan mesaj '
                'zincirinde göndericiye ilişkin olarak elektronik transfer mesajlarında bulunması gereken bilgilere', 'MUST',
                lambda t: t[t.index('transfere aracılık'):t.index(' gösterilir') + len(' gösterilir')])


def md24a(number):
    if number == 2:
        return duty('24/A', 2, 'Kripto varlık hizmet sağlayıcılar', 'MUST', lambda t: t[t.index('tarafından'):t.index(' yer verilir') + 12])
    return duty('24/A', 6, 'Herhangi bir kripto varlık hizmet sağlayıcı nezdinde kayıtlı olmayan bir cüzdan adresine gönderilen veya böyle '
                'bir cüzdan adresinden alınan kripto varlık transferlerinde kripto varlık hizmet sağlayıcı nezdinde söz konusu transferin '
                'tarafı olan müşteriden', 'MUST', lambda t: t[t.index('kayıtlı olmayan cüzdan adresi sahibi'):t.index(' alınır.') + 7])


def disclosure():
    """Kanun 5549 md. 4(2) as the judge reads it (hand-built: the Kanun is not in the Tedbirler fixture)."""
    action = ('Başkanlığa şüpheli işlem bildiriminde bulunulduğunu, yükümlülük denetimi ile görevlendirilen denetim elemanları ile '
              'yargılama sırasında mahkemeler dışında, işleme taraf olanlar dahil hiç kimseye açıklayamazlar')
    exception = 'yükümlülük denetimi ile görevlendirilen denetim elemanları ile yargılama sırasında mahkemeler dışında'
    return {'subject': 'Yükümlüler', 'modality': 'MUST_NOT', 'required_action': None, 'prohibited_action': action, 'conditions': [],
            'exceptions': [exception], 'elements': [{'id': 'action', 'kind': 'action', 'text': action},
                                                    {'id': 'subject', 'kind': 'subject', 'text': 'Yükümlüler'},
                                                    {'id': 'prohibition', 'kind': 'prohibition', 'text': 'açıklayamazlar'}]}


def plain(action, subject='Yükümlüler', modality='MUST', **extra):
    """A duty with an act, a subject and the given critical elements ({id: (kind, text)})."""
    elements = [{'id': 'action', 'kind': 'action', 'text': action}, {'id': 'subject', 'kind': 'subject', 'text': subject}]
    elements += [{'id': key, 'kind': kind, 'text': text} for key, (kind, text) in extra.items()]
    return {'subject': subject, 'modality': modality, 'required_action': action, 'prohibited_action': None, 'conditions': [], 'exceptions': [],
            'elements': elements}


def gate(judge, payload, texts, quantities=()):
    """passages_v19 and the engine's aggregation (strong gate on): (relations, results, (coverage, reason, control, flags))."""
    checks, _, results, _, _ = passages_v19(judge, payload, rows(*texts), frozenset(), quantities, 1, 's', 'o', False)
    outcome = coverage_of_v19(checks, results, frozenset(), quantities, payload.get('elements'), strong_gate=True)
    canonical_bytes(results)                                     # packets forbid floats: the records must be canonical
    return {c.source_id: c.relation for c in checks}, {r['source_id']: r for r in results}, outcome


def codes(result):
    return [n['code'] for n in result.get('notes') or []]


def with_lists(value, covered=(), missing=()):
    return {**value, 'covered_elements': list(covered), 'missing_elements': list(missing)}


class RoleTests(unittest.TestCase):
    def test_a_sender_sentence_is_no_conflict_with_a_receiving_institution_duty(self):
        payload, quantities = md24(4)
        self.assertEqual(precheck.duty_role(payload)['actor'], 'RECEIVING')
        judge = V19(fast=fast('POSSIBLE_CONFLICT', 'bu bilgiler için teyit aranmaz.'),
                    verify=verdict(True, 'EXEMPTION_ADDED', 'bu bilgiler için teyit aranmaz.', 'UNRELATED'))
        relations, results, (coverage, _, _, _) = gate(judge, payload, [OUTGOING_SKIP], quantities)
        self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('ACTOR_ROLE_MISMATCH', codes(results['p1']))
        # The verifier was told whose duty it is.
        self.assertEqual(judge.kinds('verify')[0]['payload']['duty']['role']['actor'], 'receiving institution')

    def test_a_sender_sentence_is_no_conflict_with_an_intermediary_duty(self):
        payload, quantities = md24(6)
        self.assertEqual(precheck.duty_role(payload)['actor'], 'INTERMEDIARY')
        claim = lambda text: verdict(True, 'SCOPE_NARROWED', OUTGOING_LIMIT, 'UNRELATED') if text == OUTGOING_LIMIT else verdict()
        judge = V19(fast=lambda text: fast('POSSIBLE_CONFLICT', OUTGOING_LIMIT) if text == OUTGOING_LIMIT else fast('POSSIBLE_PARTIAL', BRANCH,
                                                                                                                    covered=('act_1',)),
                    verify=claim)
        relations, results, (coverage, _, _, _) = gate(judge, payload, [OUTGOING_LIMIT, BRANCH], quantities)
        self.assertEqual((relations['p1'], relations['p2'], coverage), ('UNRELATED', 'UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('ACTOR_ROLE_MISMATCH', codes(results['p1']))
        # A passage that names no transfer at all is not about an intermediary's duty: settled without the verifier.
        self.assertIn('ACTOR_ROLE_MISMATCH', codes(results['p2']))
        self.assertEqual([c['passage'] for c in judge.kinds('verify')], [OUTGOING_LIMIT])

    def test_customers_exempted_from_their_declaration_is_no_conflict_with_the_institutions_duty(self):
        payload, quantities = md24a(2)
        self.assertFalse(precheck.duty_role(payload)['customer_act'])
        judge = V19(fast=fast('POSSIBLE_CONFLICT', DECLARATION_SPAN), verify=verdict(True, 'EXEMPTION_ADDED', DECLARATION_SPAN, 'UNRELATED'))
        relations, results, (coverage, _, _, _) = gate(judge, payload, [DECLARATION], quantities)
        self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('ACTOR_ROLE_MISMATCH', codes(results['p1']))

    def test_a_same_role_true_conflict_stays(self):
        # The duty to take the customer's declaration: the same sentence is a real exemption from it (I06 md. 24/A(6)).
        payload, quantities = md24a(6)
        self.assertTrue(precheck.duty_role(payload)['customer_act'])
        judge = V19(fast=fast('POSSIBLE_CONFLICT', DECLARATION_SPAN), verify=verdict(True, 'EXEMPTION_ADDED', DECLARATION_SPAN, 'NOT_APPLICABLE'))
        relations, _, (coverage, _, _, _) = gate(judge, payload, [DECLARATION], quantities)
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        # The receiving institution's duty against a sentence about received messages: the same role, a conflict.
        payload, quantities = md24(4)
        received = 'Eksik bilgi içeren bir transfer mesajı alındığında transfer iade edilmez ve eksik bilgiler de talep edilmez.'
        judge = V19(fast=fast('POSSIBLE_CONFLICT', received), verify=verdict(True, 'DIRECT_OPPOSITE', received, 'NOT_APPLICABLE'))
        relations, _, (coverage, _, _, _) = gate(judge, payload, [received], quantities)
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))

    def test_role_equivalent_wording_is_the_same_party(self):
        report = plain('bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi')
        text = "Bankamız, şüpheli işlemleri MASAK'a bildirir."
        # The verifier listed "Yükümlüler" missing for a bank policy: the same obliged party (I04 md. 4(1) was OTHER_SUBJECT).
        judge = V19(fast=fast('POSSIBLE_PARTIAL', text, covered=('action',), missing=('subject',)),
                    verify=with_lists(verdict(relation='PARTIAL', support=text), ['action'], ['subject']))
        relations, results, (coverage, _, _, _) = gate(judge, report, [text])
        self.assertEqual((relations['p1'], coverage), ('PARTIAL', 'PARTIAL'))
        self.assertIn('ROLE_EQUIVALENT_SUBJECT', codes(results['p1']))
        for subject, passage in (('Yükümlüler', 'Şirketimiz kayıtları saklar.'), ('finansal kuruluş', 'Bankamız kayıtları saklar.')):
            self.assertTrue(precheck.same_subject(plain('saklamak', subject), passage))
        self.assertEqual(precheck.role_mismatch(plain('bildirmek', 'finansal kuruluş'), 'Bankamız, şüpheli işlemleri bildirir.'), '')
        # Customer families: "müşteri" states a duty about natural-person customers; registered companies do not state one
        # about associations (C03 md. 8(1)).
        self.assertTrue(precheck.same_subject(plain('kimliğini tespit etmek', 'Gerçek kişi müşteriler'), 'Müşterilerin kimliği tespit edilir.'))
        self.assertFalse(precheck.same_subject(plain('kimliğini tespit etmek', 'Dernekler'),
                                               'Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde unvan alınır.'))


class PolarityTests(unittest.TestCase):
    def test_a_prohibition_restated_as_a_prohibition_is_no_conflict(self):
        payload = disclosure()
        restated = ('Şüpheli işlem bildiriminde bulunulduğu bilgisi, müşteriler dâhil hiç kimseyle paylaşılmaz; yükümlülük denetimi yapan '
                    'denetim elemanlarına ve yargılama sırasında mahkemelere verilen bilgiler bu kuralın dışındadır.')
        judge = V19(fast=fast('POSSIBLE_CONFLICT', restated), verify=verdict(True, 'DIRECT_OPPOSITE', restated, 'NOT_APPLICABLE'))
        relations, results, (coverage, _, _, _) = gate(judge, payload, [restated])
        self.assertEqual(relations['p1'], 'SUPPORTS')
        self.assertNotEqual(coverage, 'CONFLICT')
        self.assertIn('CONFLICT_WITHDRAWN', codes(results['p1']))
        simple = {'modality': 'MUST_NOT', 'prohibited_action': 'isimsiz hesap açamazlar', 'conditions': [], 'exceptions': []}
        self.assertTrue(precheck.same_direction(simple, 'Bankamız isimsiz hesap kesinlikle açamaz.'))
        self.assertFalse(precheck.same_direction(simple, '50.000 TL üzerindeki işlemlerde isimsiz hesap açamaz.'))   # narrower
        # Reporting to MASAK is another act than disclosing to others (I02 md. 4(2), I04 md. 4(2)).
        report = 'Şüpheli İşlem Bildirimi: Tüm şüpheli işlemler ilgili otoriteye raporlanır.'
        judge = V19(fast=fast('POSSIBLE_CONFLICT', report), verify=verdict(True, 'DIRECT_OPPOSITE', report, 'UNRELATED'))
        relations, results, _ = gate(judge, payload, [report])
        self.assertEqual(relations['p1'], 'UNRELATED')
        self.assertIn('DIFFERENT_ACTION', codes(results['p1']))
        # ... but telling the customer is disclosing (no mismatch on the act).
        self.assertEqual(precheck.different_action(payload, 'Bildirimde bulunulduğu müşteriye bildirilir.'), '')

    def test_a_requirement_restated_as_a_requirement_is_no_conflict(self):
        payload = plain('müşterinin kimliğini tespit etmek')
        text = 'Müşterinin kimliği her durumda tespit edilir.'
        self.assertEqual(precheck.polarities(text, precheck.verb_keys(payload)), {'REQUIRED'})
        judge = V19(fast=fast('POSSIBLE_CONFLICT', text), verify=verdict(True, 'DIRECT_OPPOSITE', text, 'NOT_APPLICABLE'))
        relations, results, (coverage, _, _, _) = gate(judge, payload, [text])
        self.assertEqual(relations['p1'], 'SUPPORTS')
        self.assertIn('SAME_DIRECTION', codes(results['p1']))
        self.assertNotEqual(coverage, 'CONFLICT')

    def test_a_permission_against_a_prohibition_may_conflict(self):
        payload, quantities = md26()
        text = 'Kampanya dönemlerinde basitleştirilmiş tedbirler bütün müşterilere uygulanabilir.'
        self.assertEqual(precheck.polarities(text, precheck.verb_keys(payload)), {'PERMITTED'})
        self.assertFalse(precheck.same_direction(payload, text))
        judge = V19(fast=fast('POSSIBLE_CONFLICT', text), verify=verdict(True, 'PROHIBITED_ACTION_ALLOWED', text, 'NOT_APPLICABLE'))
        relations, _, (coverage, _, _, _) = gate(judge, payload, [text], quantities)
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))

    def test_an_exemption_against_a_requirement_may_conflict(self):
        payload = plain('müşterinin kimliğini tespit etmek')
        text = "Tutarı 1.000 TL'nin altındaki işlemlerde müşteriler kimlik tespitinden muaftır."
        self.assertEqual(precheck.polarities(text, precheck.verb_keys(payload)), {'EXEMPTED'})
        judge = V19(fast=fast('POSSIBLE_CONFLICT', text), verify=verdict(True, 'EXEMPTION_ADDED', text, 'NOT_APPLICABLE'))
        relations, _, (coverage, _, _, _) = gate(judge, payload, [text])
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))


class PartialGateTests(unittest.TestCase):
    def test_a_missing_deadline_is_partial(self):
        payload, quantities = md28()
        text = "Şüpheli işlemler MASAK'a bildirilir."
        judge = V19(fast=fast('POSSIBLE_PARTIAL', text, covered=('action',), missing=('deadline_1',)),
                    verify=with_lists(verdict(relation='PARTIAL', support=text), ['action'], ['deadline_1']))
        relations, _, (coverage, reason, _, flags) = gate(judge, payload, [text], quantities)
        self.assertEqual((relations['p1'], coverage, flags), ('PARTIAL', 'PARTIAL', []))
        self.assertEqual([c['task'] for c in judge.kinds('verify')], ['judge.verify.support'])        # one CONFIRM_PARTIAL

    def test_a_missing_threshold_is_partial(self):
        payload = plain('müşterinin kimliğini tespit etmek', threshold_1=('threshold', 'onbeşbin TL veya üzerinde'))
        text = 'Müşterinin kimliği tespit edilir.'
        judge = V19(fast=fast('POSSIBLE_PARTIAL', text, missing=('threshold_1',)),
                    verify=with_lists(verdict(relation='PARTIAL', support=text), ['action'], ['threshold_1']))
        _, _, (coverage, reason, _, _) = gate(judge, payload, [text])
        self.assertEqual(coverage, 'PARTIAL')
        self.assertIn('onbeşbin TL veya üzerinde', reason)

    def test_the_same_act_with_an_incomplete_object_is_partial(self):
        payload = plain('belge, defter ve kayıtları muhafaza etmek', object=('object', 'belge, defter ve kayıtları'))
        text = 'İşlem kayıtları saklanır.'
        judge = V19(fast=fast('POSSIBLE_PARTIAL', text, missing=('object',)),
                    verify=with_lists(verdict(relation='PARTIAL', support=text), ['action'], ['object']))
        relations, _, (coverage, _, _, _) = gate(judge, payload, [text])
        self.assertEqual((relations['p1'], coverage), ('PARTIAL', 'PARTIAL'))

    def test_every_critical_element_stated_covers(self):
        payload = plain('belge, defter ve kayıtları muhafaza etmek', object=('object', 'belge, defter ve kayıtları'))
        text = 'Her türlü belge, defter ve kayıt muhafaza edilir.'
        judge = V19(fast=fast('POSSIBLE_SUPPORT', text, covered=('action', 'object')),
                    verify=with_lists(verdict(relation='SUPPORTS', support=text), ['action', 'object', 'subject'], []))
        relations, _, (coverage, _, _, flags) = gate(judge, payload, [text])
        self.assertEqual((relations['p1'], coverage, flags), ('SUPPORTS', 'COVERS_TEXT', []))

    def test_another_role_or_act_is_no_evidence(self):
        payload, quantities = md24(4)
        sent = ('Bankamızca gönderilen yurt içi ve yurt dışı elektronik transfer mesajlarında gönderenin adı soyadı ve hesap numarası yer '
                'alır.')
        judge = V19(fast=lambda text: fast('POSSIBLE_SUPPORT', sent, covered=('action',)) if text == sent else fast('IRRELEVANT', covered=()))
        relations, results, (coverage, _, _, _) = gate(judge, payload, [sent, LEAVE], quantities)
        self.assertEqual((relations['p1'], relations['p2'], coverage), ('UNRELATED', 'UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('ACTOR_ROLE_MISMATCH', codes(results['p1']))
        self.assertEqual(judge.kinds('verify'), [])                                  # settled by the role record, no verifier call

    def test_a_fast_element_list_never_completes_a_cover(self):
        # I01 md. 3(1), I04 md. 8: the verifier read the identification passage (act_2 missing); the records passage the fast
        # model said states act_2 was never read by it, so it cannot complete a joint COVERS_TEXT.
        payload = {**plain('kimliklerini tespit etmek ve gerekli diğer tedbirleri almak'), 'elements': [
            {'id': 'act_1', 'kind': 'action', 'text': 'kimliklerini tespit etmek'},
            {'id': 'act_2', 'kind': 'action', 'text': 'gerekli diğer tedbirleri almak'},
            {'id': 'subject', 'kind': 'subject', 'text': 'Yükümlüler'}]}
        identify = 'Sözleşmeyi imzalayan müşterinin kimliği resmi kimlik belgesinin aslı üzerinden tespit edilir.'
        records = 'Müşteri kimlik belgeleri ve işlem kayıtları beş yıl süreyle saklanır.'
        judge = V19(fast=lambda text: fast('POSSIBLE_PARTIAL', text, covered=('act_1',) if text == identify else ('act_1', 'act_2'),
                                           missing=('act_2',) if text == identify else ()),
                    verify=with_lists(verdict(relation='PARTIAL', support=identify), ['act_1'], ['act_2']))
        relations, _, (coverage, reason, _, _) = gate(judge, payload, [identify, records])
        self.assertEqual(coverage, 'PARTIAL')
        self.assertIn('gerekli diğer tedbirleri almak', reason)
        self.assertEqual(len(judge.kinds('verify')), 1)

    def test_an_unrelated_answer_resting_only_on_the_subject_is_partial(self):
        # I13 md. 21(3): the act stated, the timing missing, the obliged party read as "another subject".
        payload = plain('müşterinin kimlik bilgilerini üçüncü taraftan derhal alır', subject='finansal kuruluş',
                        deadline_1=('deadline', 'derhal'))
        text = 'Güvenilen kuruluştan müşterinin kimlik bilgileri temin edilerek müşteri dosyasına konur.'
        judge = V19(fast=fast('POSSIBLE_PARTIAL', text, missing=('deadline_1',)),
                    verify=with_lists(verdict(relation='UNRELATED', support=text), ['action'], ['subject', 'deadline_1']))
        relations, results, (coverage, _, _, _) = gate(judge, payload, [text])
        self.assertEqual((relations['p1'], coverage), ('PARTIAL', 'PARTIAL'))
        self.assertIn('ROLE_EQUIVALENT_SUBJECT', codes(results['p1']))
        # Another customer group stays UNRELATED (C03 md. 8(1)).
        groups = plain('kimlik tespitinde derneğin adını almak', subject='Dernekler')
        other = 'Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde tüzel kişinin unvanı alınır.'
        judge = V19(fast=fast('POSSIBLE_PARTIAL', other, missing=('subject',)),
                    verify=with_lists(verdict(relation='PARTIAL', support=other), ['action'], ['subject']))
        relations, results, (coverage, _, _, _) = gate(judge, groups, [other])
        self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('OTHER_SUBJECT', codes(results['p1']))


class WindowTests(unittest.TestCase):
    def setUp(self):
        token = AI_WINDOW.set(None)
        self.addCleanup(AI_WINDOW.reset, token)

    def reading(self, payload, quantities, text):
        signals = precheck_passage(payload, text, quantities)[0]
        return {'duty': payload, 'row': {'source_id': 'p1', 'text': text}, 'signals': signals, 'quantities': quantities}

    def test_the_compact_verifier_request_of_a_long_duty_fits_8k_and_keeps_the_deciding_sentence(self):
        for payload, quantities in (md5_1(), md46()):
            with self.subTest(duty=payload['required_action'][:40]):
                text = ('Müşteri kimlik belgeleri, işlem kayıtları ve defterler son işlem tarihinden itibaren sekiz yıl süreyle saklanır. '
                        'Yurt dışı ve yurt içi transferlerde işlem tutarı onbeşbin TL veya üzerinde olduğunda kimlik tespiti yapılır.')
                ids = [e['id'] for e in payload['elements']]
                request = verify_payload(self.reading(payload, quantities, text), 'CONFIRM_PARTIAL')
                legacy = {'duty': payload, 'passage': text, 'automatic_signals': precheck.compact(self.reading(payload, quantities, text)['signals']),
                          'question': request['question']}
                limit = admissible_tokens(type('Base', (), {'num_ctx': 8192, 'num_predict': 4096})())
                self.assertLessEqual(estimate_tokens(VERIFY_PROMPT, request, verify_schema(ids)), limit)
                self.assertLess(payload_tokens(request), payload_tokens(legacy))
                # Every element once: no legacy key repeats an element's text (required_action, items, deadline, threshold ...).
                self.assertLessEqual(set(request['duty']), {'modality', 'role', 'elements', 'subject', 'conditions', 'exceptions',
                                                            'source_sentence', 'definitions'})
                self.assertEqual(request['duty']['elements'], {e['id']: e['text'] for e in payload['elements']})
                self.assertLess(payload_tokens(request['duty']), 0.7 * payload_tokens(payload))
                self.assertEqual(request['passage'], text)                 # a request that fits is sent whole
        # When it must be cut, the deciding sentence stays: the fast reading's quote comes first.
        payload, quantities = md46()
        deciding = 'İşlem kayıtları ve müşteri dosyaları belge yönetim sisteminde saklanır.'
        text = ' '.join(['Toplantı odaları takvim üzerinden ayrılır ve kullanım sonrası temiz bırakılır.'] * 6 + [deciding] +
                        ['Kurum logosu yalın ve sade biçimde kullanılır.'] * 6)
        request = verify_payload(self.reading(payload, quantities, text), 'CONFIRM_PARTIAL')
        compress = window_compressor(payload, text, self.reading(payload, quantities, text)['signals'], quantities, deciding)
        out = compress(dict(request), payload_tokens({**request, 'passage': deciding + ' [...] [...]'}))
        self.assertIn(deciding, out['passage'])
        self.assertGreater(compress.dropped, 0)
        # A sentence repeated verbatim is sent once (quotes are still checked against the full passage).
        self.assertEqual(verify_payload(self.reading(payload, quantities, text), 'CONFIRM_PARTIAL')['passage'].count('Toplantı odaları'), 1)

    def test_a_16k_fallback_is_logged_with_its_reason(self):
        big = adaptive()
        payload = sized(9000)
        used, sent, info = fit_call(big, 'p', payload, SCHEMA)
        self.assertEqual((used, info['action'], info['fallback_reason']), (big.wide, 'fallback_large', 'NO_COMPRESSOR'))

        def short(value, target):                       # compresses, drops two sentences, but cannot reach 8k
            compress.dropped = 2
            value['passage'] = value['passage'][:-100]
            return value
        compress = short
        used, sent, info = fit_call(big, 'p', payload, SCHEMA, compress)
        self.assertEqual((info['action'], info['fallback_reason']), ('fallback_large', 'ESTIMATE_OVER_BASE_AFTER_COMPRESSION'))
        with patch(CLIENT) as factory:
            fake_client(factory, [answer(9000)])
            used.generate_structured('p', sent, SCHEMA)
        entry = big.call_log[-1]
        self.assertEqual((entry['window_action'], entry['num_ctx'], entry['16k_fallback_reason'], entry['compressed_evidence_count']),
                         ('fallback_large', 16384, 'ESTIMATE_OVER_BASE_AFTER_COMPRESSION', 0))      # the original payload went
        self.assertIn('estimated_prompt_tokens', entry)
        self.assertEqual(redact_call(entry, RedactionPolicy('minimal')), entry)
        self.assertLessEqual(set(entry), ALLOWED_CALL_FIELDS['minimal'])
        # A request that fits carries no reason and a zero count.
        used, sent, info = fit_call(big, 'p', sized(2000), SCHEMA)
        with patch(CLIENT) as factory:
            fake_client(factory, [answer(2000)])
            used.generate_structured('p', sent, SCHEMA)
        entry = big.call_log[-1]
        self.assertEqual((entry['window_action'], entry['num_ctx'], entry['compressed_evidence_count']), ('fits', 8192, 0))
        self.assertNotIn('16k_fallback_reason', entry)


if __name__ == '__main__':
    unittest.main()

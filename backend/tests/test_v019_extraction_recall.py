"""v0.19 extraction recall: duties phrased without a v0.18 marker, provision headings, and span salvage.

Forensics of the three v019t3 evidence runs (25 September 2026): required duties never reached the model
because their sub-paragraph had no marker the gate knew (Tedbirler md. 21(3) "... derhal alır", md. 24(1)
"... zorunlu olup ... teyit edilir", md. 22(1) "... gerçekleştirmezler", md. 15 "... doğrular"), because a
permission's condition is the duty (md. 21(2) "... emin olunması şartıyla mümkündür") or a limit read as a
scope exemption (md. 21(5) "... durumunda uygulanmaz"), or because the gate rejected a correct answer over
a copy difference (md. 31(3): five words skipped inside a long clause). The v0.18 markers stay binding and
exactly as strict; the new ones are soft (grounding.binding): extractable, alignable, never demanded.
"""
import json
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction.classify import classify, duty_modals
from regchain.extraction.grounding import CONDITIONS, EXCEPTIONS, MODAL, binding, modality, required_qualifiers, verify
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import CONTEXT_PROMPT, OllamaProvider, RulesProvider
from regchain.extraction.retrieval import retrieve
from regchain.extraction.schema import ExtractionOutput
from regchain.extraction.spans import normalized_span, overlap_span, salvage

from test_v019_extraction import Scripted, action_of, answer, candidate, tedbirler, unit

LOCAL = 'http://localhost:11434'
KANUN_5549_4_2 = ('(2) Yükümlüler, şüpheli işlem bildiriminde bulunulduğunu, yükümlülük denetimi ile görevlendirilen denetim '
                  'elemanları ile yargılama sırasında mahkemeler dışında, işleme taraf olanlar dahil hiç kimseye açıklayamazlar.')
CONC_MUST_WHERE = ('A firm must suspend any steps it takes to recover a debt from a customer where the customer disputes the '
                   'debt on valid grounds. [Note: paragraph 3.9e of DCG]')


def outcome(text, *values):
    return extract(text, Scripted(answer(*values)))


def kinds(text):
    return [(m.group(), modality(m.group()), binding(m)) for m in duty_modals(text)]


class ConditionedPermissionTests(unittest.TestCase):
    """md. 21: a bare permission stays a permission; its conditions and limits are duties."""

    def test_the_md21_sub_paragraphs_are_classed_on_their_own_wording(self):
        classes = {n: classify(unit('21', n)[1], 'Üçüncü tarafa güven').kind for n in (1, 2, 3, 4, 5)}
        self.assertEqual(classes, {1: 'PERMISSION', 2: 'OBLIGATION', 3: 'OBLIGATION', 4: 'OTHER', 5: 'PROHIBITION'})
        for n in (2, 3, 5):
            self.assertTrue(MODAL.search(unit('21', n)[1]), n)            # the engine's gate sends the unit to extraction
        self.assertEqual(kinds(unit('21', 2)[1]), [('şartıyla', 'MUST', False)])
        self.assertEqual(kinds(unit('21', 5)[1]), [('uygulanmaz', 'MUST_NOT', False)])

    def test_the_condition_of_a_permission_is_extracted_as_a_duty_with_its_own_qualifiers(self):
        _, text = unit('21', 2)
        subject = 'Üçüncü tarafa güvenilebilmesi'
        action = text[text.index('a) (Değişik'):text.index(' şartıyla mümkündür')]
        self.assertTrue(action.endswith('emin olunması'))
        # Nothing is demanded of an answer that does not use the soft marker ...
        self.assertEqual(required_qualifiers(text, CONDITIONS), [])
        # ... and the conditions of its sentence are, once an answer anchors a duty on it.
        demanded = required_qualifiers(text, CONDITIONS, ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, subject, 'MUST', action)]))
        self.assertEqual(demanded, ['yurt dışında yerleşik olması durumunda',
                                    'güvenilen kuruluş tarafından uzaktan kimlik tespiti yoluyla sürekli iş ilişkisi tesis edilmesi halinde'])
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, subject, 'MUST', action, conditions=demanded)]))
        # The marker word itself ("... emin olunması şartıyla") is the duty's end, not one more condition.
        self.assertNotIn('emin olunması şartıyla', ' '.join(demanded))

    def test_a_limit_on_a_permission_is_a_prohibition_and_the_rules_baseline_reads_md21(self):
        _, text = unit('21', 5)
        result = extract(text, RulesProvider())
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        value = result.output.obligations[0]
        self.assertEqual((value.modality, action_of(value)[-10:]), ('MUST_NOT', 'uygulanmaz'))
        self.assertEqual(value.conditions, ['üçüncü tarafın riskli ülkelerde yerleşik olması durumunda'])
        self.assertEqual(value.exceptions, [])                             # the marker is not also an exception
        _, third = unit('21', 3)
        value = extract(third, RulesProvider()).output.obligations[0]
        self.assertTrue(action_of(value).endswith('müşterinin kimlik bilgilerini üçüncü taraftan derhal alır'))

    def test_a_bare_permission_stays_a_permission(self):
        for text in [unit('21', 1)[1], '(1) Kurul ek tedbir alabilir.', '(2) Başkanlık yükümlülere bu konuda izin verir.',
                     '(3) Yükümlüler, müşterinin talep etmesi halinde bilgileri paylaşabilir.']:     # a condition without "yalnızca"
            with self.subTest(text=text[:40]):
                self.assertEqual(classify(text).kind, 'PERMISSION')
                self.assertEqual(duty_modals(text), [])
                self.assertEqual(json.loads(RulesProvider().generate(text))['status'], 'NO_EXPLICIT_OBLIGATION')
        self.assertEqual(classify('A firm may rely on a third party.').kind, 'PERMISSION')

    def test_a_permission_only_under_a_condition_is_a_duty_to_meet_it(self):
        text = '(1) Yükümlüler, yalnızca müşterinin yazılı onayının bulunması halinde bilgileri üçüncü kişilerle paylaşabilir.'
        self.assertEqual(classify(text).kind, 'OBLIGATION')
        self.assertEqual(kinds(text), [('halinde', 'MUST', False)])
        value = candidate(text, 'Yükümlüler', 'MUST', 'yalnızca müşterinin yazılı onayının bulunması')
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[value]))

    def test_a_provision_that_does_not_apply_stays_an_exemption(self):
        for text in ['(3) Bu madde hükümleri yabancı kuruluşlar hakkında uygulanmaz.',
                     '(4) Birinci fıkra hükmü, işlemin yurt dışında yapılması durumunda uygulanmaz.']:
            with self.subTest(text=text[:40]):
                self.assertEqual(classify(text).kind, 'EXEMPTION')
                self.assertEqual(duty_modals(text), [])


class AoristAndPassiveDutyTests(unittest.TestCase):
    """md. 24 / 24/A: duties in the aorist and the "edilir" passive reach the model and pass the gate."""

    def test_the_md24_and_24a_sub_paragraphs_carry_duty_markers(self):
        expected = {('24', 1): [('zorunlu', 'MUST', False), ('edilir', 'MUST', False)],
                    ('24/A', 1): [('zorunlu', 'MUST', False), ('edilir', 'MUST', False)],
                    ('24', 4): [('sağlar', 'MUST', False)],
                    ('24/A', 8): [('edilir', 'MUST', False), ('bulundurulur', 'MUST', True)],
                    ('22', 1): [('gerçekleştirmezler', 'MUST_NOT', False), ('açamazlar', 'MUST_NOT', False)],
                    ('15', 1): [('doğrular', 'MUST', False)]}
        for (article, number), markers in expected.items():
            with self.subTest(article=article, number=number):
                _, text = unit(article, number)
                self.assertEqual(kinds(text), markers)
                self.assertTrue(classify(text).extractable)
                self.assertTrue(MODAL.search(text))
        self.assertEqual([m for m, _, _ in kinds(unit('24/A', 3)[1])], ['eder', 'eder', 'bulundurulur'])

    def test_a_passive_teyit_edilir_duty_passes_the_gate(self):
        _, text = unit('24', 1)
        subject = 'Onbeşbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında'
        first = candidate(text, subject, 'MUST', text[text.index('gönderenin;'):text.index(' zorunlu olup')])
        second = candidate(text, 'bu bilgilerin doğruluğu', 'MUST', 'ayrıca teyit edilir')
        result = outcome(text, first, second)
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertIn('gönderen', action_of(result.output.obligations[0]))
        _, request = unit('24/A', 8)
        value = candidate(request, 'Altıncı ve yedinci fıkraların uygulanmasında', 'MUST',
                          'risk temelli yaklaşım çerçevesinde müşteriden transferin tarafları hakkında ilave bilgi ve belge talep edilir')
        tail = request[request.index('İşlemin tarafları'):]
        other = candidate(request, 'İşlemin tarafları hakkında', 'MUST', tail[tail.index('transferin gerçekleştirilmemesi'):tail.index(' hususları') + len(' hususları göz önünde bulundurulur')],
                          conditions=['İşlemin tarafları hakkında yeterli bilgi elde edilememesi durumunda'])
        verify(request, ExtractionOutput(status='EXTRACTED', obligations=[value, other]))

    def test_an_aorist_return_duty_is_extracted_by_the_rules_baseline(self):
        _, text = unit('24', 4)
        result = extract(text, RulesProvider())
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertIn('elektronik transferi iade eder veya', action_of(result.output.obligations[0]))

    def test_soft_markers_are_never_demanded_so_every_v018_answer_still_passes(self):
        _, text = unit('24/A', 3)
        # The run's salvaged answer covers only the v0.18 marker "bulundurulur"; "talep eder." and "iade eder."
        # are soft, so it is not rejected for leaving them out, and their conditions are not demanded of it.
        action = ('gönderici kripto varlık hizmet sağlayıcıdan gelen transferlerin reddedilmesi veya söz konusu kripto varlık hizmet '
                  'sağlayıcıyla yapılan işlemlerin sınırlandırılması ya da iş ilişkisine son verilmesi hususları göz önünde bulundurulur')
        condition = ('Gönderilen mesajların sürekli olarak eksik bilgiler içermesi ve bu bilgilerin talep edildiği halde '
                     'tamamlanmaması halinde')
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[
            candidate(text, 'alıcı kripto varlık hizmet sağlayıcı', 'MUST', action, conditions=[condition])]))
        # A dismissal of a unit with a soft duty is still refused, so the repair turn asks again.
        with self.assertRaisesRegex(ValueError, 'Cannot dismiss'):
            verify(unit('21', 3)[1], ExtractionOutput(status='NO_EXPLICIT_OBLIGATION'))

    def test_words_that_only_look_like_duties_are_not_markers(self):
        for text in ['(1) Bu Kanunda geçen; a) Bakanlık: Maliye Bakanlığını ifade eder.',
                     '(2) Bu şekilde yapılan işlemler bildirim yapılmış kabul edilir.',
                     '(4) Bildirim yapılmış olması, yükümlülerin diğer yükümlülüklerini ortadan kaldırmaz.',
                     '(1) Müşterinin kimliği işlem öncesinde tespit edilir.']:
            with self.subTest(text=text[:40]):
                self.assertNotIn(classify(text).kind, ('OBLIGATION', 'PROHIBITION'))
        self.assertEqual(duty_modals('(4) Bildirim yapılmış olması, yükümlülerin diğer yükümlülüklerini ortadan kaldırmaz.'), [])


class EnglishTests(unittest.TestCase):
    def test_a_conc_must_where_provision_keeps_its_condition(self):
        self.assertEqual(classify(CONC_MUST_WHERE, legal_type='RULE').kind, 'OBLIGATION')
        result = extract(CONC_MUST_WHERE, RulesProvider())
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        conditions = result.output.obligations[0].conditions             # the baseline copies the whole English tail
        self.assertEqual(len(conditions), 1)
        self.assertTrue(conditions[0].startswith('where the customer disputes the debt on valid grounds.'))
        # The condition cannot be dropped: not even by the salvage, which only files what the answer carries.
        action = 'suspend any steps it takes to recover a debt from a customer'
        self.assertEqual(outcome(CONC_MUST_WHERE, candidate(CONC_MUST_WHERE, 'A firm', 'MUST', action)).reason, 'GROUNDING_REJECTED')

    def test_what_a_firm_will_or_needs_to_do_is_a_duty(self):
        for text, mode in [('The firm will notify the customer of the outcome within 5 business days.', 'MUST'),
                           ('A lender needs to record the reasons for the decision.', 'MUST'),
                           ('The firm will not charge a fee for the first reminder.', 'MUST_NOT')]:
            with self.subTest(text=text[:30]):
                self.assertEqual(classify(text, legal_type='RULE').kind, 'PROHIBITION' if mode == 'MUST_NOT' else 'OBLIGATION')
                result = extract(text, RulesProvider())
                self.assertEqual(result.output.obligations[0].modality, mode)
        self.assertEqual(classify('This will help firms plan.', legal_type='RULE').kind, 'OTHER')


class HeadingContextTests(unittest.TestCase):
    def test_the_provision_heading_reaches_the_model_as_context_only(self):
        section = tedbirler()['21']
        _, text = unit('21', 3)
        same = [s for s in tedbirler().values() if s['version_id'] == section['version_id']]
        packet = retrieve(dict(section, text=text), same)
        self.assertEqual(packet.heading, 'Üçüncü tarafa güven')
        self.assertNotIn('heading', packet.manifest())                   # the packet shape is unchanged
        self.assertIn('provision_heading', CONTEXT_PROMPT)
        self.assertIn('context only', CONTEXT_PROMPT)
        sent = []

        def chat(prompt, payload, schema):
            sent.append(payload)
            return json.dumps({'status': 'NO_EXPLICIT_OBLIGATION', 'obligations': []})
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch.object(provider, '_chat', side_effect=chat):
            provider.generate_with_context(text, packet)
            provider.repair(text, packet, 'bad')
            provider.generate_with_context(text, None)
        self.assertEqual([p.get('provision_heading') for p in sent], ['Üçüncü tarafa güven', 'Üçüncü tarafa güven', None])
        self.assertEqual(sent[0]['source_paragraph'], text)

    def test_a_range_citation_is_context_that_does_not_govern_the_duty(self):
        section = tedbirler()['15']
        packet = retrieve(section, list(tedbirler().values()))
        cited = [i for i in packet.items if i['reason'] == 'exact_reference']
        self.assertEqual([(i['printed_label'], i['required']) for i in cited], [('Yönetmelik 200713012 md. 14', False)])
        _, text = unit('15', 1)
        action = text[text.index('6 ila 14'):]
        result = extract(text, Scripted(answer(candidate(text, 'Yükümlüler', 'MUST', action))), packet)
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertIn('gerçekliğini doğrular', action_of(result.output.obligations[0]))


class SpanRepairTests(unittest.TestCase):
    def test_a_semicolon_after_an_enumerated_subject_is_no_sentence_end(self):
        _, text = unit('31', 1)
        subject = 'Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar'
        cut = ('her türlü ortamdaki kayıtlarını , bu kayıtlara erişimi sağlamak veya okunabilir hale getirmek için gerekli tüm bilgi '
               've şifreleri tam ve doğru olarak, istenilen usul, şekil ve sürede gecikmeksizin vermek ve gerekli kolaylığı sağlamak')
        result = outcome(text, candidate(text, subject, 'MUST', cut))                # the run's answer (I08 md. 31(1))
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        action = action_of(result.output.obligations[0])
        self.assertTrue(action.startswith('Başkanlık ve denetim elemanları tarafından istenilecek her türlü bilgi, belge'))
        self.assertTrue(action.endswith(cut))
        self.assertIn('ACTION_SPAN_EXPANDED', [d.get('code') for d in result.diagnostics])


class SalvageTests(unittest.TestCase):
    def test_a_span_that_differs_only_by_punctuation_is_snapped_to_the_source(self):
        _, text = unit('28', 3)
        dropped = ('yeni bilgi ve bulgular elde edildiği takdirde tekrar Şüpheli İşlem Bildirim Formu doldurulur ve daha önce yapılan '
                   'bildirime ek olduğu belirtilerek gecikmeksizin Başkanlığa gönderilir.')                   # the comma is gone
        self.assertNotIn(dropped, text)
        span = normalized_span(dropped, [text])
        self.assertIn(span, text)
        self.assertTrue(span.startswith('yeni bilgi ve bulgular elde edildiği takdirde, tekrar'))
        self.assertTrue(span.endswith('gönderilir.'))
        subject = 'Bildirimde bulunulan işlemle ilgili olarak daha sonra yeni bilgi ve bulgular elde edildiği takdirde'
        condition = subject
        model = candidate(text, 'Bildirimde bulunulan işlem', 'MUST', dropped, conditions=[condition])
        result = outcome(text, model)
        record = next(d for d in result.diagnostics if d.get('code') == 'GROUNDING_SALVAGED')
        self.assertEqual(record['steps'][0], 'normalize')              # then the usual span repair of the clause start
        self.assertEqual(record['fields'][0]['source'], span)
        canonical_bytes(list(result.diagnostics))                          # digest-safe: strings and ints only
        self.assertIn(action_of(result.output.obligations[0]), text)

    def test_a_long_clause_with_skipped_words_is_snapped_to_the_whole_source_clause(self):
        _, text = unit('31', 3)
        short = ('yerinde yapılacak denetimler kapsamında defter ve belgeleri denetime hazır bulundurmak; tüm bilgi işlem sistemini, '
                 'denetim elemanlarına açmak ve verilerin güvenliğini sağlamak')          # the run's answer (md. 31(3))
        full = overlap_span(short, [text])
        self.assertEqual(full, text[text.index('yerinde'):text.index(' zorundadır')])
        result = outcome(text, candidate(text, 'Yükümlüler', 'MUST', short))
        record = next(d for d in result.diagnostics if d.get('code') == 'GROUNDING_SALVAGED')
        self.assertEqual(record['steps'], ['overlap'])
        self.assertIn('denetimin amaçlarına uygun olarak', action_of(result.output.obligations[0]))

    def test_changed_or_short_values_are_never_snapped(self):
        _, text = unit('24/A', 7)
        invented = ('Yurt dışında yerleşik söz konusu hizmet sağlayıcı veya finansal kuruluşun beşinci fıkrada öngörülen bir '
                    'mesajlaşma sistemini kullanmaması halinde')                          # "kullanması" negated
        self.assertIsNone(normalized_span(invented, [text]))
        self.assertIsNone(overlap_span(invented, [text]))
        self.assertIsNone(overlap_span('beyan alınır', [text]))            # too short to say where it was copied from
        far = 'Herhangi bir kripto varlık hizmet sağlayıcı beyan alınır'    # more words skipped than a quarter of the value
        self.assertIsNone(overlap_span(far, [unit('24/A', 6)[1]]))

    def test_a_turkish_exception_carried_inside_the_action_is_filed(self):
        action = ('şüpheli işlem bildiriminde bulunulduğunu, yükümlülük denetimi ile görevlendirilen denetim elemanları ile yargılama '
                  'sırasında mahkemeler dışında, işleme taraf olanlar dahil hiç kimseye açıklayamazlar')
        result = outcome(KANUN_5549_4_2, candidate(KANUN_5549_4_2, 'Yükümlüler', 'MUST_NOT', action))
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertEqual(result.output.obligations[0].exceptions,
                         ['yükümlülük denetimi ile görevlendirilen denetim elemanları ile yargılama sırasında mahkemeler dışında'])
        record = next(d for d in result.diagnostics if d.get('code') == 'GROUNDING_SALVAGED')
        self.assertIn('qualifier', record['steps'])


if __name__ == '__main__':
    unittest.main()

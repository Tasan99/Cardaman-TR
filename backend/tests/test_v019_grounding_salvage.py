"""v0.19 grounding salvage: whole duties the gate rejected for a copy error, not for a wrong answer.

The answers below are the kinds the v0.19 independent run (24 September 2026) rejected as
GROUNDING_REJECTED: a qualifier filed under the wrong field or copied inside the action (CONC 7.7.2,
7.7.3), a qualifier demand that was garbage ('7) Yurt dışında', '2) Yukarıdaki ...', a whole CONC
7.7.1 sentence), a Turkish action that stops short of its passive marker (Tedbirler md. 24/A(3)),
and a field that differs from the source only by case or quotes (md. 24/A(4)). verify stays strict;
spans.salvage selects source spans after the last attempt and verify decides again.
"""
import unittest

from regchain.evidence import canonical_bytes
from regchain.extraction.grounding import CONDITIONS, EXCEPTIONS, qualifier_spans, required_qualifiers, verify
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import RulesProvider
from regchain.extraction.schema import ExtractionOutput
from regchain.extraction.spans import salvage, source_span

from test_v019_extraction import Scripted, action_of, answer, candidate, tedbirler, unit

CONC_7_7_1 = ('When levying charges for debt recovery on a customer in default or arrears difficulties firms should consider '
              'their obligations under Principle 6 to pay due regard to the interests of customers and treat them fairly and the '
              'requirement to act to deliver good outcomes for retail customers under the Consumer Duty, as applicable. '
              '[Note: paragraphs 3.1 and 3.10 of DCG]')
CONC_7_7_2 = ('A firm must not claim the costs of recovering a debt from a customer if it has no contractual right to claim such '
              'costs. [Note: paragraph 3.11b of DCG]')
CONC_7_7_3 = ('A firm must not cause a customer to believe that the customer is legally liable to pay the costs of recovery where '
              'no such obligation exists. [Note: paragraph 3.11a of DCG]')
KANUN_5549_7_2 = ('(2) Yukarıdaki fıkraya göre talepte bulunulanlar savunma hakkına ilişkin hükümler saklı kalmak kaydıyla, özel '
                  'kanunlarda yazılı hükümleri ileri sürerek bilgi ve belge vermekten kaçınamazlar.')


def salvaged(test, text, result):
    """The one salvage record of a result, after checking the result is a clean, exact-span candidate."""
    test.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
    records = [d for d in result.diagnostics if d.get('code') == 'GROUNDING_SALVAGED']
    test.assertEqual(len(records), 1)
    test.assertEqual(set(records[0]) - {'attempt'}, {'stage', 'code', 'steps', 'fields'})
    test.assertEqual(records[0]['stage'], 'grounding')
    canonical_bytes(list(result.diagnostics))                              # digest-safe: no floats
    for value in result.output.obligations:
        for field in [value.subject, action_of(value), *value.conditions, *value.exceptions]:
            test.assertIn(field, text)                                     # every stored field is a source substring
    verify(text, result.output)
    return records[0]


class MisfiledQualifierTests(unittest.TestCase):
    def test_a_condition_filed_as_an_exception_is_moved(self):
        span = 'if it has no contractual right to claim such costs.'
        model = answer(candidate(CONC_7_7_2, 'A firm', 'MUST_NOT', 'claim the costs of recovering a debt from a customer',
                                 exceptions=[span]))
        provider = Scripted(model)
        result = extract(CONC_7_7_2, provider)
        self.assertEqual(provider.calls, 2)                                # the repair turn is spent first
        record = salvaged(self, CONC_7_7_2, result)
        self.assertEqual(record['steps'], ['qualifier'])
        self.assertEqual(result.output.obligations[0].conditions, [span])
        self.assertEqual(result.output.obligations[0].exceptions, [])

    def test_a_condition_copied_inside_the_action_is_filed_as_a_condition(self):
        action = 'cause a customer to believe that the customer is legally liable to pay the costs of recovery where no such obligation exists.'
        result = extract(CONC_7_7_3, Scripted(answer(candidate(CONC_7_7_3, 'A firm', 'MUST_NOT', action))))
        salvaged(self, CONC_7_7_3, result)
        self.assertEqual(result.output.obligations[0].conditions, ['where no such obligation exists.'])

    def test_a_qualifier_the_answer_does_not_carry_stays_rejected(self):
        # v0.19 t5 (intentional change): a Turkish CONDITION of the duty's own sentence, left out by the
        # model, is filed from the source on that duty (spans.own_sentence_condition, step
        # 'qualifier_source'); the v019t4 run lost md. 24(5) and 24/A(3) this way. What the answer carries
        # nowhere and the source does not tie to one of its duties still stays rejected: an English
        # condition, and any exception.
        _, text = unit('24/A', 3)
        action = ('gönderici kripto varlık hizmet sağlayıcıdan gelen transferlerin reddedilmesi veya söz konusu kripto varlık hizmet '
                  'sağlayıcıyla yapılan işlemlerin sınırlandırılması ya da iş ilişkisine son verilmesi hususları göz önünde bulundurulur')
        result = extract(text, Scripted(answer(candidate(text, 'alıcı kripto varlık hizmet sağlayıcı', 'MUST', action))))
        record = salvaged(self, text, result)
        self.assertEqual(record['steps'], ['qualifier_source'])
        self.assertEqual(result.output.obligations[0].conditions,
                         ['Gönderilen mesajların sürekli olarak eksik bilgiler içermesi ve bu bilgilerin talep edildiği halde tamamlanmaması halinde'])
        english = extract(CONC_7_7_2, Scripted(answer(candidate(CONC_7_7_2, 'A firm', 'MUST_NOT',
                                                                'claim the costs of recovering a debt from a customer'))))
        self.assertEqual(english.reason, 'GROUNDING_REJECTED')              # the condition was omitted, not misfiled
        self.assertNotIn('GROUNDING_SALVAGED', [d.get('code') for d in english.diagnostics])


class QualifierDemandTests(unittest.TestCase):
    def test_yurt_disinda_is_abroad_not_an_exception(self):
        _, text = unit('24/A', 7)
        self.assertIn('Yurt dışında', text)
        self.assertEqual(required_qualifiers(text, EXCEPTIONS), [])
        action = ('nezdinde söz konusu transferin tarafı olan müşteriden; göndericinin ya da alıcının gerçek kişilerde adı ve soyadı, '
                  'tüzel kişilerde ise unvan bilgisi ile adresi veya doğum yeri ve tarihi veya müşteri numarası, vatandaşlık numarası, '
                  'pasaport numarası, vergi kimlik numarası gibi kişiyi belirlemeye yarayan bilgilerden en az birine ilişkin beyan alınır')
        # The run's first answer: no exception, and no condition of the permission sentence that follows.
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, 'kripto varlık hizmet sağlayıcı', 'MUST', action)]))

    def test_a_condition_of_a_sentence_without_a_duty_marker_is_not_demanded(self):
        _, text = unit('24/A', 7)
        # "... kullanması halinde ise ... yer verilebilir.": the condition of a permission.
        self.assertTrue(any('kullanması halinde' in span for span in qualifier_spans(text, CONDITIONS)))
        self.assertEqual(required_qualifiers(text, CONDITIONS), [])

    def test_a_sub_paragraph_number_is_not_part_of_the_demanded_qualifier(self):
        demanded = required_qualifiers(KANUN_5549_7_2, CONDITIONS)
        self.assertEqual(demanded, ['Yukarıdaki fıkraya göre talepte bulunulanlar savunma hakkına ilişkin hükümler saklı kalmak kaydıyla'])
        model = candidate(KANUN_5549_7_2, 'talepte bulunulanlar', 'MUST_NOT',
                          'özel kanunlarda yazılı hükümleri ileri sürerek bilgi ve belge vermekten kaçınamazlar', conditions=demanded)
        verify(KANUN_5549_7_2, ExtractionOutput(status='EXTRACTED', obligations=[model]))

    def test_an_english_leading_clause_stops_before_the_duty_it_governs(self):
        condition = 'When levying charges for debt recovery on a customer in default or arrears difficulties'
        action = ('consider their obligations under Principle 6 to pay due regard to the interests of customers and treat them fairly and '
                  'the requirement to act to deliver good outcomes for retail customers under the Consumer Duty, as applicable.')
        output = ExtractionOutput(status='EXTRACTED', obligations=[candidate(CONC_7_7_1, 'firms', 'SHOULD', action, conditions=[condition])])
        self.assertEqual(required_qualifiers(CONC_7_7_1, CONDITIONS, output), [condition])
        verify(CONC_7_7_1, output)
        # Without the duty's subject there is nothing to cut at, and the demand stays as strict as before.
        self.assertTrue(required_qualifiers(CONC_7_7_1, CONDITIONS)[0].endswith('as applicable.'))


class ActionAndSpanTests(unittest.TestCase):
    def test_a_turkish_action_short_of_its_passive_marker_ends_at_it(self):
        _, text = unit('24/A', 3)
        short = ('gönderici kripto varlık hizmet sağlayıcıdan gelen transferlerin reddedilmesi veya söz konusu kripto varlık hizmet '
                 'sağlayıcıyla yapılan işlemlerin sınırlandırılması ya da iş ilişkisine son verilmesi')
        condition = ('gönderilen mesajların sürekli olarak eksik bilgiler içermesi ve bu bilgilerin talep edildiği halde '
                     'tamamlanmaması halinde')                                     # lower case: the source opens a sentence here
        model = answer(candidate(text, 'alıcı kripto varlık hizmet sağlayıcı', 'MUST', short, conditions=[condition]))
        result = extract(text, Scripted(model))
        record = salvaged(self, text, result)
        self.assertEqual(record['steps'], ['snap', 'action'])
        value = result.output.obligations[0]
        self.assertEqual(value.required_action, short + ' hususları göz önünde bulundurulur')
        self.assertEqual(value.conditions, ['G' + condition[1:]])

    def test_a_field_that_differs_only_by_case_quotes_or_whitespace_is_snapped(self):
        _, text = unit('24/A', 4)
        subject = text[4:text.index(' yer verilir')]
        action = 'yer verilir ve bu bilgilerin transferin her aşamasında aktarılmasına özel dikkat gösterilir.'
        result = extract(text, Scripted(answer(candidate(text, subject[0].lower() + subject[1:], 'MUST', action))))
        record = salvaged(self, text, result)
        self.assertEqual(record['steps'], ['snap'])
        self.assertEqual(result.output.obligations[0].subject, subject)
        _, quoted = unit('24/A', 2)
        self.assertIn('TL’nin', quoted)
        self.assertEqual(source_span("onbeşbin  TL'nin altındaki", [quoted]), 'onbeşbin TL’nin altındaki')
        self.assertEqual(source_span('ONBEŞBİN TL’NİN', [quoted]), 'onbeşbin TL’nin')
        self.assertEqual(source_span('onbesbin tl’nin', [quoted]), 'onbeşbin TL’nin')

    def test_invented_text_is_never_salvaged(self):
        _, text = unit('24/A', 7)
        action = ('nezdinde söz konusu transferin tarafı olan müşteriden; göndericinin ya da alıcının gerçek kişilerde adı ve soyadı, '
                  'tüzel kişilerde ise unvan bilgisi ile adresi veya doğum yeri ve tarihi veya müşteri numarası, vatandaşlık numarası, '
                  'pasaport numarası, vergi kimlik numarası gibi kişiyi belirlemeye yarayan bilgilerden en az birine ilişkin beyan alınır')
        # The run's second answer: the exception negates the source's condition ("kullanması" -> "kullanmaması").
        invented = ('Yurt dışında yerleşik söz konusu hizmet sağlayıcı veya finansal kuruluşun beşinci fıkrada öngörülen bir '
                    'mesajlaşma sistemini kullanmaması halinde')
        model = candidate(text, 'kripto varlık hizmet sağlayıcı', 'MUST', action, exceptions=[invented])
        self.assertIsNone(salvage(text, ExtractionOutput(status='EXTRACTED', obligations=[model])))
        result = extract(text, Scripted(answer(model)))
        self.assertEqual(result.reason, 'GROUNDING_REJECTED')
        self.assertEqual(result.output.status, 'INSUFFICIENT_EVIDENCE')
        self.assertNotIn('GROUNDING_SALVAGED', [d.get('code') for d in result.diagnostics])

    def test_an_action_of_another_duty_is_not_realigned(self):
        _, text = unit('24/A', 7)
        other = 'birinci fıkrada yer alan bilgilere kullanılan sistem vasıtasıyla mesajlarda yer verilebilir'   # a permission
        model = candidate(text, 'kripto varlık hizmet sağlayıcı', 'MUST', other)
        self.assertIsNone(salvage(text, ExtractionOutput(status='EXTRACTED', obligations=[model])))


class RulesBaselineTests(unittest.TestCase):
    def test_the_rules_provider_over_tedbirler_is_never_salvaged_and_keeps_its_candidates(self):
        from regchain.pilot.engine import split_units
        checked = 0
        for section in tedbirler().values():
            for text in [section['text'], *[t for _, t in split_units(section['text'])]]:
                result = extract(text, RulesProvider())
                self.assertNotIn('GROUNDING_SALVAGED', [d.get('code') for d in result.diagnostics], text[:80])
                if result.output.status == 'EXTRACTED':
                    self.assertEqual(result.output, ExtractionOutput.model_validate_json(RulesProvider().generate(text)), text[:80])
                    checked += 1
        self.assertGreater(checked, 20)


if __name__ == '__main__':
    unittest.main()

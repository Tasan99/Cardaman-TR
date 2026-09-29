"""v0.19 t5: the grounding ladder (spans.ground) and its reason codes.

The v019t4 runs (25 September 2026) rejected whole units whose duty the model had right: Tedbirler md. 46(1)
came back with a correct duty plus a "duty" copied from the marker-less sentence "... başlangıç tarihi hesabın
kapatıldığı tarihtir." (C11 and C12, twice each, after a hint that named the other candidate's marker); md.
21(2) came back as one candidate per list item; md. 24(5) and 24/A(3) without the condition of their own
sentence; md. 24(1) with "veyadoğum"; md. 31(1) with "isterilen". The answers below are those recorded
answers. Every stored field must stay an exact source substring, and an invented duty must stay rejected.
"""
import json
import unittest

from regchain.evidence import canonical_bytes
from regchain.extraction.grounding import CONDITIONS, qualifier_spans, required_qualifiers, verify
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import TURKISH_PROMPT
from regchain.extraction.schema import ExtractionOutput
from regchain.extraction.spans import (LOW_LEXICAL_OVERLAP, NO_SOURCE_SPAN, REASON_CODES, SEMANTIC_ONLY, SOURCE_BOUND_OK,
                                       STRUCTURED_MISMATCH, compact_span, field_rejection, ground, salvage, semantic_span)

from test_v019_extraction import Scripted, action_of, answer, candidate, unit
from test_v019_grounding_salvage import CONC_7_7_2

MD46_DUTY = ('her türlü ortamdaki, yükümlülüklerine ve işlemlerine ilişkin belgeleri düzenleme tarihinden, defter ve kayıtları ise '
             'son kayıt tarihinden; kimlik tespitine ilişkin belge ve kayıtları son işlem tarihinden itibaren sekiz yıl süre ile '
             'muhafaza etmek ve istenmesi halinde yetkililere ibraz etmek')
MD46_NOT_A_DUTY = 'kimlik tespitine ilişkin belgelerin muhafaza süresinin başlangıç tarihi hesabın kapatıldığı tarihtir'
MD24_ITEMS = ('a) Adı ve soyadına, ticaret siciline kayıtlı tüzel kişinin unvanına, diğer tüzel kişiler ve tüzel kişiliği olmayan '
              'teşekküllerin tam adına, b) Hesap numarasına, hesap numarasının bulunmadığı durumda işlemle ilgili referans numarasına, '
              'c) Adresi veya doğum yeri ve tarihi veya müşteri numarası, vatandaşlık numarası, pasaport numarası, vergi kimlik numarası '
              'gibi göndereni belirlemeye yarayan bilgilerden en az birine, yer verilmesi')


def records(result, code):
    return [d for d in result.diagnostics if d.get('code') == code]


def trace_of(result):
    """The grounding trace of the accepted answer (the 'complete' record) or of the rejection."""
    for d in reversed(result.diagnostics):
        if 'grounding' in d and d.get('code') in ('CANDIDATE_REQUIRES_LEGAL_REVIEW', 'GROUNDING_REJECTED'):
            return d['grounding']
    raise AssertionError('no grounding trace')


class Checks(unittest.TestCase):
    def accepted(self, text, result):
        """The salvage record of an accepted answer, after checking every stored field is a source span."""
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        canonical_bytes(list(result.diagnostics))                          # digest-safe: strings and ints only
        for value in result.output.obligations:
            for field in [value.subject, action_of(value), *value.conditions, *value.exceptions]:
                self.assertIn(field, text)
        verify(text, result.output)
        found = records(result, 'GROUNDING_SALVAGED')
        self.assertEqual(len(found), 1)
        self.assertEqual(set(found[0]) - {'attempt'}, {'stage', 'code', 'steps', 'fields'})
        return found[0]


class EnumerationDutyTests(Checks):
    def test_md46_duty_survives_a_marker_less_second_candidate(self):
        _, text = unit('46', 1)
        model = answer(candidate(text, 'Yükümlüler', 'MUST', MD46_DUTY),
                       candidate(text, 'Yükümlü nezdindeki hesaplarla ilgili', 'MUST', MD46_NOT_A_DUTY))
        provider = Scripted(model)
        result = extract(text, provider)
        self.assertEqual(provider.calls, 2)                                # the repair turn is still spent
        record = self.accepted(text, result)
        self.assertEqual(record['steps'], ['drop', 'qualifier'])
        [duty] = result.output.obligations
        self.assertEqual(duty.required_action, MD46_DUTY)
        self.assertEqual(duty.conditions, ['istenmesi halinde'])
        trace = trace_of(result)
        self.assertEqual(trace[0], {'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'structured'})
        self.assertEqual(trace[1], {'candidate': 1, 'code': STRUCTURED_MISMATCH, 'detail': 'NO_ALIGNED_MARKER'})

    def test_the_repair_hint_names_the_candidate_no_marker_governs(self):
        _, text = unit('46', 1)
        output = ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, 'Yükümlüler', 'MUST', MD46_DUTY),
                                                                   candidate(text, 'Yükümlü nezdindeki hesaplarla ilgili', 'MUST', MD46_NOT_A_DUTY)])
        with self.assertRaises(ValueError) as caught:
            verify(text, output)
        message = str(caught.exception)
        self.assertTrue(message.startswith('ACTION_ALIGNMENT'))
        self.assertIn('candidate 2 of 2', message)
        self.assertIn('no duty marker occurs in its sentence', message)

    def test_a_condition_after_a_coordinated_infinitive_is_demanded_from_ve(self):
        _, text = unit('46', 1)
        self.assertEqual(required_qualifiers(text, CONDITIONS), ['istenmesi halinde'])
        # What the rules baseline and the providers copy is unchanged (and still contains the demand).
        self.assertTrue(any(span.endswith('muhafaza etmek ve istenmesi halinde') for span in qualifier_spans(text, CONDITIONS)))

    def test_md21_list_items_become_the_one_conditioned_permission_duty(self):
        _, text = unit('21', 2)
        items = ['a) (Değişik: 28/12/2009-2009/15720 K.) Üçüncü tarafın, kimlik tespiti, kayıtların saklanması ve müşterinin tanınması '
                 'kuralının gereklerini sağlayacak diğer tedbirleri aldığından, yurt dışında yerleşik olması durumunda ise ayrıca aklama '
                 've terörün finansmanıyla mücadele alanındaki uluslararası standartlara uygun düzenleme ve denetlemelere tabi olduğundan',
                 'b) Kimlik tespitine ilişkin belgelerin onaylı örneklerinin (güvenilen kuruluş tarafından uzaktan kimlik tespiti yoluyla '
                 'sürekli iş ilişkisi tesis edilmesi halinde, alınan dijital ortamdaki görüntülerinin), talep edildiğinde üçüncü taraftan '
                 'derhal temin edileceğinden',
                 'c) (Ek:RG-25/12/2024-32763-C.K.-9305/11 md.) Üçüncü tarafça, bilgileri paylaşılan müşterinin kimlik tespitinin '
                 'basitleştirilmiş tedbirler kapsamında yapılmadığından, emin olunması şartıyla mümkündür.']
        model = answer(*[candidate(text, 'Üçüncü tarafa güvenilebilmesi', 'MUST', item) for item in items])
        result = extract(text, Scripted(model))
        record = self.accepted(text, result)
        self.assertEqual(record['steps'], ['merge', 'qualifier'])
        [duty] = result.output.obligations
        self.assertTrue(duty.required_action.startswith('a) (Değişik'))
        self.assertTrue(duty.required_action.endswith('yapılmadığından, emin olunması'))
        self.assertIn('yurt dışında yerleşik olması durumunda', duty.conditions)
        self.assertEqual([t['code'] for t in trace_of(result)], [SOURCE_BOUND_OK] * 3)
        self.assertEqual([t.get('merged_into') for t in trace_of(result)], [None, 0, 0])


class PassiveDutyTests(Checks):
    def test_md24_5_passive_duty_gets_its_own_sentence_condition(self):
        _, text = unit('24', 5)
        short = ('gönderici finansal kuruluştan gelen elektronik transferlerin reddedilmesi veya söz konusu finansal kuruluşla yapılan '
                 'işlemlerin sınırlandırılması ya da iş ilişkisine son verilmesi')
        result = extract(text, Scripted(answer(candidate(text, 'alıcı finansal kuruluş', 'MUST', short))))
        record = self.accepted(text, result)
        self.assertEqual(record['steps'], ['action', 'qualifier_source'])
        [duty] = result.output.obligations
        self.assertEqual(duty.required_action, short + ' hususları göz önünde bulundurulur')
        self.assertEqual(duty.conditions, ['Gönderilen mesajların sürekli olarak eksik bilgiler içermesi ve bu bilgilerin talep edildiği '
                                           'halde tamamlanmaması halinde'])

    def test_md24_1_words_run_together_are_the_source_span(self):
        _, text = unit('24', 1)
        subject = 'Onbeşbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında, gönderenin'
        joined = MD24_ITEMS.replace('Adresi veya doğum', 'Adresi veyadoğum')
        self.assertEqual(compact_span(joined, [text]), MD24_ITEMS)
        result = extract(text, Scripted(answer(candidate(text, subject, 'MUST', joined))))
        record = self.accepted(text, result)
        self.assertEqual(record['steps'], ['compact'])
        self.assertEqual(result.output.obligations[0].required_action, MD24_ITEMS)
        self.assertEqual(trace_of(result), [{'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'normalized'}])

    def test_24A_1_an_earlier_answer_is_kept_when_the_repair_made_it_worse(self):
        _, text = unit('24/A', 1)
        subject = 'Kripto varlık hizmet sağlayıcılar'
        first = text[text.index('tarafından aracılık'):text.index(' zorunlu olup')]
        skipped = first.replace(' cüzdan adresinin bulunmadığı durumda işlemle ilgili referans numarasına,', '')   # the recorded skip
        worse = 'kripto varlık transferlerinde her türlü bilgiyi derhal silmek'           # the repair invented a duty
        result = extract(text, Scripted(answer(candidate(text, subject, 'MUST', skipped)), answer(candidate(text, subject, 'MUST', worse))))
        record = self.accepted(text, result)
        self.assertEqual(record['attempt'], 1)
        self.assertEqual(record['steps'], ['overlap'])
        self.assertEqual(result.output.obligations[0].required_action, first)


class TypoTests(Checks):
    MD31_TAIL = 'istenilen usul, şekil ve sürede gecikmeksizin vermek ve gerekli kolaylığı sağlamak zorundadır'
    MD31_SUBJECT = 'Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar'

    def test_md31_1_one_misspelt_word_is_the_semantic_rung_and_stores_the_source(self):
        _, text = unit('31', 1)
        typo = self.MD31_TAIL.replace('istenilen', 'isterilen')
        self.assertEqual(semantic_span(typo, [text]), self.MD31_TAIL)
        result = extract(text, Scripted(answer(candidate(text, self.MD31_SUBJECT, 'MUST', typo))))
        record = self.accepted(text, result)
        self.assertEqual(record['steps'], ['semantic', 'span_repair'])
        action = result.output.obligations[0].required_action
        self.assertTrue(action.startswith('Başkanlık ve denetim elemanları tarafından istenilecek her türlü bilgi, belge'))
        self.assertNotIn('isterilen', action)
        self.assertEqual(trace_of(result), [{'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'semantic'}])

    def test_two_changed_words_or_a_negation_are_semantic_only_and_rejected(self):
        _, text = unit('31', 1)
        twice = self.MD31_TAIL.replace('istenilen', 'isterilen').replace('gecikmeksizin', 'gecikmeksizn')
        self.assertIsNone(semantic_span(twice, [text]))
        self.assertEqual(field_rejection(twice, [text]), SEMANTIC_ONLY)
        negated = self.MD31_TAIL.replace('vermek', 'vermemek')                # a two-letter negative suffix
        self.assertIsNone(semantic_span(negated, [text]))
        result = extract(text, Scripted(answer(candidate(text, self.MD31_SUBJECT, 'MUST', negated))))
        self.assertEqual(result.reason, 'GROUNDING_REJECTED')
        self.assertEqual(trace_of(result)[0], {'candidate': 0, 'code': SEMANTIC_ONLY, 'field': 'required_action'})

    def test_a_changed_number_is_never_a_misspelling(self):
        _, text = unit('46', 1)
        changed = MD46_DUTY.replace('sekiz yıl', 'dokuz yıl')
        self.assertIsNone(semantic_span(changed, [text]))
        self.assertIsNone(salvage(text, ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, 'Yükümlüler', 'MUST', changed)])))


class HallucinationTests(Checks):
    def test_an_invented_duty_is_rejected_with_no_source_span(self):
        _, text = unit('46', 1)
        invented = 'müşterilerin tüm yazışmalarını on beş yıl boyunca yurt dışında saklamak'
        result = extract(text, Scripted(answer(candidate(text, 'Yükümlüler', 'MUST', invented))))
        self.assertEqual(result.reason, 'GROUNDING_REJECTED')
        self.assertEqual(result.output.status, 'INSUFFICIENT_EVIDENCE')
        self.assertEqual(records(result, 'GROUNDING_SALVAGED'), [])
        final = records(result, 'GROUNDING_REJECTED')[-1]['grounding']
        self.assertEqual(final[0], {'candidate': 0, 'code': NO_SOURCE_SPAN, 'field': 'required_action'})
        self.assertEqual(final[-1]['candidate'], -1)
        # every rejected answer carries its own trace too
        self.assertTrue(all('grounding' in d for d in records(result, 'EVIDENCE_INVALID')))
        canonical_bytes(list(result.diagnostics))

    def test_an_invented_duty_next_to_a_true_one_is_left_out_and_logged(self):
        _, text = unit('46', 1)
        invented = 'müşterilerin tüm yazışmalarını on beş yıl boyunca yurt dışında saklamak'
        model = answer(candidate(text, 'Yükümlüler', 'MUST', MD46_DUTY), candidate(text, 'Yükümlüler', 'MUST', invented))
        result = extract(text, Scripted(model))
        self.accepted(text, result)
        self.assertEqual([action_of(c) for c in result.output.obligations], [MD46_DUTY])
        self.assertEqual(trace_of(result)[1], {'candidate': 1, 'code': NO_SOURCE_SPAN, 'field': 'required_action'})

    def test_a_marker_less_sentence_alone_is_still_rejected(self):
        _, text = unit('46', 1)
        result = extract(text, Scripted(answer(candidate(text, 'Yükümlü nezdindeki hesaplarla ilgili', 'MUST', MD46_NOT_A_DUTY))))
        self.assertEqual(result.reason, 'GROUNDING_REJECTED')
        final = records(result, 'GROUNDING_REJECTED')[-1]['grounding']
        self.assertEqual(final[0], {'candidate': 0, 'code': STRUCTURED_MISMATCH, 'detail': 'NO_ALIGNED_MARKER'})

    def test_scattered_source_words_are_low_lexical_overlap(self):
        _, text = unit('46', 1)
        scattered = 'kayıtları muhafaza etmek'
        self.assertEqual(field_rejection(scattered, [text]), LOW_LEXICAL_OVERLAP)
        output, record, trace = ground(text, ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, 'Yükümlüler', 'MUST', scattered)]))
        self.assertIsNone(output)
        self.assertIsNone(record)
        self.assertEqual(trace[0], {'candidate': 0, 'code': LOW_LEXICAL_OVERLAP, 'field': 'required_action'})

    def test_an_english_duty_is_not_merged_trimmed_or_given_a_condition(self):
        result = extract(CONC_7_7_2, Scripted(answer(candidate(CONC_7_7_2, 'A firm', 'MUST_NOT', 'claim the costs of recovering a debt from a customer'))))
        self.assertEqual(result.reason, 'GROUNDING_REJECTED')
        self.assertEqual(records(result, 'GROUNDING_REJECTED')[-1]['grounding'][-1],
                         {'candidate': -1, 'code': STRUCTURED_MISMATCH, 'detail': 'MISSING_QUALIFIER'})


class ReasonCodeTests(Checks):
    def test_an_answer_the_gate_accepts_as_it_came_is_source_bound_exact(self):
        _, text = unit('46', 1)
        model = answer(candidate(text, 'Yükümlüler', 'MUST', MD46_DUTY, conditions=['istenmesi halinde']))
        result = extract(text, Scripted(model))
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertEqual([d['code'] for d in result.diagnostics], ['CANDIDATE_REQUIRES_LEGAL_REVIEW'])
        self.assertEqual(result.diagnostics[-1]['grounding'], [{'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'exact'}])

    def test_every_reason_code_is_produced(self):
        _, text = unit('46', 1)
        seen = set()
        for action in [MD46_DUTY, 'müşterilerin tüm yazışmalarını on beş yıl boyunca yurt dışında saklamak', 'kayıtları muhafaza etmek',
                       MD46_DUTY.replace('muhafaza etmek', 'muhafaza etmemek').replace('ibraz etmek', 'ibraz etmemek')]:
            output = ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, 'Yükümlüler', 'MUST', action),
                                                                       candidate(text, 'Yükümlü nezdindeki hesaplarla ilgili', 'MUST', MD46_NOT_A_DUTY)])
            _, _, trace = ground(text, output)
            seen.update(entry['code'] for entry in trace)
            json.dumps(trace)
        self.assertEqual(seen, set(REASON_CODES))

    def test_a_condition_the_action_swallowed_half_of_is_filed_whole(self):
        _, text = unit('28', 3)
        # v019t3 rep8: the action began inside the condition and dropped its comma.
        action = ('yeni bilgi ve bulgular elde edildiği takdirde tekrar Şüpheli İşlem Bildirim Formu doldurulur ve daha önce yapılan '
                  'bildirime ek olduğu belirtilerek gecikmeksizin Başkanlığa gönderilir.')
        result = extract(text, Scripted(answer(candidate(text, 'Bildirimde bulunulan işlem', 'MUST', action))))
        record = self.accepted(text, result)
        self.assertEqual(record['steps'], ['normalize', 'qualifier_source'])
        [duty] = result.output.obligations
        self.assertTrue(duty.required_action.startswith('tekrar Şüpheli İşlem Bildirim Formu doldurulur'))
        self.assertEqual(duty.conditions, ['Bildirimde bulunulan işlemle ilgili olarak daha sonra yeni bilgi ve bulgular elde edildiği takdirde'])

    def test_the_v019_salvage_path_runs_first_and_unchanged(self):
        # A condition filed as an exception (CONC 7.7.2): phase A salvages it exactly as before.
        span = 'if it has no contractual right to claim such costs.'
        output = ExtractionOutput(status='EXTRACTED', obligations=[candidate(CONC_7_7_2, 'A firm', 'MUST_NOT',
                                                                            'claim the costs of recovering a debt from a customer', exceptions=[span])])
        salvaged, record, trace = ground(CONC_7_7_2, output)
        self.assertEqual(record['steps'], ['qualifier'])
        self.assertEqual(salvaged.obligations[0].conditions, [span])
        self.assertEqual(trace, [{'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'structured'}])

    def test_the_turkish_prompt_says_a_marker_less_sentence_and_a_list_item_are_no_candidates(self):
        self.assertIn('a sentence with none of these markers', TURKISH_PROMPT)
        self.assertIn('is ONE candidate', TURKISH_PROMPT)


if __name__ == '__main__':
    unittest.main()

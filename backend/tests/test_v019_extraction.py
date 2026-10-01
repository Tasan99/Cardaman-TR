"""v0.19 extraction hardening: truncated Turkish actions, quantities, and the duty structure.

The truncations below are the four the v0.18 final run shipped to the judge (Tedbirler
Yönetmeliği md. 26(2), 28(2), 5(3), 20(2)); the candidates are copied from its packets.
"""
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction.contract import EXAMPLE, materialize
from regchain.extraction.grounding import verify
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import RulesProvider
from regchain.extraction.quantities import compare, evidence_safe, parse_quantities, period_phrases
from regchain.extraction.schema import Candidate, ExtractionOutput
from regchain.extraction.spans import expand_action, repair_actions
from regchain.extraction.structure import LEGACY_KEYS, duty_payload, duty_quantities, structure_of

FIXTURE = Path(__file__).resolve().parents[2] / 'evaluation' / 'fixtures' / 'regulations' / 'tedbirler-200713012'
STR_DUTY = ('(1) Yükümlüler nezdinde veya bunlar aracılığıyla yapılan veya yapılmaya teşebbüs edilen işlemlere konu malvarlığının yasa dışı '
            'yollardan elde edildiğine veya yasa dışı amaçlarla kullanıldığına dair herhangi bir bilgi, şüphe veya şüpheyi gerektirecek bir '
            'hususun bulunması halinde bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi zorunludur.')
_SECTIONS = []


def tedbirler():
    """{article number: section} of the retained Tedbirler Yönetmeliği snapshot (loaded once)."""
    if not _SECTIONS:
        from regchain.pilot.sources import load_sources
        _SECTIONS.extend(load_sources(FIXTURE)[1])
    return {s['printed_label'].rsplit(' ', 1)[-1]: s for s in _SECTIONS if s['printed_label'].startswith('Yönetmelik 200713012 md. ')}


def unit(article: str, number: int) -> tuple[int, str]:
    from regchain.pilot.engine import split_units
    for offset, text in split_units(tedbirler()[article]['text']):
        if text.startswith('(%d) ' % number):
            return offset, text
    raise AssertionError(f'md. {article}({number}) not in the fixture')


def candidate(text, subject, modality, action, conditions=(), exceptions=()) -> Candidate:
    negative = modality.endswith('_NOT')
    return Candidate(source_quote=text, subject=subject, modality=modality, required_action=None if negative else action,
                     prohibited_action=action if negative else None, conditions=list(conditions), exceptions=list(exceptions),
                     confidence_score='0.5000')


def action_of(value: Candidate) -> str:
    return value.prohibited_action if value.modality.endswith('_NOT') else value.required_action


class Scripted:
    """Answers with fixed ExtractionOutput JSON, like a model that always cuts the same way."""
    name = 'scripted'
    model_version = 'fixture-v1'

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def generate(self, text):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return answer

    def repair(self, text, context, feedback):
        return self.generate(text)


def answer(*values: Candidate) -> str:
    return ExtractionOutput(status='EXTRACTED', obligations=list(values)).model_dump_json()


# (article, unit, subject, modality, model action, model conditions, repaired action): the v0.18 final run.
TRUNCATED = [
    ('26', 2, 'Yükümlüler', 'MUST_NOT', 'uygulayamazlar', [], 'basitleştirilmiş tedbirleri uygulayamazlar'),
    ('28', 2, 'Şüpheli işlemler', 'MUST', 'bildirilir', ['işleme ilişkin şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde'],
     'işleme ilişkin şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde Başkanlığa bildirilir'),
    ('5', 3, 'Sürekli iş ilişkisi tesisinde', 'MUST', 'bilgi alınır.', [], 'iş ilişkisinin amacı ve mahiyeti hakkında bilgi alınır.'),
    ('20', 2, 'Finansal kuruluşlar', 'MUST', 'uygun ve etkili tedbirleri almak zorundadır.', [],
     'yüz yüze olmayan işlemler yapılmasını mümkün kılan yöntem veya sistemleri kullanarak gerçekleştirilen sürekli iş ilişkisi tesisi, '
     'hesaba para yatırma, hesaptan para çekme ve elektronik transfer gibi işlemlere özel dikkat göstermek, müşterinin mali profiline ve '
     'faaliyetlerine uygun olmayan veya faaliyetleriyle ilgisi bulunmayan işlemleri yakından izlemek, tutar ve işlem sayısı limiti '
     'belirlemek de dâhil uygun ve etkili tedbirleri almak zorundadır.'),
]


class TruncatedActionRegressionTests(unittest.TestCase):
    def test_the_four_real_truncations_are_expanded_and_still_pass_the_gate(self):
        for article, number, subject, mode, cut, conditions, full in TRUNCATED:
            with self.subTest(article=article):
                _, text = unit(article, number)
                output = ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, subject, mode, cut, conditions)])
                verify(text, output)                                   # the gate accepted the cut action (the v0.18 defect)
                repaired, repairs = repair_actions(text, output)
                self.assertEqual(action_of(repaired.obligations[0]), full)
                self.assertEqual(len(repairs), 1)
                self.assertEqual({k: repairs[0][k] for k in ('stage', 'code', 'model_action', 'action')},
                                 {'stage': 'span_repair', 'code': 'ACTION_SPAN_EXPANDED', 'model_action': cut, 'action': full})
                self.assertEqual(repairs[0]['added'], full[:-len(cut)].strip())
                verify(text, repaired)
                # Everything else about the candidate is the model's.
                self.assertEqual(repaired.obligations[0].model_dump(exclude={'required_action', 'prohibited_action'}),
                                 output.obligations[0].model_dump(exclude={'required_action', 'prohibited_action'}))

    def test_the_pipeline_repairs_before_the_gate_and_records_it_on_success_only(self):
        _, text = unit('5', 3)
        cut = candidate(text, 'Sürekli iş ilişkisi tesisinde', 'MUST', 'bilgi alınır.')
        result = extract(text, Scripted(answer(cut)))
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertEqual(result.output.obligations[0].required_action, 'iş ilişkisinin amacı ve mahiyeti hakkında bilgi alınır.')
        self.assertEqual([d['code'] for d in result.diagnostics], ['ACTION_SPAN_EXPANDED', 'CANDIDATE_REQUIRES_LEGAL_REVIEW'])
        self.assertEqual({k: result.diagnostics[0][k] for k in ('attempt', 'stage', 'model_action', 'action', 'added')},
                         {'attempt': 1, 'stage': 'span_repair', 'model_action': 'bilgi alınır.',
                          'action': 'iş ilişkisinin amacı ve mahiyeti hakkında bilgi alınır.', 'added': 'iş ilişkisinin amacı ve mahiyeti hakkında'})
        # A repaired first answer that the gate still rejects leaves no span_repair record behind.
        wrong = candidate(text, 'Bakanlık', 'MUST', 'bilgi alınır.')
        failed = extract(text, Scripted(answer(wrong), answer(wrong)))
        self.assertEqual(failed.reason, 'GROUNDING_REJECTED')
        self.assertNotIn('span_repair', [d['stage'] for d in failed.diagnostics])

    def test_a_repair_the_gate_rejects_falls_back_to_the_model_span(self):
        text = 'Yükümlüler, bunu hiç kimseye açıklayamazlar.'
        model = candidate(text, 'Yükümlüler', 'MUST_NOT', 'bunu hiç kimseye açıklayamazlar.')
        broken = ExtractionOutput(status='EXTRACTED', obligations=[candidate(text, 'Yükümlüler', 'MUST_NOT', 'not in the source')])
        record = {'stage': 'span_repair', 'code': 'ACTION_SPAN_EXPANDED', 'candidate': 0, 'model_action': 'x', 'action': 'y', 'added': 'z'}
        with patch('regchain.extraction.pipeline.repair_actions', return_value=(broken, [record])):
            result = extract(text, Scripted(answer(model)))
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertEqual(result.output.obligations[0].prohibited_action, 'bunu hiç kimseye açıklayamazlar.')
        self.assertEqual([d['stage'] for d in result.diagnostics], ['complete'])

    def test_two_synthetic_truncations_are_repaired_within_their_clause(self):
        text = 'Yükümlüler, Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar.'
        new, note = expand_action(text, candidate(text, 'Yükümlüler', 'MUST_NOT', 'açıklayamazlar'))
        self.assertEqual(new, 'Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar')
        text = 'Yükümlüler, şüphe oluşması hâlinde, mahkeme kararları hariç, işlemi Başkanlığa bildirmek zorundadır.'
        new, note = expand_action(text, candidate(text, 'Yükümlüler', 'MUST', 'bildirmek'))
        self.assertEqual((new, note['added']), ('işlemi Başkanlığa bildirmek', 'işlemi Başkanlığa'))     # stops after "... hariç,"


class NoChangeTests(unittest.TestCase):
    def assertUnchanged(self, text, *values):
        output = ExtractionOutput(status='EXTRACTED', obligations=list(values))
        repaired, repairs = repair_actions(text, output)
        self.assertEqual(repairs, [])
        self.assertIs(repaired, output)

    def test_english_and_already_full_actions_are_left_alone(self):
        english = materialize(json.dumps(EXAMPLE), 'A firm must retain records.')
        self.assertEqual(repair_actions('A firm must retain records.', english), (english, []))
        self.assertUnchanged('A firm must record consent and must keep evidence.',
                             *ExtractionOutput.model_validate_json(RulesProvider().generate('A firm must record consent and must keep evidence.')).obligations)
        fixtures = [('Yükümlüler, bunu hiç kimseye açıklayamazlar.', 'MUST_NOT', 'bunu hiç kimseye açıklayamazlar.'),
                    ('(1) Yükümlüler kayıtları sekiz yıl saklamak zorundadır. Usûl ve esaslar yönetmelikle belirlenir.', 'MUST', 'kayıtları sekiz yıl saklamak'),
                    ('Yükümlüler, işlem yapanların kimliklerini tespit etmek zorundadır.', 'MUST', 'işlem yapanların kimliklerini tespit etmek'),
                    ('Yükümlüler, işlem yapanların kimliklerini tespit etmek zorundadır.', 'MUST', 'işlem yapanların kimliklerini tespit etmek zorundadır.'),
                    ('Yükümlüler, Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar.', 'MUST_NOT',
                     'Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar')]
        for text, mode, action in fixtures:
            with self.subTest(action=action):
                self.assertUnchanged(text, candidate(text, 'Yükümlüler', mode, action))
        aorist = 'Şüpheli işlemler on iş günü içinde Başkanlığa bildirilir.'
        self.assertUnchanged(aorist, candidate(aorist, 'Şüpheli işlemler', 'MUST', 'on iş günü içinde Başkanlığa bildirilir'))
        self.assertUnchanged(STR_DUTY, candidate(STR_DUTY, 'Yükümlüler', 'MUST', 'bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi',
                                                 ['şüphe veya şüpheyi gerektirecek bir hususun bulunması halinde']))

    def test_the_gate_still_rejects_what_it_rejected(self):
        # "tespit" does not end at the marker: it is not aligned, so it is neither repaired nor accepted.
        text = 'Yükümlüler, işlem yapanların kimliklerini tespit etmek zorundadır.'
        for action in ('tespit', 'kimliklerini tespit'):
            with self.subTest(action=action):
                value = candidate(text, 'Yükümlüler', 'MUST', action)
                self.assertUnchanged(text, value)
                self.assertIsNone(expand_action(text, value)[1])
                with self.assertRaises(ValueError):
                    verify(text, ExtractionOutput(status='EXTRACTED', obligations=[value]))
        # The live cut of 22 September 2026 still fails alignment and still needs the repair round.
        cut = candidate(STR_DUTY, 'Yükümlüler', 'MUST', 'bulunması halinde bu işlemlerin')
        self.assertUnchanged(STR_DUTY, cut)

    def test_safeguards_leave_a_visible_gap_alone(self):
        _, text = unit('42', 1)                                                # "... yolcu ile ilgili tespitler ..."
        value = candidate(text, 'Kendisinden açıklama talep edilen yolcu', 'MUST', 'tespitler bir tutanağa bağlanır.')
        self.assertEqual(expand_action(text, value)[1]['code'], 'ACTION_GAP_FUNCTION_WORDS')
        self.assertUnchanged(text, value)
        _, text = unit('5', 3)                                                 # the rules baseline's amendment-note subject
        value = candidate(text, '(Değişik: 28/12/2009-2009/15720 K.) Sürekli iş ilişkisi tesisinde', 'MUST',
                          'iş ilişkisinin amacı ve mahiyeti hakkında bilgi alınır')
        self.assertEqual(expand_action(text, value)[1]['code'], 'SUBJECT_NOT_IN_SENTENCE')
        self.assertUnchanged(text, value)
        text = 'Yükümlüler ile bunların çalışanları, bunu hiç kimseye açıklayamazlar.'
        value = candidate(text, 'Yükümlüler', 'MUST_NOT', 'açıklayamazlar')
        self.assertEqual(expand_action(text, value)[1]['code'], 'SUBJECT_COORDINATION_UNRESOLVED')
        # A coordinated obliged party is part of the subject, not of the action (md. 20(1)).
        _, text = unit('20', 1)
        self.assertUnchanged(text, candidate(text, 'Finansal kuruluşlar', 'MUST', text[text.index('yeni ve gelişen'):text.index(' zorundadır')]))
        # A case of the list before the duty (md. 5(1) e) "... tutar gözetmeksizin,") stays out of the action.
        _, text = unit('5', 1)
        self.assertUnchanged(text, candidate(text, 'Yükümlüler', 'MUST', text[text.index('kimliğe ilişkin'):text.index(' zorundadır')]))

    def test_a_sibling_duty_is_never_swallowed(self):
        text = 'Yükümlüler, kayıtları saklamak ve kayıtları ibraz etmek zorundadır.'
        whole = candidate(text, 'Yükümlüler', 'MUST', 'kayıtları saklamak ve kayıtları ibraz etmek')
        tail = candidate(text, 'Yükümlüler', 'MUST', 'ibraz etmek')
        repaired, repairs = repair_actions(text, ExtractionOutput(status='EXTRACTED', obligations=[whole, tail]))
        self.assertEqual(repairs, [])
        self.assertEqual(expand_action(text, tail)[0], 'kayıtları saklamak ve kayıtları ibraz etmek')   # without the sibling


class RulesBaselineTests(unittest.TestCase):
    def test_the_rules_provider_over_tedbirler_expands_nothing(self):
        from regchain.pilot.engine import split_units
        checked = 0
        for section in tedbirler().values():
            for _, text in split_units(section['text']):
                output = ExtractionOutput.model_validate_json(RulesProvider().generate(text))
                repaired, repairs = repair_actions(text, output)
                self.assertEqual(repairs, [], text[:80])
                self.assertIs(repaired, output)
                result = extract(text, RulesProvider())
                self.assertNotIn('span_repair', [d['stage'] for d in result.diagnostics])
                for value in output.obligations:
                    structure = structure_of(text, value.model_dump(mode='json'))
                    self.assertNotIn('STRUCTURE_FAILED', structure['flags'], text[:80])
                    self.assertIsNone((structure['action'] or {}).get('repair'), text[:80])
                    canonical_bytes(structure)                            # evidence-safe: no floats
                    checked += 1
        self.assertGreater(checked, 20)

    def test_the_review_json_carries_one_structure_per_candidate_and_the_gate_version_moved(self):
        import inspect
        from regchain.evidence import digest
        from regchain.extraction import service
        _, text = unit('28', 2)
        output = ExtractionOutput.model_validate_json(RulesProvider().generate(text))
        structures = service.structures(text, output)
        self.assertEqual(len(structures), len(output.obligations))
        self.assertEqual(structures[0]['deadline'][0]['text'], 'en geç on iş günü içinde')
        digest([{'structures': structures}])                                    # the run's output hash takes it
        self.assertEqual(service.structures(text, ExtractionOutput(status='NO_EXPLICIT_OBLIGATION')), [])
        # t6 review 1 (PROV-1): v3.4 named the sentence coverage step (spans.cover_sentences); t7 review 1 (PROV-P2-1): v3.5
        # names the per-candidate review (pipeline.second_reading / apply_review); the version moved again.
        self.assertIn("'grounding_version':'scoped-evidence-v3.5-candidate-review'", inspect.getsource(service))


class RepairPropertyTests(unittest.TestCase):
    def test_every_word_suffix_of_a_full_action_repairs_to_a_superstring_ending_in_place(self):
        from regchain.pilot.engine import split_units
        cases = 0
        for article in ('4', '5', '8', '20', '26', '27', '28', '42', '46'):
            for _, text in split_units(tedbirler()[article]['text']):
                full = ExtractionOutput.model_validate_json(RulesProvider().generate(text))
                if full.status != 'EXTRACTED':
                    continue
                for index, value in enumerate(full.obligations):
                    action = action_of(value)
                    starts = [m.start() for m in re.finditer(r'(?<=\s)\S', action)]
                    for start in starts[-6:]:
                        cut = action[start:]
                        field = 'prohibited_action' if value.modality.endswith('_NOT') else 'required_action'
                        values = list(full.obligations)
                        values[index] = Candidate.model_validate({**value.model_dump(), field: cut})
                        output = ExtractionOutput(status='EXTRACTED', obligations=values)
                        try:
                            verify(text, output)
                        except ValueError:
                            continue
                        repaired, _ = repair_actions(text, output)
                        new = action_of(repaired.obligations[index])
                        with self.subTest(article=article, cut=cut[:40]):
                            self.assertIn(new, text)
                            self.assertTrue(new.endswith(cut))
                            self.assertLessEqual(len(new), len(action) + len(value.subject) + 400)
                            verify(text, repaired)
                            self.assertEqual(repair_actions(text, repaired)[1], [])          # idempotent
                        cases += 1
        self.assertGreater(cases, 20)


class QuantityTests(unittest.TestCase):
    def test_parse_table(self):
        table = [
            ('15.000 TL', (15000, 'TL', 'money', None)),
            ('185.000 TL ve üzeri işlemler', (185000, 'TL', 'money', '>=')),
            ('75.000 Türk lirası', (75000, 'TL', 'money', None)),
            ('yüzseksenbeşbin TL', (185000, 'TL', 'money', None)),
            ('iki yüz elli bin lira', (250000, 'TL', 'money', None)),
            ('on iş günü', (10, 'iş günü', 'duration', None)),
            ('on beş gün', (15, 'gün', 'duration', None)),
            ('otuz gün', (30, 'gün', 'duration', None)),
            ('iki sene', (2, 'yıl', 'duration', None)),
            ('beş yıl', (5, 'yıl', 'duration', None)),
            ('sekiz yıl süre ile', (8, 'yıl', 'duration', None)),
            ('8 (sekiz) yıl', (8, 'yıl', 'duration', None)),
            ('2,5 milyon TL', (2500000, 'TL', 'money', None)),
            ('1.500,50 TL', (1500.5, 'TL', 'money', None)),
            ('en az beş yıl', (5, 'yıl', 'duration', '>=')),
            ('en fazla 30 gün', (30, 'gün', 'duration', '<=')),
            ("15.000 TL'yi aşan", (15000, 'TL', 'money', '>')),
            ("15.000 TL'nin altındaki", (15000, 'TL', 'money', '<')),
            ('%10', (10, '%', 'percent', None)),
            ('yüzde on beş', (15, '%', 'percent', None)),
            ('10 years', (10, 'yıl', 'duration', None)),
            ('within 5 business days', (5, 'iş günü', 'duration', None)),
            ('within ten working days', (10, 'iş günü', 'duration', None)),
            ('30 calendar days', (30, 'gün', 'duration', None)),
            ('24 hours', (24, 'saat', 'duration', None)),
            ('£15,000', (15000, 'GBP', 'money', None)),
            ('€ 1.000', (1000, 'EUR', 'money', None)),
            ('more than EUR 15,000', (15000, 'EUR', 'money', '>')),
            ('at least 10 years', (10, 'yıl', 'duration', '>=')),
        ]
        for text, expected in table:
            with self.subTest(text=text):
                found = parse_quantities(text)
                self.assertEqual(len(found), 1, found)
                q = found[0]
                self.assertEqual((q['amount'], q['unit'], q['unit_class'], q['comparator']), expected)
                self.assertEqual(text[q['start']:q['end']], q['text'])

    def test_days_leads_and_what_is_not_a_quantity(self):
        deadline = parse_quantities('en geç on iş günü içinde Başkanlığa')[0]
        self.assertEqual((deadline['days'], deadline['lead'], deadline['trail']), (14, 'en geç', 'içinde'))
        self.assertEqual(parse_quantities('beş yıl süreyle saklanır')[0]['lead'], 'süreyle')
        self.assertEqual(parse_quantities('within 30 days')[0]['lead'], 'within')
        self.assertEqual([q['days'] for q in parse_quantities('1 hafta, 2 ay, 1 yıl, 10 gün, 10 iş günü, 48 saat')], [7, 60, 365, 10, 14, 2])
        self.assertIsNone(parse_quantities('15.000 TL')[0]['days'])
        for text in ('yüz yüze olmayan işlemler', 'iki ayrı işlem', 'onay verilir', 'altında', 'bir işlem', '27 nci maddeye',
                     '(Mülga: 28/12/2009-2009/15720 K.)', 'on the same terms'):
            with self.subTest(text=text):
                self.assertEqual(parse_quantities(text), [])
        nested = parse_quantities('yüzseksenbeşbin TL (kripto varlık hizmet sağlayıcılar için onbeşbin TL) veya üzerinde olduğunda')
        self.assertEqual([(q['amount'], q['comparator'], q['parenthetical']) for q in nested], [(185000, '>=', False), (15000, '>=', True)])

    def test_period_phrases(self):
        text = 'İşlemler ay sonunda toplu olarak bildirilir; acil olanlar derhal, diğerleri en kısa sürede ve işlemden önce kontrol edilir.'
        self.assertEqual([(p['text'], p['kind'], p['days']) for p in period_phrases(text)],
                         [('ay sonunda', 'PERIOD_END', 30), ('toplu olarak', 'BATCH', None), ('derhal', 'IMMEDIATE', 0),
                          ('en kısa sürede', 'ASAP', None), ('işlemden önce', 'BEFORE', 0)])
        self.assertEqual(period_phrases('iş ilişkisi tesisinden veya işlem yapılmadan önce')[0]['text'], 'tesisinden veya işlem yapılmadan önce')
        self.assertEqual([p['kind'] for p in period_phrases('ayın son iş günü; o ay sonunda; dönem sonunda; aynı gün içinde; gecikmeksizin')],
                         ['PERIOD_END', 'PERIOD_END', 'PERIOD_END', 'IMMEDIATE', 'IMMEDIATE'])
        self.assertEqual([p['kind'] for p in period_phrases('at the end of the month in batches, immediately, without undue delay, '
                                                            'as soon as possible, before the transaction')],
                         ['PERIOD_END', 'BATCH', 'IMMEDIATE', 'IMMEDIATE', 'ASAP', 'BEFORE'])
        for text in ('hemen hemen tüm işlemler', 'üç ay sonra', 'hafta sonu', 'Daha önce elde edilen'):
            with self.subTest(text=text):
                self.assertEqual(period_phrases(text), [])
        # A long "...den veya ...den veya" run once backtracked exponentially (60 repeats never returned).
        chain = 'Yükümlüler ' + 'işlemden veya ' * 300 + 'bu önce bildirir. Yükümlüler hesaptan veya işlemden önce bildirir.'
        self.assertEqual([(p['text'], p['kind']) for p in period_phrases(chain)], [('hesaptan veya işlemden önce', 'BEFORE')])
        for p in period_phrases(text):
            self.assertEqual(text[p['start']:p['end']], p['text'])

    def test_compare_directions(self):
        q = lambda text: parse_quantities(text)[0]
        ten_business = q('on iş günü')
        self.assertEqual(compare(ten_business, q('10 iş günü'), 'max'), 'SAME')
        self.assertEqual(compare(ten_business, q('5 gün'), 'max'), 'STRICTER')
        self.assertEqual(compare(ten_business, q('30 gün'), 'max'), 'WEAKER')
        eight = q('sekiz yıl')
        self.assertEqual(compare(eight, q('10 yıl'), 'min'), 'STRICTER')
        self.assertEqual(compare(eight, q('beş yıl'), 'min'), 'WEAKER')
        self.assertEqual(compare(eight, q('8 sene'), 'min'), 'SAME')
        floor = q('185.000 TL ve üzeri')
        self.assertEqual(compare(floor, q('250.000 TL üzerindeki'), 'floor'), 'WEAKER')
        self.assertEqual(compare(floor, q('100.000 TL ve üzeri'), 'floor'), 'STRICTER')
        self.assertEqual(compare(floor, q('185.000 TL ve üzeri'), 'floor'), 'SAME')
        self.assertEqual(compare(floor, q("185.000 TL'yi aşan"), 'floor'), 'WEAKER')          # misses 185.000 itself
        self.assertEqual(compare(floor, q('€ 185.000'), 'floor'), 'INCOMPARABLE')             # another currency
        self.assertEqual(compare(floor, q('10 gün'), 'floor'), 'INCOMPARABLE')                # another class
        self.assertEqual(compare(ten_business, period_phrases('ay sonunda')[0], 'max'), 'INCOMPARABLE')
        self.assertEqual(compare(ten_business, q('5 gün'), 'sideways'), 'INCOMPARABLE')
        # Evidence-safe records (decimal strings) compare the same way.
        self.assertEqual(compare(evidence_safe(q('bir iş günü')), evidence_safe(q('24 saat')), 'max'), 'STRICTER')   # 1.4 against 1 day
        self.assertEqual(compare(evidence_safe(q('bir iş günü')), evidence_safe(q('36 saat')), 'max'), 'WEAKER')     # against 1.5 days
        self.assertEqual(evidence_safe({'a': 1.4, 'b': [14.000000000000002], 'c': 'x'}), {'a': '1.4', 'b': [14], 'c': 'x'})


class StructureTests(unittest.TestCase):
    def structure(self, article, number, subject, mode, action, conditions=()):
        offset, text = unit(article, number)
        value = candidate(text, subject, mode, action, conditions).model_dump(mode='json')
        structure = structure_of(text, value, offset)
        self.assertSpansHold(text, structure)
        canonical_bytes(structure)
        return text, value, structure

    def assertSpansHold(self, text, structure):
        spans = [structure[k] for k in ('subject', 'modal', 'action', 'source_span') if structure[k]]
        spans += [s for k in ('conditions', 'exceptions', 'deadline', 'threshold') for s in structure[k]]
        for span in spans:
            self.assertEqual(text[span['start']:span['end']], span['text'])

    def test_md_26_2_carries_the_risk_condition_and_the_repaired_action(self):
        _, value, s = self.structure('26', 2, 'Yükümlüler', 'MUST_NOT', 'uygulayamazlar')
        self.assertEqual(s['version'], 'structure-v1')
        self.assertEqual((s['action']['text'], s['action']['repair']), ('basitleştirilmiş tedbirleri uygulayamazlar', 'uygulayamazlar'))
        self.assertEqual((s['modal']['text'], s['modal']['polarity'], s['modal']['force']), ('uygulayamazlar', 'NEGATIVE', 'MUST'))
        self.assertEqual(s['object'], {'text': 'basitleştirilmiş tedbirleri', 'heuristic': True})
        self.assertIn({'text': 'işlem nedeniyle aklama veya terörün finansmanı riskinin oluşabileceği durumlarda', 'start': 15, 'end': 95,
                       'origin': 'marker'}, s['conditions'])
        self.assertIn('DROPPED_AFTER_ACTION', s['flags'])                       # "... göz önünde bulundururlar"
        self.assertEqual(s['unit_offset'], unit('26', 2)[0])

    def test_md_28_2_deadline_and_md_46_1_retention(self):
        _, _, s = self.structure('28', 2, 'Şüpheli işlemler', 'MUST', 'bildirilir',
                                 ['işleme ilişkin şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde'])
        deadline, = s['deadline']
        self.assertEqual({k: deadline[k] for k in ('text', 'kind', 'amount', 'unit', 'days', 'direction', 'anchor')},
                         {'text': 'en geç on iş günü içinde', 'kind': 'DEADLINE', 'amount': 10, 'unit': 'iş günü', 'days': 14,
                          'direction': 'max', 'anchor': 'işleme ilişkin şüphenin oluştuğu tarihten itibaren'})
        _, text = unit('46', 1)
        action = text[text.index('her türlü'):text.index(' zorundadır')]
        _, _, s = self.structure('46', 1, 'Yükümlüler', 'MUST', action)
        retention, = s['deadline']
        self.assertEqual({k: retention[k] for k in ('text', 'kind', 'amount', 'unit', 'direction', 'anchor')},
                         {'text': 'sekiz yıl süre ile', 'kind': 'DURATION', 'amount': 8, 'unit': 'yıl', 'direction': 'min',
                          'anchor': 'son işlem tarihinden itibaren'})
        self.assertIn({'text': 'istenmesi halinde', 'start': text.index('istenmesi halinde'),
                       'end': text.index('istenmesi halinde') + len('istenmesi halinde'), 'origin': 'marker'}, s['conditions'])

    def test_md_5_thresholds_and_before_md_42_2_enumeration_md_28_3_immediate(self):
        _, text = unit('5', 1)
        _, _, s = self.structure('5', 1, 'Yükümlüler', 'MUST', text[text.index('kimliğe ilişkin'):text.index(' zorundadır')])
        self.assertEqual([(t['amount'], t['unit'], t['comparator'], t['direction'], t['parenthetical']) for t in s['threshold']],
                         [(185000, 'TL', '>=', 'floor', False), (15000, 'TL', '>=', 'floor', True), (15000, 'TL', '>=', 'floor', False)])
        self.assertEqual(len(s['enumeration']), 5)                                # a)-e); the repealed ç) is skipped
        self.assertEqual(s['enumeration'][-1], 'Daha önce elde edilen müşteri kimlik bilgilerinin yeterliliği ve doğruluğu konusunda şüphe '
                                               'olduğunda tutar gözetmeksizin')
        self.assertEqual([q['amount'] for q in duty_quantities(s)], [185000, 15000])     # the bracketed crypto variant is left out
        _, _, s = self.structure('5', 2, 'Kimlik tespiti', 'MUST', 'iş ilişkisi tesisinden veya işlem yapılmadan önce tamamlanır.')
        self.assertEqual([(d['kind'], d['direction'], d['days']) for d in s['deadline']], [('BEFORE', 'max', 0)])
        _, _, s = self.structure('42', 2, 'Düzenlenen tutanak', 'MUST', 'asgari aşağıdaki bilgilerin yer alması zorunludur')
        self.assertEqual(len(s['enumeration']), 10)
        self.assertEqual((s['enumeration'][0], s['enumeration'][-1]),
                         ('Tutanağın düzenlendiği yer ve tarih', 'Tutanağı düzenleyen görevlinin adı, soyadı, unvanı, sicil numarası ve imzası'))
        self.assertEqual((s['subject']['word_text'], s['subject']['word_complete']), ('Düzenlenen tutanakta', False))
        self.assertIn('SUBJECT_WORD_CUT', s['flags'])
        self.assertNotIn('DROPPED_AFTER_ACTION', s['flags'])
        _, text = unit('28', 3)
        _, _, s = self.structure('28', 3, 'Bildirimde', 'MUST', text[text.index('bulunulan'):text.index(' Başkanlığa gönderilir')] + ' Başkanlığa gönderilir')
        self.assertEqual([(d['text'], d['kind']) for d in s['deadline']], [('gecikmeksizin', 'IMMEDIATE')])

    def test_english_structure_and_the_failure_shape(self):
        text = '(1) Where a customer is high risk, a firm must retain records for at least five years and report suspicions within 10 business days.'
        value = candidate(text, 'a firm', 'MUST', 'retain records for at least five years and report suspicions within 10 business days.').model_dump(mode='json')
        s = structure_of(text, value)
        self.assertSpansHold(text, s)
        self.assertEqual([(d['kind'], d['amount'], d['unit'], d['direction']) for d in s['deadline']],
                         [('DURATION', 5, 'yıl', 'min'), ('DEADLINE', 10, 'iş günü', 'max')])
        self.assertEqual(s['conditions'][0]['origin'], 'marker')
        self.assertIsNone(s['action']['repair'])
        failed = structure_of('Yükümlüler bildirir.', {'modality': 5, 'subject': 'Yükümlüler'})
        self.assertEqual((failed['version'], failed['flags'], failed['action'], failed['deadline']), ('structure-v1', ['STRUCTURE_FAILED'], None, []))
        self.assertEqual(duty_quantities(failed), [])

    def test_duty_payload_shape(self):
        text, value, s = self.structure('28', 2, 'Şüpheli işlemler', 'MUST', 'bildirilir',
                                        ['işleme ilişkin şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde'])
        legacy = duty_payload(value, None)
        self.assertEqual(set(legacy), {*LEGACY_KEYS, 'elements'})
        self.assertEqual(legacy['elements'], [{'id': 'action', 'kind': 'action', 'text': 'bildirilir'}])
        self.assertEqual(legacy['required_action'], 'bildirilir')
        payload = duty_payload(value, s)
        self.assertTrue(set(LEGACY_KEYS) <= set(payload))
        self.assertFalse({'text', 'passage'} & set(payload))
        self.assertEqual(payload['required_action'], 'işleme ilişkin şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde Başkanlığa bildirilir')
        self.assertEqual(payload['deadline'], 'en geç on iş günü içinde (işleme ilişkin şüphenin oluştuğu tarihten itibaren)')
        self.assertEqual(payload['source_sentence'], 'Şüpheli işlemler, işleme ilişkin şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde Başkanlığa bildirilir.')
        # Short round 3: the subject is an element too (a scope element: never a gap); the repaired action already
        # holds the condition, and the heuristic object carries the deadline, so neither is its own element.
        self.assertEqual([e['id'] for e in payload['elements']], ['action', 'subject', 'deadline_1'])
        self.assertNotIn('threshold', payload)
        self.assertNotIn('items', payload)
        self.assertEqual(duty_payload(value, structure_of('x', {'modality': 5})), legacy)   # a failed structure reads as none
        _, text = unit('5', 1)
        _, value, s = self.structure('5', 1, 'Yükümlüler', 'MUST', text[text.index('kimliğe ilişkin'):text.index(' zorundadır')])
        payload = duty_payload(value, s)
        self.assertEqual([e['id'] for e in payload['elements']],
                         ['act_1', 'act_2', 'act_3', 'subject', 'threshold_1', 'threshold_2', 'item_1', 'item_2', 'item_3', 'item_4', 'item_5'])
        # v0.19 COVERS gate: md. 5(1) coordinates three verb clauses (almak ve ... teyit etmek suretiyle ... tespit etmek
        # ve ... tedbirleri almak); each is its own element, an exact substring of the action.
        self.assertTrue(all(e['text'] in payload['required_action'] for e in payload['elements'] if e['kind'] == 'action'))
        self.assertEqual(len(payload['conditions']), len(set(payload['conditions'])))
        _, _, s = self.structure('42', 2, 'Düzenlenen tutanak', 'MUST', 'asgari aşağıdaki bilgilerin yer alması zorunludur')
        payload = duty_payload(value, s)
        self.assertLessEqual(len(payload['items']), 12)
        self.assertLessEqual(len(payload['source_sentence']), 600)
        self.assertNotIn('(Değişik', payload['source_sentence'])


class RestatementTests(unittest.TestCase):
    def test_the_repaired_c14_duty_is_restated_by_the_expected_policy_sentence(self):
        from regchain.pilot.engine import restates_prohibition
        offset, text = unit('26', 2)
        model = candidate(text, 'Yükümlüler', 'MUST_NOT', 'uygulayamazlar')
        repaired, _ = repair_actions(text, ExtractionOutput(status='EXTRACTED', obligations=[model]))
        value = repaired.obligations[0].model_dump(mode='json')
        duty = duty_payload(value, structure_of(text, value, offset))
        self.assertEqual((duty['prohibited_action'], duty['conditions']),
                         ('basitleştirilmiş tedbirleri uygulayamazlar', ['işlem nedeniyle aklama veya terörün finansmanı riskinin oluşabileceği durumlarda']))
        quote = ('Aklama veya terörün finansmanı riski oluşabilecek durumlarda basitleştirilmiş tedbir uygulanmaz ve işlem şüpheli işlem '
                 'değerlendirmesine konu edilir.')
        self.assertTrue(restates_prohibition(duty, quote))
        # The truncated v0.18 duty was not restated: this is the C14 false conflict.
        self.assertFalse(restates_prohibition(duty_payload(model.model_dump(mode='json'), None), quote))


if __name__ == '__main__':
    unittest.main()


class ExtractionDiagnosticsDigestTests(unittest.TestCase):
    """Live smoke run 2026-09-24 (C12, C14): the v0.19 call records carry tokens_per_second and gpu_fraction as
    floats; copied into the extraction diagnostics they made make_event refuse the whole packet."""

    def test_call_records_with_measured_floats_stay_digest_safe(self):
        from regchain.evidence import digest
        from regchain.extraction.pipeline import extract
        from regchain.extraction.providers import RulesProvider

        class Measured(RulesProvider):
            def generate(self, text):
                self.last_calls.append({'elapsed_ms': 12, 'tokens_per_second': 64.1, 'prompt_tokens_per_second': 9836.57,
                                        'gpu_fraction': 1.0, 'processor': '100% GPU'})
                return super().generate(text)

        provider = Measured()
        provider.last_calls = []
        result = extract('Yükümlüler, kimlik tespitine ilişkin belgeleri sekiz yıl süre ile muhafaza etmek zorundadır.', provider)
        calls = [d for d in result.diagnostics if d.get('calls')]
        self.assertTrue(calls, 'the complete diagnostic carries the call records')
        self.assertEqual(calls[0]['calls'][0]['tokens_per_second'], '64.1')
        self.assertEqual(calls[0]['calls'][0]['gpu_fraction'], 1)
        digest(list(result.diagnostics))                                  # raises on any float

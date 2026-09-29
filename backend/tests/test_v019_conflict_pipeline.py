"""v0.19 coverage/conflict pipeline (COVERAGE_PIPELINE=v19): deterministic pre-check, fast classifier,
thinking verifier only on a possible conflict, element- and quantity-aware aggregation; and the v0.18
path's reliability fixes (CONFLICT_UNCONFIRMED, non-object JSON from enrich/draft, per-row timings).

The duties are cut from the retained Tedbirler Yönetmeliği snapshot and located with the WS1
structure; the passages are the planted ones of the evaluation fixtures (aml-conflict.md). Every
model is a fake: `_chat` answers by the schema it is given (label = fast, conflict = verifier,
items = relevance screen).
"""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction.providers import AI_CONTEXT, AI_TASK, AI_UNCACHED, ProviderFailure, failure
from regchain.extraction.schema import Candidate, ExtractionOutput
from regchain.extraction.spans import repair_actions
from regchain.extraction.structure import duty_payload, structure_of
from regchain.pilot import conflict as precheck
from regchain.pilot.engine import (DEFAULT_SETTINGS, STAGES, analyze, autonomous_gate, coverage_of_v19, draft_failure_code, passages_v19,
                                   pipeline_settings, precheck_passage, prompt_registry, settings_of, split_units)
from regchain.pilot.schema import PolicyCheck
from test_pilot import FixtureProvider, company, policies, sections

FIXTURE = Path(__file__).resolve().parents[2] / 'evaluation' / 'fixtures' / 'regulations' / 'tedbirler-200713012'
_SECTIONS = []
# The planted passages of evaluation/fixtures/policies/aml-conflict.md and aml-mixed.md.
KYC_SKIP = ("Müşteri deneyimini hızlandırmak için, tek seferlik ve tutarı 15.000 TL'nin altında kalan havale işlemlerinde kimlik tespiti ve "
            "teyit adımı atlanır; müşteri yalnızca telefon numarasını doğrular. Bu istisna kampanya dönemlerinde bütün tutarlar için uygulanabilir.")
KYC_SKIP_SPAN = 'Bu istisna kampanya dönemlerinde bütün tutarlar için uygulanabilir.'
RESTATED = ('Aklama veya terörün finansmanı riski oluşabilecek durumlarda basitleştirilmiş tedbir uygulanmaz ve işlem şüpheli işlem '
            'değerlendirmesine konu edilir.')
MONTH_END = ("Şüpheli işlem formları uyum birimince her ayın son iş günü toplu olarak değerlendirilir ve uygun görülenler o ay sonunda "
             "MASAK'a topluca bildirilir.")
FIVE_YEARS = ('Müşteri işlem kayıtları, kimlik tespiti belgeleri ve hesap hareketleri işlem tarihinden itibaren beş yıl saklanır; beş yılın '
              'sonunda imha edilir.')
NOISE = ['Parolalar karmaşık olur ve düzenli değiştirilir.', 'Ofis girişinde kartlı geçiş sistemi kullanılır.',
         'Personel eğitim takvimi insan kaynakları biriminde tutulur.', 'Toplantı odaları takvim üzerinden ayrılır.',
         'Kurum logosu yalın ve sade biçimde kullanılır.']


def tedbirler():
    if not _SECTIONS:
        from regchain.pilot.sources import load_sources
        _SECTIONS.extend(load_sources(FIXTURE)[1])
    return {s['printed_label'].rsplit(' ', 1)[-1]: s for s in _SECTIONS if s['printed_label'].startswith('Yönetmelik 200713012 md. ')}


def unit(article, number):
    for offset, text in split_units(tedbirler()[article]['text']):
        if text.startswith('(%d) ' % number):
            return offset, text
    raise AssertionError(f'md. {article}({number}) not in the fixture')


def candidate(text, subject, modality, action, conditions=()):
    negative = modality.endswith('_NOT')
    return Candidate(source_quote=text, subject=subject, modality=modality, required_action=None if negative else action,
                     prohibited_action=action if negative else None, conditions=list(conditions), exceptions=[], confidence_score='0.5000')


def duty(article, number, subject, modality, action, conditions=(), repair=False):
    """(duty payload, labelled quantities) of one Tedbirler duty, as analyze() hands them to the v19 judge."""
    offset, text = unit(article, number)
    action = action(text) if callable(action) else action
    value = candidate(text, subject, modality, action, conditions)
    if repair:
        value = repair_actions(text, ExtractionOutput(status='EXTRACTED', obligations=[value]))[0].obligations[0]
    value = value.model_dump(mode='json')
    structure = structure_of(text, value, offset)
    return duty_payload(value, structure), precheck.labelled_quantities(structure)


def md28():
    return duty('28', 2, 'Şüpheli işlemler', 'MUST', 'bildirilir', ['işleme ilişkin şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde'])


def md46():
    return duty('46', 1, 'Yükümlüler', 'MUST', lambda t: t[t.index('her türlü'):t.index(' zorundadır')])


def md5_1():
    return duty('5', 1, 'Yükümlüler', 'MUST', lambda t: t[t.index('kimliğe ilişkin'):t.index(' zorundadır')])


def md5_2():
    return duty('5', 2, 'Kimlik tespiti', 'MUST', 'iş ilişkisi tesisinden veya işlem yapılmadan önce tamamlanır.')


def md26(repaired=True):
    """Tedbirler md. 26(2), C14: the v0.18 extraction kept only "uygulayamazlar"."""
    if repaired:
        return duty('26', 2, 'Yükümlüler', 'MUST_NOT', 'uygulayamazlar', repair=True)
    _, text = unit('26', 2)
    return duty_payload(candidate(text, 'Yükümlüler', 'MUST_NOT', 'uygulayamazlar').model_dump(mode='json'), None), []


def fast(label, quote='', covered=('action',), missing=(), reason='fast reading'):
    return {'label': label, 'quote': quote, 'covered_elements': list(covered), 'missing_elements': list(missing), 'reason': reason}


def verdict(conflict=False, kind='NONE', span='', relation='UNRELATED', support='', confidence='HIGH', rationale='verifier reading'):
    return {'conflict': conflict, 'contradiction_type': kind, 'contradiction_span': span, 'regulation_requirement': 'the duty',
            'policy_statement': 'the passage', 'confidence': confidence, 'relation_if_no_conflict': relation, 'support_quote': support,
            'rationale': rationale}


class V19(FixtureProvider):
    """Answers the v0.19 questions by schema; `fast`, `verify`, `screen` map a passage text to an answer (or an exception)."""

    def __init__(self, fast=None, verify=None, screen=None):
        self.fast_answer, self.verify_answer, self.screen_answer = fast, verify, screen
        self.calls = []

    def reply(self, answer, payload):
        value = answer(payload['passage']) if callable(answer) else answer
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)

    def _chat(self, prompt, payload, schema):
        fields = schema['properties']
        if 'label' in fields:
            self.calls.append({'kind': 'fast', 'task': AI_TASK.get(), 'passage': payload['passage']})
            return self.reply(self.fast_answer or (lambda text: fast('SUPPORTS', text)), payload)
        if 'conflict' in fields:
            self.calls.append({'kind': 'verify', 'task': AI_TASK.get(), 'passage': payload['passage'], 'uncached': AI_UNCACHED.get(),
                               'payload': payload})
            return self.reply(self.verify_answer or verdict(relation='UNRELATED'), payload)
        if 'items' in fields:
            self.calls.append({'kind': 'screen', 'task': AI_TASK.get(), 'ids': [p['text'] for p in payload['passages']]})
            topic = self.screen_answer or (lambda text: 'SAME_SUBJECT')
            return json.dumps({'items': [{'id': p['id'], 'topic': topic(p['text'])} for p in payload['passages']]})
        if 'clause' in fields:
            self.calls.append({'kind': 'draft', 'task': AI_TASK.get(), 'context': dict(AI_CONTEXT.get() or {})})
        return super()._chat(prompt, payload, schema)

    def kinds(self, kind):
        return [c for c in self.calls if c['kind'] == kind]


def rows(*texts):
    return [{'source_id': f'p{i}', 'text': text} for i, text in enumerate(texts, 1)]


def run(provider, payload, quantities, texts, controls=frozenset(), screen=False):
    """(relations by source id, results by source id, (coverage, reason, control coverage, flags))."""
    checks, _, results, _, _ = passages_v19(provider, payload, rows(*texts), controls, quantities, 1, 's', 'o', screen)
    outcome = coverage_of_v19(checks, results, controls, quantities, payload.get('elements'))
    canonical_bytes(results)                                     # packets forbid floats: the records must be canonical
    return {c.source_id: c.relation for c in checks}, {r['source_id']: r for r in results}, outcome


def signals(payload, text, quantities=()):
    return [(s['type'], s['strength'], s['text']) for s in precheck.structural_signals(payload, text, quantities)]


class ConflictKeywordSafetyTests(unittest.TestCase):
    MUST = {'modality': 'MUST', 'required_action': 'kimlik tespiti yapmak'}
    MUST_NOT = {'modality': 'MUST_NOT', 'prohibited_action': 'anonim hesap açamazlar'}

    def test_waiver_exemption_and_limiter_wording_are_strong_signals(self):
        cases = [('Kimlik tespiti adımı atlanır.', 'WAIVER', 'atlanır'), ('Bu işlemlerde teyit uygulanmaz.', 'WAIVER', 'uygulanmaz'),
                 ('Kurumsal müşteriler hariç tutulur.', 'EXEMPTION', 'hariç'), ('Bu istisna kampanyada geçerlidir.', 'EXEMPTION', 'istisna'),
                 ('Teyit zorunlu değildir.', 'WAIVER', 'zorunlu değildir'), ('Kimlik yalnızca şubede tespit edilir.', 'LIMITER', 'yalnızca'),
                 ('Identity checks are skipped for small payments.', 'WAIVER', 'skipped'),
                 ('Customers are exempt from verification.', 'EXEMPTION', 'exempt')]
        for text, kind, word in cases:
            with self.subTest(text=text):
                self.assertIn((kind, 'strong', word), signals(self.MUST, text))

    def test_permission_of_the_prohibited_and_weaker_modality_depend_on_the_duty(self):
        self.assertIn(('PERMISSION_OF_PROHIBITED', 'strong', 'açılabilir'), signals(self.MUST_NOT, 'Talep halinde anonim hesap açılabilir.'))
        self.assertIn(('PERMISSION_OF_PROHIBITED', 'strong', 'may'), signals(self.MUST_NOT, 'Staff may open anonymous accounts.'))
        # The same permissive verb against a duty to act is no permission of a prohibited act.
        self.assertNotIn('PERMISSION_OF_PROHIBITED', [s[0] for s in signals(self.MUST, 'Talep halinde anonim hesap açılabilir.')])
        self.assertIn(('WEAKER_MODALITY', 'strong', 'takdirine'), signals(self.MUST, 'Kimlik tespiti şube müdürünün takdirine bırakılır.'))
        self.assertIn(('WEAKER_MODALITY', 'strong', 'should'), signals(self.MUST, 'Staff should verify identity where convenient.'))
        self.assertNotIn('WEAKER_MODALITY', [s[0] for s in signals(self.MUST_NOT, 'Staff should verify identity.')])

    def test_negation_and_prohibition_are_weak_and_abroad_is_no_exception(self):
        self.assertEqual(signals(self.MUST, 'Kimlik bilgileri üçüncü kişilere verilmez.'), [('NEGATION', 'weak', 'verilmez')])
        self.assertEqual(signals(self.MUST, 'Şifreler e-postayla paylaşılamaz.'), [('PROHIBITION', 'weak', 'paylaşılamaz')])
        self.assertEqual(signals(self.MUST, 'Merkezi yurt dışında bulunan şubeler de kimlik tespiti yapar.'), [])
        self.assertEqual(precheck.strong(precheck.structural_signals(self.MUST, 'Kimlik bilgileri üçüncü kişilere verilmez.')), [])

    def test_deadline_period_and_threshold_numbers_are_compared_with_the_duty(self):
        payload, quantities = md28()
        self.assertEqual([q['element'] for q in quantities], ['deadline_1'])
        self.assertIn(('DEADLINE_MISMATCH', 'strong', 'o ay sonunda'), signals(payload, MONTH_END, quantities))
        self.assertIn(('DEADLINE_MISMATCH', 'strong', 'on beş iş günü'),
                      signals(payload, 'Şüpheli işlemler en geç on beş iş günü içinde Başkanlığa bildirilir.', quantities))
        self.assertEqual(signals(payload, 'Şüpheli işlemler en geç beş iş günü içinde Başkanlığa bildirilir.', quantities),
                         [('STRICTER_QUANTITY', 'weak', 'beş iş günü')])
        self.assertEqual(signals(payload, 'Şüpheli işlemler en geç on iş günü içinde Başkanlığa bildirilir.', quantities), [])
        payload, quantities = md46()
        self.assertIn(('DEADLINE_MISMATCH', 'strong', 'beş yıl'), signals(payload, FIVE_YEARS, quantities))
        self.assertEqual(signals(payload, 'Kimlik belgeleri on yıl süreyle saklanır.', quantities), [('STRICTER_QUANTITY', 'weak', 'on yıl')])
        payload, quantities = md5_1()
        self.assertEqual([(q['element'], q['amount']) for q in quantities], [('threshold_1', 185000), ('threshold_2', 15000)])
        self.assertIn(('THRESHOLD_MISMATCH', 'strong', '250.000 TL'),
                      signals(payload, '250.000 TL ve üzerindeki işlemlerde müşterinin kimlik tespiti yapılır.', quantities))
        # 185.000 TL states the general floor; it is not a weaker version of the 15.000 TL wire-transfer floor.
        text = '185.000 TL ve üzerindeki işlemlerde kimlik tespiti yapılır.'
        self.assertEqual(signals(payload, text, quantities), [])
        self.assertEqual(precheck.quantity_match(quantities, text), {'threshold_1': 'SAME', 'threshold_2': 'NOT_STATED'})

    def test_a_number_of_a_kind_the_duty_does_not_state_is_a_weak_quantity_signal(self):
        payload, quantities = md5_2()
        found = signals(payload, KYC_SKIP, quantities)
        self.assertIn(('QUANTITY', 'weak', "15.000 TL'nin"), found)
        self.assertEqual([s[0] for s in found if s[1] == 'strong'], ['WAIVER', 'LIMITER', 'EXEMPTION'])

    def test_topic_overlap_reads_content_stems(self):
        payload, _ = md26()
        self.assertEqual(precheck.topic_overlap(payload, RESTATED), 1.0)
        self.assertEqual(precheck.topic_overlap(payload, NOISE[0]), 0.0)
        self.assertEqual(precheck.topic_overlap({}, 'anything'), 0.0)
        self.assertGreater(precheck.topic_overlap(md5_2()[0], KYC_SKIP), 0.25)

    def test_the_duty_s_own_prohibition_restated_is_only_a_weak_signal(self):
        payload, quantities = md26()
        found, overlap, _ = precheck_passage(payload, RESTATED, quantities)
        self.assertEqual([(s['type'], s['strength']) for s in found], [('WAIVER', 'weak')])
        # The truncated v0.18 duty cannot tell a restatement from a waiver.
        found, _, _ = precheck_passage(md26(repaired=False)[0], RESTATED, [])
        self.assertEqual([(s['type'], s['strength']) for s in found], [('WAIVER', 'strong')])


class PrefilterConflictBypassTests(unittest.TestCase):
    def test_a_passage_with_conflict_wording_or_a_quantity_is_never_set_aside_by_the_screen(self):
        payload, quantities = md46()
        texts = [NOISE[1], NOISE[2], NOISE[3],                     # the best-ranked three are always judged
                 'Kimlik tespiti adımı kampanyada atlanır.',        # waiver wording
                 'Belgeler beş yıl saklanır.',                      # a quantity
                 'Kontrol No: RET-01 — Arşiv birimi belgeleri düzenler.',  # a control-row id is no number with a unit
                 NOISE[0]]
        judge = V19(fast=lambda text: fast('IRRELEVANT'), screen=lambda text: 'OTHER')
        relations, results, _ = run(judge, payload, quantities, texts, screen=True)
        screened = sorted(sid for sid, r in results.items() if r.get('screen') == 'SCREENED_OUT')
        self.assertEqual(screened, ['p6', 'p7'])
        self.assertEqual(len(judge.kinds('screen')), 1)
        self.assertEqual(judge.kinds('screen')[0]['ids'], [texts[5], texts[6]])     # the protected passages were not even shown
        self.assertEqual(sorted(c['passage'] for c in judge.kinds('fast')), sorted(texts[:5]))

    def test_the_safety_list(self):
        for text in ('Tespit atlanır.', 'Bu istisnadır.', 'En geç ay sonunda bildirilir.', 'Sadece şubede yapılır.', 'Kayıtlar 5 yıl tutulur.',
                     'Kayıtlar sekiz yıl tutulur.', 'Formlar toplu olarak iletilir.', 'Onay aranmaz.', 'Payments are waived.', 'within 30 days'):
            with self.subTest(text=text):
                self.assertTrue(precheck.screen_protected(text))
        for text in ('Kontrol No: RET-01', 'Kontrol No: KYC-03 Müşteri kabul formu', 'Ziyaretçiler resepsiyonda kayıt altına alınır.'):
            with self.subTest(text=text):
                self.assertEqual(precheck.screen_protected(text), '')


class ConflictDetectionTests(unittest.TestCase):
    def test_a_direct_contradiction_is_a_conflict_with_its_exact_span(self):
        payload, quantities = md28()
        text = "Şüpheli işlemler MASAK'a bildirilmez; uyum birimi dosyayı kendi arşivinde tutar."
        span = "Şüpheli işlemler MASAK'a bildirilmez;"
        judge = V19(fast=fast('POSSIBLE_CONFLICT', span), verify=verdict(True, 'DIRECT_OPPOSITE', span, 'NOT_APPLICABLE'))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage, flags), ('CONFLICTS', 'CONFLICT', []))
        self.assertEqual(results['p1']['escalation'], 'FAST_POSSIBLE_CONFLICT')
        self.assertEqual(results['p1']['verifier']['contradiction_span'], span)
        self.assertEqual([c['task'] for c in judge.calls], ['judge.fast', 'judge.verify'])
        self.assertIn('DIRECT_OPPOSITE', reason)
        self.assertIn('the thinking verifier read each (sent to it by FAST_POSSIBLE_CONFLICT) and quoted the contradicting sentence', reason)
        self.assertNotIn('second reading', reason)

    def test_a_month_end_batch_against_a_ten_business_day_deadline_is_escalated_and_a_conflict(self):
        payload, quantities = md28()
        span = "uygun görülenler o ay sonunda MASAK'a topluca bildirilir."
        judge = V19(fast=fast('SUPPORTS', MONTH_END), verify=verdict(True, 'DEADLINE_MISMATCH', span, 'NOT_APPLICABLE'))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [MONTH_END])
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        self.assertEqual(results['p1']['escalation'], 'STRUCTURAL_SIGNAL')
        shown = judge.kinds('verify')[0]['payload']
        self.assertIn('DEADLINE_MISMATCH', [s['type'] for s in shown['automatic_signals']])
        # Round 5: the verifier reads the compact duty (engine.model_duty): the deadline once, as its element.
        self.assertEqual(shown['duty']['elements']['deadline_1'], 'en geç on iş günü içinde (işleme ilişkin şüphenin oluştuğu tarihten itibaren)')

    def test_a_higher_trigger_amount_is_a_threshold_mismatch(self):
        payload, quantities = md5_1()
        text = '250.000 TL ve üzerindeki işlemlerde müşterinin kimlik tespiti yapılır.'
        judge = V19(fast=fast('SUPPORTS', text), verify=verdict(True, 'THRESHOLD_MISMATCH', text, 'NOT_APPLICABLE'))
        relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage, results['p1']['escalation']), ('CONFLICTS', 'CONFLICT', 'STRUCTURAL_SIGNAL'))
        self.assertIn('THRESHOLD_MISMATCH', reason)

    def test_the_c13_kyc_skip_reaches_the_verifier_through_its_wording_though_the_fast_reading_supports(self):
        payload, quantities = md5_2()
        judge = V19(fast=fast('SUPPORTS', 'müşteri yalnızca telefon numarasını doğrular.'),
                    verify=verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE'))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [KYC_SKIP])
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        self.assertEqual((results['p1']['fast']['label'], results['p1']['escalation']), ('POSSIBLE_SUPPORT', 'STRUCTURAL_SIGNAL'))
        self.assertEqual({s['type'] for s in judge.kinds('verify')[0]['payload']['automatic_signals'] if s['strength'] == 'strong'},
                         {'WAIVER', 'LIMITER', 'EXEMPTION'})

    def test_a_low_confidence_conflict_stands_and_routes_to_a_person(self):
        payload, quantities = md5_2()
        judge = V19(fast=fast('POSSIBLE_CONFLICT'), verify=verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE', confidence='LOW'))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [KYC_SKIP])
        self.assertEqual((relations['p1'], coverage, flags), ('CONFLICTS', 'CONFLICT', ['CONFLICT_LOW_CONFIDENCE']))
        self.assertIn('LOW confidence', reason)
        self.assertIn('CONFLICT_LOW_CONFIDENCE', [n['code'] for n in results['p1']['notes']])


class FalseConflictPreventionTests(unittest.TestCase):
    def test_a_stricter_retention_period_is_no_conflict_when_the_verifier_says_so(self):
        payload, quantities = md46()
        text = 'Kimlik tespitine ilişkin belge ve kayıtlar son işlem tarihinden itibaren on yıl süreyle saklanır ve istenmesi halinde ibraz edilir.'
        judge = V19(fast=fast('POSSIBLE_CONFLICT'), verify=verdict(relation='SUPPORTS', support=text, rationale='Ten years is stricter.'))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage, flags), ('SUPPORTS', 'COVERS_TEXT', []))
        self.assertEqual(results['p1']['quantity_match'], {'deadline_1': 'STRICTER'})
        self.assertIn(('STRICTER_QUANTITY', 'weak'), [(s['type'], s['strength']) for s in results['p1']['signals']])
        self.assertIn('the same or stricter', reason)

    def test_a_restated_prohibition_is_withdrawn_by_rule_even_when_the_verifier_calls_it_a_conflict(self):
        payload, quantities = md26()
        judge = V19(fast=fast('POSSIBLE_CONFLICT'), verify=verdict(True, 'DIRECT_OPPOSITE', RESTATED, 'NOT_APPLICABLE'))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [RESTATED])
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'COVERS_TEXT'))
        codes = [n['code'] for n in results['p1']['notes']]
        self.assertIn('CONFLICT_WITHDRAWN', codes)
        self.assertIn('RESTATED_PROHIBITION', codes)

    def test_a_conflict_claim_without_an_exact_span_is_no_conflict_but_a_strong_uncertainty(self):
        payload, quantities = md5_2()
        judge = V19(fast=fast('POSSIBLE_CONFLICT'), verify=verdict(True, 'EXEMPTION_ADDED', 'Kimlik tespiti hiçbir zaman yapılmaz.', 'NOT_APPLICABLE'))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [KYC_SKIP])
        self.assertEqual((relations['p1'], coverage, flags), ('UNCLEAR', 'UNKNOWN', ['POSSIBLE_CONFLICT_UNRESOLVED']))
        self.assertEqual(results['p1']['uncertainty'], 'strong')
        self.assertEqual(len(judge.kinds('verify')), 2)                                   # one repair with the reason
        self.assertIn('validation_feedback', judge.kinds('verify')[1]['payload'])
        self.assertIn('CONFLICT_CLAIM_WITHOUT_SPAN', [n['code'] for n in results['p1']['notes']])
        self.assertIn('CONFLICT_CLAIM_WITHOUT_SPAN', reason)
        # A NONE type is a claim without its kind: the same rule.
        judge = V19(fast=fast('POSSIBLE_CONFLICT'), verify=verdict(True, 'NONE', KYC_SKIP_SPAN, 'NOT_APPLICABLE'))
        self.assertEqual(run(judge, payload, quantities, [KYC_SKIP])[0]['p1'], 'UNCLEAR')


class TruncatedExtractionTests(unittest.TestCase):
    def test_the_repaired_c14_duty_and_its_restating_passage_give_no_conflict(self):
        payload, quantities = md26()
        self.assertEqual(payload['prohibited_action'], 'basitleştirilmiş tedbirleri uygulayamazlar')
        self.assertIn('işlem nedeniyle aklama veya terörün finansmanı riskinin oluşabileceği durumlarda', payload['conditions'])
        claim = lambda text: verdict(True, 'DIRECT_OPPOSITE', RESTATED, 'NOT_APPLICABLE') if text == RESTATED else verdict(relation='UNRELATED')
        judge = V19(fast=lambda text: fast('POSSIBLE_CONFLICT') if text == RESTATED else fast('IRRELEVANT'), verify=claim)
        relations, _, (coverage, _, _, _) = run(judge, payload, quantities, [RESTATED, KYC_SKIP, *NOISE[:2]])
        self.assertEqual((relations['p1'], relations['p2'], coverage), ('SUPPORTS', 'UNRELATED', 'COVERS_TEXT'))
        # Round 5: the compact duty carries the repaired action once, as its element (engine.model_duty).
        self.assertEqual(judge.kinds('verify')[0]['payload']['duty']['elements']['action'], 'basitleştirilmiş tedbirleri uygulayamazlar')
        # The v0.18 truncated duty ("uygulayamazlar", no condition) could not tell the restatement from a waiver.
        truncated, _ = md26(repaired=False)
        judge = V19(fast=lambda text: fast('POSSIBLE_CONFLICT') if text == RESTATED else fast('IRRELEVANT'), verify=claim)
        self.assertEqual(run(judge, truncated, [], [RESTATED])[2][0], 'CONFLICT')


class AggregationTests(unittest.TestCase):
    def test_one_support_among_irrelevant_noise_covers_the_duty(self):
        payload, quantities = md26()
        # v0.19 COVERS gate: the fast SUPPORTS is confirmed once by the strong verifier before it covers.
        judge = V19(fast=lambda text: fast('SUPPORTS', RESTATED) if text == RESTATED else fast('IRRELEVANT', covered=()),
                    verify={**verdict(relation='SUPPORTS', support=RESTATED), 'covered_elements': ['action'], 'missing_elements': []})
        relations, _, (coverage, reason, _, flags) = run(judge, payload, quantities, [*NOISE, RESTATED])
        self.assertEqual((coverage, flags), ('COVERS_TEXT', []))
        self.assertEqual([(c['passage'], c['payload']['question'][:30]) for c in judge.kinds('verify')],
                         [(RESTATED, 'The fast reading says this pas')])
        self.assertEqual(list(relations.values()).count('UNRELATED'), 5)
        self.assertIn('none of the 6 passages read conflicts', reason)

    def test_a_weak_uncertainty_never_turns_support_into_unknown(self):
        payload, quantities = md26()
        down = ProviderFailure('down')
        answers = {RESTATED: fast('SUPPORTS', RESTATED), NOISE[3]: down}
        judge = V19(fast=lambda text: answers.get(text) or fast('IRRELEVANT', covered=()),
                    verify={**verdict(relation='SUPPORTS', support=RESTATED), 'covered_elements': ['action'], 'missing_elements': []})
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [RESTATED, NOISE[0], NOISE[1], NOISE[3]])
        self.assertEqual((relations['p4'], results['p4']['uncertainty'], coverage, flags), ('UNCLEAR', 'weak', 'COVERS_TEXT', []))
        self.assertIn('could not be judged', reason)
        # rank 4, no signal: not escalated; the one verifier call is the COVERS confirmation of the support.
        self.assertEqual([c['passage'] for c in judge.kinds('verify')], [RESTATED])
        # The same counted directly: SUPPORTS beside a weak UNCLEAR and noise.
        checks = [PolicyCheck(source_id='a', quote='q', relation='SUPPORTS'), PolicyCheck(source_id='b', quote='q', relation='UNCLEAR'),
                  PolicyCheck(source_id='c', quote='q', relation='UNRELATED')]
        results = [{'source_id': 'a', 'missing_elements': []}, {'source_id': 'b', 'uncertainty': 'weak'}, {'source_id': 'c'}]
        self.assertEqual(coverage_of_v19(checks, results)[0], 'COVERS_TEXT')
        results[1]['uncertainty'] = 'strong'
        self.assertEqual(coverage_of_v19(checks, results)[::3], ('UNKNOWN', ['POSSIBLE_CONFLICT_UNRESOLVED']))

    def test_no_evidence_and_control_rows_keep_the_v18_wording(self):
        checks = [PolicyCheck(source_id='k1', quote='q', relation='SUPPORTS'), PolicyCheck(source_id='p1', quote='q', relation='UNRELATED')]
        coverage, reason, control, flags = coverage_of_v19(checks, [{'source_id': 'k1'}, {'source_id': 'p1'}], {'k1'})
        self.assertEqual((coverage, control, flags), ('NO_EVIDENCE', 'SUPPORTS', []))
        self.assertIn('operational evidence', reason)
        self.assertEqual(coverage_of_v19([], [])[0], 'NO_EVIDENCE')
        conflict = [PolicyCheck(source_id='k1', quote='q', relation='CONFLICTS')]
        self.assertEqual(coverage_of_v19(conflict, [{'source_id': 'k1'}], {'k1'})[::2], ('CONFLICT', 'CONFLICT'))


class PartialCoverageTests(unittest.TestCase):
    def test_the_retention_act_without_its_period_is_partial(self):
        payload, quantities = md46()
        text = 'İşlem kayıtları ve müşteri dosyaları belge yönetim sisteminde saklanır.'
        # Short round 3: a PARTIAL resting on the fast reading alone is confirmed once by the strong verifier.
        judge = V19(fast=fast('SUPPORTS', text), verify={**verdict(relation='SUPPORTS', support=text), 'covered_elements': ['act_1'],
                                                         'missing_elements': []})
        relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage, results['p1']['partial_confirmation']), ('SUPPORTS', 'PARTIAL', 'SUPPORTS'))
        self.assertEqual(results['p1']['quantity_match'], {'deadline_1': 'NOT_STATED'})
        self.assertIn('QUANTITY_NOT_STATED', reason)
        self.assertIn('sekiz yıl süre ile', reason)
        # A weaker period is named as such (when the verifier found no conflict in it).
        judge = V19(fast=fast('SUPPORTS', FIVE_YEARS), verify=verdict(relation='SUPPORTS', support=FIVE_YEARS))
        self.assertIn('a weaker one is stated', run(judge, payload, quantities, [FIVE_YEARS])[2][1])

    def test_missing_elements_of_the_fast_reading_make_it_partial(self):
        payload, quantities = duty('42', 2, 'Düzenlenen tutanak', 'MUST', 'asgari aşağıdaki bilgilerin yer alması zorunludur')
        self.assertEqual(quantities, [])
        text = 'Tutanakta yer ve tarih, yolcunun kimlik bilgileri ve yolcunun imzası bulunur.'
        # Short round 3: the strong verifier confirms the PARTIAL the fast reading alone would decide.
        judge = V19(fast=fast('SUPPORTS', text, covered=('action', 'item_1', 'item_2', 'item_9'), missing=('item_3', 'item_4')),
                    verify={**verdict(relation='PARTIAL', support=text), 'covered_elements': ['action', 'item_1', 'item_2'],
                            'missing_elements': ['item_3', 'item_4']})
        relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [text])
        # WS6b: a SUPPORTS that lists missing elements is recorded as the PARTIAL it describes.
        self.assertEqual((relations['p1'], coverage), ('PARTIAL', 'PARTIAL'))
        self.assertIn('FAST_NORMALISED', [n['code'] for n in results['p1']['notes']])
        self.assertEqual((results['p1']['fast']['label'], results['p1']['partial_confirmation']), ('POSSIBLE_PARTIAL', 'CONFIRMED'))
        self.assertEqual(results['p1']['missing_elements'], ['item_3', 'item_4'])
        self.assertIn('Yolcunun mesleği veya iştigal konusu', reason)
        # An element id the duty does not have is dropped, not trusted.
        judge = V19(fast=fast('PARTIAL', text, covered=('action', 'item_99'), missing=()))
        self.assertEqual(run(judge, payload, quantities, [text])[1]['p1']['fast']['covered'], ['action'])

    def test_the_act_and_its_period_in_two_passages_cover_the_duty_together(self):
        payload, quantities = md46()
        act = 'Kimlik tespitine ilişkin belge ve kayıtlar muhafaza edilir ve istenmesi halinde yetkililere ibraz edilir.'
        period = 'Belgeler ve kayıtlar son işlem tarihinden itibaren sekiz yıl süreyle saklanır.'
        # The md. 46(1) action coordinates two acts (keep; produce on request): act_1 and act_2.
        self.assertEqual([e['id'] for e in payload['elements']], ['act_1', 'act_2', 'subject', 'deadline_1'])
        answers = {act: fast('SUPPORTS', act, covered=('act_1', 'act_2'), missing=('deadline_1',)),
                   period: fast('PARTIAL', period, covered=('act_1', 'deadline_1'), missing=('act_2',))}
        # v0.19 COVERS gate: the joint cover is confirmed once by the strong verifier (the best-ranked passage).
        confirm = {**verdict(relation='PARTIAL', support=act), 'covered_elements': ['act_1', 'act_2'], 'missing_elements': ['deadline_1']}
        judge = V19(fast=lambda text: answers[text], verify=confirm)
        relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [act, period])
        # WS6b: each is a PARTIAL on its own (the SUPPORTS listing a missing period is normalised); together
        # they state every element of the duty.
        self.assertEqual((relations, coverage), ({'p1': 'PARTIAL', 'p2': 'PARTIAL'}, 'COVERS_TEXT'))
        self.assertEqual(([c['passage'] for c in judge.kinds('verify')], results['p1']['covers_confirmation']), ([act], 'PARTIAL'))
        self.assertIn('together state every element', reason)
        self.assertIn('sekiz yıl süre ile', reason)
        # Either one alone stays partial (short round 3: once the strong verifier confirms it).
        self.assertEqual(run(V19(fast=lambda text: answers[text], verify=confirm), payload, quantities, [act])[2][0], 'PARTIAL')


class VerifierEconomyTests(unittest.TestCase):
    def test_the_thinking_verifier_is_asked_only_where_a_conflict_is_possible(self):
        payload, quantities = md28()
        support = 'Şüpheli işlemler, şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde MASAK\'a bildirilir.'
        judge = V19(fast=lambda text: fast('SUPPORTS', support) if text == support else fast('IRRELEVANT', covered=()),
                    verify={**verdict(relation='SUPPORTS', support=support), 'covered_elements': ['action', 'deadline_1'], 'missing_elements': []})
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [support, *NOISE])
        self.assertEqual(coverage, 'COVERS_TEXT')
        # No conflict question; the one strong call is the COVERS confirmation of the fast support (v0.19 gate).
        self.assertEqual([(c['passage'], c['task']) for c in judge.kinds('verify')], [(support, 'judge.verify.support')])
        self.assertEqual({c['task'] for c in judge.calls}, {'judge.fast', 'judge.verify.support'})
        self.assertEqual(results['p1']['quantity_match'], {'deadline_1': 'SAME'})
        self.assertEqual((results['p1']['escalation'], results['p1']['covers_confirmation']), ('CONFIRM_COVERS', 'CONFIRMED'))
        self.assertTrue(all(r['escalation'] is None for r in results.values() if r['source_id'] != 'p1'))

    def test_ambiguous_answers_get_a_support_check_under_their_own_task(self):
        payload, quantities = md46()
        visitors = 'Ziyaretçiler resepsiyonda bilgilerini yazdırır.'          # no word of the duty: a support here is doubtful
        judge = V19(fast=fast('SUPPORTS', visitors), verify=verdict(relation='UNRELATED', rationale='Visitors are another actor.'))
        relations, results, _ = run(judge, payload, quantities, [visitors])
        self.assertEqual((relations['p1'], results['p1']['escalation']), ('UNRELATED', 'AMBIGUOUS_SUPPORT'))
        self.assertEqual(judge.kinds('verify')[0]['task'], 'judge.verify.support')
        near = 'Kimlik tespitine ilişkin belge ve kayıtlar son işlem tarihinden itibaren muhafaza edilir ve yetkililere ibraz edilir.'
        judge = V19(fast=fast('IRRELEVANT', covered=()), verify=verdict(relation='PARTIAL', support=near))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [near])
        self.assertEqual((relations['p1'], results['p1']['escalation'], coverage), ('PARTIAL', 'AMBIGUOUS_IRRELEVANT', 'PARTIAL'))
        self.assertEqual(judge.kinds('verify')[0]['task'], 'judge.verify.support')
        # The same passage below the best-ranked three keeps the fast answer.
        judge = V19(fast=fast('IRRELEVANT', covered=()), verify=verdict(relation='PARTIAL', support=near))
        self.assertEqual(run(judge, payload, quantities, [*NOISE[:3], near])[0]['p4'], 'UNRELATED')
        self.assertEqual(judge.kinds('verify'), [])

    def test_a_failed_fast_reading_falls_back_to_the_verifier_for_top_ranked_or_signalled_passages_only(self):
        payload, quantities = md5_2()
        judge = V19(fast=ProviderFailure('down'), verify=lambda text: verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE')
                    if text == KYC_SKIP else verdict(relation='UNRELATED'))
        texts = [NOISE[0], NOISE[1], NOISE[2], KYC_SKIP, NOISE[3]]
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, texts)
        self.assertEqual(relations, {'p1': 'UNRELATED', 'p2': 'UNRELATED', 'p3': 'UNRELATED', 'p4': 'CONFLICTS', 'p5': 'UNCLEAR'})
        self.assertEqual(sorted(c['passage'] for c in judge.kinds('verify')), sorted(texts[:4]))
        self.assertEqual({results[k]['escalation'] for k in ('p1', 'p4')}, {'FAST_FAILED'})
        self.assertEqual((results['p5']['uncertainty'], results['p5'].get('unjudged')), ('weak', True))
        self.assertIn('FAST_FAILED', [n['code'] for n in results['p5']['notes']])
        self.assertEqual(coverage, 'CONFLICT')


class VerifierFailureTests(unittest.TestCase):
    def test_a_verifier_that_truncates_twice_leaves_a_strong_uncertainty_and_the_duty_unknown(self):
        payload, quantities = md5_2()
        cut = failure('Provider output incomplete or truncated', 'output_truncated')
        judge = V19(fast=lambda text: fast('POSSIBLE_CONFLICT') if text == KYC_SKIP else fast('IRRELEVANT', covered=()), verify=cut)
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [KYC_SKIP, NOISE[0]])
        self.assertEqual((relations['p1'], results['p1']['uncertainty'], coverage, flags),
                         ('UNCLEAR', 'strong', 'UNKNOWN', ['POSSIBLE_CONFLICT_UNRESOLVED']))
        self.assertEqual([c['uncached'] for c in judge.kinds('verify')], [False, True])     # one retry, not from the cache
        self.assertIn({'question': 'verify', 'code': 'VERIFIER_FAILED', 'failure_code': 'OUTPUT_TRUNCATED'}, results['p1']['notes'])
        self.assertIn('OUTPUT_TRUNCATED', reason)

    def test_a_verifier_that_times_out_once_is_retried_and_answered(self):
        payload, quantities = md5_2()
        answers = [failure('Ollama request exceeded the time limit', 'transport', 'TIMEOUT'),
                   verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE')]
        judge = V19(fast=fast('POSSIBLE_CONFLICT'), verify=lambda text: answers.pop(0))
        relations, results, (coverage, _, _, flags) = run(judge, payload, quantities, [KYC_SKIP])
        self.assertEqual((relations['p1'], coverage, flags), ('CONFLICTS', 'CONFLICT', []))
        self.assertEqual([n.get('failure_code') for n in results['p1']['notes'] if n['code'] == 'ProviderFailure'], ['TIMEOUT'])

    def test_a_refused_request_is_not_retried(self):
        payload, quantities = md5_2()
        judge = V19(fast=fast('POSSIBLE_CONFLICT'), verify=failure('HTTP 400', 'refused'))
        relations, results, _ = run(judge, payload, quantities, [KYC_SKIP])
        self.assertEqual((relations['p1'], len(judge.kinds('verify'))), ('UNCLEAR', 1))

    def test_a_verifier_failure_does_not_fail_the_analysis(self):
        judge = V19(fast=lambda text: fast('POSSIBLE_CONFLICT', text), verify=failure('truncated', 'output_truncated'))
        row = v19_analyze(judge)['obligations'][0]
        proposal = row['proposal']
        self.assertEqual((proposal['coverage'], proposal['review_flags']), ('UNKNOWN', ['POSSIBLE_CONFLICT_UNRESOLVED']))
        self.assertEqual(proposal['remediation_status'], 'UNDETERMINED')
        self.assertIsNotNone(autonomous_gate(row))
        self.assertNotIn('Analysis unavailable', proposal['coverage_reason'])


def v19_analyze(judge, **settings):
    return analyze(company(), policies(), sections(), judge, ['CONC 7.3.4'],
                   settings=pipeline_settings(coverage_pipeline='v19', **settings))['events'][0]['payload']


class PipelineWiringTests(unittest.TestCase):
    def test_settings_record_the_pipeline_and_old_packets_read_as_v18(self):
        with patch.dict(os.environ, {'COVERAGE_PIPELINE': 'v19'}):
            self.assertEqual(pipeline_settings()['coverage_pipeline'], 'v19')
        with patch.dict(os.environ):
            os.environ.pop('COVERAGE_PIPELINE', None)
            current = pipeline_settings()
        self.assertEqual((current['coverage_pipeline'], current['coverage_prompts_sha256']), ('v18', None))
        self.assertEqual(len(pipeline_settings(coverage_pipeline='v19')['coverage_prompts_sha256']), 64)
        with self.assertRaises(ValueError):
            pipeline_settings(coverage_pipeline='v20')
        old = {k: v for k, v in current.items() if k not in ('coverage_pipeline', 'coverage_prompts_sha256')}
        self.assertEqual(settings_of({'pipeline_settings': old}), current)                 # a v0.18 packet is not blocked by the new keys
        self.assertEqual(settings_of({}), DEFAULT_SETTINGS)
        self.assertNotIn('judge.fast', prompt_registry())
        self.assertIn('judge.verify', prompt_registry(pipeline_settings(coverage_pipeline='v19')))

    def test_a_v19_analysis_records_the_pipeline_structure_and_its_readings(self):
        judge = V19(fast=lambda text: fast('SUPPORTS', text), verify=lambda text: verdict(relation='SUPPORTS', support=text))
        payload = v19_analyze(judge)
        row = payload['obligations'][0]
        self.assertEqual(payload['pipeline_settings']['coverage_pipeline'], 'v19')
        self.assertIn('judge.fast', payload['prompt_registry'])
        self.assertEqual(row['proposal']['coverage'], 'COVERS_TEXT')
        self.assertEqual(row['structure']['version'], 'structure-v1')
        result = next(d for d in row['diagnostics'] if d.get('stage') == 'passages')['results'][0]
        # v0.19 COVERS gate: the fast SUPPORTS was confirmed by the strong verifier before it covered.
        self.assertEqual((result['pipeline'], result['fast']['label'], result['escalation'], result['covers_confirmation']),
                         ('v19', 'POSSIBLE_SUPPORT', 'CONFIRM_COVERS', 'CONFIRMED'))
        self.assertTrue(any('fast unreasoned classifier' in item for item in payload['limitations']))

    def test_the_adaptive_window_hook_is_asked_once_per_obligation_in_v19_only(self):
        with patch('regchain.pilot.engine.window_for', side_effect=lambda judge, need: judge) as hook:
            v19_analyze(V19())
            self.assertEqual(hook.call_count, 1)
            (judge, need), _ = hook.call_args
            self.assertIsInstance(need, int)
            self.assertGreater(need, 0)
            with patch.dict(os.environ):
                os.environ.pop('COVERAGE_PIPELINE', None)
                analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
            self.assertEqual(hook.call_count, 1)

    def test_rows_carry_their_own_stage_timings(self):
        payload = v19_analyze(V19(fast=lambda text: fast('POSSIBLE_CONFLICT', text), verify=verdict(relation='SUPPORTS', support='All staff must retain records.')))
        timings = payload['obligations'][0]['timings_ms']
        for stage in ('retrieval', 'enrichment', 'conflict_precheck', 'fast_classifier', 'thinking_verifier', 'aggregation'):
            self.assertIn(stage, timings, stage)
        self.assertTrue(set(timings) <= set(STAGES))
        self.assertTrue(all(isinstance(v, int) for v in timings.values()))
        for stage in ('conflict_precheck', 'fast_classifier', 'thinking_verifier', 'support_judgement', 'aggregation'):
            self.assertIn(stage, payload['timings'])
        with patch.dict(os.environ):
            os.environ.pop('COVERAGE_PIPELINE', None)
            legacy = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']
        self.assertTrue({'conflict', 'coverage', 'aggregation'} <= set(legacy['obligations'][0]['timings_ms']))


class LegacyPathTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop('COVERAGE_PIPELINE', None)

    def test_the_v18_path_is_unchanged(self):
        payload = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']
        row = payload['obligations'][0]
        self.assertEqual([c['relation'] for c in row['proposal']['policy_checks']], ['SUPPORTS'])
        self.assertEqual((row['proposal']['coverage'], row['proposal']['review_flags']), ('COVERS_TEXT', []))
        result = next(d for d in row['diagnostics'] if d.get('stage') == 'passages')['results'][0]
        self.assertEqual(result['screen'], 'NO')
        self.assertNotIn('pipeline', result)
        self.assertEqual(payload['pipeline_settings']['coverage_pipeline'], 'v18')
        self.assertNotIn('judge.fast', payload['prompt_registry'])

    def test_an_unconfirmed_conflict_routes_to_a_person_and_says_so(self):
        class Outage(FixtureProvider):
            def passage(self, payload):
                return {'relation': 'CONFLICTS', 'quote': payload['passage'], 'reason': 'Contradicts.'}

            def _chat(self, prompt, payload, schema):
                if 'verdict' in schema['properties']:
                    raise ProviderFailure('down')
                return super()._chat(prompt, payload, schema)
        row = analyze(company(), policies(), sections(), Outage(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        proposal = row['proposal']
        self.assertEqual((proposal['coverage'], proposal['review_flags']), ('CONFLICT', ['CONFLICT_UNCONFIRMED']))
        self.assertIn('1 not confirmed', proposal['coverage_reason'])
        self.assertIn('CONFLICT_UNCONFIRMED', autonomous_gate(row))

    def test_a_json_null_from_enrich_or_draft_does_not_end_the_analysis(self):
        class Null(FixtureProvider):
            def passage(self, payload):
                return {'relation': 'UNRELATED', 'quote': '', 'reason': 'About something else.'}

            def _chat(self, prompt, payload, schema):
                if 'summary' in schema['properties'] or 'clause' in schema['properties']:
                    return 'null'
                return super()._chat(prompt, payload, schema)
        for settings in (None, pipeline_settings(coverage_pipeline='v19')):
            judge = Null() if settings is None else type('NullV19', (Null, V19), {})(fast=fast('IRRELEVANT', covered=()))
            with self.subTest(pipeline=settings and settings['coverage_pipeline']):
                row = analyze(company(), policies(), sections(), judge, ['CONC 7.3.4'], categories=['Diğer'],
                              settings=settings)['events'][0]['payload']['obligations'][0]
                item = row['proposal']['remediation']
                self.assertEqual((row['proposal']['coverage'], item['draft_status'], item['draft_failure']),
                                 ('NO_EVIDENCE', 'DRAFT_UNAVAILABLE', 'MALFORMED_JSON'))
                self.assertIn('ENRICHMENT_UNAVAILABLE', [d.get('code') for d in row['diagnostics']])

    def test_a_json_null_to_a_question_is_named_malformed_not_a_crash(self):
        from regchain.pilot.engine import answer_question

        class Null(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                return 'null'
        with self.assertRaises(json.JSONDecodeError) as caught:
            answer_question(Null(), 'Kayıtlar ne kadar saklanır?', sections(), policies(), [])
        self.assertEqual(getattr(caught.exception, 'code', None), 'MALFORMED_JSON')

    def test_draft_calls_are_attributed_to_their_obligation_and_failures_named_by_code(self):
        judge = V19(fast=fast('IRRELEVANT', covered=()))
        row = v19_analyze(judge)['obligations'][0]
        self.assertEqual(row['proposal']['remediation_status'], 'PROPOSED')
        self.assertEqual(judge.kinds('draft')[0]['context'].get('obligation_id'), row['id'])
        cases = [(failure('x', 'output_truncated'), 'OUTPUT_TRUNCATED'), (failure('x', 'circuit'), 'CIRCUIT_OPEN'),
                 (failure('x', 'transport', 'TIMEOUT'), 'TIMEOUT'), (failure('x', 'prompt_cut'), 'CONTEXT_LENGTH'),
                 (ProviderFailure('down'), 'MODEL_ERROR'), (ProviderFailure('request exceeded the time limit'), 'TIMEOUT')]
        for exc, code in cases:
            with self.subTest(code=code):
                self.assertEqual(draft_failure_code(exc), code)


if __name__ == '__main__':
    unittest.main()

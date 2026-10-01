"""v0.15: pre-classification, repair hints, four-state applicability with a basis, remediation
proposals, the AI call log and provider error handling.

Every case here is a fixture; no model, no network. The live measurements are in docs/PHASE15.md.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from regchain.extraction.classify import NOT_EXTRACTED, classify, delegation_spans, duty_modals
from regchain.extraction.contract import materialize
from regchain.extraction.grounding import MODAL, verify
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import (AI_TASK, OllamaProvider, ProviderFailure, RulesProvider, ai_task, usage_summary)
from regchain.extraction.schema import Candidate, ExtractionOutput
from regchain.pilot.engine import analyze, autonomous_gate, autonomous_review, company_facts, prompt_registry
from regchain.pilot.report import render
from regchain.pilot.review import apply_review
from regchain.pilot.schema import Company, Review
from regchain.pilot.workspace import Workspace, create_app, metrics_of
from test_pilot import FixtureProvider, company, policies, sections
from test_workspace import request_input, source_fixture

LOCAL = 'http://localhost:11434'
STR_DUTY = ('(1) Yükümlüler nezdinde veya bunlar aracılığıyla yapılan veya yapılmaya teşebbüs edilen işlemlere konu malvarlığının yasa dışı '
            'yollardan elde edildiğine veya yasa dışı amaçlarla kullanıldığına dair herhangi bir bilgi, şüphe veya şüpheyi gerektirecek bir '
            'hususun bulunması halinde bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi zorunludur.')
DELEGATION = '(3) Yükümlülerin hangi faaliyetlerinden dolayı ve hangi usûl ve esaslara göre şüpheli işlem bildiriminde bulunacağı yönetmelikle belirlenir.'


class ClassificationTests(unittest.TestCase):
    def test_turkish_units_are_classed_from_their_wording(self):
        cases = {
            STR_DUTY: 'OBLIGATION',
            '(1) Yükümlüler, işlem yapanların kimliklerini tespit etmek zorundadır.': 'OBLIGATION',
            '(2) Yükümlüler, bildirimde bulunulduğunu hiç kimseye açıklayamazlar.': 'PROHIBITION',
            DELEGATION: 'DELEGATION',
            '(2) Kimlik tespitine esas belge nevilerini belirlemeye Bakanlık yetkili olup, usûl ve esaslar yönetmelikle belirlenir.': 'DELEGATION',
            '(2) Başkanlık altyapıyı kurmaya yetkilidir.': 'DELEGATION',
            '(1) Bu Kanunda geçen; a) Bakanlık: Maliye Bakanlığını ifade eder.': 'DEFINITION',
            '(1) Bu Kanunun 4 üncü maddesindeki yükümlülükleri ihlâl eden kimse bir yıldan üç yıla kadar hapis ve adlî para cezası ile cezalandırılır.': 'ENFORCEMENT',
            '(1) Bu Yönetmelik, 5549 sayılı Kanun kapsamındaki yükümlüler hakkında uygulanır.': 'SCOPE',
            '(3) Bu madde hükümleri yabancı kuruluşlar hakkında uygulanmaz.': 'EXEMPTION',
            '(1) Kurul ek tedbir alabilir.': 'PERMISSION',
            '(1) Şüpheli işlemler on iş günü içinde Başkanlığa bildirilir.': 'OBLIGATION',
            '(1) Müşterinin kimliği işlem öncesinde tespit edilir.': 'OTHER',      # aorist without a duty marker the pattern knows
        }
        for text, expected in cases.items():
            with self.subTest(text=text[:50]):
                self.assertEqual(classify(text).kind, expected)
        # The article heading decides definitions, scope and penalty articles; a duty marker still wins.
        self.assertEqual(classify('(1) Bu Kanunda geçen; d) Yükümlü: bankaları,', heading='Tanımlar').kind, 'DEFINITION')
        self.assertEqual(classify('(1) Başkanlıkça yükümlülere idari para cezası verilir.', heading='Yükümlülük ihlâlinde idarî ceza').kind, 'ENFORCEMENT')
        self.assertEqual(classify('(1) Yükümlüler kayıt tutmak zorundadır.', heading='Kapsam').kind, 'OBLIGATION')

    def test_english_units_are_classed_from_their_wording(self):
        cases = {'(1) A firm must retain records.': 'OBLIGATION',
                 '(2) This rule applies in relation to: (a) a communication by the firm; (b) a financial promotion.': 'SCOPE',
                 'A firm must not pressurise a customer.': 'PROHIBITION',
                 'A firm may rely on a third party.': 'PERMISSION',
                 'In this chapter, "customer" means an individual who is a borrower.': 'DEFINITION',
                 'A firm should consider the circumstances.': 'RECOMMENDATION',
                 'The FCA may make rules specifying the information.': 'DELEGATION',
                 'A firm which fails to comply is liable to a financial penalty.': 'ENFORCEMENT',
                 'Some descriptive sentence.': 'OTHER'}
        for text, expected in cases.items():
            with self.subTest(text=text[:40]):
                self.assertEqual(classify(text).kind, expected)
        self.assertEqual(classify('A firm should consider the circumstances.', legal_type='GUIDANCE').kind, 'GUIDANCE')
        self.assertTrue(classify('A firm must retain records.').extractable)
        self.assertFalse(classify(DELEGATION).extractable)
        self.assertEqual(NOT_EXTRACTED, {'DEFINITION', 'SCOPE', 'DELEGATION', 'ENFORCEMENT', 'PERMISSION', 'EXEMPTION'})

    def test_a_delegation_sentence_is_neither_a_duty_nor_required_to_be_covered(self):
        mixed = '(1) Yükümlüler kayıtları sekiz yıl saklamak zorundadır. Usûl ve esaslar yönetmelikle belirlenir.'
        self.assertEqual([mixed[a:b] for a, b in delegation_spans(mixed)], ['Usûl ve esaslar yönetmelikle belirlenir.'])
        self.assertEqual([m.group() for m in MODAL.finditer(mixed)], ['zorundadır', 'belirlenir'])
        self.assertEqual([m.group() for m in duty_modals(mixed)], ['zorundadır'])
        output = materialize(json.dumps({'status': 'EXTRACTED', 'obligations': [{'subject': 'Yükümlüler', 'modality': 'MUST',
                                         'action': 'kayıtları sekiz yıl saklamak', 'conditions': [], 'exceptions': [], 'evidence': []}]}), mixed)
        verify(mixed, output)                                                   # no UNCOVERED_MODAL for "belirlenir"
        # The rules baseline reads the same modals, so it does not turn the delegation into a duty.
        result = extract(mixed, RulesProvider())
        self.assertEqual([o.required_action for o in result.output.obligations], ['kayıtları sekiz yıl saklamak'])
        self.assertEqual(extract(DELEGATION, RulesProvider()).output.status, 'NO_EXPLICIT_OBLIGATION')


class RepairHintTests(unittest.TestCase):
    def test_the_alignment_failure_names_the_words_before_the_marker(self):
        # The live failure of 22 September 2026: the model stopped at the conditional clause.
        cut = json.dumps({'status': 'EXTRACTED', 'obligations': [{'subject': 'Yükümlüler', 'modality': 'MUST', 'conditions': [], 'exceptions': [], 'evidence': [],
                          'action': 'nezdinde veya bunlar aracılığıyla yapılan veya yapılmaya teşebbüs edilen işlemlere konu malvarlığının yasa dışı yollardan '
                                    'elde edildiğine veya yasa dışı amaçlarla kullanıldığına dair herhangi bir bilgi, şüphe veya şüpheyi gerektirecek bir hususun '
                                    'bulunması halinde bu işlemlerin'}]})
        with self.assertRaises(ValueError) as caught:
            verify(STR_DUTY, materialize(cut, STR_DUTY))
        message = str(caught.exception)
        self.assertTrue(message.startswith('ACTION_ALIGNMENT'))
        self.assertIn("marker 'zorunludur'", message)
        self.assertIn('yükümlüler tarafından Başkanlığa bildirilmesi', message)
        whole = json.dumps({'status': 'EXTRACTED', 'obligations': [{'subject': 'Yükümlüler', 'modality': 'MUST', 'evidence': [], 'exceptions': [],
                            'action': 'bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi',
                            'conditions': ['şüphe veya şüpheyi gerektirecek bir hususun bulunması halinde']}]})
        verify(STR_DUTY, materialize(whole, STR_DUTY))
        english = 'A firm must retain records for six years.'
        with self.assertRaises(ValueError) as caught:
            verify(english, ExtractionOutput(status='EXTRACTED', obligations=[Candidate(source_quote=english, subject='A firm', modality='MUST_NOT',
                                                                                        prohibited_action='retain records', confidence_score='0.5')]))
        self.assertIn('No marker of modality MUST_NOT', str(caught.exception))

    def test_the_repair_prompt_carries_the_hint(self):
        class Cutting(RulesProvider):
            """Answers like the live model: the conditional clause as the action, then the repaired copy."""
            fed = []

            def generate(self, text):
                return json.dumps({'status': 'EXTRACTED', 'obligations': [{'source_quote': text, 'subject': 'Yükümlüler', 'modality': 'MUST',
                                   'required_action': 'bulunması halinde bu işlemlerin', 'prohibited_action': None, 'conditions': [], 'exceptions': [],
                                   'confidence_score': '0.5000'}]})

            def repair(self, text, context, feedback):
                Cutting.fed.append(feedback)
                return json.dumps({'status': 'EXTRACTED', 'obligations': [{'source_quote': text, 'subject': 'Yükümlüler', 'modality': 'MUST',
                                   'required_action': 'bu işlemlerin yükümlüler tarafından Başkanlığa bildirilmesi', 'prohibited_action': None,
                                   'conditions': ['şüphe veya şüpheyi gerektirecek bir hususun bulunması halinde'], 'exceptions': [], 'confidence_score': '0.5000'}]})
        result = extract(STR_DUTY, Cutting())
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertEqual(result.attempts, 2)
        self.assertIn('Başkanlığa bildirilmesi', Cutting.fed[0])


def turkish_sections():
    common = dict(source_kind='CONSOLIDATED', quality_flags=[], defined_terms=[], legal_type='RULE', locator_kind='mevzuat_madde',
                  effective_from='2006-10-18', source_url='https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=1&MevzuatNo=5549&MevzuatTertip=5',
                  fetched_at='2026-09-22T00:00:00+00:00', external_key='MEVZUAT-1.5.5549', version_hash='a'*64)
    title = 'SUÇ GELİRLERİNİN AKLANMASININ ÖNLENMESİ HAKKINDA KANUN'
    return [dict(common, id='tr-scope', version_id='v', section_number='BİRİNCİ BÖLÜM', paragraph_number='2', ordinal=1, printed_label='Kanun 5549 md. 2',
                 heading_path=[title, 'BİRİNCİ BÖLÜM Amaç ve Tanımlar', 'Tanımlar'], content_hash='b'*64,
                 text='(1) Bu Kanunda geçen; d) Yükümlü: Bankacılık, sigortacılık, ödünç para verme ve diğer finansal hizmetler alanında faaliyet gösterenleri, ifade eder.'),
            dict(common, id='tr-duty', version_id='v', section_number='İKİNCİ BÖLÜM', paragraph_number='4', ordinal=3, printed_label='Kanun 5549 md. 4',
                 heading_path=[title, 'İKİNCİ BÖLÜM Yükümlülükler', 'Şüpheli işlem bildirimi'], content_hash='c'*64,
                 text=STR_DUTY + ' (2) Yükümlüler, Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar. ' + DELEGATION),
            dict(common, id='tr-penalty', version_id='v', section_number='DÖRDÜNCÜ BÖLÜM', paragraph_number='14', ordinal=9, printed_label='Kanun 5549 md. 14',
                 heading_path=[title, 'DÖRDÜNCÜ BÖLÜM Cezalar', 'Yükümlülük ihlâlinde adlî ceza'], content_hash='d'*64,
                 text='(1) Bu Kanunun 4 üncü maddesinin ikinci fıkrası ile 7 ve 8 inci maddelerindeki yükümlülükleri ihlâl eden kimse bir yıldan üç yıla kadar hapis ve beşbin güne kadar adlî para cezası ile cezalandırılır.')]


def anadolu():
    return Company(id='anadolu-odeme-test', name='Anadolu Ödeme ve Elektronik Para A.Ş. — sentetik', version='1', synthetic=True,
                   jurisdictions=['Türkiye'], activities=['ödeme hizmetleri sunma', 'elektronik para ihracı'],
                   licences=['Sentetik senaryoda varsayılan TCMB ödeme kuruluşu faaliyet izni; gerçek bir izin değildir'],
                   products=['dijital cüzdan'], customer_types=['bireysel müşteriler'], description='Kurgusal Türk ödeme kuruluşu; gerçek bir kuruluşu temsil etmez.')


class TurkishRules(RulesProvider):
    """The rules baseline with the judge fixture on top, for a Turkish provision."""
    name, base_url = 'fixture', LOCAL
    _chat = FixtureProvider._chat
    scope = FixtureProvider.scope
    passage = FixtureProvider.passage


class AnalysisClassificationTests(unittest.TestCase):
    def test_a_delegation_sub_paragraph_is_reported_not_extracted(self):
        calls = []

        class Counting(TurkishRules):
            def generate(self, text):
                calls.append(text[:20])
                return super().generate(text)
        payload = analyze(anadolu(), policies(), turkish_sections(), Counting(), ['Kanun 5549 md. 4'])['events'][0]['payload']
        case = payload['cases'][0]
        self.assertEqual([u['kind'] for u in case['classification']], ['OBLIGATION', 'PROHIBITION', 'DELEGATION'])
        self.assertEqual([u['reason'] for u in case['units']], ['CANDIDATE_REQUIRES_LEGAL_REVIEW', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 'CLASSIFIED_DELEGATION'])
        self.assertEqual(calls, ['(1) Yükümlüler nezdi', '(2) Yükümlüler, Başk'])                  # the delegation cost no model call
        self.assertEqual([o['candidate']['modality'] for o in payload['obligations']], ['MUST', 'MUST_NOT'])
        self.assertNotIn('yönetmelikle belirlenir', json.dumps([o['candidate'] for o in payload['obligations']], ensure_ascii=False))
        self.assertTrue(any('classified as definition' in text for text in payload['limitations']))

    def test_a_penalty_article_and_a_definitions_article_yield_no_candidate_and_say_why(self):
        payload = analyze(anadolu(), policies(), turkish_sections(), TurkishRules(), ['Kanun 5549 md. 14', 'Kanun 5549 md. 2'])['events'][0]['payload']
        reasons = {c['source']['printed_label']: c['reason'] for c in payload['cases']}
        self.assertEqual(reasons, {'Kanun 5549 md. 2': 'CLASSIFIED_DEFINITION', 'Kanun 5549 md. 14': 'CLASSIFIED_ENFORCEMENT'})
        self.assertEqual(payload['obligations'], [])
        page = render({'format': 'regchain-pilot-packet-v1', 'events': [{'payload': payload, 'event_hash': 'e'*64}], 'head': 'e'*64, 'count': 1})
        self.assertIn('Yaptırım hükmü', page)
        self.assertIn('Tanım hükmü', page)


class ApplicabilityTests(unittest.TestCase):
    def test_the_judge_sees_the_facts_but_not_the_synthetic_flag(self):
        seen = {}

        class Watching(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'applicability' in schema['properties']:
                    seen.update(payload)
                return super()._chat(prompt, payload, schema)
        analyze(company(), policies(), sections(), Watching(), ['CONC 7.3.4'])
        self.assertEqual(set(seen['company']), {'name', 'jurisdictions', 'activities', 'licences', 'products', 'customer_types', 'description'})
        self.assertNotIn('synthetic', seen['company'])
        self.assertEqual(seen['provision'], {'source_id': 'p0', 'text': 'A firm must retain records.'})
        self.assertIn('test-data marker', seen and __import__('regchain.pilot.engine', fromlist=['SCOPE_PROMPT']).SCOPE_PROMPT)
        self.assertEqual(company_facts(company())['activities'], ['consumer credit lending'])

    def test_a_basis_is_verified_against_the_profile_and_the_scope_text(self):
        row = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'APPLIES')
        self.assertEqual(row['proposal']['applicability_rule'], 'MODEL')
        self.assertEqual(row['proposal']['basis'], [{'company_fact': 'consumer credit lending', 'match': 'YES', 'source_id': 's-scope',
                                                     'regulatory_condition': 'This chapter applies to consumer credit lending.',
                                                     'company_fact_key': 'activities', 'exclusionary': None, 'note': ''}])

        class Inventing(FixtureProvider):
            calls = 0

            def scope(self, payload):
                Inventing.calls += 1
                value = super().scope(payload)
                value['basis'][0]['regulatory_condition'] = 'a condition the scope never states'
                return value
        row = analyze(company(), policies(), sections(), Inventing(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(Inventing.calls, 2)
        self.assertEqual(row['proposal']['applicability'], 'UNKNOWN')
        notes = [d for d in row['diagnostics'] if d.get('code') == 'PROPOSAL_INVALID']
        self.assertEqual(len(notes), 2)
        self.assertIn('regulatory_condition', notes[0]['detail'])
        self.assertIn('a condition the scope never states', notes[0]['response_excerpt'])    # what was answered is kept

        class Unfounded(FixtureProvider):
            def scope(self, payload):
                return {**super().scope(payload), 'basis': []}
        row = analyze(company(), policies(), sections(), Unfounded(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'UNKNOWN')
        self.assertIn('APPLIES requires at least one verified basis with match YES', row['diagnostics'][0]['detail'])

    def test_an_unknown_with_a_matching_basis_becomes_possibly_applies_by_rule(self):
        class Hedging(FixtureProvider):
            def scope(self, payload):
                return {**super().scope(payload), 'applicability': 'UNKNOWN', 'missing_information': ['Whether the licence is current'],
                        'applicability_reason': 'The activity is within scope; the licence status is not stated.'}
        packet = analyze(company(), policies(), sections(), Hedging(), ['CONC 7.3.4'])
        row = packet['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'POSSIBLY_APPLIES')
        self.assertEqual(row['proposal']['applicability_rule'], 'UPGRADED_FROM_UNKNOWN_BY_BASIS')
        self.assertIn('[Rule:', row['proposal']['applicability_reason'])
        self.assertIn('Olası uygulanabilir', render(packet))
        self.assertIn('kural gereği', render(packet))
        # Possibly is not a decision: the gate escalates and the recorded decision says UNKNOWN.
        self.assertIn('only possible', autonomous_gate(row))
        review, summary = autonomous_review(packet)
        self.assertEqual((summary['decided'], summary['escalated']), (0, 1))
        self.assertEqual(review.decisions[0].applicability, 'UNKNOWN')
        self.assertEqual(apply_review(packet, review)['events'][-1]['payload']['outcomes'][0]['status'], 'APPLICABILITY_REVIEW_REQUIRED')
        metrics = metrics_of(packet['events'][0]['payload'])
        self.assertEqual((metrics['applicable'], metrics['possibly_applicable']), (0, 1))

        class Contradicted(Hedging):
            def scope(self, payload):
                value = super().scope(payload)
                value['basis'].append({'company_fact': 'UK', 'regulatory_condition': 'consumer credit lending', 'match': 'NO'})
                return value
        row = analyze(company(), policies(), sections(), Contradicted(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'UNKNOWN')                    # a NO match blocks the upgrade

    def test_the_model_may_answer_possibly_or_does_not_apply_itself(self):
        class Possibly(FixtureProvider):
            def scope(self, payload):
                value = super().scope(payload)
                value['basis'][0]['match'] = 'UNCLEAR'
                return {**value, 'applicability': 'POSSIBLY_APPLIES', 'missing_information': ['entity type']}
        row = analyze(company(), policies(), sections(), Possibly(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((row['proposal']['applicability'], row['proposal']['applicability_rule']), ('POSSIBLY_APPLIES', 'MODEL'))

        class Outside(FixtureProvider):
            def scope(self, payload):
                value = super().scope(payload)
                value['basis'][0]['match'] = 'NO'
                return {**value, 'applicability': 'DOES_NOT_APPLY'}
        row = analyze(company(), policies(), sections(), Outside(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'DOES_NOT_APPLY')
        self.assertIsNone(row['proposal']['remediation'])                                 # nothing to propose for an inapplicable duty
        # A reviewer's decision stays three-valued.
        with self.assertRaises(ValueError):
            Review.model_validate({'format': 'regchain-pilot-review-v1', 'analysis_head': 'a'*64, 'reviewer': 'r', 'reviewer_role': 'x',
                                   'decisions': [{'obligation_id': 'o', 'extraction': 'ACCEPT', 'applicability': 'POSSIBLY_APPLIES', 'coverage': 'UNKNOWN',
                                                  'company_fact_keys': [], 'scope_evidence': [], 'policy_evidence': [], 'rationale': 'r'}]})


def many(count, conflict='Records are deleted every night.'):
    texts = [f'Staff must retain records in archive {i}.' for i in range(count)] + [conflict]
    return [dict(policies()[0], chunks=[dict(policies()[0]['chunks'][0], source_id=f'c{i:03d}', text=text) for i, text in enumerate(texts)])]


class Deleting(FixtureProvider):
    def passage(self, payload):
        if 'deleted' not in payload['passage']:
            return super().passage(payload)
        return {'relation': 'CONFLICTS', 'quote': 'Records are deleted every night.', 'reason': 'Fixture.'}


class Unrelated(FixtureProvider):
    def passage(self, payload):
        return {'relation': 'UNRELATED', 'quote': '', 'reason': 'About something else.'}


class RemediationTests(unittest.TestCase):
    def test_a_conflict_yields_a_high_priority_proposal_with_a_drafted_clause(self):
        packet = analyze(company(), many(1), sections(), Deleting(), ['CONC 7.3.4'])
        row = packet['events'][0]['payload']['obligations'][0]
        item = row['proposal']['remediation']
        self.assertEqual((item['label'], item['type'], item['trigger'], item['priority'], item['status'], item['requires_approval']),
                         ('AI-GENERATED PROPOSAL', 'CONFLICT_RESOLUTION', 'CONFLICT', 'HIGH', 'PROPOSED', True))
        self.assertEqual(item['priority_inputs'], ['coverage=CONFLICT'])
        self.assertIn('1 policy passage(s) contradict this duty', item['recommended_action'])
        self.assertIn('retain records', item['recommended_action'])
        self.assertEqual(item['suggested_policy_language'], 'Fixture policy clause requiring the duty.')
        self.assertIn('approval', item['implementation_notes'])
        page = render(packet)
        self.assertIn('AI öneri — onay gerektirir', page)
        self.assertIn('Çelişki giderme', page)
        self.assertIn('data-field="remediation"', page)
        metrics = metrics_of(packet['events'][0]['payload'])
        self.assertEqual((metrics['remediations'], metrics['remediations_high']), (1, 1))

    def test_no_evidence_yields_a_new_clause_proposal_and_a_covered_duty_none(self):
        row = analyze(company(), policies(), sections(), Unrelated(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        item = row['proposal']['remediation']
        self.assertEqual((item['type'], item['priority'], item['trigger']), ('NEW_POLICY_CLAUSE', 'MEDIUM', 'NO_EVIDENCE'))
        self.assertIn('add a clause', item['recommended_action'])
        covered = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertIsNone(covered['proposal']['remediation'])
        off = analyze(company(), policies(), sections(), Unrelated(), ['CONC 7.3.4'], remediation=False)['events'][0]['payload']
        self.assertIsNone(off['obligations'][0]['proposal']['remediation'])
        self.assertFalse(off['remediation_enabled'])

    def test_turkish_gaps_are_proposed_in_turkish_and_a_criminal_sanction_raises_the_priority(self):
        class TurkishUnrelated(TurkishRules):
            passage = Unrelated.passage
        payload = analyze(anadolu(), policies(), turkish_sections(), TurkishUnrelated(), ['Kanun 5549 md. 4'])['events'][0]['payload']
        items = [o['proposal']['remediation'] for o in payload['obligations']]
        self.assertTrue(all(item['type'] == 'NEW_POLICY_CLAUSE' for item in items))
        self.assertIn('policy\'ye yükümlülüğü açıkça düzenleyen bir madde ekle', items[0]['recommended_action'])
        # md. 14 threatens imprisonment for breaches of md. 4(2): the prohibition's gap is HIGH.
        self.assertEqual([item['priority'] for item in items], ['HIGH', 'HIGH'])
        self.assertIn('sanction=criminal', items[0]['priority_inputs'])
        self.assertIn('AI önerisidir', items[0]['implementation_notes'])

    def test_a_draft_failure_leaves_the_rule_based_proposal_and_is_noted(self):
        class NoDraft(Deleting):
            def _chat(self, prompt, payload, schema):
                if 'clause' in schema['properties']:
                    raise ProviderFailure('down')
                return super()._chat(prompt, payload, schema)
        row = analyze(company(), many(1), sections(), NoDraft(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        item = row['proposal']['remediation']
        self.assertEqual((item['priority'], item['suggested_policy_language']), ('HIGH', ''))
        # v0.16: the failure is named and the draft is retried once; the rule-based proposal stands.
        self.assertEqual((item['draft_status'], item['draft_failure'], item['draft_attempts']), ('DRAFT_UNAVAILABLE', 'MODEL_ERROR', 2))
        notes = [d for d in row['diagnostics'] if d.get('code') == 'DRAFT_UNAVAILABLE']
        self.assertEqual([(n['detail'], n['failure'], n['attempt']) for n in notes], [('ProviderFailure', 'MODEL_ERROR', 1), ('ProviderFailure', 'MODEL_ERROR', 2)])
        self.assertEqual(row['proposal']['remediation_status'], 'PROPOSED')
        # A reviewer can accept or reject the proposal; the field is part of the recorded decision.
        packet = analyze(company(), many(1), sections(), Deleting(), ['CONC 7.3.4'])
        review, _ = autonomous_review(packet)
        decision = review.decisions[0].model_dump()
        decision['remediation'] = 'REJECT'
        reviewed = apply_review(packet, Review(format='regchain-pilot-review-v1', analysis_head=packet['head'], reviewer='r', reviewer_role='role', decisions=[decision]))
        self.assertEqual(reviewed['events'][-1]['payload']['review']['decisions'][0]['remediation'], 'REJECT')


class CallLogTests(unittest.TestCase):
    def client(self, factory, chunks):
        client = factory.return_value.__enter__.return_value
        client.get.return_value.json.return_value = {'models': [{'name': 'm', 'digest': 'd'}]}
        stream = client.stream.return_value.__enter__.return_value
        stream.iter_bytes.return_value = chunks
        return client

    def test_every_call_is_recorded_with_its_task_and_without_its_text(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        chunk = [b'{"message": {"content": "{}"}, "done": true, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 9}']
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            self.client(factory, chunk)
            with ai_task('judge.applicability'):
                provider.generate_structured('secret prompt', {'company': 'confidential'}, {'type': 'object'})
            self.assertEqual(AI_TASK.get(), '')
        entry = provider.call_log[-1]
        self.assertEqual((entry['task'], entry['status'], entry['prompt_tokens'], entry['output_tokens'], entry['model_version']),
                         ('judge.applicability', 'OK', 300, 9, 'm@d'))
        self.assertNotIn('confidential', json.dumps(entry))
        self.assertNotIn('secret', json.dumps(entry))
        self.assertEqual(len(entry['prompt_sha256']), 64)
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            self.client(factory, [b'{"message": {"content": "{"}, "done": false}\n'])
            with self.assertRaises(ProviderFailure):
                provider._chat('p', {'text': 'x'}, {'type': 'object'})
        self.assertEqual(provider.call_log[-1]['status'], 'PROVIDER_FAILURE')
        summary = usage_summary([provider.call_log, provider.call_log])
        self.assertEqual((summary['calls'], summary['failures'], summary['prompt_tokens']), (2, 1, 300))
        self.assertEqual(summary['by_task']['judge.applicability']['calls'], 1)
        self.assertEqual(sorted(prompt_registry()), ['ask', 'draft', 'enrich', 'extraction.context', 'extraction.primary', 'extraction.review',
                                                     'extraction.turkish', 'judge.conflict', 'judge.conflict_confirm', 'judge.scope', 'judge.support'])

    def test_transient_http_errors_back_off_and_the_status_is_named(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        response = httpx.Response(429, request=httpx.Request('POST', LOCAL + '/api/chat'))
        with patch('regchain.extraction.providers.httpx.Client') as factory, patch('regchain.extraction.providers.time.sleep') as sleep:
            client = self.client(factory, [])
            client.stream.return_value.__enter__.return_value.raise_for_status.side_effect = httpx.HTTPStatusError('x', request=response.request, response=response)
            with self.assertRaisesRegex(ProviderFailure, 'HTTP 429: rate limited after 3 attempts'):
                provider._chat('p', {'text': 'x'}, {'type': 'object'})
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])
        self.assertEqual(provider.call_log[-1]['status'], 'PROVIDER_FAILURE')
        response = httpx.Response(401, request=httpx.Request('POST', LOCAL + '/api/chat'))
        with patch('regchain.extraction.providers.httpx.Client') as factory, patch('regchain.extraction.providers.time.sleep') as sleep:
            client = self.client(factory, [])
            client.stream.return_value.__enter__.return_value.raise_for_status.side_effect = httpx.HTTPStatusError('x', request=response.request, response=response)
            with self.assertRaisesRegex(ProviderFailure, 'HTTP 401: authentication failed'):
                provider._chat('p', {'text': 'x'}, {'type': 'object'})
            sleep.assert_not_called()


class WorkspaceLogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        class Logging(FixtureProvider):
            call_log = []

            def _chat(self, prompt, payload, schema):
                Logging.call_log.append({'task': AI_TASK.get(), 'status': 'OK', 'prompt_tokens': 10, 'output_tokens': 2, 'elapsed_ms': 5})
                return super()._chat(prompt, payload, schema)
        Logging.call_log = []
        self.provider = Logging
        self.workspace = Workspace(Path(self.temp.name)/'workspace', provider_factory=lambda _: Logging(), source_fetcher=source_fixture)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                                 headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def finish(self, **changes):
        run_id = self.client.post('/api/runs', json=request_input(**changes)).json()['id']
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def test_the_run_keeps_its_ai_call_log_and_usage_and_the_form_flag_is_honoured(self):
        row = self.finish(provider='ollama')
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        log = (self.workspace.directory(row['id'])/'ai-calls.jsonl').read_text(encoding='utf-8').splitlines()
        tasks = [json.loads(line)['task'] for line in log]
        self.assertIn('judge.applicability', tasks)
        self.assertIn('judge.contradicts', tasks)
        self.assertEqual(row['ai_usage']['calls'], len(log))
        self.assertIn('judge.applicability', row['ai_usage']['by_task'])
        self.assertIn('possibly_applicable', row['metrics'])
        self.assertEqual(row['metrics']['remediations'], 0)                        # the fixture covers the duty
        packet = self.client.get(f'/api/runs/{row["id"]}/packet').json()
        self.assertTrue(packet['events'][0]['payload']['remediation_enabled'])
        self.assertIn('judge.scope', packet['events'][0]['payload']['prompt_registry'])
        off = self.finish(provider='ollama', remediation=False)
        self.assertFalse(self.client.get(f'/api/runs/{off["id"]}/packet').json()['events'][0]['payload']['remediation_enabled'])
        self.assertIn('remediation', self.client.get('/').text)


if __name__ == '__main__':
    unittest.main()


class ResidualTests(unittest.TestCase):
    """The audit's remaining findings: operator visibility, repair context, expert labels, DB-path parity."""

    def test_the_report_shows_what_the_scope_judge_answered_and_why_it_was_refused(self):
        class Inventing(FixtureProvider):
            def scope(self, payload):
                value = super().scope(payload)
                value['scope_evidence'][0]['quote'] = 'A clause the scope never contained'
                return value
        packet = analyze(company(), policies(), sections(), Inventing(), ['CONC 7.3.4'])
        page = render(packet)
        self.assertIn('Uygulanabilirlik yargısının notları', page)
        self.assertIn('SCOPE_QUOTE_DROPPED', page)
        self.assertIn('A clause the scope never contained', page)

    def test_the_scope_repair_sees_its_own_previous_answer(self):
        seen = []

        class Watching(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'applicability' in schema['properties']:
                    seen.append(dict(payload))
                    value = json.loads(super()._chat(prompt, payload, schema))
                    value['basis'] = []
                    return json.dumps(value)
                return super()._chat(prompt, payload, schema)
        analyze(company(), policies(), sections(), Watching(), ['CONC 7.3.4'])
        self.assertEqual(len(seen), 2)
        self.assertNotIn('previous_response', seen[0])
        self.assertIn('"basis": []', seen[1]['previous_response'])
        self.assertIn('APPLIES requires', seen[1]['validation_feedback'])

    def test_an_expert_may_label_possibly_applies(self):
        from regchain.pilot.evaluation import evaluate, label_template
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        labels = label_template(packet)
        labels.update(reviewer='Expert', reviewer_role='Fixture')
        labels['decisions'][0].update(status='EXPERT_REVIEWED', expected_applicability='POSSIBLY_APPLIES', expected_coverage='COVERS_TEXT', rationale='r')
        result = evaluate(packet, labels)
        self.assertEqual(result['decisions']['applicability'], {'correct': 0, 'labelled': 1, 'accuracy': 0.0})   # exact state only
        self.assertEqual(result['decisions']['coverage']['correct'], 1)

    def test_the_database_path_applies_the_same_classification_gate(self):
        from regchain.extraction.service import classified
        row = {'text': DELEGATION, 'heading_path': ['t', 'c', 'Şüpheli işlem bildirimi'], 'legal_type': 'RULE'}
        kind, skipped = classified(row)
        self.assertEqual((kind.kind, skipped.reason, skipped.attempts), ('DELEGATION', 'CLASSIFIED_DELEGATION', 0))
        self.assertEqual(skipped.diagnostics[0]['code'], 'CLASSIFIED_DELEGATION')
        kind, skipped = classified({'text': 'A firm must retain records.', 'heading_path': [], 'legal_type': 'RULE'})
        self.assertEqual((kind.kind, skipped), ('OBLIGATION', None))
        kind, skipped = classified({'text': '(1) Bu Kanunda geçen; a) Bakanlık: Maliye Bakanlığını,', 'heading_path': ['t', 'c', 'Tanımlar'], 'legal_type': 'RULE'})
        self.assertEqual(kind.kind, 'DEFINITION')


class BasisToleranceTests(unittest.TestCase):
    """Measured live (23 September 2026): the judge answered APPLIES with two to eight basis pairs,
    most exact copies with a folded first letter, one or two paraphrased; one bad pair must not
    turn the whole answer into 'manual analysis required'."""

    def test_a_case_folded_copy_is_verified_and_an_invented_pair_is_dropped_and_named(self):
        class Live(FixtureProvider):
            def scope(self, payload):
                value = super().scope(payload)
                value['basis'] = [{'company_fact': 'consumer credit lending', 'regulatory_condition': 'this chapter applies to consumer credit lending', 'match': 'YES'},
                                  {'company_fact': 'consumer credit lending', 'regulatory_condition': 'payment services are obliged parties', 'match': 'YES'},
                                  {'company_fact': 'a fact the profile never stated', 'regulatory_condition': 'consumer credit lending', 'match': 'YES'},
                                  {'company_fact': 'consumer credit lending, loans', 'regulatory_condition': 'consumer credit lending.', 'match': 'YES'}]
                return value
        row = analyze(company(), policies(), sections(), Live(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'APPLIES')
        kept = row['proposal']['basis']
        self.assertEqual([b['regulatory_condition'] for b in kept], ['This chapter applies to consumer credit lending', 'consumer credit lending.'])
        self.assertEqual(kept[1]['company_fact'], 'consumer credit lending, loans')                 # a joined list still names the profile value
        dropped = [d for d in row['diagnostics'] if d.get('code') == 'BASIS_DROPPED']
        self.assertEqual(len(dropped), 1)
        self.assertIn('payment services are obliged parties', dropped[0]['detail'])
        self.assertIn('a fact the profile never stated', dropped[0]['detail'])
        self.assertIn('BASIS_DROPPED', render(analyze(company(), policies(), sections(), Live(), ['CONC 7.3.4'])))

    def test_an_answer_whose_every_pair_is_invented_is_refused_with_the_reason(self):
        class Invented(FixtureProvider):
            calls = 0

            def scope(self, payload):
                Invented.calls += 1
                value = super().scope(payload)
                value['basis'] = [{'company_fact': 'consumer credit lending', 'regulatory_condition': 'a phrase the scope never contains', 'match': 'YES'}]
                return value
        row = analyze(company(), policies(), sections(), Invented(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((Invented.calls, row['proposal']['applicability']), (2, 'UNKNOWN'))
        refusal = [d for d in row['diagnostics'] if d.get('code') == 'PROPOSAL_INVALID'][0]['detail']
        self.assertIn('APPLIES requires at least one verified basis', refusal)
        self.assertIn('a phrase the scope never contains', refusal)


class RealPdfReaderTests(unittest.TestCase):
    """Two habits of real regulator PDFs (measured 23 September 2026 on 88 public documents)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def blank(self, name, pages=1):
        from pypdf import PdfWriter
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(300, 300)
        path = self.root / name
        writer.write(path)
        return path

    def test_an_owner_locked_pdf_opens_and_a_user_locked_one_is_refused(self):
        from pypdf import PdfReader, PdfWriter
        from regchain.pilot.policies import read_policy
        source = self.blank('plain.pdf')
        # Owner-password-only encryption (empty user password): the FCA's publishing habit.
        locked = PdfWriter(clone_from=str(source))
        locked.encrypt(user_password='', owner_password='owner-only')
        locked_path = self.root / 'owner-locked.pdf'
        locked.write(locked_path)
        self.assertTrue(PdfReader(locked_path).is_encrypted)
        # The blank page still fails the text check, which proves decrypt('') ran before it.
        with self.assertRaisesRegex(ValueError, 'OCR'):
            read_policy(locked_path)
        with patch('pypdf._page.PageObject.extract_text', return_value='Section 1. Staff must retain records.'):
            self.assertEqual(read_policy(locked_path)['chunks'][0]['text'], 'Section 1. Staff must retain records.')
        user_locked = PdfWriter(clone_from=str(source))
        user_locked.encrypt(user_password='secret', owner_password='owner')
        user_path = self.root / 'user-locked.pdf'
        user_locked.write(user_path)
        with self.assertRaisesRegex(ValueError, 'decrypted by its owner'):
            read_policy(user_path)

    def test_a_minority_of_image_only_pages_is_skipped_and_counted_but_a_scanned_document_is_refused(self):
        from regchain.pilot.policies import read_policy
        path = self.blank('cover.pdf', pages=4)
        texts = ['', 'Section 1. Staff must retain records.', 'Section 2. Records are kept for eight years.', 'Section 3. Disposal is logged.']
        with patch('pypdf._page.PageObject.extract_text', side_effect=texts):
            value = read_policy(path)
        self.assertEqual(value['empty_pages'], [1])
        self.assertEqual([c['number'] for c in value['chunks']], [2, 3, 4])
        with patch('pypdf._page.PageObject.extract_text', side_effect=['', '', '', 'only one page has text']):
            with self.assertRaisesRegex(ValueError, "requires_ocr"):
                read_policy(path)

"""v0.16 regressions (23 September 2026, DemoPay / Garanti Ödeme / Tedbirler Yönetmeliği run):
the article filter is the only source of candidates and helper provisions are context; the
structural evidence gate; a CONFLICT needs a confirming second reading; support outweighs an
unclear crumb; applicability is shared per provision and a state that disagrees with its basis
is recorded as UNKNOWN, not "manual analysis required"; the repair prompt fits the window;
proposal failures are named and never end a run; the optional reranker falls back to RRF; the
answer cache and the enriched call log; the workspace shows the real scope of the analysis."""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from regchain.extraction.providers import (AI_TASK, ContextBudgetError, OllamaProvider, ProviderFailure, ai_context, ai_task, uncached,
                                           usage_summary)
from regchain.ingestion.mevzuat import article_heading, parse_mevzuat, text_url
from regchain.pilot.engine import (JUDGE_ALL_UNDER, JUDGE_WINDOW, JUDGE_WINDOW_RERANKED, analyze, coverage_of, judge_passage,
                                   passages_to_judge)
from regchain.pilot.policies import PARSER, cut_points, read_policy
from regchain.pilot.rerank import configured_reranker
from regchain.pilot.report import render
from regchain.pilot.schema import Company, PolicyCheck
from regchain.pilot.semantic import PolicyIndex, evidence_gate
from regchain.pilot.sources import application_rows, load_sources, save_sources, select_targets
from regchain.pilot.workspace import Workspace, create_app, metrics_of
from test_phase15 import LOCAL, TurkishRules, anadolu, turkish_sections
from test_pilot import FixtureProvider, company, policies, sections
from test_semantic import ConceptEmbedder, chunk, policy
from test_turkiye import download, fixture_fetch, p
from test_workspace import request_input, source_fixture

FIXTURES = Path(__file__).with_name('fixtures')
TEDBIRLER_URL = text_url('21', '200713012')


def madde(number, body):
    return f"<p class=MsoNormal style='text-align:justify'><b><span>MADDE {number} –</span></b><span> {body}</span></p>"


def regulation_html():
    """A small by-law in the shape of Tedbirler Yönetmeliği: scope/definition articles, duties in
    md. 3, 4(2), 5, 8 and 28, a commencement article."""
    return ('<html><head><meta http-equiv=Content-Type content="text/html; charset=Windows-1254"></head><body lang=TR><div class=WordSection1>'
            + p('TEDBİRLER HAKKINDA YÖNETMELİK (TEST)', center=True, bold=True)
            + p('Resmî Gazete Tarihi : 09/01/2008 Resmî Gazete Sayısı : 26751')
            + p('BİRİNCİ BÖLÜM', center=True, bold=True) + p('Amaç, Kapsam ve Tanımlar', center=True, bold=True)
            + p('Amaç ve kapsam', bold=True) + madde(1, '(1) Bu Yönetmeliğin amacı, suç gelirlerinin aklanmasının önlenmesine ilişkin yükümlülükleri düzenlemektir.')
            + p('Tanımlar', bold=True) + madde(2, '(1) Bu Yönetmelikte geçen; a) Yükümlü: Bankalar ile ödeme kuruluşları ve elektronik para kuruluşlarını, ifade eder.')
            + p('Kimlik tespiti', bold=True) + madde(3, '(1) Yükümlüler, müşterilerinin kimliğini tespit etmek zorundadır.')
            + p('Yükümlü', bold=True) + madde(4, '(1) Kanunun uygulanmasında yükümlü; a) Bankalar, b) Ödeme kuruluşları ile elektronik para kuruluşlarıdır. '
                                              '(2) Yükümlülerin şubeleri de yükümlü sayılır. '
                                              '(3) Merkezi yurt dışında bulunan yükümlünün Türkiye’deki şube ve temsilcileri de yükümlü sayılır.')
            + p('Gerçek kişilerde kimlik tespiti', bold=True) + madde(5, '(1) Gerçek kişilerin kimlik tespitinde ilgilinin adı ve doğum tarihi alınır.')
            + p('Ticaret siciline kayıtlı tüzel kişilerde kimlik tespiti', bold=True)
            + madde(7, '(1) Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde tüzel kişinin unvanı alınır.')
            + p('Dernek ve vakıflarda kimlik tespiti', bold=True) + madde(8, '(1) Derneklerin kimlik tespitinde derneğin adı ve kütük numarası alınır. '
                                                                          '(2) Vakıfların kimlik tespiti, sicil belgeleri esas alınmak suretiyle yapılır. '
                                                                          '(3) Yetkililerce istenildiğinde teyide esas belgelerin fotokopisi alınır.')
            + p('Yabancı dernek ve vakıflarda kimlik tespiti', bold=True)
            + madde(9, '(1) Yabancı dernek ve vakıfların Türkiye’deki şube ve temsilciliklerinin kimlik tespiti, İçişleri Bakanlığındaki kayda ilişkin belgeler esas alınmak suretiyle yapılır.')
            + p('Bankalarda muhabir ilişkisi', bold=True) + madde(10, '(1) Bankalar, muhabir ilişkilerinde ek tedbir almak zorundadır.')
            + p('Şüpheli işlem bildiriminde süre', bold=True) + madde(28, '(1) Yükümlüler, şüpheli işlemleri on iş günü içinde Başkanlığa bildirmek zorundadır.')
            + p('Yürürlük', bold=True) + madde(46, '(1) Bu Yönetmelik yayımı tarihinde yürürlüğe girer.')
            + '</div></body></html>')


def regulation_sections(root):
    save_sources(root, [download(regulation_html(), TEDBIRLER_URL)], mevzuat=('21', '200713012', '5'))
    return load_sources(root)[1]


class TargetFilterTests(unittest.TestCase):
    """Seen live: the form sent an empty filter, 51 articles were processed and md. 5 became a
    candidate although the operator had asked for md. 3, 4 and 8 only."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sections = regulation_sections(Path(self.temp.name) / 'reg')

    def test_only_the_filtered_articles_are_targets_and_the_rest_are_never_candidates(self):
        labels = select_targets(self.sections, 'YONETMELIK', '200713012', 'all', ['3', '4', '8'])
        self.assertEqual(labels, ['Yönetmelik 200713012 md. 3', 'Yönetmelik 200713012 md. 4', 'Yönetmelik 200713012 md. 8'])
        payload = analyze(anadolu(), policies(), self.sections, TurkishRules(), labels, target_filter=['3', '4', '8'])['events'][0]['payload']
        self.assertEqual([c['source']['printed_label'] for c in payload['cases']], labels)
        sources = {o['source_label'] for o in payload['obligations']}
        self.assertTrue(sources)                                                            # the targets yield candidates
        self.assertTrue(sources <= set(labels))
        for other in ('md. 5', 'md. 28', 'md. 1', 'md. 2', 'md. 46'):
            self.assertFalse(any(o['source_label'].endswith(other) for o in payload['obligations']), other)
        regulation = payload['regulation']
        self.assertEqual((regulation['target_filter'], regulation['targets'], regulation['target_provisions']), (['3', '4', '8'], labels, 3))
        self.assertEqual(regulation['targets_also_scope'], ['Yönetmelik 200713012 md. 4'])      # the obliged-party list is a target here
        page = render({'format': 'regchain-pilot-packet-v1', 'events': [{'payload': payload, 'event_hash': 'e' * 64}], 'head': 'e' * 64, 'count': 1})
        self.assertIn('Kullanıcının seçtiği hedef maddeler:</b> 3, 4, 8', page)
        self.assertIn('Analiz edilen hedef provision:</b> 3', page)

    def test_helper_provisions_are_read_as_context_but_never_become_candidates(self):
        labels = select_targets(self.sections, 'YONETMELIK', '200713012', 'all', ['3', '8'])
        payload = analyze(anadolu(), policies(), self.sections, TurkishRules(), labels, target_filter=['3', '8'])['events'][0]['payload']
        helpers = {h['label'].split(' md. ')[1]: h for h in payload['regulation']['helper_provisions']}
        # The scope, obliged-party and definition articles were read for applicability ...
        self.assertEqual({n for n, h in helpers.items() if h['role'] == 'scope'}, {'1', '2', '4'})
        self.assertEqual(helpers['4']['heading'], 'Yükümlü')
        self.assertEqual([s['printed_label'].split(' md. ')[1] for s in payload['scope_sources']], ['1', '2', '4'])
        # ... the neighbouring articles were read as extraction context (never as targets) ...
        context = {n: h for n, h in helpers.items() if h['role'] == 'context'}
        self.assertTrue(set(context) <= {'5', '7', '9', '10', '28', '46'}, set(context))
        self.assertTrue(all('neighbor' in h['reasons'] for h in context.values()))
        # ... and none of them, nor md. 5, is a case or a candidate.
        self.assertEqual([c['source']['printed_label'].split(' md. ')[1] for c in payload['cases']], ['3', '8'])
        self.assertFalse({'1', '2', '4', '5', '28', '46'} & {o['source_label'].split(' md. ')[1] for o in payload['obligations']})
        self.assertEqual(payload['regulation']['helper_count'], len(helpers))
        self.assertTrue(any('helper provisions' in text for text in payload['limitations']))

    def test_without_a_filter_every_operative_article_is_a_target_and_the_packet_says_so(self):
        labels = select_targets(self.sections, 'YONETMELIK', '200713012', 'all', [])
        self.assertEqual([label.split(' md. ')[1] for label in labels], ['3', '4', '5', '7', '8', '9', '10', '28'])
        payload = analyze(anadolu(), policies(), self.sections, TurkishRules(), labels, target_filter=[])['events'][0]['payload']
        self.assertEqual((payload['regulation']['target_filter'], payload['regulation']['target_provisions']), ([], 8))


def demopay(**changes):
    """The profile of the live run: a Turkish payment / e-money institution serving individuals, SMEs and merchants."""
    profile = dict(id='demopay-tr', name='DemoPay Teknoloji A.Ş.', version='1', synthetic=True, jurisdictions=['Türkiye'],
                   activities=['ödeme hizmetleri, elektronik para, dijital cüzdan'],
                   licences=['6493 sayılı Kanun kapsamında ödeme hizmetleri ve elektronik para faaliyeti'],
                   products=['dijital cüzdan, para transferi, sanal kart, üye işyeri ödemeleri'],
                   customer_types=["bireysel müşteriler, KOBİ'ler, üye işyerleri"],
                   description="Türkiye'de bireysel kullanıcılar ve üye işyerleri için dijital cüzdan, para transferi ve ödeme hizmetleri sunan sentetik fintech şirketi.")
    profile.update(changes)
    return Company(**profile)


class EntityGateTests(unittest.TestCase):
    """Seen live: md. 8 (associations and foundations) came back APPLIES for a payment institution
    whose customers are individuals, SMEs and merchants, because md. 4 lists payment institutions."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sections = regulation_sections(Path(self.temp.name) / 'reg')
        self.labels = select_targets(self.sections, 'YONETMELIK', '200713012', 'all', ['3', '4', '5', '7', '8', '9', '10'])

    def by_clause(self, payload):
        return {(o['source_label'].split(' md. ')[1], o['proposal']['applicability_scope']['child_clause']): o for o in payload['obligations']}

    def test_sub_clause_entities_decide_applicability_for_a_payment_institution(self):
        asked, scoped = [], []

        class Counting(TurkishRules):
            def _chat(self, prompt, payload, schema):
                if 'contradicts' in schema['properties']:
                    asked.append(payload['duty'])
                if 'applicability' in schema['properties']:
                    scoped.append(payload['source_label'])
                return super()._chat(prompt, payload, schema)
        payload = analyze(demopay(), policies(), self.sections, Counting(), self.labels)['events'][0]['payload']
        rows = self.by_clause(payload)
        final = {key: (row['proposal']['applicability'], row['proposal']['applicability_rule']) for key, row in rows.items()}
        # Generic KYC duty of every obliged party, individuals, registered legal entities: the provision-level APPLIES stands.
        self.assertEqual(final[('3', '(1)')], ('APPLIES', 'MODEL'))
        self.assertEqual(final[('5', '(1)')], ('APPLIES', 'MODEL'))
        self.assertEqual(final[('7', '(1)')], ('APPLIES', 'MODEL'))
        self.assertEqual(final[('4', '(2)')], ('APPLIES', 'MODEL'))
        # Associations, foundations, their foreign branches, banks and foreign-headquartered firms: not this profile.
        for key in (('8', '(1)'), ('8', '(2)'), ('8', '(3)'), ('9', '(1)'), ('10', '(1)'), ('4', '(3)')):
            self.assertEqual(final[key], ('DOES_NOT_APPLY', 'ENTITY_GATE'), key)
        scope = rows[('8', '(1)')]['proposal']['applicability_scope']
        self.assertEqual((scope['parent_provision'], scope['child_clause'], scope['match'], scope['rule'], scope['provision_assessed'], scope['final']),
                         ('Yönetmelik 200713012 md. 8', '(1)', 'MISMATCH', 'ENTITY_GATE', False, 'DOES_NOT_APPLY'))
        self.assertEqual([(r['role'], r['type'], r['match']) for r in scope['required_entities']], [('counterparty', 'ASSOCIATION', 'MISMATCH')])
        self.assertEqual(scope['company_entity_types'], ['PAYMENT_INSTITUTION', 'EMONEY_INSTITUTION'])
        self.assertEqual(scope['company_customer_families'], ['INDIVIDUAL', 'BUSINESS'])
        self.assertIn('DOES_NOT_APPLY for this profile', scope['reason'])
        self.assertIn('model judgement was not requested', rows[('8', '(1)')]['proposal']['applicability_reason'])
        self.assertIn('APPLICABILITY_SKIPPED', [d.get('code') for d in rows[('8', '(1)')]['diagnostics']])
        # The rule settles md. 8, 9 and 10 outright: the provision-level judge was asked only for md. 3, 4, 5 and 7.
        self.assertEqual(sorted(scoped), ['Yönetmelik 200713012 md. 3', 'Yönetmelik 200713012 md. 4', 'Yönetmelik 200713012 md. 5', 'Yönetmelik 200713012 md. 7'])
        self.assertEqual(rows[('4', '(2)')]['proposal']['applicability_scope']['provision_assessed'], True)
        # (3) names no party of its own: the article heading (dernek ve vakıflar) decides.
        self.assertEqual({r['source'] for r in rows[('8', '(3)')]['proposal']['applicability_scope']['required_entities']}, {'heading'})
        self.assertEqual(rows[('9', '(1)')]['proposal']['applicability_scope']['required_entities'][0]['type'], 'FOREIGN_ASSOCIATION_BRANCH')
        self.assertEqual([(r['role'], r['type']) for r in rows[('10', '(1)')]['proposal']['applicability_scope']['required_entities']][0], ('obliged_party', 'BANK'))
        self.assertEqual(rows[('4', '(3)')]['proposal']['applicability_scope']['required_entities'][0]['type'], 'FOREIGN_HQ_OBLIGED')
        # A clause that does not apply is not put to the judge and gets no proposal.
        excluded = rows[('8', '(1)')]['proposal']
        self.assertEqual((excluded['coverage_assessed'], excluded['policy_checks'], excluded['remediation_status'], excluded['remediation']),
                         (False, [], 'NOT_APPLICABLE', None))
        self.assertEqual(rows[('8', '(1)')]['judged_policy_ids'], [])
        self.assertEqual(len(asked), 4)                                                        # only the four applicable duties were judged
        metrics = metrics_of(payload)
        self.assertEqual((metrics['entity_gate_excluded'], metrics['coverage_not_assessed'], metrics['applicability']['DOES_NOT_APPLY']), (6, 6, 6))
        self.assertEqual(sum(metrics['coverage'].values()) + metrics['coverage_not_assessed'], metrics['candidates'])
        page = render({'format': 'regchain-pilot-packet-v1', 'events': [{'payload': payload, 'event_hash': 'e' * 64}], 'head': 'e' * 64, 'count': 1})
        for text in ('Ana provision', 'Alt bent', 'Gerekli varlık türü', 'Şirket profili varlık türü', 'Eşleşme', 'Sonuç',
                     'kural: entity gate', 'Değerlendirilmedi — bent şirkete uygulanmıyor', 'entity gate) profil dışı'):
            self.assertIn(text, page, text)

    def test_a_profile_with_association_customers_keeps_the_provision_level_answer(self):
        payload = analyze(demopay(customer_types=["bireysel müşteriler, KOBİ'ler, dernekler ve vakıflar"]), policies(), self.sections, TurkishRules(),
                          self.labels)['events'][0]['payload']
        rows = self.by_clause(payload)
        for key in (('8', '(1)'), ('8', '(2)'), ('8', '(3)'), ('9', '(1)')):
            self.assertEqual(rows[key]['proposal']['applicability'], 'APPLIES', key)
            self.assertEqual(rows[key]['proposal']['applicability_scope']['match'], 'MATCH', key)
        self.assertEqual(rows[('10', '(1)')]['proposal']['applicability'], 'DOES_NOT_APPLY')             # still not a bank

    def test_an_unstated_customer_base_or_a_generic_one_never_excludes(self):
        payload = analyze(demopay(customer_types=None), policies(), self.sections, TurkishRules(), self.labels)['events'][0]['payload']
        row = self.by_clause(payload)[('8', '(1)')]
        self.assertEqual((row['proposal']['applicability'], row['proposal']['applicability_scope']['match'], row['proposal']['applicability_scope']['rule']),
                         ('UNKNOWN', 'UNDETERMINED', 'PROVISION_LEVEL'))                              # incomplete profile: the existing rule
        payload = analyze(demopay(customer_types=['tüzel kişi müşteriler']), policies(), self.sections, TurkishRules(), self.labels)['events'][0]['payload']
        row = self.by_clause(payload)[('8', '(1)')]
        self.assertEqual((row['proposal']['applicability'], row['proposal']['applicability_scope']['match']), ('APPLIES', 'UNDETERMINED'))

    def test_the_gate_never_lifts_a_provision_level_does_not_apply(self):
        class Excluding(TurkishRules):
            def scope(self, payload):
                value = super().scope(payload)
                condition = payload['scope'][0]['text'][:120]
                return {**value, 'applicability': 'DOES_NOT_APPLY', 'basis': [{'company_fact': value['basis'][0]['company_fact'],
                                                                             'regulatory_condition': condition, 'match': 'NO'}],
                        'applicability_reason': 'Fixture: outside the scope.'}
        payload = analyze(demopay(), policies(), self.sections, Excluding(), ['Yönetmelik 200713012 md. 3'])['events'][0]['payload']
        row = payload['obligations'][0]['proposal']
        self.assertEqual((row['applicability'], row['applicability_rule'], row['applicability_scope']['match']), ('DOES_NOT_APPLY', 'MODEL', 'NOT_RESTRICTED'))


class BasisNoTests(unittest.TestCase):
    """The user's md. 4 case: YES, YES, NO. The NO must name its source, and only an explicit
    exclusion may veto two verified YES pairs."""

    def test_a_non_exclusionary_no_is_recorded_with_its_source_and_does_not_veto(self):
        class ListItemNo(FixtureProvider):
            calls = 0

            def scope(self, payload):
                ListItemNo.calls += 1
                value = super().scope(payload)
                scope_text = payload['scope'][0]['text']
                value['basis'] = [{'company_fact': 'consumer credit lending', 'regulatory_condition': scope_text[:48], 'match': 'YES'},
                                  {'company_fact': 'loans', 'regulatory_condition': scope_text[:48], 'match': 'YES'},
                                  {'company_fact': 'UK', 'regulatory_condition': 'consumer credit lending', 'match': 'NO'}]
                return value
        packet = analyze(company(), policies(), sections(), ListItemNo(), ['CONC 7.3.4'])
        row = packet['events'][0]['payload']['obligations'][0]
        self.assertEqual((ListItemNo.calls, row['proposal']['applicability'], row['proposal']['applicability_rule']), (1, 'APPLIES', 'MODEL'))
        no = [b for b in row['proposal']['basis'] if b['match'] == 'NO']
        self.assertEqual(len(no), 1)
        self.assertEqual((no[0]['exclusionary'], no[0]['source_id'], no[0]['company_fact_key']), (False, 's-scope', 'jurisdictions'))
        self.assertIn('not a veto', no[0]['note'])
        notes = [d for d in row['diagnostics'] if d.get('code') == 'BASIS_NO_NOT_EXCLUSIONARY']
        self.assertEqual(len(notes), 1)
        self.assertIn('[s1]', notes[0]['detail'])                                          # the judge's source id; the basis row carries the canonical one
        self.assertIn('"UK"', notes[0]['detail'])
        self.assertIn('not a veto', notes[0]['detail'])
        page = render(packet)
        self.assertIn('açık istisna değil, veto etmedi', page)
        self.assertIn('[CONC 7.1.1]', page)

    def test_an_exclusionary_no_still_refuses_an_applies_and_is_named_in_the_downgrade(self):
        scoped = [dict(sections()[0], text='This chapter applies to consumer credit lending, except firms that lend only to businesses.'), sections()[1]]

        class ExcludedNo(FixtureProvider):
            calls = 0

            def scope(self, payload):
                ExcludedNo.calls += 1
                value = super().scope(payload)
                value['basis'] = [{'company_fact': 'consumer credit lending', 'regulatory_condition': 'This chapter applies to consumer credit lending', 'match': 'YES'},
                                  {'company_fact': 'loans', 'regulatory_condition': 'consumer credit lending', 'match': 'YES'},
                                  {'company_fact': 'UK', 'regulatory_condition': 'except firms that lend only to businesses', 'match': 'NO'}]
                return value
        row = analyze(company(), policies(), scoped, ExcludedNo(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((ExcludedNo.calls, row['proposal']['applicability'], row['proposal']['applicability_rule']), (2, 'UNKNOWN', 'DOWNGRADED_BASIS_INCONSISTENT'))
        self.assertIn('exclusionary NO', row['proposal']['applicability_reason'])
        self.assertIn('except firms that lend only to businesses', row['proposal']['applicability_reason'])
        self.assertIn('BASIS_NO_EXCLUSIONARY', [d.get('code') for d in row['diagnostics']])
        refusal = [d for d in row['diagnostics'] if d.get('code') == 'PROPOSAL_INVALID'][0]['detail']
        self.assertIn('explicit exclusion', refusal)

    def test_exclusion_wording_is_recognised_and_abroad_is_not_outside_of(self):
        from regchain.pilot.engine import exclusion_marker
        self.assertEqual(exclusion_marker('Bu madde ödeme kuruluşları hakkında uygulanmaz.'), 'uygulanmaz')
        self.assertEqual(exclusion_marker('Bankalar hariç yükümlüler'), 'hariç')
        self.assertEqual(exclusion_marker('applies to firms other than banks'), 'other than')
        self.assertEqual(exclusion_marker('Merkezi yurt dışında bulunan yükümlünün Türkiye’deki şube ve temsilcileri'), '')
        self.assertEqual(exclusion_marker('e) Ödeme kuruluşları ile elektronik para kuruluşları.'), '')


class EvidenceGateTests(unittest.TestCase):
    def test_structural_crumbs_are_named_and_sentences_and_control_rows_pass(self):
        self.assertEqual(evidence_gate(chunk('h', 'Kurum İçi Sınırsız Kullanım / Kişisel Veri')), 'HEADING_FRAGMENT')
        self.assertEqual(evidence_gate(chunk('h2', 'Madde 5 – Kimlik Tespiti')), 'HEADING_FRAGMENT')
        self.assertEqual(evidence_gate(chunk('t', 'edürleri, • Programın uygulamadaki etkinliğini ve gelişmelerini test edecek denetimi,')), 'TORN_FRAGMENT')
        self.assertEqual(evidence_gate(chunk('n', '12')), 'PAGE_CRUMB')
        self.assertEqual(evidence_gate(chunk('n2', 'Sayfa 3 / 17')), 'PAGE_CRUMB')
        self.assertIsNone(evidence_gate(chunk('s', 'Müşteri bilgi ve belgeleri 8 yıl süre ile muhafaza edilir.')))
        self.assertIsNone(evidence_gate(chunk('s2', 'Staff retain records in archive 0. <script>alert(0)</script>')))
        self.assertIsNone(evidence_gate(chunk('m', 'Records must be kept')))                       # duty wording is never a crumb
        self.assertIsNone(evidence_gate(dict(chunk('k', 'Kontrol No: R-1 — Kanıt: yok'), locator='control_row')))
        # A long passage that starts mid-sentence (a page break) is text, not a torn crumb.
        self.assertIsNone(evidence_gate(chunk('l', 'uygun faaliyet gösterilmesini güvence altına alacak iç kontrol sistemini kurar ve ' * 3)))
        with patch.dict(os.environ, {'EVIDENCE_MIN_SIMILARITY': '0.35'}):
            self.assertEqual(evidence_gate(chunk('x', 'A sentence about something else entirely.'), similarity=0.1, lexical=0), 'LOW_RELEVANCE')
            self.assertIsNone(evidence_gate(chunk('x', 'A sentence about something else entirely.'), similarity=0.1, lexical=1))
        self.assertIsNone(evidence_gate(chunk('x', 'A sentence about something else entirely.'), similarity=0.1, lexical=0))  # off by default

    def test_a_heading_fragment_cannot_flip_the_verdict_on_its_own(self):
        class Unsure(FixtureProvider):
            def passage(self, payload):
                if 'Kişisel Veri' in payload['passage']:
                    return {'relation': 'UNCLEAR', 'quote': '', 'reason': 'Fixture cannot judge a heading.'}
                return super().passage(payload)
        docs = policy(chunk('crumb', 'Kurum İçi Sınırsız Kullanım / Kişisel Veri'), chunk('p1', 'All staff must retain records.'))
        for embedder in (None, ConceptEmbedder()):
            with self.subTest(retrieval='hybrid' if embedder else 'lexical'):
                packet = analyze(company(), docs, sections(), Unsure(), ['CONC 7.3.4'], embedder=embedder)
                row = packet['events'][0]['payload']['obligations'][0]
                self.assertEqual(row['proposal']['coverage'], 'COVERS_TEXT')                  # was UNKNOWN before the gate
                self.assertEqual(row['judged_policy_ids'], ['p1'])
                self.assertEqual(row['filtered_policy_ids'], ['crumb'])
                self.assertEqual(row['proposal']['filtered_passages'], [{'source_id': 'crumb', 'reason': 'HEADING_FRAGMENT'}])
                self.assertEqual([c['source_id'] for c in row['proposal']['policy_checks']], ['p1'])
                signals = {s['source_id']: s for s in row['evidence_signals']}
                self.assertEqual((signals['crumb']['gate'], signals['crumb']['judged'], signals['p1']['relation']), ('HEADING_FRAGMENT', False, 'SUPPORTS'))
                page = render(packet)
                self.assertIn('Kanıt kapısı', page)
                self.assertIn('başlık kırıntısı', page)
                self.assertIn('Kanıt hattı', page)
                self.assertNotIn('pasaj okunmadı', page)

    def test_a_reranker_shrinks_the_judge_window_and_is_never_a_gate(self):
        class Fake:
            calls = 0

            def manifest(self):
                return {'method': 'fake-v1', 'model': 'fake', 'max_length': 8, 'device': 'cpu', 'candidates': 20}

            def score(self, query, texts):
                Fake.calls += 1
                return [1.0 if 'archive 7' in text else 0.1 for text in texts]
        many = policy(*[chunk(f'p{i:02d}', f'Staff retain records in archive {i}.') for i in range(JUDGE_ALL_UNDER + 5)])
        packet = analyze(company(), many, sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder(), reranker=Fake())
        payload = packet['events'][0]['payload']
        row = payload['obligations'][0]
        self.assertEqual(len(row['judged_policy_ids']), JUDGE_WINDOW_RERANKED)
        self.assertEqual(row['judged_policy_ids'][0], 'p07')                                # the reranker's favourite leads
        self.assertEqual(payload['policy_retrieval']['method'], 'hybrid-rrf-rerank-v1')
        self.assertEqual(payload['policy_retrieval']['reranker']['status'], 'active')
        self.assertEqual(row['retrieval']['shown'][0]['rerank_score'], '1.0000')
        self.assertGreaterEqual(Fake.calls, 1)
        self.assertIn('Reranker', render(packet))
        # Without a reranker the window is the measured one.
        plain = analyze(company(), many, sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder())
        self.assertEqual(len(plain['events'][0]['payload']['obligations'][0]['judged_policy_ids']), JUDGE_WINDOW)


class RerankerFallbackTests(unittest.TestCase):
    def test_a_missing_or_unconfigured_reranker_falls_back_to_the_fused_order(self):
        with patch.dict(os.environ, {'RERANK_MODEL': ''}):
            reranker, status = configured_reranker()
        self.assertIsNone(reranker)
        self.assertEqual(status['status'], 'off')
        with patch.dict(os.environ, {'RERANK_MODEL': 'no-such-org/no-such-reranker-model', 'RERANK_ALLOW_DOWNLOAD': ''}):
            reranker, status = configured_reranker()
        self.assertIsNone(reranker)
        self.assertEqual(status['status'], 'unavailable')
        self.assertIn('RRF order', status['reason'])

    def test_a_faulty_reranker_is_recorded_and_the_run_keeps_the_fused_order(self):
        class Broken:
            def manifest(self):
                return {'method': 'broken'}

            def score(self, query, texts):
                raise RuntimeError('boom')
        index = PolicyIndex(policy(chunk('a', 'Staff keep archive files.'), chunk('b', 'Paper files are destroyed by shredding.')), ConceptEmbedder(), Broken())
        duty = {'subject': 'a firm', 'modality': 'MUST', 'required_action': 'retain records', 'prohibited_action': None, 'conditions': [], 'exceptions': []}
        chosen, record = index.select(duty)
        self.assertEqual([c['source_id'] for c in chosen], ['a', 'b'])
        self.assertIsNone(record['shown'][0]['rerank_score'])
        self.assertEqual(index.rerank_failures, ['RuntimeError'])
        self.assertEqual(index.manifest()['reranker']['failures'], 1)
        packet = analyze(company(), policy(chunk('a', 'All staff must retain records.')), sections(), FixtureProvider(), ['CONC 7.3.4'],
                         embedder=ConceptEmbedder(), reranker=Broken())
        self.assertEqual(packet['events'][0]['payload']['obligations'][0]['proposal']['coverage'], 'COVERS_TEXT')


class ConflictConfirmationTests(unittest.TestCase):
    """Seen live: a supportive sentence ('basitleştirilmiş tedbir uygulanmaz ve konu şüpheli işlem
    bildirimine ...') was called a contradiction of the reporting duty."""

    class TwoReadings(FixtureProvider):
        verdict = 'NOT_A_CONTRADICTION'

        def _chat(self, prompt, payload, schema):
            fields = schema['properties']
            if 'contradicts' in fields:
                return json.dumps({'contradicts': 'YES', 'quote': payload['passage'], 'reason': 'First reading saw a negative word.'})
            if 'verdict' in fields:
                return json.dumps({'verdict': self.verdict, 'duty_requires': 'report suspicious transactions',
                                   'passage_instructs': 'report the matter as a suspicious transaction', 'reason': 'The sentence requires the duty.'})
            return super()._chat(prompt, payload, schema)

    def test_a_supporting_sentence_is_not_a_conflict_after_the_second_reading(self):
        row = analyze(company(), policies(), sections(), self.TwoReadings(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual([c['relation'] for c in row['proposal']['policy_checks']], ['SUPPORTS'])
        self.assertEqual((row['proposal']['coverage'], row['signals']), ('COVERS_TEXT', []))
        result = row['diagnostics'][-1]['results'][0]
        self.assertEqual(result['screen'], 'WITHDRAWN')
        self.assertEqual([n['code'] for n in result['notes']], ['CONFLICT_WITHDRAWN'])
        self.assertIn('NOT_A_CONTRADICTION', result['notes'][0]['detail'])

    def test_a_confirmed_contradiction_keeps_its_quote_and_carries_the_rationale(self):
        class Confirmed(self.TwoReadings):
            verdict = 'CONTRADICTS'
        row = analyze(company(), policies(), sections(), Confirmed(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        check = row['proposal']['policy_checks'][0]
        self.assertEqual((check['relation'], check['quote']), ('CONFLICTS', 'All staff must retain records.'))
        reason = row['diagnostics'][-1]['results'][0]['reason']
        self.assertIn('Confirmed on a second reading', reason)
        self.assertIn('report suspicious transactions', reason)
        self.assertEqual((row['proposal']['coverage'], row['signals']), ('CONFLICT', ['POTENTIAL_POLICY_CONFLICT']))
        self.assertIn('Çelişki gerekçesi (ikinci okumayla doğrulandı)', render(analyze(company(), policies(), sections(), Confirmed(), ['CONC 7.3.4'])))

    def test_an_unanswered_second_reading_leaves_the_conflict_flagged_for_review(self):
        class Outage(self.TwoReadings):
            def _chat(self, prompt, payload, schema):
                if 'verdict' in schema['properties']:
                    raise ProviderFailure('down')
                return super()._chat(prompt, payload, schema)
        row = analyze(company(), policies(), sections(), Outage(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['coverage'], 'CONFLICT')                             # fail closed: a person looks at it
        self.assertIn('CONFLICT_UNCONFIRMED', [n['code'] for n in row['diagnostics'][-1]['results'][0]['notes']])

    def test_an_unclear_contradiction_answer_without_a_quote_falls_through_to_the_support_question(self):
        class Silent(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'contradicts' in schema['properties']:
                    return json.dumps({'contradicts': 'UNCLEAR', 'quote': '', 'reason': 'The passage is silent on the duty.'})
                return super()._chat(prompt, payload, schema)
        judgement, notes, screen = judge_passage(Silent(), {'required_action': 'retain records'}, policies()[0]['chunks'][0])
        self.assertEqual((judgement.relation, screen), ('SUPPORTS', 'UNCLEAR'))
        self.assertEqual([n['code'] for n in notes], ['SCREEN_UNCLEAR'])

        class Both(Silent):
            def passage(self, payload):
                return {'relation': 'UNCLEAR', 'quote': '', 'reason': 'Cannot tell.'}
        judgement, notes, screen = judge_passage(Both(), {'required_action': 'retain records'}, policies()[0]['chunks'][0])
        self.assertEqual(judgement.relation, 'UNCLEAR')


class CoverageAggregationTests(unittest.TestCase):
    def test_strong_support_outweighs_one_unclear_passage_but_a_tie_does_not(self):
        checks = lambda *relations: [PolicyCheck(source_id=f'p{i}', quote='q', relation=r) for i, r in enumerate(relations)]
        cases = [(('SUPPORTS',) * 5 + ('UNCLEAR', 'UNRELATED'), 'COVERS_TEXT'), (('SUPPORTS', 'SUPPORTS', 'UNCLEAR'), 'COVERS_TEXT'),
                 (('SUPPORTS', 'PARTIAL', 'UNCLEAR'), 'COVERS_TEXT'), (('PARTIAL', 'PARTIAL', 'UNCLEAR'), 'PARTIAL'),
                 (('SUPPORTS', 'UNCLEAR'), 'UNKNOWN'), (('UNCLEAR', 'UNRELATED'), 'UNKNOWN'), (('PARTIAL', 'UNCLEAR'), 'UNKNOWN'),
                 (('SUPPORTS',) * 5 + ('UNCLEAR', 'CONFLICTS'), 'CONFLICT')]
        for relations, expected in cases:
            with self.subTest(relations=relations):
                coverage, reason, _ = coverage_of(checks(*relations))
                self.assertEqual(coverage, expected)
        coverage, reason, _ = coverage_of(checks('SUPPORTS', 'SUPPORTS', 'UNCLEAR'))
        self.assertIn('1 passage(s) could not be judged', reason)
        self.assertIn('do not outweigh the support', reason)

    def test_a_control_row_alone_never_makes_written_coverage(self):
        controls = {'k1'}
        row = lambda sid, relation: PolicyCheck(source_id=sid, quote='q', relation=relation)
        self.assertEqual(coverage_of([row('k1', 'SUPPORTS')], controls)[::2], ('NO_EVIDENCE', 'SUPPORTS'))
        self.assertIn('operational evidence', coverage_of([row('k1', 'SUPPORTS')], controls)[1])
        self.assertEqual(coverage_of([row('p1', 'SUPPORTS'), row('k1', 'SUPPORTS')], controls)[::2], ('COVERS_TEXT', 'SUPPORTS'))
        self.assertEqual(coverage_of([row('p1', 'SUPPORTS'), row('k1', 'CONFLICTS')], controls)[::2], ('CONFLICT', 'CONFLICT'))
        self.assertEqual(coverage_of([row('p1', 'UNRELATED')], controls)[2], 'NOT_READ')
        self.assertEqual(coverage_of([row('p1', 'UNRELATED'), row('k1', 'UNRELATED')], controls)[2], 'NONE')
        # End to end: the register supports, the policy is silent; the metric counts the row apart.
        register = {'filename': 'kontroller.csv', 'raw_hash': 'e' * 64, 'bytes': 30, 'parser': 'fixture',
                    'chunks': [{'source_id': 'k1', 'policy_hash': 'e' * 64, 'filename': 'kontroller.csv', 'locator': 'control_row', 'number': 1,
                                'start': 0, 'end': 40, 'text': 'kontrol_no: K-01 — ad: Staff must retain records.'}]}

        class ControlOnly(FixtureProvider):
            def passage(self, payload):
                if payload['passage'].startswith('kontrol_no'):
                    return super().passage(payload)
                return {'relation': 'UNRELATED', 'quote': '', 'reason': 'About something else.'}
        packet = analyze(company(), [*policies(), register], sections(), ControlOnly(), ['CONC 7.3.4'])
        proposal = packet['events'][0]['payload']['obligations'][0]['proposal']
        self.assertEqual((proposal['coverage'], proposal['control_coverage']), ('NO_EVIDENCE', 'SUPPORTS'))
        self.assertEqual(metrics_of(packet['events'][0]['payload'])['covered'], 0)
        page = render(packet)
        self.assertIn('kontrol kaydı destekliyor', page)
        self.assertIn('Kontrol kaydı satırı', page)


class ApplicabilityTests(unittest.TestCase):
    def test_a_state_that_disagrees_with_its_verified_basis_is_recorded_as_unknown_not_as_unavailable(self):
        # Seen live (md. 10, 31, 44): DOES_NOT_APPLY with only YES pairs, twice; the duty ended
        # "PROPOSAL_INVALID: manual analysis required" and counted as "AI could not propose".
        class Contrary(FixtureProvider):
            calls = 0

            def scope(self, payload):
                Contrary.calls += 1
                return {**super().scope(payload), 'applicability': 'DOES_NOT_APPLY',
                        'applicability_reason': 'The duty concerns political parties, not the company.'}
        packet = analyze(company(), policies(), sections(), Contrary(), ['CONC 7.3.4'])
        row = packet['events'][0]['payload']['obligations'][0]
        self.assertEqual(Contrary.calls, 2)
        self.assertEqual((row['proposal']['applicability'], row['proposal']['applicability_rule']), ('UNKNOWN', 'DOWNGRADED_BASIS_INCONSISTENT'))
        self.assertTrue(row['proposal']['applicability_reason'].startswith('The duty concerns political parties'))
        self.assertIn('[Rule: the model answered DOES_NOT_APPLY', row['proposal']['applicability_reason'])
        self.assertEqual(len(row['proposal']['basis']), 1)                                    # the verified pairs are kept for the reviewer
        self.assertIn('DOWNGRADED_BASIS_INCONSISTENT', [d.get('code') for d in row['diagnostics']])
        self.assertEqual(metrics_of(packet['events'][0]['payload'])['ai_unavailable'], 0)
        self.assertIn('DOWNGRADED_BASIS_INCONSISTENT', render(packet))

    def test_applicability_is_judged_once_per_provision_and_shared_by_its_duties(self):
        seen = []

        class Counting(TurkishRules):
            def _chat(self, prompt, payload, schema):
                if 'applicability' in schema['properties']:
                    seen.append(payload)
                return super()._chat(prompt, payload, schema)
        payload = analyze(anadolu(), policies(), turkish_sections(), Counting(), ['Kanun 5549 md. 4'])['events'][0]['payload']
        self.assertEqual(len(payload['obligations']), 2)
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]['sibling_obligations']), 1)
        first, second = payload['obligations']
        self.assertEqual((first['proposal']['applicability'], second['proposal']['applicability']), ('APPLIES', 'APPLIES'))
        self.assertEqual((first['applicability_siblings'], second['applicability_siblings']), (1, 1))
        self.assertIn('APPLICABILITY_SHARED', [d.get('code') for d in second['diagnostics']])
        self.assertEqual(first['proposal']['basis'][0]['company_fact_key'], 'activities')

    def test_a_repair_prompt_that_no_longer_fits_is_repeated_without_the_previous_answer(self):
        # Seen live three times: the repair prompt carried 6,000 characters of the model's own
        # answer and no longer fitted the window, so the duty ended "Analysis unavailable".
        class Budget(FixtureProvider):
            attempts = 0

            def _chat(self, prompt, payload, schema):
                if 'applicability' not in schema['properties']:
                    return super()._chat(prompt, payload, schema)
                Budget.attempts += 1
                if Budget.attempts == 1:
                    value = self.scope(payload)
                    value['scope_evidence'][0]['quote'] = 'an invented clause'
                    return json.dumps(value)
                if 'previous_response' in payload:
                    raise ContextBudgetError('CONTEXT_BUDGET_EXCEEDED: prompt and reserved output exceed admission budget')
                return json.dumps(self.scope(payload))
        row = analyze(company(), policies(), sections(), Budget(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(Budget.attempts, 3)
        self.assertEqual(row['proposal']['applicability'], 'APPLIES')
        notes = [d for d in row['diagnostics'] if d.get('code') == 'ContextBudgetError']
        self.assertEqual(notes[0]['retry'], 'repeated without previous_response')


class ProposalFallbackTests(unittest.TestCase):
    def test_a_malformed_draft_is_named_retried_once_and_never_ends_the_run(self):
        class Garbled(FixtureProvider):
            def passage(self, payload):
                return {'relation': 'UNRELATED', 'quote': '', 'reason': 'About something else.'}

            def _chat(self, prompt, payload, schema):
                if 'clause' in schema['properties']:
                    return 'not json at all'
                return super()._chat(prompt, payload, schema)
        packet = analyze(company(), policies(), sections(), Garbled(), ['CONC 7.3.4'])
        row = packet['events'][0]['payload']['obligations'][0]
        item = row['proposal']['remediation']
        self.assertEqual((item['type'], item['draft_status'], item['draft_failure'], item['draft_attempts']), ('NEW_POLICY_CLAUSE', 'DRAFT_UNAVAILABLE', 'MALFORMED_JSON', 2))
        self.assertEqual(item['suggested_policy_language'], '')
        self.assertIn('could not be drafted (MALFORMED_JSON)', item['implementation_notes'])
        self.assertEqual(row['proposal']['remediation_status'], 'PROPOSED')
        metrics = metrics_of(packet['events'][0]['payload'])
        self.assertEqual((metrics['remediations'], metrics['proposal_failed'], metrics['proposal_status']), (1, 1, {'PROPOSED': 1}))
        self.assertIn('taslağı üretilemedi', render(packet))

    def test_every_remediation_status_is_explicit(self):
        covered = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((covered['proposal']['remediation_status'], covered['proposal']['remediation']), ('NOT_NEEDED', None))

        class Unsure(FixtureProvider):
            def passage(self, payload):
                return {'relation': 'UNCLEAR', 'quote': '', 'reason': 'Fixture cannot decide.'}
        unknown = analyze(company(), policies(), sections(), Unsure(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((unknown['proposal']['coverage'], unknown['proposal']['remediation_status'], unknown['proposal']['remediation']),
                         ('UNKNOWN', 'UNDETERMINED', None))
        off = analyze(company(), policies(), sections(), Unsure(), ['CONC 7.3.4'], remediation=False)['events'][0]['payload']['obligations'][0]
        self.assertEqual(off['proposal']['remediation_status'], 'DISABLED')
        page = render(analyze(company(), policies(), sections(), Unsure(), ['CONC 7.3.4']))
        self.assertIn('kapsam belirsiz; öneri üretilmedi', page)


class CacheAndLogTests(unittest.TestCase):
    def client(self, factory):
        client = factory.return_value.__enter__.return_value
        client.get.return_value.json.return_value = {'models': [{'name': 'm', 'digest': 'd'}]}
        stream = client.stream.return_value.__enter__.return_value
        stream.iter_bytes.return_value = [b'{"message": {"content": "{\\"a\\": 1}"}, "done": true, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 9}']
        return client

    def test_the_same_request_is_answered_once_and_the_record_names_what_it_was_about(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client = self.client(factory)
            with ai_task('judge.contradicts'), ai_context(provision_id='prov-1', obligation_id='obl-1', evidence_ids=['pass-1']):
                first = provider.generate_structured('p', {'duty': 'd', 'passage': 'x'}, {'type': 'object'})
                second = provider.generate_structured('p', {'duty': 'd', 'passage': 'x'}, {'type': 'object'})
                with uncached():
                    third = provider.generate_structured('p', {'duty': 'd', 'passage': 'x'}, {'type': 'object'})
                provider.generate_structured('p', {'duty': 'd', 'passage': 'y'}, {'type': 'object'})
            self.assertEqual(client.stream.call_count, 3)                                      # x once, x again uncached, y once
        self.assertEqual((first, second, third), ('{"a": 1}', '{"a": 1}', '{"a": 1}'))
        entries = provider.call_log
        self.assertEqual([e['cache_hit'] for e in entries], [False, True, False, False])
        for key in ('stage', 'provision_id', 'obligation_id', 'evidence_ids', 'cache_hit', 'elapsed_ms', 'prompt_tokens', 'output_tokens',
                    'model', 'model_version', 'retry_count', 'payload_sha256', 'num_predict'):
            self.assertIn(key, entries[0], key)
        self.assertEqual((entries[0]['stage'], entries[0]['provision_id'], entries[0]['obligation_id'], entries[0]['evidence_ids'], entries[0]['model']),
                         ('judge.contradicts', 'prov-1', 'obl-1', ['pass-1'], 'm'))
        self.assertEqual((entries[1]['elapsed_ms'], entries[1]['prompt_tokens'], entries[1]['done_reason']), (0, 0, 'cache'))
        self.assertNotEqual(entries[0]['payload_sha256'], entries[3]['payload_sha256'])
        for entry in entries:
            self.assertNotIn('passage', json.dumps(entry).replace('"passage"', ''))         # keys only, never the text
        summary = usage_summary([provider.call_log])
        self.assertEqual((summary['calls'], summary['cache_hits'], summary['by_task']['judge.contradicts']['cache_hits']), (4, 1, 1))
        self.assertEqual(AI_TASK.get(), '')

    def test_the_same_passage_and_duty_pair_is_judged_once_per_run(self):
        asked = []

        class Counting(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'contradicts' in schema['properties']:
                    asked.append(payload['passage'])
                return super()._chat(prompt, payload, schema)
        docs = policy(chunk('a', 'All staff must retain records.'), chunk('b', 'Staff keep archive files for six years.'))
        analyze(company(), docs, sections(), Counting(), ['CONC 7.3.4'], embedder=ConceptEmbedder())
        self.assertEqual(sorted(asked), sorted(['All staff must retain records.', 'Staff keep archive files for six years.']))


class PolicyReaderTests(unittest.TestCase):
    def test_v3_cuts_long_blocks_at_sentence_ends_and_leaves_no_crumb(self):
        sentence = 'Yükümlüler müşterilerinin kimliğini tespit eder ve kayıtları sekiz yıl saklar. '
        block = (sentence * 40).strip()                                                       # about 3,200 characters
        pieces = cut_points(block, 0, len(block), PARSER)
        self.assertEqual(''.join(block[a:b] for a, b in pieces), block)
        self.assertTrue(all(b - a <= 2000 for a, b in pieces))
        self.assertTrue(all(block[a:b].rstrip().endswith('.') for a, b in pieces))          # cut at sentence ends
        self.assertGreaterEqual(min(b - a for a, b in pieces), 300)                           # no tail crumb
        short = (sentence * 26).strip()                                                       # about 2,080: one piece, not 2,000 + 80
        self.assertEqual(cut_points(short, 0, len(short), PARSER), [(0, len(short))])
        self.assertEqual(cut_points(short, 0, len(short), 'pilot-policy-v2'), [(0, 2000), (2000, len(short))])
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'policy.txt'
        path.write_text(block + '\n', encoding='utf-8')
        value = read_policy(path)
        self.assertEqual(value['parser'], 'pilot-policy-v3')
        self.assertEqual(''.join(c['text'] for c in value['chunks']), block)
        self.assertEqual(len(read_policy(path, parser='pilot-policy-v2')['chunks']), 2)


class TedbirlerSnapshotTests(unittest.TestCase):
    """The retained live download of Tedbirler Yönetmeliği (23 September 2026)."""

    def test_the_regulation_parses_into_62_articles_without_footnotes(self):
        html = FIXTURES.joinpath('tedbirler-200713012.html').read_text(encoding='utf-8')
        document = parse_mevzuat(download(html, TEDBIRLER_URL))
        self.assertEqual(len(document.paragraphs), 62)
        numbers = [a.number for a in document.paragraphs]
        self.assertTrue({'3', '4', '5', '6/A', '8', '28', '32', '46'} <= set(numbers))
        for article in document.paragraphs:
            self.assertNotIn('––––', article.text, article.number)
            self.assertNotIn('_____', article.text, article.number)
            self.assertNotIn('değiştirilmiştir.', article.text, article.number)                # the page-bottom footnotes
            self.assertNotIn('ibaresi eklenmiştir.', article.text, article.number)
        by_number = {a.number: a for a in document.paragraphs}
        self.assertIn('Ödeme kuruluşları ile elektronik para kuruluşları', by_number['4'].text)
        self.assertTrue(by_number['8'].text.rstrip().endswith('esas alınmak suretiyle yapılır.'))
        self.assertEqual(article_heading({'heading_path': list(by_number['4'].heading_path)}), 'Yükümlü')
        # v2 read the underscore-ruled footnote block of md. 8 into the article; that reading is
        # still reproducible for the retained snapshot, and only there.
        old = parse_mevzuat(download(html, TEDBIRLER_URL), 'mevzuat-parser-v2')
        self.assertIn('değiştirilmiştir.', {a.number: a for a in old.paragraphs}['8'].text)
        self.assertNotEqual(old.content_hash, document.content_hash)

    def test_the_retained_snapshot_still_verifies_and_the_filter_selects_three_targets(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / 'sources'
        root.mkdir()
        snapshot = json.loads(FIXTURES.joinpath('tedbirler-snapshot.json').read_text(encoding='utf-8'))
        shutil.copy(FIXTURES / 'tedbirler-200713012.html', root / snapshot['sources'][0]['raw_file'])
        (root / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf-8')
        _, sections_ = load_sources(root)                                                    # bytes and parsed hash reproduce
        self.assertEqual(len(sections_), 62)
        self.assertEqual(select_targets(sections_, 'YONETMELIK', '200713012', 'all', ['3', '4', '8']),
                         ['Yönetmelik 200713012 md. 3', 'Yönetmelik 200713012 md. 4', 'Yönetmelik 200713012 md. 8'])
        # Scope evidence: the purpose/scope article and the obliged-party list. Before v0.16 the
        # heading heuristic also took md. 32 ("Yükümlüler tarafından devamlı bilgi verme", a duty)
        # and md. 35 ("Denetimin kapsamı", a procedure). The 3,800-character definitions article
        # (md. 3) still does not fit the 7,000-character scope budget beside md. 4.
        self.assertEqual([r['paragraph_number'] for r in application_rows(sections_, 'YONETMELIK', '200713012')], ['1', '4'])
        self.assertEqual(len(select_targets(sections_, 'YONETMELIK', '200713012', 'all', [])), 51)


class WorkspaceScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source_fixture(root / 'launcher-sources')

        def fetch(directory, **kwargs):
            return fixture_fetch(directory, **kwargs) if kwargs.get('mevzuat') else source_fixture(directory, **kwargs)
        self.workspace = Workspace(root / 'workspace', cached_sources=root / 'launcher-sources',
                                   provider_factory=lambda _: TurkishRules(), source_fetcher=fetch,
                                   reranker_factory=lambda: (None, {'status': 'unavailable', 'reason': 'test'}))
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                                 headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def finish(self, **changes):
        response = self.client.post('/api/runs', json=request_input(**changes))
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['id']
        self.assertEqual(response.json()['targets_requested'], changes.get('sections', []))    # visible from the first second
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def test_the_job_record_shows_targets_helpers_counts_and_duration(self):
        row = self.finish(provider='ollama', regulator='TR', mevzuat_kind='1', mevzuat_number='5549', sections=['3', '4'], labels=[])
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        self.assertEqual((row['targets_requested'], row['targets'], row['target_count']), (['3', '4'], ['Kanun 5549 md. 3', 'Kanun 5549 md. 4'], 2))
        self.assertEqual([h['label'] for h in row['helper_provisions']], ['Kanun 5549 md. 2'])
        self.assertEqual(row['helper_count'], 1)
        self.assertIsInstance(row['duration_seconds'], int)
        self.assertEqual(row['reranker']['status'], 'off')                                    # lexical retrieval: no reranking
        self.assertIn('kelime', row['reranker']['reason'])
        metrics = row['metrics']
        for key in ('applicability', 'coverage', 'proposal_status', 'proposal_failed', 'ai_unavailable', 'pending_human_review'):
            self.assertIn(key, metrics)
        self.assertEqual(sum(metrics['applicability'].values()), metrics['candidates'])
        self.assertEqual(sum(metrics['coverage'].values()), metrics['candidates'])
        self.assertEqual(metrics['pending_human_review'], metrics['candidates'])
        self.assertIn('passages_filtered', row)
        packet = self.client.get(f'/api/runs/{row["id"]}/packet').json()['events'][0]['payload']
        self.assertEqual(packet['regulation']['target_filter'], ['3', '4'])
        self.assertEqual(packet['regulation']['targets'], row['targets'])
        page = self.client.get(f'/api/runs/{row["id"]}/report').text
        self.assertIn('Analiz kapsamı', page)
        self.assertIn('Kullanıcının seçtiği hedef maddeler:</b> 3, 4', page)

    def test_the_form_previews_the_filter_and_names_both_source_labels(self):
        html = self.client.get('/').text
        for text in ('articlesPreview', 'parseArticles', "mevzuat.gov.tr'dan güncel kaynağı indir", "FCA'dan güncel kaynakları indir",
                     'Madde filtresi boş', 'Analizin gerçek kapsamı', 'AI önerilerinin sayımı; uyum oranı değildir.', 'Pilot v0.16'):
            self.assertIn(text, html, text)
        self.assertIn('rerank_model', self.client.get('/api/state').json())


if __name__ == '__main__':
    unittest.main()

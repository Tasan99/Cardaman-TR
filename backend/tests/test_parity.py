"""Cardamon parity: control register, summaries and categories, sanction links, ask/draft/assign."""
import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from regchain.pilot.engine import DEFAULT_CATEGORIES, analyze, answer_question, draft_clause, sanction_links
from regchain.pilot.policies import read_policy
from regchain.pilot.report import render
from regchain.pilot.sources import load_sources, save_sources
from regchain.pilot.workspace import Workspace, create_app, metrics_of
from test_pilot import FixtureProvider, company, policies, sections
from test_turkiye import download, law_html, p
from test_workspace import request_input, source_fixture


class ControlRegisterTests(unittest.TestCase):
    def test_a_csv_control_register_becomes_one_passage_per_control(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'kontroller.csv'
            path.write_text('kontrol_no,ad,açıklama,sorumlu\nK-01,Kayıt saklama,"Müşteri kayıtları sekiz yıl saklanır",Uyum\nK-02,,"Kimlik tespiti\nher işlemde",Operasyon\n', encoding='utf-8')
            value = read_policy(path)
        self.assertEqual([c['locator'] for c in value['chunks']], ['control_row', 'control_row'])
        self.assertEqual(value['chunks'][0]['text'], 'kontrol_no: K-01 — ad: Kayıt saklama — açıklama: Müşteri kayıtları sekiz yıl saklanır — sorumlu: Uyum')
        self.assertEqual(value['chunks'][1]['text'], 'kontrol_no: K-02 — açıklama: Kimlik tespiti her işlemde — sorumlu: Operasyon')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bos.csv'
            path.write_text('kontrol_no,ad\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'en az bir kontrol'):
                read_policy(path)

    def test_control_coverage_is_counted_separately_from_policy_coverage(self):
        control = {'filename': 'kontroller.csv', 'raw_hash': 'e'*64, 'bytes': 30, 'parser': 'fixture',
                   'chunks': [{'source_id': 'k1', 'policy_hash': 'e'*64, 'filename': 'kontroller.csv', 'locator': 'control_row', 'number': 1,
                               'start': 0, 'end': 40, 'text': 'kontrol_no: K-01 — ad: Staff must retain records.'}]}
        packet = analyze(company(), [*policies(), control], sections(), FixtureProvider(), ['CONC 7.3.4'], categories=DEFAULT_CATEGORIES)
        data = packet['events'][0]['payload']
        metrics = metrics_of(data)
        self.assertEqual((metrics['candidates'], metrics['applicable'], metrics['covered'], metrics['control_rows'], metrics['control_covered']), (1, 1, 1, 1, 1))
        page = render(packet)
        self.assertIn('Kontrol kaydı kanıtı: 1 / 1 (%100)', page)
        self.assertIn('Uygulanabilir (APPLIES): 1 / 1 (%100)', page)
        # v0.16: a control row alone never makes written coverage; here the policy passage does.
        row = data['obligations'][0]['proposal']
        self.assertEqual((row['coverage'], row['control_coverage']), ('COVERS_TEXT', 'SUPPORTS'))
        without = metrics_of(analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload'])
        self.assertEqual((without['control_rows'], without['control_covered']), (0, 0))


class OrientationTests(unittest.TestCase):
    def test_every_candidate_gets_a_summary_and_a_category_that_are_marked_as_orientation(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'], categories=DEFAULT_CATEGORIES)
        data = packet['events'][0]['payload']
        row = data['obligations'][0]
        self.assertEqual((row['summary'], row['category']), ('Fixture summary of the duty.', DEFAULT_CATEGORIES[0]))
        self.assertEqual(data['risk_taxonomy'], list(DEFAULT_CATEGORIES))
        self.assertTrue(any('paraphrases' in text for text in data['limitations']))
        page = render(packet)
        self.assertIn('AI özeti (yönlendirme amaçlı, kanıt değil)', page)
        self.assertIn('Risk kategorisine göre', page)
        # No categories, or a provider without a model: nothing is invented.
        plain = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']
        self.assertEqual((plain['obligations'][0]['summary'], plain['obligations'][0]['category']), ('', None))

    def test_turkish_penalty_articles_are_linked_to_the_duty_articles_they_cite(self):
        html = law_html().replace("<p class=MsoNormal style='text-align:justify'><b><span>MADDE 5/A- (Ek: 18/6/2014-6545/87 md.)</span></b></p>",
                                  p('Yükümlülük ihlâlinde idarî ceza', bold=True)
                                  + "<p class=MsoNormal style='text-align:justify'><b><span>MADDE 5/B –</span></b><span> (1) Bu Kanunun 3 üncü maddesi kapsamındaki yükümlülüğü ihlal eden yükümlülere elli bin lira idari para cezası verilir.</span></p>"
                                  + "<p class=MsoNormal style='text-align:justify'><b><span>MADDE 5/A- (Ek: 18/6/2014-6545/87 md.)</span></b></p>")
        with tempfile.TemporaryDirectory() as directory:
            save_sources(Path(directory)/'tr', [download(html)], mevzuat=('1', '5549', '5'))
            _, tr_sections = load_sources(Path(directory)/'tr')
        links = sanction_links(tr_sections, 'KANUN')
        self.assertEqual(list(links), ['3'])
        self.assertEqual(links['3'][0]['label'], 'Kanun 5549 md. 5/B')
        self.assertIn('idari para cezası verilir', links['3'][0]['quote'])
        self.assertEqual(sanction_links(sections(), 'CONC'), {})
        packet = analyze(company(), policies(), tr_sections, FixtureProvider(), ['Kanun 5549 md. 3', 'Kanun 5549 md. 4'])
        rows = {o['source_label']: o for o in packet['events'][0]['payload']['obligations']}
        self.assertEqual(rows['Kanun 5549 md. 3']['sanctions'][0]['label'], 'Kanun 5549 md. 5/B')
        self.assertEqual(rows['Kanun 5549 md. 4']['sanctions'], [])
        self.assertIn('Yaptırım bağlantısı · Kanun 5549 md. 5/B', render(packet))


class AskAndDraftTests(unittest.TestCase):
    def test_an_answer_carries_only_citations_that_exist_in_the_sources(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        data = packet['events'][0]['payload']
        result = answer_question(FixtureProvider(), 'Kayıtlar ne kadar saklanmalı?', sections(), data['policies'], data['obligations'])
        self.assertEqual(result['answer'], 'Fixture answer.')
        self.assertEqual(len(result['citations']), 1)
        self.assertEqual(result['citations_dropped'], 0)
        self.assertTrue(any(s['kind'] == 'analysis' for s in result['sources_read']))

        class Inventing(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'answer' in schema['properties']:
                    return json.dumps({'answer': 'Made up.', 'citations': [{'source_id': payload['sources'][0]['source_id'], 'quote': 'never in the source'}]})
                return super()._chat(prompt, payload, schema)
        result = answer_question(Inventing(), 'Soru?', sections(), data['policies'], data['obligations'])
        self.assertEqual((result['citations'], result['citations_dropped']), ([], 1))
        with self.assertRaisesRegex(ValueError, '3 ile 500'):
            answer_question(FixtureProvider(), 'a', sections(), data['policies'], data['obligations'])
        from regchain.extraction.providers import RulesProvider
        with self.assertRaisesRegex(ValueError, 'Ollama'):
            answer_question(RulesProvider(), 'Soru nedir?', sections(), data['policies'], data['obligations'])

    def test_a_draft_clause_is_returned_as_a_draft(self):
        candidate = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]['candidate']
        draft = draft_clause(FixtureProvider(), candidate, 'Sample policy text.')
        self.assertEqual(draft['clause'], 'Fixture policy clause requiring the duty.')
        self.assertIn('Kanıt değildir', draft['note'])


class WorkspaceParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name)/'workspace', provider_factory=lambda _: FixtureProvider(), source_fetcher=source_fixture)
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

    def test_metrics_ask_draft_and_assignment_through_the_api(self):
        upload = [dict(name='policy.txt', content=base64.b64encode(b'All staff must retain records.').decode()),
                  dict(name='kontroller.csv', content=base64.b64encode('kontrol_no,ad\nK-01,All staff must retain records.\n'.encode()).decode())]
        row = self.finish(provider='ollama', policies=upload, risk_categories=['Kayıt', 'Diğer'])
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        self.assertEqual(row['metrics']['control_rows'], 1)
        self.assertEqual(row['metrics']['categories'], {'Kayıt': row['candidates']})
        run = '/api/runs/'+row['id']
        asked = self.client.post(run+'/ask', json={'question': 'Kayıtlar ne kadar saklanır?'})
        self.assertEqual(asked.status_code, 200, asked.text)
        self.assertEqual(asked.json()['answer'], 'Fixture answer.')
        self.assertTrue((self.workspace.result(row['id'])/'questions.jsonl').exists())
        self.assertEqual(self.client.post(run+'/ask', json={'question': 'ab'}).status_code, 422)
        packet = self.client.get(run+'/packet').json()
        obligation_id = packet['events'][0]['payload']['obligations'][0]['id']
        drafted = self.client.post(run+'/draft', json={'obligation_id': obligation_id})
        self.assertEqual(drafted.status_code, 200, drafted.text)
        self.assertEqual(drafted.json()['clause'], 'Fixture policy clause requiring the duty.')
        self.assertEqual(self.client.post(run+'/draft', json={'obligation_id': 'f'*64}).status_code, 422)
        assigned = self.client.post(run+'/assign', json={'obligation_id': obligation_id, 'assignee': 'Ayşe Uyum'})
        self.assertEqual(assigned.status_code, 200, assigned.text)
        self.assertEqual(assigned.json()['assignments'][obligation_id]['assignee'], 'Ayşe Uyum')
        self.assertEqual(self.workspace.metadata(row['id'])['assignments'][obligation_id]['assignee'], 'Ayşe Uyum')
        self.assertEqual(self.client.post(run+'/assign', json={'obligation_id': obligation_id, 'assignee': ''}).json()['assignments'], {})
        # The original packet is untouched by questions, drafts and assignments.
        self.assertEqual(self.client.get(run+'/packet').json()['count'], 1)
        self.assertEqual(self.client.get(run+'/verify').status_code, 200)
        page = self.client.get('/').text
        for needle in ('id="askButton"', 'id="draftButton"', 'id="assignButton"', 'name="risk_categories"', '.csv'):
            self.assertIn(needle, page)

    def test_rules_workflow_refuses_questions_with_a_clear_message(self):
        self.workspace.provider_factory = lambda _: __import__('regchain.extraction.providers', fromlist=['RulesProvider']).RulesProvider()
        row = self.finish(provider='rules')
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        asked = self.client.post('/api/runs/'+row['id']+'/ask', json={'question': 'Kayıtlar ne kadar saklanır?'})
        self.assertEqual(asked.status_code, 422)
        self.assertIn('Ollama', asked.json()['detail'])


if __name__ == '__main__':
    unittest.main()

import base64
import copy
import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from regchain.evidence import verify_chain
from regchain.extraction.providers import RulesProvider
from regchain.pilot.artifacts import save_bundle
from regchain.pilot.engine import load_packet
from regchain.pilot.evaluation import label_template
from regchain.pilot.paths import extended
from regchain.pilot.report import render
from regchain.pilot.schema import Review
from regchain.pilot.sources import save_sources
from regchain.pilot.verification import verify_artifacts
from regchain.pilot.workspace import Workspace, create_app, RunInput, Upload, filename, running_instance
from test_pilot import company, packet, review_input
from test_consolidated import card, page
from test_consolidated_postgres import handbook_download


def source_fixture(directory, **_):
    source = handbook_download(page(
        card('CONC 7.1.1', 'R', '04/11/2024', '<p>This chapter applies to consumer credit lenders.</p>'),
        card('CONC 7.3.4', 'R', '04/11/2024', '<p>A firm must retain records.</p>')))
    return save_sources(directory, [source])


def request_input(**changes):
    value = dict(company=company().model_dump(), provider='rules', source_mode='refresh',
        labels=['CONC 7.3.4'], policies=[dict(name='retention.txt',
            content=base64.b64encode(b'Staff must retain records.').decode())])
    value.update(changes)
    return value


def uncertain_review(packet):
    return dict(format='regchain-pilot-review-v1', analysis_head=packet['head'],
        reviewer='Automated test only', reviewer_role='Software QA, not an expert', decisions=[
            dict(obligation_id=o['id'], extraction='UNCERTAIN', applicability='UNKNOWN',
                 coverage='UNKNOWN', company_fact_keys=[], scope_evidence=[], policy_evidence=[],
                 rationale='Automated test; no legal judgement.')
            for o in packet['events'][0]['payload']['obligations']])


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source_fixture(self.root/'cached')
        self.workspace = Workspace(self.root/'workspace', self.root/'cached',
            provider_factory=lambda _: RulesProvider(), source_fetcher=source_fixture)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'),
            base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
            headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def complete(self, **changes):
        response = self.client.post('/api/runs', json=request_input(**changes))
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['id']
        for _ in range(300):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED', 'INTERRUPTED') and not self.workspace.active:
                self.assertEqual(row['state'], 'COMPLETED', row)
                return run_id
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def test_auth_required_for_every_private_endpoint_but_not_shell(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        response = self.client.get('/api/state', headers={'Authorization': ''})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn('runs', response.json())
        self.assertEqual(self.client.get('/openapi.json').status_code, 404)

    def test_rebinding_cross_origin_remote_client_and_spoofed_proxy_blocked(self):
        for headers in ({'Host': 'attacker.example:8767'}, {'Origin': 'https://attacker.example'},
                        {'Origin': 'null'}, {'Sec-Fetch-Site': 'cross-site'}):
            self.assertEqual(self.client.get('/api/state', headers=headers).status_code, 403)
        with TestClient(create_app(self.workspace, 'test-only-token'),
                        base_url='http://127.0.0.1:8767', client=('10.0.0.2', 1000),
                        headers={'Authorization': 'Bearer test-only-token'}) as remote:
            self.assertEqual(remote.get('/api/state', headers={'X-Forwarded-For': '127.0.0.1'}).status_code, 403)

    def test_policy_not_echoed_in_validation_error(self):
        secret = 'PRIVATE-DOCUMENT-SENTINEL'
        response = self.client.post('/api/runs', json=request_input(policies=[{'name': '../outside.txt', 'content': secret}]))
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(secret, response.text)

    def test_body_limit_and_non_json_are_rejected(self):
        self.assertEqual(self.client.post('/api/runs', content='{}').status_code, 415)
        with patch('regchain.pilot.workspace.MAX_REQUEST', 20):
            self.assertEqual(self.client.post('/api/runs', json={'oversize': 'x'*30}).status_code, 413)

    def test_filename_cannot_traverse_or_use_windows_devices(self):
        for name in ('../escape.txt', r'C:\secret.txt', 'x:stream.txt', 'CON.txt', 'aux.pdf',
                     'x.txt.', 'x.txt\n', 'x.exe', 'x/../a.pdf'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                filename(name)

    def test_duplicate_policy_bytes_or_case_insensitive_names_rejected(self):
        a = request_input()['policies'][0]
        for b in (dict(a, name='other.txt'), dict(a, name='RETENTION.TXT', content=base64.b64encode(b'Other').decode())):
            response = self.client.post('/api/runs', json=request_input(policies=[a,b]))
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.workspace.list_runs(), [])

    def test_invalid_base64_is_rejected_without_a_job(self):
        response = self.client.post('/api/runs', json=request_input(policies=[{'name': 'a.txt', 'content': '!!'}]))
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.workspace.list_runs(), [])

    def test_end_to_end_company_upload_analysis_review_export_verify(self):
        run_id = self.complete()
        original = self.workspace.packet(run_id)
        response = self.client.get(f'/api/runs/{run_id}/report')
        self.assertEqual(response.status_code, 200)
        self.assertIn('cardaman-review', response.text)
        review = uncertain_review(original)
        saved = self.client.post(f'/api/runs/{run_id}/reviews', json=review)
        self.assertEqual(saved.status_code, 200, saved.text)
        repeat = self.client.post(f'/api/runs/{run_id}/reviews', json=review)
        self.assertEqual(saved.json(), repeat.json())
        self.assertEqual(len(self.workspace.metadata(run_id)['reviews']), 1)
        self.assertEqual(self.workspace.packet(run_id), original)
        rid = saved.json()['id']
        reviewed = self.client.get(f'/api/runs/{run_id}/packet?review_id={rid}').json()
        self.assertTrue(verify_chain(reviewed['events'], saved.json()['head'], 2))
        self.assertEqual(reviewed['events'][-1]['payload']['outcomes'][0]['status'], 'EXTRACTION_REVIEW_REQUIRED')
        download = self.client.get(f'/api/runs/{run_id}/export?review_id={rid}')
        self.assertEqual(download.status_code, 200)
        output = self.root/'export'
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            self.assertIn('policy-originals/', '/'.join(archive.namelist()))
            archive.extractall(output)
        exported = load_packet(output/'packet.json')
        verify_artifacts(exported, output)
        self.assertEqual(exported['head'], saved.json()['head'])

    def test_stale_review_and_missing_evidence_cannot_be_saved(self):
        run_id = self.complete()
        value = uncertain_review(self.workspace.packet(run_id))
        stale = copy.deepcopy(value)
        stale['analysis_head'] = '0'*64
        self.assertEqual(self.client.post(f'/api/runs/{run_id}/reviews', json=stale).status_code, 422)
        value['decisions'][0].update(extraction='ACCEPT', applicability='APPLIES', coverage='COVERS_TEXT')
        self.assertEqual(self.client.post(f'/api/runs/{run_id}/reviews', json=value).status_code, 422)
        self.assertEqual(self.workspace.metadata(run_id)['reviews'], [])

    def test_corrupt_original_blocks_report_review_and_export(self):
        run_id = self.complete()
        review = uncertain_review(self.workspace.packet(run_id))
        next((self.workspace.result(run_id)/'policy-originals').iterdir()).write_bytes(b'tampered')
        for operation in ('report','export','verify'):
            self.assertEqual(self.client.get(f'/api/runs/{run_id}/{operation}').status_code, 422)
        self.assertEqual(self.client.post(f'/api/runs/{run_id}/reviews', json=review).status_code, 422)

    def test_supplied_anchor_is_checked_and_internal_check_not_external_claim(self):
        run_id = self.complete()
        url = f'/api/runs/{run_id}/verify'
        result = self.client.get(url).json()
        self.assertFalse(result['external_reference_checked'])
        self.assertEqual(self.client.get(url, params={'expected_head':'a'*64, 'expected_count':1}).status_code, 422)
        self.assertTrue(self.client.get(url, params={'expected_head':result['head'], 'expected_count':1}).json()['external_reference_checked'])

    def test_reused_source_company_change_requires_new_review(self):
        first = self.complete()
        profile = company().model_dump()
        profile['version'] = '2'
        second = self.complete(company=profile, previous_id=first, source_mode='reuse')
        p = self.workspace.packet(second)['events'][0]['payload']
        self.assertTrue(p['input_changes']['company_changed'])
        self.assertEqual(p['cases'][0]['change']['status'], 'TEXT_UNCHANGED')
        self.assertEqual(p['previous_analysis_head'], self.workspace.packet(first)['head'])
        self.assertEqual(self.workspace.metadata(second)['reviews'], [])
        profile['id'] = 'different-company'
        self.assertEqual(self.client.post('/api/runs', json=request_input(company=profile, previous_id=first)).status_code, 422)

    def test_unreviewed_expert_labels_remain_unmeasured_and_are_saved(self):
        run_id = self.complete()
        labels = self.client.get(f'/api/runs/{run_id}/labels').json()
        result = self.client.post(f'/api/runs/{run_id}/evaluate', json=labels)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['metrics']['status'], 'EXPERT_REVIEW_PENDING')
        self.assertIsNone(result.json()['metrics']['decisions']['coverage']['accuracy'])
        self.assertEqual(len(list((self.workspace.result(run_id)/'evaluations').glob('*.json'))), 1)
        with zipfile.ZipFile(io.BytesIO(self.workspace.export(run_id))) as archive:
            self.assertEqual(len([n for n in archive.namelist() if n.startswith('evaluations/')]), 1)
        self.assertEqual(self.client.post(f'/api/runs/{run_id}/evaluate', json={}).status_code, 422)

    def test_restart_recovers_committed_review_missing_from_job_index(self):
        run_id = self.complete()
        saved = self.workspace.save_review(run_id, Review.model_validate(uncertain_review(self.workspace.packet(run_id))))
        self.workspace.update(run_id, reviews=[])
        self.workspace.close()
        restarted = Workspace(self.workspace.root)
        self.addCleanup(restarted.close)
        restarted.start()
        self.assertEqual(restarted.metadata(run_id)['reviews'][0], saved)

    def test_single_worker_backpressure_and_failure_retry_keep_old_evidence(self):
        entered, release = threading.Event(), threading.Event()
        def blocked(directory, **_):
            entered.set()
            release.wait(5)
            raise ValueError('Fixture source unavailable')
        self.workspace.source_fetcher = blocked
        first = self.client.post('/api/runs', json=request_input()).json()['id']
        self.assertTrue(entered.wait(3))
        self.assertEqual(self.client.post('/api/runs', json=request_input()).status_code, 409)
        release.set()
        for _ in range(200):
            if not self.workspace.active:
                break
            time.sleep(.01)
        self.assertEqual(self.workspace.metadata(first)['state'], 'FAILED')
        self.assertFalse((self.workspace.directory(first)/'result').exists())
        self.workspace.source_fetcher = source_fixture
        response = self.client.post(f'/api/runs/{first}/retry', json={})
        self.assertEqual(response.status_code, 202)
        self.assertNotEqual(response.json()['id'], first)
        self.assertEqual(self.workspace.metadata(first)['state'], 'FAILED')

    def test_restart_marks_inflight_interrupted_and_preserves_completed(self):
        first = self.complete()
        self.workspace.update(first, state='ANALYZING')
        self.workspace.close()
        restarted = Workspace(self.workspace.root)
        restarted.start()
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.metadata(first)['state'], 'INTERRUPTED')
        self.assertTrue((restarted.directory(first)/'result/packet.json').exists())

    def test_second_process_lock_prevents_duplicate_worker(self):
        second = Workspace(self.workspace.root)
        self.addCleanup(second.close)
        with self.assertRaisesRegex(ValueError, 'başka'):
            second.start()

    def test_import_preserves_original_hash_and_deduplicates(self):
        first = self.complete()
        other = Workspace(self.root/'other')
        self.addCleanup(other.close)
        original = self.workspace.packet(first)
        other.import_bundle(self.workspace.result(first))
        other.import_bundle(self.workspace.result(first))
        self.assertEqual(len(other.list_runs()), 1)
        self.assertEqual(other.packet(other.list_runs()[0]['id']), original)

    def test_reviewed_report_is_read_only_and_shows_saved_rationale(self):
        from regchain.pilot.review import apply_review
        original = packet()
        reviewed = apply_review(original, Review.model_validate(review_input(original)))
        page_text = render(reviewed)
        self.assertIn('Synthetic fixture decision; not legal review.', page_text)
        self.assertNotIn('<select data-field=', page_text)
        self.assertIn('"read_only": true', page_text)

    def test_bad_policy_is_failed_job_not_empty_success(self):
        response = self.client.post('/api/runs', json=request_input(policies=[
            dict(name='blank.txt', content=base64.b64encode(b'   ').decode())]))
        run_id = response.json()['id']
        for _ in range(100):
            if not self.workspace.active:
                break
            time.sleep(.01)
        self.assertEqual(self.workspace.metadata(run_id)['state'], 'FAILED')
        self.assertEqual(self.client.get(f'/api/runs/{run_id}/packet').status_code, 409)

    def finished(self, workspace, run_id):
        for _ in range(300):
            row = workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED', 'INTERRUPTED') and not workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    @unittest.skipUnless(os.name == 'nt', 'MAX_PATH only limits Windows')
    def test_analysis_publishes_under_a_root_deeper_than_max_path(self):
        # Registered after the TemporaryDirectory cleanup, so it runs before it.
        self.addCleanup(shutil.rmtree, extended(self.root/('d'*110)), True)
        deep = Workspace(self.root/('d'*110)/('e'*60), self.root/'cached',
            provider_factory=lambda _: RulesProvider(), source_fetcher=source_fixture)
        self.addCleanup(deep.close)
        for mode in ('refresh', 'reuse'):
            with self.subTest(source_mode=mode):
                run_id = deep.submit(RunInput.model_validate(request_input(source_mode=mode)))['id']
                self.assertEqual(self.finished(deep, run_id)['state'], 'COMPLETED')
                result = deep.result(run_id)
                retained = next((result/'regulatory-sources').glob('*.html'))
                self.assertGreater(len(str(retained)), 260)
                verify_artifacts(deep.packet(run_id), result)
                self.assertTrue(zipfile.ZipFile(io.BytesIO(deep.export(run_id))).namelist())

    def test_file_failure_is_explained_without_paths_and_keeps_local_diagnostics(self):
        failure = shutil.Error([('C:/private/source.html', 'C:/private/target.html', 'simulated')])
        with patch('regchain.pilot.artifacts.shutil.copytree', side_effect=failure):
            run_id = self.client.post('/api/runs', json=request_input()).json()['id']
            row = self.finished(self.workspace, run_id)
        self.assertEqual(row['state'], 'FAILED')
        self.assertIn('Dosya işlemi tamamlanamadı', row['error'])
        self.assertIn('error.log', row['error'])
        self.assertNotIn('private', row['error'])
        directory = self.workspace.directory(run_id)
        self.assertIn('Traceback', (directory/'error.log').read_text(encoding='utf-8'))
        self.assertFalse((directory/'result').exists())
        partial = [p.name for p in directory.iterdir() if '.partial-' in p.name]
        self.assertEqual([len(name) for name in partial], [len('result.partial-')+8])
        self.assertEqual(self.client.get(f'/api/runs/{run_id}/packet').status_code, 409)

    def test_hybrid_search_is_refused_up_front_when_no_embedding_model_is_configured(self):
        with patch.dict(os.environ, {'EMBED_MODEL': ''}):
            response = self.client.post('/api/runs', json=request_input(retrieval='hybrid'))
        self.assertEqual(response.status_code, 422)
        self.assertIn('EMBED_MODEL', response.json()['detail'])
        self.assertEqual(self.workspace.list_runs(), [])

    def test_hybrid_search_is_recorded_in_job_packet_and_survives_verification(self):
        from test_semantic import ConceptEmbedder
        self.workspace.embedder_factory = ConceptEmbedder
        run_id = self.complete(retrieval='hybrid')
        self.assertEqual(self.workspace.metadata(run_id)['retrieval'], 'hybrid')
        payload = self.workspace.packet(run_id)['events'][0]['payload']
        self.assertEqual(payload['policy_retrieval']['method'], 'hybrid-rrf-v1')
        self.assertEqual(self.client.get(f'/api/runs/{run_id}/verify').json()['integrity'], 'VERIFIED')
        self.assertEqual(self.workspace.metadata(self.complete())['retrieval'], 'lexical')
        self.assertIn('embed_model', self.client.get('/api/state').json())

    def test_embedder_outage_fails_the_job_with_a_clear_message_not_a_lexical_result(self):
        from regchain.pilot.semantic import EmbeddingFailure
        from test_semantic import ConceptEmbedder

        class Down(ConceptEmbedder):
            def embed(self, texts):
                raise EmbeddingFailure('Ollama embedding request failed; check service, model and configuration')
        self.workspace.embedder_factory = Down
        run_id = self.client.post('/api/runs', json=request_input(retrieval='hybrid')).json()['id']
        row = self.finished(self.workspace, run_id)
        self.assertEqual(row['state'], 'FAILED')
        self.assertIn('embedding request failed', row['error'])
        self.assertFalse((self.workspace.directory(run_id)/'result').exists())

    def test_completed_job_says_when_the_model_never_answered(self):
        from regchain.extraction.providers import ContextBudgetError
        from test_pilot import FixtureProvider

        class TooLarge(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                raise ContextBudgetError('CONTEXT_BUDGET_EXCEEDED: prompt and reserved output exceed admission budget')
        self.workspace.provider_factory = lambda _: TooLarge()
        row = self.workspace.metadata(self.complete(provider='ollama'))
        self.assertEqual(row['state'], 'COMPLETED')
        self.assertEqual((row['ai_unavailable'], row['candidates']), (1, 1))
        self.assertEqual((row['passages_judged'], row['passages_unclear'], row['conflicts_flagged']), (1, 1, 0))
        self.workspace.provider_factory = lambda _: FixtureProvider()
        self.assertEqual(self.workspace.metadata(self.complete(provider='ollama'))['ai_unavailable'], 0)
        self.assertIn('ai_unavailable', self.client.get('/').text)

    def test_completed_job_counts_what_the_judge_read_and_what_it_flagged(self):
        from test_pilot import FixtureProvider

        class Forbidding(FixtureProvider):
            def passage(self, payload):
                return dict(super().passage(payload), relation='CONFLICTS')
        self.workspace.provider_factory = lambda _: Forbidding()
        row = self.workspace.metadata(self.complete(provider='ollama'))
        self.assertEqual((row['passages_judged'], row['passages_unclear'], row['conflicts_flagged']), (1, 0, 1))
        self.assertIn('conflicts_flagged', self.client.get('/').text)

    def test_held_lock_hint_names_the_running_instance(self):
        self.assertIn('terminal', running_instance(self.workspace.root))
        (self.workspace.root/'server.json').write_text(
            json.dumps({'pid': 4242, 'port': 8767, 'started_at': '2026-09-21T11:07:29+00:00'}), encoding='utf-8')
        hint = running_instance(self.workspace.root)
        self.assertIn('PID 4242', hint)
        self.assertIn('port 8767', hint)
        self.assertIn('Stop-Process -Id 4242', hint)
        (self.workspace.root/'server.json').write_text('{"pid": "rm -rf", "port": 1}', encoding='utf-8')
        self.assertNotIn('rm -rf', running_instance(self.workspace.root))

    def test_security_headers_and_script_escape(self):
        response = self.client.get('/')
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertIn("frame-ancestors 'none'", response.headers['content-security-policy'])
        self.assertNotIn('http://cdn', response.text)


if __name__ == '__main__':
    unittest.main()

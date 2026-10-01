"""v0.17: applicability trace, evidence provenance, the CONFLICT quote rule, stage timings,
document quality and the workspace's observability record."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from regchain.evidence import canonical_bytes, verify_chain
from regchain.pilot.engine import STAGES, analyze, autonomous_gate, autonomous_review
from regchain.pilot.policies import document_quality, read_policy
from regchain.pilot.report import render
from regchain.pilot.review import apply_review
from regchain.pilot.schema import PolicyCheck, Proposal, Review
from regchain.pilot.workspace import Workspace, create_app
from test_phase15 import TurkishRules, anadolu, turkish_sections
from test_pilot import FixtureProvider, company, packet, policies, review_input, sections
from test_semantic import ConceptEmbedder, chunk, policy
from test_workspace import request_input, source_fixture, uncertain_review


class TraceAndProvenanceTests(unittest.TestCase):
    def test_every_proposal_carries_its_trace_and_the_provenance_of_each_quote(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder())
        payload = packet['events'][0]['payload']
        canonical_bytes(payload)                                                   # no floats leaked into evidence
        proposal = payload['obligations'][0]['proposal']
        trace = proposal['trace']
        for key in ('parent_provision', 'child_clause', 'rule_subject', 'target_entity', 'company_entity', 'company_customer_types',
                    'matching_company_field', 'matching_regulation_text', 'positive_evidence', 'negative_evidence',
                    'explicit_exclusion_evidence', 'rule_decision', 'model_decision', 'aggregate_decision', 'disagreement'):
            self.assertIn(key, trace, key)
        self.assertEqual((trace['rule_decision'], trace['model_decision'], trace['aggregate_decision'], trace['disagreement']),
                         ('NO_RESTRICTION', 'APPLIES', 'APPLIES', False))
        self.assertEqual(trace['matching_company_field'], ['activities'])
        kinds = {row['kind'] for row in proposal['provenance']}
        self.assertEqual(kinds, {'regulation', 'policy'})
        regulation = next(row for row in proposal['provenance'] if row['kind'] == 'regulation')
        self.assertEqual((regulation['label'], regulation['passage_id'], regulation['snapshot_hash']), ('CONC 7.1.1', 's-scope', 'a' * 64))
        cited = next(row for row in proposal['provenance'] if row['kind'] == 'policy')
        self.assertEqual((cited['label'], cited['passage_id'], cited['snapshot_hash'], cited['paragraph']), ('fixture.txt', 'policy-1', 'd' * 64, 1))
        self.assertEqual(cited['retrieval_scores']['rank'], 1)
        self.assertIsNone(cited['confidence'])                                     # never invented
        self.assertIn('no calibrated confidence', cited['confidence_note'])
        self.assertEqual(proposal['review_flags'], [])

    def test_rule_and_model_disagreement_is_flagged_and_escalated(self):
        class Denying(TurkishRules):
            def scope(self, payload):
                value = super().scope(payload)
                condition = payload['scope'][0]['text'][:100]
                return {**value, 'applicability': 'DOES_NOT_APPLY', 'basis': [{'company_fact': value['basis'][0]['company_fact'],
                                                                             'regulatory_condition': condition, 'match': 'NO'}],
                        'applicability_reason': 'Fixture denies.'}
        # md. 4(1) names no counterparty, so the gate cannot MATCH there; use a clause it matches: gerçek kişi ↔ bireysel müşteriler.
        real_person = dict(turkish_sections()[1], id='tr-real', printed_label='Kanun 5549 md. 6', paragraph_number='6', ordinal=5,
                           heading_path=[turkish_sections()[1]['heading_path'][0], 'İKİNCİ BÖLÜM', 'Gerçek kişilerde kimlik tespiti'],
                           text='(1) Gerçek kişilerin kimlik tespitinde ilgilinin adı ve doğum tarihi alınır.')
        packet = analyze(anadolu(), policies(), [*turkish_sections(), real_person], Denying(), ['Kanun 5549 md. 6'])
        row = packet['events'][0]['payload']['obligations'][0]
        trace = row['proposal']['trace']
        self.assertEqual((trace['rule_decision'], trace['model_decision'], trace['disagreement']), ('NO_OBJECTION', 'DOES_NOT_APPLY', True))
        self.assertEqual(row['proposal']['review_flags'], ['RULE_MODEL_DISAGREEMENT'])
        self.assertIn('Flagged for a person', autonomous_gate(row))
        review, summary = autonomous_review(packet)
        self.assertEqual((summary['decided'], summary['escalated']), (0, 1))

    def test_a_conflict_without_its_contradiction_quote_is_refused_by_the_schema(self):
        base = dict(applicability='APPLIES', company_fact_keys=[], scope_evidence=[], applicability_reason='r', policy_evidence=[],
                    coverage_reason='r', missing_information=[])
        with self.assertRaisesRegex(ValueError, 'explicit contradiction quote'):
            Proposal(**base, coverage='CONFLICT', policy_checks=[PolicyCheck(source_id='p', quote='x', relation='SUPPORTS')])
        Proposal(**base, coverage='CONFLICT', policy_checks=[PolicyCheck(source_id='p', quote='Records are deleted nightly.', relation='CONFLICTS')])


class TimingTests(unittest.TestCase):
    def test_the_packet_records_stage_timings_in_milliseconds_and_rows_their_elapsed_time(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder())
        payload = packet['events'][0]['payload']
        timings = payload['timings']
        for stage in STAGES:
            self.assertIn(stage, timings, stage)
            self.assertIsInstance(timings[stage], int)
        self.assertIn('embedding_cache_hits', timings)
        self.assertIn('elapsed_ms', payload['obligations'][0])
        self.assertIsInstance(payload['obligations'][0]['elapsed_ms'], int)

    def test_stage_timing_accumulates_the_time_spent_in_each_stage(self):
        class Slow(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'contradicts' in schema['properties']:
                    time.sleep(0.05)
                return super()._chat(prompt, payload, schema)
        timings = analyze(company(), policies(), sections(), Slow(), ['CONC 7.3.4'])['events'][0]['payload']['timings']
        self.assertGreaterEqual(timings['conflict'], 40)                            # the sleep landed in the conflict stage
        self.assertGreaterEqual(timings['applicability'], 0)


class DocumentQualityTests(unittest.TestCase):
    def test_pdf_quality_is_classified_and_a_scanned_document_requires_ocr(self):
        parts = lambda texts: [(i + 1, t, 'pdf_page') for i, t in enumerate(texts)]
        empties = lambda texts: [i + 1 for i, t in enumerate(texts) if not t.strip()]
        long = 'Kimlik tespiti işlem yapılmadan önce tamamlanır. ' * 10
        for texts, kind in [([long] * 4, 'text'), ([''] + [long] * 4, 'mixed'), (['', '', long], 'scanned'), (['kısa metin'] * 3, 'low_density')]:
            with self.subTest(kind=kind):
                quality = document_quality(parts(texts), empties(texts))
                self.assertEqual(quality['kind'], kind)
                self.assertEqual(quality['requires_ocr'], kind == 'scanned')
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'doc.pdf'
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(300, 300)
        writer.write(path)
        with patch('pypdf._page.PageObject.extract_text', side_effect=['', long, long]):
            value = read_policy(path)
        self.assertEqual((value['document_quality']['kind'], value['document_quality']['empty_pages'], value['document_quality']['requires_ocr']), ('mixed', 1, False))
        with patch('pypdf._page.PageObject.extract_text', side_effect=['', '', long]):
            with self.assertRaisesRegex(ValueError, 'requires_ocr'):
                read_policy(path)
        text = Path(temp.name) / 'policy.txt'
        text.write_text('All staff must retain records.\n', encoding='utf-8')
        self.assertEqual(read_policy(text)['document_quality']['kind'], 'text')


class WorkspaceObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name) / 'workspace', provider_factory=lambda _: FixtureProvider(), source_fetcher=source_fixture)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                                 headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_the_job_carries_stage_timings_calls_by_stage_document_quality_and_the_slowest_duties(self):
        run_id = self.client.post('/api/runs', json=request_input(provider='ollama')).json()['id']
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                break
            time.sleep(.01)
        self.assertEqual(row['state'], 'COMPLETED', row.get('error'))
        observability = row['observability']
        for key in ('download_ms', 'parse_ms', 'obligation_extraction', 'applicability', 'coverage', 'conflict', 'proposal_generation'):
            self.assertIn(key, observability['stages_ms'], key)
        self.assertEqual(observability['slowest_obligations'][0]['label'], 'CONC 7.3.4')
        self.assertIn('calls_by_stage', observability)
        self.assertEqual(row['policy_quality'][0]['name'], 'retention.txt')
        self.assertEqual(row['policy_quality'][0]['kind'], 'text')
        html = self.client.get('/').text
        self.assertIn('Gözlemlenebilirlik (debug)', html)
        self.assertIn('Belge kalitesi', html)


class PlatformWiringTests(unittest.TestCase):
    """v0.17 Phase 11: request ids, audit chain, RBAC, redacted model call logs, retention, export and delete
    on the single-operator workspace."""

    def make(self, **environ):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with patch.dict(os.environ, environ, clear=False):
            for key in ('OPERATOR_ROLES', 'RETENTION_DAYS', 'AI_LOG_REDACTION', 'TENANT_ID', 'OPERATOR_NAME'):
                if key not in environ:
                    os.environ.pop(key, None)
            workspace = Workspace(Path(temp.name) / 'workspace', provider_factory=lambda _: FixtureProvider(), source_fetcher=source_fixture)
        client = TestClient(create_app(workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                            headers={'Authorization': 'Bearer test-only-token'})
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return workspace, client

    def completed(self, workspace, client):
        run_id = client.post('/api/runs', json=request_input(provider='ollama')).json()['id']
        for _ in range(400):
            row = workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not workspace.active:
                break
            time.sleep(.01)
        self.assertEqual(row['state'], 'COMPLETED', row.get('error'))
        return run_id

    def test_every_response_names_its_request_and_a_well_formed_client_id_is_kept(self):
        workspace, client = self.make(TENANT_ID='acme', OPERATOR_NAME='ayse')
        first = client.get('/api/state')
        self.assertRegex(first.headers['X-Request-ID'], r'^[0-9a-f]{32}$')
        self.assertEqual(client.get('/api/state', headers={'X-Request-ID': 'gw-2026.09.23_abc-123'}).headers['X-Request-ID'], 'gw-2026.09.23_abc-123')
        self.assertRegex(client.get('/api/state', headers={'X-Request-ID': 'has space'}).headers['X-Request-ID'], r'^[0-9a-f]{32}$')
        # A refused request carries an id too.
        self.assertIn('X-Request-ID', client.get('/api/state', headers={'Authorization': 'Bearer wrong'}).headers)
        self.assertEqual((workspace.tenant_id, workspace.operator, workspace.roles), ('acme', 'ayse', frozenset({'ADMIN'})))

    def test_actions_are_audited_on_a_verifiable_chain_and_logs_leave_redacted(self):
        workspace, client = self.make(AI_LOG_REDACTION='standard')
        run_id = self.completed(workspace, client)
        packet = workspace.packet(run_id)
        review = uncertain_review(packet)
        review['decisions'][0]['action'] = 'NEEDS_EVIDENCE'
        saved = client.post(f'/api/runs/{run_id}/reviews', json=review, headers={'X-Request-ID': 'review-request-1'})
        self.assertEqual(saved.status_code, 200, saved.text)
        # The fixture provider keeps no call log; a model run writes one like this (hashes and numbers, no text).
        with (workspace.directory(run_id) / 'ai-calls.jsonl').open('w', encoding='utf-8') as handle:
            for stage in ('extraction', 'applicability'):
                handle.write(json.dumps({'at': '2026-09-23T10:00:00+00:00', 'task': stage, 'stage': stage, 'provider': 'ollama', 'model': 'qwen3:8b',
                                         'model_version': 'qwen3:8b@abc', 'thinking': True, 'num_predict': 6144, 'request_bytes': 1200,
                                         'prompt_sha256': 'a' * 64, 'payload_sha256': 'b' * 64, 'schema_sha256': 'c' * 64,
                                         'provision_id': 'd' * 64, 'obligation_id': 'e' * 64, 'evidence_ids': ['f' * 64],
                                         'cache_hit': False, 'retry_count': 0, 'status': 'ok', 'elapsed_ms': 1200,
                                         'prompt_tokens': 900, 'output_tokens': 120, 'error': None}) + '\n')
        export = client.get(f'/api/runs/{run_id}/export')
        self.assertEqual(export.status_code, 200)
        import io
        import zipfile
        names = zipfile.ZipFile(io.BytesIO(export.content)).namelist()
        self.assertIn('ai-calls.redacted.jsonl', names)
        redacted = [json.loads(line) for line in zipfile.ZipFile(io.BytesIO(export.content)).read('ai-calls.redacted.jsonl').decode('utf-8').splitlines()]
        self.assertTrue(redacted)
        for entry in redacted:
            self.assertNotIn('thinking', entry)                                    # standard drops tuning internals
            self.assertNotIn('prompt', entry)
            self.assertIn('prompt_sha256', entry)
        audit = client.get('/api/audit').json()
        self.assertTrue(audit['chain_valid'])
        actions = [(e['action'], e['outcome']) for e in audit['entries']]
        self.assertIn(('start_analysis', 'OK'), actions)
        self.assertIn(('save_review', 'OK'), actions)
        self.assertIn(('export_data', 'OK'), actions)
        review_entry = next(e for e in audit['entries'] if e['action'] == 'save_review')
        self.assertEqual((review_entry['request_id'], review_entry['actor'], review_entry['tenant_id']), ('review-request-1', 'local-operator', 'local'))
        self.assertIn('NEEDS_EVIDENCE=1', review_entry['detail'])
        self.assertTrue(workspace.audit_log.verify())
        # Model call records: the configured level or stricter, never weaker; never a prompt.
        calls = client.get(f'/api/runs/{run_id}/ai-calls').json()
        self.assertEqual(calls['level'], 'standard')
        self.assertTrue(calls['calls'])
        self.assertEqual(client.get(f'/api/runs/{run_id}/ai-calls', params={'level': 'minimal'}).status_code, 422)
        strict = client.get(f'/api/runs/{run_id}/ai-calls', params={'level': 'strict'}).json()['calls']
        self.assertTrue(all('provision_id' not in entry and 'payload_sha256' not in entry for entry in strict))
        # The job record names the deterministic request id of the submission.
        self.assertIsNone(workspace.retention.days)
        self.assertEqual(client.get('/api/retention').json()['expired'], [])

    def test_delete_and_retention_sweep_are_admin_actions_and_a_viewer_is_refused_and_audited(self):
        workspace, client = self.make(RETENTION_DAYS='1')
        run_id = self.completed(workspace, client)
        # Nothing is old enough yet; backdate the job and the sweep finds it.
        self.assertEqual(client.get('/api/retention').json(), {'days': 1, 'keep_reviewed': True, 'redaction_level': 'standard', 'expired': []})
        workspace.update(run_id, created_at='2026-01-01T00:00:00+00:00')
        self.assertEqual(client.get('/api/retention').json()['expired'], [run_id])
        swept = client.post('/api/retention/sweep', json={}).json()
        self.assertEqual(swept['deleted'], [{'id': run_id, 'outcome': 'OK'}])
        self.assertFalse((workspace.runs / run_id).exists())
        entries = workspace.audit_log.entries()
        self.assertIn(('delete_run', 'OK'), [(e['action'], e['outcome']) for e in entries])
        self.assertTrue(workspace.audit_log.verify())
        # A viewer may look but not delete or export; the refusal is on the chain.
        viewer_workspace, viewer = self.make(OPERATOR_ROLES='VIEWER')
        other = self.completed(*self.make())                                       # a run in another workspace, just for an id
        self.assertEqual(viewer.get('/api/state').status_code, 200)
        denied = viewer.delete(f'/api/runs/{other}')
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(viewer.get('/api/tenant/export').status_code, 403)
        self.assertEqual(viewer.post('/api/runs', json=request_input()).status_code, 403)
        outcomes = [(e['action'], e['outcome']) for e in viewer_workspace.audit_log.entries()]
        self.assertIn(('delete_data', 'DENIED'), outcomes)
        self.assertIn(('start_analysis', 'DENIED'), outcomes)

    def test_the_tenant_export_holds_every_completed_run_with_a_manifest(self):
        workspace, client = self.make()
        run_id = self.completed(workspace, client)
        response = client.get('/api/tenant/export')
        self.assertEqual(response.status_code, 200)
        import io
        import zipfile
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        self.assertIn('manifest.json', archive.namelist())
        self.assertTrue(any(name.startswith(run_id + '/') for name in archive.namelist()))
        self.assertFalse(list((workspace.root / 'exports').iterdir()))                # nothing left behind
        self.assertEqual(client.delete(f'/api/runs/{run_id}').json()['deleted'], [{'id': run_id, 'outcome': 'OK'}])
        self.assertEqual(client.get('/api/tenant/export').status_code, 404)


class ChangeImpactTests(unittest.TestCase):
    def test_change_kinds_are_read_from_the_wording(self):
        from regchain.pilot.impact import change_kinds, suggested_action
        self.assertEqual(change_kinds('Aynı metin.', 'Aynı  metin.'), [])
        self.assertIn('THRESHOLD_CHANGED', change_kinds('İşlem tutarı 75.000 TL üzerinde ise kimlik tespiti yapılır.',
                                                        'İşlem tutarı 185.000 TL üzerinde ise kimlik tespiti yapılır.'))
        self.assertIn('DEADLINE_CHANGED', change_kinds('Bildirim on gün içinde yapılır.', 'Bildirim yirmi gün içinde yapılır.'))
        self.assertIn('ENTITY_SCOPE_CHANGED', change_kinds('Müşteri gerçek kişi ise kimlik tespiti yapılır.',
                                                           'Müşteri gerçek kişi veya dernek ise kimlik tespiti yapılır.'))
        self.assertIn('EXEMPTION_CHANGED', change_kinds('Bu madde bütün işlemlere uygulanır.',
                                                        'Bu madde bütün işlemlere uygulanır; kamu kurumlarına uygulanmaz.'))
        self.assertEqual(change_kinds('A firm must retain records.', 'A firm must keep records.'), ['TEXT_CHANGED'])
        self.assertIn('Eşik', suggested_action(['THRESHOLD_CHANGED']))
        self.assertIn('Süre', suggested_action(['DEADLINE_CHANGED', 'TEXT_CHANGED']))

    def test_unchanged_provisions_are_carried_forward_without_model_calls_and_changed_ones_are_reread(self):
        class Counting(FixtureProvider):
            calls = 0

            def _chat(self, prompt, payload, schema):
                Counting.calls += 1
                return super()._chat(prompt, payload, schema)
        first = analyze(company(), policies(), sections(), Counting(), ['CONC 7.3.4'])
        old = first['events'][0]['payload']
        previous_sources = [c['source'] for c in old['cases']]
        # A new snapshot of the same text: every section id changes, nothing else does.
        renewed = [dict(s, id='n-' + s['id'], version_id='n-' + s['version_id']) for s in sections()]
        Counting.calls = 0
        second = analyze(company(), policies(), renewed, Counting(), ['CONC 7.3.4'], previous_sources, first['head'], old)
        payload = second['events'][0]['payload']
        self.assertEqual(Counting.calls, 0)
        row = payload['obligations'][0]
        self.assertEqual(row['carried_forward']['from_head'], first['head'])
        self.assertEqual(row['source_id'], 'n-s-duty')
        self.assertEqual(row['proposal']['scope_evidence'][0]['source_id'], 'n-s-scope')   # quotes follow the new ids
        self.assertEqual((payload['regulation']['carried_forward'], payload['regulation']['reanalysed']), (1, 0))
        self.assertTrue(payload['impact']['reused_unchanged'])
        self.assertFalse(payload['impact']['affects_company'])
        self.assertEqual(payload['cases'][0]['change']['status'], 'TEXT_UNCHANGED')
        self.assertTrue(verify_chain(second['events'], second['head'], second['count']))
        canonical_bytes(payload)
        # A carried row is still reviewable: its quotes resolve against the new snapshot.
        review = review_input(second)
        review['decisions'][0]['scope_evidence'][0]['source_id'] = 'n-s-scope'
        self.assertEqual(apply_review(second, Review.model_validate(review))['count'], 2)
        self.assertIn('önceki analizden taşındı', render(second))
        # The duty's text changes: it is read again and the impact names the kind of change.
        changed = [dict(s, text='A firm must retain records for 6 years.') if s['printed_label'] == 'CONC 7.3.4' else s for s in renewed]
        third = analyze(company(), policies(), changed, Counting(), ['CONC 7.3.4'], previous_sources, first['head'], old)
        impact = third['events'][0]['payload']['impact']
        self.assertGreater(Counting.calls, 0)
        self.assertNotIn('carried_forward', third['events'][0]['payload']['obligations'][0])
        self.assertTrue(impact['affects_company'])
        self.assertEqual((impact['changed'][0]['label'], impact['changed'][0]['affected_policies']), ('CONC 7.3.4', ['fixture.txt']))
        self.assertIn('DEADLINE_CHANGED', impact['changed'][0]['kinds'])
        self.assertTrue(impact['changed'][0]['suggested_action'])
        self.assertIn('Bu regülasyon güncellemesi şirketinizi etkileyebilir', render(third))
        # A different company profile blocks reuse: everything is read again, and the record says why.
        other = company()
        other.version = '2'
        Counting.calls = 0
        fourth = analyze(other, policies(), renewed, Counting(), ['CONC 7.3.4'], previous_sources, first['head'], old)
        self.assertGreater(Counting.calls, 0)
        self.assertEqual(fourth['events'][0]['payload']['impact']['reuse_blocked_reason'], 'company profile changed')
        self.assertEqual(fourth['events'][0]['payload']['impact']['carried_forward'], 0)

    def test_a_changed_scope_article_blocks_reuse_even_when_the_duty_text_is_the_same(self):
        first = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        old = first['events'][0]['payload']
        widened = [dict(s, text='This chapter applies to consumer credit lending and hiring.') if s['id'] == 's-scope' else s for s in sections()]
        second = analyze(company(), policies(), widened, FixtureProvider(), ['CONC 7.3.4'], [c['source'] for c in old['cases']], first['head'], old)
        impact = second['events'][0]['payload']['impact']
        self.assertEqual(impact['reuse_blocked_reason'], 'scope provisions changed')
        self.assertNotIn('carried_forward', second['events'][0]['payload']['obligations'][0])


class ReviewUXTests(unittest.TestCase):
    def test_override_needs_a_reason_approve_keeps_the_proposal_and_ai_output_is_never_erased(self):
        value = packet()
        review = review_input(value)
        review['decisions'][0]['action'] = 'OVERRIDE'
        with self.assertRaisesRegex(ValueError, 'override_reason'):
            Review.model_validate(review)
        review['decisions'][0]['override_reason'] = 'Policy metni saklama süresini belirtmiyor.'
        review['decisions'][0]['coverage'] = 'PARTIAL'
        reviewed = apply_review(value, Review.model_validate(review))
        entry = reviewed['events'][-1]['payload']['audit'][0]
        self.assertEqual((entry['action'], entry['changed_fields'], entry['override_reason']),
                         ('OVERRIDE', ['coverage'], 'Policy metni saklama süresini belirtmiyor.'))
        self.assertEqual(entry['ai_proposal'], {'applicability': 'APPLIES', 'coverage': 'COVERS_TEXT'})
        self.assertEqual((entry['reviewer'], entry['reviewer_role']), ('Fixture reviewer', 'Synthetic test actor'))
        self.assertEqual(reviewed['events'][0], value['events'][0])                     # the analysis event is untouched
        self.assertTrue(verify_chain(reviewed['events'], reviewed['head'], 2))
        html = render(reviewed)
        self.assertIn('Denetim kaydı', html)
        self.assertIn('override gerekçesi', html)
        # APPROVE that silently changes the AI's verdict is refused; the reviewer must override with a reason.
        approve = review_input(value)
        approve['decisions'][0].update(action='APPROVE', applicability='DOES_NOT_APPLY')
        with self.assertRaisesRegex(ValueError, 'OVERRIDE'):
            apply_review(value, Review.model_validate(approve))
        approve['decisions'][0]['applicability'] = 'APPLIES'
        self.assertEqual(apply_review(value, Review.model_validate(approve))['events'][-1]['payload']['outcomes'][0]['status'], 'REVIEWED_POLICY_COVERAGE')
        needs = review_input(value)
        needs['decisions'][0]['action'] = 'NEEDS_EVIDENCE'
        self.assertEqual(apply_review(value, Review.model_validate(needs))['events'][-1]['payload']['outcomes'][0]['status'], 'MORE_EVIDENCE_REQUIRED')
        # A review written before v0.17 carries no action and still validates.
        self.assertIsNone(Review.model_validate(review_input(value)).decisions[0].action)
        with self.assertRaisesRegex(ValueError, 'only recorded with an OVERRIDE'):
            Review.model_validate({**review_input(value), 'decisions': [{**review_input(value)['decisions'][0], 'action': 'REJECT', 'override_reason': 'x'}]})

    def test_the_review_screen_shows_three_columns_and_the_four_actions(self):
        html = render(packet())
        for text in ('class="tri"', '<h4>Regülasyon</h4>', '<h4>Policy kanıtı</h4>', '<h4>AI önerisi ve karar</h4>', 'data-field="action"',
                     'value="APPROVE"', 'value="REJECT"', 'value="OVERRIDE"', 'value="NEEDS_EVIDENCE"', 'data-field="override_reason"',
                     '"proposals":'):
            self.assertIn(text, html, text)


if __name__ == '__main__':
    unittest.main()

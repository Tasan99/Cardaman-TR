"""v0.18 regression: the v0.17 platform baseline, end to end through the workspace app.

Request ids on every kind of response, the audit chain across a session and its tamper
detection, model call records at every redaction level, the tenant export (manifest, and call
logs redacted like the single-run export), audited deletion, the retention sweep, and normal
logs that never carry a policy passage, a company description or a secret. Fixtures only: no
model, no network.
"""
import base64
import contextlib
import io
import json
import logging
import os
import tempfile
import time
import traceback
import unittest
import zipfile
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from regchain.evidence import GENESIS, digest
from regchain.extraction.providers import AI_CONTEXT, AI_TASK
from regchain.platform import retention
from regchain.platform.audit import AuditLog
from regchain.platform.redaction import ALLOWED_CALL_FIELDS, FORBIDDEN_CALL_FIELDS, RedactionPolicy
from regchain.pilot.workspace import Workspace, create_app
from test_pilot import FixtureProvider, company
from test_platform import call_entry
from test_workspace import request_input, source_fixture, uncertain_review

TOKEN = 'test-only-token'
# Marker text that exists only in the inputs; a log line or an export that carries it has leaked.
PASSAGE = 'Staff must retain records. Customer files stay in the Leeds archive for seven years under dual control.'
DESCRIPTION = 'Synthetic lender whose back office sits above the bakery on Mill Lane.'
PROMPT_MARK = 'PROMPT-TEXT-MARK'
SECRET = 'sk-canary-5f2d9c81e7a4'
DB_PASSWORD = 'Canary-Pass-7731'
SECRETS = {'LLM_API_KEY': SECRET, 'DATABASE_URL': f'postgresql://app:{DB_PASSWORD}@db.internal:5432/regchain'}
LEAKS = (PASSAGE, 'Leeds archive', DESCRIPTION, 'Mill Lane', PROMPT_MARK, SECRET, DB_PASSWORD, TOKEN)
HEX32 = r'^[0-9a-f]{32}$'


class Logging(FixtureProvider):
    """The fixture judge with a call log shaped like OllamaProvider's: hashes, numbers, links, no text."""
    planted = False

    def __init__(self):
        self.call_log = []

    def _chat(self, prompt, payload, schema):
        answer = super()._chat(prompt, payload, schema)
        about = AI_CONTEXT.get() or {}
        entry = {'at': '2026-09-24T10:00:00+00:00', 'task': AI_TASK.get(), 'stage': AI_TASK.get(), 'provider': 'ollama',
                 'model': 'qwen3:8b', 'model_version': 'qwen3:8b@abc', 'thinking': True, 'num_predict': 6144,
                 'request_bytes': len(json.dumps(payload)), 'prompt_sha256': sha256(prompt.encode()).hexdigest(),
                 'payload_sha256': 'b' * 64, 'schema_sha256': 'c' * 64, 'provision_id': about.get('provision_id'),
                 'obligation_id': about.get('obligation_id'), 'evidence_ids': list(about.get('evidence_ids') or []),
                 'cache_hit': False, 'retry_count': 0, 'status': 'OK', 'elapsed_ms': 5, 'prompt_tokens': 10,
                 'output_tokens': 2, 'done_reason': 'stop', 'error': None}
        if self.planted:
            entry.update(prompt=PROMPT_MARK + ' ' + prompt[:80], payload=payload, response=answer, text=PASSAGE,
                         policy_excerpt=PASSAGE, error=f'HTTP 503: provider said {SECRET}')
        self.call_log.append(entry)
        return answer


class Planted(Logging):
    """A provider that wrongly records text, the prompt and a secret: what redaction must stop at the door."""
    planted = True


def canary_input(**changes):
    value = request_input(provider='ollama', company=dict(company().model_dump(), description=DESCRIPTION),
                          policies=[dict(name='retention.txt', content=base64.b64encode(PASSAGE.encode()).decode())])
    value.update(changes)
    return value


class Harness(unittest.TestCase):
    provider = Logging

    def make(self, **environ):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        provider = self.provider
        with patch.dict(os.environ, environ, clear=False):
            for key in ('OPERATOR_ROLES', 'RETENTION_DAYS', 'RETENTION_KEEP_REVIEWED', 'AI_LOG_REDACTION', 'TENANT_ID', 'OPERATOR_NAME'):
                if key not in environ:
                    os.environ.pop(key, None)
            workspace = Workspace(Path(temp.name) / 'workspace', provider_factory=lambda _: provider(), source_fetcher=source_fixture)
        client = TestClient(create_app(workspace, TOKEN), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                            headers={'Authorization': 'Bearer ' + TOKEN})
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return workspace, client

    def finished(self, workspace, client, **changes):
        response = client.post('/api/runs', json=canary_input(**changes))
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['id']
        for _ in range(600):
            row = workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED', 'INTERRUPTED') and not workspace.active:
                return row
            time.sleep(.01)
        self.fail('Fixture analysis did not finish')

    def completed(self, workspace, client, **changes):
        row = self.finished(workspace, client, **changes)
        self.assertEqual(row['state'], 'COMPLETED', row.get('error'))
        return row['id']

    def last_audit(self, workspace):
        return workspace.audit_log.entries()[-1]

    def assertRedacted(self, entries, level):
        """Call records at ``level``: allowlisted fields only, no text, the planted secret masked."""
        self.assertTrue(entries)
        for entry in entries:
            self.assertLessEqual(set(entry), ALLOWED_CALL_FIELDS[level])
            self.assertTrue(FORBIDDEN_CALL_FIELDS.isdisjoint(entry))
            self.assertNotIn('policy_excerpt', entry)
        text = json.dumps(entries, ensure_ascii=False)
        for leaked in (PASSAGE, 'Leeds archive', PROMPT_MARK, SECRET):
            self.assertNotIn(leaked, text)
        self.assertIn('[redacted LLM_API_KEY]', text)


class RequestIdTests(Harness):
    def test_success_4xx_and_403_responses_all_name_their_request(self):
        workspace, client = self.make()
        missing = '0' * 32
        cases = [
            ('success', 200, lambda h: client.get('/api/state', headers=h)),
            ('not found', 404, lambda h: client.get(f'/api/runs/{missing}', headers=h)),
            ('invalid body', 422, lambda h: client.post('/api/runs', json={'company': {}}, headers=h)),
            ('invalid level', 422, lambda h: client.get(f'/api/runs/{missing}/ai-calls', params={'level': 'public'}, headers=h)),
            ('no session token', 401, lambda h: client.get('/api/state', headers={**h, 'Authorization': 'Bearer wrong'})),
            ('not JSON', 415, lambda h: client.post('/api/runs', content=b'x', headers={**h, 'Content-Type': 'text/plain'})),
            ('foreign host', 403, lambda h: client.get('/api/state', headers={**h, 'Host': 'evil.example:8767'})),
            ('cross-site', 403, lambda h: client.get('/api/state', headers={**h, 'Sec-Fetch-Site': 'cross-site'})),
        ]
        malformed = ('has space', 'short', 'x' * 65, 'quote"inside"x', 'çok-güzel-kimlik'.encode('utf-8'))
        for name, status, send in cases:
            with self.subTest(name):
                kept = send({'X-Request-ID': 'client-req-0001'})
                self.assertEqual(kept.status_code, status, kept.text)
                self.assertEqual(kept.headers['X-Request-ID'], 'client-req-0001')
                self.assertRegex(send({}).headers['X-Request-ID'], HEX32)
                for bad in malformed:
                    replaced = send({'X-Request-ID': bad})
                    self.assertEqual(replaced.status_code, status)
                    self.assertRegex(replaced.headers['X-Request-ID'], HEX32)

    def test_a_permission_denial_names_its_request_and_the_audit_line_carries_the_same_id(self):
        workspace, client = self.make()
        run_id = self.completed(workspace, client)
        workspace.roles = frozenset({'VIEWER'})
        denied = client.delete(f'/api/runs/{run_id}', headers={'X-Request-ID': 'viewer-delete-01'})
        self.assertEqual(denied.status_code, 403)
        self.assertEqual(denied.headers['X-Request-ID'], 'viewer-delete-01')
        self.assertEqual(self.last_audit(workspace)['request_id'], 'viewer-delete-01')
        # A malformed id is replaced, and the replacement is what the audit line records.
        denied = client.get('/api/tenant/export', headers={'X-Request-ID': 'bad id'})
        self.assertEqual(denied.status_code, 403)
        self.assertRegex(denied.headers['X-Request-ID'], HEX32)
        self.assertEqual((self.last_audit(workspace)['action'], self.last_audit(workspace)['request_id']), ('export_data', denied.headers['X-Request-ID']))


class AuditChainTests(Harness):
    def test_a_session_of_actions_is_one_verifiable_chain_and_each_line_names_its_request(self):
        workspace, client = self.make(TENANT_ID='acme', OPERATOR_NAME='ayse')
        first, second = self.completed(workspace, client), self.completed(workspace, client)
        review = uncertain_review(workspace.packet(first))
        review['decisions'][0]['action'] = 'NEEDS_EVIDENCE'
        steps = [
            ('save_review', 'OK', 200, lambda h: client.post(f'/api/runs/{first}/reviews', json=review, headers=h)),
            ('export_data', 'OK', 200, lambda h: client.get(f'/api/runs/{first}/export', headers=h)),
            ('export_data', 'OK', 200, lambda h: client.get('/api/tenant/export', headers=h)),
            ('start_analysis', 'REFUSED', 404, lambda h: client.post('/api/runs', json=canary_input(previous_id='0' * 32), headers=h)),
            ('delete_run', 'REFUSED', 422, lambda h: client.delete('/api/runs/not-a-run-id', headers=h)),
            ('delete_run', 'OK', 200, lambda h: client.delete(f'/api/runs/{second}', headers=h)),
        ]
        for number, (action, outcome, status, send) in enumerate(steps):
            with self.subTest(action=action, outcome=outcome):
                request_id = f'audit-step-{number:04d}'
                self.assertEqual(send({'X-Request-ID': request_id}).status_code, status)
                entry = self.last_audit(workspace)
                self.assertEqual((entry['action'], entry['outcome'], entry['request_id'], entry['actor'], entry['tenant_id']),
                                 (action, outcome, request_id, 'ayse', 'acme'))
        workspace.roles = frozenset({'VIEWER'})
        self.assertEqual(client.delete(f'/api/runs/{first}', headers={'X-Request-ID': 'audit-step-denied'}).status_code, 403)
        self.assertEqual((self.last_audit(workspace)['action'], self.last_audit(workspace)['outcome']), ('delete_data', 'DENIED'))
        audit = client.get('/api/audit').json()
        self.assertTrue(audit['chain_valid'])
        entries = audit['entries']
        self.assertEqual([e['action'] for e in entries[:2]], ['start_analysis', 'start_analysis'])
        self.assertEqual(len(entries), 2 + len(steps) + 1)
        self.assertEqual(entries[0]['previous_hash'], GENESIS)
        for earlier, later in zip(entries, entries[1:]):
            self.assertEqual(later['previous_hash'], earlier['entry_hash'])
        # The chain holds identifiers and counts, never document text.
        text = (workspace.root / 'audit.jsonl').read_text(encoding='utf-8')
        for leaked in LEAKS:
            self.assertNotIn(leaked, text)

    def test_tampering_with_the_audit_file_is_detected_by_verify_and_by_the_endpoint(self):
        workspace, client = self.make()
        run_id = self.completed(workspace, client)
        client.get(f'/api/runs/{run_id}/export')
        client.get('/api/tenant/export')
        path = workspace.root / 'audit.jsonl'
        original = path.read_text(encoding='utf-8')
        lines = original.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(client.get('/api/audit').json()['chain_valid'])
        edited = dict(json.loads(lines[1]), detail='nothing was exported')
        # Rehashing the edited line is not enough: the next line still names the old hash.
        forged = dict(json.loads(lines[1]), actor='someone-else')
        forged['entry_hash'] = digest({key: value for key, value in forged.items() if key != 'entry_hash'})
        variants = {'edited': [lines[0], json.dumps(edited, ensure_ascii=False), lines[2]],
                    'rehashed': [lines[0], json.dumps(forged, ensure_ascii=False), lines[2]],
                    'removed': [lines[0], lines[2]], 'reordered': [lines[1], lines[0], lines[2]],
                    'extra field': [lines[0], json.dumps(dict(json.loads(lines[1]), note='x'), ensure_ascii=False), lines[2]]}
        for name, rows in variants.items():
            with self.subTest(name):
                path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
                self.assertFalse(AuditLog(path).verify())
                self.assertFalse(workspace.audit_log.verify())
                self.assertFalse(client.get('/api/audit').json()['chain_valid'])
        path.write_text(original + 'not json\n', encoding='utf-8')
        self.assertFalse(workspace.audit_log.verify())
        self.assertIn('Denetim kaydı bozuk', client.get('/api/audit').json()['detail'])
        path.write_text(original, encoding='utf-8')
        self.assertTrue(workspace.audit_log.verify())
        client.get(f'/api/runs/{run_id}/export')                                    # appending to the restored file keeps it whole
        self.assertTrue(client.get('/api/audit').json()['chain_valid'])


class AiCallLogTests(Harness):
    provider = Planted

    def test_every_level_drops_text_bearing_fields_and_a_level_weaker_than_configured_is_refused(self):
        self.enterContext(patch.dict(os.environ, SECRETS))
        workspace, client = self.make(AI_LOG_REDACTION='strict')
        self.assertEqual(workspace.redaction.level, 'strict')
        run_id = self.completed(workspace, client)
        raw = (workspace.directory(run_id) / 'ai-calls.jsonl').read_text(encoding='utf-8')
        self.assertIn(PROMPT_MARK, raw)                                             # the planted record really holds text
        levels = list(ALLOWED_CALL_FIELDS)
        for configured in levels:
            workspace.redaction = RedactionPolicy(configured)
            for requested in (None, *levels):
                with self.subTest(configured=configured, requested=requested):
                    response = client.get(f'/api/runs/{run_id}/ai-calls', params={'level': requested} if requested else {})
                    if requested and levels.index(requested) < levels.index(configured):
                        self.assertEqual(response.status_code, 422)
                        self.assertIn(configured, response.json()['detail'])
                        continue
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()['level'], requested or configured)
                    self.assertRedacted(response.json()['calls'], requested or configured)
        self.assertEqual(client.get(f'/api/runs/{run_id}/ai-calls', params={'level': 'public'}).status_code, 422)
        for role in ('VIEWER', 'REVIEWER'):
            workspace.roles = frozenset({role})
            self.assertEqual(client.get(f'/api/runs/{run_id}/ai-calls').status_code, 403)
            self.assertEqual((self.last_audit(workspace)['action'], self.last_audit(workspace)['outcome']), ('read_ai_logs', 'DENIED'))


class TenantExportTests(Harness):
    provider = Planted

    def test_the_tenant_export_has_a_manifest_and_its_call_logs_are_redacted_like_the_single_run_export(self):
        self.enterContext(patch.dict(os.environ, SECRETS))
        workspace, client = self.make(AI_LOG_REDACTION='strict', TENANT_ID='acme')
        first, second = self.completed(workspace, client), self.completed(workspace, client)
        self.assertEqual(client.post(f'/api/runs/{first}/reviews', json=uncertain_review(workspace.packet(first))).status_code, 200)
        response = client.get('/api/tenant/export')
        self.assertEqual(response.status_code, 200)
        self.assertIn('cardaman-acme-export.zip', response.headers['Content-Disposition'])
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        manifest = json.loads(archive.read('manifest.json'))
        self.assertEqual(set(archive.namelist()), set(manifest['files']) | {'manifest.json'})
        for name, expected in manifest['files'].items():
            self.assertEqual(sha256(archive.read(name)).hexdigest(), expected, name)
        self.assertEqual((manifest['format'], set(manifest['runs']), manifest['ai_calls_redaction']),
                         (retention.EXPORT_FORMAT, {first, second}, 'strict'))
        for run_id in (first, second):
            names = {name.partition('/')[2] for name in archive.namelist() if name.startswith(run_id + '/')}
            self.assertLessEqual({'job.json', 'input.json', 'ai-calls.redacted.jsonl', 'result/packet.json'}, names)
            self.assertNotIn('ai-calls.jsonl', names)                                # never the raw log
            self.assertNotIn('error.log', names)
            self.assertFalse(any(name.startswith('inputs/') for name in names))
            calls = archive.read(f'{run_id}/ai-calls.redacted.jsonl').decode('utf-8')
            self.assertRedacted([json.loads(line) for line in calls.splitlines()], 'strict')
            # One redaction, two export paths: the single-run export carries the same bytes.
            single = zipfile.ZipFile(io.BytesIO(client.get(f'/api/runs/{run_id}/export').content))
            self.assertEqual(single.read('ai-calls.redacted.jsonl').decode('utf-8'), calls)
            self.assertIn(PROMPT_MARK, (workspace.directory(run_id) / 'ai-calls.jsonl').read_text(encoding='utf-8'))  # the run is untouched
        self.assertTrue(any(name.startswith(f'{first}/result/reviews/') for name in archive.namelist()))
        self.assertFalse(list((workspace.root / 'exports').iterdir()))
        self.assertIn(('export_data', 'tenant:acme', 'OK'), [(e['action'], e['subject'], e['outcome']) for e in workspace.audit_log.entries()])
        # A viewer cannot take the tenant's data, and the attempt is on the chain.
        workspace.roles = frozenset({'VIEWER'})
        self.assertEqual(client.get('/api/tenant/export').status_code, 403)
        self.assertEqual((self.last_audit(workspace)['action'], self.last_audit(workspace)['outcome']), ('export_data', 'DENIED'))

    def test_export_runs_redacts_the_call_log_when_given_a_level_and_hashes_what_it_archived(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        runs = Path(temp.name) / 'runs'
        run_id, empty = f'{1:032x}', f'{2:032x}'
        for name, log in ((run_id, json.dumps(call_entry(prompt='secret prompt', text=PASSAGE)) + '\n'), (empty, '\n\n')):
            (runs / name / 'result').mkdir(parents=True)
            (runs / name / 'job.json').write_text(json.dumps({'id': name, 'state': 'COMPLETED'}), encoding='utf-8')
            (runs / name / 'result' / 'packet.json').write_text('{}', encoding='utf-8')
            (runs / name / 'ai-calls.jsonl').write_text(log, encoding='utf-8')
        destination = Path(temp.name) / 'standard.zip'
        manifest = retention.export_runs(runs, [run_id, empty], destination, RedactionPolicy('standard'))
        self.assertEqual(manifest['ai_calls_redaction'], 'standard')
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            self.assertIn(f'{run_id}/ai-calls.redacted.jsonl', names)
            self.assertNotIn(f'{run_id}/ai-calls.jsonl', names)
            self.assertFalse(any(name.startswith(empty + '/ai-calls') for name in names))    # nothing to redact, nothing written
            data = archive.read(f'{run_id}/ai-calls.redacted.jsonl')
            self.assertEqual(sha256(data).hexdigest(), manifest['files'][f'{run_id}/ai-calls.redacted.jsonl'])
            entry = json.loads(data)
            self.assertEqual(set(entry), set(call_entry()) & ALLOWED_CALL_FIELDS['standard'])
        # Without a level the log is the operator's own copy, byte for byte (the v0.17 behaviour).
        verbatim = retention.export_runs(runs, [run_id], Path(temp.name) / 'operator.zip')
        self.assertIsNone(verbatim['ai_calls_redaction'])
        self.assertEqual(verbatim['files'][f'{run_id}/ai-calls.jsonl'], sha256((runs / run_id / 'ai-calls.jsonl').read_bytes()).hexdigest())


class DeleteTests(Harness):
    def test_delete_needs_delete_data_is_audited_and_removes_only_the_named_run(self):
        workspace, client = self.make(OPERATOR_NAME='ayse')
        keep, remove = self.completed(workspace, client), self.completed(workspace, client)
        for role in ('VIEWER', 'REVIEWER', 'ANALYST'):
            with self.subTest(role=role):
                workspace.roles = frozenset({role})
                denied = client.delete(f'/api/runs/{remove}', headers={'X-Request-ID': f'delete-as-{role.lower()}'})
                self.assertEqual(denied.status_code, 403)
                self.assertIn('delete_data', denied.json()['detail'])
                entry = self.last_audit(workspace)
                self.assertEqual((entry['action'], entry['outcome'], entry['subject'], entry['request_id']),
                                 ('delete_data', 'DENIED', f'/api/runs/{remove}', f'delete-as-{role.lower()}'))
                self.assertTrue((workspace.runs / remove / 'job.json').is_file())
        workspace.roles = frozenset({'ADMIN'})
        # The run in progress is never deleted; the refusal is on the chain too.
        workspace.active = remove
        try:
            self.assertEqual(client.delete(f'/api/runs/{remove}').status_code, 409)
        finally:
            workspace.active = None
        self.assertEqual((self.last_audit(workspace)['action'], self.last_audit(workspace)['outcome']), ('delete_run', 'REFUSED'))
        self.assertEqual(client.delete(f'/api/runs/{remove}').json(), {'deleted': [{'id': remove, 'outcome': 'OK'}]})
        self.assertFalse((workspace.runs / remove).exists())
        self.assertEqual(client.get(f'/api/runs/{remove}').status_code, 404)
        self.assertEqual([row['id'] for row in client.get('/api/state').json()['runs']], [keep])
        self.assertEqual(client.get(f'/api/runs/{keep}/verify').json()['integrity'], 'VERIFIED')
        self.assertEqual(client.delete(f'/api/runs/{remove}').json(), {'deleted': [{'id': remove, 'outcome': 'NOT_FOUND'}]})
        self.assertEqual(client.delete('/api/runs/' + 'A' * 32).status_code, 422)
        self.assertEqual((self.last_audit(workspace)['action'], self.last_audit(workspace)['outcome']), ('delete_run', 'REFUSED'))
        self.assertTrue(workspace.audit_log.verify())


class RetentionTests(Harness):
    def test_the_sweep_never_deletes_a_reviewed_an_undated_or_the_active_run(self):
        workspace, client = self.make(RETENTION_DAYS='30')
        reviewed, plain, undated, active, recent = (self.completed(workspace, client) for _ in range(5))
        self.assertEqual(client.post(f'/api/runs/{reviewed}/reviews', json=uncertain_review(workspace.packet(reviewed))).status_code, 200)
        for run_id in (reviewed, plain, active):
            workspace.update(run_id, created_at='2026-01-01T00:00:00+00:00')
        workspace.update(undated, created_at='unknown')
        self.assertEqual(set(client.get('/api/retention').json()['expired']), {plain, active})
        workspace.active = active
        try:
            self.assertEqual(client.get('/api/retention').json()['expired'], [plain])
            workspace.roles = frozenset({'VIEWER'})
            self.assertEqual(client.post('/api/retention/sweep', json={}).status_code, 403)
            self.assertEqual((self.last_audit(workspace)['action'], self.last_audit(workspace)['outcome']), ('delete_data', 'DENIED'))
            workspace.roles = frozenset({'ADMIN'})
            self.assertEqual(client.post('/api/retention/sweep', json={}).json(), {'deleted': [{'id': plain, 'outcome': 'OK'}], 'days': 30})
        finally:
            workspace.active = None
        # Once it is no longer running, the old unreviewed run goes; the reviewed and the undated one never do.
        self.assertEqual(client.post('/api/retention/sweep', json={}).json()['deleted'], [{'id': active, 'outcome': 'OK'}])
        self.assertEqual(client.post('/api/retention/sweep', json={}).json(), {'deleted': [], 'days': 30})
        self.assertEqual({row['id'] for row in workspace.list_runs()}, {reviewed, undated, recent})
        # Reviewed runs are kept by setting, not by accident; an undated run is kept whatever the setting.
        workspace.retention = retention.RetentionSettings(30, False)
        self.assertEqual(workspace.expired_runs(), [reviewed])
        deletes = [e for e in workspace.audit_log.entries() if e['action'] == 'delete_run']
        self.assertEqual([(e['subject'], e['outcome']) for e in deletes], [(plain, 'OK'), (active, 'OK')])


class Collect(logging.Handler):
    """Every record any logger emits, formatted with its traceback, as one searchable text."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        text = f'{record.name} {record.levelname} {record.getMessage()}'
        if record.exc_info:
            text += '\n' + ''.join(traceback.format_exception(*record.exc_info))
        self.lines.append(text)


class NormalLogTests(Harness):
    def capture(self):
        handler = Collect()
        loggers = [logging.getLogger(), logging.getLogger('regchain')]
        for logger in loggers:
            previous = logger.level
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            self.addCleanup(logger.setLevel, previous)
            self.addCleanup(logger.removeHandler, handler)
        stdout, stderr = io.StringIO(), io.StringIO()
        self.enterContext(contextlib.redirect_stdout(stdout))
        self.enterContext(contextlib.redirect_stderr(stderr))
        return handler, stdout, stderr

    def test_a_full_analysis_through_the_workspace_logs_no_passage_description_or_secret(self):
        self.enterContext(patch.dict(os.environ, SECRETS))
        handler, stdout, stderr = self.capture()
        workspace, client = self.make(TENANT_ID='acme')
        run_id = self.completed(workspace, client)
        # The markers went through the analysis: the retained packet holds them, so a clean log is not an empty test.
        packet = json.dumps(workspace.packet(run_id), ensure_ascii=False)
        self.assertTrue(PASSAGE in packet and DESCRIPTION in packet)
        for path in ('', '/packet', '/report', '/ai-calls', '/export', '/labels', '/verify'):
            self.assertEqual(client.get(f'/api/runs/{run_id}{path}').status_code, 200, path)
        for path in ('/api/state', '/api/audit', '/api/retention', '/api/tenant/export'):
            self.assertEqual(client.get(path).status_code, 200, path)
        self.assertEqual(client.post(f'/api/runs/{run_id}/reviews', json=uncertain_review(workspace.packet(run_id))).status_code, 200)
        invalid = client.post(f'/api/runs/{run_id}/reviews', json={'decisions': PASSAGE})
        self.assertEqual(invalid.status_code, 422)
        self.assertNotIn('Leeds archive', invalid.text)                             # validation errors do not echo input
        # A failure whose message carries the description and a secret: the job says what failed, not the text.
        def refusing(_):
            raise RuntimeError(f'model refused {DESCRIPTION} with {SECRET}')
        workspace.provider_factory = refusing
        failed = self.finished(workspace, client)
        self.assertEqual(failed['state'], 'FAILED')
        self.assertIn('RuntimeError', failed['error'])
        self.assertTrue(handler.lines)                                              # the capture sees the request log
        captured = '\n'.join(handler.lines) + stdout.getvalue() + stderr.getvalue()
        records = {'logs': captured, 'job': json.dumps(failed, ensure_ascii=False),
                   'audit.jsonl': (workspace.root / 'audit.jsonl').read_text(encoding='utf-8'),
                   'ai-calls.jsonl': (workspace.directory(run_id) / 'ai-calls.jsonl').read_text(encoding='utf-8'),
                   'job.json': (workspace.directory(run_id) / 'job.json').read_text(encoding='utf-8')}
        for name, text in records.items():
            for leaked in LEAKS:
                with self.subTest(record=name, leaked=leaked[:20]):
                    self.assertNotIn(leaked, text)


if __name__ == '__main__':
    unittest.main()

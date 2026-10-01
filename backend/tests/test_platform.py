import json
import os
import tempfile
import threading
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from regchain.evidence import GENESIS
from regchain.platform import audit, context, rbac, redaction, retention, secrets, storage


def make_link(link: Path, target: Path) -> bool:
    """A symlink, or on Windows a junction (no privilege needed); False when neither can be made."""
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == 'nt' and target.is_dir():
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
            return True
        except OSError:
            return False
    return False


def analyst(tenant='acme', request_id='req-0000001'):
    return context.RequestContext(request_id, tenant, 'ayse@example.test', frozenset({rbac.ANALYST}),
                                  '2026-09-23T10:00:00+00:00')


class RequestIdTests(unittest.TestCase):
    def test_well_formed_client_id_is_kept(self):
        self.assertEqual(context.new_request_id('gw-2026.09.23_abc-123'), 'gw-2026.09.23_abc-123')

    def test_malformed_client_id_is_replaced(self):
        for bad in (None, '', 'short', 'has space here', 'quote"inside"x', 'x' * 65, 'çok-güzel-id', 42):
            value = context.new_request_id(bad)
            self.assertNotEqual(value, bad)
            self.assertRegex(value, r'^[0-9a-f]{32}$')
        self.assertNotEqual(context.new_request_id(), context.new_request_id())

    def test_deterministic_id_is_stable_and_input_sensitive(self):
        body = sha256(b'{"company": 1}').hexdigest()
        base = ('acme', 'post', '/api/runs', body, '2026-09-23T10:00:00+00:00')
        first = context.deterministic_request_id(*base)
        self.assertEqual(first, context.deterministic_request_id(*base))
        self.assertEqual(first, context.deterministic_request_id('acme', 'POST', '/api/runs', body, base[4]))
        self.assertRegex(first, r'^[0-9a-f]{32}$')
        # A deterministic id is a valid X-Request-ID, so a client can echo it back.
        self.assertEqual(context.new_request_id(first), first)
        variants = [('beta', 'POST', '/api/runs', body, base[4]), ('acme', 'DELETE', '/api/runs', body, base[4]),
                    ('acme', 'POST', '/api/runs/x', body, base[4]), ('acme', 'POST', '/api/runs', sha256(b'other').hexdigest(), base[4]),
                    ('acme', 'POST', '/api/runs', body, '2026-09-23T10:00:01+00:00')]
        ids = {context.deterministic_request_id(*v) for v in variants}
        self.assertEqual(len(ids), len(variants))
        self.assertNotIn(first, ids)
        with self.assertRaises(ValueError):
            context.deterministic_request_id('acme', 'POST', '/api/runs', 'not-a-hash', base[4])

    def test_bind_context_nests_and_restores(self):
        self.assertIsNone(context.current_context())
        with context.bind_context(tenant_id='acme', actor='ayse', roles=[rbac.VIEWER]) as outer:
            self.assertIs(context.current_context(), outer)
            self.assertRegex(outer.request_id, r'^[0-9a-f]{32}$')
            self.assertEqual(outer.roles, frozenset({'VIEWER'}))
            with context.bind_context(analyst('beta')) as inner:
                self.assertEqual(context.current_context().tenant_id, 'beta')
                self.assertIsNot(inner, outer)
            self.assertIs(context.current_context(), outer)
        self.assertIsNone(context.current_context())
        with self.assertRaises(ValueError):
            context.RequestContext('bad id', 'acme', 'ayse', frozenset(), '2026-09-23T10:00:00+00:00')
        with self.assertRaises(ValueError):
            context.RequestContext('req-0000001', '', 'ayse', frozenset(), '2026-09-23T10:00:00+00:00')


class RbacTests(unittest.TestCase):
    def test_permission_table(self):
        self.assertTrue(rbac.permitted([rbac.VIEWER], 'view_runs'))
        self.assertFalse(rbac.permitted([rbac.VIEWER], 'start_analysis'))
        self.assertFalse(rbac.permitted([rbac.VIEWER], 'export_data'))
        self.assertTrue(rbac.permitted([rbac.REVIEWER], 'save_review'))
        self.assertFalse(rbac.permitted([rbac.ANALYST], 'save_review'))
        self.assertTrue(rbac.permitted([rbac.ANALYST], 'start_analysis'))
        self.assertTrue(rbac.permitted([rbac.ANALYST], 'read_ai_logs'))
        self.assertFalse(rbac.permitted([rbac.ANALYST, rbac.REVIEWER], 'delete_data'))
        for permission in rbac.PERMISSIONS:
            self.assertTrue(rbac.permitted({rbac.ADMIN}, permission), permission)
        self.assertFalse(rbac.permitted([], 'view_runs'))
        self.assertFalse(rbac.permitted(None, 'view_runs'))
        with self.assertRaises(ValueError):
            rbac.permitted([rbac.ADMIN], 'launch_missiles')

    def test_require_raises_with_permission_name(self):
        with self.assertRaises(rbac.PermissionDenied) as caught:
            rbac.require('view_runs')
        self.assertEqual(caught.exception.permission, 'view_runs')
        with context.bind_context(analyst()):
            self.assertEqual(rbac.require('start_analysis').actor, 'ayse@example.test')
            with self.assertRaises(rbac.PermissionDenied) as caught:
                rbac.require('delete_data')
        self.assertEqual(caught.exception.permission, 'delete_data')
        self.assertIn('delete_data', str(caught.exception))
        self.assertIn('yetkiniz yok', str(caught.exception))
        self.assertIsInstance(caught.exception, ValueError)

    def test_tenant_isolation(self):
        ctx = analyst('acme')
        self.assertTrue(rbac.same_tenant(ctx, 'acme'))
        self.assertFalse(rbac.same_tenant(ctx, 'beta'))
        self.assertFalse(rbac.same_tenant(ctx, ''))
        self.assertFalse(rbac.same_tenant(ctx, None))
        self.assertFalse(rbac.same_tenant(None, 'acme'))
        rbac.assert_tenant(ctx, 'acme')
        with self.assertRaises(rbac.PermissionDenied) as caught:
            rbac.assert_tenant(ctx, 'beta', 'export_data')
        self.assertEqual(caught.exception.permission, 'export_data')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'audit' / 'audit.jsonl'
        self.log = audit.AuditLog(self.path)

    def test_append_and_verify(self):
        self.assertTrue(self.log.verify())
        self.assertEqual(self.log.entries(), [])
        first = self.log.record('start_analysis', 'a' * 32, 'OK', 'labels=2', context=analyst())
        with context.bind_context(analyst('acme', 'req-0000002')):
            second = self.log.record('save_review', 'b' * 32, 'DENIED', 'x' * 500)
        self.assertEqual(first['previous_hash'], GENESIS)
        self.assertEqual(second['previous_hash'], first['entry_hash'])
        self.assertEqual(second['request_id'], 'req-0000002')
        self.assertEqual(len(second['detail']), audit.DETAIL_LIMIT)
        rows = self.log.entries()
        self.assertEqual([r['action'] for r in rows], ['start_analysis', 'save_review'])
        self.assertEqual(sorted(rows[0]), sorted(audit.FIELDS))
        self.assertTrue(self.log.verify())
        self.assertEqual(audit.AuditLog(self.path).head, second['entry_hash'])
        self.assertEqual(audit.AuditLog(self.path).record('export', 'c' * 32, 'OK', context=analyst())['previous_hash'],
                         second['entry_hash'])

    def test_tamper_detection(self):
        for index in range(3):
            self.log.record('view_run', f'{index:032x}', 'OK', context=analyst())
        lines = self.path.read_text(encoding='utf-8').splitlines()
        edited = json.loads(lines[1])
        edited['outcome'] = 'DENIED'
        self.path.write_text('\n'.join([lines[0], json.dumps(edited), lines[2]]) + '\n', encoding='utf-8')
        self.assertFalse(audit.AuditLog(self.path).verify())
        self.path.write_text('\n'.join([lines[0], lines[2]]) + '\n', encoding='utf-8')
        self.assertFalse(audit.AuditLog(self.path).verify())
        forged = json.loads(lines[2])
        forged['subject'] = 'f' * 32
        self.path.write_text('\n'.join(lines + [json.dumps(forged)]) + '\n', encoding='utf-8')
        self.assertFalse(audit.AuditLog(self.path).verify())
        self.path.write_text('\n'.join(lines) + '\nnot json\n', encoding='utf-8')
        self.assertFalse(audit.AuditLog(self.path).verify())
        self.path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        self.assertTrue(audit.AuditLog(self.path).verify())

    def test_refuses_entry_without_context_or_with_bad_outcome(self):
        with self.assertRaises(ValueError):
            self.log.record('view_run', 'x', 'OK')
        with self.assertRaises(ValueError):
            self.log.record('view_run', 'x', 'MAYBE', context=analyst())
        self.assertEqual(self.log.entries(), [])

    def test_concurrent_appends_keep_the_chain(self):
        def worker(number):
            with context.bind_context(analyst('acme', f'req-{number:07d}')):
                for _ in range(10):
                    self.log.record('view_run', 'a' * 32, 'OK')
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        rows = self.log.entries()
        self.assertEqual(len(rows), 50)
        self.assertEqual(len({r['entry_hash'] for r in rows}), 50)
        self.assertTrue(self.log.verify())


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'store'
        self.store = storage.LocalDocumentStore(self.root)

    def test_content_addressed_key_round_trip(self):
        key = self.store.put('acme', 'Uyum Politikası v2.pdf', b'%PDF-1.4 policy')
        self.assertEqual(key, sha256(b'%PDF-1.4 policy').hexdigest() + '-Uyum_Politikas_v2.pdf')
        self.assertEqual(self.store.put('acme', 'Uyum Politikası v2.pdf', b'%PDF-1.4 policy'), key)
        self.assertEqual(self.store.get('acme', key), b'%PDF-1.4 policy')
        self.assertEqual(self.store.list('acme'), [key])
        self.assertEqual(len(list((self.root / 'acme').iterdir())), 1)
        other = self.store.put('acme', 'Uyum Politikası v2.pdf', b'%PDF-1.4 policy (rev)')
        self.assertNotEqual(other, key)
        self.store.delete('acme', key)
        with self.assertRaises(storage.DocumentNotFound):
            self.store.get('acme', key)
        self.assertEqual(self.store.list('acme'), [other])

    def test_traversal_and_bad_identifiers_are_refused(self):
        key = self.store.put('acme', 'p.txt', b'policy')
        for bad_key in ('../acme/' + key, key + '/../x', 'x' * 64 + '-p.txt', 'p.txt', key.replace('-', '\\')):
            with self.assertRaises(storage.StorageRefused):
                self.store.get('acme', bad_key)
        for bad_tenant in ('../acme', 'Acme', 'ac me', '', '.hidden', 'a' * 65):
            with self.assertRaises(storage.StorageRefused):
                self.store.put(bad_tenant, 'p.txt', b'policy')
        with self.assertRaises(storage.StorageRefused):
            self.store.put('acme', 'p.txt', b'')

    def test_no_cross_tenant_reads(self):
        key = self.store.put('acme', 'p.txt', b'acme policy')
        with self.assertRaises(storage.DocumentNotFound):
            self.store.get('beta', key)
        with self.assertRaises(storage.DocumentNotFound):
            self.store.delete('beta', key)
        self.assertEqual(self.store.list('beta'), [])
        self.assertEqual(self.store.get('acme', key), b'acme policy')

    def test_tampered_file_is_refused(self):
        key = self.store.put('acme', 'p.txt', b'acme policy')
        (self.root / 'acme' / key).write_bytes(b'edited on disk')
        with self.assertRaises(storage.StorageRefused):
            self.store.get('acme', key)

    def test_links_are_refused(self):
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        key = storage.content_key('p.txt', b'outside')
        (outside / key).write_bytes(b'outside')
        if not make_link(self.root / 'evil', outside):
            self.skipTest('Neither symlinks nor junctions can be created here')
        with self.assertRaises(storage.StorageRefused):
            self.store.get('evil', key)
        with self.assertRaises(storage.StorageRefused):
            self.store.put('evil', 'p.txt', b'new')
        with self.assertRaises(storage.StorageRefused):
            self.store.list('evil')
        self.assertEqual((outside / key).read_bytes(), b'outside')
        # A linked file inside a real tenant directory, named like a valid key.
        target = Path(self.temp.name) / 'target.bin'
        target.write_bytes(b'target')
        good = self.store.put('acme', 'p.txt', b'acme')
        linked = self.root / 'acme' / (sha256(b'target').hexdigest() + '-p.txt')
        if make_link(linked, target):
            with self.assertRaises(storage.StorageRefused):
                self.store.get('acme', linked.name)
            self.assertEqual(self.store.list('acme'), [good])


class SecretsTests(unittest.TestCase):
    def test_allowlist(self):
        env = {'LLM_API_KEY': 'sk-live-1234567890', 'PATH': '/usr/bin', 'APP_DB_PASSWORD': '   '}
        self.assertEqual(secrets.get_secret('LLM_API_KEY', env), 'sk-live-1234567890')
        with self.assertRaises(secrets.SecretUnavailable):
            secrets.get_secret('PATH', env)
        with self.assertRaises(secrets.SecretUnavailable):
            secrets.get_secret('DATABASE_URL', env)
        with self.assertRaises(secrets.SecretUnavailable):
            secrets.get_secret('APP_DB_PASSWORD', env)
        with patch.dict(os.environ, {'EVM_RPC_URL': 'https://rpc.example.test/key123'}):
            self.assertEqual(secrets.get_secret('EVM_RPC_URL'), 'https://rpc.example.test/key123')

    def test_redacts_values_and_url_passwords(self):
        env = {'LLM_API_KEY': 'sk-live-1234567890', 'DATABASE_URL': 'postgresql://app:Sup3rS3cret@db.internal:5432/regchain'}
        line = ('connect failed for postgresql://app:Sup3rS3cret@db.internal:5432/regchain; retry with '
                'password Sup3rS3cret and header Bearer sk-live-1234567890')
        masked = secrets.redact_secrets(line, env)
        self.assertNotIn('Sup3rS3cret', masked)
        self.assertNotIn('sk-live-1234567890', masked)
        self.assertIn('[redacted DATABASE_URL]', masked)
        self.assertIn('[redacted LLM_API_KEY]', masked)
        self.assertIn('connect failed for', masked)
        self.assertEqual(secrets.redact_secrets('nothing here', env), 'nothing here')
        self.assertEqual(secrets.redact_secrets('nothing here', {}), 'nothing here')


def call_entry(**extra):
    entry = {'at': '2026-09-23T10:00:00+00:00', 'task': 'judge.applicability', 'stage': 'judge.applicability',
             'provider': 'ollama', 'model': 'qwen3:8b', 'model_version': 'qwen3:8b@sha256:abc', 'thinking': True,
             'prompt_sha256': 'a' * 64, 'payload_sha256': 'b' * 64, 'schema_sha256': 'c' * 64, 'request_bytes': 18234,
             'provision_id': 'd' * 64, 'obligation_id': 'e' * 64, 'evidence_ids': ['f' * 64], 'cache_hit': False,
             'retry_count': 1, 'status': 'PROVIDER_FAILURE', 'elapsed_ms': 4210, 'prompt_tokens': 3900,
             'output_tokens': 0, 'done_reason': None, 'error': 'HTTP 503: provider error after 3 attempts ' + 'x' * 200,
             'num_predict': 6144, 'num_predict_reduced': True}
    entry.update(extra)
    return entry


class RedactionTests(unittest.TestCase):
    def test_levels_keep_and_drop_the_right_fields(self):
        entry = call_entry(prompt='Extract regulatory obligation candidates...', payload={'source_paragraph': 'A firm must'},
                           response='{"status": "EXTRACTED"}')
        minimal = redaction.redact_call(entry, redaction.RedactionPolicy('minimal'))
        standard = redaction.redact_call(entry, redaction.RedactionPolicy('standard'))
        strict = redaction.redact_call(entry, redaction.RedactionPolicy('strict'))
        for kept in (minimal, standard, strict):
            for forbidden in ('prompt', 'payload', 'response'):
                self.assertNotIn(forbidden, kept)
            self.assertEqual(len(kept['error']), 120)
            self.assertTrue(kept['error'].startswith('HTTP 503'))
            for field in ('at', 'task', 'model_version', 'prompt_sha256', 'status', 'elapsed_ms', 'prompt_tokens', 'retry_count'):
                self.assertIn(field, kept)
        self.assertEqual(set(minimal), set(call_entry()))
        self.assertIn('request_bytes', minimal)
        self.assertIn('thinking', minimal)
        for field in ('request_bytes', 'thinking', 'num_predict', 'num_predict_reduced', 'done_reason', 'schema_sha256'):
            self.assertNotIn(field, standard)
            self.assertNotIn(field, strict)
        for field in ('provision_id', 'obligation_id', 'evidence_ids', 'payload_sha256'):
            self.assertIn(field, standard)
            self.assertNotIn(field, strict)
        self.assertEqual(standard['evidence_ids'], ['f' * 64])
        self.assertIsNot(standard['evidence_ids'], entry['evidence_ids'])
        self.assertEqual(redaction.redact_call({'status': 'OK', 'error': None}, redaction.RedactionPolicy('strict')),
                         {'status': 'OK', 'error': None})
        with self.assertRaises(ValueError):
            redaction.RedactionPolicy('public')

    def test_scrub_text_replaces_long_passages_only(self):
        passage = ('A firm must establish and maintain adequate policies and procedures sufficient to ensure compliance '
                   'of the firm and its managers and employees with its obligations under the regulatory system and for '
                   'countering the risk that the firm might be used to further financial crime.')
        self.assertGreater(len(passage), 200)
        line = f'judge failed for CONC 7.3.4: passage="{passage}" status=PROVIDER_FAILURE'
        scrubbed = redaction.scrub_text(line)
        self.assertEqual(scrubbed, f'judge failed for CONC 7.3.4: passage="[redacted {len(passage)} chars]" status=PROVIDER_FAILURE')
        self.assertNotIn('financial crime', redaction.scrub_text('unquoted: ' + passage))
        short = 'judge failed for CONC 7.3.4: HTTP 503 after 3 attempts'
        self.assertEqual(redaction.scrub_text(short), short)
        hashes = 'evidence_ids=' + ', '.join(f'{n:064x}' for n in range(5))
        self.assertEqual(redaction.scrub_text(hashes), hashes)
        blob = 'payload=' + 'QUJD' * 100
        self.assertEqual(redaction.scrub_text(blob), 'payload=[redacted 400 chars]')


def run_row(run_id, created, reviews=()):
    return {'id': run_id, 'created_at': created, 'reviews': list(reviews)}


class RetentionSettingsTests(unittest.TestCase):
    def test_from_env(self):
        self.assertEqual(retention.RetentionSettings.from_env({}), retention.RetentionSettings(None, True))
        self.assertEqual(retention.RetentionSettings.from_env({'RETENTION_DAYS': '30', 'RETENTION_KEEP_REVIEWED': 'off'}),
                         retention.RetentionSettings(30, False))
        self.assertEqual(retention.RetentionSettings.from_env({'RETENTION_DAYS': ' 7 ', 'RETENTION_KEEP_REVIEWED': 'ON'}),
                         retention.RetentionSettings(7, True))
        for env in ({'RETENTION_DAYS': '0'}, {'RETENTION_DAYS': 'abc'}, {'RETENTION_DAYS': '-3'},
                    {'RETENTION_DAYS': '99999'}, {'RETENTION_KEEP_REVIEWED': 'maybe'}):
            with self.assertRaises(ValueError):
                retention.RetentionSettings.from_env(env)
        with patch.dict(os.environ, {'RETENTION_DAYS': '90', 'RETENTION_KEEP_REVIEWED': ''}):
            self.assertEqual(retention.RetentionSettings.from_env(), retention.RetentionSettings(90, True))

    def test_expired_selection(self):
        now = datetime(2026, 9, 23, 12, tzinfo=timezone.utc)
        old = (now - timedelta(days=40)).isoformat()
        edge = (now - timedelta(days=30)).isoformat()
        recent = (now - timedelta(days=5)).isoformat()
        rows = [run_row('1' * 32, old), run_row('2' * 32, old, [{'id': 'r'}]), run_row('3' * 32, recent),
                run_row('4' * 32, edge), {'id': '5' * 32, 'reviews': []}, run_row('6' * 32, 'not a date'),
                run_row('7' * 32, (now - timedelta(days=31)).replace(tzinfo=None).isoformat())]
        keep = retention.RetentionSettings(30, True)
        self.assertEqual([r['id'] for r in retention.expired(rows, now, keep)], ['1' * 32, '7' * 32])
        self.assertEqual([r['id'] for r in keep.expired(rows, now.isoformat())], ['1' * 32, '7' * 32])
        purge = retention.RetentionSettings(30, False)
        self.assertEqual([r['id'] for r in retention.expired(rows, now, purge)], ['1' * 32, '2' * 32, '7' * 32])
        self.assertEqual(retention.expired(rows, now, retention.RetentionSettings(None, False)), [])
        with self.assertRaises(ValueError):
            retention.RetentionSettings(0, True)


class RetentionFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name) / 'workspace' / 'runs'
        self.runs.mkdir(parents=True)
        self.audit = audit.AuditLog(Path(self.temp.name) / 'audit.jsonl')
        self.ids = [f'{n:032x}' for n in (1, 2)]
        for run_id in self.ids:
            directory = self.runs / run_id
            (directory / 'result' / 'reviews').mkdir(parents=True)
            (directory / 'inputs').mkdir()
            (directory / 'job.json').write_text(json.dumps({'id': run_id, 'state': 'COMPLETED'}), encoding='utf-8')
            (directory / 'input.json').write_text(json.dumps({'provider': 'rules'}), encoding='utf-8')
            (directory / 'ai-calls.jsonl').write_text(json.dumps(call_entry()) + '\n', encoding='utf-8')
            (directory / 'result' / 'packet.json').write_text('{"head": "' + run_id + '"}', encoding='utf-8')
            (directory / 'result' / 'reviews' / ('a' * 64 + '.json')).write_text('{"review": 1}', encoding='utf-8')
            (directory / 'inputs' / 'policy.txt').write_text('Staff must retain records.', encoding='utf-8')
            (directory / 'error.log').write_text('Traceback: local diagnostics', encoding='utf-8')

    def test_export_zip_contents_and_manifest(self):
        destination = Path(self.temp.name) / 'exports' / 'runs.zip'
        destination.parent.mkdir()
        manifest = retention.export_runs(self.runs, self.ids, destination)
        self.assertTrue(destination.is_file())
        self.assertEqual(manifest['runs'], self.ids)
        self.assertEqual(manifest['format'], retention.EXPORT_FORMAT)
        with zipfile.ZipFile(destination) as archive:
            names = set(archive.namelist())
            for run_id in self.ids:
                for name in ('job.json', 'input.json', 'ai-calls.jsonl', 'result/packet.json', 'result/reviews/' + 'a' * 64 + '.json'):
                    self.assertIn(run_id + '/' + name, names)
                self.assertNotIn(run_id + '/inputs/policy.txt', names)
                self.assertNotIn(run_id + '/error.log', names)
            self.assertEqual(names, set(manifest['files']) | {'manifest.json'})
            written = json.loads(archive.read('manifest.json'))
            self.assertEqual(written['files'], manifest['files'])
            for arcname, expected in manifest['files'].items():
                self.assertEqual(sha256(archive.read(arcname)).hexdigest(), expected, arcname)
                run_id, _, relative = arcname.partition('/')
                self.assertEqual(sha256((self.runs / run_id / relative).read_bytes()).hexdigest(), expected)
        with self.assertRaises(ValueError):
            retention.export_runs(self.runs, self.ids, destination)
        self.assertEqual([p.name for p in destination.parent.iterdir()], ['runs.zip'])

    def test_export_refuses_bad_ids_and_destinations(self):
        elsewhere = Path(self.temp.name) / 'other.zip'
        for bad in ('../workspace', 'A' * 32, self.ids[0][:-1], '0' * 32):
            with self.assertRaises(ValueError):
                retention.export_runs(self.runs, [bad], elsewhere)
        with self.assertRaises(ValueError):
            retention.export_runs(self.runs, [], elsewhere)
        with self.assertRaises(ValueError):
            retention.export_runs(self.runs, self.ids, self.runs / 'inside.zip')
        self.assertFalse(elsewhere.exists())
        self.assertEqual(sorted(p.name for p in self.runs.iterdir()), self.ids)

    def test_delete_refuses_bad_ids_without_touching_anything(self):
        for bad in ('../workspace', 'A' * 32, self.ids[0][:-1], 'x' * 32, None):
            with self.assertRaises(ValueError):
                retention.delete_runs(self.runs, [self.ids[0], bad], self.audit, analyst())
        self.assertEqual(sorted(p.name for p in self.runs.iterdir()), self.ids)
        rows = self.audit.entries()
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r['action'] == 'delete_run' and r['outcome'] == 'REFUSED' for r in rows))
        self.assertEqual(rows[0]['subject'], '..?workspace')
        self.assertTrue(self.audit.verify())

    def test_delete_removes_only_named_runs_and_audits_each(self):
        missing = f'{9:032x}'
        ctx = analyst('acme', 'req-delete-1')
        result = retention.delete_runs(self.runs, [self.ids[0], missing], self.audit, ctx)
        self.assertEqual(result, [{'id': self.ids[0], 'outcome': 'OK'}, {'id': missing, 'outcome': 'NOT_FOUND'}])
        self.assertEqual([p.name for p in self.runs.iterdir()], [self.ids[1]])
        self.assertTrue((self.runs / self.ids[1] / 'result' / 'packet.json').is_file())
        rows = self.audit.entries()
        self.assertEqual([(r['subject'], r['outcome'], r['request_id'], r['actor']) for r in rows],
                         [(self.ids[0], 'OK', 'req-delete-1', 'ayse@example.test'), (missing, 'NOT_FOUND', 'req-delete-1', 'ayse@example.test')])
        self.assertTrue(self.audit.verify())

    def test_delete_refuses_a_run_directory_that_points_outside(self):
        outside = Path(self.temp.name) / 'precious'
        outside.mkdir()
        (outside / 'keep.txt').write_text('keep', encoding='utf-8')
        linked = f'{7:032x}'
        if not make_link(self.runs / linked, outside):
            self.skipTest('Neither symlinks nor junctions can be created here')
        result = retention.delete_runs(self.runs, [linked, self.ids[1]], self.audit, analyst())
        self.assertEqual(result, [{'id': linked, 'outcome': 'REFUSED'}, {'id': self.ids[1], 'outcome': 'OK'}])
        self.assertEqual((outside / 'keep.txt').read_text(encoding='utf-8'), 'keep')
        self.assertTrue((self.runs / linked).exists())
        self.assertFalse((self.runs / self.ids[1]).exists())
        self.assertEqual([r['outcome'] for r in self.audit.entries()], ['REFUSED', 'OK'])
        with self.assertRaises(ValueError):
            retention.export_runs(self.runs, [linked], Path(self.temp.name) / 'linked.zip')


if __name__ == '__main__':
    unittest.main()

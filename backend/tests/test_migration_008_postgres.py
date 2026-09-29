"""Upgrade populated v0.6 history, not merely an empty schema."""
import os
import unittest
from pathlib import Path

import psycopg

from regchain.ingestion.parse import parse
from regchain.ingestion.repository import save
from test_ingestion import html_download


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Requires disposable PostgreSQL + pgvector')
class Migration008Tests(unittest.TestCase):
    def test_populated_context_history_is_backfilled_and_remains_immutable(self):
        conn = psycopg.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(conn.close)
        self.addCleanup(conn.rollback)
        migrations = Path(__file__).resolve().parents[1] / 'migrations'
        for file in sorted(migrations.glob('00[1-7]_*.sql')):
            conn.execute(file.read_text(encoding='utf-8'))
        download = html_download('<main><p>A firm must retain records.</p></main>')
        version = save(conn, 'migration-fixture', download, parse(download))['version_id']
        section, text, source_hash = conn.execute(
            'SELECT id,text,content_hash FROM regulation_sections WHERE version_id=%s', (version,)).fetchone()
        run = conn.execute('''INSERT INTO extraction_runs(version_id,provider,model_version,prompt_hash,
            schema_version,input_hash,status) VALUES(%s,'fixture','v1',%s,'v2',%s,'REVIEW_REQUIRED') RETURNING id''',
            (version, 'a'*64, 'b'*64)).fetchone()[0]
        obligation = conn.execute('''INSERT INTO obligations(version_id,section_id,extraction_run_id,
            source_quote,quote_start,quote_end,subject,modality,required_action,jurisdiction,risk_category,confidence)
            VALUES(%s,%s,%s,%s,0,%s,'A firm','MUST','retain records.','UK','UNKNOWN',0.5) RETURNING id''',
            (version, section, run, text, len(text))).fetchone()[0]
        conn.execute('''INSERT INTO obligation_context_evidence(obligation_id,version_id,section_id,role,quote,source_hash)
            VALUES(%s,%s,%s,'scope',%s,%s)''', (obligation, version, section, text, source_hash))
        conn.execute((migrations/'008_extraction_contract.sql').read_text(encoding='utf-8'))
        row = conn.execute('SELECT version_id=source_version_id,quote,source_hash FROM obligation_context_evidence').fetchone()
        self.assertEqual(row, (True, text, source_hash))
        self.assertEqual(conn.execute('SELECT runtime_manifest FROM extraction_runs').fetchone()[0], {})
        with self.assertRaises(psycopg.errors.RaiseException):
            conn.execute("UPDATE obligation_context_evidence SET quote='changed'")

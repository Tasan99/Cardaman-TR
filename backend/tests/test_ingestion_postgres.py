import os
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import psycopg
from fastapi.testclient import TestClient
from regchain.api import app

from regchain.ingestion.parse import parse
from regchain.ingestion.repository import save, record_rejected
from test_ingestion import HTML, html_download


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Requires disposable PostgreSQL + pgvector')
class IngestionDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        for file in sorted((Path(__file__).resolve().parents[1] / 'migrations').glob('*.sql')):
            self.conn.execute(file.read_text(encoding='utf-8'))

    def ingest(self, html=HTML):
        source = html_download(html)
        return save(self.conn, 'synthetic-fixture', source, parse(source))

    def count(self, table):
        # Only test-owned constant table names are supplied.
        return self.conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]

    def test_duplicate_fetch_keeps_one_version(self):
        first, second = self.ingest(), self.ingest()
        self.assertEqual(second['status'], 'UNCHANGED')
        self.assertEqual(first['version_id'], second['version_id'])
        self.assertEqual(self.count('regulation_versions'), 1)
        self.assertEqual(self.count('regulatory_fetches'), 2)

    def test_parser_upgrade_is_not_reported_as_regulatory_change(self):
        with patch('regchain.ingestion.repository.PARSER_VERSION','fixture-old-parser'):
            first=self.ingest()
        second=self.ingest()
        self.assertNotEqual(first['version_id'],second['version_id'])
        self.assertEqual(second['change_origin'],'PARSER_REPROCESSING')
        self.assertEqual(self.count('regulatory_changes'),0)
        self.assertEqual(self.count('regulation_versions'),2)

    def test_simultaneous_parser_and_source_change_requires_explicit_comparison(self):
        with patch('regchain.ingestion.repository.PARSER_VERSION','fixture-old-parser'):
            self.ingest()
        second=self.ingest(HTML.replace('must record approval','must record written approval'))
        self.assertEqual(second['change_origin'],'SOURCE_AND_PARSER_CHANGE')
        self.assertEqual(self.count('regulatory_changes'),0)

    def test_reversion_is_a_new_version(self):
        self.ingest()
        self.ingest(HTML.replace('must record approval', 'must record written approval'))
        third = self.ingest()
        self.assertEqual(third['version_no'], 3)
        self.assertEqual(self.count('regulation_versions'), 3)
        self.assertEqual(self.count('regulatory_changes'), 2)

    def test_raw_only_change_retains_both_objects(self):
        self.ingest()
        result = self.ingest(HTML.replace('must record', 'must   record'))
        self.assertEqual(result['status'], 'UNCHANGED')
        self.assertEqual(self.count('regulatory_source_objects'), 2)
        self.assertEqual(self.count('regulation_versions'), 1)

    def test_change_links_old_and_new_source_paragraphs(self):
        first = self.ingest()
        second = self.ingest(HTML.replace('unless the account is a sandbox account', 'unless explicitly exempt'))
        row = self.conn.execute('SELECT old_version_id,new_version_id,old_section_id,new_section_id,diff_text FROM regulatory_changes').fetchone()
        self.assertEqual(str(row[0]),first['version_id'])
        self.assertEqual(str(row[1]),second['version_id'])
        self.assertNotEqual(row[2],row[3])
        self.assertIn('sandbox account',row[4])
        self.assertIn('explicitly exempt',row[4])

    def test_failed_write_rolls_back_entire_version(self):
        source = html_download(HTML)
        document = parse(source)
        invalid = replace(document, paragraphs=(document.paragraphs[0],document.paragraphs[0]))
        with self.assertRaises(psycopg.errors.UniqueViolation):
            save(self.conn, 'synthetic-fixture', source, invalid)
        self.assertEqual(self.count('regulations'),0)
        self.assertEqual(self.count('regulation_versions'),0)
        self.assertEqual(self.count('regulatory_source_objects'),0)

    def test_rejection_is_recorded_without_version(self):
        record_rejected(self.conn, html_download('<h1>No content</h1>'), 'No main content')
        self.assertEqual(self.count('regulatory_fetches'),1)
        self.assertEqual(self.count('regulation_versions'),0)

    def test_raw_hash_must_match_bytes(self):
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.conn.execute("INSERT INTO regulatory_source_objects(raw_hash,body,media_type) VALUES(%s,%s,'text/html')", ('0'*64,b'wrong hash'))

    def test_read_api_returns_version_citations_and_original_bytes(self):
        result = self.ingest()
        @contextmanager
        def connection():
            from psycopg.rows import dict_row
            previous = self.conn.row_factory
            self.conn.row_factory = dict_row
            try:
                yield self.conn
            finally:
                self.conn.row_factory = previous
        with patch('regchain.regulatory_api.database', connection):
            client = TestClient(app)
            self.assertEqual(client.get('/regulations').json()[0]['latest_version'],1)
            version = client.get('/versions/' + result['version_id'])
            self.assertEqual(version.status_code,200)
            paragraphs = client.get('/versions/' + result['version_id'] + '/paragraphs').json()
            self.assertEqual(paragraphs[0]['locator_kind'],'html_id')
            source = client.get('/versions/' + result['version_id'] + '/source')
            self.assertEqual(source.content,html_download(HTML).body)
            self.assertIn('attachment',source.headers['Content-Disposition'])

    def test_ingestion_role_can_append_but_cannot_read_companies(self):
        self.conn.execute('CREATE ROLE regchain_ingestion_test NOSUPERUSER NOBYPASSRLS')
        self.conn.execute('GRANT USAGE ON SCHEMA public TO regchain_ingestion_test')
        self.conn.execute('GRANT SELECT ON regulators TO regchain_ingestion_test')
        self.conn.execute('''GRANT SELECT,INSERT ON regulations,regulation_versions,regulation_sections,
            regulatory_changes,regulatory_chunks,regulatory_source_objects,regulatory_fetches TO regchain_ingestion_test''')
        self.conn.execute('SET LOCAL ROLE regchain_ingestion_test')
        self.assertEqual(self.ingest()['status'],'CREATED')
        with self.assertRaises(psycopg.errors.InsufficientPrivilege):
            self.conn.execute('SELECT * FROM companies')

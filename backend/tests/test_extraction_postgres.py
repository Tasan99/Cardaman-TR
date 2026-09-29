import os
import unittest
from pathlib import Path
from uuid import UUID
from contextlib import contextmanager
from unittest.mock import patch

import psycopg
from fastapi.testclient import TestClient
from regchain.api import app

from regchain.ingestion.parse import parse
from regchain.ingestion.repository import save
from regchain.extraction.service import run_version
from regchain.extraction.providers import RulesProvider, ProviderFailure
from test_ingestion import html_download
from test_extraction import ScriptedProvider
from test_quality import ContextProvider


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'),'Requires disposable PostgreSQL + pgvector')
class ExtractionDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        for file in sorted((Path(__file__).resolve().parents[1]/'migrations').glob('*.sql')):
            self.conn.execute(file.read_text(encoding='utf-8'))
        source = html_download('<main><h1>Synthetic fixture</h1><p>A firm must keep records unless exempt.</p></main>')
        result = save(self.conn,'test-extraction',source,parse(source))
        self.version = UUID(result['version_id'])

    def test_candidate_has_valid_source_and_review_record(self):
        result = run_version(self.conn,self.version,RulesProvider())
        self.assertEqual(result['status'],'REVIEW_REQUIRED')
        self.assertEqual(result['candidates'],1)
        row = self.conn.execute('''SELECT o.source_quote,s.text,o.quote_end,o.human_review_required
             FROM obligations o JOIN regulation_sections s ON s.id=o.section_id''').fetchone()
        self.assertEqual(row[0],row[1])
        self.assertEqual(row[2],len(row[0]))
        self.assertTrue(row[3])
        self.assertEqual(self.conn.execute('SELECT count(*) FROM extraction_reviews').fetchone()[0],1)

    def test_repeated_identical_run_is_idempotent(self):
        first = run_version(self.conn,self.version,RulesProvider())
        second = run_version(self.conn,self.version,RulesProvider())
        self.assertEqual(second['status'],'UNCHANGED')
        self.assertEqual(first['run_id'],second['run_id'])
        self.assertEqual(self.conn.execute('SELECT count(*) FROM obligations').fetchone()[0],1)

    def test_failed_provider_stores_failure_without_obligations(self):
        result = run_version(self.conn,self.version,ScriptedProvider([ProviderFailure('offline')]))
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(self.conn.execute('SELECT count(*) FROM obligations').fetchone()[0],0)
        self.assertEqual(self.conn.execute('SELECT status FROM extraction_reviews').fetchone()[0],'INSUFFICIENT_EVIDENCE')

    def test_historical_run_cannot_be_changed(self):
        run_version(self.conn,self.version,RulesProvider())
        with self.assertRaises(psycopg.errors.RaiseException):
            self.conn.execute("UPDATE extraction_runs SET model_version='changed'")

    def test_different_model_version_creates_separate_run(self):
        run_version(self.conn,self.version,RulesProvider())
        provider = RulesProvider()
        provider.model_version = 'next-test-version'
        run_version(self.conn,self.version,provider)
        self.assertEqual(self.conn.execute('SELECT count(*) FROM extraction_runs').fetchone()[0],2)

    def test_context_evidence_persists_under_restricted_extractor_role(self):
        source=html_download('<main><h1>Fixture</h1><p>A firm must retain records.</p><p>Unless the account is exempt.</p></main>')
        version=UUID(save(self.conn,'context-fixture',source,parse(source))['version_id'])
        self.conn.execute('CREATE ROLE test_context_role NOSUPERUSER NOBYPASSRLS')
        self.conn.execute('GRANT USAGE ON SCHEMA public TO test_context_role')
        # regulations is needed to pin a corpus version by its external key; company
        # tables stay out of reach, which is what this test is about.
        self.conn.execute('GRANT SELECT ON regulations,regulation_versions,regulation_sections TO test_context_role')
        self.conn.execute('GRANT SELECT,INSERT ON extraction_runs,obligations,extraction_reviews,obligation_context_evidence TO test_context_role')
        self.conn.execute('SET LOCAL ROLE test_context_role')
        self.assertEqual(run_version(self.conn,version,ContextProvider())['candidates'],1)
        row=self.conn.execute('SELECT quote,version_id FROM obligation_context_evidence').fetchone()
        self.assertEqual(row[0],'Unless the account is exempt.')
        self.assertEqual(row[1],version)

    def test_local_grounding_failure_does_not_abort_later_paragraphs(self):
        source=html_download('<main><h1>Fixture</h1><p>A firm must retain records.</p><p>A firm must review controls.</p></main>')
        version=UUID(save(self.conn,'partial-fixture',source,parse(source))['version_id'])
        responses=['bad JSON']*2+[RulesProvider().generate('A firm must review controls.')]
        result=run_version(self.conn,version,ScriptedProvider(responses))
        self.assertEqual(result['status'],'REVIEW_REQUIRED')
        self.assertEqual(result['candidates'],1)

    def test_context_cannot_link_an_obligation_to_a_different_version(self):
        run_version(self.conn,self.version,RulesProvider())
        obligation=self.conn.execute('SELECT id FROM obligations').fetchone()[0]
        source=html_download('<main><h1>Fixture</h1><p>Unless exempt.</p></main>')
        other=UUID(save(self.conn,'different-context',source,parse(source))['version_id'])
        section=self.conn.execute('SELECT id,text,content_hash FROM regulation_sections WHERE version_id=%s',(other,)).fetchone()
        with self.assertRaises((psycopg.errors.ForeignKeyViolation,psycopg.errors.RaiseException)):
            self.conn.execute('INSERT INTO obligation_context_evidence(obligation_id,version_id,section_id,role,quote,source_hash) VALUES(%s,%s,%s,%s,%s,%s)',
                (obligation,other,section[0],'exception',section[1],section[2]))

    def test_api_exposes_backend_citations_and_review_state(self):
        result = run_version(self.conn,self.version,RulesProvider())
        @contextmanager
        def database():
            from psycopg.rows import dict_row
            previous = self.conn.row_factory
            self.conn.row_factory = dict_row
            try:
                yield self.conn
            finally:
                self.conn.row_factory = previous
        with patch('regchain.extraction_api.database',database):
            client = TestClient(app)
            response = client.get(f'/versions/{self.version}/obligations')
            self.assertEqual(response.status_code,200)
            value = response.json()[0]
            self.assertEqual(value['citation']['source_url'],'https://www.fca.org.uk/test')
            self.assertEqual(value['provider'],'rules')
            self.assertTrue(value['human_review_required'])
            self.assertEqual(client.get(f"/extractions/{result['run_id']}/reviews").json()[0]['status'],'EXTRACTED')

    def test_extractor_role_can_append_but_cannot_read_company_data(self):
        self.conn.execute('CREATE ROLE test_extract_role NOSUPERUSER NOBYPASSRLS')
        self.conn.execute('GRANT USAGE ON SCHEMA public TO test_extract_role')
        self.conn.execute('GRANT SELECT ON regulations,regulation_versions,regulation_sections TO test_extract_role')
        self.conn.execute('GRANT SELECT,INSERT ON extraction_runs,obligations,extraction_reviews TO test_extract_role')
        self.conn.execute('SET LOCAL ROLE test_extract_role')
        self.assertEqual(run_version(self.conn,self.version,RulesProvider())['candidates'],1)
        with self.assertRaises(psycopg.errors.InsufficientPrivilege):
            self.conn.execute('SELECT * FROM company_profiles')

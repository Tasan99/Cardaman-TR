"""Consolidated corpus behaviour against a real disposable PostgreSQL database."""
import os
import unittest
import json
from contextlib import contextmanager
from unittest.mock import patch
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID

import psycopg

from regchain.extraction.benchmark import measure
from regchain.extraction.coverage import report
from regchain.extraction.providers import RulesProvider
from regchain.extraction.service import run_version
from regchain.ingestion.models import Download
from regchain.ingestion.parse import parse
from regchain.ingestion.repository import save
from test_consolidated import card, page


def handbook_download(html: str, url: str = 'https://handbook.fca.org.uk/handbook/conc7/conc7s3') -> Download:
    return Download(url, url, html.encode(), 'text/html', datetime.now(timezone.utc))


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Requires disposable PostgreSQL + pgvector')
class ConsolidatedDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        for file in sorted((Path(__file__).resolve().parents[1] / 'migrations').glob('*.sql')):
            self.conn.execute(file.read_text(encoding='utf-8'))

    def ingest(self, key: str, html: str, url: str | None = None) -> UUID:
        source = handbook_download(html, url) if url else handbook_download(html)
        return UUID(save(self.conn, key, source, parse(source))['version_id'])

    def test_provision_type_date_and_defined_terms_are_stored(self):
        self.ingest('FCA-HB-CONC-7-3', page(card(
            'CONC 7.3.4', 'R', '04/11/2024',
            '<div class="rule"><p>A <a class="autodeftext" href="/glossary/G430">firm</a> '
            'must treat customers fairly.</p></div>')))
        row = self.conn.execute('''SELECT legal_type,effective_from,defined_terms,source_kind,
                                   printed_label,locator_kind FROM regulation_sections''').fetchone()
        self.assertEqual(row[0], 'RULE')
        self.assertEqual(row[1], date(2024, 11, 4))
        self.assertEqual(row[2], [{'glossary_id': 'G430', 'term': 'firm'}])
        self.assertEqual((row[3], row[4], row[5]), ('CONSOLIDATED', 'CONC 7.3.4', 'handbook_provision'))

    def test_cross_document_candidate_persists_and_cites_its_own_source_url(self):
        support_url='https://handbook.fca.org.uk/handbook/conc7/conc7s1'
        source_version=self.ingest('FCA-HB-CONC-7-1',page(card('CONC 7.1.4','R','01/04/2014',
            '<div class="rule"><p>This chapter applies to lenders.</p></div>')),support_url)
        target=self.ingest('FCA-HB-CONC-7-3',page(card('CONC 7.3.4','R','04/11/2024',
            '<div class="rule"><p>A firm must act under CONC 7.1.4R.</p></div>')))
        class Attributed:
            name='fixture'; model_version='v1'
            def generate_with_context(self,text,context):
                data=json.loads(RulesProvider().generate(text))
                item=context.items[0]
                data['obligations'][0]['supporting_evidence']=[dict(section_id=item['section_id'],role='scope',quote=item['text'])]
                return json.dumps(data)
        result=run_version(self.conn,target,Attributed())
        self.assertEqual(result['candidates'],1)
        self.assertEqual(self.conn.execute('SELECT version_id,source_version_id FROM obligation_context_evidence').fetchone(),(target,source_version))
        from psycopg.rows import dict_row
        from fastapi.testclient import TestClient
        from regchain.api import app
        @contextmanager
        def db():
            previous=self.conn.row_factory
            self.conn.row_factory=dict_row
            try: yield self.conn
            finally: self.conn.row_factory=previous
        with patch('regchain.extraction_api.database',db):
            evidence=TestClient(app).get(f'/versions/{target}/obligations').json()[0]['supporting_evidence'][0]
        self.assertEqual(evidence['source_url'],support_url)
        self.assertEqual(evidence['version_id'],str(source_version))

    def test_changed_runtime_settings_create_a_new_run(self):
        version=self.ingest('fixture',page(card('CONC 7.3.4','R','04/11/2024','<p>A firm must retain records.</p>')))
        class RuntimeRules(RulesProvider):
            thinking=False
            def runtime_manifest(self): return {'thinking':self.thinking}
        provider=RuntimeRules()
        first=run_version(self.conn,version,provider)
        provider.thinking=True
        second=run_version(self.conn,version,provider)
        self.assertNotEqual(first['run_id'],second['run_id'])
        self.assertEqual(run_version(self.conn,version,provider)['status'],'UNCHANGED')

    def test_unpinned_cross_document_evidence_is_rejected(self):
        version=self.ingest('first',page(card('CONC 7.3.4','R','04/11/2024','<p>A firm must retain records.</p>')))
        run_version(self.conn,version,RulesProvider())
        obligation=self.conn.execute('SELECT id FROM obligations').fetchone()[0]
        other=self.ingest('later',page(card('CONC 7.1.4','R','04/11/2024','<p>This chapter applies to lenders.</p>')))
        source=self.conn.execute('SELECT id,text,content_hash FROM regulation_sections WHERE version_id=%s',(other,)).fetchone()
        with self.assertRaisesRegex(psycopg.errors.RaiseException,'pinned'):
            self.conn.execute('''INSERT INTO obligation_context_evidence(obligation_id,version_id,source_version_id,section_id,role,quote,source_hash)
                VALUES(%s,%s,%s,%s,'scope',%s,%s)''',(obligation,version,other,source[0],source[1],source[2]))

    def test_run_pins_every_corpus_version_it_could_resolve_against(self):
        self.ingest('FCA-HB-CONC-7-1', page(card(
            'CONC 7.1.4', 'R', '01/04/2014', '<div class="rule"><p>This chapter applies to lenders.</p></div>')),
            'https://handbook.fca.org.uk/handbook/conc7/conc7s1')
        version = self.ingest('FCA-HB-CONC-7-3', page(card(
            'CONC 7.3.4', 'R', '04/11/2024',
            '<div class="rule"><p>A firm must act in accordance with CONC 7.1.4R.</p></div>')))
        result = run_version(self.conn, version, RulesProvider())
        self.assertEqual(result['corpus_versions'], 1)
        pinned = self.conn.execute('''SELECT c.external_key,c.role,c.normalized_hash=v.normalized_hash
            FROM extraction_corpus_sources c JOIN regulation_versions v ON v.id=c.corpus_version_id''').fetchone()
        self.assertEqual((pinned[0], pinned[1], pinned[2]), ('FCA-HB-CONC-7-1', 'CONSOLIDATED', True))

    def test_pinned_corpus_row_must_match_the_stored_version(self):
        version = self.ingest('FCA-HB-CONC-7-3', page(card(
            'CONC 7.3.4', 'R', '04/11/2024', '<div class="rule"><p>A firm must act.</p></div>')))
        run = run_version(self.conn, version, RulesProvider())['run_id']
        with self.assertRaises(psycopg.errors.RaiseException):
            self.conn.execute('''INSERT INTO extraction_corpus_sources(run_id,corpus_version_id,
                external_key,normalized_hash,role) VALUES(%s,%s,'FCA-HB-CONC-7-3',%s,'CONSOLIDATED')''',
                (run, version, 'f' * 64))

    def test_a_larger_corpus_is_a_new_analysis_not_an_unchanged_one(self):
        version = self.ingest('FCA-HB-CONC-7-3', page(card(
            'CONC 7.3.4', 'R', '04/11/2024',
            '<div class="rule"><p>A firm must act in accordance with CONC 7.1.4R.</p></div>')))
        first = run_version(self.conn, version, RulesProvider())
        self.ingest('FCA-HB-CONC-7-1', page(card(
            'CONC 7.1.4', 'R', '01/04/2014', '<div class="rule"><p>This chapter applies to lenders.</p></div>')),
            'https://handbook.fca.org.uk/handbook/conc7/conc7s1')
        second = run_version(self.conn, version, RulesProvider())
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertEqual(second['status'], 'REVIEW_REQUIRED')
        self.assertEqual(self.conn.execute('SELECT count(*) FROM extraction_runs').fetchone()[0], 2)

    def test_coverage_report_names_the_chapter_that_is_still_missing(self):
        self.ingest('FCA-HB-CONC-7-3', page(card(
            'CONC 7.3.4', 'R', '04/11/2024',
            '<div class="rule"><p>A firm must act under MCOB 13.3.1R.</p></div>')))
        worklist = report(self.conn)['missing_chapters']
        self.assertEqual(worklist[0]['chapter'], 'MCOB 13.3')
        self.assertEqual(worklist[0]['external_key'], 'FCA-HB-MCOB-13-3')
        self.assertEqual(worklist[0]['url'], 'https://www.handbook.fca.org.uk/handbook/MCOB/13/3.html')

    def test_coverage_does_not_report_redline_as_an_ingestion_problem(self):
        from test_amendments_postgres import instrument_document, publication_download
        save(self.conn, 'FIXTURE-INSTRUMENT', publication_download(), instrument_document())
        document = next(d for d in report(self.conn)['documents']
                        if d['external_key'] == 'FIXTURE-INSTRUMENT')
        # The four annex paragraphs are counted apart; only the preamble is assessed.
        self.assertEqual(document['paragraph_context']['amendment_not_extracted'], 4)
        self.assertNotIn('missing_source', document['paragraph_context'])

    def test_benchmark_measures_only_consolidated_paragraphs_and_writes_nothing(self):
        self.ingest('FCA-HB-CONC-7-3', page(
            card('CONC 7.3.4', 'R', '04/11/2024',
                 '<div class="rule"><p>A firm must retain records.</p></div>'),
            card('CONC 7.3.5', 'G', '04/11/2024',
                 '<div class="guidance"><p>A firm should review records.</p></div>')))
        result = measure(self.conn, RulesProvider())
        self.assertEqual(result['paragraphs'], {'consolidated': 2, 'context_complete': 2, 'evaluated': 2})
        self.assertEqual(result['outcomes']['CANDIDATE_REQUIRES_LEGAL_REVIEW'], 2)
        self.assertEqual(result['rates']['candidate_rate'], 1.0)
        self.assertEqual(result['rates']['grounding_rejection_rate'], 0.0)
        self.assertEqual(sorted(result['by_legal_type']), ['GUIDANCE', 'RULE'])
        # The baseline is the provider here, so the two must agree with each other.
        self.assertEqual(result['baseline_agreement']['identical'], 2)
        self.assertEqual(result['disagreements'], [])
        # A measurement must never look like a recorded analysis.
        self.assertEqual(self.conn.execute('SELECT count(*) FROM extraction_runs').fetchone()[0], 0)
        self.assertEqual(self.conn.execute('SELECT count(*) FROM obligations').fetchone()[0], 0)

    def test_benchmark_limit_bounds_the_work_without_hiding_the_total(self):
        self.ingest('FCA-HB-CONC-7-3', page(
            card('CONC 7.3.4', 'R', '04/11/2024', '<div class="rule"><p>A firm must retain records.</p></div>'),
            card('CONC 7.3.5', 'G', '04/11/2024', '<div class="guidance"><p>A firm should review records.</p></div>')))
        result = measure(self.conn, RulesProvider(), limit=1)
        self.assertEqual(result['paragraphs'], {'consolidated': 2, 'context_complete': 2, 'evaluated': 1})
        self.assertEqual(sum(result['outcomes'].values()), 1)

    def test_glossary_identifier_case_survives_storage(self):
        self.ingest('FCA-HB-GLOSSARY-G3497p',
                    '<html><body><main><div class="p-card-content"><div class="header">'
                    '<h1>PRA Rulebook</h1></div><div class="details_wrap"><div class="section">'
                    '<p>the rulebook published by the PRA.</p></div></div></div></main></body></html>',
                    'https://handbook.fca.org.uk/glossary/G3497p')
        row = self.conn.execute('SELECT paragraph_number,legal_type FROM regulation_sections').fetchone()
        self.assertEqual(row, ('G3497p', 'DEFINITION'))


if __name__ == '__main__':
    unittest.main()

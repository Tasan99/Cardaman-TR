"""Amendment links against a real disposable PostgreSQL database."""
import os
import unittest
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import psycopg
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from regchain.api import app
from regchain.extraction.amendments import run_version
from regchain.ingestion.models import Document, Download, Paragraph
from regchain.ingestion.parse import parse
from regchain.ingestion.repository import save
from test_consolidated import card, page

PREAMBLE = ('FIXTURE INSTRUMENT Powers exercised A. The FCA makes this instrument. '
            'Commencement C. This instrument comes into force on 4 November 2024.')


def amendment(number: str, text: str, label: str | None) -> Paragraph:
    return Paragraph('Annex D', number, text, ('Annex D',), page=1, locator_kind='pdf_printed',
                     printed_label=label, source_kind='AMENDMENT')


def instrument_document() -> Document:
    """An amendment instrument shaped like a policy statement appendix.

    Built directly rather than parsed: PDF segmentation has its own tests, and
    regulation_sections is immutable, so amendment paragraphs cannot be marked
    after the fact.
    """
    return Document('Fixture policy statement, not FCA law', (
        Paragraph('preamble', '1', PREAMBLE, (), page=1, source_kind='COMMENTARY'),
        amendment('annex-d', 'Annex D Amendments to the Consumer Credit sourcebook (CONC)', None),
        amendment('7.3', '7.3 Treatment of customers in arrears', '7.3'),
        amendment('7.3.4', '7.3.4 R A firm must treat customers with forbearance.', '7.3.4'),
        amendment('7.9.1', '7.9.1 R A firm must report annually.', '7.9.1'),
    ))


def publication_download() -> Download:
    return Download('https://www.fca.org.uk/publication/policy/fixture',
                    'https://www.fca.org.uk/publication/policy/fixture',
                    b'%PDF-fixture-bytes-not-parsed', 'application/pdf', datetime.now(timezone.utc))


def handbook_download(html: str, url: str) -> Download:
    return Download(url, url, html.encode(), 'text/html', datetime.now(timezone.utc))


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Requires disposable PostgreSQL + pgvector')
class AmendmentLinkDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        for file in sorted((Path(__file__).resolve().parents[1] / 'migrations').glob('*.sql')):
            self.conn.execute(file.read_text(encoding='utf-8'))
        self.instrument = self.ingest_instrument()

    def ingest_instrument(self) -> UUID:
        return UUID(save(self.conn, 'FIXTURE-INSTRUMENT', publication_download(),
                         instrument_document())['version_id'])

    def ingest_chapter(self, key: str, url: str, *cards: str) -> UUID:
        source = handbook_download(page(*cards), url)
        return UUID(save(self.conn, key, source, parse(source))['version_id'])

    def conc_7_3(self) -> UUID:
        return self.ingest_chapter(
            'FCA-HB-CONC-7-3', 'https://www.handbook.fca.org.uk/handbook/CONC/7/3.html',
            card('CONC 7.3.4', 'R', '04/11/2024',
                 '<div class="rule"><p>A firm must treat customers with forbearance.</p></div>'))

    def test_link_records_commencement_pinned_corpus_and_agreement(self):
        self.conc_7_3()
        result = run_version(self.conn, self.instrument)
        self.assertEqual(result['status'], 'REVIEW_REQUIRED')
        self.assertEqual(result['commencement_date'], '2024-11-04')
        self.assertEqual(result['annexes'], {'D': 'CONC'})
        self.assertEqual(result['links'], 1)
        self.assertEqual(result['date_agreement']['MATCH'], 1)
        row = self.conn.execute('''SELECT l.locator,l.date_agreement,l.commencement_date,
            l.consolidated_effective_from,l.human_review_required,s.text
            FROM amendment_links l JOIN regulation_sections s ON s.id=l.consolidated_section_id''').fetchone()
        self.assertEqual(row[0], 'CONC 7.3.4')
        self.assertEqual((row[1], row[2], row[3]), ('MATCH', date(2024, 11, 4), date(2024, 11, 4)))
        self.assertTrue(row[4])
        self.assertEqual(row[5], 'A firm must treat customers with forbearance.')
        self.assertEqual(self.conn.execute(
            "SELECT corpus->0->>'external_key' FROM amendment_runs").fetchone()[0], 'FCA-HB-CONC-7-3')

    def test_unlinked_amendment_paragraph_is_recorded_not_dropped(self):
        self.conc_7_3()
        run_version(self.conn, self.instrument)
        gaps = dict(self.conn.execute('SELECT reason,count(*) FROM amendment_link_gaps GROUP BY reason').fetchall())
        self.assertEqual(gaps.get('NOT_IN_CONSOLIDATED_CORPUS'), 1)
        self.assertEqual(self.conn.execute(
            "SELECT locator FROM amendment_link_gaps WHERE reason='NOT_IN_CONSOLIDATED_CORPUS'").fetchone()[0],
            'CONC 7.9.1')

    def test_superseded_provision_is_flagged_rather_than_presented_as_current(self):
        self.ingest_chapter('FCA-HB-CONC-7-3', 'https://www.handbook.fca.org.uk/handbook/CONC/7/3.html',
                            card('CONC 7.3.4', 'R', '26/06/2026',
                                 '<div class="rule"><p>A firm must treat customers with forbearance.</p></div>'))
        result = run_version(self.conn, self.instrument)
        self.assertEqual(result['date_agreement']['SUPERSEDED'], 1)

    def test_commentary_cannot_be_recorded_as_an_amended_provision(self):
        consolidated = self.conc_7_3()
        run = run_version(self.conn, self.instrument)['run_id']
        commentary = self.conn.execute(
            "SELECT id FROM regulation_sections WHERE version_id=%s AND source_kind='COMMENTARY' LIMIT 1",
            (self.instrument,)).fetchone()[0]
        provision = self.conn.execute(
            'SELECT id FROM regulation_sections WHERE version_id=%s LIMIT 1', (consolidated,)).fetchone()[0]
        with self.assertRaises(psycopg.errors.RaiseException):
            self.conn.execute('''INSERT INTO amendment_links(run_id,instrument_version_id,
                instrument_section_id,consolidated_version_id,consolidated_section_id,locator,
                match_basis,date_agreement) VALUES(%s,%s,%s,%s,%s,'CONC 7.3.4','PRINTED_LOCATOR','UNKNOWN')''',
                (run, self.instrument, commentary, consolidated, provision))

    def test_identical_input_is_idempotent_and_a_new_corpus_is_a_new_run(self):
        self.conc_7_3()
        first = run_version(self.conn, self.instrument)
        self.assertEqual(run_version(self.conn, self.instrument)['status'], 'UNCHANGED')
        self.ingest_chapter('FCA-HB-CONC-7-9', 'https://www.handbook.fca.org.uk/handbook/CONC/7/9.html',
                            card('CONC 7.9.1', 'R', '04/11/2024',
                                 '<div class="rule"><p>A firm must report annually.</p></div>'))
        second = run_version(self.conn, self.instrument)
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertEqual(second['links'], 2)

    def test_history_cannot_be_rewritten(self):
        self.conc_7_3()
        run_version(self.conn, self.instrument)
        # A rejected statement aborts its transaction, so each attempt needs its own
        # savepoint; otherwise the second one only reports the first one's failure.
        for statement in ("UPDATE amendment_links SET date_agreement='MATCH'",
                          'DELETE FROM amendment_links',
                          'DELETE FROM amendment_runs',
                          "UPDATE amendment_link_gaps SET reason='NOT_IN_CONSOLIDATED_CORPUS'"):
            with self.subTest(statement=statement), self.assertRaises(psycopg.errors.RaiseException):
                with self.conn.transaction():
                    self.conn.execute(statement)

    def test_read_api_returns_links_with_both_citations(self):
        self.conc_7_3()
        run = run_version(self.conn, self.instrument)['run_id']

        @contextmanager
        def database():
            previous = self.conn.row_factory
            self.conn.row_factory = dict_row
            try:
                yield self.conn
            finally:
                self.conn.row_factory = previous

        with patch('regchain.extraction_api.database', database):
            client = TestClient(app)
            runs = client.get(f'/versions/{self.instrument}/amendment-runs')
            self.assertEqual(runs.status_code, 200)
            self.assertEqual(runs.json()[0]['commencement_date'], '2024-11-04')
            links = client.get(f'/amendment-runs/{run}/links')
            self.assertEqual(links.status_code, 200)
            link = links.json()[0]
            self.assertEqual(link['locator'], 'CONC 7.3.4')
            self.assertEqual(link['instrument']['source_kind'], 'AMENDMENT')
            self.assertEqual(link['consolidated']['source_kind'], 'CONSOLIDATED')
            self.assertEqual(link['consolidated']['printed_label'], 'CONC 7.3.4')
            self.assertEqual(client.get(f'/amendment-runs/{run}/links?date_agreement=SUPERSEDED').json(), [])
            gaps = client.get(f'/amendment-runs/{run}/gaps')
            self.assertEqual(gaps.json()[0]['reason'], 'NOT_IN_CONSOLIDATED_CORPUS')


if __name__ == '__main__':
    unittest.main()

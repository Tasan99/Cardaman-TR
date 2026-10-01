"""Telling a regulator's change from our own parser's change, on a real database."""
import os
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import psycopg
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from regchain.api import app
from regchain.ingestion.compare import PENDING, compare_version
from regchain.ingestion.models import NORMALIZER_VERSION, PARSER_VERSION, Download
from regchain.ingestion.parse import parse
from regchain.ingestion.repository import save
from test_consolidated import card, page

URL = 'https://www.handbook.fca.org.uk/handbook/CONC/7/3.html'
KEY = 'FCA-HB-CONC-7-3'


def handbook(html: str) -> Download:
    return Download(URL, URL, html.encode(), 'text/html', datetime.now(timezone.utc))


def chapter(text: str, *, comment: str = '') -> str:
    return page(card('CONC 7.3.4', 'R', '04/11/2024',
                     f'<div class="rule">{comment}<p>{text}</p></div>'))


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'Requires disposable PostgreSQL + pgvector')
class SourceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ['TEST_DATABASE_URL'])
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        for file in sorted((Path(__file__).resolve().parents[1] / 'migrations').glob('*.sql')):
            self.conn.execute(file.read_text(encoding='utf-8'))

    def simultaneous_change(self, first: str, second: str) -> UUID:
        """Store one version under an older parser, then another under the current one."""
        with patch('regchain.ingestion.repository.PARSER_VERSION', 'fca-parser-v2'):
            source = handbook(first)
            save(self.conn, KEY, source, parse(source))
        source = handbook(second)
        result = save(self.conn, KEY, source, parse(source))
        self.assertEqual(result['change_origin'], 'SOURCE_AND_PARSER_CHANGE')
        self.assertEqual(result['changed_paragraphs'], 0)
        return UUID(result['version_id'])

    def test_a_real_source_edit_survives_holding_the_parser_constant(self):
        version = self.simultaneous_change(chapter('A firm must act.'),
                                           chapter('A firm must act promptly.'))
        result = compare_version(self.conn, version)
        self.assertEqual(result['verdict'], 'SOURCE_CHANGED')
        self.assertEqual(result['changed_paragraphs'], 1)
        self.assertEqual(result['kinds']['MODIFIED'], 1)
        self.assertEqual(result['parser_version'], PARSER_VERSION)
        self.assertNotEqual(result['old_raw_hash'], result['new_raw_hash'])
        row = self.conn.execute('''SELECT kind,old_locator,old_text,diff_text,new_section_id
            FROM source_comparison_changes''').fetchone()
        self.assertEqual(row[0], 'MODIFIED')
        self.assertEqual(row[2], 'A firm must act.')
        self.assertIn('promptly', row[3])
        self.assertIsNotNone(row[4])

    def test_bytes_that_change_nothing_we_read_are_reported_as_parser_only(self):
        # The markup differs, so the archived bytes differ, but the provision does not.
        version = self.simultaneous_change(chapter('A firm must act.'),
                                           chapter('A firm must act.', comment='<!-- cms rebuild -->'))
        result = compare_version(self.conn, version)
        self.assertEqual(result['verdict'], 'SOURCE_UNCHANGED')
        self.assertEqual(result['changed_paragraphs'], 0)
        self.assertNotEqual(result['old_raw_hash'], result['new_raw_hash'])
        self.assertEqual(self.conn.execute('SELECT count(*) FROM source_comparison_changes').fetchone()[0], 0)

    def test_comparison_records_both_hashes_and_the_single_parser_used(self):
        earlier = chapter('A firm must act.')
        version = self.simultaneous_change(earlier, chapter('A firm must act promptly.'))
        compare_version(self.conn, version)
        row = self.conn.execute('''SELECT parser_version,normalizer_version,old_reparsed_hash,
            new_normalized_hash,human_review_required FROM source_comparisons''').fetchone()
        self.assertEqual((row[0], row[1]), (PARSER_VERSION, NORMALIZER_VERSION))
        # The earlier side is hashed as the current parser reads the archived bytes,
        # which is what makes the two sides comparable. It is stored separately from
        # the hash that version carries, because that one came from another parser.
        self.assertEqual(row[2], parse(handbook(earlier)).content_hash)
        self.assertEqual(row[3], self.conn.execute(
            'SELECT normalized_hash FROM regulation_versions WHERE id=%s', (version,)).fetchone()[0])
        self.assertTrue(row[4])

    def test_pending_lists_the_version_until_it_is_compared(self):
        version = self.simultaneous_change(chapter('A firm must act.'),
                                           chapter('A firm must act promptly.'))
        pending = [row[0] for row in self.conn.execute(PENDING, {'parser': PARSER_VERSION, 'normalizer': NORMALIZER_VERSION}).fetchall()]
        self.assertEqual(pending, [version])
        compare_version(self.conn, version)
        self.assertEqual(self.conn.execute(PENDING, {'parser': PARSER_VERSION, 'normalizer': NORMALIZER_VERSION}).fetchall(), [])

    def test_repeating_the_comparison_adds_nothing(self):
        version = self.simultaneous_change(chapter('A firm must act.'),
                                           chapter('A firm must act promptly.'))
        compare_version(self.conn, version)
        self.assertEqual(compare_version(self.conn, version)['status'], 'UNCHANGED')
        self.assertEqual(self.conn.execute('SELECT count(*) FROM source_comparisons').fetchone()[0], 1)
        self.assertEqual(self.conn.execute('SELECT count(*) FROM source_comparison_changes').fetchone()[0], 1)

    def test_a_first_version_has_nothing_to_compare_against(self):
        source = handbook(chapter('A firm must act.'))
        version = UUID(save(self.conn, KEY, source, parse(source))['version_id'])
        with self.assertRaises(ValueError) as error:
            compare_version(self.conn, version)
        self.assertIn('different archived bytes', str(error.exception))

    def test_a_version_from_an_older_parser_must_be_reparsed_first(self):
        # Holding the parser constant is the whole point; comparing a v2-stored
        # version against v4-reparsed bytes would reintroduce the mixed cause.
        with patch('regchain.ingestion.repository.PARSER_VERSION', 'fca-parser-v2'):
            first = handbook(chapter('A firm must act.'))
            save(self.conn, KEY, first, parse(first))
            second = handbook(chapter('A firm must act promptly.'))
            stale = UUID(save(self.conn, KEY, second, parse(second))['version_id'])
        with self.assertRaises(ValueError) as error:
            compare_version(self.conn, stale)
        self.assertIn('reparse', str(error.exception))
        self.assertEqual(self.conn.execute(PENDING, {'parser': PARSER_VERSION,
                                                     'normalizer': NORMALIZER_VERSION}).fetchall(), [])

    def test_parser_only_reprocessing_in_between_is_skipped(self):
        """The comparison reaches back to the last version whose bytes actually differ."""
        edited = handbook(chapter('A firm must act promptly.'))
        with patch('regchain.ingestion.repository.PARSER_VERSION', 'fca-parser-v2'):
            original = handbook(chapter('A firm must act.'))
            save(self.conn, KEY, original, parse(original))
            save(self.conn, KEY, edited, parse(edited))
        # The same bytes reprocessed by the current parser: no source change of its own.
        latest = UUID(save(self.conn, KEY, edited, parse(edited))['version_id'])
        self.assertEqual(self.conn.execute(
            "SELECT metadata->>'change_origin' FROM regulation_versions WHERE id=%s",
            (latest,)).fetchone()[0], 'PARSER_REPROCESSING')
        result = compare_version(self.conn, latest)
        self.assertEqual(result['verdict'], 'SOURCE_CHANGED')
        self.assertEqual(result['old_raw_hash'], original.raw_hash)

    def test_a_comparison_cannot_be_rewritten(self):
        version = self.simultaneous_change(chapter('A firm must act.'),
                                           chapter('A firm must act promptly.'))
        compare_version(self.conn, version)
        for statement in ("UPDATE source_comparisons SET verdict='SOURCE_UNCHANGED'",
                          'DELETE FROM source_comparison_changes',
                          'DELETE FROM source_comparisons'):
            with self.subTest(statement=statement), self.assertRaises(psycopg.errors.RaiseException):
                with self.conn.transaction():
                    self.conn.execute(statement)

    def test_read_api_exposes_the_verdict_and_the_attributable_changes(self):
        version = self.simultaneous_change(chapter('A firm must act.'),
                                           chapter('A firm must act promptly.'))
        comparison = compare_version(self.conn, version)['comparison_id']

        @contextmanager
        def database():
            previous = self.conn.row_factory
            self.conn.row_factory = dict_row
            try:
                yield self.conn
            finally:
                self.conn.row_factory = previous

        with patch('regchain.regulatory_api.database', database):
            client = TestClient(app)
            listed = client.get(f'/versions/{version}/source-comparisons')
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()[0]['verdict'], 'SOURCE_CHANGED')
            self.assertEqual(listed.json()[0]['parser_version'], PARSER_VERSION)
            changes = client.get(f'/source-comparisons/{comparison}/changes')
            self.assertEqual(changes.status_code, 200)
            self.assertEqual(changes.json()[0]['old_text'], 'A firm must act.')
            self.assertIn('promptly', changes.json()[0]['diff_text'])


if __name__ == '__main__':
    unittest.main()

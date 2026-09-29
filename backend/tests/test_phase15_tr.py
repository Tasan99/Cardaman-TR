"""Türkiye fixes of 23 September 2026: 'reuse' resolves the retained snapshot of the requested
regulation (or downloads and says so), errors name the right source, and the mevzuat parser
no longer reads page-bottom amendment footnotes into the article above them."""
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from regchain.extraction.providers import RulesProvider
from regchain.ingestion.mevzuat import MEVZUAT_PARSER_VERSION, parse_mevzuat, text_url
from regchain.ingestion.models import Download
from regchain.pilot.sources import load_sources, save_sources, select_targets
from regchain.pilot.workspace import Workspace, create_app
from test_turkiye import download, fixture_fetch, law_html, p
from test_workspace import request_input, source_fixture

TR = dict(regulator='TR', mevzuat_kind='1', mevzuat_number='5549', sections=['3', '4'], labels=[])


class FootnoteParserTests(unittest.TestCase):
    def test_page_bottom_amendment_notes_are_not_part_of_the_article(self):
        html = law_html().replace(
            p('(2) Kimlik tespitine esas belge nevilerini belirlemeye Bakanlık yetkilidir.'),
            p('(2) Kimlik tespitine esas belge nevilerini belirlemeye Bakanlık yetkilidir. a) Bankalar. –––––––––––––––– '
              '(1) 29/2/2016 tarihli ve 2016/8566 sayılı Bakanlar Kurulu Kararı Eki Yönetmeliğin 1 inci maddesiyle bu bentte yer alan ibare değiştirilmiştir.')
            + p('(2) 24/2/2021 tarihli ve 31405 sayılı Resmî Gazete’de yayımlanan 3580 sayılı Cumhurbaşkanı Kararının 1 inci maddesiyle ibare eklenmiştir.')
            + p('b) Sigorta şirketleri.'))
        doc = parse_mevzuat(download(html))
        text = next(a.text for a in doc.paragraphs if a.number == '3')
        self.assertIn('a) Bankalar. b) Sigorta şirketleri.', text)
        self.assertNotIn('––––', text)
        self.assertNotIn('değiştirilmiştir', text)
        self.assertNotIn('eklenmiştir', text)
        self.assertEqual(MEVZUAT_PARSER_VERSION, 'mevzuat-parser-v3')          # v3 (v0.16) also reads an underscore rule

    def test_a_snapshot_from_the_earlier_parser_loads_when_its_hash_still_reproduces(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / 'tr'
        save_sources(root, [download()], mevzuat=('1', '5549', '5'))
        snapshot = json.loads((root / 'snapshot.json').read_text(encoding='utf-8'))
        snapshot['sources'][0]['parser_version'] = 'mevzuat-parser-v1'
        (root / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf-8')
        _, sections = load_sources(root)                                    # same hash: the newer parser reads it identically
        self.assertEqual(len(sections), 6)
        snapshot['sources'][0]['parsed_hash'] = 'f' * 64
        (root / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 're-ingest'):
            load_sources(root)


class ErrorWordingTests(unittest.TestCase):
    def test_a_missing_regulation_names_the_right_source_to_download_from(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name) / 'tr'
        save_sources(root, [download()], mevzuat=('1', '5549', '5'))
        _, sections = load_sources(root)
        with self.assertRaises(ValueError) as caught:
            select_targets(sections, 'YONETMELIK', '200713012', 'all')
        self.assertIn("mevzuat.gov.tr'dan güncel kaynağı indir", str(caught.exception))
        self.assertIn('Yönetmelik 200713012', str(caught.exception))
        self.assertIn('Kanun 5549', str(caught.exception))
        with self.assertRaisesRegex(ValueError, "FCA'dan güncel kaynağı indir"):
            select_targets(sections, 'CONC', '7', 'all')


class ReuseTests(unittest.TestCase):
    """Reuse means the snapshot of the regulation being analysed, never whatever folder is around."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        # The launcher's cached folder holds an FCA chapter, as on the user's machine.
        source_fixture(root / 'launcher-sources')
        self.fetches = []

        def fetch(directory, **kwargs):
            self.fetches.append(kwargs.get('mevzuat') or (kwargs.get('module'), kwargs.get('chapter')))
            return fixture_fetch(directory, **kwargs) if kwargs.get('mevzuat') else source_fixture(directory, **kwargs)
        self.workspace = Workspace(root / 'workspace', cached_sources=root / 'launcher-sources',
                                   provider_factory=lambda _: RulesProvider(), source_fetcher=fetch)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                                 headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def finish(self, **changes):
        response = self.client.post('/api/runs', json=request_input(**changes))
        self.assertEqual(response.status_code, 202, response.text)
        run_id = response.json()['id']
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def test_reuse_of_a_regulation_never_retained_downloads_it_and_says_so(self):
        row = self.finish(source_mode='reuse', **TR)
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        self.assertEqual(row['regulation'], 'Kanun 5549')
        self.assertEqual(row['source_mode_effective'], 'refresh')
        self.assertIn("mevzuat.gov.tr'dan güncel kaynak indirildi", row['source_note'])
        self.assertEqual(self.fetches, [('1', '5549', '5')])
        # The next reuse of the same regulation takes that run's retained snapshot, nothing is fetched.
        again = self.finish(source_mode='reuse', **TR)
        self.assertEqual(again['state'], 'COMPLETED', again['error'])
        self.assertEqual(again['source_mode_effective'], 'reuse')
        self.assertIn('Saklanan kaynak', again['source_note'])
        self.assertIn('Kanun 5549', again['source_note'])
        self.assertEqual(len(self.fetches), 1)
        # The launcher's FCA folder still serves an FCA reuse.
        fca = self.finish(source_mode='reuse')
        self.assertEqual(fca['state'], 'COMPLETED', fca['error'])
        self.assertIn('başlatıcının kaynak klasörü', fca['source_note'])
        self.assertEqual(len(self.fetches), 1)

    def test_comparing_with_a_run_of_another_regulation_is_refused_clearly(self):
        fca = self.finish()
        self.assertEqual(fca['state'], 'COMPLETED', fca['error'])
        row = self.finish(source_mode='reuse', previous_id=fca['id'], **TR)
        self.assertEqual(row['state'], 'FAILED')
        self.assertIn('bu düzenlemeye ait değil', row['error'])
        self.assertNotIn('FCA', row['error'])


if __name__ == '__main__':
    unittest.main()


class RealDocumentLessonsTests(unittest.TestCase):
    """Measured on a real bank policy beside a control register (23 September 2026)."""

    def test_running_headers_and_page_numbers_are_stripped_by_the_v2_reader_only(self):
        from unittest.mock import patch
        from pypdf import PdfWriter
        from regchain.pilot.policies import PARSER, read_policy
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'bank.pdf'
        writer = PdfWriter()
        for _ in range(4):
            writer.add_blank_page(300, 300)
        writer.write(path)
        pages = [f'Kurum İçi Sınırsız Kullanım / Kişisel Veri\nMadde {n}. Yükümlüler kimlik tespiti yapmak zorundadır.\n{n}' for n in range(1, 5)]
        with patch('pypdf._page.PageObject.extract_text', side_effect=pages):
            value = read_policy(path)
        self.assertEqual(value['parser'], PARSER)
        self.assertEqual([c['text'] for c in value['chunks']], [f'Madde {n}. Yükümlüler kimlik tespiti yapmak zorundadır.' for n in range(1, 5)])
        with patch('pypdf._page.PageObject.extract_text', side_effect=pages):
            old = read_policy(path, parser='pilot-policy-v1')
        self.assertEqual(old['parser'], 'pilot-policy-v1')
        self.assertTrue(old['chunks'][0]['text'].startswith('Kurum İçi Sınırsız Kullanım'))
        # A line that opens only two pages is content, not a header.
        pages = ['Amaç\nMadde 1.', 'Amaç\nMadde 2.', 'Kapsam\nMadde 3.', 'Diğer\nMadde 4.']
        with patch('pypdf._page.PageObject.extract_text', side_effect=pages):
            value = read_policy(path)
        self.assertEqual(value['chunks'][0]['text'], 'Amaç\nMadde 1.')

    def test_the_best_control_rows_are_judged_beyond_the_ranked_window(self):
        from regchain.pilot.engine import CONTROL_WINDOW, JUDGE_ALL_UNDER, JUDGE_WINDOW, analyze
        from test_pilot import FixtureProvider, company, policies, sections
        base = policies()[0]
        texts = [f'Staff must retain records in archive {i}.' for i in range(JUDGE_ALL_UNDER + 5)]
        policy = dict(base, chunks=[dict(base['chunks'][0], source_id=f'c{i:03d}', text=text) for i, text in enumerate(texts)])
        rows = [dict(base['chunks'][0], source_id=f'k{i:03d}', locator='control_row', number=i + 1,
                     text=f'Kontrol No: R-{i} — Açıklama: records are retained for {i} years') for i in range(10)]
        register = dict(base, filename='controls.csv', raw_hash='e' * 64, chunks=rows)
        row = analyze(company(), [policy, register], sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        judged = row['judged_policy_ids']
        self.assertGreaterEqual(len(judged), JUDGE_WINDOW)
        self.assertLessEqual(len(judged), JUDGE_WINDOW + CONTROL_WINDOW)
        self.assertGreaterEqual(sum(1 for s in judged if s.startswith('k')), 1)
        self.assertEqual(len(set(judged)), len(judged))

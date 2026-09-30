"""The official-source corpus: fetch, version and verify (no network; the packaged texts are read from disk)."""
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from regchain.ingestion.mevzuat import text_url
from regchain.ingestion.models import Download, IngestionError
from regchain.tr.adapters import institution_of
from regchain.tr.core import RegulationMeta, SourceRef
from regchain.tr.corpus import (CORPUS, CorpusError, CorpusStore, catalogue_record, download_text, gazette_date, provision_labels,
                                refresh, same_title, verified_fields, verify_catalogue)
from regchain.tr.packs import Registry

REGISTRY = Registry.load()
STORE = CorpusStore()
REF = SourceRef(adapter='mevzuat', kind='9', number='99001')
URL = text_url('9', '99001')


def page(limit: str = '150') -> bytes:
    """A synthetic regulation shaped like the served Word export; a parser fixture, not law."""
    return ('<html><body><table><tr><td>'
            '<p align=center><b>SENTETİK ÖRNEK İÇECEK TEBLİĞİ</b></p>'
            '<p><b>Kapsam</b></p><p><b>MADDE 1 –</b> (1) Bu Tebliğ örnek içecekleri kapsar.</p>'
            f'<p><b>Ürün özellikleri</b></p><p><b>MADDE 2 –</b> (1) Örnek içecekte kafein {limit} mg/L’den fazla olamaz.</p>'
            '<p><b>Yürürlük</b></p><p><b>MADDE 3 –</b> (1) Bu Tebliğ yayımı tarihinde yürürlüğe girer.</p>'
            '</td></tr></table></body></html>').encode('utf-8')


def download(body: bytes, url: str = URL, when=datetime(2026, 9, 30, 12, tzinfo=timezone.utc)) -> Download:
    return Download(url, url, body, 'text/html', when)


def meta(**overrides) -> RegulationMeta:
    base = dict(regulation_id='TR:TEBLIG:SENTETIK', title='Sentetik Örnek İçecek Tebliği', jurisdiction='TR',
                regulator='TARIM_ORMAN', regulation_type='TEBLIG', binding_status='BINDING', number=None,
                effective_date=None, effective_status='UNKNOWN', metadata_status='UNVERIFIED',
                verification_note='synthetic test entry', sector_tags=['BEVERAGE'], source_ref=REF.model_dump())
    return RegulationMeta.model_validate(base | overrides)


def search(query, family, limit):
    return [{'kind': '9', 'number': '99001', 'tertip': '5', 'title': 'SENTETİK ÖRNEK İÇECEK TEBLİĞİ', 'kind_name': 'Tebliğ',
             'gazette_date': '30.06.2017', 'gazette_number': '30110'}]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = CorpusStore(Path(self.directory.name))
        self.addCleanup(self.directory.cleanup)

    def fetcher(self, body):
        return lambda url: download(body, url)

    def test_a_fetch_is_stored_as_a_version_with_bytes_hashes_and_the_catalogue_record(self):
        report = refresh([meta()], self.store, fetcher=self.fetcher(page()), search=search, pause=0)
        self.assertEqual([r['status'] for r in report], ['NEW_VERSION'])
        version = self.store.head('TR:TEBLIG:SENTETIK')
        self.assertEqual((version.articles, version.previous_version_id, version.synthetic), (3, None, False))
        self.assertEqual(version.publication_date, date(2017, 6, 30))
        self.assertEqual(version.effective_date, date(2017, 6, 30))          # "yayımı tarihinde yürürlüğe girer"
        self.assertEqual(version.catalogue.gazette_number, '30110')
        raw = self.store.raw_path(version)
        self.assertEqual(raw.read_bytes(), page())
        sections = self.store.sections('TR:TEBLIG:SENTETIK')
        self.assertEqual([s['printed_label'] for s in sections], ['Tebliğ 99001 md. 1', 'Tebliğ 99001 md. 2', 'Tebliğ 99001 md. 3'])
        self.assertEqual(sections[1]['text'][sections[1]['lines'][0]['start']:sections[1]['lines'][0]['end']], sections[1]['text'])

    def test_the_same_text_again_is_unchanged_and_a_changed_text_is_the_next_version(self):
        refresh([meta()], self.store, fetcher=self.fetcher(page()), search=search, pause=0)
        again = refresh([meta()], self.store, fetcher=self.fetcher(page()), search=search, pause=0)
        self.assertEqual(again[0]['status'], 'UNCHANGED')
        self.assertEqual(len(self.store.versions('TR:TEBLIG:SENTETIK')), 1)
        changed = refresh([meta()], self.store, fetcher=lambda url: download(page('120'), url, datetime(2026, 10, 2, tzinfo=timezone.utc)),
                          search=search, pause=0)
        self.assertEqual(changed[0]['status'], 'NEW_VERSION')
        first, second = self.store.versions('TR:TEBLIG:SENTETIK')
        self.assertEqual(second.previous_version_id, first.version_id)
        self.assertNotEqual(second.parsed_hash, first.parsed_hash)
        self.assertIn('120 mg/L', self.store.section('TR:TEBLIG:SENTETIK', 'Tebliğ 99001 md. 2')['text'])
        self.assertIn('150 mg/L', self.store.section('TR:TEBLIG:SENTETIK', 'Tebliğ 99001 md. 2', first.version_id)['text'])

    def test_re_rendered_bytes_with_the_same_text_are_not_a_new_version(self):
        refresh([meta()], self.store, fetcher=self.fetcher(page()), search=search, pause=0)
        rendered = page().replace(b'<html>', b'<html><!-- rendered again -->')
        self.assertEqual(refresh([meta()], self.store, fetcher=self.fetcher(rendered), search=search, pause=0)[0]['status'], 'UNCHANGED')

    def test_a_failed_fetch_is_reported_and_nothing_is_stored(self):
        def failing(url):
            raise IngestionError('Source returned HTTP 503; retry later')
        report = refresh([meta()], self.store, fetcher=failing, search=search, pause=0)
        self.assertEqual(report[0]['status'], 'FAILED')
        self.assertIn('503', report[0]['error'])
        self.assertEqual(self.store.versions('TR:TEBLIG:SENTETIK'), [])

    def test_a_text_that_answers_for_another_record_is_refused(self):
        other = text_url('9', '99002')
        with self.assertRaises(CorpusError):
            download_text(REF, lambda url: download(page(), other))

    def test_only_an_allowlisted_https_source_can_be_named(self):
        with self.assertRaises(ValueError):
            download_text(SourceRef(adapter='mevzuat', kind='99', number='1'), self.fetcher(page()))
        self.assertEqual(institution_of(URL).institution_id, 'MEVZUAT')

    def test_a_tampered_stored_text_is_refused_on_load(self):
        refresh([meta()], self.store, fetcher=self.fetcher(page()), search=search, pause=0)
        version = self.store.head('TR:TEBLIG:SENTETIK')
        raw = self.store.raw_path(version)
        raw.write_bytes(page('999'))
        with self.assertRaises(ValueError):
            CorpusStore(Path(self.directory.name)).sections('TR:TEBLIG:SENTETIK')

    def test_a_synthetic_version_is_flagged_and_never_enters_the_packaged_corpus(self):
        source, document = download_text(REF, self.fetcher(page()))
        status, version = self.store.record('TR:TEBLIG:SENTETIK', REF, source, document, synthetic=True)
        self.assertTrue(version.synthetic)
        with self.assertRaises(CorpusError):
            CorpusStore(CORPUS).record('TR:TEBLIG:SENTETIK', REF, source, document, synthetic=True)

    def test_verification_needs_the_printed_title_and_the_stored_hash(self):
        refresh([meta()], self.store, fetcher=self.fetcher(page()), search=search, pause=0)
        fields = verified_fields(meta(), self.store)
        self.assertTrue(fields['title_matches'])
        verified = meta(metadata_status='VERIFIED', source_url=fields['source_url'], versions=fields['versions'],
                        gazette_date=fields['gazette_date'], effective_date=fields['effective_date'], effective_status='IN_FORCE')
        self.assertEqual(verify_catalogue([verified], self.store), [])
        wrong_title = meta(title='Başka Bir Tebliğ', metadata_status='VERIFIED', source_url=fields['source_url'],
                           versions=fields['versions'], gazette_date=fields['gazette_date'])
        self.assertTrue(any('printed title' in p for p in verify_catalogue([wrong_title], self.store)))
        typed_hash = [fields['versions'][0] | {'source_hash': 'a' * 64}]
        self.assertTrue(any('stored versions' in p for p in verify_catalogue(
            [meta(metadata_status='VERIFIED', source_url=fields['source_url'], versions=typed_hash,
                  gazette_date=fields['gazette_date'])], self.store)))

    def test_a_verified_entry_without_source_or_hash_is_not_a_valid_catalogue_entry(self):
        with self.assertRaises(ValueError):
            meta(metadata_status='VERIFIED')


class HelperTests(unittest.TestCase):
    def test_titles_are_compared_in_turkish_case(self):
        self.assertTrue(same_title('TÜRK GIDA KODEKSİ BİRA TEBLİĞİ (TEBLİĞ NO: 2006/33)', 'Türk Gıda Kodeksi Bira Tebliği (Tebliğ No: 2006/33)'))
        self.assertTrue(same_title('ISPARTA İLİ', 'Isparta ili'))
        self.assertFalse(same_title('Bira Tebliği', 'Şarap Tebliği'))

    def test_gazette_dates_and_provision_references(self):
        self.assertEqual(gazette_date('07.01.2011'), date(2011, 1, 7))
        self.assertIsNone(gazette_date(''))
        self.assertEqual(provision_labels('Kanun 4250 md. 6/f.1; Kanun 4250 md. 6/f.2/c.1; Kanun 4250 md. 9'),
                         ['Kanun 4250 md. 6', 'Kanun 4250 md. 9'])

    def test_the_catalogue_record_is_matched_by_kind_and_number_not_by_title_alone(self):
        self.assertEqual(catalogue_record(REF, ['Sentetik Örnek İçecek Tebliği'], search).gazette_date, date(2017, 6, 30))
        other = lambda q, f, n: [{'kind': '9', 'number': '1', 'tertip': '5', 'title': 'SENTETİK ÖRNEK İÇECEK TEBLİĞİ'}]
        self.assertIsNone(catalogue_record(REF, ['Sentetik Örnek İçecek Tebliği'], other))


class PackagedCorpusTests(unittest.TestCase):
    """The texts shipped with the package were fetched from mevzuat.gov.tr; these checks never touch the network."""

    def test_every_catalogued_beverage_regulation_has_a_stored_official_text(self):
        beverage = [m for m in REGISTRY.regulations.values() if m.source_ref is not None]
        self.assertGreaterEqual(len(beverage), 20)
        for regulation in beverage:
            with self.subTest(regulation.regulation_id):
                head = STORE.head(regulation.regulation_id)
                self.assertIsNotNone(head)
                self.assertFalse(head.synthetic)
                self.assertEqual(institution_of(head.url).institution_id, 'MEVZUAT')
                self.assertEqual(STORE.official_source(regulation.regulation_id, regulation).authority, 'BINDING')

    def test_every_stored_text_reproduces_its_hashes(self):
        for regulation_id in STORE.regulation_ids():
            with self.subTest(regulation_id):
                head = STORE.head(regulation_id)
                sections = STORE.sections(regulation_id)          # load_sources re-hashes the bytes and re-parses them
                self.assertEqual(len(sections), head.articles)
                self.assertEqual(sections[0]['version_hash'], head.parsed_hash)

    def test_the_catalogue_and_the_scopes_claim_nothing_the_corpus_does_not_support(self):
        self.assertEqual(verify_catalogue(REGISTRY.regulations.values(), STORE, REGISTRY.scopes.values()), [])

    def test_no_beverage_catalogue_entry_is_left_unverified(self):
        unverified = [m.regulation_id for m in REGISTRY.regulations.values()
                      if m.source_ref is not None and m.metadata_status != 'VERIFIED']
        self.assertEqual(unverified, [])

    def test_the_reenacted_alcohol_article_is_in_force_with_its_eleven_paragraphs(self):
        article = STORE.section('TR:KANUN:4250', 'Kanun 4250 md. 6')
        self.assertNotIn('DELETED_PROVISION', article['quality_flags'])
        lines = [article['text'][s['start']:s['end']] for s in article['lines']]
        self.assertEqual(len(lines), 12)                                  # the enactment note, then eleven fıkra
        self.assertTrue(lines[1].startswith('Alkollü içkilerin her ne surette olursa olsun reklamı'))
        self.assertIn('22:00 ila 06:00', lines[5])

    def test_the_wrapped_food_codex_texts_are_read(self):
        energy = STORE.section('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 5')
        self.assertIn('toplam kafein miktarı 150 mg/L’den fazla olamaz', energy['text'])
        self.assertGreaterEqual(len(STORE.sections('TR:YONETMELIK:TGK_ETIKETLEME')), 50)


if __name__ == '__main__':
    unittest.main()

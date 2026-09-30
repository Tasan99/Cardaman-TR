"""mevzuat-parser-v4: the wrapped-text rules the TR corpus adapter reads official texts with.

The HTML below is synthetic (shaped like the Word export mevzuat.gov.tr serves); it is a parser fixture, not law.
v3 stays the default parser, so the retained AML snapshots are read exactly as before.
"""
import unittest
from datetime import datetime, timezone

from regchain.ingestion.mevzuat import (MEVZUAT_PARSER_VERSION, WRAPPED_TEXT_VERSION, in_force_text, parse_mevzuat, text_url)
from regchain.ingestion.models import Download, IngestionError

URL = text_url('9', '99001')


def download(html: str) -> Download:
    return Download(URL, URL, html.encode('utf-8'), 'text/html', datetime(2026, 9, 30, tzinfo=timezone.utc))


def page(body: str) -> str:
    return f'<html><body><div class=WordSection1>{body}</div></body></html>'


ARTICLES = ('<p align=center><b>SENTETİK ÖRNEK TEBLİĞİ</b></p>'
            '<p><b>Kapsam</b></p><p><b>MADDE 1 –</b> (1) Bu Tebliğ örnek içecekleri kapsar.</p>'
            '<p><b>Ürün özellikleri</b></p><p><b>MADDE 2 –</b> (1) Örnek içecekte kafein 150 mg/L’den fazla olamaz.</p>'
            '<p>a) Birinci bent.</p><p>b) İkinci bent.</p>'
            '<table><tr><td><p>Çizelge hücresi: MADDE dışı veri</p></td></tr></table>'
            '<p>(2) İkinci fıkra.</p>')
WRAPPED = page(f'<table><tr><td><table><tr><td>{ARTICLES}</td></tr></table></td></tr></table>')
PLAIN = page(ARTICLES)


class WrappedTextTests(unittest.TestCase):
    def test_v3_is_still_the_default_and_finds_nothing_inside_a_layout_table(self):
        self.assertEqual(MEVZUAT_PARSER_VERSION, 'mevzuat-parser-v3')
        with self.assertRaises(IngestionError):
            parse_mevzuat(download(WRAPPED))

    def test_v4_reads_the_text_a_layout_table_wraps_and_still_drops_the_data_table(self):
        document = parse_mevzuat(download(WRAPPED), WRAPPED_TEXT_VERSION)
        self.assertEqual([p.printed_label for p in document.paragraphs], ['Tebliğ 99001 md. 1', 'Tebliğ 99001 md. 2'])
        self.assertEqual(document.title, 'SENTETİK ÖRNEK TEBLİĞİ')
        second = document.paragraphs[1]
        self.assertIn('150 mg/L', second.text)
        self.assertIn('(2) İkinci fıkra.', second.text)
        self.assertNotIn('Çizelge hücresi', second.text)

    def test_v4_reads_an_unwrapped_text_exactly_as_v3_does(self):
        old, new = parse_mevzuat(download(PLAIN)), parse_mevzuat(download(PLAIN), WRAPPED_TEXT_VERSION)
        self.assertEqual([(p.printed_label, p.text, p.quality_flags) for p in old.paragraphs],
                         [(p.printed_label, p.text, p.quality_flags) for p in new.paragraphs])

    def test_v4_records_where_each_source_paragraph_sits_and_v3_records_nothing(self):
        second = parse_mevzuat(download(PLAIN), WRAPPED_TEXT_VERSION).paragraphs[1]
        lines = [second.text[s['start']:s['end']] for s in second.source_spans]
        self.assertEqual(lines, ['(1) Örnek içecekte kafein 150 mg/L’den fazla olamaz.', 'a) Birinci bent.', 'b) İkinci bent.',
                                 '(2) İkinci fıkra.'])
        self.assertEqual(' '.join(lines), second.text)
        self.assertEqual(parse_mevzuat(download(PLAIN)).paragraphs[1].source_spans, ())


class ReenactedArticleTests(unittest.TestCase):
    BODY = page('<p align=center><b>SENTETİK ÖRNEK TEBLİĞİ</b></p>'
                '<p><b>MADDE 1 –</b> Bu Tebliğ örnek içecekleri kapsar.</p>'
                '<p><b>MADDE 2 –</b> (Mülga: 11/1/2001-4619/5 md.; Yeniden düzenleme: 24/5/2013-6487/2 md.)</p>'
                '<p>Örnek içeceklerin reklamı yapılamaz.</p><p>Örnek içecekler otomatik makineler ile satılamaz.</p>'
                '<p><b>MADDE 3 –</b> (Mülga: 27/3/1969-1137/37-1 md.)</p>'
                '<p><b>MADDE 4 –</b> (Mülga birinci fıkra: 1/1/2020-7000/1 md.)</p><p>İkinci fıkra yürürlüktedir.</p>'
                '<p><b>GEÇİCİ MADDE 1 –</b> Birinci geçici hüküm.</p>'
                '<p><b>GEÇİCİ MADDE 1 –</b> İşlenemeyen geçici hüküm.</p>')

    def test_a_repeal_note_followed_by_text_is_in_force(self):
        self.assertTrue(in_force_text('(Mülga: 11/1/2001-4619/5 md.; Yeniden düzenleme: 24/5/2013-6487/2 md.) Reklamı yapılamaz.'))
        self.assertTrue(in_force_text('(Mülga birinci fıkra: 1/1/2020-7000/1 md.) İkinci fıkra yürürlüktedir.'))
        self.assertFalse(in_force_text('(Mülga: 27/3/1969-1137/37-1 md.)'))
        self.assertFalse(in_force_text('(Mülga: 20/11/2017-KHK-696/82 md.; Aynen kabul: 1/2/2018-7079/77 md.)'))
        self.assertFalse(in_force_text('(Mülga: 1/1/2001-1/1 md.) (Mülga: 2/2/2002-2/2 md.)'))

    def test_v4_keeps_a_reenacted_article_and_v3_still_reads_it_as_repealed(self):
        flags = lambda version: {p.number: 'DELETED_PROVISION' in p.quality_flags
                                 for p in parse_mevzuat(download(self.BODY), version).paragraphs}
        old, new = flags('mevzuat-parser-v3'), flags(WRAPPED_TEXT_VERSION)
        self.assertEqual((old['2'], old['3'], old['4']), (True, True, True))
        self.assertEqual((new['2'], new['3'], new['4']), (False, True, False))

    def test_v4_gives_a_repeated_article_number_its_own_label(self):
        document = parse_mevzuat(download(self.BODY), WRAPPED_TEXT_VERSION)
        labels = [p.printed_label for p in document.paragraphs]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn('Tebliğ 99001 md. Geçici 1 (mükerrer 2)', labels)
        self.assertIn('Duplicate article number Geçici 1', document.warnings)
        repeated = [p.printed_label for p in parse_mevzuat(download(self.BODY)).paragraphs]
        self.assertEqual(repeated.count('Tebliğ 99001 md. Geçici 1'), 2)          # v3: unchanged behaviour

    def test_the_unnumbered_paragraphs_of_a_reenacted_article_are_separate_lines(self):
        article = next(p for p in parse_mevzuat(download(self.BODY), WRAPPED_TEXT_VERSION).paragraphs if p.number == '2')
        lines = [article.text[s['start']:s['end']] for s in article.source_spans]
        self.assertEqual(lines[1:], ['Örnek içeceklerin reklamı yapılamaz.', 'Örnek içecekler otomatik makineler ile satılamaz.'])


if __name__ == '__main__':
    unittest.main()

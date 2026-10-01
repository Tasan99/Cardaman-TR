"""v0.18 Phase Q: document quality states (text_pdf, scanned_pdf, mixed_pdf, low_text_density,
corrupted, text), the OCR recommendation beside the requires_ocr hard stop, and the refusal of a
corrupted or empty policy instead of an analysis of nothing."""
import base64
import io
import itertools
import logging
import random
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.errors import PdfReadError
from regchain.evidence import canonical_bytes
from regchain.pilot.policies import document_quality, read_policy
from regchain.pilot.workspace import Workspace, create_app
from test_pilot import FixtureProvider
from test_workspace import request_input, source_fixture

LONG = 'Kimlik tespiti işlem yapılmadan önce tamamlanır. ' * 10


def articles(count):
    # Distinct pages: one line repeated on three or more pages is a running header and is stripped.
    return [f'Madde {n}. {LONG}' for n in range(1, count + 1)]


def parts(texts):
    return [(i + 1, t, 'pdf_page') for i, t in enumerate(texts)]


def empties(texts):
    return [i + 1 for i, t in enumerate(texts) if not t.strip()]


def blank_pdf(pages):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(300, 300)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def page_texts(texts):
    """extract_text patched to return the pages' texts in order, again on every later read (the
    workspace re-reads a retained policy when it verifies the packet)."""
    cycle = itertools.cycle(texts)
    return patch('pypdf._page.PageObject.extract_text', side_effect=lambda *_, **__: next(cycle))


def docx(paragraphs):
    ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    body = ''.join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('word/document.xml', f'<w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>')
    return buffer.getvalue()


class DocumentStateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        # pypdf warns about the broken bytes these tests hand it; the test output needs none of it.
        logger = logging.getLogger('pypdf')
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel(logging.ERROR)

    def file(self, name, content):
        path = self.root / name
        path.write_bytes(content if isinstance(content, bytes) else content.encode('utf-8'))
        return path

    def test_each_pdf_state_is_named_and_only_a_scanned_pdf_requires_ocr(self):
        cases = [([LONG] * 4, 'text', 'text_pdf', False), ([''] + [LONG] * 4, 'mixed', 'mixed_pdf', True),
                 (['', '', LONG], 'scanned', 'scanned_pdf', True), (['kısa metin'] * 3, 'low_density', 'low_text_density', True)]
        for texts, kind, state, recommended in cases:
            with self.subTest(state=state):
                quality = document_quality(parts(texts), empties(texts))
                self.assertEqual((quality['kind'], quality['state']), (kind, state))       # v0.17 kind unchanged
                self.assertEqual(quality['requires_ocr'], state == 'scanned_pdf')           # the hard stop only
                self.assertEqual(quality['ocr_recommended'], recommended)
                self.assertTrue(quality['quality_note'].strip())
                canonical_bytes(quality)                                                    # packet-safe: no floats

    def test_mixed_and_thin_pdfs_are_read_with_ocr_recommended_and_a_scanned_pdf_is_refused(self):
        path = self.file('politika.pdf', blank_pdf(5))
        with page_texts([''] + articles(4)):
            quality = read_policy(path)['document_quality']
        self.assertEqual((quality['state'], quality['ocr_recommended'], quality['requires_ocr']), ('mixed_pdf', True, False))
        self.assertIn('1/5', quality['quality_note'])
        with page_texts([f'Madde {n}. Kısa metin.' for n in range(1, 6)]):
            quality = read_policy(path)['document_quality']
        self.assertEqual((quality['state'], quality['ocr_recommended'], quality['requires_ocr']), ('low_text_density', True, False))
        with page_texts(['', '', '', '', LONG]):
            with self.assertRaisesRegex(ValueError, '^requires_ocr:'):
                read_policy(path)
        with page_texts(articles(5)):
            quality = read_policy(path)['document_quality']
        self.assertEqual((quality['state'], quality['ocr_recommended'], quality['requires_ocr']), ('text_pdf', False, False))

    def test_a_pdf_that_cannot_be_opened_is_refused_as_corrupted(self):
        whole = blank_pdf(2)
        for name, content in [('rastgele.pdf', random.Random(18).randbytes(4096)), ('yarim.pdf', whole[:len(whole) // 2]),
                              ('bos.pdf', b'')]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f'^corrupted: "{name}"'):
                    read_policy(self.file(name, content))

    def test_a_pdf_whose_pages_fail_to_extract_is_corrupted_but_one_failed_page_is_counted(self):
        path = self.file('hasarli.pdf', blank_pdf(4))
        with patch('pypdf._page.PageObject.extract_text', side_effect=PdfReadError('bad content stream')):
            with self.assertRaisesRegex(ValueError, r'^corrupted: .*4/4 sayfanın metni çıkarılamadı'):
                read_policy(path)
        with patch('pypdf._page.PageObject.extract_text', side_effect=[*articles(1), KeyError('/Contents'), *articles(2)]):
            value = read_policy(path)
        quality = value['document_quality']
        self.assertEqual((quality['state'], quality['failed_pages'], quality['empty_pages']), ('mixed_pdf', 1, 1))
        self.assertEqual(value['empty_pages'], [2])
        self.assertEqual({c['number'] for c in value['chunks']}, {1, 3, 4})
        quality = document_quality(parts([''] * 3 + [LONG]), [1, 2, 3], failed=[1, 2, 3])
        self.assertEqual((quality['state'], quality['requires_ocr']), ('corrupted', False))

    def test_text_that_is_mostly_not_letters_is_corrupted_and_never_quoted_in_the_refusal(self):
        garbage = 'Gizli müşteri bilgisi ' + '�' * 300 + '\x01\x02\x03' * 50
        quality = document_quality(parts([garbage] * 3), [])
        self.assertEqual((quality['kind'], quality['state']), ('text', 'corrupted'))
        self.assertLess(quality['letter_percent'], 30)
        for name, content in [('bozuk.txt', garbage), ('bozuk.md', '\x00\x07' * 400), ('bozuk.docx', docx(['�' * 80]))]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    read_policy(self.file(name, content))
                message = str(caught.exception)
                self.assertTrue(message.startswith(f'corrupted: "{name}"'), message)
                self.assertNotIn('müşteri', message)
                self.assertNotIn('�', message)
        path = self.file('bozuk.pdf', blank_pdf(3))
        with page_texts(['\x03\x04\x05\x06 ' * 60] * 3):
            with self.assertRaisesRegex(ValueError, '^corrupted: "bozuk.pdf"'):
                read_policy(path)

    def test_a_word_file_that_is_not_a_word_package_is_refused_as_corrupted(self):
        whole = docx(['Kayıtlar sekiz yıl saklanır.'] * 50)
        package = io.BytesIO()
        with zipfile.ZipFile(package, 'w') as archive:
            archive.writestr('word/styles.xml', '<styles/>')
        broken = io.BytesIO()
        with zipfile.ZipFile(broken, 'w') as archive:
            archive.writestr('word/document.xml', '<w:document><w:body><w:p>')
        for name, content in [('metin.docx', 'Bu bir Word dosyası değil.'), ('yarim.docx', whole[:len(whole) // 2]),
                              ('govdesiz.docx', package.getvalue()), ('kirik.docx', broken.getvalue())]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f'^corrupted: "{name}" Word belgesi olarak açılamadı'):
                    read_policy(self.file(name, content))
        # The refusals that were already there keep their own messages.
        entity = io.BytesIO()
        with zipfile.ZipFile(entity, 'w') as archive:
            archive.writestr('word/document.xml', '<!DOCTYPE x [<!ENTITY a "b">]><w:document/>')
        with self.assertRaisesRegex(ValueError, '^XML entities are not allowed'):
            read_policy(self.file('varlik.docx', entity.getvalue()))

    def test_an_empty_policy_is_refused_instead_of_analysed(self):
        documents = [('bos.txt', b''), ('bosluk.md', ' \n\n\t \n'), ('bos.docx', docx(['', ''])),
                     ('bos.csv', 'kontrol_no,ad\n\n'), ('sayfasiz.pdf', blank_pdf(0))]
        for name, content in documents:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, f'^empty: "{name}"'):
                    read_policy(self.file(name, content))
        # A running header and page numbers are all this PDF has: stripped, nothing is left to judge.
        path = self.file('baslik.pdf', blank_pdf(3))
        with page_texts([f'Örnek A.Ş. Uyum Politikası\n{n}' for n in (1, 2, 3)]):
            with self.assertRaisesRegex(ValueError, '^empty: "baslik.pdf"'):
                read_policy(path)
        # A blank PDF is still refused as scanned, as in v0.17.
        with self.assertRaisesRegex(ValueError, '^requires_ocr:'):
            read_policy(self.file('beyaz.pdf', blank_pdf(2)))

    def test_a_normal_text_document_is_state_text(self):
        documents = [('politika.txt', 'Personel kayıtları sekiz yıl saklar.\n'), ('politika.md', '# Saklama\n\nKayıtlar sekiz yıl saklanır.\n'),
                     ('politika.docx', docx(['Kayıtlar sekiz yıl saklanır.'])),
                     ('kontroller.csv', 'kontrol_no,açıklama\nK-01,Kayıtlar sekiz yıl saklanır\n')]
        for name, content in documents:
            with self.subTest(name=name):
                quality = read_policy(self.file(name, content))['document_quality']
                self.assertEqual((quality['kind'], quality['state'], quality['requires_ocr'], quality['ocr_recommended']),
                                 ('text', 'text', False, False))
                self.assertEqual(quality['quality_note'], 'Metin belgesi; metin doğrudan okundu.')
                canonical_bytes(quality)


class QualityFlowTests(unittest.TestCase):
    """The new fields reach the job record (policy_quality) and the packet (policies[].document_quality),
    and a refusal reaches the job's error, through the existing plumbing, with no change to workspace.py."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.workspace = Workspace(Path(temp.name) / 'workspace', provider_factory=lambda _: FixtureProvider(), source_fetcher=source_fixture)
        client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                            headers={'Authorization': 'Bearer test-only-token'})
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        self.client = client

    def run_with(self, name, content):
        upload = dict(name=name, content=base64.b64encode(content).decode())
        run_id = self.client.post('/api/runs', json=request_input(provider='ollama', policies=[upload])).json()['id']
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                break
            time.sleep(.01)
        return run_id, row

    def test_a_mixed_pdf_is_analysed_and_its_state_reaches_the_job_and_the_packet(self):
        with page_texts(['', 'Staff must retain records for eight years. ' * 6, 'Records are reviewed every year. ' * 6,
                         'Disposal of records is logged. ' * 6]):
            run_id, row = self.run_with('karma.pdf', blank_pdf(4))
            self.assertEqual(row['state'], 'COMPLETED', row.get('error'))
            packet = self.workspace.packet(run_id)                                          # verification re-reads the PDF
        job = row['policy_quality'][0]
        self.assertEqual((job['name'], job['kind'], job['state'], job['ocr_recommended'], job['requires_ocr']),
                         ('karma.pdf', 'mixed', 'mixed_pdf', True, False))
        self.assertIn('OCR önerilir', job['quality_note'])
        recorded = packet['events'][0]['payload']['policies'][0]['document_quality']
        self.assertEqual({k: v for k, v in job.items() if k != 'name'}, recorded)

    def test_a_corrupted_upload_fails_the_job_with_the_whole_refusal_and_none_of_its_text(self):
        logger = logging.getLogger('pypdf')
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel(logging.ERROR)
        garbage =('Gizli müşteri bilgisi ' + '�' * 300 + '\x01\x02\x03' * 50).encode('utf-8')
        for name, content, prefix in [('bozuk.txt', garbage, 'corrupted: "bozuk.txt" okunabilir policy metni vermiyor.'),
                                      ('rastgele.pdf', random.Random(18).randbytes(4096), 'corrupted: "rastgele.pdf" PDF olarak açılamadı'),
                                      ('bos.md', b' \n\n', 'empty: "bos.md" içinde analiz edilecek metin yok')]:
            with self.subTest(name=name):
                _, row =self.run_with(name, content)
                self.assertEqual(row['state'], 'FAILED')
                self.assertTrue(row['error'].startswith(prefix), row['error'])
                self.assertTrue(row['error'].endswith(('yükleyin.', 'OCR gerekir.')), row['error'])      # not cut at 600 characters
                self.assertNotIn('müşteri', row['error'])
                self.assertNotIn('policy_quality', row)                                # refused before anything was recorded


class RetainedOriginalTests(unittest.TestCase):
    def test_a_retained_original_is_re_read_without_the_v018_refusals(self):
        # A v0.17 packet may hold a policy whose text is mostly digits or a file that is now "empty";
        # re-reading it for verification must reproduce its chunks, not refuse it.
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        digits = Path(temp.name) / 'tablo.txt'
        digits.write_text('1234 5678 9012 3456 7890 1234 5678 9012 3456 7890 ab\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '^corrupted:'):
            read_policy(digits)
        self.assertEqual(len(read_policy(digits, refuse_unreadable=False)['chunks']), 1)
        blank = Path(temp.name) / 'bos.txt'
        blank.write_text('\n\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '^empty:'):
            read_policy(blank)
        self.assertEqual(read_policy(blank, refuse_unreadable=False)['chunks'], [])


if __name__ == '__main__':
    unittest.main()

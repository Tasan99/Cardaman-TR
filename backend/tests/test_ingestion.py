import unittest
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from regchain.ingestion.diff import paragraph_diff
from regchain.ingestion.fetch import fetch, public_address, validate_url
from regchain.ingestion.models import Download, IngestionError, Paragraph
from regchain.ingestion.parse import parse
from regchain.ingestion.worker import bounded_parse


def html_download(html: str) -> Download:
    return Download('https://www.fca.org.uk/test', 'https://www.fca.org.uk/test',
                    html.encode(), 'text/html', datetime.now(timezone.utc))


HTML = '''<html><head><meta property="article:published_time" content="2024-04-10T10:00:00Z"></head>
<body><nav><p>Ignore navigation</p></nav><main><h1>TEST fixture, not FCA law</h1>
<h2 id="scope">Scope</h2><p id="p1">A firm must record approval, unless the account is a sandbox account.</p>
<ul><li>Check the condition:<ul><li>except where explicitly exempt.</li></ul></li></ul>
<form><p>Ignore feedback form</p></form></main></body></html>'''


class ParseTests(unittest.TestCase):
    def test_exception_and_nested_list_preserved_once(self):
        document = parse(html_download(HTML))
        self.assertEqual(len(document.paragraphs), 2)
        self.assertIn('unless', document.paragraphs[0].text)
        self.assertIn('except where', document.paragraphs[1].text)
        self.assertNotIn('Ignore', document.normalized_text)
        self.assertEqual(str(document.publication_date), '2024-04-10')
        self.assertIsNone(document.effective_date)
        self.assertEqual(document.paragraphs[0].locator_kind, 'html_id')
        self.assertEqual(document.paragraphs[1].locator_kind, 'generated')

    def test_whitespace_changes_do_not_change_content_hash(self):
        first = parse(html_download(HTML))
        second = parse(html_download(HTML.replace('must record', 'must   record')))
        self.assertEqual(first.content_hash, second.content_hash)

    def test_date_and_title_changes_change_content_hash(self):
        first = parse(html_download(HTML))
        self.assertNotEqual(first.content_hash, parse(html_download(HTML.replace('2024-04-10', '2024-04-11'))).content_hash)
        self.assertNotEqual(first.content_hash, replace(first, title='New title').content_hash)

    def test_script_only_source_rejected(self):
        with self.assertRaises(IngestionError):
            parse(html_download('<h1>Handbook</h1><script>render()</script>'))

    def test_fca_article_selected_over_timeline_and_publication_date_over_creation(self):
        html = '''<h1>FCA fixture</h1><meta property="article:published_time" content="2023-05-23T01:00:00Z">
        <meta property="funnelback:published-date" content="2024/04/10">
        <article class="timeline"></article><article data-history-node-id="1"><p>Rule text.</p></article>'''
        doc = parse(html_download(html))
        self.assertEqual(str(doc.publication_date),'2024-04-10')
        self.assertEqual(doc.paragraphs[0].text,'Rule text.')

    def test_password_protected_pdf_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612,height=792)
        writer.encrypt('not-an-empty-password')
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(IngestionError,'Encrypted PDF'):
            parse(replace(html_download(''),body=output.getvalue(),media_type='application/pdf'))

    def test_blank_pdf_requires_review(self):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(IngestionError, 'OCR/manual review'):
            parse(replace(html_download(''), body=output.getvalue(), media_type='application/pdf'))

    def test_pdf_page_and_text_preserved(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 50 700 Td (A firm must record approval unless exempt.) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        writer.add_metadata({'/Title': 'Synthetic test rule'})
        output = BytesIO()
        writer.write(output)
        document = parse(replace(html_download(''), body=output.getvalue(), media_type='application/pdf'))
        self.assertEqual(document.title, 'Synthetic test rule')
        self.assertEqual(document.paragraphs[0].page, 1)
        self.assertIn('unless exempt', document.paragraphs[0].text)
        self.assertIsNone(document.publication_date)

    def test_pdf_signature_required(self):
        with self.assertRaises(IngestionError):
            parse(replace(html_download('not a PDF'), media_type='application/pdf'))

    def test_subprocess_parser(self):
        self.assertEqual(bounded_parse(html_download(HTML)).content_hash, parse(html_download(HTML)).content_hash)


class DiffTests(unittest.TestCase):
    def paras(self, *texts):
        return tuple(Paragraph('scope', str(i), text) for i, text in enumerate(texts, 1))

    def test_insert_does_not_mark_following_paragraphs_modified(self):
        diff = paragraph_diff(self.paras('A', 'B'), self.paras('new', 'A', 'B'))
        self.assertEqual([(c.kind,c.old_index,c.new_index) for c in diff], [('ADDED',None,0)])

    def test_removed_exception_is_visible(self):
        diff = paragraph_diff(self.paras('Must do X unless Y.'), self.paras('Must do X.'))
        self.assertEqual(diff[0].kind, 'MODIFIED')
        self.assertIn('-Must do X unless Y.', diff[0].diff_text)

    def test_removed_paragraph(self):
        self.assertEqual(paragraph_diff(self.paras('A','B'),self.paras('B'))[0].kind, 'REMOVED')

    def test_identical_paragraphs(self):
        self.assertEqual(paragraph_diff(self.paras('A'),self.paras('A')), [])


class SourceSecurityTests(unittest.TestCase):
    def test_non_fca_and_unsafe_urls_rejected(self):
        for url in ['http://www.fca.org.uk/a', 'https://www.fca.org.uk.evil.test/a',
                    'https://user@www.fca.org.uk/a', 'https://127.0.0.1/a',
                    'https://www.fca.org.uk:8080/a', 'https://www.fca.org.uk/a\r\nX:1']:
            with self.subTest(url=url), self.assertRaises(IngestionError):
                validate_url(url)

    def test_fragment_removed(self):
        self.assertEqual(validate_url('https://www.fca.org.uk/a#b'), 'https://www.fca.org.uk/a')

    def test_private_dns_answer_rejected(self):
        with patch('socket.getaddrinfo', return_value=[(2,1,6,'',('127.0.0.1',443))]):
            with self.assertRaises(IngestionError):
                public_address('www.fca.org.uk')

    def test_redirect_to_private_host_rejected(self):
        with patch('regchain.ingestion.fetch.PinnedHTTPSConnection') as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 302
            response.getheader.return_value = 'https://127.0.0.1/secret'
            with self.assertRaises(IngestionError):
                fetch('https://www.fca.org.uk/a')
            self.assertEqual(connection.call_count, 1)

    def test_oversized_response_rejected(self):
        with patch('regchain.ingestion.fetch.PinnedHTTPSConnection') as connection:
            response = connection.return_value.getresponse.return_value
            response.status = 200
            response.getheader.side_effect = lambda key, default=None: {'Content-Type':'text/html','Content-Length':'999999999'}.get(key,default)
            with self.assertRaisesRegex(IngestionError, '12 MiB'):
                fetch('https://www.fca.org.uk/a')

    def test_retry_is_bounded(self):
        with patch('regchain.ingestion.fetch.PinnedHTTPSConnection') as connection, patch('regchain.ingestion.fetch.time.sleep'):
            connection.return_value.getresponse.return_value.status = 503
            with self.assertRaises(IngestionError):
                fetch('https://www.fca.org.uk/a')
            self.assertEqual(connection.call_count, 3)

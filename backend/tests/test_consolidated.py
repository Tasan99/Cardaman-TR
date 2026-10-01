"""Consolidated Handbook sources and cross-document reference resolution.

Fixtures imitate the structure of the official pages. They are not FCA law and the
assertions are about source handling, never about legal correctness.
"""
import unittest
from datetime import date, datetime, timezone
from hashlib import sha256

from regchain.extraction.pipeline import extract
from regchain.extraction.providers import RulesProvider
from regchain.extraction.retrieval import operative_text, retrieve
from regchain.ingestion.handbook import parse_handbook
from regchain.ingestion.models import Download, IngestionError
from regchain.ingestion.parse import parse
from regchain.ingestion.sources import chapter_key, chapter_of, chapter_url, glossary_source

HANDBOOK_URL = 'https://handbook.fca.org.uk/handbook/conc7/conc7s3'


def card(label: str, code: str, effective: str, body: str) -> str:
    return f'''<div class="p-card-content"><div class="header">
      <label><span>{label}</span></label><span class="font-bold">{effective}</span>
      <span class="provison-type">{code}</span></div>
      <div class="provision-meta"><div class="details_wrap"><div class="section-content">
      {body}</div></div></div></div>'''


def page(*cards: str, heading: str = 'Forbearance and due consideration') -> str:
    return ('<html><body><main><h1>CONC 7.3 Fixture, not FCA law</h1>'
            f'<h3>{heading}</h3><div class="app">' + ''.join(cards) + '</div></main></body></html>')


def download(html: str, url: str = HANDBOOK_URL) -> Download:
    return Download(url, url, html.encode(), 'text/html', datetime.now(timezone.utc))


def section(sid, text, label=None, version='v1', **values):
    values.setdefault('section_number', 'CONC 7.3')
    values.setdefault('heading_path', ['CONC 7.3'])
    values.setdefault('paragraph_number', label or sid)
    return dict(id=sid, version_id=version, text=text, printed_label=label,
                content_hash=sha256(text.encode()).hexdigest(), **values)


def corpus_section(sid, text, label, version, key='FCA-HB-CONC-7-1', **values):
    return section(sid, text, label, version, external_key=key, version_hash='0' * 64,
                   source_kind='CONSOLIDATED', **values)


class HandbookParserTests(unittest.TestCase):
    def test_provision_label_type_date_and_defined_terms(self):
        html = page(card('CONC 7.3.4', 'R', '04/11/2024',
                         '<div class="rule"><p>A <a class="autodeftext" href="/glossary/G430">firm</a> '
                         'must treat <a class="autodeftext" href="/glossary/G252">customers</a> fairly.</p>'
                         '<p>[<strong>Note</strong>: paragraph 7.3 of ILG]</p></div>'))
        paragraph = parse_handbook(download(html)).paragraphs[0]
        self.assertEqual(paragraph.printed_label, 'CONC 7.3.4')
        self.assertEqual(paragraph.legal_type, 'RULE')
        self.assertEqual(paragraph.effective_from, date(2024, 11, 4))
        self.assertEqual(paragraph.source_kind, 'CONSOLIDATED')
        self.assertEqual((paragraph.section, paragraph.number), ('CONC 7.3', '7.3.4'))
        self.assertEqual(paragraph.text, 'A firm must treat customers fairly. [Note: paragraph 7.3 of ILG]')
        self.assertEqual([t['glossary_id'] for t in paragraph.defined_terms], ['G252', 'G430'])

    def test_inline_links_do_not_insert_spaces_but_blocks_separate(self):
        html = page(card('CONC 7.3.9', 'R', '01/04/2014',
                         '<div class="rule"><p>A firm must protect the '
                         '<a class="autodeftext" href="/glossary/G252">customer</a>&#8217;s interest:</p>'
                         '<ol><li class="subpara1"><a></a>(1) <p>by acting promptly;</p></li></ol></div>'))
        self.assertEqual(parse_handbook(download(html)).paragraphs[0].text,
                         'A firm must protect the customer’s interest: (1) by acting promptly;')

    def test_markup_that_is_not_text_never_enters_a_quote(self):
        # Comments and scripts are NavigableString subclasses in the parser library;
        # treating them as text would corrupt a quote that must match the source.
        html = page(card('CONC 7.3.4', 'R', '04/11/2024',
                         '<div class="rule"><!-- cms rebuild 2026 -->'
                         '<script>track("x")</script><p>A firm must act.</p></div>'))
        self.assertEqual(parse_handbook(download(html)).paragraphs[0].text, 'A firm must act.')

    def test_guidance_and_deleted_provisions_are_marked(self):
        html = page(card('CONC 7.3.2', 'G', '26/06/2026', '<div class="guidance"><p>A firm should act.</p></div>'),
                    card('CONC 7.3.3', 'R', '21/03/2016', '<div class="rule"><p>[deleted]</p></div>'))
        guidance, deleted = parse_handbook(download(html)).paragraphs
        self.assertEqual(guidance.legal_type, 'GUIDANCE')
        self.assertIn('DELETED_PROVISION', deleted.quality_flags)

    def test_client_rendered_view_is_rejected_not_guessed(self):
        html = '<html><body><main><h1>CONC 7.3</h1><div class="app"></div></main></body></html>'
        with self.assertRaises(IngestionError) as error:
            parse_handbook(download(html))
        self.assertIn('server-rendered', str(error.exception))

    def test_unrecognised_label_is_rejected(self):
        with self.assertRaises(IngestionError):
            parse_handbook(download(page(card('Some heading', 'R', '04/11/2024', '<p>Text.</p>'))))

    def test_glossary_definition_keeps_case_sensitive_identifier(self):
        html = ('<html><body><main><div class="p-card-content"><div class="header">'
                '<h1>customer</h1></div><div class="details_wrap"><div class="section">'
                '<p>a client who is not an eligible counterparty.</p></div></div></div></main></body></html>')
        paragraph = parse_handbook(download(html, 'https://handbook.fca.org.uk/glossary/G3497p')).paragraphs[0]
        self.assertEqual((paragraph.legal_type, paragraph.number), ('DEFINITION', 'G3497p'))
        self.assertEqual(paragraph.locator_kind, 'handbook_glossary')

    def test_handbook_host_selects_the_consolidated_adapter(self):
        html = page(card('CONC 7.3.4', 'R', '04/11/2024', '<div class="rule"><p>A firm must act.</p></div>'))
        self.assertEqual(parse(download(html)).paragraphs[0].source_kind, 'CONSOLIDATED')


class ReferenceResolutionTests(unittest.TestCase):
    def test_reference_resolves_into_a_pinned_corpus_document(self):
        target = section('a', 'A firm must act in accordance with CONC 7.1.4R.', 'CONC 7.3.4')
        other = corpus_section('b', 'This chapter applies to lenders.', 'CONC 7.1.4', 'v2',
                               legal_type='RULE', section_number='CONC 7.1')
        packet = retrieve(target, [target], corpus=[other])
        self.assertEqual([i['section_id'] for i in packet.items], ['b'])
        self.assertEqual(packet.unresolved, ())
        self.assertEqual(packet.corpus, ({'version_id': 'v2', 'external_key': 'FCA-HB-CONC-7-1',
                                          'normalized_hash': '0' * 64},))

    def test_corpus_identity_is_in_the_manifest_for_later_verification(self):
        target = section('a', 'A firm must act under CONC 7.1.4R.', 'CONC 7.3.4')
        other = corpus_section('b', 'Applies to lenders.', 'CONC 7.1.4', 'v2', legal_type='RULE')
        manifest = retrieve(target, [target], corpus=[other]).manifest()
        self.assertEqual(manifest['corpus'][0]['normalized_hash'], '0' * 64)
        self.assertEqual(manifest['items'][0]['version_id'], 'v2')
        self.assertEqual(manifest['items'][0]['legal_type'], 'RULE')

    def test_uningested_chapter_is_reported_as_a_missing_source(self):
        target = section('a', 'A firm must act under MCOB 13.3.1R.', 'CONC 7.3.4')
        packet = retrieve(target, [target])
        self.assertEqual(packet.missing_sources, ('MCOB 13.3.1R',))
        self.assertEqual(extract(target['text'], RulesProvider(), packet).reason, 'MISSING_CONSOLIDATED_SOURCE')
        self.assertEqual(chapter_of(packet.missing_sources[0]), 'MCOB 13.3')

    def test_editorial_note_is_not_an_operative_cross_reference(self):
        text = 'A firm must treat customers fairly. [Note: paragraph 7.3 of ILG and 2.2 of DCG]'
        self.assertNotIn('7.3', operative_text(text))
        target = section('a', text, 'CONC 7.3.4')
        packet = retrieve(target, [target])
        self.assertEqual((packet.unresolved, packet.missing_sources), ((), ()))
        self.assertEqual(extract(text, RulesProvider(), packet).output.status, 'EXTRACTED')

    def test_sub_lettered_provision_wins_over_a_type_letter_reading(self):
        target = section('a', 'A firm must act under CONC 7.3.5D.', 'CONC 7.3.4')
        sub = corpus_section('b', 'Sub-lettered provision.', 'CONC 7.3.5D', 'v2', legal_type='RULE')
        base = corpus_section('c', 'Base guidance.', 'CONC 7.3.5', 'v2', legal_type='GUIDANCE')
        packet = retrieve(target, [target], corpus=[sub, base])
        self.assertEqual([i['section_id'] for i in packet.items], ['b'])

    def test_type_letter_is_dropped_when_the_recorded_type_agrees(self):
        target = section('a', 'A firm must act under CONC 7.1.4R.', 'CONC 7.3.4')
        base = corpus_section('b', 'The rule text.', 'CONC 7.1.4', 'v2', legal_type='RULE')
        self.assertEqual([i['section_id'] for i in retrieve(target, [target], corpus=[base]).items], ['b'])

    def test_two_readings_of_one_citation_abstain_instead_of_guessing(self):
        target = section('a', 'A firm must act under CONC 7.3.5G.', 'CONC 7.3.4')
        sub = corpus_section('b', 'Sub-lettered provision.', 'CONC 7.3.5G', 'v2', legal_type='RULE')
        base = corpus_section('c', 'Base guidance.', 'CONC 7.3.5', 'v2', legal_type='GUIDANCE')
        packet = retrieve(target, [target], corpus=[sub, base])
        self.assertEqual(packet.items, ())
        self.assertIn('ambiguous', packet.unresolved[0])

    def test_chapter_citation_expands_to_its_provisions(self):
        target = section('a', 'A firm must act in accordance with CONC 5D.3.', 'CONC 7.3.4')
        first = corpus_section('b', 'First provision.', 'CONC 5D.3.1', 'v2', legal_type='RULE')
        second = corpus_section('c', 'Second provision.', 'CONC 5D.3.2', 'v2', legal_type='RULE')
        packet = retrieve(target, [target], corpus=[first, second])
        self.assertEqual(sorted(i['section_id'] for i in packet.items), ['b', 'c'])
        self.assertEqual(packet.unresolved, ())

    def test_context_gap_is_recorded_but_only_the_target_blocks(self):
        target = section('a', 'A firm must act under CONC 7.1.4R.', 'CONC 7.3.4')
        cited = corpus_section('b', 'Applies as set out in MCOB 13.9.1R.', 'CONC 7.1.4', 'v2', legal_type='RULE')
        packet = retrieve(target, [target], corpus=[cited])
        self.assertEqual((packet.unresolved, packet.missing_sources), ((), ()))
        self.assertIn('MCOB 13.9.1R', packet.dependency_gaps)
        self.assertIn('MCOB 13.9.1R', packet.manifest()['dependency_gaps'])

    def test_targets_own_missing_reference_still_blocks_extraction(self):
        target = section('a', 'A firm must act under MCOB 13.9.1R.', 'CONC 7.3.4')
        packet = retrieve(target, [target])
        self.assertTrue(packet.unresolved)
        self.assertEqual(extract(target['text'], RulesProvider(), packet).output.obligations, [])

    def test_glossary_definition_needs_an_explicit_definitional_cue(self):
        definition = corpus_section('g', 'customer means a client.', 'customer', 'v3',
                                    key='FCA-HB-GLOSSARY-G252', legal_type='DEFINITION',
                                    paragraph_number='G252', section_number='glossary')
        terms = [{'glossary_id': 'G252', 'term': 'customer'}]
        plain = section('a', 'A firm must treat customers fairly.', 'CONC 7.3.4', defined_terms=terms)
        self.assertEqual(retrieve(plain, [plain], corpus=[definition]).items, ())
        cued = section('a', 'A firm must treat a customer as defined in the glossary.',
                       'CONC 7.3.4', defined_terms=terms)
        packet = retrieve(cued, [cued], corpus=[definition])
        self.assertEqual([i['reason'] for i in packet.items], ['glossary_definition'])

    def test_amendment_text_is_never_used_as_a_consolidated_rule(self):
        target = section('a', 'A firm must retain records.', '7.3.4', source_kind='AMENDMENT')
        self.assertEqual(extract(target['text'], RulesProvider(), retrieve(target, [target])).reason,
                         'AMENDMENT_REQUIRES_CONSOLIDATED_SOURCE')


class SourceCatalogueTests(unittest.TestCase):
    def test_chapter_locator_maps_to_official_url_and_key(self):
        self.assertEqual(chapter_url('CONC 7.3'), 'https://www.handbook.fca.org.uk/handbook/CONC/7/3.html')
        self.assertEqual(chapter_url('CONC 5D.1'), 'https://www.handbook.fca.org.uk/handbook/CONC/5D/1.html')
        self.assertEqual(chapter_key('PRIN 2A.2'), 'FCA-HB-PRIN-2A-2')
        self.assertIsNone(chapter_url('paragraph 2.26'))

    def test_glossary_source_rejects_an_invented_identifier(self):
        self.assertEqual(glossary_source('G3497p')[1], 'https://www.handbook.fca.org.uk/glossary/G3497p')
        with self.assertRaises(ValueError):
            glossary_source('customer')


if __name__ == '__main__':
    unittest.main()

"""Türkiye: mevzuat.gov.tr ingestion, Turkish grounding and the Official Gazette scan.

Fixtures imitate the shapes of the live pages read on 22 September 2026 (Word-exported
HTML, footnote anchors and superscripts, page numbers, a heading sharing its paragraph with
the MADDE line). No network is used here.
"""
import base64
import json
import tempfile
import time
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from regchain.extraction.grounding import CONDITIONS, EXCEPTIONS, MODAL, modality, qualifier_spans, turkish, turkish_references, verify
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import RulesProvider
from regchain.extraction.retrieval import retrieve
from regchain.extraction.schema import Candidate, ExtractionOutput
from regchain.ingestion.mevzuat import (KINDS, MEVZUAT_PARSER_VERSION, document_label, identity, parse_mevzuat, preliminary,
                                        scope_rank, text_url)
from regchain.ingestion.models import Download, IngestionError
from regchain.pilot.engine import analyze, split_units
from regchain.pilot.policies import mevzuat_extract
from regchain.pilot.report import render
from regchain.pilot.sources import application_rows, chapter_of, is_turkish, load_sources, save_sources, section_of, select_targets
from regchain.pilot.turkiye import gazette_day, gazette_scan
from regchain.pilot.workspace import RunInput, Workspace, create_app
from test_pilot import FixtureProvider
from test_workspace import request_input

URL = text_url('1', '5549')


def p(text, center=False, bold=False):
    style = "text-align:center" if center else "text-align:justify;text-indent:35.45pt"
    inner = f'<b><span>{text}</span></b>' if bold else f'<span>{text}</span>'
    return f"<p class=MsoNormal {'align=center ' if center else ''}style='{style}'>{inner}</p>"


def law_html():
    return ('<html><head><meta http-equiv=Content-Type content="text/html; charset=Windows-1254"></head><body lang=TR><div class=WordSection1>'
            + p('SUÇ GELİRLERİNİN AKLANMASININ ÖNLENMESİ', center=True, bold=True) + p('HAKKINDA KANUN', center=True, bold=True)
            + p('Kanun Numarası : 5549') + p('Yayımlandığı Resmî Gazete : Tarih: 18/10/2006 Sayı: 26323')
            + p('BİRİNCİ BÖLÜM', center=True, bold=True) + p('Amaç ve Tanımlar', center=True, bold=True)
            + p('Amaç', bold=True)
            + "<p class=MsoNormal style='text-align:justify'><b><span>MADDE 1 –</span></b><span> (1) Bu Kanunun amacı, suç gelirlerinin\n  aklanmasının önlenmesine ilişkin usûl ve esasları belirlemektir.</span></p>"
            + "<p class=MsoNormal style='text-align:justify'><b><span>Tanımlar<br> MADDE 2 – </span></b><span>(1) Bu Kanunda geçen;<sup> (1)</sup></span></p>"
            + p('d) Yükümlü: Bankacılık, sigortacılık ve ödeme hizmetleri alanında faaliyet gösterenleri,')
            + p('ifade eder.')
            + p('5337', center=True)
            + p('İKİNCİ BÖLÜM', center=True, bold=True) + p('Yükümlülükler', center=True, bold=True)
            + "<p class=MsoNormal style='text-align:justify'><b><span>Müşterinin tanınması</span></b><a href=\"#_ftn8\" name=\"_ftnref8\"><span class=MsoFootnoteReference>[8]</span></a></p>"
            + "<p class=MsoNormal style='text-align:justify'><b><span>MADDE 3 –</span></b><span> (1) Yükümlüler, işlem yapanların kimliklerini tespit etmek zorundadır. </span></p>"
            + p('(2) Kimlik tespitine esas belge nevilerini belirlemeye Bakanlık yetkilidir.')
            + p('Şüpheli işlem bildirimi', bold=True)
            + "<p class=MsoNormal style='text-align:justify'><b><span>MADDE 4 –</span></b><span> (1) Şüpheli işlemler, bu Kanunun 3 üncü maddesi saklı kalmak kaydıyla Başkanlığa bildirilir.</span></p>"
            + p('(2) Yükümlüler, Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar.')
            + p('Uluslararası bilgi değişimi', bold=True)
            + "<p class=MsoNormal style='text-align:justify'><b><span>MADDE 5 –</span></b><span> <b>(Mülga: 2/7/2018-KHK-703/15 md.)</b></span></p>"
            + p('Elektronik tebligat', bold=True)
            + "<p class=MsoNormal style='text-align:justify'><b><span>MADDE 5/A- (Ek: 18/6/2014-6545/87 md.)</span></b></p>"
            + p('(1) Tebligatlar, 7201 sayılı Tebligat Kanununun 7 nci maddesine bağlı olmaksızın elektronik ortamda yapılır.')
            + '<table><tr><td>' + p('26/7/2024', center=True) + '</td></tr></table>'
            + '<div><div id=ftn1><p class=MsoFootnoteText><a href="#_ftnref1" name="_ftn1">[1]</a> Anayasa Mahkemesinin kararı ile iptal edilmiştir.</p></div></div>'
            + '</div></body></html>')


def download(html=None, url=URL):
    return Download(url, url, (html or law_html()).encode('utf-8'), 'text/html', datetime(2026, 9, 22, tzinfo=timezone.utc))


def fixture_fetch(directory, **kwargs):
    return save_sources(directory, [download()], **{k: v for k, v in kwargs.items() if k in ('module', 'chapter', 'mevzuat')})


class ParserTests(unittest.TestCase):
    def test_articles_headings_chapters_and_repeals_are_read_from_word_html(self):
        doc = parse_mevzuat(download())
        self.assertEqual(doc.title, 'SUÇ GELİRLERİNİN AKLANMASININ ÖNLENMESİ HAKKINDA KANUN')
        self.assertEqual(doc.publication_date, date(2006, 10, 18))
        self.assertEqual([a.printed_label for a in doc.paragraphs],
                         ['Kanun 5549 md. 1', 'Kanun 5549 md. 2', 'Kanun 5549 md. 3', 'Kanun 5549 md. 4', 'Kanun 5549 md. 5', 'Kanun 5549 md. 5/A'])
        by = {a.number: a for a in doc.paragraphs}
        self.assertEqual(by['1'].text, '(1) Bu Kanunun amacı, suç gelirlerinin aklanmasının önlenmesine ilişkin usûl ve esasları belirlemektir.')
        self.assertEqual(by['1'].heading_path, (doc.title, 'BİRİNCİ BÖLÜM Amaç ve Tanımlar', 'Amaç'))
        # A heading that shares its paragraph with the MADDE line, and a superscript footnote mark.
        self.assertEqual(by['2'].heading_path[2], 'Tanımlar')
        self.assertTrue(by['2'].text.startswith('(1) Bu Kanunda geçen; d) Yükümlü:'))
        self.assertNotIn('(1) (1)', by['2'].text)
        # Footnote anchors, page numbers, the date table and the footnote block are not law.
        self.assertEqual(by['3'].heading_path[2], 'Müşterinin tanınması')
        self.assertNotIn('[8]', by['3'].heading_path[2])
        self.assertNotIn('5337', ' '.join(a.text for a in doc.paragraphs))
        self.assertNotIn('26/7/2024', ' '.join(a.text for a in doc.paragraphs))
        self.assertNotIn('Anayasa Mahkemesinin', ' '.join(a.text for a in doc.paragraphs))
        self.assertEqual(by['3'].text, '(1) Yükümlüler, işlem yapanların kimliklerini tespit etmek zorundadır. (2) Kimlik tespitine esas belge nevilerini belirlemeye Bakanlık yetkilidir.')
        self.assertEqual(by['3'].section, 'İKİNCİ BÖLÜM')
        self.assertIn('DELETED_PROVISION', by['5'].quality_flags)
        self.assertEqual(by['5/A'].text, '(Ek: 18/6/2014-6545/87 md.) (1) Tebligatlar, 7201 sayılı Tebligat Kanununun 7 nci maddesine bağlı olmaksızın elektronik ortamda yapılır.')
        self.assertTrue(all(a.legal_type == 'RULE' and a.locator_kind == 'mevzuat_madde' and a.source_kind == 'CONSOLIDATED' for a in doc.paragraphs))

    def test_scope_and_preliminary_articles_are_recognised_by_heading(self):
        _, sections = load_sources(self.snapshot())
        ranks = {s['printed_label']: scope_rank(s) for s in sections}
        self.assertEqual(ranks['Kanun 5549 md. 2'], 2)                 # Tanımlar defines "Yükümlü"
        self.assertEqual(ranks['Kanun 5549 md. 3'], 0)
        # Seen live: "Yükümlülük ihlâlinde idarî ceza" is about breaches, not about who is bound.
        self.assertEqual(scope_rank({'heading_path': ['t', 'c', 'Yükümlülük ihlâlinde idarî ceza']}), 0)
        self.assertEqual(scope_rank({'heading_path': ['t', 'c', 'Yükümlüler']}), 3)
        self.assertEqual(scope_rank({'heading_path': ['t', 'c', 'Amaç ve kapsam']}), 3)
        self.assertTrue(preliminary(next(s for s in sections if s['printed_label'] == 'Kanun 5549 md. 1')))
        self.assertFalse(preliminary(next(s for s in sections if s['printed_label'] == 'Kanun 5549 md. 4')))

    def snapshot(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        save_sources(Path(temp.name)/'tr', [download()], mevzuat=('1', '5549', '5'))
        return Path(temp.name)/'tr'

    def test_only_the_mevzuat_text_endpoint_is_parsed_and_identifiers_are_validated(self):
        self.assertEqual(text_url('21', '200713012'), 'https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=21&MevzuatNo=200713012&MevzuatTertip=5')
        self.assertEqual(identity(URL), ('1', '5549', '5'))
        self.assertEqual(document_label('7', '12426'), 'Yönetmelik 12426')
        for kind, number in (('99', '5549'), ('1', 'abc'), ('1', ''), ('', '5549')):
            with self.subTest(kind=kind, number=number), self.assertRaises(ValueError):
                text_url(kind, number)
        with self.assertRaises(IngestionError):
            parse_mevzuat(download(url='https://handbook.fca.org.uk/handbook/cobs4'))
        with self.assertRaisesRegex(IngestionError, 'No MADDE'):
            parse_mevzuat(download('<html><body><p>Kayıt bulunamadı</p></body></html>'))


class GroundingTests(unittest.TestCase):
    def test_turkish_modality_is_read_from_the_sentence_end(self):
        cases = {'Yükümlüler, kimlikleri tespit etmek zorundadır.': [('zorundadır', 'MUST')],
                 'Yükümlüler bunu hiç kimseye açıklayamazlar.': [('açıklayamazlar', 'MUST_NOT')],
                 'Şüpheli işlemler on iş günü içinde Başkanlığa bildirilir.': [('bildirilir', 'MUST')],
                 'Bu işlemlerde kimlik tespiti yapılmaz.': [('yapılmaz', 'MUST_NOT')],
                 'Kurul ek tedbir alabilir; kayıtlar imha edilebilir.': [],                  # permissions are not duties
                 'Bu hükümler yabancı kuruluşlara uygulanmaz.': [],                          # an exception marker, not a duty
                 'Belgelerin sekiz yıl saklanması gerekir.': [('gerekir', 'MUST')],
                 'A firm must not delete records.': [('must not', 'MUST_NOT')]}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual([(m.group(), modality(m.group())) for m in MODAL.finditer(text)], expected)
        self.assertTrue(turkish('Yükümlüler bildirmek zorundadır.'))
        self.assertFalse(turkish('A firm must retain records.'))

    def test_turkish_qualifiers_are_the_clause_before_the_marker(self):
        text = 'Yükümlüler, şüphe oluşması hâlinde, mahkeme kararları hariç, işlemi Başkanlığa bildirmek zorundadır.'
        self.assertEqual(qualifier_spans(text, CONDITIONS), ['şüphe oluşması hâlinde'])
        self.assertEqual(qualifier_spans(text, EXCEPTIONS), ['mahkeme kararları hariç'])

    def test_citations_of_this_document_are_references_and_of_other_statutes_are_not(self):
        text = ('Bu Kanunun 4 üncü maddesi ile 5237 sayılı Türk Ceza Kanununun 282 nci maddesi saklıdır. '
                '3 üncü maddede belirtilen tedbirler alınır. Kanunun 5 inci maddesi uyarınca çıkarılır.')
        self.assertEqual([(n, i) for _, n, i in turkish_references(text)], [('4', True), ('282', False), ('3', True), ('5', False)])

    def test_the_grounding_gate_aligns_subject_action_and_modal_in_turkish_order(self):
        text = 'Yükümlüler, işlem yapanların kimliklerini tespit etmek zorundadır.'
        good = Candidate(source_quote=text, subject='Yükümlüler', modality='MUST', required_action='işlem yapanların kimliklerini tespit etmek', confidence_score='0.5')
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[good]))
        # The marker and the full stop may be copied with the predicate (measured model habit).
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[Candidate(source_quote=text, subject='Yükümlüler', modality='MUST', required_action='işlem yapanların kimliklerini tespit etmek zorundadır.', confidence_score='0.5')]))
        for subject, action in (('Yükümlüler', 'tespit'), ('Bakanlık', 'kimliklerini tespit etmek'), ('Yükümlüler', 'kimliklerini tespit')):
            with self.subTest(subject=subject, action=action), self.assertRaises(ValueError):
                verify(text, ExtractionOutput(status='EXTRACTED', obligations=[Candidate(source_quote=text, subject=subject, modality='MUST', required_action=action, confidence_score='0.5')]))
        # Polarity belongs to the text: a MUST proposal against "açıklayamazlar" is stored as MUST_NOT.
        from regchain.extraction.contract import materialize
        prohibition = 'Yükümlüler, bunu hiç kimseye açıklayamazlar.'
        stored = materialize(json.dumps({'status': 'EXTRACTED', 'obligations': [{'subject': 'Yükümlüler', 'modality': 'MUST', 'action': 'bunu hiç kimseye açıklayamazlar.', 'conditions': [], 'exceptions': [], 'evidence': []}]}), prohibition)
        self.assertEqual((stored.obligations[0].modality, stored.obligations[0].prohibited_action), ('MUST_NOT', 'bunu hiç kimseye açıklayamazlar.'))
        verify(prohibition, stored)
        aorist = 'Şüpheli işlemler on iş günü içinde Başkanlığa bildirilir.'
        verify(aorist, ExtractionOutput(status='EXTRACTED', obligations=[Candidate(source_quote=aorist, subject='Şüpheli işlemler', modality='MUST', required_action='on iş günü içinde Başkanlığa bildirilir', confidence_score='0.5')]))

    def test_the_rules_baseline_extracts_a_turkish_duty_and_a_leading_note_does_not_break_units(self):
        result = extract('Yükümlüler, Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar.', RulesProvider())
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        self.assertEqual((result.output.obligations[0].subject, result.output.obligations[0].modality, result.output.obligations[0].prohibited_action),
                         ('Yükümlüler', 'MUST_NOT', 'Başkanlığa şüpheli işlem bildiriminde bulunulduğunu hiç kimseye açıklayamazlar'))
        text = '(Ek: 18/6/2014-6545/87 md.) (1) Tebligatlar elektronik ortamda yapılır. (2) Başkanlık altyapıyı kurmaya yetkilidir.'
        self.assertEqual([unit for _, unit in split_units(text)], ['(1) Tebligatlar elektronik ortamda yapılır.', '(2) Başkanlık altyapıyı kurmaya yetkilidir.'])

    def test_an_internal_article_reference_is_resolved_within_the_regulation(self):
        _, sections = load_sources(ParserTests.snapshot(self))
        target = next(s for s in sections if s['printed_label'] == 'Kanun 5549 md. 4')
        packet = retrieve(target, sections)
        self.assertEqual([i['printed_label'] for i in packet.items if i['reason'] == 'exact_reference'], ['Kanun 5549 md. 3'])
        self.assertEqual(packet.unresolved, ())
        # The statute citation in 5/A is a citation; nothing is missing and nothing blocks.
        other = next(s for s in sections if s['printed_label'] == 'Kanun 5549 md. 5/A')
        self.assertEqual(retrieve(other, sections).unresolved, ())


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        save_sources(self.root/'tr', [download()], mevzuat=('1', '5549', '5'))
        self.bundle, self.sections = load_sources(self.root/'tr')

    def test_snapshot_records_the_turkish_parser_and_verifies_on_reload(self):
        source = self.bundle['sources'][0]
        self.assertEqual(source['parser_version'], MEVZUAT_PARSER_VERSION)
        self.assertEqual(source['title'], 'SUÇ GELİRLERİNİN AKLANMASININ ÖNLENMESİ HAKKINDA KANUN')
        self.assertEqual(self.sections[0]['external_key'], 'MEVZUAT-1.5.5549')
        self.assertEqual(chapter_of('Kanun 5549 md. 4'), ('KANUN', '5549'))
        self.assertEqual(chapter_of('Yönetmelik 200713012 md. 9/A'), ('YONETMELIK', '200713012'))
        self.assertEqual(section_of('Kanun 5549 md. 4'), 'Kanun 5549')
        self.assertTrue(is_turkish('KANUN') and not is_turkish('COBS'))
        with self.assertRaisesRegex(ValueError, 'başka bir metin'):
            save_sources(self.root/'wrong', [download()], mevzuat=('1', '6493', '5'))

    def test_scope_evidence_is_the_definitions_article_and_targets_skip_preliminaries(self):
        self.assertEqual([r['printed_label'] for r in application_rows(self.sections, 'KANUN', '5549')], ['Kanun 5549 md. 2'])
        self.assertEqual(select_targets(self.sections, 'KANUN', '5549', 'all'), ['Kanun 5549 md. 3', 'Kanun 5549 md. 4', 'Kanun 5549 md. 5/A'])
        self.assertEqual(select_targets(self.sections, 'KANUN', '5549', 'all', ['4', '5/A']), ['Kanun 5549 md. 4', 'Kanun 5549 md. 5/A'])
        self.assertEqual(select_targets(self.sections, 'KANUN', '5549', 'labels', labels=['Kanun 5549 md. 3']), ['Kanun 5549 md. 3'])
        with self.assertRaisesRegex(ValueError, 'olmayan madde'):
            select_targets(self.sections, 'KANUN', '5549', 'all', ['99'])

    def test_a_turkish_regulation_is_analysed_with_its_own_scope_and_reported_in_turkish(self):
        from test_pilot import company
        policies = [{'filename': 'uyum.md', 'raw_hash': 'd'*64, 'bytes': 30, 'parser': 'fixture',
                     'chunks': [{'source_id': 'p1', 'policy_hash': 'd'*64, 'filename': 'uyum.md', 'locator': 'text_block', 'number': 1,
                                 'start': 0, 'end': 40, 'text': 'Müşterinin kimliği işlem öncesinde tespit edilir.'}]}]
        packet = analyze(company(), policies, self.sections, FixtureProvider(), ['Kanun 5549 md. 3', 'Kanun 5549 md. 4'])
        payload = packet['events'][0]['payload']
        self.assertEqual(payload['regulation']['regulator'], 'TR')
        self.assertEqual((payload['regulation']['module'], payload['regulation']['chapter']), ('KANUN', '5549'))
        self.assertEqual(payload['regulation']['title'], 'SUÇ GELİRLERİNİN AKLANMASININ ÖNLENMESİ HAKKINDA KANUN')
        self.assertEqual([s['printed_label'] for s in payload['scope_sources']], ['Kanun 5549 md. 2'])
        labels = [o['source_label'] for o in payload['obligations']]
        self.assertIn('Kanun 5549 md. 3', labels)
        self.assertIn('Kanun 5549 md. 4', labels)
        self.assertTrue(any('own scope' in text for text in payload['limitations']))
        page = render(packet)
        self.assertIn('Kanun 5549 md. 3', page)
        self.assertIn('T.C. mevzuatı', page)


class WorkspaceTurkiyeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name)/'workspace', provider_factory=lambda _: RulesProvider(), source_fetcher=fixture_fetch)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                                 headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def finish(self, **changes):
        run_id = self.client.post('/api/runs', json=request_input(**changes)).json()['id']
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def test_a_turkish_run_names_the_regulation_and_reads_only_the_chosen_articles(self):
        upload = [dict(name='uyum-politikasi.md', content=base64.b64encode('Müşterinin kimliği işlem öncesinde tespit edilir.'.encode()).decode())]
        row = self.finish(policies=upload, regulator='TR', mevzuat_kind='1', mevzuat_number='5549', sections=['3', '4'], labels=[])
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        self.assertEqual((row['regulator'], row['regulation'], row['provisions']), ('TR', 'Kanun 5549', 2))
        report = self.client.get('/api/runs/'+row['id']+'/report').text
        self.assertIn('Kanun 5549 md. 3', report)
        self.assertIn('Kanun 5549 md. 4', report)
        self.assertNotIn('Kanun 5549 md. 5/A', report)
        self.assertEqual(self.client.get('/api/runs/'+row['id']+'/verify').status_code, 200)

    def test_the_turkish_choice_is_validated_before_a_job_exists(self):
        for bad in ({'regulator': 'TR', 'mevzuat_kind': '', 'mevzuat_number': '5549'}, {'regulator': 'TR', 'mevzuat_kind': '1', 'mevzuat_number': ''},
                    {'regulator': 'TR', 'mevzuat_kind': '77', 'mevzuat_number': '5549'}):
            with self.subTest(bad=bad):
                self.assertEqual(self.client.post('/api/runs', json=request_input(**bad)).status_code, 422)
        value = RunInput.model_validate(request_input(regulator='TR', mevzuat_kind='21', mevzuat_number='200713012', sections=['28'], labels=[]))
        self.assertEqual((value.module, value.chapter, value.selection, value.sections), ('YONETMELIK', '200713012', 'all', ['28']))

    def test_legislation_uploaded_as_a_policy_is_refused_with_directions(self):
        text = 'Kanun Numarası : 5549\n\n' + '\n\n'.join(f'MADDE {n} – (1) Yükümlüler, fixture yükümlülüğü {n} yerine getirmek zorundadır.' for n in range(1, 20))
        policy = {'filename': 'kanun.txt', 'chunks': [{'text': block} for block in text.split('\n\n')]}
        self.assertEqual(mevzuat_extract(policy), 19)
        self.assertEqual(mevzuat_extract({'filename': 'p.txt', 'chunks': [{'text': 'Kimlik tespiti 5549 sayılı Kanunun 3 üncü maddesine göre yapılır.'}]}), 0)
        upload = [dict(name='kanun.txt', content=base64.b64encode(text.encode()).decode())]
        turkish = dict(regulator='TR', mevzuat_kind='1', mevzuat_number='5549', sections=[], labels=[])
        refused = self.finish(policies=upload, **turkish)
        self.assertEqual(refused['state'], 'FAILED')
        self.assertIn('19 maddelik bir mevzuat metni', refused['error'])
        self.assertIn('Türkiye', refused['error'])
        self.assertEqual(self.finish(policies=upload, allow_regulatory_text=True, **turkish)['state'], 'COMPLETED')

    def test_sample_scenarios_ship_with_their_policies_and_the_page_offers_them(self):
        scenarios = self.client.get('/api/samples').json()['scenarios']
        ids = [s['id'] for s in scenarios]
        self.assertIn('tr-masak-5549', ids)
        self.assertIn('uk-cobs4-promotions', ids)
        turkish = next(s for s in scenarios if s['id'] == 'tr-masak-5549')
        self.assertEqual(len(turkish['policies']), 3)
        self.assertEqual(turkish['regulation']['mevzuat_number'], '5549')
        decoded = base64.b64decode(turkish['policies'][0]['content']).decode('utf-8')
        self.assertIn('SENTETİK TEST BELGESİ', decoded)
        page = self.client.get('/').text
        for needle in ("json('samples')", 'mevzuat/search?q=', 'gazette?days=', 'id="regulator"', 'Ufuk taraması'):
            self.assertIn(needle, page)


class GazetteTests(unittest.TestCase):
    INDEX = ('<html><body><a href="20260922.pdf"></a><a href="20260922-1.htm">&nbsp;–– Sosyal Güvenlik Kurumu Personeli Yönetmeliğinde Değişiklik Yapılmasına Dair Yönetmelik</a>'
             '<a href="20260922-2.htm">–– Mali Suçları Araştırma Kurulu Genel Tebliği (Sıra No: 31)</a>'
             '<a href="http://www.resmigazete.gov.tr/main.aspx?home=http://www.resmigazete.gov.tr/ilanlar/eskiilanlar/2026/09/20260922.htm">a - Yargı İlânları</a></body></html>')

    def test_the_daily_index_becomes_typed_items_and_the_scan_filters_by_title(self):
        page = Download('https://www.resmigazete.gov.tr/x', 'https://www.resmigazete.gov.tr/x', self.INDEX.encode('cp1254'), 'text/html', datetime.now(timezone.utc))
        with patch('regchain.pilot.turkiye.fetch', return_value=page):
            items = gazette_day(date(2026, 9, 22))
            scan = gazette_scan(2, 'tebliğ', today=date(2026, 9, 22))
        self.assertEqual([(i['kind'], i['title'][:30]) for i in items],
                         [('YONETMELIK', 'Sosyal Güvenlik Kurumu Persone'), ('TEBLIG', 'Mali Suçları Araştırma Kurulu ')])
        self.assertEqual(items[0]['url'], 'https://www.resmigazete.gov.tr/eskiler/2026/09/20260922-1.htm')
        self.assertEqual([i['title'][:20] for i in scan['items']], ['Mali Suçları Araştır', 'Mali Suçları Araştır'])
        self.assertEqual(scan['checked'], [{'date': '2026-09-22', 'items': 2}, {'date': '2026-09-21', 'items': 2}])

    def test_a_day_without_a_gazette_is_empty_not_an_error(self):
        with patch('regchain.pilot.turkiye.fetch', side_effect=IngestionError('Source returned HTTP 404')):
            self.assertEqual(gazette_day(date(2026, 9, 21)), [])


if __name__ == '__main__':
    unittest.main()

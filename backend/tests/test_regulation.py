"""Any FCA Handbook chapter can be selected; the pilot is no longer bound to CONC 7.3.4."""
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
from regchain.extraction.providers import RulesProvider
from regchain.ingestion.handbook import parse_handbook
from regchain.pilot.engine import analyze
from regchain.pilot.report import render
from regchain.pilot.sources import application_rows, chapter_url, load_sources, save_sources, section_of, select_targets
from regchain.pilot.workspace import RunInput, Workspace, create_app
from test_consolidated import card, page
from test_consolidated_postgres import handbook_download
from test_pilot import FixtureProvider, company, policies
from test_workspace import request_input

CHAPTER = 'https://handbook.fca.org.uk/handbook/cobs4'


def chapter_page():
    # Fixture wording, not FCA law. Shapes follow the live chapter page: every section on
    # one page, a negative provision number, a guidance paragraph and a deleted provision.
    return handbook_download(page(
        card('COBS 4.1.1', 'R', '01/11/2007', '<p>This chapter applies to a firm communicating with a client.</p>'),
        card('COBS 4.1.2', 'G', '01/11/2007', '<p>Guidance on application, shown after the rule.</p>'),
        card('COBS 4.2.1', 'R', '01/11/2007', '<p>A firm must retain records.</p>'),
        card('COBS 4.2.2', 'G', '01/11/2007', '<p>A firm should review records.</p>'),
        card('COBS 4.5A.3', 'R', '03/01/2018', '<p>A firm must retain client records.</p>'),
        card('COBS 4.7.-2', 'R', '01/02/2023', '<p>A firm must retain archive records.</p>'),
        card('COBS 4.7.9', 'R', '01/02/2023', '<p>[deleted]</p>')), CHAPTER)


def chapter_fixture(directory, **_):
    return save_sources(directory, [chapter_page()])


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        chapter_fixture(self.root/'cobs4')
        _, self.sections = load_sources(self.root/'cobs4')

    def test_one_url_serves_a_whole_chapter_and_bad_input_never_reaches_the_network(self):
        self.assertEqual(chapter_url('cobs', '4'), CHAPTER)
        self.assertEqual(chapter_url(' CONC ', '5d'), 'https://handbook.fca.org.uk/handbook/conc5d')
        for module, chapter in [('COBS4', '4'), ('../x', '4'), ('COBS', '4/../../etc'), ('', '4'), ('COBS', ''), ('COBS', '123')]:
            with self.subTest(module=module, chapter=chapter), self.assertRaises(ValueError):
                chapter_url(module, chapter)

    def test_negative_provision_numbers_no_longer_reject_the_whole_chapter(self):
        labels = [p.printed_label for p in parse_handbook(chapter_page()).paragraphs]
        self.assertIn('COBS 4.7.-2', labels)
        self.assertEqual(section_of('COBS 4.7.-2'), 'COBS 4.7')
        self.assertEqual(section_of('COBS 4.5A.3'), 'COBS 4.5A')

    def test_a_page_that_delivers_another_chapter_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'başka bir bölüm'):
            save_sources(self.root/'wrong', [chapter_page()], module='COBS', chapter='9')
        save_sources(self.root/'right', [chapter_page()], module='cobs', chapter='4')

    def test_rules_selection_skips_application_guidance_and_deleted_provisions(self):
        self.assertEqual(select_targets(self.sections, 'COBS', '4', 'rules'), ['COBS 4.2.1', 'COBS 4.5A.3', 'COBS 4.7.-2'])
        self.assertEqual(select_targets(self.sections, 'cobs', '4', 'all'),
                         ['COBS 4.2.1', 'COBS 4.2.2', 'COBS 4.5A.3', 'COBS 4.7.-2'])

    def test_sections_narrow_the_selection_and_may_name_the_application_section(self):
        self.assertEqual(select_targets(self.sections, 'COBS', '4', 'all', ['4.2']), ['COBS 4.2.1', 'COBS 4.2.2'])
        self.assertEqual(select_targets(self.sections, 'COBS', '4', 'rules', ['COBS 4.5A', '7']), ['COBS 4.5A.3', 'COBS 4.7.-2'])
        self.assertEqual(select_targets(self.sections, 'COBS', '4', 'rules', ['4.1']), ['COBS 4.1.1'])
        with self.assertRaisesRegex(ValueError, 'COBS 4.99'):
            select_targets(self.sections, 'COBS', '4', 'rules', ['4.99'])

    def test_named_provisions_must_exist_and_a_missing_chapter_says_what_to_do(self):
        self.assertEqual(select_targets(self.sections, 'COBS', '4', 'labels', labels=['COBS 4.7.-2', 'COBS 4.2.1']),
                         ['COBS 4.2.1', 'COBS 4.7.-2'])
        with self.assertRaisesRegex(ValueError, 'COBS 4.2.99'):
            select_targets(self.sections, 'COBS', '4', 'labels', labels=['COBS 4.2.99'])
        with self.assertRaisesRegex(ValueError, 'indir'):
            select_targets(self.sections, 'CONC', '7', 'rules')

    def test_scope_evidence_is_the_chapters_own_application_section_rules_first(self):
        rows = application_rows(self.sections, 'COBS', '4')
        self.assertEqual([r['printed_label'] for r in rows], ['COBS 4.1.1', 'COBS 4.1.2'])
        long = [dict(s, text='x'*2990) if s['printed_label'] == 'COBS 4.1.1' else s for s in self.sections]
        self.assertEqual([r['printed_label'] for r in application_rows(long, 'COBS', '4')], ['COBS 4.1.1'])
        self.assertEqual(application_rows(self.sections, 'COBS', '9'), [])


class ChapterAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        chapter_fixture(Path(self.temp.name)/'cobs4')
        _, self.sections = load_sources(Path(self.temp.name)/'cobs4')
        self.targets = select_targets(self.sections, 'COBS', '4', 'rules')

    def test_a_non_conc_chapter_is_analysed_against_its_own_application_section(self):
        payload = analyze(company(), policies(), self.sections, FixtureProvider(), self.targets)['events'][0]['payload']
        regulation = payload['regulation']
        self.assertEqual({k: regulation[k] for k in ('regulator', 'module', 'chapter', 'selected', 'processed', 'assessed', 'stopped_early')},
                         {'regulator': 'FCA', 'module': 'COBS', 'chapter': '4', 'selected': 3, 'processed': 3, 'assessed': 3, 'stopped_early': False})
        # v0.16: the packet names the targets and the helper provisions read around them.
        self.assertEqual((regulation['targets'], regulation['target_provisions']), (self.targets, 3))
        self.assertEqual([h['label'] for h in regulation['helper_provisions'] if h['role'] == 'scope'], ['COBS 4.1.1', 'COBS 4.1.2'])
        self.assertTrue({h['role'] for h in regulation['helper_provisions']} <= {'scope', 'context'})
        self.assertFalse(set(self.targets) & {h['label'] for h in regulation['helper_provisions']})   # a target is never a helper
        self.assertEqual([s['printed_label'] for s in payload['scope_sources']], ['COBS 4.1.1', 'COBS 4.1.2'])
        self.assertEqual([o['source_label'] for o in payload['obligations']], ['COBS 4.2.1', 'COBS 4.5A.3', 'COBS 4.7.-2'])
        self.assertEqual(payload['obligations'][0]['proposal']['applicability'], 'APPLIES')
        self.assertTrue(any('application section' in text for text in payload['limitations']))

    def test_one_analysis_covers_one_chapter(self):
        other = dict(self.sections[0], id='x', printed_label='CONC 7.3.4', text='A firm must retain records.')
        with self.assertRaisesRegex(ValueError, 'one Handbook chapter'):
            analyze(company(), policies(), [*self.sections, other], FixtureProvider(), ['COBS 4.2.1', 'CONC 7.3.4'])

    def test_progress_is_reported_and_a_stop_keeps_the_provisions_already_done(self):
        seen = []
        packet = analyze(company(), policies(), self.sections, FixtureProvider(), self.targets,
                         progress=lambda done, total, label, phase: seen.append((done, total, label, phase)),
                         should_stop=lambda: len(seen) >= 2)
        payload = packet['events'][0]['payload']
        # Every duty is extracted before any is judged; a stop ends both passes.
        self.assertEqual(seen, [(0, 3, 'COBS 4.2.1', 'extraction'), (1, 3, 'COBS 4.5A.3', 'extraction'),
                                (0, 2, 'COBS 4.2.1', 'assessment'), (1, 2, 'COBS 4.5A.3', 'assessment')])
        self.assertEqual({k: payload['regulation'][k] for k in ('processed', 'assessed', 'stopped_early')},
                         {'processed': 2, 'assessed': 0, 'stopped_early': True})
        self.assertEqual(len(payload['cases']), 2)
        self.assertTrue(all(o['proposal']['coverage_reason'].startswith('Analysis stopped') for o in payload['obligations']))
        self.assertIn('analiz durduruldu', render(packet))

    def test_a_stop_during_the_second_pass_keeps_the_assessments_already_made(self):
        seen = []
        payload = analyze(company(), policies(), self.sections, FixtureProvider(), self.targets,
                          progress=lambda done, total, label, phase: seen.append(phase),
                          should_stop=lambda: seen.count('assessment') >= 1)['events'][0]['payload']
        self.assertEqual({k: payload['regulation'][k] for k in ('processed', 'assessed', 'stopped_early')},
                         {'processed': 3, 'assessed': 1, 'stopped_early': True})
        self.assertEqual([o['proposal']['coverage'] for o in payload['obligations']], ['COVERS_TEXT', 'UNKNOWN', 'UNKNOWN'])

    def test_a_separate_judge_makes_every_judgement_and_the_extraction_model_none(self):
        class Extractor(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                raise AssertionError('the extraction model must not judge company documents')

        class Judge(FixtureProvider):
            name, model_version, judged = 'judge-fixture', 'judge@1', []

            def generate(self, text):
                raise AssertionError('the judge must not extract duties')

            def _chat(self, prompt, payload, schema):
                fields = schema['properties']
                Judge.judged.append('scope' if 'applicability' in fields else 'conflict' if 'contradicts' in fields else 'support')
                return super()._chat(prompt, payload, schema)
        one = analyze(company(), policies(), self.sections, Extractor(), ['COBS 4.2.1'], judge=Judge())['events'][0]['payload']
        self.assertEqual(one['obligations'][0]['proposal']['coverage'], 'COVERS_TEXT')
        self.assertEqual(Judge.judged, ['scope', 'conflict', 'support'])       # narrow questions, nothing joint
        self.assertEqual(one['judge_runtime'], {'provider': 'judge-fixture', 'model_version': 'judge@1', 'votes': 1})
        same = analyze(company(), policies(), self.sections, FixtureProvider(), ['COBS 4.2.1'])['events'][0]['payload']
        self.assertEqual(same['judge_runtime'], {**same['runtime'], 'votes': 1})
        self.assertNotEqual(one['input_hash'], same['input_hash'])

    def test_overview_lists_every_provision_with_counts_and_the_reason_for_no_candidate(self):
        page_text = render(analyze(company(), policies(), self.sections, FixtureProvider(),
                                   select_targets(self.sections, 'COBS', '4', 'all', ['4.2', '4.1'])))
        self.assertIn('Bölüm özeti', page_text)
        self.assertIn('FCA COBS 4', page_text)
        self.assertIn('Aday üretilemedi', page_text)      # the application rule has no modal duty
        self.assertIn('href="#o-', page_text)
        self.assertIn('uyum oranı değildir', page_text)

    def test_long_policies_do_not_repeat_every_passage_under_every_card(self):
        many = [dict(policies()[0], chunks=[dict(policies()[0]['chunks'][0], source_id=f'c{i}', text=f'Staff must retain records in archive {i}.')
                                            for i in range(40)])]
        class Selective(FixtureProvider):
            """Every passage is read, and like a real policy most are about something else."""
            def passage(self, payload):
                if payload['passage'].endswith(('archive 3.', 'archive 37.')):
                    return super().passage(payload)
                return {'relation': 'UNRELATED', 'quote': '', 'reason': 'Fixture.'}
        packet = analyze(company(), many, self.sections, Selective(), ['COBS 4.2.1'])
        self.assertEqual(len(packet['events'][0]['payload']['obligations'][0]['judged_policy_ids']), 40)
        page_text = render(packet)
        self.assertIn('bu yükümlülük için sıralanan pasajlar', page_text)
        self.assertLess(page_text.count('data-policy='), 12)
        self.assertIn('data-policy="c37"', page_text)          # decided far below the ranking, still offered as evidence
        self.assertIn('40 pasajın 38 tanesi bu yükümlülükle ilgisiz bulundu', page_text)


class MultipartProvisionTests(unittest.TestCase):
    RULE = ('(1) A firm must retain records. (2) This rule applies in relation to: (a) a communication by the firm; '
            '(b) a financial promotion under COBS 4.2.1R(2). [Note: article 24] (3) A firm must retain archive records.')

    def test_only_top_level_numbering_from_the_first_character_is_cut(self):
        from regchain.pilot.engine import split_units
        units = split_units(self.RULE)
        self.assertEqual([offset for offset, _ in units], [0, self.RULE.index('(2) This'), self.RULE.index('(3) A firm')])
        self.assertTrue(all(self.RULE[offset:offset+len(unit)] == unit for offset, unit in units))
        listed = 'A firm must not pressurise a customer: (1) to pay in one repayment; (2) to pay within a short period.'
        self.assertEqual(split_units(listed), [(0, listed)])
        self.assertEqual(split_units('(1) A single numbered paragraph.'), [(0, '(1) A single numbered paragraph.')])
        self.assertEqual(len(split_units('(1) First duty. (3) Numbering that does not run on is not a cut.')), 1)

    def test_each_sub_paragraph_is_extracted_on_its_own_and_the_row_says_so(self):
        sections = [dict(s) for s in __import__('test_pilot').sections()]
        sections[1]['text'] = self.RULE
        calls = []

        class Counting(FixtureProvider):
            def generate(self, text):
                calls.append(text)
                return super().generate(text)
        packet = analyze(company(), policies(), sections, Counting(), ['CONC 7.3.4'])
        payload = packet['events'][0]['payload']
        # The scope sub-paragraph has no modal wording: it costs no model call.
        self.assertEqual(calls, ['(1) A firm must retain records.', '(3) A firm must retain archive records.'])
        self.assertEqual([o['candidate']['required_action'] for o in payload['obligations']], ['retain records.', 'retain archive records.'])
        self.assertEqual([o['multipart'] for o in payload['obligations']],
                         [{'unit_offset': 0, 'units': 3}, {'unit_offset': self.RULE.index('(3) A firm'), 'units': 3}])
        self.assertEqual(payload['cases'][0]['unit_count'], 3)
        self.assertIn('Çok parçalı provision', render(packet))
        self.assertTrue(any('multipart' in text for text in payload['limitations']))

    def test_statute_citations_are_not_unresolved_handbook_references(self):
        from regchain.extraction.grounding import CROSS_REFERENCE
        self.assertIsNone(CROSS_REFERENCE.search('a right of action under section 138D of the Act.'))
        self.assertIsNone(CROSS_REFERENCE.search('within section 21 of the Act'))
        for text in ('as set out in section 7.3 of this chapter', 'see COBS 4.2.1R', 'in accordance with the rules', 'paragraph 3 applies'):
            with self.subTest(text=text):
                self.assertIsNotNone(CROSS_REFERENCE.search(text))

    def test_a_condition_ends_at_the_next_numbered_sub_paragraph(self):
        from regchain.extraction.grounding import CONDITIONS, qualifier_spans
        text = ('(1) This rule applies where: (a) a firm has put in place a repayment arrangement; and (b) the customer is '
                'meeting its terms. (2) The firm must reduce any further interest.')
        self.assertEqual(qualifier_spans(text, CONDITIONS),
                         ['where: (a) a firm has put in place a repayment arrangement; and (b) the customer is meeting its terms.'])

    def test_optional_context_can_never_crowd_out_the_provision(self):
        from regchain.extraction.retrieval import OPTIONAL_CHARS, retrieve
        from test_consolidated import section
        target = section('t', 'A firm must retain records of each compensation scheme reference.', label='7.3.4')
        table = section('annex', 'compensation scheme ' * (OPTIONAL_CHARS // 10), label='7.3.5')
        small = section('near', 'A compensation scheme reference is a factual reference.', label='7.3.6')
        packet = retrieve(target, [target, table, small])
        self.assertNotIn('annex', [item['section_id'] for item in packet.items])
        self.assertFalse(packet.truncated)


class WorkspaceRegulationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.requested = []

        def fetcher(directory, **where):
            self.requested.append(where)
            return chapter_fixture(directory)
        self.workspace = Workspace(Path(self.temp.name)/'workspace', provider_factory=lambda _: RulesProvider(), source_fetcher=fetcher)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767',
                                 client=('127.0.0.1', 51000), headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def finish(self, **changes):
        response = self.client.post('/api/runs', json=request_input(**changes))
        self.assertEqual(response.status_code, 202, response.text)
        for _ in range(300):
            row = self.workspace.metadata(response.json()['id'])
            if row['state'] in ('COMPLETED', 'FAILED', 'INTERRUPTED') and not self.workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def test_a_whole_chapter_is_fetched_selected_and_reported_with_progress(self):
        row = self.finish(module='cobs', chapter='4', selection='rules', labels=[])
        self.assertEqual(row['state'], 'COMPLETED', row)
        self.assertEqual(self.requested, [{'module': 'cobs', 'chapter': '4'}])
        self.assertEqual((row['regulation'], row['provisions'], row['stopped_early']), ('COBS 4', 3, False))
        self.assertEqual(row['progress'], {'done': 3, 'total': 3, 'label': None, 'phase': 'done'})
        self.assertIn('Bölüm özeti', self.client.get(f'/api/runs/{row["id"]}/report').text)
        self.assertEqual(self.client.get(f'/api/runs/{row["id"]}/verify').json()['integrity'], 'VERIFIED')

    def test_a_selection_that_matches_nothing_fails_with_a_reason_not_an_empty_success(self):
        row = self.finish(module='COBS', chapter='4', selection='rules', sections=['4.99'], labels=[])
        self.assertEqual(row['state'], 'FAILED')
        self.assertIn('COBS 4.99', row['error'])
        row = self.finish(module='CONC', chapter='7', selection='labels', labels=['CONC 7.3.4'])
        self.assertIn('CONC 7', row['error'])

    def test_stop_is_only_accepted_for_the_running_analysis(self):
        row = self.finish(module='COBS', chapter='4', selection='labels', labels=['COBS 4.2.1'])
        self.assertEqual(self.client.post(f'/api/runs/{row["id"]}/stop', json={}).status_code, 409)

    def test_printed_fca_locators_are_accepted_and_everything_else_is_not(self):
        RunInput.model_validate(request_input(labels=['COBS 4.7.-2', 'CONC 7.3.5-A', 'COBS 4.5A.4', 'SYSC 10A.1.11']))
        for bad in (['COBS 4'], ['cobs 4.2.1'], ['COBS 4.2.1; rm'], ['COBS 4.2.1', 'COBS 4.2.1']):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                RunInput.model_validate(request_input(labels=bad))
        for module, chapter in (('COBS4', '4'), ('COBS', '4.2'), ('../x', '4')):
            with self.subTest(module=module), self.assertRaises(ValidationError):
                RunInput.model_validate(request_input(module=module, chapter=chapter))


if __name__ == '__main__':
    unittest.main()

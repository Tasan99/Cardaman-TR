import copy
import json
import tempfile
import unittest
import zipfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from regchain.evidence import digest, verify_chain
from regchain.extraction.providers import RulesProvider
from regchain.pilot.engine import analyze, propose, load_packet, duty_changes
from regchain.pilot.evaluation import evaluate, label_template
from regchain.pilot.policies import read_policy, select_chunks
from regchain.pilot.report import render
from regchain.pilot.review import apply_review
from regchain.pilot.schema import Company, Review
from regchain.pilot.sources import change_for, load_sources, save_sources
from regchain.pilot.verification import verify_artifacts
from test_consolidated import card, page
from test_consolidated_postgres import handbook_download


def company():
    return Company(id='fixture', name='Fixture company', version='1', synthetic=True,
        jurisdictions=['UK'], activities=['consumer credit lending'], licences=['synthetic permission'],
        products=['loans'], customer_types=['individual borrowers'], description='Synthetic fixture')


def sections():
    common = dict(source_kind='CONSOLIDATED', quality_flags=[], heading_path=[], defined_terms=[],
        legal_type='RULE', locator_kind='handbook_provision', effective_from='2024-01-01',
        source_url='https://handbook.fca.org.uk/handbook/conc7/conc7s3', fetched_at='2026-09-20T00:00:00+00:00',
        external_key='FIXTURE', version_hash='a'*64, ordinal=0)
    return [dict(common, id='s-scope', version_id='v-scope', section_number='7.1', paragraph_number='1',
                 printed_label='CONC 7.1.1', text='This chapter applies to consumer credit lending.', content_hash='b'*64),
            dict(common, id='s-duty', version_id='v-duty', section_number='7.3', paragraph_number='4',
                 printed_label='CONC 7.3.4', text='A firm must retain records.', content_hash='c'*64)]


def policies():
    return [{'filename': 'fixture.txt', 'raw_hash': 'd'*64, 'bytes': 30, 'parser': 'fixture',
             'chunks': [{'source_id': 'policy-1', 'policy_hash': 'd'*64, 'filename': 'fixture.txt',
                         'locator': 'text_block', 'number': 1, 'start': 0, 'end': 28, 'text': 'All staff must retain records.'}]}]


class FixtureProvider(RulesProvider):
    name = 'fixture'
    base_url = 'http://localhost:11434'

    def _chat(self, prompt, payload, schema):
        fields = schema['properties']
        if 'summary' in fields:
            return json.dumps({'summary': 'Fixture summary of the duty.', 'category': fields['category']['enum'][0]})
        if 'answer' in fields:
            src = (payload.get('sources') or [None])[0]
            return json.dumps({'answer': 'Fixture answer.', 'citations': [{'source_id': src['source_id'], 'quote': src['text'][:40]}] if src else []})
        if 'clause' in fields:
            return json.dumps({'clause': 'Fixture policy clause requiring the duty.', 'notes': 'fixture'})
        if 'contradicts' in fields:
            answer = self.passage(payload)
            verdict = {'CONFLICTS': 'YES', 'UNCLEAR': 'UNCLEAR'}.get(answer['relation'], 'NO')
            return json.dumps({'contradicts': verdict, 'quote': answer['quote'] if verdict == 'YES' else '', 'reason': answer['reason']})
        if 'verdict' in fields:
            # The second reading of an alleged contradiction (v0.16) agrees with this fixture's first reading.
            answer = self.passage(payload)
            verdict = 'CONTRADICTS' if answer['relation'] == 'CONFLICTS' else 'NOT_A_CONTRADICTION'
            return json.dumps({'verdict': verdict, 'duty_requires': 'the duty', 'passage_instructs': 'the passage', 'reason': answer['reason']})
        if 'supports' in fields:
            answer = self.passage(payload)
            return json.dumps({'supports': answer['relation'], 'quote': answer['quote'], 'reason': answer['reason']})
        return json.dumps(self.scope(payload))

    def scope(self, payload):
        """The applicability answer: APPLIES, with one fact/condition pair copied exactly."""
        company = payload['company']
        fact = next((company[k][0] for k in ('activities', 'products', 'jurisdictions', 'licences', 'customer_types')
                     if isinstance(company.get(k), list) and company[k]), company.get('description') or company.get('name'))
        scope_text = payload['scope'][0]['text'] if payload.get('scope') else payload['provision']['text']
        source = payload['scope'][0]['source_id'] if payload.get('scope') else payload['provision']['source_id']
        return {'applicability': 'APPLIES', 'company_fact_keys': ['activities'],
                'scope_evidence': [{'source_id': source, 'quote': scope_text[:400]}],
                'basis': [{'company_fact': fact, 'regulatory_condition': scope_text[:300], 'match': 'YES'}],
                'applicability_reason': 'Synthetic activity matches fixture scope.', 'missing_information': []}

    def passage(self, payload):
        """What this fixture thinks of one passage; _chat turns it into the two narrow answers."""
        return {'relation': 'SUPPORTS', 'quote': payload['passage'], 'reason': 'Fixture passage requires what the duty requires.'}


def packet():
    return analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])


def review_input(value, coverage='COVERS_TEXT'):
    row = value['events'][0]['payload']['obligations'][0]
    return {'format': 'regchain-pilot-review-v1', 'analysis_head': value['head'],
        'reviewer': 'Fixture reviewer', 'reviewer_role': 'Synthetic test actor', 'decisions': [{
            'obligation_id': row['id'], 'extraction': 'ACCEPT', 'applicability': 'APPLIES',
            'coverage': coverage, 'company_fact_keys': ['activities'],
            'scope_evidence': [{'source_id': 's-scope', 'quote': sections()[0]['text']}],
            'policy_evidence': [{'source_id': 'policy-1', 'quote': policies()[0]['chunks'][0]['text']}],
            'rationale': 'Synthetic fixture decision; not legal review.'}]}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_text_citations_preserve_exact_extracted_substrings_and_hash(self):
        path = self.root/'policy.txt'
        path.write_text('Başlık\n\nWe retain records.\n', encoding='utf-8')
        value = read_policy(path)
        self.assertEqual(value['chunks'][1]['text'], 'We retain records.')
        self.assertEqual(len(value['raw_hash']), 64)
        self.assertEqual(select_chunks('retain records', [value])[0]['text'], 'We retain records.')

    def test_word_uses_paragraph_locators_not_invented_pages(self):
        path = self.root/'policy.docx'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Retain records.</w:t></w:r></w:p></w:body></w:document>')
        value = read_policy(path)
        self.assertEqual(value['chunks'][0]['locator'], 'docx_paragraph')
        self.assertEqual(value['chunks'][0]['text'], 'Retain records.')

    def test_scanned_or_blank_pdf_is_not_silently_empty_policy(self):
        path = self.root/'blank.pdf'
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.write(path)
        with self.assertRaisesRegex(ValueError, 'OCR'):
            read_policy(path)

    def test_unrelated_policy_has_no_invented_retrieval_match(self):
        self.assertEqual(select_chunks('forbearance', policies()), [])


class PilotTests(unittest.TestCase):
    def test_cli_builds_complete_review_artifacts_without_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = handbook_download(page(
                card('CONC 7.1.1', 'R', '04/11/2024', '<p>This chapter applies to lenders.</p>'),
                card('CONC 7.3.4', 'R', '04/11/2024', '<p>A firm must retain records.</p>')))
            save_sources(root/'sources', [source])
            (root/'company.json').write_text(company().model_dump_json(), encoding='utf-8')
            (root/'policy.txt').write_text('Staff must retain records.', encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'regchain.pilot', 'run',
                '--company', str(root/'company.json'), '--policy', str(root/'policy.txt'),
                '--sources', str(root/'sources'), '--output', str(root/'result'), '--provider', 'rules',
                '--label', 'CONC 7.3.4'], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root/'result/review.html').exists())
            self.assertTrue((root/'result/receipt.json').exists())
            self.assertTrue((root/'result/regulatory-sources/snapshot.json').exists())
            self.assertEqual(json.loads((root/'result/evaluation.json').read_text())['status'], 'EXPERT_REVIEW_PENDING')
            value = load_packet(root/'result/packet.json')
            verify_artifacts(value, root/'result')
            retained = next((root/'result/policy-originals').iterdir())
            retained.write_bytes(b'tampered policy')
            with self.assertRaisesRegex(ValueError, 'missing or modified'):
                verify_artifacts(value, root/'result')

    def many(self, count, conflict='Records are deleted every night.'):
        texts = [f'Staff must retain records in archive {i}.' for i in range(count)] + [conflict]
        return [dict(policies()[0], chunks=[dict(policies()[0]['chunks'][0], source_id=f'c{i:03d}', text=text)
                                           for i, text in enumerate(texts)])]

    class Deleting(FixtureProvider):
        """Supports everything except a passage that deletes records."""
        def passage(self, payload):
            self.seen = [*getattr(self, 'seen', []), payload]
            if 'deleted' not in payload['passage']:
                return super().passage(payload)
            return {'relation': 'CONFLICTS', 'quote': 'Records are deleted every night.', 'reason': 'Fixture.'}

    def test_one_conflicting_passage_decides_coverage_however_many_support_it(self):
        # Observed live with one joint request: the model quoted a prohibition, labelled it
        # SUPPORTS beside four supportive passages, and approved itself as second pass.
        provider = self.Deleting()
        packet = analyze(company(), self.many(3), sections(), provider, ['CONC 7.3.4'])
        row = packet['events'][0]['payload']['obligations'][0]
        # Nothing else in sight: one passage per question; the second reading of an alleged
        # contradiction (v0.16) adds only that passage's own quoted sentence.
        self.assertTrue(all(set(payload) in ({'duty', 'passage'}, {'duty', 'passage', 'alleged_contradiction', 'first_reading'})
                            for payload in provider.seen))
        self.assertEqual(row['proposal']['coverage'], 'CONFLICT')
        self.assertIn('3 supporting passage(s) do not resolve that', row['proposal']['coverage_reason'])
        self.assertEqual(row['proposal']['applicability'], 'APPLIES')
        self.assertEqual(row['signals'], ['POTENTIAL_POLICY_CONFLICT'])
        self.assertIn({'source_id': 'c003', 'quote': 'Records are deleted every night.', 'relation': 'CONFLICTS'}, row['proposal']['policy_checks'])
        self.assertIn({'source_id': 'c003', 'quote': 'Records are deleted every night.'}, row['proposal']['policy_evidence'])
        self.assertEqual(row['status'], 'HUMAN_REVIEW_REQUIRED')
        self.assertIn('Records are deleted every night.', render(packet))

    def test_a_contradiction_that_ranks_below_the_best_passages_is_still_judged(self):
        # Measured live: the contradicting passage ranked 9 to 18 of 18, never among the six best.
        provider = self.Deleting()
        row = analyze(company(), self.many(7), sections(), provider, ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(len(row['retrieved_policy_ids']), 6)
        self.assertNotIn('c007', row['retrieved_policy_ids'])
        self.assertEqual(len(row['judged_policy_ids']), 8)                 # a small policy set is read in full
        self.assertEqual(row['judged_policy_ids'][:6], row['retrieved_policy_ids'])
        self.assertEqual((row['proposal']['coverage'], row['signals']), ('CONFLICT', ['POTENTIAL_POLICY_CONFLICT']))

    def test_a_large_policy_set_is_judged_only_down_to_the_leading_ranks(self):
        from regchain.pilot.engine import JUDGE_ALL_UNDER, JUDGE_WINDOW
        row = analyze(company(), self.many(JUDGE_ALL_UNDER + 5), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(len(row['judged_policy_ids']), JUDGE_WINDOW)
        self.assertEqual(len(row['proposal']['policy_checks']), JUDGE_WINDOW)

    def test_coverage_is_counted_from_the_passages_not_asked_of_the_model(self):
        from regchain.pilot.engine import coverage_of
        from regchain.pilot.schema import PolicyCheck
        checks = lambda *relations: [PolicyCheck(source_id=f'p{i}', quote='q', relation=r) for i, r in enumerate(relations)]
        for relations, expected in [((), 'NO_EVIDENCE'), (('UNRELATED', 'UNRELATED'), 'NO_EVIDENCE'), (('PARTIAL', 'UNRELATED'), 'PARTIAL'),
                                    (('SUPPORTS', 'PARTIAL'), 'COVERS_TEXT'), (('SUPPORTS', 'UNCLEAR'), 'UNKNOWN'),
                                    (('SUPPORTS', 'SUPPORTS', 'UNCLEAR', 'CONFLICTS'), 'CONFLICT')]:
            with self.subTest(relations=relations):
                self.assertEqual(coverage_of(checks(*relations))[0], expected)
        self.assertIn('not proof of a gap', coverage_of(checks())[1])

    def test_an_invented_quote_is_retried_once_and_then_blocks_a_favourable_verdict(self):
        class Vague(FixtureProvider):
            calls = 0

            def passage(self, payload):
                Vague.calls += 1
                return {'relation': 'CONFLICTS', 'quote': 'A sentence the policy never contained', 'reason': 'Fixture.'}
        row = analyze(company(), policies(), sections(), Vague(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(Vague.calls, 2)
        self.assertEqual(row['proposal']['coverage'], 'UNKNOWN')            # not a conflict without its sentence,
        self.assertEqual(row['signals'], [])                                # and not cleared either
        self.assertEqual(row['proposal']['policy_evidence'], [])
        notes = row['diagnostics'][-1]['results'][0]['notes']
        self.assertEqual([(n['question'], n['code']) for n in notes], [('contradicts', 'JUDGEMENT_INVALID')]*2)

    def test_reasoning_is_spent_on_contradiction_and_on_confirming_a_favourable_answer(self):
        asked = []

        class Quick(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                asked.append(('quick', sorted(schema['properties'])[0]))
                return super()._chat(prompt, payload, schema)

        class Reasoning(self.Deleting):
            def _chat(self, prompt, payload, schema):
                asked.append(('reasoning', sorted(schema['properties'])[0]))
                return super()._chat(prompt, payload, schema)
        judge = Reasoning()
        judge.quick = Quick()
        row = analyze(company(), self.many(1), sections(), judge, ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        # Scope; then per passage the contradiction question, and only after a NO the support
        # question: unreasoned first, reasoned again because the answer was favourable. A YES is
        # read a second time by the reasoning judge (v0.16, 'duty_requires' is the confirmation
        # schema). The conflict leaves a gap, so the remediation draft is the last call, unreasoned.
        self.assertEqual(asked, [('reasoning', 'applicability'),
                                 ('reasoning', 'contradicts'), ('quick', 'quote'), ('reasoning', 'quote'),
                                 ('reasoning', 'contradicts'), ('reasoning', 'duty_requires'), ('quick', 'clause')])
        self.assertEqual([c['relation'] for c in row['proposal']['policy_checks']], ['SUPPORTS', 'CONFLICTS'])

    def test_a_favourable_answer_stands_only_if_the_reasoning_judge_agrees(self):
        # Measured live: without reasoning the model said a passage about reminder letters
        # "supports" a duty of forbearance.
        class Sceptical(FixtureProvider):
            def passage(self, payload):
                return {'relation': 'UNRELATED', 'quote': '', 'reason': 'The passage is about something else.'}
        judge = Sceptical()
        judge.quick = FixtureProvider()                                     # calls everything SUPPORTS
        row = analyze(company(), policies(), sections(), judge, ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((row['proposal']['coverage'], row['proposal']['policy_evidence']), ('NO_EVIDENCE', []))
        self.assertEqual([c['relation'] for c in row['proposal']['policy_checks']], ['UNRELATED'])
        # What the unreasoned model sets aside is not asked again.
        asked = []

        class Counting(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                asked.append(sorted(schema['properties'])[0])
                return super()._chat(prompt, payload, schema)
        judge = Counting()
        judge.quick = Sceptical()
        analyze(company(), policies(), sections(), judge, ['CONC 7.3.4'])
        self.assertEqual(asked, ['applicability', 'contradicts'])

    def test_a_judge_outage_fails_closed_without_ending_the_run(self):
        from regchain.extraction.providers import ProviderFailure

        class Outage(FixtureProvider):
            def passage(self, payload):
                raise ProviderFailure('fixture outage')
        row = analyze(company(), self.many(2), sections(), Outage(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['coverage'], 'UNKNOWN')
        self.assertEqual(row['proposal']['coverage_reason'], 'Analysis unavailable: ProviderFailure')
        self.assertEqual(row['proposal']['applicability'], 'APPLIES')       # the scope question still succeeded
        self.assertEqual({c['relation'] for c in row['proposal']['policy_checks']}, {'UNCLEAR'})

        class NoScope(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'applicability' in schema['properties']:
                    raise ProviderFailure('fixture outage')
                return super()._chat(prompt, payload, schema)
        row = analyze(company(), policies(), sections(), NoScope(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((row['proposal']['applicability'], row['proposal']['coverage']), ('UNKNOWN', 'COVERS_TEXT'))
        self.assertEqual(row['proposal']['applicability_reason'], 'Analysis unavailable: ProviderFailure')

    def test_pdf_line_breaks_inside_a_quote_are_tolerated_but_changed_words_are_not(self):
        from regchain.pilot.engine import source_span
        broken = [{'filename': 'p.pdf', 'raw_hash': 'd'*64, 'bytes': 1, 'parser': 'fixture', 'chunks': [{
            'source_id': 'policy-1', 'policy_hash': 'd'*64, 'filename': 'p.pdf', 'locator': 'pdf_page', 'number': 1,
            'start': 0, 'end': 31, 'text': 'All staff must \nretain  records.'}]}]

        class Reflowed(FixtureProvider):
            def passage(self, payload):
                return dict(super().passage(payload), quote='All staff must retain records.')
        row = analyze(company(), broken, sections(), Reflowed(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['coverage'], 'COVERS_TEXT')
        # What is retained is the source's own characters, line break included.
        self.assertEqual(row['proposal']['policy_evidence'][0]['quote'], 'All staff must \nretain  records.')
        self.assertIn(row['proposal']['policy_checks'][0]['quote'], broken[0]['chunks'][0]['text'])
        self.assertIsNone(source_span('All staff should retain records.', broken[0]['chunks'][0]['text']))
        self.assertIsNone(source_span('   ', 'anything'))
        self.assertEqual(source_span('retain rec', 'All staff must \nretain  records.'), 'retain  rec')
        # Observed live: the 400-character schema bound mangles the final token of an exact copy.
        from regchain.pilot.schema import QUOTE_LIMIT
        head = ('word '*((QUOTE_LIMIT-6)//5)).strip()
        source = head + ' effect before completion day.'
        mangled = head + ' 效果'
        self.assertGreaterEqual(len(mangled), QUOTE_LIMIT-8)
        self.assertEqual(source_span(mangled, source), head)
        self.assertIsNone(source_span('short 效果', source))  # a short quote gets no such tolerance

    def test_an_elided_quote_names_the_source_span_between_its_fragments(self):
        from regchain.pilot.engine import source_span
        # Seen live: the judge shortened a long Turkish definition list with "...".
        text = 'd) Yükümlü: Bankacılık, sigortacılık, sermaye piyasaları, ödünç para verme ve diğer finansal hizmetler alanında faaliyet gösterenler;'
        self.assertEqual(source_span('d) Yükümlü: Bankacılık... finansal hizmetler... faaliyet gösterenler;', text), text)
        self.assertEqual(source_span('Bankacılık, sigortacılık… ödünç para', text), 'Bankacılık, sigortacılık, sermaye piyasaları, ödünç para')
        self.assertIsNone(source_span('Bankacılık... yoktur böyle bir şey', text))
        # Over the limit, the longest fragment of at least six words stands in for the span.
        long = 'd) Yükümlü: ' + 'x '*300 + 'ödünç para verme ve diğer finansal hizmetler alanında faaliyet gösterenler;'
        self.assertEqual(source_span('d) Yükümlü... ödünç para verme ve diğer finansal hizmetler alanında', long),
                         'ödünç para verme ve diğer finansal hizmetler alanında')
        self.assertIsNone(source_span('d) Yükümlü... faaliyet gösterenler;', long))
        # A quote that silently skips from a list's opening to its item names the copy at its end.
        listed = '(1) Bu Kanunda geçen; a) Bakanlık: Maliye Bakanlığını, b) Bakan: Maliye Bakanını, d) Yükümlü: Bankacılık, sigortacılık ve ödeme hizmetleri alanında faaliyet gösterenleri,'
        self.assertEqual(source_span('(1) Bu Kanunda geçen; d) Yükümlü: Bankacılık, sigortacılık ve ödeme hizmetleri alanında faaliyet gösterenleri,', listed),
                         'd) Yükümlü: Bankacılık, sigortacılık ve ödeme hizmetleri alanında faaliyet gösterenleri,')
        self.assertIsNone(source_span('(1) Bu Kanunda geçen; d) Yükümlü: uydurma', listed))

    def test_model_quotes_are_bounded_but_a_reviewer_may_cite_a_whole_passage(self):
        from pydantic import ValidationError
        from regchain.pilot.schema import ModelQuote, Quote, QUOTE_LIMIT
        Quote(source_id='p', quote='x'*2000)
        with self.assertRaises(ValidationError):
            ModelQuote(source_id='p', quote='x'*(QUOTE_LIMIT+1))

    def test_every_passage_handed_to_the_judge_is_accounted_for(self):
        row = analyze(company(), self.many(4), sections(), FixtureProvider(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual([c['source_id'] for c in row['proposal']['policy_checks']], row['judged_policy_ids'])
        self.assertEqual(row['diagnostics'][-1]['judged'], 5)

    def test_missing_extraction_never_means_old_duties_were_removed(self):
        before = {'status': 'EXTRACTED', 'obligations': [{'subject': 'A firm', 'modality': 'MUST', 'required_action': 'retain records.'}]}
        current = {'status': 'INSUFFICIENT_EVIDENCE', 'obligations': []}
        self.assertEqual(duty_changes(current, before)['status'], 'EXTRACTION_INCOMPLETE')
        self.assertEqual(duty_changes(current, before)['removed'], [])

    def test_changed_duty_is_presented_for_review(self):
        before = {'status': 'EXTRACTED', 'obligations': [{'subject': 'A firm', 'modality': 'MUST', 'required_action': 'retain records.'}]}
        after = copy.deepcopy(before)
        after['obligations'][0]['required_action'] = 'review records.'
        result = duty_changes(after, before)
        self.assertEqual(result['status'], 'CANDIDATES_CHANGED_REVIEW_REQUIRED')
        self.assertEqual(len(result['added']), 1)
        self.assertEqual(len(result['removed']), 1)

    def test_end_to_end_proposal_is_review_only_and_ids_are_backend_owned(self):
        result = packet()
        row = result['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['status'], 'HUMAN_REVIEW_REQUIRED')
        self.assertEqual(row['proposal']['policy_evidence'][0]['source_id'], 'policy-1')
        self.assertEqual(row['proposal']['scope_evidence'][0]['source_id'], 's-scope')
        self.assertTrue(verify_chain(result['events'], result['head'], 1))

    def test_missing_company_information_forces_unknown(self):
        profile = company()
        profile.licences = None
        result = analyze(profile, policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        self.assertEqual(result['events'][0]['payload']['obligations'][0]['proposal']['applicability'], 'UNKNOWN')

    def test_a_stray_fact_key_is_dropped_and_noted_but_an_answer_without_facts_is_not(self):
        # Seen live: the model listed the 'synthetic' flag as a company fact in every Turkish call.
        class Stray(FixtureProvider):
            keys = ['synthetic', 'activities']

            def _chat(self, prompt, payload, schema):
                value = json.loads(super()._chat(prompt, payload, schema))
                if 'company_fact_keys' in value:
                    value['company_fact_keys'] = list(Stray.keys)
                return json.dumps(value)
        row = analyze(company(), policies(), sections(), Stray(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual((row['proposal']['applicability'], row['proposal']['company_fact_keys']), ('APPLIES', ['activities']))
        self.assertEqual([d['code'] for d in row['diagnostics'] if 'stage' not in d], ['FACT_KEYS_FILTERED'])
        Stray.keys = ['synthetic']
        row = analyze(company(), policies(), sections(), Stray(), ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'UNKNOWN')
        self.assertTrue(row['proposal']['applicability_reason'].startswith('PROPOSAL_INVALID'))

    def test_fabricated_scope_quote_is_rejected_with_bounded_repair(self):
        class Invented(FixtureProvider):
            calls = 0
            def _chat(self, prompt, payload, schema):
                value = json.loads(super()._chat(prompt, payload, schema))
                if 'scope_evidence' in value:
                    self.calls += 1
                    value['scope_evidence'][0]['quote'] = 'Invented scope clause'
                return json.dumps(value)
        provider = Invented()
        row = analyze(company(), policies(), sections(), provider, ['CONC 7.3.4'])['events'][0]['payload']['obligations'][0]
        self.assertEqual(provider.calls, 2)
        self.assertEqual(row['proposal']['applicability'], 'UNKNOWN')
        self.assertEqual(row['proposal']['applicability_reason'], 'PROPOSAL_INVALID: manual analysis required')
        self.assertEqual(row['proposal']['scope_evidence'], [])
        self.assertEqual(row['proposal']['coverage'], 'COVERS_TEXT')         # the policy question is separate and stood

    def test_private_documents_cannot_be_sent_to_remote_ollama(self):
        provider = FixtureProvider()
        provider.base_url = 'https://remote.example'
        with self.assertRaisesRegex(ValueError, 'local Ollama'):
            propose(provider, company(), sections()[1], {}, sections()[:1], policies()[0]['chunks'])

    def test_review_adds_hash_event_and_stale_review_is_rejected(self):
        first = packet()
        reviewed = apply_review(first, Review.model_validate(review_input(first)))
        self.assertTrue(verify_chain(reviewed['events'], reviewed['head'], 2))
        self.assertEqual(first['count'], 1)
        self.assertEqual(reviewed['events'][-1]['payload']['outcomes'][0]['status'], 'REVIEWED_POLICY_COVERAGE')
        wrong = review_input(first)
        wrong['analysis_head'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'different analysis'):
            apply_review(first, Review.model_validate(wrong))

    def test_no_evidence_never_becomes_confirmed_gap(self):
        value = packet()
        review = review_input(value, 'NO_EVIDENCE')
        review['decisions'][0]['policy_evidence'] = []
        result = apply_review(value, Review.model_validate(review))
        self.assertEqual(result['events'][-1]['payload']['outcomes'][0]['status'], 'POLICY_REVIEW_REQUIRED')

    def test_confirmed_gap_requires_reviewed_applicability_and_policy_evidence(self):
        value = packet()
        review = review_input(value, 'CONFLICT')
        self.assertEqual(apply_review(value, Review.model_validate(review))['events'][-1]['payload']['outcomes'][0]['status'], 'REVIEWED_POLICY_GAP')
        review['decisions'][0]['policy_evidence'] = []
        with self.assertRaisesRegex(ValueError, 'exact policy quote'):
            apply_review(value, Review.model_validate(review))

    def test_html_escapes_document_prompt_injection_as_text(self):
        value = packet()
        value['events'][0]['payload']['company']['name'] = '</script><script>alert(1)</script>'
        html = render(value)
        self.assertNotIn('</script><script>alert(1)</script>', html)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', html)
        embedded = html.split('<script type="application/json" id="data">')[1].split('</script>')[0]
        self.assertIsInstance(json.loads(embedded), dict)

    def test_company_change_reopens_review_even_with_same_regulation(self):
        old = packet()
        previous = old['events'][0]['payload']
        new_company = company()
        new_company.version = '2'
        new = analyze(new_company, policies(), sections(), FixtureProvider(), ['CONC 7.3.4'],
            [c['source'] for c in previous['cases']], old['head'], previous)
        result = new['events'][0]['payload']
        self.assertTrue(result['input_changes']['company_changed'])
        self.assertEqual(result['cases'][0]['change']['status'], 'TEXT_UNCHANGED')
        self.assertEqual(result['obligations'][0]['status'], 'HUMAN_REVIEW_REQUIRED')

    def test_hash_tampering_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'packet.json'
            value = packet()
            value['events'][0]['payload']['company']['name'] = 'tampered'
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_packet(path)

    def test_missing_previous_source_does_not_invent_regulatory_change(self):
        self.assertEqual(change_for(sections()[1], None)['status'], 'NO_BASELINE')

    def test_unreviewed_labels_produce_no_accuracy_claim(self):
        value = packet()
        result = evaluate(value, label_template(value))
        self.assertEqual(result['status'], 'EXPERT_REVIEW_PENDING')
        self.assertIsNone(result['extraction_exact_span']['precision'])
        self.assertIsNone(result['decisions']['coverage']['accuracy'])

    def test_exact_gold_metrics_count_a_missed_obligation(self):
        value = packet()
        value['events'][0]['payload']['cases'][0]['output'] = {'status': 'INSUFFICIENT_EVIDENCE', 'obligations': []}
        labels = label_template(value)
        labels.update(reviewer='Synthetic expert fixture', reviewer_role='Test only')
        labels['cases'][0].update(status='EXPERT_REVIEWED', rationale='Fixture annotation',
            expected_obligations=[{'subject': 'A firm', 'modality': 'MUST', 'action': 'retain records.'}])
        result = evaluate(value, labels)
        self.assertEqual(result['extraction_exact_span']['tp'], 0)
        self.assertEqual(result['extraction_exact_span']['fn'], 1)
        self.assertEqual(result['extraction_exact_span']['recall'], 0)
        self.assertEqual(result['status'], 'PARTIAL')

    def test_source_archive_detects_byte_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'sources'
            download = handbook_download(page(card('CONC 7.3.4', 'R', '04/11/2024', '<p>A firm must retain records.</p>')))
            bundle = save_sources(root, [download])
            self.assertEqual(len(load_sources(root)[1]), 1)
            (root/bundle['sources'][0]['raw_file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_sources(root)

    def test_source_path_cannot_escape_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'sources'
            download = handbook_download(page(card('CONC 7.3.4', 'R', '04/11/2024', '<p>A firm must retain records.</p>')))
            bundle = save_sources(root, [download])
            bundle['sources'][0]['raw_file'] = '../outside.html'
            (root/'snapshot.json').write_text(json.dumps(bundle), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'inside'):
                load_sources(root)

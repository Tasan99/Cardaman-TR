import json
import os
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes, verify_chain
from regchain.pilot.engine import analyze
from regchain.pilot.policies import select_chunks
from regchain.pilot.report import render
from regchain.pilot.semantic import (EmbeddingFailure, OllamaEmbedder, PolicyIndex, configured_embedder,
                                     heading_only, query_text, shared_ranks)
from test_pilot import FixtureProvider, company, sections

# Synthetic concept axes: words of one meaning share an axis, so a paraphrase with
# no shared word still points the same way. This is a fixture, not a language model.
CONCEPTS = [{'individual', 'circumstances', 'personal', 'situation', 'person'},
            {'retain', 'records', 'keep', 'files', 'archive'},
            {'shredding', 'paper', 'destroyed', 'disposal'},
            {'vary', 'schedule', 'permitted', 'fixed'}]


class ConceptEmbedder:
    base_url = 'http://localhost:11434'

    def __init__(self):
        self.calls = []

    def manifest(self):
        return {'provider': 'fixture', 'model_version': 'concepts@1', 'adapter': 'fixture'}

    def embed(self, texts):
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            words = set(text.lower().replace('.', ' ').replace(',', ' ').replace("'s", ' ').split())
            vector = [float(len(words & concept)) for concept in CONCEPTS] + [0.01]
            norm = sum(x*x for x in vector) ** .5
            rows.append([x/norm for x in vector])
        return rows


def chunk(name, text, filename='policy.txt'):
    return {'source_id': name, 'policy_hash': 'd'*64, 'filename': filename, 'locator': 'text_block',
            'number': 1, 'start': 0, 'end': len(text), 'text': text}


def policy(*chunks):
    return [{'filename': chunks[0]['filename'], 'raw_hash': 'd'*64, 'bytes': 1, 'parser': 'fixture', 'chunks': list(chunks)}]


DUTY = {'subject': 'a firm', 'modality': 'MUST', 'prohibited_action': None, 'exceptions': [],
        'required_action': 'take into account the individual circumstances of the customer',
        'conditions': ['When determining appropriate forbearance']}


class RankingTests(unittest.TestCase):
    def test_paraphrase_without_a_shared_word_is_found_only_by_meaning(self):
        value = policy(chunk('a-noise', 'Paper files are destroyed by shredding.'),
                       chunk('b-support', 'Agents look at the personal situation of each borrower.'))
        self.assertEqual(select_chunks(DUTY['required_action'], value), [])
        chosen, record = PolicyIndex(value, ConceptEmbedder()).select(DUTY)
        self.assertEqual(chosen[0]['source_id'], 'b-support')
        self.assertIsNone(record['shown'][0]['lexical_rank'])
        self.assertEqual(record['shown'][0]['semantic_rank'], 1)

    def test_one_contradicting_sentence_inside_a_long_passage_is_not_diluted(self):
        long = ('Paper files are destroyed by shredding. Disposal is logged. Paper is destroyed weekly. '
                'Agents are not permitted to look at the personal situation of a borrower.')
        value = policy(chunk('a-long', long), chunk('b-other', 'Paper files are destroyed by shredding and disposal.'))
        index = PolicyIndex(value, ConceptEmbedder())
        self.assertTrue(index.sentence_level)
        chosen, record = index.select(DUTY)
        self.assertEqual(chosen[0]['source_id'], 'a-long')
        whole = ConceptEmbedder().embed([long, query_text(DUTY)])
        passage_only = sum(a*b for a, b in zip(*whole))
        self.assertGreater(float(record['shown'][0]['similarity']), passage_only + .2)

    def test_exact_terms_still_count_and_ties_share_a_rank(self):
        self.assertEqual(shared_ranks([3, 1, 1, 0, 1]), [1, 2, 2, 5, 2])
        value = policy(chunk('z-last-by-id', 'The individual circumstances of the customer are recorded.'),
                       chunk('a-first-by-id', 'Agents look at the personal situation of each borrower.'))
        chosen, record = PolicyIndex(value, ConceptEmbedder()).select(DUTY)
        # Identical in meaning for this fixture; the literal wording breaks the tie, not the id order.
        self.assertEqual([r['semantic_rank'] for r in record['shown']], [1, 1])
        self.assertEqual([c['source_id'] for c in chosen], ['z-last-by-id', 'a-first-by-id'])
        self.assertEqual([r['lexical_rank'] for r in record['shown']], [1, None])

    def test_markdown_headings_never_take_a_slot_but_are_counted(self):
        value = policy(chunk('h', '## 5. Looking at the person, not only the account', 'p.md'),
                       chunk('b', 'Agents look at the personal situation of each borrower.', 'p.md'),
                       chunk('t', '# Not a heading in a text file', 'p.txt'))
        self.assertTrue(heading_only(value[0]['chunks'][0]))
        self.assertFalse(heading_only(value[0]['chunks'][2]))
        index = PolicyIndex(value, ConceptEmbedder())
        self.assertEqual(index.manifest()['headings_excluded'], 1)
        self.assertNotIn('h', [c['source_id'] for c in index.select(DUTY)[0]])

    def test_budget_is_respected_and_near_misses_are_disclosed(self):
        value = policy(*[chunk(f'c{i}', f'Agents look at the personal situation of person number {i}.') for i in range(12)])
        chosen, record = PolicyIndex(value, ConceptEmbedder()).select(DUTY, limit=6, max_chars=6500)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len(record['not_shown']), 4)
        self.assertFalse({r['source_id'] for r in record['shown']} & {r['source_id'] for r in record['not_shown']})
        tight, _ = PolicyIndex(value, ConceptEmbedder()).select(DUTY, limit=6, max_chars=130)
        self.assertLessEqual(sum(len(c['text']) for c in tight), 130)

    def test_large_top_passages_are_never_padded_from_the_bottom_of_the_ranking(self):
        # Observed on a 118-page PDF: rank 184, the single letter "R", was shown because it fitted.
        relevant = [chunk(f'big{i}', 'Agents look at the personal situation of each borrower. ' + 'Further detail follows. '*70) for i in range(3)]
        crumbs = [chunk('crumb-r', 'R'), chunk('crumb-cut', 'e \nclient'), chunk('short-unrelated', 'Paper is destroyed weekly.')]
        index = PolicyIndex(policy(*relevant, *[chunk(f'pad{i}', f'Paper files number {i} are destroyed by shredding.') for i in range(30)], *crumbs), ConceptEmbedder())
        self.assertEqual(index.manifest()['fragments_excluded'], 2)
        chosen, record = index.select(DUTY, limit=6, max_chars=4000)
        self.assertTrue(all(row['rank'] <= 10 for row in record['shown'] + record['not_shown']))
        self.assertNotIn('crumb-r', [c['source_id'] for c in chosen])
        self.assertEqual([c['source_id'] for c in chosen][:2], ['big0', 'big1'])
        self.assertIn('big2', [r['source_id'] for r in record['not_shown']])

    def test_passages_are_embedded_once_and_all_duties_are_embedded_in_one_batch(self):
        embedder = ConceptEmbedder()
        index = PolicyIndex(policy(chunk('a', 'Agents look at the personal situation of each borrower.')), embedder)
        other = dict(DUTY, required_action='retain records of each decision')
        # A query vector per duty would make the runtime swap the judge and the embedder
        # for every obligation once the judge no longer fits beside it in memory.
        index.prepare([DUTY, other, DUTY])
        index.select(DUTY)
        index.select(other)
        index.select(DUTY)
        self.assertEqual([len(call) for call in embedder.calls], [1, 2])

    def test_query_keeps_conditions_polarity_and_exceptions(self):
        duty = dict(DUTY, modality='MUST_NOT', required_action=None, prohibited_action='repossess a home',
                    exceptions=['other than as a last resort.'])
        self.assertEqual(query_text(duty),
            'When determining appropriate forbearance a firm must not repossess a home other than as a last resort.')


class EvidenceTests(unittest.TestCase):
    def policies(self):
        return policy(chunk('policy-1', 'All staff must retain records.'),
                      chunk('policy-2', 'Staff keep archive files for six years.'))

    def test_hybrid_analysis_records_the_method_without_floats_and_changes_identity(self):
        lexical = analyze(company(), self.policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        hybrid = analyze(company(), self.policies(), sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder())
        payload = hybrid['events'][0]['payload']
        canonical_bytes(payload)  # rejects any float in retained evidence
        self.assertTrue(verify_chain(hybrid['events'], hybrid['head'], 1))
        self.assertEqual(payload['policy_retrieval']['method'], 'hybrid-rrf-v1')
        self.assertEqual(payload['policy_retrieval']['embedding']['model_version'], 'concepts@1')
        self.assertEqual(lexical['events'][0]['payload']['policy_retrieval']['method'], 'lexical-v1')
        self.assertEqual(lexical['events'][0]['payload']['policy_retrieval']['reranker'], {'status': 'off'})
        self.assertEqual(payload['policy_retrieval']['reranker'], {'status': 'off'})
        self.assertNotEqual(payload['input_hash'], lexical['events'][0]['payload']['input_hash'])
        row = payload['obligations'][0]
        self.assertEqual(row['retrieved_policy_ids'], [r['source_id'] for r in row['retrieval']['shown']])
        self.assertIsInstance(row['retrieval']['shown'][0]['similarity'], str)
        self.assertNotIn('retrieval', lexical['events'][0]['payload']['obligations'][0])
        self.assertIn('similarity', ' '.join(payload['limitations']))

    def test_the_judge_reads_beyond_the_ranking_and_is_never_shown_it(self):
        seen = []

        class Recording(FixtureProvider):
            def passage(self, payload):
                seen.append(set(payload))
                return super().passage(payload)
        many = policy(*[chunk(f'p{i}', f'Staff retain records in archive {i}.') for i in range(9)])
        row = analyze(company(), many, sections(), Recording(), ['CONC 7.3.4'], embedder=ConceptEmbedder())['events'][0]['payload']['obligations'][0]
        # Similarity orders the reading; it is no longer a gate the judge cannot see past,
        # and a rank is not evidence, so the judge is given the duty and one passage only.
        self.assertEqual(seen, [{'duty', 'passage'}]*18)      # two narrow questions for each of nine passages
        self.assertEqual(len(row['retrieved_policy_ids']), 6)
        self.assertEqual(row['judged_policy_ids'][:6], row['retrieved_policy_ids'])
        self.assertEqual(sorted(row['judged_policy_ids']), [f'p{i}' for i in range(9)])

    def test_review_screen_discloses_what_the_judge_read_and_did_not_read_and_escapes_it(self):
        many = policy(*[chunk(f'p{i}', f'Staff retain records in archive {i}. <script>alert({i})</script>') for i in range(9)])
        page = render(analyze(company(), many, sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder()))
        self.assertIn('9 policy pasajını tek tek', page)
        self.assertIn('Yüklenen policy setinin tamamı okundu', page)
        self.assertIn('yargı: SUPPORTS', page)
        self.assertIn('Yargı modelinin gerekçesi', page)
        self.assertIn('anlam benzerliği + kelime örtüşmesi', page)
        self.assertNotIn('<script>alert(', page)
        self.assertIn('yalnızca kelime örtüşmesi', render(analyze(company(), many, sections(), FixtureProvider(), ['CONC 7.3.4'])))
        large = policy(*[chunk(f'p{i:02d}', f'Staff retain records in archive {i}.') for i in range(45)])
        page = render(analyze(company(), large, sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder()))
        self.assertIn('Policy setindeki 33 pasaj okunmadı', page)

    def test_headings_and_crumbs_are_neither_judged_nor_reported_as_unread(self):
        # Seen live: a two-file Markdown policy set has 16 headings; the review screen called
        # them "16 passages not read", which is what a reader would take for a coverage gap.
        seen = []

        class Recording(FixtureProvider):
            def passage(self, payload):
                seen.append(payload['passage'])
                return super().passage(payload)
        docs = policy(chunk('h1', '# Records policy', 'policy.md'), chunk('h2', '## 1. Purpose and scope', 'policy.md'),
                      chunk('p1', 'All staff must retain records.', 'policy.md'), chunk('crumb', 'R', 'policy.md'))
        for embedder in (ConceptEmbedder(), None):
            with self.subTest(retrieval='hybrid' if embedder else 'lexical'):
                del seen[:]
                packet = analyze(company(), docs, sections(), Recording(), ['CONC 7.3.4'], embedder=embedder)
                self.assertEqual(seen, ['All staff must retain records.']*2)          # two narrow questions
                self.assertEqual(packet['events'][0]['payload']['obligations'][0]['judged_policy_ids'], ['p1'])
                page = render(packet)
                self.assertIn('Yüklenen policy setinin tamamı okundu', page)
                self.assertNotIn('pasaj okunmadı', page)

    def test_a_packet_from_before_the_separate_judge_keeps_its_own_wording(self):
        many = policy(*[chunk(f'p{i}', f'Staff retain records in archive {i}.') for i in range(9)])
        packet = analyze(company(), many, sections(), FixtureProvider(), ['CONC 7.3.4'], embedder=ConceptEmbedder())
        packet['events'][0]['payload']['obligations'][0].pop('judged_policy_ids')
        page = render(packet)
        self.assertIn('AI’a gösterilmeyen pasajlar', page)      # there the model read only what it was shown
        self.assertNotIn('yargı: SUPPORTS', page)

    def test_embedder_failure_stops_the_analysis_before_any_model_call(self):
        class Down(ConceptEmbedder):
            def embed(self, texts):
                raise EmbeddingFailure('Ollama embedding request failed; check service, model and configuration')

        class Counting(FixtureProvider):
            calls = 0

            def _chat(self, prompt, payload, schema):
                Counting.calls += 1
                return super()._chat(prompt, payload, schema)
        with self.assertRaisesRegex(ValueError, 'embedding request failed'):
            analyze(company(), self.policies(), sections(), Counting(), ['CONC 7.3.4'], embedder=Down())
        self.assertEqual(Counting.calls, 0)


class OllamaEmbedderTests(unittest.TestCase):
    def client(self, factory, embeddings, models=None):
        client = factory.return_value.__enter__.return_value
        client.get.return_value.json.return_value = {'models': models if models is not None else [{'name': 'e:latest', 'digest': 'abc'}]}
        client.post.return_value.json.return_value = {'embeddings': embeddings}
        return client

    def test_private_policy_text_is_never_embedded_remotely_and_model_is_pinned(self):
        for model, digest, url in [('e', 'abc', 'https://embeddings.example'), ('e', 'abc', 'http://10.0.0.5:11434'),
                                   ('', 'abc', 'http://localhost:11434'), ('e', '', 'http://localhost:11434'),
                                   ('e', 'abc', 'http://user:pw@localhost:11434')]:
            with self.subTest(url=url, model=model, digest=digest), self.assertRaises(ValueError):
                OllamaEmbedder(model, digest, url)
        with self.assertRaises(ValueError):
            OllamaEmbedder('e', 'abc', 'http://localhost:11434', 5)

    def test_vectors_are_normalised_batched_and_never_silently_truncated(self):
        with patch('regchain.pilot.semantic.httpx.Client') as factory:
            client = self.client(factory, None)
            client.post.return_value.json.side_effect = [{'embeddings': [[3.0, 4.0]]*32}, {'embeddings': [[0.0, 2.0]]*3}]
            vectors = OllamaEmbedder('e:latest', 'abc', 'http://localhost:11434').embed(['text']*35)
        self.assertEqual(len(vectors), 35)
        self.assertEqual(vectors[0], [0.6, 0.8])
        self.assertEqual(vectors[-1], [0.0, 1.0])
        self.assertEqual(client.post.call_count, 2)
        self.assertIs(client.post.call_args.kwargs['json']['truncate'], False)

    def test_digest_mismatch_and_malformed_vectors_fail_closed(self):
        embedder = OllamaEmbedder('e:latest', 'abc', 'http://localhost:11434')
        with patch('regchain.pilot.semantic.httpx.Client') as factory:
            client = self.client(factory, [[1.0]], models=[{'name': 'e:latest', 'digest': 'other'}])
            with self.assertRaisesRegex(EmbeddingFailure, 'digest'):
                embedder.embed(['text'])
            client.post.assert_not_called()
        for texts, bad in [(1, [[1.0], [1.0]]),          # more vectors than inputs
                           (2, [[1.0, 2.0]]),            # fewer vectors than inputs
                           (1, [[0.0, 0.0]]),            # no direction
                           (1, [[float('nan'), 1.0]]),
                           (1, [[True, 1.0]]),
                           (1, ['not-a-vector']),
                           (2, [[1.0, 2.0], [1.0]])]:    # mixed sizes
            with self.subTest(bad=str(bad)), patch('regchain.pilot.semantic.httpx.Client') as factory:
                self.client(factory, bad)
                with self.assertRaises(EmbeddingFailure):
                    embedder.embed(['text'] * texts)

    def test_failure_message_never_carries_policy_text(self):
        import httpx
        with patch('regchain.pilot.semantic.httpx.Client') as factory:
            client = self.client(factory, [[1.0]])
            client.post.side_effect = httpx.ConnectError('PRIVATE-POLICY-SENTINEL')
            with self.assertRaises(EmbeddingFailure) as error:
                OllamaEmbedder('e:latest', 'abc', 'http://localhost:11434').embed(['PRIVATE-POLICY-SENTINEL'])
        self.assertNotIn('SENTINEL', str(error.exception))

    def test_configuration_is_explicit(self):
        with patch.dict(os.environ, {'EMBED_MODEL': '', 'EMBED_MODEL_DIGEST': ''}):
            with self.assertRaisesRegex(ValueError, 'EMBED_MODEL'):
                configured_embedder()
        with patch.dict(os.environ, {'EMBED_MODEL': 'e:latest', 'EMBED_MODEL_DIGEST': 'abc',
                                     'OLLAMA_BASE_URL': 'http://localhost:11434', 'EMBED_TIMEOUT_SECONDS': 'soon'}):
            with self.assertRaises(ValueError):
                configured_embedder()
        with patch.dict(os.environ, {'EMBED_MODEL': 'e:latest', 'EMBED_MODEL_DIGEST': 'abc',
                                     'OLLAMA_BASE_URL': 'http://localhost:11434', 'EMBED_TIMEOUT_SECONDS': ''}):
            self.assertEqual(configured_embedder().model_version, 'e:latest@abc')


if __name__ == '__main__':
    unittest.main()

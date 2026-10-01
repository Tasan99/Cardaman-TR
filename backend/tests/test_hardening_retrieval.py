"""Candidate expansion invariants; fixture vectors are not model-quality evidence."""
import unittest
from unittest.mock import patch

from regchain.model_router import ModelPolicyError, ModelScope, bind_model_scope
from regchain.pilot.semantic import PolicyIndex, candidate_union
from test_semantic import ConceptEmbedder, DUTY, chunk, policy


class CandidateUnionTests(unittest.TestCase):
    def test_independent_channels_recover_outside_head_without_losing_or_reordering_it(self):
        original = list(range(50))
        semantic = [30, 31, *range(30), *range(32, 50)]
        lexical = [40, 41, 0]
        actual = candidate_union(original, semantic, lexical, 12, 2, 2)
        self.assertEqual(actual[:12], original[:12])
        self.assertEqual(actual[12:], [30, 31, 40, 41])
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(original, list(range(50)))

    def test_union_never_introduces_foreign_ids_and_zero_limits_disable_expansion(self):
        self.assertEqual(candidate_union(['a', 'b'], ['foreign'], ['foreign'], 1, 1, 1), ['a'])
        self.assertEqual(candidate_union([3, 1, 2], [2, 1, 3], [2], 2, 0, 0), [3, 1])
        self.assertEqual(candidate_union([], [], [], 12), [])
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidate_union([1], [1], [1], value)

    def test_positive_lexical_channel_only_and_similarity_ties_are_source_id_stable(self):
        chunks = [chunk(str(i), 'Every customer is assessed with due consideration.') for i in range(5)]
        index = PolicyIndex(policy(*chunks), ConceptEmbedder())
        # Fused order is intentionally different from each channel and stays intact.
        frozen = ([4, 3, 2, 1, 0], [1., 1., .1, .2, .9], [1, 1, 5, 4, 3],
                  [2, 2, 2, 1, 2], [0, 0, 0, 2, 0], [0] * 5, {})
        with patch.object(index, 'order', return_value=frozen):
            selected = index.candidate_window(DUTY, 1, semantic_limit=1, lexical_limit=10)
        self.assertEqual([c['source_id'] for c in selected], ['4', '3', '0'])
        self.assertIs(selected[1], index.candidates[3])

    def test_scoped_index_cannot_be_reused_in_another_company(self):
        with bind_model_scope(ModelScope('t', 'a', 'run')):
            index = PolicyIndex(policy(chunk('a', 'Each customer receives consideration.')), ConceptEmbedder())
        with bind_model_scope(ModelScope('t', 'b', 'run')):
            with self.assertRaises(ModelPolicyError):
                index.candidate_window(DUTY, 12)

    def test_structural_crumbs_stay_out_and_order_scores_and_select_are_unchanged(self):
        index = PolicyIndex(policy(chunk('heading', '# Heading', 'p.md'),
                                   chunk('crumb', '23'),
                                   chunk('a', 'Agents look at the personal situation of each borrower.'),
                                   chunk('b', 'Paper files are destroyed by shredding.')), ConceptEmbedder())
        before = (index.ranked(DUTY), index.scores(DUTY), index.select(DUTY))
        selected = index.candidate_window(DUTY, 1)
        self.assertNotIn('heading', [c['source_id'] for c in selected])
        self.assertNotIn('crumb', [c['source_id'] for c in selected])
        self.assertEqual(before, (index.ranked(DUTY), index.scores(DUTY), index.select(DUTY)))
        self.assertEqual(before[0][:1], selected[:1])

    def test_reranker_order_is_preserved_when_adding_independent_candidates(self):
        class Reranker:
            def manifest(self):
                return {'model': 'fixture'}

            def score(self, query, texts):
                return list(range(len(texts)))

        chunks = [chunk(str(i), f'Each customer record number {i} is retained for review.') for i in range(15)]
        index = PolicyIndex(policy(*chunks), ConceptEmbedder(), Reranker())
        head = index.ranked(DUTY)[:3]
        self.assertEqual(index.candidate_window(DUTY, 3)[:3], head)


if __name__ == '__main__':
    unittest.main()

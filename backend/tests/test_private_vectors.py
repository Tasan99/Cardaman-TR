"""Private embedding and reranker boundaries, using mocked HTTP and model weights only."""
import os
import sys
import types
import unittest
from unittest.mock import patch

from regchain.model_router import ModelPolicyError, ModelScope, bind_model_scope
from regchain.pilot import rerank, semantic
from regchain.pilot.rerank import CrossEncoderReranker, cached_scores, configured_reranker
from regchain.pilot.semantic import OllamaEmbedder, PolicyIndex
from test_semantic import ConceptEmbedder, DUTY, chunk, policy


LOCAL = 'http://localhost:11434'
DIGEST = 'a' * 64
TEXT = 'PRIVATE-POLICY-SENTINEL: employees retain records for six years.'


class FakeEncoder:
    loaded = []
    predicted = []

    def __init__(self, model, max_length=512, device=None, local_files_only=True):
        self.device = device or 'cpu'
        self.loaded.append((model, max_length, device, local_files_only))

    def predict(self, pairs, batch_size=8, show_progress_bar=False):
        self.predicted.append(len(pairs))
        return [0.75] * len(pairs)


class PrivateVectorTests(unittest.TestCase):
    def setUp(self):
        semantic._EMBED_CACHE.clear()
        rerank._SCORE_CACHE.clear()
        rerank._INSTANCES.clear()
        rerank._ENCODERS.clear()
        FakeEncoder.loaded.clear()
        FakeEncoder.predicted.clear()
        self.addCleanup(semantic._EMBED_CACHE.clear)
        self.addCleanup(rerank._SCORE_CACHE.clear)
        self.addCleanup(rerank._INSTANCES.clear)
        self.addCleanup(rerank._ENCODERS.clear)
        env = patch.dict(os.environ, {'CARDAMAN_MODE': 'development', 'RERANK_MODEL': 'fixture/private-reranker',
                                     'RERANK_DEVICE': '', 'RERANK_ALLOW_DOWNLOAD': ''})
        env.start()
        self.addCleanup(env.stop)
        module = types.ModuleType('sentence_transformers')
        module.CrossEncoder = FakeEncoder
        weights = patch.dict(sys.modules, {'sentence_transformers': module})
        weights.start()
        self.addCleanup(weights.stop)

    def client(self, factory, extra=None):
        client = factory.return_value.__enter__.return_value
        entry = {'name': 'bge-m3:latest', 'digest': DIGEST, **(extra or {})}
        client.get.return_value.json.return_value = {'models': [entry]}
        client.post.return_value.json.return_value = {'embeddings': [[3.0, 4.0]]}
        return client

    def embedder(self):
        return OllamaEmbedder('bge-m3:latest', DIGEST, LOCAL, cache=True)

    def test_embedding_cache_is_partitioned_by_both_tenant_and_company(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}), \
                patch('regchain.pilot.semantic.httpx.Client') as factory:
            client = self.client(factory)
            for tenant, company in [('tenant-a', 'company-a'), ('tenant-a', 'company-b'), ('tenant-b', 'company-a')]:
                with self.subTest(tenant=tenant, company=company), \
                        bind_model_scope(ModelScope(tenant, company, 'run-1')):
                    embedder = self.embedder()
                    self.assertEqual(embedder.embed([TEXT]), [[0.6, 0.8]])
                    self.assertEqual(embedder.cache_hits, 0)
                    embedder.embed([TEXT])
                    self.assertEqual(embedder.cache_hits, 1)
            self.assertEqual(client.post.call_count, 3)
            # New runs may reuse vectors belonging to the same tenant and company.
            with bind_model_scope(ModelScope('tenant-a', 'company-a', 'run-2')):
                again = self.embedder()
                again.embed([TEXT])
                self.assertEqual(again.cache_hits, 1)
            self.assertEqual(client.post.call_count, 3)
            self.assertEqual(len(semantic._EMBED_CACHE), 3)
            self.assertNotIn(TEXT, repr(semantic._EMBED_CACHE))
            self.assertNotIn("'company-a'", repr(semantic._EMBED_CACHE))
            self.assertFalse(factory.call_args.kwargs['trust_env'])
            self.assertFalse(factory.call_args.kwargs['follow_redirects'])
            self.assertIs(client.post.call_args.kwargs['json']['truncate'], False)

    def test_embedding_cache_cannot_hide_mutated_endpoint(self):
        with patch('regchain.pilot.semantic.httpx.Client') as factory:
            self.client(factory)
            embedder = self.embedder()
            embedder.embed([TEXT])
            factory.reset_mock()
            embedder.base_url = 'https://external.example'
            with self.assertRaises(ModelPolicyError):
                embedder.embed([TEXT])
            factory.assert_not_called()

    def test_embedding_cache_cannot_hide_mutated_model_identity(self):
        with patch('regchain.pilot.semantic.httpx.Client') as factory:
            self.client(factory)
            embedder = self.embedder()
            embedder.embed([TEXT])
            factory.reset_mock()
            embedder.model_digest = 'b' * 64
            with self.assertRaises(ModelPolicyError):
                embedder.embed([TEXT])
            factory.assert_not_called()

    def test_embedding_rejects_local_origin_and_coordinated_model_identity_mutations(self):
        for mutation in ('origin', 'model', 'digest'):
            with self.subTest(mutation=mutation), patch('regchain.pilot.semantic.httpx.Client') as factory:
                self.client(factory)
                embedder = self.embedder()
                embedder.embed([TEXT])
                factory.reset_mock()
                if mutation == 'origin':
                    embedder.base_url = 'http://127.0.0.1:11434'
                elif mutation == 'model':
                    embedder.model = 'another-local-model:latest'
                else:
                    embedder.model_digest = 'b' * 64
                embedder.model_version = embedder.model + '@' + embedder.model_digest
                with self.assertRaises(ModelPolicyError):
                    embedder.embed([TEXT])
                factory.assert_not_called()

    def test_server_embedding_accepts_ollama_default_tag_and_equivalent_full_sha256(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}), \
                bind_model_scope(ModelScope('tenant', 'company', 'run')):
            for configured, installed in [('sha256:' + DIGEST.upper(), DIGEST),
                                           (DIGEST, 'sha256:' + DIGEST.upper())]:
                with self.subTest(configured=configured, installed=installed), \
                        patch('regchain.pilot.semantic.httpx.Client') as factory:
                    client = self.client(factory, {'digest': installed})
                    embedder = OllamaEmbedder('bge-m3', configured, LOCAL)
                    self.assertEqual(embedder.embed([TEXT]), [[0.6, 0.8]])
                    self.assertEqual(client.post.call_args.kwargs['json']['model'], 'bge-m3')
                    self.assertIs(client.post.call_args.kwargs['json']['truncate'], False)

    def test_embedding_checks_all_matching_aliases_for_cloud_metadata_before_digest(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}), \
                bind_model_scope(ModelScope('tenant', 'company', 'run')), \
                patch('regchain.pilot.semantic.httpx.Client') as factory:
            client = self.client(factory)
            client.get.return_value.json.return_value['models'].append(
                {'name': 'bge-m3:latest', 'digest': 'b' * 64, 'remote_host': 'https://ollama.com'})
            with self.assertRaises(ModelPolicyError):
                OllamaEmbedder('bge-m3', DIGEST, LOCAL).embed([TEXT])
            client.post.assert_not_called()

    def test_cloud_metadata_on_a_local_alias_is_rejected_before_embedding_post(self):
        for metadata in ({'remote_host': 'https://ollama.com'}, {'remote_model': 'remote:cloud'},
                         {'details': {'is_cloud': True}}, {'details': {'cloud': True}}):
            with self.subTest(metadata=metadata), patch('regchain.pilot.semantic.httpx.Client') as factory:
                client = self.client(factory, metadata)
                with self.assertRaises(ModelPolicyError):
                    self.embedder().embed([TEXT])
                client.post.assert_not_called()
                self.assertEqual(semantic._EMBED_CACHE, {})

    def test_cloud_metadata_is_rechecked_even_for_cached_vectors(self):
        with patch('regchain.pilot.semantic.httpx.Client') as factory:
            client = self.client(factory)
            embedder = self.embedder()
            embedder.embed([TEXT])
            client.post.reset_mock()
            self.client(factory, {'remote_host': 'https://ollama.com'})
            with self.assertRaises(ModelPolicyError):
                embedder.embed([TEXT])
            client.post.assert_not_called()

    def test_embedding_instance_cannot_cross_runs_or_companies(self):
        with bind_model_scope(ModelScope('tenant', 'company-a', 'run-1')), \
                patch('regchain.pilot.semantic.httpx.Client') as factory:
            self.client(factory)
            embedder = self.embedder()
            embedder.embed([TEXT])
        for scope in (ModelScope('tenant', 'company-a', 'run-2'), ModelScope('tenant', 'company-b', 'run-1')):
            with bind_model_scope(scope), patch('regchain.pilot.semantic.httpx.Client') as factory:
                with self.assertRaises(ModelPolicyError):
                    embedder.embed([TEXT])
                with self.assertRaises(ModelPolicyError):
                    embedder.manifest()
                factory.assert_not_called()

    def test_server_requires_scope_before_embedding_or_reranker_construction(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}), \
                patch('regchain.pilot.semantic.httpx.Client') as factory:
            for construct in (self.embedder, CrossEncoderReranker, configured_reranker):
                with self.subTest(construct=construct), self.assertRaises(ModelPolicyError):
                    construct()
            factory.assert_not_called()
            self.assertEqual(FakeEncoder.loaded, [])

    def test_switching_to_server_cannot_reuse_unscoped_development_objects(self):
        embedder = self.embedder()
        ranker = CrossEncoderReranker()
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}), \
                bind_model_scope(ModelScope('tenant', 'company', 'run')), \
                patch('regchain.pilot.semantic.httpx.Client') as factory:
            for act in (lambda: embedder.embed([TEXT]), lambda: ranker.score('query', [TEXT]),
                        lambda: cached_scores(ranker, 'query', [TEXT])):
                with self.assertRaises(ModelPolicyError):
                    act()
            factory.assert_not_called()
            self.assertEqual(FakeEncoder.predicted, [])

    def test_server_forbids_reranker_download_before_loading_weights(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server', 'RERANK_ALLOW_DOWNLOAD': 'on'}), \
                bind_model_scope(ModelScope('tenant', 'company', 'run')):
            with self.assertRaises(ModelPolicyError):
                CrossEncoderReranker(allow_download=True)
            with self.assertRaises(ModelPolicyError):
                configured_reranker()
        self.assertEqual(FakeEncoder.loaded, [])

    def test_reranker_weights_are_shared_but_scores_and_calls_are_private(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}):
            with bind_model_scope(ModelScope('tenant', 'company-a', 'run-1')):
                first, status = configured_reranker()
                self.assertEqual(status['status'], 'active')
                self.assertIs(configured_reranker()[0], first)
                self.assertEqual(cached_scores(first, 'query', [TEXT]), ([0.75], 0))
            with bind_model_scope(ModelScope('tenant', 'company-a', 'run-2')):
                second, _ = configured_reranker()
                self.assertIsNot(first, second)
                self.assertIs(first.encoder, second.encoder)
                self.assertEqual(second.calls, [])
                self.assertEqual(cached_scores(second, 'query', [TEXT]), ([0.75], 1))
                self.assertEqual(second.calls, [{'pairs': 0, 'cache_hits': 1, 'elapsed_ms': 0}])
                self.assertEqual(len(first.calls), 1)
                self.assertEqual(first.calls[0]['pairs'], 1)
                for act in (lambda: first.score('query', [TEXT]), lambda: cached_scores(first, 'query', [TEXT])):
                    with self.assertRaises(ModelPolicyError):
                        act()
                self.assertEqual(len(first.calls), 1)
            with bind_model_scope(ModelScope('tenant', 'company-b', 'run-1')):
                third, _ = configured_reranker()
                self.assertIs(first.encoder, third.encoder)
                self.assertEqual(cached_scores(third, 'query', [TEXT]), ([0.75], 0))
            with bind_model_scope(ModelScope('other-tenant', 'company-a', 'run-1')):
                fourth, _ = configured_reranker()
                self.assertEqual(cached_scores(fourth, 'query', [TEXT]), ([0.75], 0))
        self.assertEqual(len(FakeEncoder.loaded), 1)
        self.assertTrue(FakeEncoder.loaded[0][3])
        self.assertEqual(FakeEncoder.predicted, [1, 1, 1])
        self.assertEqual(len(rerank._SCORE_CACHE), 3)
        self.assertNotIn(TEXT, repr(rerank._SCORE_CACHE))
        self.assertNotIn("'company-a'", repr(rerank._SCORE_CACHE))

    def test_server_checks_scope_before_cached_reranker_hits(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}):
            with bind_model_scope(ModelScope('tenant', 'company', 'run')):
                ranker = CrossEncoderReranker()
                cached_scores(ranker, 'query', [TEXT])
            with self.assertRaises(ModelPolicyError):
                cached_scores(ranker, 'query', [TEXT])
            self.assertEqual(len(ranker.calls), 1)
        self.assertEqual(FakeEncoder.predicted, [1])

    def test_reranker_instance_settings_are_not_reused_after_device_or_download_change(self):
        first, _ = configured_reranker()
        with patch.dict(os.environ, {'RERANK_DEVICE': 'cuda'}):
            device, _ = configured_reranker()
        with patch.dict(os.environ, {'RERANK_ALLOW_DOWNLOAD': 'on'}):
            downloadable, _ = configured_reranker()
        self.assertIsNot(first, device)
        self.assertIsNot(first.encoder, device.encoder)
        self.assertIsNot(first.encoder, downloadable.encoder)
        self.assertEqual([entry[3] for entry in FakeEncoder.loaded], [True, True, False])

    def test_reranker_mutation_cannot_return_cached_scores_from_different_model_settings(self):
        ranker = CrossEncoderReranker()
        cached_scores(ranker, 'query', [TEXT])
        ranker.max_length = 256
        with self.assertRaises(ModelPolicyError):
            cached_scores(ranker, 'query', [TEXT])
        self.assertEqual(FakeEncoder.predicted, [1])

    def test_policy_index_memo_cannot_bypass_run_scope(self):
        with bind_model_scope(ModelScope('tenant', 'company', 'run-1')):
            index = PolicyIndex(policy(chunk('p', TEXT)), ConceptEmbedder())
            index.order(DUTY)
        with bind_model_scope(ModelScope('tenant', 'company', 'run-2')):
            for act in (lambda: index.order(DUTY), lambda: index.select(DUTY), lambda: index.ranked(DUTY),
                        lambda: index.scores(DUTY), lambda: index.prepare([DUTY]), index.manifest):
                with self.assertRaises(ModelPolicyError):
                    act()

    def test_policy_index_does_not_swallow_reranker_policy_failure(self):
        with bind_model_scope(ModelScope('tenant', 'company', 'run-1')):
            ranker = CrossEncoderReranker()
        with bind_model_scope(ModelScope('tenant', 'company', 'run-2')):
            index = PolicyIndex(policy(chunk('p', TEXT)), ConceptEmbedder(), ranker)
            with self.assertRaises(ModelPolicyError):
                index.order(DUTY)
            self.assertEqual(index.rerank_failures, [])


if __name__ == '__main__':
    unittest.main()

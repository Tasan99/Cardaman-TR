"""Private routing boundaries, without network access or an installed model."""

import asyncio
import os
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch

from regchain.model_router import (
    ModelPolicyError, ModelRouter, ModelScope, bind_model_scope, cache_namespace,
    current_model_scope, model_digests_match, model_mode, model_names_match, validate_inventory_entry, validate_model,
    validate_ollama_endpoint,
)


DIGEST = 'a' * 64
SERVER = {
    'CARDAMAN_MODE': 'server', 'OLLAMA_BASE_URL': 'http://localhost:11434',
    'LLM_MODEL': 'qwen3:4b', 'LLM_MODEL_DIGEST': DIGEST,
    'JUDGE_MODEL': 'qwen3:8b', 'JUDGE_MODEL_DIGEST': 'b' * 64,
    'EMBED_MODEL': 'bge-m3:latest', 'EMBED_MODEL_DIGEST': 'c' * 64,
}
SCOPE = ModelScope('tenant-a', 'company-a', 'run-a')


class ModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_mode_defaults_only_when_unset_and_rejects_unknown_values(self):
        self.assertEqual(model_mode({}), 'development')
        self.assertEqual(model_mode(SERVER), 'server')
        for value in ('', 'production', 'Server', ' server ', None):
            with self.subTest(value=value), self.assertRaises(ModelPolicyError):
                model_mode({'CARDAMAN_MODE': value})

    def test_local_origins_normalize_without_widening_the_allowlist(self):
        for origin, expected in (
            ('http://LOCALHOST:11434/', 'http://localhost:11434'),
            ('https://127.0.0.1', 'https://127.0.0.1'),
            ('http://[::1]:11434/', 'http://[::1]:11434'),
            ('http://host.docker.internal:011434', 'http://host.docker.internal:11434'),
        ):
            for environment in ({}, SERVER):
                with self.subTest(origin=origin, mode=model_mode(environment)):
                    self.assertEqual(validate_ollama_endpoint(origin, environ=environment), expected)

    def test_remote_and_ambiguous_origins_are_denied_in_both_modes(self):
        origins = (
            'http://public.example:11434', 'https://public.example', 'https://ollama.com',
            'http://192.168.1.1:11434', 'http://127.1:11434', 'http://2130706433',
            'http://localhost.public.example', 'http://localhost.', 'http://[::ffff:127.0.0.1]',
            'http://localhost@public.example', 'http://user:pass@localhost', 'http://@localhost',
            'http://localhost/api/chat', 'http://localhost//', 'http://localhost?',
            'http://localhost#', 'http://localhost?host=public.example', 'http://localhost/#',
            'http://localhost:', 'http://localhost:0', 'http://localhost:65536',
            'http://localhost:-1', 'http://localhost:abc', 'http://[::1]suffix',
            'http://[::1', 'http://localhost\\@public.example', 'http://local\thost',
            ' http://localhost', 'http://localhost\n', 'http://localhost\x00',
            'ftp://localhost', '//localhost:11434', '', None,
        )
        for origin in origins:
            for environment in ({}, SERVER):
                with self.subTest(origin=origin, mode=model_mode(environment)), self.assertRaises(ModelPolicyError):
                    validate_ollama_endpoint(origin, environ=environment)

    def test_development_keeps_pinned_local_alternate_models(self):
        for model in ('fixture:latest', 'm', 'local/model:q4', 'gpt-oss:20b'):
            validate_model(model, 'sha256:test-fixture')
        validate_model('embedding-fixture', 'test', kind='embedding')

    def test_cloud_model_names_and_remote_paths_are_always_denied(self):
        for model in ('qwen3:4b-cloud', 'cloud/model', 'qwen3:cloud', 'qwen3-CLOUD:latest',
                      'https://ollama.com/model', 'hf.co/owner/model', 'registry.example/model',
                      '../model', '/model', 'model\\other', 'model\n', '', None):
            with self.subTest(model=model), self.assertRaises(ModelPolicyError):
                validate_model(model, DIGEST)

    def test_model_and_digest_are_required_without_implicit_fallback(self):
        for digest in ('', ' ', 'abc\n', None):
            with self.subTest(digest=digest), self.assertRaises(ModelPolicyError):
                validate_model('fixture', digest)
        with self.assertRaises(ModelPolicyError):
            validate_model('fixture', 'digest', kind='unknown')

    def test_inventory_name_matching_only_expands_the_implicit_latest_tag(self):
        for actual, configured in (('bge-m3:latest', 'bge-m3'), ('bge-m3', 'bge-m3:latest'),
                                   ('bge-m3', 'bge-m3'), ('qwen3:4b', 'qwen3:4b'),
                                   ('local/model:latest', 'local/model')):
            with self.subTest(actual=actual, configured=configured):
                self.assertTrue(model_names_match(actual, configured))
        for actual, configured in (('bge-m3:v2', 'bge-m3'), ('bge-m3:latest', 'BGE-M3'),
                                   ('bge-m3:latest', 'other/bge-m3'), ('qwen3:8b', 'qwen3:4b'),
                                   ('bge-m3:latest ', 'bge-m3'), ('', ''), (None, 'bge-m3'),
                                   ('https://remote/model', 'https://remote/model')):
            with self.subTest(actual=actual, configured=configured):
                self.assertFalse(model_names_match(actual, configured))

    def test_inventory_digest_matching_normalizes_only_full_sha256_identities(self):
        for actual, configured in ((DIGEST.upper(), 'sha256:' + DIGEST),
                                   ('sha256:' + DIGEST, DIGEST.upper()),
                                   ('sha256:' + DIGEST.upper(), DIGEST),
                                   ('sha256:fixture', 'sha256:fixture'), ('abc', 'abc')):
            with self.subTest(actual=actual, configured=configured):
                self.assertTrue(model_digests_match(actual, configured))
        for actual, configured in (('sha256:abc', 'abc'), ('ABC', 'abc'), (DIGEST, 'b' * 64),
                                   ('SHA256:' + DIGEST, DIGEST), ('sha256:' + 'z' * 64, 'z' * 64),
                                   (' ' + DIGEST, DIGEST), ('', ''), (None, None), (123, 123)):
            with self.subTest(actual=actual, configured=configured):
                self.assertFalse(model_digests_match(actual, configured))

    def test_server_approved_models_require_complete_pins(self):
        for model, kind in (('qwen3:4b', 'chat'), ('qwen3:8b', 'chat'),
                            ('bge-m3', 'embedding'), ('bge-m3:latest', 'embedding')):
            for digest in (DIGEST, 'sha256:' + DIGEST):
                validate_model(model, digest, kind=kind, environ=SERVER)
        for model, kind, digest in (('fixture', 'chat', DIGEST), ('bge-m3', 'chat', DIGEST),
                                    ('qwen3:4b', 'embedding', DIGEST), ('qwen3:4b', 'chat', 'abc'),
                                    ('qwen3:4b', 'chat', 'sha256:abc'), ('qwen3:4b', 'chat', 'z' * 64)):
            with self.subTest(model=model, kind=kind, digest=digest), self.assertRaises(ModelPolicyError):
                validate_model(model, digest, kind=kind, environ=SERVER)

    def test_inventory_checks_alias_metadata_and_advertised_cloud_names(self):
        validate_inventory_entry({'name': 'local-alias', 'digest': DIGEST,
                                  'details': {'format': 'gguf', 'family': 'qwen3'}})
        for entry in (
            {'name': 'qwen3:4b', 'remote_host': 'https://ollama.com'},
            {'name': 'qwen3:4b', 'remote_model': 'remote-model'},
            {'name': 'qwen3:4b', 'cloud': True},
            {'name': 'qwen3:4b', 'details': {'is_cloud': True}},
            {'name': 'qwen3:4b', 'details': {'remote_host': 'https://ollama.com'}},
            {'name': 'qwen3:4b', 'model': 'qwen3:4b-cloud'},
            {'name': 'remote:cloud'}, None, [],
        ):
            with self.subTest(entry=entry), self.assertRaises(ModelPolicyError):
                validate_inventory_entry(entry)

    def test_scope_is_immutable_and_identifiers_are_validated(self):
        with self.assertRaises(FrozenInstanceError):
            SCOPE.company_id = 'other'
        for value in ('', ' ', ' leading', 'trailing ', 'x' * 257, 'company\nprivate', None):
            for field in ('tenant_id', 'company_id', 'run_id'):
                fields = dict(tenant_id='tenant', company_id='company', run_id='run')
                fields[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ModelPolicyError):
                    ModelScope(**fields)

    def test_scope_restores_nested_context_even_on_failure(self):
        self.assertIsNone(current_model_scope())
        with bind_model_scope(SCOPE):
            other = ModelScope('tenant-b', 'company-b', 'run-b')
            with self.assertRaises(RuntimeError):
                with bind_model_scope(other):
                    self.assertEqual(current_model_scope(), other)
                    raise RuntimeError('stop')
            self.assertEqual(current_model_scope(), SCOPE)
        self.assertIsNone(current_model_scope())
        with self.assertRaises(ModelPolicyError):
            with bind_model_scope({'tenant_id': 'tenant'}):
                pass

    def test_concurrent_contexts_do_not_share_scope_or_namespace(self):
        async def task(scope):
            with bind_model_scope(scope):
                before = cache_namespace()
                await asyncio.sleep(0)
                self.assertEqual(current_model_scope(), scope)
                self.assertEqual(cache_namespace(), before)
                return before

        async def run():
            return await asyncio.gather(task(SCOPE), task(ModelScope('tenant-b', 'company-a', 'run-a')))

        namespaces = asyncio.run(run())
        self.assertNotEqual(*namespaces)
        self.assertIsNone(current_model_scope())

    def test_cache_namespace_separates_tenants_and_companies_but_reuses_same_company_runs(self):
        namespaces = []
        for scope in (SCOPE, ModelScope('tenant-a', 'company-a', 'run-b'),
                      ModelScope('tenant-b', 'company-a', 'run-a'), ModelScope('tenant-a', 'company-b', 'run-a'),
                      ModelScope('tenant-a|company-a', 'company-b', 'run-a'),
                      ModelScope('tenant-a', 'company-a|company-b', 'run-a')):
            with bind_model_scope(scope):
                namespaces.append(cache_namespace())
                self.assertNotIn(scope.tenant_id, namespaces[-1])
                self.assertNotIn(scope.company_id, namespaces[-1])
        self.assertEqual(namespaces[0], namespaces[1])
        self.assertEqual(len(set(namespaces)), 5)

    def test_server_cannot_use_unscoped_cache(self):
        self.assertEqual(cache_namespace(), 'development-local')
        with self.assertRaises(ModelPolicyError):
            cache_namespace(SERVER)
        with bind_model_scope(SCOPE):
            self.assertEqual(cache_namespace(), cache_namespace(SERVER))

    def test_router_factories_are_lazy_and_reuse_the_original_objects(self):
        provider, embedding = object(), object()
        factory = Mock(return_value=provider)
        judge = Mock(side_effect=lambda default: default)
        embedder = Mock(return_value=embedding)
        router = ModelRouter('rules', provider_factory=factory, judge_factory=judge, embedder_factory=embedder)
        factory.assert_not_called()
        judge.assert_not_called()
        embedder.assert_not_called()
        self.assertIs(router.judge_provider(), provider)
        self.assertIs(router.extraction_provider(), provider)
        self.assertIs(router.judge_provider(), provider)
        self.assertIs(router.embedder(), embedding)
        self.assertIs(router.embedder(), embedding)
        factory.assert_called_once_with('rules')
        judge.assert_called_once_with(provider)
        embedder.assert_called_once_with()

    def test_default_router_factories_keep_existing_factory_signatures(self):
        provider, judge, embedding = object(), object(), object()
        with patch('regchain.extraction.providers.configured_provider', return_value=provider) as factory, \
                patch('regchain.extraction.providers.configured_judge', return_value=judge) as judge_factory, \
                patch('regchain.pilot.semantic.configured_embedder', return_value=embedding) as embedder_factory:
            router = ModelRouter()
            factory.assert_not_called()
            self.assertIs(router.judge_provider(), judge)
            self.assertIs(router.embedder(), embedding)
            factory.assert_called_once_with('ollama')
            judge_factory.assert_called_once_with(provider)
            embedder_factory.assert_called_once_with()

    def test_failed_factory_does_not_silently_fallback_or_cache_failure(self):
        provider = object()
        factory = Mock(side_effect=[RuntimeError('offline'), provider])
        router = ModelRouter(provider_factory=factory)
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            router.extraction_provider()
        self.assertIs(router.extraction_provider(), provider)
        self.assertEqual(factory.call_count, 2)

    def test_router_rejects_scope_changes_even_within_same_company(self):
        factory = Mock(return_value=object())
        with bind_model_scope(SCOPE):
            router = ModelRouter(provider_factory=factory)
            provider = router.extraction_provider()
            for other in (ModelScope('tenant-b', 'company-a', 'run-a'),
                          ModelScope('tenant-a', 'company-b', 'run-a'),
                          ModelScope('tenant-a', 'company-a', 'run-b')):
                with bind_model_scope(other):
                    for method in (router.extraction_provider, router.judge_provider, router.embedder):
                        with self.subTest(scope=other, method=method.__name__), self.assertRaises(ModelPolicyError):
                            method()
            self.assertIs(router.extraction_provider(), provider)
        with self.assertRaises(ModelPolicyError):
            router.extraction_provider()
        factory.assert_called_once()

    def test_router_cannot_move_from_unscoped_development_to_scoped_calls(self):
        router = ModelRouter(provider_factory=Mock())
        with bind_model_scope(SCOPE), self.assertRaises(ModelPolicyError):
            router.extraction_provider()

    def test_router_rejects_configuration_changes_before_or_after_lazy_construction(self):
        factory = Mock(return_value=object())
        router = ModelRouter(provider_factory=factory)
        for changes in ({'CARDAMAN_MODE': 'server'}, {'LLM_MODEL': 'changed'},
                        {'JUDGE_THINKING': 'on'}, {'EMBED_MODEL_DIGEST': 'new-pin'},
                        {'OLLAMA_BASE_URL': 'https://remote.example'}):
            with patch.dict(os.environ, changes), self.assertRaises(ModelPolicyError):
                router.extraction_provider()
        factory.assert_not_called()
        provider = router.extraction_provider()
        with patch.dict(os.environ, {'LLM_TIMEOUT_SECONDS': '180'}), self.assertRaises(ModelPolicyError):
            router.extraction_provider()
        self.assertIs(router.extraction_provider(), provider)
        factory.assert_called_once()

    def test_router_rejects_provider_selection_mutation_before_any_factory_or_cached_access(self):
        factory, judge, embedding = Mock(), Mock(), Mock()
        with patch.dict(os.environ, SERVER, clear=True), bind_model_scope(SCOPE):
            router = ModelRouter(provider_factory=factory, judge_factory=judge, embedder_factory=embedding)
            router.provider_name = 'rules'
            for method in (router.extraction_provider, router.judge_provider, router.embedder):
                with self.subTest(method=method.__name__), self.assertRaises(ModelPolicyError):
                    method()
            factory.assert_not_called()
            judge.assert_not_called()
            embedding.assert_not_called()
            router.provider_name = 'ollama'
            provider = router.extraction_provider()
            router.provider_name = 'rules'
            with self.assertRaises(ModelPolicyError):
                router.extraction_provider()
            router.provider_name = 'ollama'
            self.assertIs(router.extraction_provider(), provider)
        factory.assert_called_once_with('ollama')

    def test_server_requires_scope_and_all_three_pinned_roles_before_any_factory(self):
        factory = Mock()
        with patch.dict(os.environ, SERVER, clear=True):
            with self.assertRaises(ModelPolicyError):
                ModelRouter(provider_factory=factory)
            with bind_model_scope(SCOPE):
                for key in ('LLM_MODEL', 'LLM_MODEL_DIGEST', 'JUDGE_MODEL', 'JUDGE_MODEL_DIGEST',
                            'EMBED_MODEL', 'EMBED_MODEL_DIGEST'):
                    with self.subTest(missing=key), patch.dict(os.environ, {key: ''}), self.assertRaises(ModelPolicyError):
                        ModelRouter(provider_factory=factory)
                for changes in ({'LLM_MODEL': 'qwen3:8b'}, {'JUDGE_MODEL': 'qwen3:4b'},
                                {'EMBED_MODEL': 'another-local-model'}, {'LLM_MODEL_DIGEST': 'abc'},
                                {'OLLAMA_BASE_URL': 'https://remote.example'},
                                {'FAST_MODEL': 'qwen3:8b', 'FAST_MODEL_DIGEST': DIGEST},
                                {'FAST_MODEL': 'qwen3:4b', 'FAST_MODEL_DIGEST': ''}):
                    with self.subTest(changes=changes), patch.dict(os.environ, changes), self.assertRaises(ModelPolicyError):
                        ModelRouter(provider_factory=factory)
                with self.assertRaises(ModelPolicyError):
                    ModelRouter('rules', provider_factory=factory)
        factory.assert_not_called()

    def test_valid_server_preserves_factory_settings_and_shared_variants(self):
        provider, judge = Mock(), Mock()
        judge.quick = object()
        provider.fast = object()
        factory = Mock(return_value=provider)
        judge_factory = Mock(return_value=judge)
        settings = {**SERVER, 'LLM_THINKING': 'off', 'JUDGE_THINKING': 'on',
                    'JUDGE_NUM_CTX': '8192', 'FAST_MODEL': 'qwen3:4b', 'FAST_MODEL_DIGEST': DIGEST}
        with patch.dict(os.environ, settings, clear=True), bind_model_scope(SCOPE):
            router = ModelRouter(provider_factory=factory, judge_factory=judge_factory)
            self.assertIs(router.extraction_provider(), provider)
            self.assertIs(router.judge_provider(), judge)
            self.assertIs(router.judge_provider().quick, judge.quick)
            self.assertIs(router.extraction_provider().fast, provider.fast)
            self.assertEqual(dict(os.environ), settings)
        factory.assert_called_once_with('ollama')
        judge_factory.assert_called_once_with(provider)


if __name__ == '__main__':
    unittest.main()

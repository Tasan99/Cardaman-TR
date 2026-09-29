"""Private Ollama boundaries. Transport is mocked; no model or network is used."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from regchain.extraction.providers import (AI_RAW, OllamaProvider, ProviderFailure, raw_answers,
                                           request_hash, request_messages)
from regchain.model_router import ModelPolicyError, ModelScope, bind_model_scope


LOCAL = 'http://localhost:11434'
DIGEST = 'a' * 64
SCHEMA = {'type': 'object'}
SCOPE = ModelScope('tenant-a', 'company-a', 'run-a')
CLIENT = 'regchain.extraction.providers.httpx.Client'


def fake_client(factory, model='m', digest='d', **metadata):
    client = factory.return_value.__enter__.return_value
    client.get.return_value.json.return_value = {'models': [{'name': model, 'digest': digest, **metadata}]}
    answer = {'message': {'content': '{"private": "answer"}'}, 'done': True, 'done_reason': 'stop',
              'prompt_eval_count': 300, 'eval_count': 9}
    client.stream.return_value.__enter__.return_value.iter_bytes.return_value = [json.dumps(answer).encode()]
    client.request.return_value.json.return_value = {'models': []}
    return client


class PrivateProviderTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'CARDAMAN_MODE': 'development', 'AI_CACHE_DIR': ''}))
        self.factory = self.enterContext(patch(CLIENT))

    def call(self, provider, payload=None):
        return provider.generate_structured('instruction', payload or {'passage': 'private company text'}, SCHEMA)

    def test_remote_https_constructor_never_opens_a_client(self):
        with self.assertRaises(ModelPolicyError):
            OllamaProvider('m', 'd', 'https://models.example.test')
        self.factory.assert_not_called()

    def test_cloud_model_name_is_rejected_at_direct_constructor(self):
        with self.assertRaises(ModelPolicyError):
            OllamaProvider('qwen3:cloud', 'd', LOCAL)
        self.factory.assert_not_called()

    def test_local_alias_to_cloud_is_rejected_before_private_payload(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory, remote_host='https://ollama.com', remote_model='qwen3:cloud')
        with self.assertRaises(ModelPolicyError):
            self.call(provider)
        client.get.assert_called_once_with(LOCAL + '/api/tags')
        client.stream.assert_not_called()
        self.assertEqual(provider.cache, {})
        self.assertEqual(provider.call_log, [])

    def test_duplicate_matching_alias_with_remote_metadata_is_rejected(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory)
        client.get.return_value.json.return_value['models'].append(
            {'name': 'm', 'digest': 'other', 'details': {'remote_host': 'https://ollama.com'}})
        with self.assertRaises(ModelPolicyError):
            self.call(provider)
        client.stream.assert_not_called()

    def test_remote_inventory_entry_for_an_unrelated_model_does_not_block_local_model(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory)
        client.get.return_value.json.return_value['models'].append(
            {'name': 'other', 'digest': 'other', 'remote_host': 'https://ollama.com'})
        self.assertEqual(self.call(provider), '{"private": "answer"}')
        self.assertEqual(client.stream.call_count, 1)

    def test_bare_model_name_matches_the_installed_latest_tag(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory, 'm:latest')
        self.call(provider)
        self.assertEqual(client.stream.call_args.kwargs['json']['model'], 'm')

    def test_matching_latest_alias_metadata_is_checked_before_digest(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory)
        client.get.return_value.json.return_value['models'].append(
            {'name': 'm:latest', 'digest': 'different', 'remote_host': 'https://ollama.com'})
        with self.assertRaises(ModelPolicyError):
            self.call(provider)
        client.stream.assert_not_called()

    def test_development_complete_sha256_digest_supports_prefix_and_case(self):
        provider = OllamaProvider('m', 'sha256:' + DIGEST.upper(), LOCAL)
        client = fake_client(self.factory, 'm', DIGEST)
        self.call(provider)
        self.assertEqual(client.stream.call_count, 1)

    def test_development_fixture_digests_still_require_exact_equality(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory, 'm', 'D')
        with self.assertRaisesRegex(ProviderFailure, 'digest does not match'):
            self.call(provider)
        client.stream.assert_not_called()

    def test_mutated_endpoint_is_rejected_before_cached_answer_and_repair(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        fake_client(self.factory)
        self.call(provider)
        provider.last_raw = Mock()
        provider.base_url = 'https://models.example.test'
        self.factory.reset_mock()
        with self.assertRaises(ModelPolicyError):
            self.call(provider)
        with self.assertRaises(ModelPolicyError):
            provider.repair('A firm must act.', None, 'error')
        provider.last_raw.assert_not_called()
        self.factory.assert_not_called()
        self.assertEqual(len(provider.call_log), 1)

    def test_changed_valid_model_identity_cannot_reuse_digest_memo(self):
        for field, value in (('model', 'different'), ('model_digest', 'new-digest'), ('base_url', 'http://127.0.0.1:11434')):
            with self.subTest(field=field):
                provider = OllamaProvider('m', 'd', LOCAL)
                provider.digest_verified_at = time.monotonic()
                setattr(provider, field, value)
                with self.assertRaises(ModelPolicyError):
                    self.call(provider)
        self.factory.assert_not_called()

    def test_direct_request_and_processor_probe_revalidate_before_http(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        provider.base_url = 'https://models.example.test'
        with self.assertRaises(ModelPolicyError):
            provider._request([], SCHEMA, 1, time.monotonic())
        client = Mock()
        with self.assertRaises(ModelPolicyError):
            provider.placement_of(client)
        self.factory.assert_not_called()
        client.request.assert_not_called()

    def test_endpoint_changed_during_inventory_read_is_rejected_before_post(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory)

        def inventory():
            provider.base_url = 'https://models.example.test'
            return {'models': [{'name': 'm', 'digest': 'd'}]}

        client.get.return_value.json.side_effect = inventory
        with self.assertRaises(ModelPolicyError):
            self.call(provider)
        client.stream.assert_not_called()

    def test_provider_scope_checks_precede_cache_repair_and_call_log_access(self):
        with bind_model_scope(SCOPE):
            provider = OllamaProvider('m', 'd', LOCAL)
            fake_client(self.factory)
            self.call(provider)
        self.factory.reset_mock()
        for other in (ModelScope('tenant-b', 'company-a', 'run-a'),
                      ModelScope('tenant-a', 'company-b', 'run-a'),
                      ModelScope('tenant-a', 'company-a', 'run-b')):
            with self.subTest(scope=other), bind_model_scope(other):
                for operation in (lambda: self.call(provider),
                                  lambda: provider.repair('A firm must act.', None, 'error'),
                                  lambda: provider.cached_answer('key'),
                                  lambda: provider.store_answer('key', 'other'),
                                  lambda: provider.record({'status': 'OTHER'})):
                    with self.assertRaises(ModelPolicyError):
                        operation()
        self.assertEqual(len(provider.call_log), 1)
        self.factory.assert_not_called()

    def test_unscoped_development_instance_cannot_enter_company_scope(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        with bind_model_scope(SCOPE), self.assertRaises(ModelPolicyError):
            self.call(provider)
        self.factory.assert_not_called()

    def test_shared_memory_cache_separates_company_and_run(self):
        with bind_model_scope(SCOPE):
            first = OllamaProvider('m', 'd', LOCAL)
            first.store_answer('same-request', 'company-a answer')
        for other in (ModelScope('tenant-a', 'company-b', 'run-a'), ModelScope('tenant-a', 'company-a', 'run-b')):
            with self.subTest(scope=other), bind_model_scope(other):
                second = OllamaProvider('m', 'd', LOCAL)
                second.cache = first.cache
                self.assertIsNone(second.cached_answer('same-request'))

    def test_development_disk_cache_separates_tenants_and_companies(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'AI_CACHE_DIR': directory}):
            with bind_model_scope(SCOPE):
                first = OllamaProvider('m', 'd', LOCAL)
                first.store_answer('same-request', 'company-a answer')
            for other in (ModelScope('tenant-a', 'company-b', 'run-a'), ModelScope('tenant-b', 'company-a', 'run-a')):
                with self.subTest(scope=other), bind_model_scope(other):
                    self.assertIsNone(OllamaProvider('m', 'd', LOCAL).cached_answer('same-request'))
            with bind_model_scope(ModelScope('tenant-a', 'company-a', 'new-run')):
                self.assertEqual(OllamaProvider('m', 'd', LOCAL).cached_answer('same-request'), 'company-a answer')
            self.assertEqual(len(list(Path(directory).glob('*.json'))), 1)

    def test_scope_namespace_does_not_change_request_body_or_replay_hash(self):
        with bind_model_scope(SCOPE):
            provider = OllamaProvider('m', 'd', LOCAL, thinking=False)
            client = fake_client(self.factory)
            self.call(provider)
        sent = client.stream.call_args.kwargs['json']
        self.assertEqual(set(sent), {'model', 'stream', 'format', 'messages', 'options', 'think'})
        self.assertEqual(sent['options'], {'temperature': 0, 'num_predict': provider.num_predict, 'num_ctx': provider.num_ctx})
        self.assertEqual(sent['messages'], request_messages('instruction', {'passage': 'private company text'}, SCHEMA))
        self.assertEqual(provider.call_log[-1]['request_hash'], request_hash(sent))
        self.assertNotIn('tenant-a', json.dumps(sent))
        self.factory.assert_called_once_with(timeout=120, follow_redirects=False, trust_env=False)

    def test_redirect_response_never_receives_a_second_request(self):
        provider = OllamaProvider('m', 'd', LOCAL)
        client = fake_client(self.factory)
        response = httpx.Response(307, headers={'location': 'https://models.example.test'},
                                  request=httpx.Request('GET', LOCAL + '/api/tags'))
        client.get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError('redirect', request=response.request, response=response)
        with self.assertRaises(ProviderFailure):
            self.call(provider)
        self.assertEqual(client.get.call_count, 1)
        client.stream.assert_not_called()


class ServerProviderTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'CARDAMAN_MODE': 'server', 'AI_CACHE_DIR': ''}))
        self.factory = self.enterContext(patch(CLIENT))

    def provider(self):
        return OllamaProvider('qwen3:4b', DIGEST, LOCAL)

    def call(self, provider, n=1):
        return provider.generate_structured('instruction', {'passage': 'private company text', 'n': n}, SCHEMA)

    def test_server_unscoped_private_call_is_rejected_before_http(self):
        provider = self.provider()
        with self.assertRaises(ModelPolicyError):
            self.call(provider)
        self.factory.assert_not_called()

    def test_server_disk_cache_is_rejected_at_creation_and_after_configuration_change(self):
        with tempfile.TemporaryDirectory() as directory, bind_model_scope(SCOPE):
            with patch.dict(os.environ, {'AI_CACHE_DIR': directory}), self.assertRaisesRegex(ModelPolicyError, 'AI_CACHE_DIR'):
                self.provider()
            provider = self.provider()
            with patch.dict(os.environ, {'AI_CACHE_DIR': directory}), self.assertRaisesRegex(ModelPolicyError, 'AI_CACHE_DIR'):
                self.call(provider)
            provider.cache_dir = directory
            with self.assertRaisesRegex(ModelPolicyError, 'AI_CACHE_DIR'):
                provider.cached_answer('key')
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.factory.assert_not_called()

    def test_server_raw_answer_context_fails_closed(self):
        with self.assertRaisesRegex(ModelPolicyError, 'raw_answers'):
            with raw_answers():
                self.fail('server must not open a plaintext raw-answer sink')
        self.assertIsNone(AI_RAW.get())

    def test_server_rejects_injected_raw_sink_before_cache_or_http(self):
        with bind_model_scope(SCOPE):
            provider = self.provider()
            token = AI_RAW.set([])
            try:
                with self.assertRaisesRegex(ModelPolicyError, 'Raw model'):
                    self.call(provider)
                self.assertEqual(AI_RAW.get(), [])
            finally:
                AI_RAW.reset(token)
        self.factory.assert_not_called()

    def test_server_refreshes_inventory_before_each_new_inference(self):
        with bind_model_scope(SCOPE):
            provider = self.provider()
            client = fake_client(self.factory, 'qwen3:4b', DIGEST)
            self.call(provider, 1)
            self.call(provider, 2)
            self.assertEqual((client.get.call_count, client.stream.call_count), (2, 2))
            client.get.return_value.json.return_value['models'][0]['remote_host'] = 'https://ollama.com'
            with self.assertRaises(ModelPolicyError):
                self.call(provider, 3)
            self.assertEqual((client.get.call_count, client.stream.call_count), (3, 2))

    def test_server_sha256_prefix_and_case_identify_the_same_pinned_digest(self):
        with bind_model_scope(SCOPE):
            provider = OllamaProvider('qwen3:4b', 'sha256:' + DIGEST.upper(), LOCAL)
            client = fake_client(self.factory, 'qwen3:4b', DIGEST)
            self.call(provider)
            self.assertEqual(client.stream.call_count, 1)

    def test_server_stream_remote_metadata_is_rejected_without_caching_or_raw_logging(self):
        with bind_model_scope(SCOPE):
            provider = self.provider()
            client = fake_client(self.factory, 'qwen3:4b', DIGEST)
            client.stream.return_value.__enter__.return_value.iter_bytes.return_value = [
                json.dumps({'message': {'content': 'answer'}, 'done': True, 'remote_host': 'https://ollama.com'}).encode()]
            with self.assertRaises(ModelPolicyError):
                self.call(provider)
            self.assertEqual((provider.cache, provider.call_log, provider.last_answer), ({}, [], None))

    def test_mode_change_cannot_turn_existing_provider_into_development_mode(self):
        with bind_model_scope(SCOPE):
            provider = self.provider()
            with patch.dict(os.environ, {'CARDAMAN_MODE': 'development'}), self.assertRaises(ModelPolicyError):
                self.call(provider)
        self.factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()

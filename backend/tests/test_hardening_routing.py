"""The 4B diagnostic provider is distinct from the configured critical route."""
import os
import unittest
from unittest.mock import patch

from regchain.extraction.providers import OllamaProvider, RulesProvider, configured_judge, fast_provider
from regchain.model_router import ModelPolicyError


class CriticalRoutingTests(unittest.TestCase):
    def test_unset_judge_uses_pinned_8b_instead_of_reusing_4b(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'development', 'JUDGE_MODEL_DIGEST': 'b' * 64}, clear=True):
            extractor = OllamaProvider('qwen3:4b', 'a' * 64, 'http://localhost:11434', thinking=False)
            judge = configured_judge(extractor)
            self.assertEqual(judge.model, 'qwen3:8b')
            self.assertEqual(judge.model_digest, 'b' * 64)
            self.assertTrue(judge.thinking)
            self.assertIsNot(judge, extractor)
            self.assertEqual(extractor.model, 'qwen3:4b')
            self.assertIs(fast_provider(extractor), extractor)

    def test_missing_8b_pin_fails_closed_without_network_or_small_model_fallback(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'development'}, clear=True), patch('regchain.extraction.providers.httpx.Client') as network:
            extractor = OllamaProvider('qwen3:4b', 'a' * 64, 'http://localhost:11434')
            with self.assertRaisesRegex(ModelPolicyError, 'JUDGE_MODEL_DIGEST'):
                configured_judge(extractor)
            network.assert_not_called()

    def test_explicit_small_critical_model_is_rejected_even_without_thinking(self):
        for thinking in ('on', 'off', ''):
            with self.subTest(thinking=thinking), patch.dict(os.environ, {
                'CARDAMAN_MODE': 'development', 'JUDGE_MODEL': 'qwen3:4b',
                'JUDGE_MODEL_DIGEST': 'a' * 64, 'JUDGE_THINKING': thinking}, clear=True):
                extractor = OllamaProvider('qwen3:4b', 'a' * 64, 'http://localhost:11434')
                with self.assertRaisesRegex(ModelPolicyError, 'critical judgement'):
                    configured_judge(extractor)

    def test_rules_and_unrelated_development_providers_keep_existing_behavior(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'development', 'JUDGE_MODEL': ''}, clear=True):
            rules = RulesProvider()
            self.assertIs(configured_judge(rules), rules)
            fake = OllamaProvider('fixture', 'fixture-digest', 'http://localhost:11434')
            self.assertIs(configured_judge(fake), fake)

    def test_direct_diagnostic_provider_still_allows_measuring_4b_budget_failures(self):
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'development'}, clear=True):
            diagnostic = OllamaProvider('qwen3:4b', 'a' * 64, 'http://localhost:11434', thinking=True)
            self.assertEqual((diagnostic.model, diagnostic.thinking), ('qwen3:4b', True))


if __name__ == '__main__':
    unittest.main()

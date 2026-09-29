"""Model policy survives engine fallbacks and local entry-point threading.

All source/model work uses fixtures. No live model, network, or database is used.
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

from regchain.extraction.pipeline import extract, read_review
from regchain.extraction.providers import RulesProvider
from regchain.model_router import ModelPolicyError, ModelScope, bind_model_scope, current_model_scope
from regchain.pilot.engine import draft_clause, enrich, structured
from regchain.pilot.__main__ import main as pilot_main
from regchain.pilot.workspace import RunInput, Workspace
from test_pilot import FixtureProvider
from test_semantic import ConceptEmbedder
from test_workspace import request_input, source_fixture


class ModelEntryPointTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'CARDAMAN_MODE': 'development'}, clear=True))

    def test_draft_rejects_remote_provider_and_remote_quick_variant_before_call(self):
        for remote_quick in (False, True):
            for fitted in (False, True):
                with self.subTest(remote_quick=remote_quick, fitted=fitted):
                    provider = FixtureProvider()
                    target = FixtureProvider() if remote_quick else provider
                    target.base_url = 'https://remote-model.example.test'
                    target._chat = Mock(side_effect=AssertionError('Private text reached a remote provider'))
                    if remote_quick:
                        provider.quick = target
                    with self.assertRaises(ModelPolicyError):
                        draft_clause(provider, {}, 'PRIVATE POLICY SENTINEL', fitted=fitted)
                    target._chat.assert_not_called()

    def test_structured_provider_without_explicit_local_origin_is_rejected(self):
        class UnscopedProvider:
            generate_structured = Mock()
        with self.assertRaises(ModelPolicyError):
            structured(UnscopedProvider(), 'instruction', {'private': 'text'}, {})
        UnscopedProvider.generate_structured.assert_not_called()

    def test_extraction_does_not_retry_or_suppress_model_policy_failure(self):
        provider = RulesProvider()
        provider.generate = Mock(side_effect=ModelPolicyError('Private model boundary violation'))
        with self.assertRaises(ModelPolicyError):
            extract('A firm must retain records.', provider)
        provider.generate.assert_called_once()

    def test_review_rescue_does_not_hide_model_policy_failure(self):
        provider = RulesProvider()
        provider.review = Mock(side_effect=ModelPolicyError('Private model boundary violation'))
        for rescue in (False, True):
            with self.subTest(rescue=rescue), self.assertRaises(ModelPolicyError):
                read_review(provider, 'A firm must retain records.', None, rescue=rescue)

    def test_optional_enrichment_does_not_hide_model_policy_failure(self):
        provider = FixtureProvider()
        provider._chat = Mock(side_effect=ModelPolicyError('Private model boundary violation'))
        with self.assertRaises(ModelPolicyError):
            enrich(provider, {}, ['records'])
        provider._chat.assert_called_once()

    def test_workspace_refuses_server_before_creating_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'workspace'
            with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}), self.assertRaises(ModelPolicyError):
                Workspace(root)
            self.assertFalse(root.exists())

    def test_existing_workspace_refuses_runtime_switch_to_server(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory))
            try:
                with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}):
                    with self.assertRaises(ModelPolicyError):
                        workspace.models()
                    with self.assertRaises(ModelPolicyError):
                        workspace.submit(RunInput.model_validate(request_input()))
                self.assertEqual(workspace.list_runs(), [])
            finally:
                workspace.close()

    def test_local_cli_refuses_server_before_reading_documents(self):
        args = ['regchain.pilot', 'run', '--company', 'missing-company.json', '--policy', 'private.txt',
                '--sources', 'sources', '--output', 'result', '--provider', 'ollama']
        error = io.StringIO()
        with patch.dict(os.environ, {'CARDAMAN_MODE': 'server'}), patch('sys.argv', args), redirect_stderr(error):
            with self.assertRaises(SystemExit) as raised:
                pilot_main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn('requires development mode', error.getvalue())
        self.assertNotIn('missing-company', error.getvalue())

    def test_scope_follows_worker_question_and_draft_and_is_always_restored(self):
        seen = []

        def observe(stage):
            seen.append((stage, current_model_scope()))

        class ScopedFixture(FixtureProvider):
            def generate(self, text):
                observe('generate')
                return super().generate(text)

            def _chat(self, *args):
                observe('chat')
                return super()._chat(*args)

        class ScopedEmbeddings(ConceptEmbedder):
            def embed(self, texts):
                observe('embed')
                return super().embed(texts)

        def provider_factory(_):
            observe('provider_factory')
            return ScopedFixture()

        def judge_factory(provider):
            observe('judge_factory')
            return provider

        def embedder_factory():
            observe('embedder_factory')
            return ScopedEmbeddings()

        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory), provider_factory=provider_factory,
                                  source_fetcher=source_fixture, judge_factory=judge_factory,
                                  embedder_factory=embedder_factory,
                                  reranker_factory=lambda: (None, {'status': 'off'}))
            outer = ModelScope('caller-tenant', 'caller-company', 'caller-run')
            try:
                workspace.start()
                with bind_model_scope(outer):
                    job = workspace.submit(RunInput.model_validate(request_input(retrieval='hybrid')))
                    self.assertEqual(current_model_scope(), outer)
                    # One worker: this sentinel runs only after the actual analysis has ended.
                    self.assertIsNone(workspace.executor.submit(current_model_scope).result(timeout=30))
                    self.assertEqual(workspace.metadata(job['id'])['state'], 'COMPLETED', workspace.metadata(job['id']))
                    data = workspace.packet(job['id'])['events'][0]['payload']
                    with patch.dict(os.environ, {'EMBED_MODEL': 'fixture'}):
                        answer = workspace.ask(job['id'], 'What records must staff retain?')
                    self.assertTrue(answer['answer'])
                    self.assertEqual(current_model_scope(), outer)
                    draft = workspace.draft(job['id'], data['obligations'][0]['id'])
                    self.assertTrue(draft['clause'])
                    self.assertEqual(current_model_scope(), outer)
                    with patch.object(workspace, 'judge_factory', side_effect=ModelPolicyError('Rejected')):
                        with self.assertRaises(ModelPolicyError):
                            workspace.draft(job['id'], data['obligations'][0]['id'])
                    self.assertEqual(current_model_scope(), outer)
                expected = ModelScope(workspace.tenant_id, data['company']['id'], job['id'])
                self.assertTrue(seen)
                self.assertTrue(all(scope == expected for _, scope in seen), seen)
                self.assertTrue({'provider_factory', 'judge_factory', 'embedder_factory', 'generate', 'chat', 'embed'}
                                <= {stage for stage, _ in seen})
                self.assertIsNone(current_model_scope())
            finally:
                workspace.close()


if __name__ == '__main__':
    unittest.main()

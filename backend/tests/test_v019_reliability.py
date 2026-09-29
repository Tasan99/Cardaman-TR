"""v0.19 reliability (Phases 15 and 16): every provider failure carries a code from FAILURE_CODES
(and keeps its kind for the circuit breaker), failure_code() names any exception, call records
keep the window, the failure code and what the runtime reported about a failed call, an empty
answer is its own failure, JUDGE_NUM_PREDICT is checked against the real window, the judge's
window modes (fixed by default: a fixed-mode judge is the v0.18 judge; adaptive falls back to the
wide twin, see test_v019_context for fit_call), and the reranker's RRF fallback when its model is
not in the local cache. Transport-level fakes only: no Ollama, no model download, no network."""
import importlib.util
import json
import os
import socket
import sys
import time
import types
import unittest
import uuid
from unittest.mock import patch

import httpx
from pydantic import BaseModel, ValidationError
from regchain.extraction.contract import CONTRACT_VERSION
from regchain.extraction.providers import (CIRCUIT_FAILURES, FAILURE_CODES, JUDGE_THINKING_PREDICT, NUM_CTX, NUM_PREDICT,
                                           ContextBudgetError, OllamaProvider, ProviderFailure, ai_task, configured_judge, failure,
                                           failure_code, runtime_manifest, window_for)
from regchain.pilot import rerank as rerank_module
from regchain.pilot.rerank import configured_reranker
from regchain.pilot.semantic import VERSION as FUSED_VERSION, PolicyIndex
from test_semantic import DUTY, ConceptEmbedder
from test_v018_reliability import LOCAL, SCHEMA, fake_client, passages


def final(content='{"a": 1}', **fields):
    """The last streamed chunk of an answer, as Ollama sends it."""
    return json.dumps({'message': {'content': content}, 'done': True, 'done_reason': 'stop', 'prompt_eval_count': 300,
                       'eval_count': 9, **fields}).encode()


def judge_env(**changes):
    """A thinking judge on the fake's model; nothing inherited from the developer's environment."""
    return {'JUDGE_MODEL': 'm', 'JUDGE_MODEL_DIGEST': 'd', 'JUDGE_THINKING': 'on', 'JUDGE_TIMEOUT_SECONDS': '', 'JUDGE_NUM_PREDICT': '',
            'JUDGE_NUM_CTX': '', 'JUDGE_CTX_MODE': '', 'JUDGE_NUM_CTX_SMALL': '', 'JUDGE_NUM_CTX_LARGE': '', 'AI_CACHE_DIR': '',
            **changes}


def judge(**changes):
    with patch.dict(os.environ, judge_env(**changes)):
        return configured_judge(OllamaProvider('small', 'd1', LOCAL, 300))


class FailureCodeTests(unittest.TestCase):
    def fails(self, provider, chunks=None, payload=None, **client_changes):
        """(exception, call record) of one call that must fail against the fake transport."""
        with patch('regchain.extraction.providers.httpx.Client') as factory, patch('regchain.extraction.providers.time.sleep'):
            client = fake_client(factory)
            # Set directly, not through fake_client's list(): a generator must run during the call.
            client.stream.return_value.__enter__.return_value.iter_bytes.return_value = chunks if chunks is not None else [final()]
            for name, value in client_changes.items():
                setattr(client.stream, name, value)
            with self.assertRaises(ProviderFailure) as caught:
                provider.generate_structured('p', payload or {'passage': uuid.uuid4().hex}, SCHEMA)
        exc = caught.exception
        self.assertIn(exc.code, FAILURE_CODES)
        self.assertEqual(provider.call_log[-1]['failure_code'], exc.code)
        self.assertEqual(provider.call_log[-1]['num_ctx'], provider.num_ctx)
        self.assertEqual(failure_code(exc), exc.code)
        return exc, provider.call_log[-1]

    def test_the_vocabulary_is_fixed_and_every_failure_is_named(self):
        self.assertEqual(FAILURE_CODES, ('TIMEOUT', 'CONTEXT_OVERFLOW', 'MALFORMED_JSON', 'EMPTY_RESPONSE', 'RETRY_EXHAUSTED',
                                         'OLLAMA_ERROR', 'OUTPUT_TRUNCATED', 'CIRCUIT_OPEN', 'PROMPT_CUT'))
        self.assertEqual(ContextBudgetError.code, 'CONTEXT_OVERFLOW')
        # A failure raised the old way (a test double, an older caller) is still named, without a kind.
        self.assertEqual((ProviderFailure('x').code, ProviderFailure('x').kind), ('OLLAMA_ERROR', None))
        # The two-argument form takes its code from the kind; the kind is unchanged.
        self.assertEqual((failure('m', 'output_truncated').code, failure('m', 'output_truncated').kind), ('OUTPUT_TRUNCATED', 'output_truncated'))
        self.assertEqual(failure('m', 'prompt_cut').code, 'PROMPT_CUT')
        self.assertEqual(failure('m', 'transport').code, 'OLLAMA_ERROR')
        self.assertEqual(failure('m', 'transport', 'TIMEOUT').code, 'TIMEOUT')
        with self.assertRaises(ValueError):
            failure('m', 'answer', 'SOMETHING_ELSE')

    def test_a_model_that_outlives_its_time_limit_is_a_timeout(self):
        provider = OllamaProvider('m', 'd', LOCAL, 10, True)

        def slow():
            yield b'{"message": {"content": ""}, "done": false}\n'
            with patch('regchain.extraction.providers.time.monotonic', return_value=time.monotonic() + 11):
                yield b'{"message": {"content": ""}, "done": false}\n'
            yield final()
        exc, entry = self.fails(provider, slow())
        self.assertEqual((exc.code, exc.kind), ('TIMEOUT', 'transport'))
        self.assertEqual(provider.transport_failures, 1)                             # still counted by the circuit breaker
        # The HTTP client's own read timeout is the same failure (v0.18 called it a generic request failure).
        exc, entry = self.fails(OllamaProvider('m', 'd', LOCAL, 30), side_effect=httpx.ReadTimeout('slow'))
        self.assertEqual((exc.code, exc.kind, type(exc.__cause__)), ('TIMEOUT', 'transport', httpx.ReadTimeout))
        self.assertEqual((entry['status'], entry['error']), ('PROVIDER_FAILURE', 'ReadTimeout'))

    def test_a_truncated_answer_keeps_what_the_runtime_reported_and_logs_the_real_window(self):
        # The C02 case of the v0.18 final run: a runaway reasoning that hit num_predict, on an 8k judge.
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192)
        chunk = final('{', done_reason='length', prompt_eval_count=624, eval_count=4096, load_duration=2_500_000_000)
        with self.assertLogs('regchain.ai', 'WARNING') as logs, ai_task('judge.supports'):
            exc, entry = self.fails(provider, [chunk])
        self.assertEqual((exc.code, exc.kind), ('OUTPUT_TRUNCATED', 'output_truncated'))
        self.assertEqual((entry['done_reason'], entry['prompt_tokens'], entry['output_tokens'], entry['load_duration_ms'], entry['num_ctx']),
                         ('length', 624, 4096, 2500, 8192))
        line = json.loads(logs.records[0].getMessage())
        self.assertEqual((line['event'], line['num_ctx'], line['prompt_tokens']), ('OUTPUT_TRUNCATED', 8192, 624))
        # A prompt the runtime cut: reported as half of the real window (4,096 of 8,192), logged as 8k.
        provider = OllamaProvider('m', 'd', LOCAL, 30, None, 4096, 8192)
        with self.assertLogs('regchain.ai', 'WARNING') as logs:
            exc, entry = self.fails(provider, [final(prompt_eval_count=4096)], {'text': 'x' * 9000})
        self.assertEqual((exc.code, exc.kind, entry['done_reason'], entry['prompt_tokens']), ('PROMPT_CUT', 'prompt_cut', 'stop', 4096))
        self.assertEqual(json.loads(logs.records[0].getMessage())['num_ctx'], 8192)
        self.assertEqual(provider.transport_failures, 0)                             # the service answered

    def test_an_empty_answer_is_its_own_failure_and_is_never_cached(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        exc, entry = self.fails(provider, [final('  \n ')], {'passage': 'x'})
        self.assertEqual((exc.code, exc.kind), ('EMPTY_RESPONSE', 'answer'))
        self.assertEqual((entry['done_reason'], entry['prompt_tokens']), ('stop', 300))
        self.assertEqual((provider.cache, provider.transport_failures), ({}, 0))
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            fake_client(factory)
            self.assertEqual(provider.generate_structured('p', {'passage': 'x'}, SCHEMA), '{"a": 1}')
        self.assertNotIn('failure_code', provider.call_log[-1])                      # an answer has no failure code

    def test_retries_that_run_out_are_retry_exhausted_and_a_refusal_is_not(self):
        busy = httpx.Response(503, request=httpx.Request('POST', LOCAL + '/api/chat'))
        refused = httpx.Response(400, request=busy.request)
        broken = httpx.Response(501, request=busy.request)
        for response, code, kind in [(busy, 'RETRY_EXHAUSTED', 'transport'), (refused, 'OLLAMA_ERROR', 'refused'),
                                     (broken, 'OLLAMA_ERROR', 'transport')]:
            with self.subTest(status=response.status_code):
                provider = OllamaProvider('m', 'd', LOCAL, 30)
                with patch('regchain.extraction.providers.httpx.Client') as factory, patch('regchain.extraction.providers.time.sleep'):
                    client = fake_client(factory)
                    client.stream.return_value.__enter__.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
                        'x', request=response.request, response=response)
                    with self.assertRaises(ProviderFailure) as caught:
                        provider.generate_structured('p', {'n': 1}, SCHEMA)
                self.assertEqual((caught.exception.code, caught.exception.kind), (code, kind))
                self.assertEqual(provider.call_log[-1]['failure_code'], code)
        exc, entry = self.fails(OllamaProvider('m', 'd', LOCAL, 30), side_effect=httpx.ConnectError('refused'))
        self.assertEqual((exc.code, exc.kind, entry['retry_count']), ('RETRY_EXHAUSTED', 'transport', 2))
        self.assertIn('not reachable', str(exc))                                     # the message is unchanged

    def test_an_open_circuit_is_named_and_fails_without_a_request(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        for n in range(CIRCUIT_FAILURES):
            self.fails(provider, payload={'n': n}, side_effect=httpx.ConnectError('down'))
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            with self.assertRaises(ProviderFailure) as caught:
                provider.generate_structured('p', {'n': 'next'}, SCHEMA)
            factory.assert_not_called()
        self.assertEqual((caught.exception.code, provider.call_log[-1]['status'], provider.call_log[-1]['failure_code']),
                         ('CIRCUIT_OPEN', 'CIRCUIT_OPEN', 'CIRCUIT_OPEN'))
        self.assertTrue(str(caught.exception).startswith('circuit open: 5 consecutive transport failures'))

    def test_the_other_faults_of_the_service_and_of_the_answer_are_named(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client = fake_client(factory)
            client.get.return_value.json.return_value = {'models': [{'name': 'm', 'digest': 'other'}]}
            with self.assertRaises(ProviderFailure) as caught:
                provider.generate_structured('p', {'n': 'digest'}, SCHEMA)
        self.assertEqual((caught.exception.code, caught.exception.kind, provider.transport_failures), ('OLLAMA_ERROR', 'digest', 0))
        cases = [('stream ended', [b'{"message": {"content": "{"}, "done": false}\n'], 'OLLAMA_ERROR', 'transport'),
                 ('size limit', [b' ' * 2_000_001], 'OLLAMA_ERROR', 'answer'),
                 ('not stop', [final(done_reason='load')], 'OLLAMA_ERROR', 'answer'),
                 ('broken chunk', [b'{"message": \n'], 'MALFORMED_JSON', 'answer'),
                 # JSON, but not an object: v0.18 let the AttributeError escape the provider unnamed.
                 ('not an object', [b'[1, 2]\n'], 'MALFORMED_JSON', 'answer')]
        for name, chunks, code, kind in cases:
            with self.subTest(name):
                provider = OllamaProvider('m', 'd', LOCAL, 30)
                exc, entry = self.fails(provider, chunks)
                self.assertEqual((exc.code, exc.kind), (code, kind))
        # A failure before any answer arrived does not borrow the numbers of the call before it.
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            fake_client(factory)
            provider.generate_structured('p', {'n': 'answered'}, SCHEMA)
        exc, entry = self.fails(provider, side_effect=httpx.ConnectError('down'))
        self.assertFalse({'done_reason', 'prompt_tokens', 'output_tokens'} & set(entry))

    def test_a_context_overflow_is_named_on_the_exception_and_on_the_record(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192)
        with patch('regchain.extraction.providers.httpx.Client') as factory, self.assertLogs('regchain.ai', 'WARNING') as logs:
            with self.assertRaises(ContextBudgetError) as caught:
                provider.generate_structured('p', {'passage': 'x' * 20000}, SCHEMA)
            factory.assert_not_called()
        self.assertEqual((caught.exception.code, failure_code(caught.exception)), ('CONTEXT_OVERFLOW', 'CONTEXT_OVERFLOW'))
        entry = provider.call_log[-1]
        self.assertEqual((entry['status'], entry['failure_code'], entry['num_ctx']), ('CONTEXT_BUDGET_EXCEEDED', 'CONTEXT_OVERFLOW', 8192))
        self.assertEqual(json.loads(logs.records[0].getMessage())['num_ctx'], 8192)

    def test_failure_code_names_any_exception_a_model_call_ends_with(self):
        class Answer(BaseModel):
            label: int
        with self.assertRaises(ValidationError) as invalid:
            Answer.model_validate_json('{"label": "many"}')
        with self.assertRaises(json.JSONDecodeError) as broken:
            json.loads('{"label": ')
        with self.assertRaises(AttributeError) as not_object:
            json.loads('null').get('label')
        for exc, code in [(invalid.exception, 'MALFORMED_JSON'), (broken.exception, 'MALFORMED_JSON'),
                          (not_object.exception, 'MALFORMED_JSON'), (TypeError('list indices'), 'MALFORMED_JSON'),
                          (httpx.ReadTimeout('slow'), 'TIMEOUT'), (TimeoutError(), 'TIMEOUT'),
                          (ContextBudgetError('over'), 'CONTEXT_OVERFLOW'), (ProviderFailure('old style'), 'OLLAMA_ERROR'),
                          (failure('t', 'transport', 'TIMEOUT'), 'TIMEOUT'), (failure('x', 'output_truncated'), 'OUTPUT_TRUNCATED'),
                          (RuntimeError('boom'), 'OLLAMA_ERROR'), (ValueError('quote is not a span'), 'OLLAMA_ERROR')]:
            with self.subTest(exc=type(exc).__name__, code=code):
                self.assertEqual(failure_code(exc), code)

    def test_an_answer_records_its_window_and_a_model_load_when_the_runtime_reports_one(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30, None, 4096, 8192)
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client = fake_client(factory, [final(load_duration=1_234_567_890)])
            provider.generate_structured('p', {'n': 1}, SCHEMA)
            client.stream.return_value.__enter__.return_value.iter_bytes.return_value = [final()]
            provider.generate_structured('p', {'n': 2}, SCHEMA)
            provider.generate_structured('p', {'n': 2}, SCHEMA)
        loaded, plain, hit = provider.call_log
        self.assertEqual((loaded['status'], loaded['num_ctx'], loaded['load_duration_ms']), ('OK', 8192, 1234))
        self.assertNotIn('load_duration_ms', plain)
        self.assertEqual((hit['cache_hit'], hit['num_ctx']), (True, 8192))


class AdaptiveWindowTests(unittest.TestCase):
    """The judge window of the v0.19 addendum (24 September 2026): JUDGE_NUM_CTX is the base window and a
    narrower judge keeps wide twins at JUDGE_NUM_CTX_LARGE; 'adaptive' lets fit_call fall back to them
    (test_v019_context). The first v0.19 design, small twins chosen per obligation with hysteresis, is retired."""
    V018_KEYS = {'provider', 'model_version', 'contract', 'thinking', 'num_ctx', 'num_predict', 'temperature', 'timeout_seconds',
                 'adapter', 'quick'}

    def test_the_fixed_default_is_the_v018_judge_and_its_manifest_is_unchanged(self):
        default = judge()
        self.assertEqual(runtime_manifest(default), {
            'provider': 'ollama', 'model_version': 'm@d', 'contract': CONTRACT_VERSION, 'thinking': True, 'num_ctx': NUM_CTX,
            'num_predict': JUDGE_THINKING_PREDICT, 'temperature': '0', 'timeout_seconds': '300.0', 'adapter': 'ollama-v3',
            'quick': {'thinking': False, 'num_predict': NUM_PREDICT}})
        # 'fixed' spelled out is the same judge; the retired small window setting is not even read.
        fixed = judge(JUDGE_CTX_MODE='fixed', JUDGE_NUM_CTX_SMALL='not a number')
        self.assertEqual(runtime_manifest(fixed), runtime_manifest(default))
        for instance in (default, default.quick, fixed):
            self.assertIsNone(getattr(instance, 'small', None))
            self.assertIsNone(getattr(instance, 'wide', None))                       # a 16k judge has nothing wider
            self.assertEqual(instance.ctx_mode, 'fixed')
        # The v0.18 narrow window keeps its full-window twin and its manifest shape in fixed mode.
        narrow = judge(JUDGE_NUM_CTX='8192')
        self.assertEqual(set(runtime_manifest(narrow)), self.V018_KEYS | {'wide_num_ctx'})
        self.assertEqual((narrow.ctx_mode, narrow.wide.ctx_mode, narrow.quick.wide.ctx_mode), ('fixed', 'fixed', 'fixed'))

    def test_adaptive_falls_back_to_the_wide_twin_which_shares_the_log_the_cache_and_the_circuit(self):
        big = judge(JUDGE_CTX_MODE='adaptive', JUDGE_NUM_CTX='8192')
        quick = big.quick
        self.assertEqual((big.num_ctx, quick.num_ctx, big.wide.num_ctx, quick.wide.num_ctx), (8192, 8192, NUM_CTX, NUM_CTX))
        # At 8k the thinking judge keeps half its window for the answer (4,096; the measured thinking answers
        # needed at most 3,211 tokens), the wide twin its whole JUDGE_THINKING_PREDICT.
        self.assertEqual((big.num_predict, quick.num_predict, big.wide.num_predict, quick.wide.num_predict),
                         (4096, 4096, JUDGE_THINKING_PREDICT, NUM_PREDICT))
        self.assertEqual((big.wide.thinking, quick.wide.thinking), (True, False))
        for twin, of in ((big.wide, big), (quick.wide, quick)):
            self.assertIs(twin.call_log, of.call_log)
            self.assertIs(twin.cache, of.cache)
            self.assertIs(twin.circuit, of.circuit)
            self.assertIs(twin.placement, of.placement)
            self.assertEqual((twin.ctx_mode, of.ctx_mode), ('adaptive', 'adaptive'))
            self.assertIsNone(getattr(twin, 'wide', None))                           # nothing wider behind the fallback
            self.assertEqual((twin.model_version, twin.base_url, twin.timeout), (of.model_version, of.base_url, of.timeout))
        self.assertIs(quick.placement, big.placement)                                 # one model on one service
        manifest = runtime_manifest(big)
        self.assertEqual((manifest['ctx_mode'], manifest['wide_num_ctx'], manifest['num_ctx']), ('adaptive', NUM_CTX, 8192))
        self.assertEqual({k: v for k, v in manifest.items() if k != 'ctx_mode'}, runtime_manifest(judge(JUDGE_NUM_CTX='8192')))
        # One question: asked at 16k through the wide twin, served to the 8k judge from the shared cache.
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client = fake_client(factory)
            self.assertEqual(big.wide.generate_structured('p', {'passage': 'x'}, SCHEMA), '{"a": 1}')
            self.assertEqual(big.generate_structured('p', {'passage': 'x'}, SCHEMA), '{"a": 1}')
            self.assertEqual(client.stream.call_count, 1)
            self.assertEqual(client.stream.call_args.kwargs['json']['options']['num_ctx'], NUM_CTX)
        self.assertEqual([(e['num_ctx'], e['cache_hit']) for e in big.call_log], [(NUM_CTX, False), (8192, True)])
        # A service that is down opens the circuit for both windows at once.
        with patch('regchain.extraction.providers.httpx.Client') as factory, patch('regchain.extraction.providers.time.sleep'):
            fake_client(factory).stream.side_effect = httpx.ConnectError('down')
            for n in range(CIRCUIT_FAILURES):
                with self.assertRaises(ProviderFailure):
                    big.generate_structured('p', {'n': n}, SCHEMA)
            factory.reset_mock()
            with self.assertRaises(ProviderFailure) as caught:
                big.wide.generate_structured('p', {'n': 'wide'}, SCHEMA)
            factory.assert_not_called()
        self.assertEqual((caught.exception.code, big.wide.transport_failures), ('CIRCUIT_OPEN', CIRCUIT_FAILURES))
        # The fallback window is a setting of its own.
        wider = judge(JUDGE_CTX_MODE='adaptive', JUDGE_NUM_CTX='8192', JUDGE_NUM_CTX_LARGE='12288')
        self.assertEqual((wider.wide.num_ctx, wider.quick.wide.num_ctx, wider.wide.num_predict), (12288, 12288, JUDGE_THINKING_PREDICT))
        # Without thinking there is one judge and one wide twin.
        plain = judge(JUDGE_CTX_MODE='adaptive', JUDGE_NUM_CTX='8192', JUDGE_THINKING='')
        self.assertIs(plain.quick, plain)
        self.assertEqual((plain.num_predict, plain.wide.num_ctx, plain.wide.num_predict), (NUM_PREDICT, NUM_CTX, NUM_PREDICT))

    def test_a_misconfigured_window_is_refused_not_guessed(self):
        for bad in ({'JUDGE_CTX_MODE': 'auto'},
                    {'JUDGE_CTX_MODE': 'adaptive'},                                   # a 16k judge has nothing to fall back to
                    {'JUDGE_CTX_MODE': 'adaptive', 'JUDGE_NUM_CTX': '8192', 'JUDGE_NUM_CTX_LARGE': '8192'},
                    {'JUDGE_NUM_CTX_LARGE': 'lots'}, {'JUDGE_NUM_CTX_LARGE': '1024'}, {'JUDGE_NUM_CTX_LARGE': '262144'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                judge(**bad)
        # The product setting (8k base, adaptive) is valid; the retired small window setting is ignored.
        product = judge(JUDGE_CTX_MODE='adaptive', JUDGE_NUM_CTX='8192', JUDGE_NUM_CTX_SMALL='lots')
        self.assertEqual((product.num_ctx, product.wide.num_ctx, product.ctx_mode), (8192, NUM_CTX, 'adaptive'))

    def test_window_for_is_retired_and_hands_back_what_it_was_given(self):
        adaptive = judge(JUDGE_CTX_MODE='adaptive', JUDGE_NUM_CTX='8192')
        for provider in (judge(), adaptive, adaptive.quick, adaptive.wide, types.SimpleNamespace(name='fixture')):
            for need in (10, 50000, None):
                self.assertIs(window_for(provider, need), provider)
        # No per-obligation window state is kept any more: fit_call decides per call.
        self.assertFalse(hasattr(adaptive, 'window_switches') or hasattr(adaptive, 'window_state'))

    def test_judge_num_predict_is_checked_against_the_real_window(self):
        # A 32k judge may reason longer than 8,192 tokens (v0.18 compared the budget with 16k // 2).
        wide = judge(JUDGE_NUM_CTX='32768', JUDGE_NUM_PREDICT='12000')
        self.assertEqual((wide.num_ctx, wide.num_predict), (32768, 12000))
        with self.assertRaises(ValueError):
            judge(JUDGE_NUM_CTX='32768', JUDGE_NUM_PREDICT='20000')
        with self.assertRaises(ValueError):
            judge(JUDGE_NUM_PREDICT=str(NUM_CTX))
        # A narrower judge keeps the v0.18 rule: half its own window, the full budget on the wide twin.
        narrow = judge(JUDGE_NUM_CTX='8192', JUDGE_NUM_PREDICT='6144')
        self.assertEqual((narrow.num_predict, narrow.wide.num_predict), (4096, 6144))


class FakeLibrary:
    """A sentence_transformers stand-in whose weights are not in the local cache."""
    seen = []

    def __init__(self, model, max_length=512, device=None, local_files_only=False):
        FakeLibrary.seen.append({'model': model, 'local_files_only': local_files_only})
        if local_files_only:
            raise OSError(f'{model} is not in the local Hugging Face cache')
        self.device = 'cpu'


class RerankerFallbackTests(unittest.TestCase):
    def setUp(self):
        FakeLibrary.seen = []
        self.name = 'fixture/not-cached-' + uuid.uuid4().hex
        self.addCleanup(rerank_module._INSTANCES.pop, self.name, None)

    def test_a_model_missing_from_the_local_cache_is_not_downloaded_and_retrieval_keeps_the_fused_order(self):
        module = types.ModuleType('sentence_transformers')
        module.CrossEncoder = FakeLibrary
        with patch.dict(sys.modules, {'sentence_transformers': module}), \
                patch.dict(os.environ, {'RERANK_MODEL': self.name, 'RERANK_ALLOW_DOWNLOAD': '', 'RERANK_DEVICE': ''}):
            reranker, status = configured_reranker()
            self.assertIsNone(reranker)
            self.assertEqual(FakeLibrary.seen, [{'model': self.name, 'local_files_only': True}])
            self.assertEqual((status['status'], status['model']), ('unavailable', self.name))
            for said in (self.name, 'not available locally', 'OSError', 'the RRF order is used', 'RERANK_ALLOW_DOWNLOAD=on'):
                self.assertIn(said, status['reason'])
            self.assertNotIn(self.name, rerank_module._INSTANCES)                     # asked again next run, never remembered
            # The retrieval pipeline runs without it: the fused order, and the manifest says reranking is off.
            index = PolicyIndex(passages(), ConceptEmbedder(), reranker)
            chosen, record = index.select(DUTY)
            self.assertEqual(index.manifest()['method'], FUSED_VERSION)
            self.assertEqual(index.manifest()['reranker'], {'status': 'off'})
            self.assertTrue(all(row['rerank_score'] is None for row in record['shown']))
            self.assertEqual([c['source_id'] for c in chosen], [c['source_id'] for c in PolicyIndex(passages(), ConceptEmbedder()).select(DUTY)[0]])
            # Only the operator's explicit switch lets the library fetch the weights.
            with patch.dict(os.environ, {'RERANK_ALLOW_DOWNLOAD': 'on'}):
                reranker, status = configured_reranker()
            self.assertEqual(FakeLibrary.seen[-1], {'model': self.name, 'local_files_only': False})
            self.assertEqual(status['status'], 'active')

    @unittest.skipUnless(importlib.util.find_spec('sentence_transformers'), 'sentence-transformers is not installed')
    def test_the_installed_library_does_not_reach_the_network_for_a_model_it_does_not_have(self):
        attempts = []

        def blocked(*args, **kwargs):
            attempts.append(args[:2])
            raise OSError('network blocked by the test')
        with patch.object(socket.socket, 'connect', blocked), patch.object(socket.socket, 'connect_ex', blocked), \
                patch.object(socket, 'create_connection', blocked), patch.object(socket, 'getaddrinfo', blocked), \
                patch.dict(os.environ, {'RERANK_MODEL': self.name, 'RERANK_ALLOW_DOWNLOAD': '', 'RERANK_DEVICE': ''}):
            reranker, status = configured_reranker()
        self.assertIsNone(reranker)
        self.assertEqual(status['status'], 'unavailable')
        self.assertIn('the RRF order is used', status['reason'])
        self.assertEqual(attempts, [])


if __name__ == '__main__':
    unittest.main()

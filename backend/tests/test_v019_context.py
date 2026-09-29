"""v0.19 context management (addendum of 24 September 2026, WS6a providers).

qwen3:8b with a 16,384-token window ran 19.5% on the CPU of an 8 GB GPU; at 8,192 it is all on the
GPU. These tests pin: fit_call's actions (fits / compressed / fallback_large / trimmed, a prompt that
fits the base window never going to 16k, fixed mode never using the wide twin), the one retry of a
truncated thinking answer on the wide twin in adaptive mode only, the per-call record (model,
num_ctx, estimates, window_action, timings, tokens per second, the processor split from GET /api/ps
asked only on a first call, a reload or a window change), fast_provider, the redaction allowlists,
and the product defaults (8k base window, adaptive as one value in the workspace). Transport-level
fakes only: no Ollama, no network.
"""
import json
import os
import re
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from regchain.extraction.providers import (ADMISSION_MARGIN, AI_WINDOW, CTX_MODES, NUM_CTX, NUM_PREDICT, RELOAD_MS, UNKNOWN_SPLIT, WINDOW_ACTIONS,
                                           ContextBudgetError, OllamaProvider, ProviderFailure, RulesProvider, admissible_tokens,
                                           call_fitted, configured_judge, configured_provider, estimate_tokens, failure,
                                           fast_provider, fit_call, payload_tokens, processor_split, runtime_manifest, timings,
                                           truncation_fallback)
from regchain.pilot.workspace import JUDGE_WINDOW_DEFAULTS, PRODUCT_DEFAULTS, product_defaults
from regchain.platform.redaction import ALLOWED_CALL_FIELDS, FORBIDDEN_CALL_FIELDS, RedactionPolicy, redact_call
from test_v018_reliability import LOCAL, SCHEMA, fake_client
from test_v019_reliability import final, judge, judge_env

ROOT = Path(__file__).resolve().parents[2]
CLIENT = 'regchain.extraction.providers.httpx.Client'
# GET /api/ps as Ollama answers it: qwen3:8b at 8k all in VRAM, at 16k with a fifth on the CPU (measured sizes).
PS_GPU = {'models': [{'name': 'm', 'model': 'm', 'size': 6_190_000_000, 'size_vram': 6_190_000_000, 'digest': 'd'}]}
PS_SPILL = {'models': [{'name': 'm', 'model': 'm', 'size': 7_810_000_000, 'size_vram': 6_287_000_000, 'digest': 'd'}]}


def sized(tokens, prompt='p', schema=SCHEMA):
    """A {'passage': ...} payload whose request is estimated at exactly `tokens`."""
    low, high = 0, tokens * 3
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_tokens(prompt, {'passage': 'x' * middle}, schema) <= tokens:
            low = middle
        else:
            high = middle - 1
    payload = {'passage': 'x' * low}
    assert estimate_tokens(prompt, payload, schema) == tokens
    return payload


def answer(tokens, content='{"a": 1}', **fields):
    """A streamed answer whose prompt count matches a request of about `tokens` estimated tokens (the
    fake's usual 300 would look like a prompt the runtime cut)."""
    return final(content, prompt_eval_count=tokens, **fields)


def shrink_to(tokens):
    """A compressor that cuts the passage until the payload is at most `tokens` (payload_tokens), or at
    most the target fit_call gives it when None: a number stands for compression that falls short."""
    def compress(payload, target):
        goal = target if tokens is None else tokens
        while payload_tokens(payload) > goal and payload['passage']:
            payload['passage'] = payload['passage'][:-100]
        return payload
    return compress


def adaptive(**changes):
    return judge(JUDGE_CTX_MODE='adaptive', JUDGE_NUM_CTX='8192', **changes)


class Isolated(unittest.TestCase):
    """Every test starts without fit decisions waiting from another test."""

    def setUp(self):
        token = AI_WINDOW.set(None)
        self.addCleanup(AI_WINDOW.reset, token)


class FitCallTests(Isolated):
    def test_a_request_that_fits_the_base_window_never_goes_to_16k(self):
        big = adaptive()
        # 8,192 - ADMISSION_MARGIN - the floor of a 4,096 budget (2,048): an estimate of 5,120 still fits.
        self.assertEqual(admissible_tokens(big), 8192 - ADMISSION_MARGIN - 2048)
        asked = []
        for tokens in (100, 2000, admissible_tokens(big)):
            payload = sized(tokens)
            used, sent, info = fit_call(big, 'p', payload, SCHEMA, lambda value, target: asked.append(target))
            self.assertIs(used, big)
            self.assertIs(sent, payload)
            self.assertEqual((info['action'], info['mode'], info['num_ctx'], info['base_num_ctx'], info['estimated_tokens']),
                             ('fits', 'adaptive', 8192, 8192, tokens))
        self.assertEqual(asked, [])                                                  # nothing to compress
        self.assertIs(fit_call(big.quick, 'p', sized(5000), SCHEMA)[0], big.quick)
        # The live call goes out at 8k and its record says what was decided.
        with patch(CLIENT) as factory:
            client = fake_client(factory, [answer(5000)])
            used.generate_structured('p', sent, SCHEMA)
            self.assertEqual(client.stream.call_args.kwargs['json']['options']['num_ctx'], 8192)
        entry = big.call_log[-1]
        self.assertEqual((entry['model'], entry['num_ctx'], entry['window_action'], entry['estimated_prompt_tokens']),
                         ('m', 8192, 'fits', admissible_tokens(big)))
        self.assertIn('elapsed_ms', entry)
        self.assertNotIn('window_estimate_before', entry)
        self.assertEqual(set(WINDOW_ACTIONS), {'fits', 'compressed', 'fallback_large', 'trimmed', 'compact_after_truncation',
                                               'fallback_large_after_truncation'})

    def test_a_long_request_is_compressed_first_and_stays_at_the_base_window(self):
        big = adaptive()
        payload = sized(7000)
        handed = []

        def compress(value, target):
            handed.append((value is payload, target))
            return shrink_to(None)(value, target)
        used, sent, info = fit_call(big, 'p', payload, SCHEMA, compress)
        self.assertIs(used, big)
        self.assertEqual((info['action'], info['num_ctx'], info['estimated_tokens']), ('compressed', 8192, 7000))
        self.assertEqual(info['estimated_tokens_after'], estimate_tokens('p', sent, SCHEMA))
        self.assertLessEqual(info['estimated_tokens_after'], admissible_tokens(big))
        # The compressor works on a copy and is told the payload's own budget (payload_tokens units).
        self.assertEqual(len(handed), 1)
        self.assertFalse(handed[0][0])
        self.assertEqual(len(payload['passage']), len(sized(7000)['passage']))
        self.assertLessEqual(payload_tokens(sent), handed[0][1])
        self.assertLess(handed[0][1], admissible_tokens(big))
        with patch(CLIENT) as factory:
            fake_client(factory, [answer(5000)])
            used.generate_structured('p', sent, SCHEMA)
        entry = big.call_log[-1]
        self.assertEqual((entry['window_action'], entry['window_estimate_before'], entry['num_ctx']), ('compressed', 7000, 8192))

    def test_only_what_compression_cannot_fit_falls_back_to_the_wide_twin(self):
        big = adaptive()
        payload = sized(9000)
        used, sent, info = fit_call(big, 'p', payload, SCHEMA)
        self.assertEqual((used, sent, info['action'], info['num_ctx'], info['base_num_ctx']), (big.wide, payload, 'fallback_large', NUM_CTX, 8192))
        # A compressor that does not reach 8k: the ORIGINAL payload goes to 16k when it fits there.
        used, sent, info = fit_call(big, 'p', payload, SCHEMA, shrink_to(6500))
        self.assertEqual((used, sent, info['action']), (big.wide, payload, 'fallback_large'))
        self.assertNotIn('estimated_tokens_after', info)
        # Too long even for 16k, but the compressed payload fits there: that one goes.
        huge = sized(14000)
        used, sent, info = fit_call(big, 'p', huge, SCHEMA, shrink_to(10000))
        self.assertIs(used, big.wide)
        self.assertEqual(info['action'], 'fallback_large')
        self.assertLessEqual(info['estimated_tokens_after'], admissible_tokens(big.wide))
        self.assertLess(len(sent['passage']), len(huge['passage']))
        # The quick instance falls back to its own wide twin (no thinking, 16k).
        self.assertIs(fit_call(big.quick, 'p', payload, SCHEMA)[0], big.quick.wide)
        with patch(CLIENT) as factory:
            client = fake_client(factory, [answer(10000)])
            used.generate_structured('p', sent, SCHEMA)
            options = client.stream.call_args.kwargs['json']['options']
        self.assertEqual((options['num_ctx'], client.stream.call_args.kwargs['json']['think']), (NUM_CTX, True))
        entry = big.call_log[-1]
        self.assertEqual((entry['window_action'], entry['num_ctx'], entry['window_estimate_before']), ('fallback_large', NUM_CTX, 14000))

    def test_fixed_mode_never_uses_the_wide_twin(self):
        fixed = judge(JUDGE_NUM_CTX='8192')
        self.assertEqual((fixed.ctx_mode, fixed.wide.num_ctx), ('fixed', NUM_CTX))   # the v18 applicability twin exists
        payload = sized(9000)                                                         # would fit 16k
        with self.assertLogs('regchain.ai', 'WARNING') as logs:
            used, sent, info = fit_call(fixed, 'p', payload, SCHEMA)
        line = json.loads(logs.records[-1].getMessage())
        self.assertEqual((line['event'], line['num_ctx'], line['estimated_prompt_tokens']), ('CONTEXT_TRIMMED', 8192, 9000))
        self.assertIs(used, fixed)
        self.assertEqual((info['action'], info['mode'], info['num_ctx']), ('trimmed', 'fixed', 8192))
        self.assertLessEqual(estimate_tokens('p', sent, SCHEMA), admissible_tokens(fixed))
        self.assertTrue(info['notes'])
        self.assertIn('[...]', sent['passage'][-10:])
        used, sent, info = fit_call(fixed, 'p', payload, SCHEMA, shrink_to(None))
        self.assertEqual((used, info['action']), (fixed, 'compressed'))
        with self.assertLogs('regchain.ai', 'WARNING'):
            used, _, info = fit_call(fixed, 'p', payload, SCHEMA, shrink_to(6500))   # compression that falls short is trimmed further
        self.assertEqual((used, info['action']), (fixed, 'trimmed'))
        self.assertIsNone(truncation_fallback(fixed, 'p', payload, SCHEMA, failure('cut', 'output_truncated')))
        # A double that carries a wider twin but no adaptive mode is not widened either.
        stand_in = types.SimpleNamespace(num_ctx=8192, num_predict=4096, wide=fixed.wide)
        with self.assertLogs('regchain.ai', 'WARNING'):
            self.assertIs(fit_call(stand_in, 'p', payload, SCHEMA)[0], stand_in)

    def test_what_trimming_cannot_fit_is_refused_and_adaptive_trims_at_16k(self):
        big = adaptive()
        with self.assertLogs('regchain.ai', 'WARNING'):
            used, sent, info = fit_call(big, 'p', sized(30000), SCHEMA)
        self.assertEqual((used, info['action'], info['num_ctx']), (big.wide, 'trimmed', NUM_CTX))
        self.assertLessEqual(estimate_tokens('p', sent, SCHEMA), admissible_tokens(big.wide))
        self.assertGreater(estimate_tokens('p', sent, SCHEMA), admissible_tokens(big))   # no more cut than 16k needs
        # A duty is never cut: nothing to trim, so the call is refused before any request.
        with patch(CLIENT) as factory, self.assertLogs('regchain.ai', 'WARNING') as logs:
            with self.assertRaises(ContextBudgetError) as caught:
                fit_call(big, 'p', {'duty': 'x' * 60000}, SCHEMA)
            factory.assert_not_called()
        self.assertEqual(caught.exception.code, 'CONTEXT_OVERFLOW')
        line = json.loads(logs.records[-1].getMessage())
        self.assertEqual((line['event'], line['num_ctx']), ('CONTEXT_BUDGET_EXCEEDED', NUM_CTX))
        with self.assertLogs('regchain.ai', 'WARNING'), self.assertRaises(ContextBudgetError):
            fit_call(judge(JUDGE_NUM_CTX='8192'), 'p', {'duty': 'x' * 30000}, SCHEMA)

    def test_a_compressor_that_fails_is_skipped_and_a_test_double_gets_the_default_window(self):
        big = adaptive()

        def broken(value, target):
            raise KeyError('passage')
        with self.assertLogs('regchain.ai', 'WARNING'):
            used, _, info = fit_call(big, 'p', sized(9000), SCHEMA, broken)
        self.assertEqual((used, info['action'], info['compress_error']), (big.wide, 'fallback_large', 'KeyError'))
        stand_in = types.SimpleNamespace(name='fixture')
        used, sent, info = fit_call(stand_in, 'p', {'passage': 'x'}, SCHEMA)
        self.assertEqual((used, info['action'], info['num_ctx'], info['mode']), (stand_in, 'fits', NUM_CTX, 'fixed'))
        # A double may take part in the window modes (the engine's tests count model switches with one).
        wide = types.SimpleNamespace(num_ctx=16384, num_predict=6144)
        narrow = types.SimpleNamespace(num_ctx=8192, num_predict=4096, ctx_mode='adaptive', wide=wide, thinking=True)
        self.assertIs(fit_call(narrow, 'p', sized(9000), SCHEMA)[0], wide)
        self.assertIs(fit_call(narrow, 'p', sized(900), SCHEMA)[0], narrow)

    def test_the_decision_goes_only_into_the_record_of_the_call_it_prepared(self):
        big = adaptive()
        payload = sized(9000)
        used, sent, info = fit_call(big, 'p', payload, SCHEMA)
        with patch(CLIENT) as factory, self.assertLogs('regchain.ai', 'WARNING'):
            fake_client(factory, [answer(9000)])
            used.generate_structured('p', {'passage': 'x' * 22000}, SCHEMA)          # another question
            with self.assertRaises(ContextBudgetError):
                big.generate_structured('p', sent, SCHEMA)                            # the right request on the base instance
            used.generate_structured('p', {**sent, 'validation_feedback': 'x'}, SCHEMA)   # a repair of it
            used.generate_structured('p', sent, SCHEMA)
            used.generate_structured('p', sent, SCHEMA)                               # asked again: a cache hit
        other, base, repair, prepared, hit = big.call_log[-5:]
        for entry in (other, base, repair, hit):
            self.assertNotIn('window_action', entry)
        self.assertEqual((prepared['window_action'], prepared['num_ctx'], prepared['cache_hit']), ('fallback_large', NUM_CTX, False))
        self.assertTrue(hit['cache_hit'])
        self.assertEqual(AI_WINDOW.get(), {})                                          # taken, not left waiting


class TruncationFallbackTests(Isolated):
    def test_a_truncated_thinking_answer_gets_one_more_try_on_the_wide_twin_in_adaptive_mode(self):
        big = adaptive()
        payload = sized(3000)
        with patch(CLIENT) as factory:
            client = fake_client(factory)
            client.stream.return_value.__enter__.return_value.iter_bytes.side_effect = [
                [answer(3000, '{', done_reason='length', eval_count=4096)], [answer(3000, '{"a": 2}', eval_count=5000)]]
            with self.assertLogs('regchain.ai', 'WARNING'):
                reply, info = call_fitted(big, 'p', payload, SCHEMA)
            sent = [call.kwargs['json']['options'] for call in client.stream.call_args_list]
        self.assertEqual(reply, '{"a": 2}')
        self.assertEqual((info['action'], info['num_ctx'], info['base_num_ctx'], info['first']['action']),
                         ('fallback_large_after_truncation', NUM_CTX, 8192, 'fits'))
        self.assertEqual([(o['num_ctx'], o['num_predict']) for o in sent], [(8192, 4096), (NUM_CTX, 6144)])
        cut, retried = big.call_log[-2:]
        self.assertEqual((cut['status'], cut['failure_code'], cut['window_action'], cut['num_ctx']),
                         ('PROVIDER_FAILURE', 'OUTPUT_TRUNCATED', 'fits', 8192))
        self.assertEqual((retried['status'], retried['window_action'], retried['num_ctx']), ('OK', 'fallback_large_after_truncation', NUM_CTX))

    def test_no_other_call_is_retried_at_16k(self):
        big = adaptive()
        truncated = failure('cut', 'output_truncated')
        self.assertIsNotNone(truncation_fallback(big, 'p', sized(500), SCHEMA, truncated))
        for instance, exc in ((big.quick, truncated),                                 # no thinking: nothing to make room for
                              (big.wide, truncated),                                  # already at 16k
                              (big, failure('slow', 'transport', 'TIMEOUT')),
                              (big, failure('empty', 'answer', 'EMPTY_RESPONSE')),
                              (judge(JUDGE_NUM_CTX='8192'), truncated)):             # fixed mode
            with self.subTest(instance=instance.num_ctx, code=exc.code):
                self.assertIsNone(truncation_fallback(instance, 'p', sized(500), SCHEMA, exc))
        # call_fitted raises what it cannot retry, after one request.
        fixed = judge(JUDGE_NUM_CTX='8192')
        with patch(CLIENT) as factory, self.assertLogs('regchain.ai', 'WARNING'):
            client = fake_client(factory, [answer(500, '{', done_reason='length', eval_count=4096)])
            with self.assertRaises(ProviderFailure) as caught:
                call_fitted(fixed, 'p', sized(500), SCHEMA)
        self.assertEqual((caught.exception.code, client.stream.call_count), ('OUTPUT_TRUNCATED', 1))


class ProcessorSplitTests(Isolated):
    def test_the_split_is_asked_on_a_first_call_after_a_load_and_at_another_window_only(self):
        big = adaptive()
        answers = [final(load_duration=6_600_000_000),       # 1 judge 8k, first call, loaded        -> asked
                   final(load_duration=8_000_000),           # 2 judge 8k, warm (8 ms)               -> reused
                   final(load_duration=7_000_000),           # 3 quick 8k, its first call            -> asked
                   final(load_duration=4_900_000_000),       # 4 wide 16k, reloaded at a new window  -> asked
                   final(load_duration=5_000_000),           # 5 wide 16k, warm                      -> reused
                   final(load_duration=9_000_000),           # 6 judge 8k again, warm by the report  -> asked (window changed)
                   final(load_duration=RELOAD_MS * 1_000_000),   # 7 judge 8k, a reload (evicted by the 4b) -> asked
                   final()]                                  # 8 judge 8k, no load reported          -> reused
        callers = [big, big, big.quick, big.wide, big.wide, big, big, big]
        with patch(CLIENT) as factory:
            client = fake_client(factory)
            client.stream.return_value.__enter__.return_value.iter_bytes.side_effect = [[chunk] for chunk in answers]
            client.request.return_value.json.side_effect = [PS_GPU, PS_GPU, PS_SPILL, PS_GPU, PS_GPU]
            for n, caller in enumerate(callers):
                caller.generate_structured('p', {'n': n}, SCHEMA)
            asked = [(call.args, call.kwargs) for call in client.request.call_args_list]
            self.assertTrue(all(call.args[0] == LOCAL + '/api/tags' for call in client.get.call_args_list))   # the digest check is apart
        self.assertEqual(asked, [(('GET', LOCAL + '/api/ps'), {'timeout': 5})] * 5)
        # The judge and its wide twin write one log; the quick instance keeps its own (as in v0.18).
        rows = [(e['num_ctx'], e['processor'], e['gpu_fraction']) for e in big.call_log]
        self.assertEqual(rows, [(8192, '100% GPU', 1.0), (8192, '100% GPU', 1.0),
                                (NUM_CTX, '20%/80% CPU/GPU', 0.805), (NUM_CTX, '20%/80% CPU/GPU', 0.805),
                                (8192, '100% GPU', 1.0), (8192, '100% GPU', 1.0), (8192, '100% GPU', 1.0)])
        self.assertEqual([(e['num_ctx'], e['processor'], e['load_duration_ms']) for e in big.quick.call_log], [(8192, '100% GPU', 7)])
        self.assertEqual([e.get('load_duration_ms') for e in big.call_log], [6600, 8, 4900, 5, 9, RELOAD_MS, None])

    def test_a_failed_or_unusable_answer_is_unknown_and_never_fails_the_call(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch(CLIENT) as factory:
            client = fake_client(factory)
            client.request.side_effect = httpx.ConnectError('ps down')
            self.assertEqual(provider.generate_structured('p', {'n': 1}, SCHEMA), '{"a": 1}')
            client.request.side_effect = None
            provider.generate_structured('p', {'n': 2}, SCHEMA)                     # kept until a load or a new window
            self.assertEqual(client.request.call_count, 1)
            provider.generate_structured('p', {'n': 2}, SCHEMA)                     # a cache hit asks nothing
        first, second, hit = provider.call_log
        self.assertEqual((first['status'], first['processor'], first['gpu_fraction']), ('OK', 'unknown', None))
        self.assertEqual(second['processor'], 'unknown')
        self.assertEqual(provider.transport_failures, 0)                            # not the service failing the answer
        self.assertFalse({'processor', 'gpu_fraction', 'tokens_per_second', 'load_duration_ms', 'window_action'} & set(hit))
        # An HTTP error or an answer that is not the expected JSON is the same 'unknown'.
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch(CLIENT) as factory:
            client = fake_client(factory)
            response = httpx.Response(404, request=httpx.Request('GET', LOCAL + '/api/ps'))
            client.request.return_value.raise_for_status.side_effect = httpx.HTTPStatusError('x', request=response.request, response=response)
            provider.generate_structured('p', {'n': 3}, SCHEMA)
        self.assertEqual(provider.call_log[-1]['processor'], 'unknown')
        for ps in ({'models': []}, {'models': [{'name': 'other', 'size': 10, 'size_vram': 10}]}, {'models': [{'name': 'm', 'size': 0, 'size_vram': 0}]},
                   {'models': [{'name': 'm', 'size': 5, 'size_vram': 9}]}, {'models': [{'name': 'm', 'size': '5', 'size_vram': 5}]},
                   {'models': 'm'}, [], 'text', None):
            with self.subTest(ps=ps):
                self.assertEqual(processor_split(ps, 'm'), UNKNOWN_SPLIT)
        self.assertEqual(processor_split({'models': [{'model': 'm', 'size': 10, 'size_vram': 0}]}, 'm'), {'gpu_fraction': 0.0, 'processor': '100% CPU'})
        self.assertEqual(processor_split(PS_SPILL, 'm'), {'gpu_fraction': 0.805, 'processor': '20%/80% CPU/GPU'})


class TimingTests(Isolated):
    def test_tokens_per_second_is_output_tokens_over_the_generation_time(self):
        self.assertEqual(timings({'eval_count': 300, 'eval_duration': 6_000_000_000, 'prompt_eval_count': 1200,
                                  'prompt_eval_duration': 500_000_000, 'load_duration': 3_200_000_000}),
                         {'load_duration_ms': 3200, 'eval_duration_ms': 6000, 'prompt_eval_duration_ms': 500,
                          'tokens_per_second': 50.0, 'prompt_tokens_per_second': 2400.0})
        self.assertEqual(timings({'eval_count': 9, 'prompt_eval_count': 30}), {})    # nothing reported, nothing made up
        self.assertEqual(timings({'eval_count': 9, 'eval_duration': 0}), {'eval_duration_ms': 0})
        self.assertEqual(timings({'eval_count': True, 'eval_duration': 1_000_000_000}), {'eval_duration_ms': 1000})
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192)
        with patch(CLIENT) as factory:
            client = fake_client(factory, [final(eval_count=250, eval_duration=5_000_000_000, prompt_eval_count=300,
                                                 prompt_eval_duration=150_000_000, load_duration=12_000_000)])
            client.request.return_value.json.return_value = PS_GPU
            provider.generate_structured('p', {'n': 1}, SCHEMA)
            client.stream.return_value.__enter__.return_value.iter_bytes.return_value = [
                final('{', done_reason='length', eval_count=4096, eval_duration=64_000_000_000)]
            with self.assertLogs('regchain.ai', 'WARNING'), self.assertRaises(ProviderFailure):
                provider.generate_structured('p', {'n': 2}, SCHEMA)
        live, cut = provider.call_log
        # Every live call names its model, window, estimate, time, throughput and where it ran.
        for key, value in (('model', 'm'), ('num_ctx', 8192), ('estimated_prompt_tokens', estimate_tokens('p', {'n': 1}, SCHEMA)),
                           ('tokens_per_second', 50.0), ('prompt_tokens_per_second', 2000.0), ('eval_duration_ms', 5000),
                           ('prompt_eval_duration_ms', 150), ('load_duration_ms', 12), ('processor', '100% GPU'), ('gpu_fraction', 1.0)):
            self.assertEqual(live[key], value, key)
        self.assertIsInstance(live['elapsed_ms'], int)
        # A truncated answer keeps them too: that is where a runaway reasoning shows its speed.
        self.assertEqual((cut['failure_code'], cut['tokens_per_second'], cut['eval_duration_ms'], cut['processor']),
                         ('OUTPUT_TRUNCATED', 64.0, 64000, '100% GPU'))


class FastProviderTests(Isolated):
    ENV = {'LLM_MODEL': 'm', 'LLM_MODEL_DIGEST': 'd', 'LLM_THINKING': '', 'LLM_TIMEOUT_SECONDS': '', 'FAST_MODEL': '',
           'FAST_MODEL_DIGEST': '', 'AI_CACHE_DIR': '', 'OLLAMA_BASE_URL': LOCAL}
    V018_KEYS = {'provider', 'model_version', 'contract', 'thinking', 'num_ctx', 'num_predict', 'temperature', 'timeout_seconds', 'adapter'}

    def extraction(self, **changes):
        with patch.dict(os.environ, {**self.ENV, **changes}):
            provider = configured_provider('ollama')
            return provider, fast_provider(provider)

    def test_the_extraction_model_without_thinking_is_its_own_fast_model(self):
        for thinking in ('off', ''):
            with self.subTest(thinking=thinking):
                provider, fast = self.extraction(LLM_THINKING=thinking)
                self.assertIs(fast, provider)
                self.assertEqual(set(runtime_manifest(provider)), self.V018_KEYS)     # the v0.18 manifest shape
        plain = OllamaProvider('m', 'd', LOCAL, 30, False)
        self.assertIs(fast_provider(plain), plain)
        self.assertEqual(set(runtime_manifest(plain)), self.V018_KEYS)

    def test_a_thinking_extraction_model_gets_a_no_thinking_twin_on_the_same_log(self):
        provider, fast = self.extraction(LLM_THINKING='on')
        self.assertIsNot(fast, provider)
        self.assertIs(fast_provider(provider), fast)                                 # built once
        self.assertEqual((fast.model_version, fast.thinking, fast.num_ctx, fast.num_predict), ('m@d', False, NUM_CTX, NUM_PREDICT))
        for shared in ('call_log', 'cache', 'circuit', 'placement'):
            self.assertIs(getattr(fast, shared), getattr(provider, shared), shared)
        self.assertEqual(runtime_manifest(provider)['fast'], {'model_version': 'm@d', 'thinking': False, 'num_ctx': NUM_CTX,
                                                              'num_predict': NUM_PREDICT})
        self.assertNotIn('fast', runtime_manifest(fast))
        with patch(CLIENT) as factory:
            client = fake_client(factory)
            fast.generate_structured('p', {'n': 1}, SCHEMA)
            self.assertIs(client.stream.call_args.kwargs['json']['think'], False)
        self.assertEqual((provider.call_log[-1]['thinking'], provider.call_log[-1]['model']), (False, 'm'))

    def test_another_fast_model_is_pinned_and_named_in_the_manifest(self):
        provider, fast = self.extraction(LLM_THINKING='off', FAST_MODEL='qwen3:1.7b', FAST_MODEL_DIGEST='f')
        self.assertEqual((fast.model_version, fast.thinking, fast.num_ctx), ('qwen3:1.7b@f', False, provider.num_ctx))
        self.assertIs(fast.call_log, provider.call_log)                              # the run's log sees its calls
        self.assertIsNot(fast.circuit, provider.circuit)                             # another model: its own digest check
        self.assertEqual(runtime_manifest(provider)['fast']['model_version'], 'qwen3:1.7b@f')
        # The extraction model named again is the extraction model; a fast model without its digest is refused.
        for named in ({'FAST_MODEL': 'm'}, {'FAST_MODEL': 'm', 'FAST_MODEL_DIGEST': 'd'}):
            provider, fast = self.extraction(**named)
            self.assertIs(fast, provider)
        with self.assertRaises(ValueError):
            self.extraction(FAST_MODEL='qwen3:1.7b')
        # A test double or the rules provider: its own `fast` if it has one, else itself.
        rules = RulesProvider()
        self.assertIs(fast_provider(rules), rules)
        double = types.SimpleNamespace(fast=rules)
        self.assertIs(fast_provider(double), rules)


class RedactionTests(Isolated):
    NUMBERS = {'failure_code': 'OUTPUT_TRUNCATED', 'load_duration_ms': 6600, 'eval_duration_ms': 5000, 'prompt_eval_duration_ms': 150,
               'tokens_per_second': 50.0, 'prompt_tokens_per_second': 2000.0, 'gpu_fraction': 0.805, 'processor': '20%/80% CPU/GPU'}
    WINDOW = {'num_ctx': 16384, 'window_action': 'fallback_large', 'window_estimate_before': 14000}

    def test_the_new_fields_are_kept_at_the_right_levels(self):
        entry = {'status': 'OK', 'model': 'qwen3:8b', **self.NUMBERS, **self.WINDOW}
        self.assertEqual(redact_call(entry, RedactionPolicy('minimal')), entry)
        for level in ('standard', 'strict'):
            with self.subTest(level=level):
                self.assertEqual(redact_call(entry, RedactionPolicy(level)), {'status': 'OK', 'model': 'qwen3:8b', **self.NUMBERS})
        for fields in ALLOWED_CALL_FIELDS.values():
            self.assertTrue(fields.isdisjoint(FORBIDDEN_CALL_FIELDS))

    def test_a_real_record_loses_nothing_at_the_operators_level(self):
        big = adaptive()
        used, sent, _ = fit_call(big, 'p', sized(14000), SCHEMA, shrink_to(10000))
        with patch(CLIENT) as factory:
            client = fake_client(factory, [answer(10000, eval_count=250, eval_duration=5_000_000_000, prompt_eval_duration=1, load_duration=1)])
            client.request.return_value.json.return_value = PS_SPILL
            used.generate_structured('p', sent, SCHEMA)
        entry = big.call_log[-1]
        self.assertTrue({'window_action', 'window_estimate_before', 'num_ctx', 'tokens_per_second', 'processor', 'gpu_fraction',
                         'eval_duration_ms'} <= set(entry))
        self.assertLessEqual(set(entry), ALLOWED_CALL_FIELDS['minimal'])
        self.assertEqual(redact_call(entry, RedactionPolicy('minimal')), entry)


class ProductDefaultTests(Isolated):
    def test_the_judge_window_defaults_follow_a_judge_model_and_the_mode_is_one_value(self):
        mode = JUDGE_WINDOW_DEFAULTS['JUDGE_CTX_MODE']
        self.assertEqual(JUDGE_WINDOW_DEFAULTS, {'JUDGE_NUM_CTX': '8192', 'JUDGE_NUM_CTX_LARGE': '16384', 'JUDGE_CTX_MODE': mode})
        self.assertIn(mode, CTX_MODES)                     # 'adaptive' until the 8-case fixed-8k vs adaptive comparison decides
        environ = {'JUDGE_MODEL': 'qwen3:8b'}
        self.assertEqual(product_defaults(environ), list(PRODUCT_DEFAULTS) + list(JUDGE_WINDOW_DEFAULTS))
        with patch.dict(os.environ, judge_env(**{key: environ[key] for key in JUDGE_WINDOW_DEFAULTS})):
            product = configured_judge(OllamaProvider('small', 'd1', LOCAL, 300))
        self.assertEqual((product.num_ctx, product.num_predict, product.ctx_mode, product.wide.num_ctx), (8192, 4096, mode, 16384))
        # Without a judge model nothing about its window is filled.
        environ = {}
        product_defaults(environ)
        self.assertFalse(set(JUDGE_WINDOW_DEFAULTS) & set(environ))
        # Explicit values win; an explicit window with nothing larger keeps the fixed mode, not a judge that refuses to start.
        with patch.dict(JUDGE_WINDOW_DEFAULTS, {'JUDGE_CTX_MODE': 'adaptive'}):
            for given, expected in (({'JUDGE_CTX_MODE': 'fixed'}, {'JUDGE_NUM_CTX': '8192', 'JUDGE_CTX_MODE': 'fixed'}),
                                    ({'JUDGE_NUM_CTX': '16384'}, {'JUDGE_NUM_CTX': '16384', 'JUDGE_CTX_MODE': None}),
                                    ({'JUDGE_NUM_CTX': '12288', 'JUDGE_NUM_CTX_LARGE': ' '},
                                     {'JUDGE_NUM_CTX_LARGE': '16384', 'JUDGE_CTX_MODE': 'adaptive'}),
                                    ({'JUDGE_NUM_CTX': 'lots'}, {'JUDGE_NUM_CTX': 'lots', 'JUDGE_CTX_MODE': None})):
                with self.subTest(given=given):
                    environ = {'JUDGE_MODEL': 'qwen3:8b', **given}
                    product_defaults(environ)
                    self.assertEqual({key: environ.get(key) for key in expected}, expected)

    def test_the_launchers_set_the_8k_window_and_leave_the_mode_to_the_server(self):
        launcher = (ROOT / 'scripts' / 'launcher' / 'CardamanLauncher.cs').read_text(encoding='utf-8')
        runner = (ROOT / 'scripts' / 'Run-Workspace.ps1').read_text(encoding='utf-8')
        example = (ROOT / '.env.example').read_text(encoding='utf-8')
        workspace = (ROOT / 'backend' / 'src' / 'regchain' / 'pilot' / 'workspace.py').read_text(encoding='utf-8')
        allowlist = re.findall(r'"([A-Z_]+)"', re.search(r'EnvKeys\s*=\s*\{(.*?)\};', launcher, re.S).group(1))
        regex = re.search(r"\$line -match '\^\(([A-Z_|]+)\)=", runner).group(1).split('|')
        for key in ('JUDGE_NUM_CTX', 'JUDGE_NUM_CTX_LARGE', 'JUDGE_CTX_MODE', 'FAST_MODEL', 'FAST_MODEL_DIGEST'):
            self.assertIn(key, allowlist)
            self.assertIn(key, regex)
            self.assertRegex(example, rf'(?m)^{key}=')
        self.assertIn('Default(env, "JUDGE_NUM_CTX", "8192");', launcher)
        self.assertIn("if (-not $env:JUDGE_NUM_CTX) { $env:JUDGE_NUM_CTX = '8192' }", runner)
        # The fixed/adaptive decision is one value in the server, no exe rebuild: the launchers only pass it on.
        self.assertNotIn('Default(env, "JUDGE_CTX_MODE"', launcher)
        self.assertNotIn('$env:JUDGE_CTX_MODE =', runner)
        self.assertEqual(len(re.findall(r"'JUDGE_CTX_MODE': '(?:fixed|adaptive)'", workspace)), 1)
        self.assertTrue(runner.isascii())


if __name__ == '__main__':
    unittest.main()

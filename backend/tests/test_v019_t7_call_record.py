"""v0.19 t7 model-variability protocol: the call record of every model call names the exact request sent (request_hash, the
sha256 of the canonical /api/chat body: model, messages, format, options with temperature, seed, num_ctx and num_predict,
think) beside the v0.19 hashes of its prompt and input, the answer it got (raw_output_sha256) and whether that answer parsed
(parse_ok with parsed_sha256, else parse_failure: the call's failure code or MALFORMED_JSON). The answer text itself goes only
to the evaluation harness's replay sidecar (ai-raw.jsonl), which nothing reads back during a run: two runs of the same code
can then be told apart call by call (same request, other answer = model variability), and a live run replayed offline.
Transport-level fakes only: no Ollama, no model, no network."""
import builtins
import json
import os
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import httpx
from regchain.evaluation import harness
from regchain.evaluation.harness import RAW_LOG, Components, run
from regchain.extraction.providers import (AI_RAW, MEASURED, SEED, TEMPERATURE, ContextBudgetError, OllamaProvider, ProviderFailure,
                                           ai_context, ai_task, raw_answers, request_hash, request_hashes, request_messages, uncached)
from regchain.platform.redaction import ALLOWED_CALL_FIELDS, FORBIDDEN_CALL_FIELDS, RedactionPolicy, redact_call
from test_v018_reliability import LOCAL, SCHEMA, fake_client

CLIENT = 'regchain.extraction.providers.httpx.Client'
REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / 'evaluation' / 'datasets' / 'tr-aml-v1.json'
# A string that must never reach a call record, a packet or results.json; only the replay sidecar holds answer text.
SENTINEL = 'T7-RAW-ANSWER-SENTINEL'
NEW_FIELDS = ('request_hash', 'temperature', 'seed', 'raw_output_sha256', 'parse_ok')


def final(content='{"a": 1}', **fields):
    """The last streamed chunk of an answer, as Ollama sends it."""
    return json.dumps({'message': {'content': content}, 'done': True, 'done_reason': 'stop', 'prompt_eval_count': 300,
                       'eval_count': 9, **fields}).encode()


def digest(text: str) -> str:
    return sha256(text.encode('utf-8')).hexdigest()


def canonical(value) -> str:
    """The canonical JSON the replay hashes are taken over, written out here independently of the provider."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


class Isolated(unittest.TestCase):
    def setUp(self):
        # Nothing inherited from the developer's environment: no on-disk answer cache.
        self.enterContext(patch.dict(os.environ, {'AI_CACHE_DIR': ''}))

    def call(self, provider, payload=None, chunks=None, prompt='p', schema=SCHEMA, **client_changes):
        """(answer, call record, fake client) of one call that must succeed."""
        with patch(CLIENT) as factory, patch('regchain.extraction.providers.time.sleep'):
            client = fake_client(factory, chunks or [final()])
            for name, value in client_changes.items():
                setattr(client.stream, name, value)
            answer = provider.generate_structured(prompt, payload if payload is not None else {'passage': 'x'}, schema)
        return answer, provider.call_log[-1], client

    def fails(self, provider, payload=None, chunks=None, error=ProviderFailure, **client_changes):
        """(exception, call record) of one call that must fail."""
        with patch(CLIENT) as factory, patch('regchain.extraction.providers.time.sleep'):
            client = fake_client(factory, chunks or [final()])
            for name, value in client_changes.items():
                setattr(client.stream, name, value)
            with self.assertRaises(error) as caught:
                provider.generate_structured('p', payload if payload is not None else {'passage': 'x'}, SCHEMA)
        return caught.exception, provider.call_log[-1]


class CallRecordTests(Isolated):
    def test_a_live_record_names_the_request_it_sent_and_the_answer_it_got(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192)
        payload = {'passage': 'Staff keep archive files for six years.'}
        with ai_task('judge.verify'), ai_context(provision_id='p1', obligation_id='o1', evidence_ids=['e1']):
            answer, entry, client = self.call(provider, payload)
        sent = client.stream.call_args.kwargs['json']
        self.assertEqual(entry['request_hash'], digest(canonical(sent)))
        # The v0.19 hashes keep their meaning: prompt_hash is prompt_sha256, input_hash is payload_sha256.
        self.assertEqual((entry['prompt_sha256'], entry['payload_sha256'], entry['schema_sha256']), request_hashes('p', payload, SCHEMA))
        self.assertEqual((entry['model'], entry['model_version'], entry['num_ctx'], entry['num_predict'], entry['thinking']),
                         ('m', 'm@d', 8192, 4096, True))
        self.assertEqual((entry['temperature'], entry['seed'], entry['retry_count']), (0, None, 0))
        self.assertEqual(entry['raw_output_sha256'], digest(answer))
        self.assertEqual((entry['parse_ok'], entry['parsed_sha256']), (True, digest('{"a":1}')))
        self.assertFalse({'parse_failure', 'failure_code'} & set(entry))
        # The v0.19 fields the reports read are still there.
        self.assertEqual((entry['status'], entry['task'], entry['output_tokens'], entry['done_reason'], entry['obligation_id']),
                         ('OK', 'judge.verify', 9, 'stop', 'o1'))

    def test_the_request_sent_is_the_t6_request_byte_for_byte_and_sets_no_seed(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192)
        payload = {'passage': 'x'}
        _, entry, client = self.call(provider, payload)
        sent = client.stream.call_args.kwargs['json']
        self.assertEqual(sent, {'model': 'm', 'stream': True, 'format': SCHEMA, 'messages': request_messages('p', payload, SCHEMA),
                                'options': {'temperature': 0, 'num_predict': 4096, 'num_ctx': 8192}, 'think': True})
        self.assertEqual(list(sent), ['model', 'stream', 'format', 'messages', 'options', 'think'])
        self.assertEqual(list(sent['options']), ['temperature', 'num_predict', 'num_ctx'])
        self.assertEqual((TEMPERATURE, SEED), (0, None))
        # Thinking left to the model's default sends no think key, as before.
        _, _, client = self.call(OllamaProvider('m', 'd', LOCAL, 30), payload)
        self.assertNotIn('think', client.stream.call_args.kwargs['json'])
        self.assertNotIn('seed', client.stream.call_args.kwargs['json']['options'])

    def test_the_request_hash_follows_every_option_sent(self):
        payload = {'passage': 'x'}

        def hash_of(provider, body=payload):
            with uncached():
                return self.call(provider, body)[1]

        base = hash_of(OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192))
        again = hash_of(OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192))
        self.assertEqual(base['request_hash'], again['request_hash'])              # the same request, the same hash
        others = {'window': hash_of(OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 16384)),
                  'thinking': hash_of(OllamaProvider('m', 'd', LOCAL, 30, False, 4096, 8192)),
                  'budget': hash_of(OllamaProvider('m', 'd', LOCAL, 30, True, 2048, 8192)),
                  'payload': hash_of(OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192), {'passage': 'y'})}
        for name, entry in others.items():
            with self.subTest(changed=name):
                self.assertNotEqual(entry['request_hash'], base['request_hash'])
        # A generation budget lowered by admission is part of the request, and the record names the budget sent.
        big = {'passage': 'x' * 30000}
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 16384)
        _, entry, client = self.call(provider, big, [final(prompt_eval_count=10000)])
        sent = client.stream.call_args.kwargs['json']
        self.assertTrue(entry['num_predict_reduced'])
        self.assertEqual((sent['options']['num_predict'], entry['request_hash']), (entry['num_predict'], digest(canonical(sent))))
        self.assertEqual(entry['request_hash'], request_hash(provider.request_body(request_messages('p', big, SCHEMA), SCHEMA,
                                                                                    entry['num_predict'])))

    def test_an_answer_that_is_not_a_json_object_is_recorded_as_not_parsed(self):
        for content in ('not json', '[1, 2]', '"text"'):
            with self.subTest(content=content):
                answer, entry, _ = self.call(OllamaProvider('m', 'd', LOCAL, 30), {'passage': content}, [final(content)])
                self.assertEqual(answer, content)                                    # the caller still gets it, as before
                self.assertEqual((entry['status'], entry['parse_ok'], entry['parse_failure']), ('OK', False, 'MALFORMED_JSON'))
                self.assertEqual(entry['raw_output_sha256'], digest(content))
                self.assertNotIn('parsed_sha256', entry)
                self.assertNotIn('failure_code', entry)                              # an answer has no failure code
        # Key order and spacing do not change what parsed.
        _, first, _ = self.call(OllamaProvider('m', 'd', LOCAL, 30), {'n': 1}, [final('{"b": 2, "a": [1]}')])
        _, second, _ = self.call(OllamaProvider('m', 'd', LOCAL, 30), {'n': 2}, [final('{"a":[1],"b":2}')])
        self.assertNotEqual(first['raw_output_sha256'], second['raw_output_sha256'])
        self.assertEqual(first['parsed_sha256'], second['parsed_sha256'])

    def test_a_failed_call_names_its_failure_as_the_parse_outcome(self):
        # A cut answer: its partial text is hashed, and the failure code is the parse outcome.
        exc, entry = self.fails(OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192),
                                chunks=[final('{"conflict": tr', done_reason='length', eval_count=4096)])
        self.assertEqual((exc.code, entry['parse_ok'], entry['parse_failure']), ('OUTPUT_TRUNCATED', False, 'OUTPUT_TRUNCATED'))
        self.assertEqual(entry['raw_output_sha256'], digest('{"conflict": tr'))
        self.assertIsNotNone(entry['request_hash'])
        # A service that never answered: no answer hash, the retries counted, the request named.
        exc, entry = self.fails(OllamaProvider('m', 'd', LOCAL, 30), side_effect=httpx.ConnectError('refused'))
        self.assertEqual((exc.code, entry['retry_count'], entry['raw_output_sha256'], entry['parse_failure']),
                         ('RETRY_EXHAUSTED', 2, None, 'RETRY_EXHAUSTED'))
        self.assertIsNotNone(entry['request_hash'])
        # A prompt refused before sending: nothing was sent, so there is no request hash.
        exc, entry = self.fails(OllamaProvider('m', 'd', LOCAL, 30), {'passage': 'x' * 40000}, error=ContextBudgetError)
        self.assertEqual((entry['status'], entry['request_hash'], entry['parse_failure']), ('CONTEXT_BUDGET_EXCEEDED', None, 'CONTEXT_OVERFLOW'))

    def test_a_failed_call_never_takes_an_earlier_calls_answer(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        _, good, _ = self.call(provider, {'n': 1})
        _, entry = self.fails(provider, {'n': 2}, side_effect=httpx.ReadTimeout('slow'))
        self.assertEqual((entry['failure_code'], entry['raw_output_sha256'], entry['parse_failure']), ('TIMEOUT', None, 'TIMEOUT'))
        self.assertIsNotNone(good['raw_output_sha256'])

    def test_a_retried_request_is_the_same_request(self):
        busy = httpx.Response(503, request=httpx.Request('POST', LOCAL + '/api/chat'))
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch(CLIENT) as factory, patch('regchain.extraction.providers.time.sleep'):
            client = fake_client(factory, [final()])
            client.stream.return_value.__enter__.return_value.raise_for_status.side_effect = [
                httpx.HTTPStatusError('x', request=busy.request, response=busy), None]
            provider.generate_structured('p', {'n': 1}, SCHEMA)
        entry = provider.call_log[-1]
        bodies = [call.kwargs['json'] for call in client.stream.call_args_list]
        self.assertEqual((entry['status'], entry['retry_count'], len(bodies)), ('OK', 1, 2))
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(entry['request_hash'], digest(canonical(bodies[1])))

    def test_a_cache_hit_names_the_answer_it_served_and_sends_nothing(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch(CLIENT) as factory:
            client = fake_client(factory, [final()])
            provider.generate_structured('p', {'n': 1}, SCHEMA)
            provider.generate_structured('p', {'n': 1}, SCHEMA)
        live, hit = provider.call_log[-2:]
        self.assertEqual(client.stream.call_count, 1)
        self.assertEqual((hit['cache_hit'], hit['request_hash'], hit['parse_ok']), (True, None, True))
        self.assertEqual((hit['raw_output_sha256'], hit['parsed_sha256']), (live['raw_output_sha256'], live['parsed_sha256']))

    def test_records_stay_text_free_and_on_the_operators_allowlist(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192)
        answer = json.dumps({'quote': SENTINEL})
        self.call(provider, {'passage': SENTINEL}, [final(answer)])
        self.fails(provider, {'passage': SENTINEL + '2'}, chunks=[final(answer[:12], done_reason='length')])
        text = json.dumps(provider.call_log, ensure_ascii=False)
        self.assertNotIn(SENTINEL, text)
        for entry in provider.call_log:
            self.assertTrue(set(NEW_FIELDS) <= set(entry))
            self.assertLessEqual(set(entry), ALLOWED_CALL_FIELDS['minimal'])
            self.assertTrue(FORBIDDEN_CALL_FIELDS.isdisjoint(entry))
            self.assertEqual(redact_call(entry, RedactionPolicy('minimal')), entry)
            for level in ('standard', 'strict'):                 # a hash over the payload or the answer stays with the operator
                self.assertFalse({'request_hash', 'raw_output_sha256', 'parsed_sha256'} & set(redact_call(entry, RedactionPolicy(level))))

    def test_the_round_trip_numbers_the_packet_copies_are_unchanged(self):
        # extraction.pipeline copies provider.last_calls into the packet's diagnostics: no hash or answer text may enter it.
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        self.call(provider, {'passage': SENTINEL}, [final(json.dumps({'quote': SENTINEL}))])
        self.assertLessEqual(set(provider.last_calls[-1]), {'elapsed_ms', *MEASURED})
        self.assertNotIn(SENTINEL, json.dumps(provider.last_calls))


class SidecarTests(Isolated):
    def test_raw_answers_collects_every_answer_with_the_hashes_of_its_record(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30, True, 4096, 8192)
        cut = '{"conflict": tr'
        with raw_answers() as sink, ai_task('judge.verify'), ai_context(provision_id='p1', obligation_id='o1'):
            self.call(provider, {'n': 1}, [final('{"a": 1}')])
            self.fails(provider, {'n': 2}, chunks=[final(cut, done_reason='length')])
            self.call(provider, {'n': 1}, [final('{"never": "sent"}')])                  # a cache hit serves the first answer
            self.fails(provider, {'n': 3}, side_effect=httpx.ConnectError('refused'))    # no answer, no line
        self.assertIsNone(AI_RAW.get())
        self.assertEqual([line['raw'] for line in sink], ['{"a": 1}', cut, '{"a": 1}'])
        records = [r for r in provider.call_log if r['raw_output_sha256']]
        for line, record in zip(sink, records):
            with self.subTest(raw=line['raw']):
                self.assertEqual(line['raw_output_sha256'], digest(line['raw']))
                for key in ('request_hash', 'raw_output_sha256', 'task', 'provision_id', 'obligation_id', 'cache_hit', 'status', 'at'):
                    self.assertEqual(line[key], record[key])
        self.assertEqual([line['status'] for line in sink], ['OK', 'PROVIDER_FAILURE', 'OK'])
        self.assertEqual((sink[1]['failure_code'], sink[2]['cache_hit'], sink[2]['request_hash']), ('OUTPUT_TRUNCATED', True, None))

    def test_outside_raw_answers_no_answer_text_is_collected(self):
        self.assertIsNone(AI_RAW.get())
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with raw_answers() as sink:
            pass
        self.call(provider, {'n': 1})
        self.assertEqual(sink, [])

    def test_the_sidecar_is_never_read_back_as_a_cache(self):
        payload = {'passage': 'x'}
        probe = OllamaProvider('m', 'd', LOCAL, 30)
        with uncached():
            _, entry, _ = self.call(probe, payload)
        planted = {'request_hash': entry['request_hash'], 'raw_output_sha256': digest('PLANTED'), 'task': '', 'raw': 'PLANTED'}
        # 1. An open sink that already holds an answer for this very request: the call still goes to the model.
        with raw_answers() as sink:
            sink.append(dict(planted))
            answer, _, client = self.call(OllamaProvider('m', 'd', LOCAL, 30), payload, [final('{"live": 1}')])
        self.assertEqual((answer, client.stream.call_count), ('{"live": 1}', 1))
        # 2. An answer cache pointed at a run folder whose sidecar holds that request: never opened, never served.
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, RAW_LOG).write_text(json.dumps(planted) + '\n', encoding='utf-8')
            opened, real_open = [], builtins.open

            def spy(file, *args, **kwargs):
                opened.append(str(file))
                return real_open(file, *args, **kwargs)
            with patch.dict(os.environ, {'AI_CACHE_DIR': folder}), patch('builtins.open', spy):
                answer, _, client = self.call(OllamaProvider('m', 'd', LOCAL, 30), payload, [final('{"live": 2}')])
            self.assertEqual((answer, client.stream.call_count), ('{"live": 2}', 1))
            self.assertFalse([path for path in opened if path.endswith(RAW_LOG)])
            self.assertEqual(Path(folder, RAW_LOG).read_text(encoding='utf-8'), json.dumps(planted) + '\n')


class HarnessSidecarTests(Isolated):
    """harness.run writes ai-raw.jsonl beside ai-calls.jsonl. The rules components plus one live call per case on an Ollama
    provider behind the fake transport, made inside analyze, stand in for a model run."""

    def run_with_probe(self, root, answer, cache_dir=''):
        live = []

        class ProbeComponents(Components):
            def __init__(self, provider, retrieval, reranker):
                super().__init__('rules', retrieval, reranker)
                self.live = OllamaProvider('m', 'd', LOCAL, 30)
                live.append(self.live)

            def logs(self):
                return [*super().logs(), self.live.call_log]

        real_analyze = harness.analyze

        def analyze(*args, **kwargs):
            with ai_task('probe'), ai_context(provision_id='probe-provision'):
                live[-1].generate_structured('probe prompt', {'passage': 'probe passage'}, SCHEMA)
            return real_analyze(*args, **kwargs)

        with patch(CLIENT) as factory, patch.object(harness, 'Components', ProbeComponents), patch.object(harness, 'analyze', analyze), \
                patch.dict(os.environ, {'AI_CACHE_DIR': cache_dir}):
            client = fake_client(factory, [final(answer)])
            run_dir, _, _ = run(DATASET, root, provider='rules', retrieval='lexical', cases=['C08'], label='raw')
        return run_dir, client

    @staticmethod
    def lines(path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def test_a_run_writes_the_sidecar_and_nothing_else_carries_the_answer(self):
        answer = json.dumps({'quote': SENTINEL})
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, client = self.run_with_probe(Path(tmp), answer)
            raw = self.lines(run_dir / RAW_LOG)
            calls = [c for c in self.lines(run_dir / 'ai-calls.jsonl') if c['task'] == 'probe']
            self.assertEqual((len(raw), len(calls), client.stream.call_count), (1, 1, 1))
            self.assertEqual((raw[0]['case_id'], raw[0]['task'], raw[0]['provision_id'], raw[0]['raw']), ('C08', 'probe', 'probe-provision', answer))
            for key in ('request_hash', 'raw_output_sha256', 'at', 'status'):
                self.assertEqual(raw[0][key], calls[0][key])
            self.assertEqual(calls[0]['raw_output_sha256'], digest(answer))
            for name in ('ai-calls.jsonl', 'results.json', 'metrics.json', 'report.md', 'packets/C08.json', 'manifest.json'):
                with self.subTest(file=name):
                    self.assertNotIn(SENTINEL, (run_dir / name).read_text(encoding='utf-8'))
            self.assertNotIn(calls[0]['request_hash'], (run_dir / 'packets' / 'C08.json').read_text(encoding='utf-8'))

    def test_a_run_without_a_model_leaves_an_empty_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, _, _ = run(DATASET, Path(tmp), provider='rules', retrieval='lexical', cases=['C08'], label='rules')
            self.assertEqual((run_dir / RAW_LOG).read_text(encoding='utf-8'), '')

    def test_a_second_run_never_reads_the_first_runs_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, _ = self.run_with_probe(Path(tmp) / 'a', '{"run": 1}')
            # Worst case: the operator points the answer cache at the first run's folder. The second run still asks the model.
            second, client = self.run_with_probe(Path(tmp) / 'b', '{"run": 2}', cache_dir=str(first))
            self.assertEqual(client.stream.call_count, 1)
            raw = self.lines(second / RAW_LOG)
            self.assertEqual([line['raw'] for line in raw], ['{"run": 2}'])
            self.assertEqual(raw[0]['request_hash'], self.lines(first / RAW_LOG)[0]['request_hash'])   # the same request
            self.assertEqual(self.lines(first / RAW_LOG)[0]['raw'], '{"run": 1}')


if __name__ == '__main__':
    unittest.main()

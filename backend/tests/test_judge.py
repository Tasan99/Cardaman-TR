import base64
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from regchain.extraction.providers import (NUM_CTX, NUM_PREDICT, ContextBudgetError, OllamaProvider, ProviderFailure, RulesProvider,
                                           configured_judge)
from regchain.pilot.policies import handbook_extract
from regchain.pilot.workspace import Workspace, create_app
from test_workspace import request_input, source_fixture

LOCAL = 'http://localhost:11434'


class JudgeConfigurationTests(unittest.TestCase):
    def test_unset_keeps_one_model_and_rules_are_never_given_a_judge(self):
        extractor = OllamaProvider('small', 'd1', LOCAL)
        with patch.dict(os.environ, {'JUDGE_MODEL': ''}):
            self.assertIs(configured_judge(extractor), extractor)
        rules = RulesProvider()
        with patch.dict(os.environ, {'JUDGE_MODEL': 'big', 'JUDGE_MODEL_DIGEST': 'd2'}):
            self.assertIs(configured_judge(rules), rules)

    def test_a_pinned_local_judge_with_room_for_its_reasoning(self):
        extractor = OllamaProvider('small', 'd1', LOCAL, 300)
        env = {'JUDGE_MODEL': 'big', 'JUDGE_MODEL_DIGEST': 'd2', 'JUDGE_THINKING': 'on', 'JUDGE_TIMEOUT_SECONDS': '', 'JUDGE_NUM_PREDICT': ''}
        with patch.dict(os.environ, env):
            judge = configured_judge(extractor)
        self.assertEqual((judge.model_version, judge.thinking, judge.timeout, judge.base_url), ('big@d2', True, 300, LOCAL))
        # Reasoning comes before the answer and shares its limit.
        self.assertGreater(judge.num_predict, NUM_PREDICT)
        self.assertEqual(judge.runtime_manifest()['num_predict'], judge.num_predict)
        # Reasoning is spent on the contradiction question only; the rest goes to the same
        # pinned model with reasoning off, and the record says so.
        self.assertEqual((judge.quick.model_version, judge.quick.thinking, judge.quick.num_predict), ('big@d2', False, NUM_PREDICT))
        self.assertEqual(judge.runtime_manifest()['quick'], {'thinking': False, 'num_predict': NUM_PREDICT})
        with patch.dict(os.environ, dict(env, JUDGE_THINKING='off')):
            plain = configured_judge(extractor)
        self.assertEqual(plain.num_predict, NUM_PREDICT)
        self.assertIs(plain.quick, plain)
        self.assertNotIn('quick', plain.runtime_manifest())

    def test_misconfiguration_is_refused_not_guessed(self):
        extractor = OllamaProvider('small', 'd1', LOCAL)
        for bad in ({'JUDGE_THINKING': 'maybe'}, {'JUDGE_NUM_PREDICT': 'lots'}, {'JUDGE_NUM_PREDICT': str(NUM_CTX)},
                    {'JUDGE_MODEL_DIGEST': ''}, {'JUDGE_TIMEOUT_SECONDS': '5'}):
            env = {'JUDGE_MODEL': 'big', 'JUDGE_MODEL_DIGEST': 'd2', 'JUDGE_THINKING': '', 'JUDGE_TIMEOUT_SECONDS': '',
                   'JUDGE_NUM_PREDICT': '', **bad}
            with self.subTest(bad=bad), patch.dict(os.environ, env), self.assertRaises(ValueError):
                configured_judge(extractor)

    def test_more_room_for_the_answer_leaves_less_for_the_prompt(self):
        small = OllamaProvider('m', 'd', LOCAL)
        roomy = OllamaProvider('m', 'd', LOCAL, 120, True, NUM_CTX // 2)
        text = 'a' * 22000
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            # v0.16: the roomy judge may give up part of its answer budget for a long prompt, but
            # never more than half of it (30,000 bytes leave about 3,300 tokens, under 4,096).
            with self.assertRaises(ContextBudgetError):
                roomy._chat('p', {'text': 'a' * 30000}, {'type': 'object'})
            factory.assert_not_called()
            client = factory.return_value.__enter__.return_value
            client.get.return_value.json.return_value = {'models': [{'name': 'm', 'digest': 'd'}]}
            stream = client.stream.return_value.__enter__.return_value
            stream.iter_bytes.return_value = [b'{"done": true, "done_reason": "stop", "prompt_eval_count": 5500, "message": {"content": "{}"}}']
            self.assertEqual(small._chat('p', {'text': text}, {'type': 'object'}), '{}')
            self.assertEqual(client.stream.call_args.kwargs['json']['options']['num_predict'], NUM_PREDICT)


class StreamedAnswerTests(unittest.TestCase):
    def client(self, factory, chunks):
        client = factory.return_value.__enter__.return_value
        client.get.return_value.json.return_value = {'models': [{'name': 'm', 'digest': 'd'}]}
        stream = client.stream.return_value.__enter__.return_value
        stream.iter_bytes.return_value = chunks
        return client

    def test_a_streamed_answer_is_assembled_from_its_pieces(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30, True)
        pieces = [b'{"message": {"thinking": "hmm", "content": ""}, "done": false}\n{"message": {"content": "{\\"a\\": "}, "done": false}\n',
                  b'{"message": {"content": "1}"}, "done": true, "done_reason": "stop", "prompt_eval_count": 300, "eval_count": 9}']
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client = self.client(factory, pieces)
            self.assertEqual(provider._chat('p', {'text': 'x'}, {'type': 'object'}), '{"a": 1}')
            self.assertTrue(client.stream.call_args.kwargs['json']['stream'])
        self.assertEqual(provider.last_calls[-1]['output_tokens'], 9)

    def test_a_stream_that_outlives_the_time_limit_is_cut_off(self):
        # Seen live: a reasoning judge that spilled to CPU ran 52 minutes on one passage.
        provider = OllamaProvider('m', 'd', LOCAL, 10, True)

        def slow():
            yield b'{"message": {"content": ""}, "done": false}\n'
            with patch('regchain.extraction.providers.time.monotonic', return_value=time.monotonic()+11):
                yield b'{"message": {"content": ""}, "done": false}\n'
            yield b'{"message": {"content": "{}"}, "done": true, "done_reason": "stop", "prompt_eval_count": 300}'
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            self.client(factory, slow())
            with self.assertRaisesRegex(ProviderFailure, 'time limit'):
                provider._chat('p', {'text': 'x'}, {'type': 'object'})

    def test_a_stream_that_ends_without_completion_fails_closed(self):
        provider = OllamaProvider('m', 'd', LOCAL, 30)
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            self.client(factory, [b'{"message": {"content": "{"}, "done": false}\n'])
            with self.assertRaises(ProviderFailure):
                provider._chat('p', {'text': 'x'}, {'type': 'object'})


class RegulationUploadedAsPolicyTests(unittest.TestCase):
    EXPORT = ('COBS 4 Communicating with clients www.handbook.fca.org.uk September 2026\n\n' +
              '\n\n'.join(f'COBS 4.{s}.{n} R A firm must ensure a fixture requirement number {n}.' for s in range(2, 6) for n in range(1, 9)))
    POLICY = ('Financial promotions policy. We follow COBS 4.2.1 and COBS 4.3.1 of the FCA Handbook '
              '(www.handbook.fca.org.uk).\n\nEvery communication must be balanced and must not mislead.')

    def parsed(self, text):
        return {'filename': 'x.txt', 'chunks': [{'text': block} for block in text.split('\n\n')]}

    def test_a_handbook_export_is_recognised_and_a_policy_that_cites_rules_is_not(self):
        self.assertEqual(handbook_extract(self.parsed(self.EXPORT)), 'COBS 4')
        self.assertIsNone(handbook_extract(self.parsed(self.POLICY)))
        self.assertIsNone(handbook_extract(self.parsed(self.EXPORT.replace('www.handbook.fca.org.uk', 'our intranet'))))

    def test_the_run_stops_with_directions_unless_the_operator_insists(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        workspace = Workspace(Path(temp.name)/'workspace', provider_factory=lambda _: RulesProvider(), source_fetcher=source_fixture)
        client = TestClient(create_app(workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                            headers={'Authorization': 'Bearer test-only-token'})
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)

        def finish(**changes):
            upload = [dict(name='COBS 4 export.txt', content=base64.b64encode(self.EXPORT.encode()).decode())]
            run_id = client.post('/api/runs', json=request_input(policies=upload, **changes)).json()['id']
            for _ in range(300):
                row = workspace.metadata(run_id)
                if row['state'] in ('COMPLETED', 'FAILED') and not workspace.active:
                    return row
                time.sleep(.01)
            self.fail('Local fixture job did not finish')
        refused = finish()
        self.assertEqual(refused['state'], 'FAILED')
        for expected in ('COBS 4 export.txt', 'FCA Handbook', 'modül = COBS', 'bölüm = 4'):
            self.assertIn(expected, refused['error'])
        self.assertEqual(finish(allow_regulatory_text=True)['state'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main()

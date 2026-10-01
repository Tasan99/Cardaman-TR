import json
import os
import unittest
from unittest.mock import patch

from regchain.extraction.pipeline import extract
from regchain.extraction.providers import RulesProvider, OllamaProvider, ProviderFailure, configured_provider
from regchain.extraction.schema import ExtractionOutput

TEXT = 'A firm must keep records unless the account is exempt.'


class ScriptedProvider:
    name = 'test'
    model_version = 'fixture-v1'

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def generate(self, text):
        value = self.responses[min(self.calls,len(self.responses)-1)]
        self.calls += 1
        if isinstance(value,Exception):
            raise value
        return value

    def repair(self,text,context,feedback):
        return self.generate(text)


class ExtractionTests(unittest.TestCase):
    def changed(self, **fields):
        data = json.loads(RulesProvider().generate(TEXT))
        data['obligations'][0].update(fields)
        return json.dumps(data)

    def test_supported_candidate_is_still_review_only(self):
        result = extract(TEXT,RulesProvider())
        self.assertEqual(result.output.status,'EXTRACTED')
        self.assertEqual(result.output.obligations[0].exceptions,['unless the account is exempt.'])
        self.assertTrue(result.human_review_required)

    def test_all_modalities_and_polarities(self):
        for token,mode in [('must','MUST'),('shall','MUST'),('must not','MUST_NOT'),
                           ('shall not','MUST_NOT'),('should','SHOULD'),('should not','SHOULD_NOT'),
                           ('may','MAY'),('may not','MAY_NOT')]:
            with self.subTest(token=token):
                result = extract('A firm ' + token + ' retain records.',RulesProvider())
                self.assertEqual(result.output.obligations[0].modality,mode)

    def test_missing_exception_is_rejected(self):
        provider = ScriptedProvider([self.changed(exceptions=[])])
        result = extract(TEXT,provider)
        self.assertEqual(result.output.status,'INSUFFICIENT_EVIDENCE')
        self.assertEqual(provider.calls,2)

    def test_missing_nested_condition_is_rejected(self):
        text = 'A firm must act unless exempt where the exception applies.'
        data = json.loads(RulesProvider().generate(text))
        data['obligations'][0]['conditions'] = []
        self.assertEqual(extract(text,ScriptedProvider([json.dumps(data)])).output.status,'INSUFFICIENT_EVIDENCE')

    def test_fabricated_action_rejected(self):
        self.assertEqual(extract(TEXT,ScriptedProvider([self.changed(required_action='pay a fine')])).output.status,'INSUFFICIENT_EVIDENCE')

    def test_short_quote_rejected(self):
        self.assertEqual(extract(TEXT,ScriptedProvider([self.changed(source_quote='A firm must keep records')])).output.status,'INSUFFICIENT_EVIDENCE')

    def test_extra_fields_and_invalid_confidence_rejected(self):
        for update in [{'official_legal_advice':True},{'confidence_score':'1.5'}]:
            self.assertEqual(extract(TEXT,ScriptedProvider([self.changed(**update)])).output.status,'INSUFFICIENT_EVIDENCE')

    def test_retry_can_recover(self):
        provider = ScriptedProvider(['not JSON',RulesProvider().generate(TEXT)])
        self.assertEqual(extract(TEXT,provider).output.status,'EXTRACTED')
        self.assertEqual(provider.calls,2)

    def test_outage_never_falls_back_to_rules(self):
        result = extract(TEXT,ScriptedProvider([ProviderFailure('offline')]))
        self.assertEqual(result.output.obligations,[])
        self.assertEqual(result.reason,'PROVIDER_FAILURE')

    def test_cross_reference_is_not_resolved_from_general_knowledge(self):
        provider = ScriptedProvider([])
        result = extract('A firm must act in accordance with CONC 7.3.',provider)
        self.assertEqual(result.reason,'UNRESOLVED_CROSS_REFERENCE')
        self.assertEqual(provider.calls,0)

    def test_large_source_is_not_silently_truncated(self):
        self.assertEqual(extract('x'*12001,RulesProvider()).reason,'SOURCE_SIZE_OR_EMPTY')

    def test_multiple_modals_not_silently_dropped_by_baseline(self):
        result = extract('A firm must record consent and must keep evidence.',RulesProvider())
        self.assertEqual(result.output.status,'EXTRACTED')
        self.assertEqual([o.required_action for o in result.output.obligations],['record consent','keep evidence.'])

    def test_no_explicit_modality_is_not_legal_clearance(self):
        result = extract('This document describes the consultation.',RulesProvider())
        self.assertEqual(result.output.status,'NO_EXPLICIT_OBLIGATION')
        self.assertTrue(result.human_review_required)

    def test_false_negative_provider_answer_is_rejected(self):
        response = '{"status":"NO_EXPLICIT_OBLIGATION","obligations":[]}'
        self.assertEqual(extract(TEXT,ScriptedProvider([response])).output.status,'INSUFFICIENT_EVIDENCE')

    def test_no_silent_default_provider(self):
        with self.assertRaises(ValueError):
            configured_provider('')

    def test_ollama_requires_pinned_model_and_safe_origin(self):
        for model,digest,url in [('','abc','http://localhost:11434'),('m','','http://localhost:11434'),
                                 ('m','abc','http://untrusted.example')]:
            with self.assertRaises(ValueError):
                OllamaProvider(model,digest,url)

    def test_ollama_adapter_sends_schema_and_validates_completion(self):
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client = factory.return_value.__enter__.return_value
            client.get.return_value.json.return_value = {'models':[{'name':'fixture:latest','digest':'sha256:abc'}]}
            response = client.stream.return_value.__enter__.return_value
            response.iter_bytes.side_effect = [[json.dumps({'done':True,'done_reason':'stop',
                'message':{'content':payload}}).encode()] for payload in
                (json.dumps({'status':'EXTRACTED','obligations':[{'subject':'A firm','modality':'MUST',
                    'action':'keep records unless the account is exempt.','conditions':[],
                    'exceptions':['unless the account is exempt.'],'evidence':[]}]}),json.dumps({'decisions':['SUPPORTED']}))]
            provider = OllamaProvider('fixture:latest','sha256:abc','http://localhost:11434')
            self.assertEqual(extract(TEXT,provider).output.status,'EXTRACTED')
            self.assertIn('properties',client.stream.call_args.kwargs['json']['format'])

    def test_reasoning_mode_is_explicit_and_only_sent_when_chosen(self):
        # Unset must not send the field at all: the model keeps its own default.
        self.assertIsNone(OllamaProvider('m','d','http://localhost:11434').thinking)
        with patch.dict(os.environ,{'LLM_MODEL':'m','LLM_MODEL_DIGEST':'d','LLM_THINKING':'off'}):
            self.assertIs(configured_provider('ollama').thinking, False)
        with patch.dict(os.environ,{'LLM_MODEL':'m','LLM_MODEL_DIGEST':'d','LLM_THINKING':'on'}):
            self.assertIs(configured_provider('ollama').thinking, True)
        with patch.dict(os.environ,{'LLM_MODEL':'m','LLM_MODEL_DIGEST':'d','LLM_THINKING':'maybe'}):
            with self.assertRaises(ValueError):
                configured_provider('ollama')

    def test_reasoning_choice_reaches_the_request(self):
        for thinking, expected in [(None, None), (False, False), (True, True)]:
            with self.subTest(thinking=thinking):
                provider = OllamaProvider('m','d','http://localhost:11434',120,thinking)
                with patch('httpx.Client') as client:
                    instance = client.return_value.__enter__.return_value
                    instance.get.return_value.json.return_value = {'models':[{'name':'m','digest':'d'}]}
                    stream = instance.stream.return_value.__enter__.return_value
                    stream.iter_bytes.return_value = [json.dumps({'done':True,'done_reason':'stop',
                        'prompt_eval_count':30,'message':{'content':'{}'}}).encode()]
                    provider._chat('fixture',{'text':'A firm must act.'},{'type':'object'})
                    sent = instance.stream.call_args.kwargs['json']
                self.assertEqual(sent.get('think', None), expected)

    def test_context_window_matches_the_retrieval_budget(self):
        from regchain.extraction.providers import NUM_CTX, NUM_PREDICT
        # Worst-case packet is 32000 characters plus prompts, about 12000 tokens.
        # The window must hold that and still leave room for the whole answer.
        self.assertGreaterEqual(NUM_CTX - NUM_PREDICT, 12000)

    def test_a_prompt_that_filled_the_window_fails_closed(self):
        from regchain.extraction.providers import NUM_CTX
        provider = OllamaProvider('m','d','http://localhost:11434')
        body = {'done': True, 'done_reason': 'stop', 'prompt_eval_count': NUM_CTX,
                'message': {'content': '{}'}}
        with patch('httpx.Client') as client:
            instance = client.return_value.__enter__.return_value
            instance.get.return_value.json.return_value = {'models':[{'name':'m','digest':'d'}]}
            stream = instance.stream.return_value.__enter__.return_value
            stream.iter_bytes.return_value = [json.dumps(body).encode()]
            with self.assertRaises(ProviderFailure) as error:
                provider.generate('A firm must act.')
        self.assertIn('truncated', str(error.exception))

    def test_call_timeout_is_configurable_but_bounded(self):
        # A local model that spills to CPU needs longer than the default; an
        # unbounded wait would hide a hung provider, so the range stays explicit.
        self.assertEqual(OllamaProvider('m','d','http://localhost:11434').timeout, 120)
        self.assertEqual(OllamaProvider('m','d','http://localhost:11434',900).timeout, 900)
        for bad in (0, 9, 3601):
            with self.subTest(timeout=bad), self.assertRaises(ValueError):
                OllamaProvider('m','d','http://localhost:11434',bad)
        with patch.dict(os.environ,{'LLM_MODEL':'m','LLM_MODEL_DIGEST':'d','LLM_TIMEOUT_SECONDS':'600'}):
            self.assertEqual(configured_provider('ollama').timeout, 600)
        with patch.dict(os.environ,{'LLM_MODEL':'m','LLM_MODEL_DIGEST':'d','LLM_TIMEOUT_SECONDS':'abc'}):
            with self.assertRaises(ValueError):
                configured_provider('ollama')

    def test_model_digest_mismatch_fails_closed(self):
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client = factory.return_value.__enter__.return_value
            client.get.return_value.json.return_value = {'models':[]}
            with self.assertRaises(ProviderFailure):
                OllamaProvider('fixture:latest','sha256:abc','http://localhost:11434').generate(TEXT)
            client.stream.assert_not_called()

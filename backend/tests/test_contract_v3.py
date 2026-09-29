import json
import unittest
from dataclasses import replace
from regchain.extraction.contract import materialize,EXAMPLE
from regchain.extraction.providers import RulesProvider,OllamaProvider,runtime_manifest
from regchain.extraction.schema import ExtractionOutput,Candidate
from regchain.extraction.grounding import verify
from regchain.extraction.pipeline import extract
from regchain.extraction.retrieval import retrieve
from test_quality import section
from test_consolidated import corpus_section

class ContractTests(unittest.TestCase):
    def test_leading_condition_does_not_include_the_duty(self):
        text='When assessing risk, a firm must review records.'
        value=Candidate(source_quote=text,subject='a firm',modality='MUST',
            required_action='review records.',conditions=['When assessing risk'],confidence_score='0.5')
        verify(text,ExtractionOutput(status='EXTRACTED',obligations=[value]))
        value.conditions=[]
        with self.assertRaisesRegex(ValueError,'When assessing risk'):
            verify(text,ExtractionOutput(status='EXTRACTED',obligations=[value]))

    def test_oversized_prompt_is_rejected_before_http(self):
        from unittest.mock import patch
        from regchain.extraction.providers import ContextBudgetError
        provider=OllamaProvider('m','d','http://localhost:11434')
        with patch('regchain.extraction.providers.httpx.Client') as client:
            with self.assertRaises(ContextBudgetError):
                provider.generate('a'*40000)
            client.assert_not_called()

    def test_admission_budget_follows_measured_tokens_not_one_token_per_byte(self):
        # One token per byte rejected every real policy PDF block before the model saw it.
        from unittest.mock import patch
        from regchain.extraction.providers import BYTES_PER_TOKEN, NUM_CTX, NUM_PREDICT, ContextBudgetError
        self.assertLess(BYTES_PER_TOKEN, 3.3)  # lowest measured ratio (Turkish); English was 4.9
        provider=OllamaProvider('m','d','http://localhost:11434')
        room=int((NUM_CTX-NUM_PREDICT-1024)*BYTES_PER_TOKEN)
        with patch('regchain.extraction.providers.httpx.Client') as factory:
            client=factory.return_value.__enter__.return_value
            client.get.return_value.json.return_value={'models':[{'name':'m','digest':'d'}]}
            stream=client.stream.return_value.__enter__.return_value
            stream.iter_bytes.return_value=[json.dumps({'done':True,'done_reason':'stop','prompt_eval_count':5000,
                'message':{'content':'{}'}}).encode()]
            self.assertEqual(provider._chat('p',{'text':'a'*(room-400)},{'type':'object'}),'{}')
            self.assertEqual(provider.call_log[-1]['num_predict'], NUM_PREDICT)
            # v0.16: a prompt past that line is admitted with a smaller generation budget, and the
            # record says so; only a prompt that leaves less than half the budget is refused.
            self.assertEqual(provider._chat('p',{'text':'a'*room},{'type':'object'}),'{}')
            self.assertTrue(provider.call_log[-1]['num_predict_reduced'])
            self.assertLess(provider.call_log[-1]['num_predict'], NUM_PREDICT)
            floor=int((NUM_CTX-NUM_PREDICT//2-1024)*BYTES_PER_TOKEN)
            with self.assertRaises(ContextBudgetError):
                provider._chat('p',{'text':'a'*(floor+400)},{'type':'object'})
            self.assertEqual(provider.call_log[-1]['status'], 'CONTEXT_BUDGET_EXCEEDED')

    def test_a_prompt_cut_by_the_runtime_is_detected_by_its_reported_size(self):
        # Ollama reports a cut prompt as half the window, never as a full one.
        from unittest.mock import patch
        from regchain.extraction.providers import NUM_CTX, ProviderFailure
        provider=OllamaProvider('m','d','http://localhost:11434')
        for reported,text,cut in [(NUM_CTX//2+2,'a'*20000,True),   # the observed signature
                                  (500,'a'*20000,True),            # 40 bytes per token is not text
                                  (5000,'a'*20000,False)]:         # 4 bytes per token: uncut English
            with self.subTest(reported=reported), patch('regchain.extraction.providers.httpx.Client') as factory:
                client=factory.return_value.__enter__.return_value
                client.get.return_value.json.return_value={'models':[{'name':'m','digest':'d'}]}
                stream=client.stream.return_value.__enter__.return_value
                stream.iter_bytes.return_value=[json.dumps({'done':True,'done_reason':'stop',
                    'prompt_eval_count':reported,'message':{'content':'{}'}}).encode()]
                if cut:
                    with self.assertRaisesRegex(ProviderFailure,'cut'):
                        provider._chat('p',{'text':text},{'type':'object'})
                else:
                    self.assertEqual(provider._chat('p',{'text':text},{'type':'object'}),'{}')

    def test_backend_owns_quote_and_polarity(self):
        value=materialize(json.dumps(EXAMPLE),'A firm must retain records.')
        self.assertEqual(value.obligations[0].source_quote,'A firm must retain records.')
        self.assertEqual(value.obligations[0].required_action,'retain records.')
        self.assertIsNone(value.obligations[0].prohibited_action)

    def test_unknown_evidence_id_cannot_be_materialized(self):
        data=json.loads(json.dumps(EXAMPLE))
        data['obligations'][0]['evidence']=[{'source_id':'999','role':'scope','quote':'made up'}]
        with self.assertRaisesRegex(ValueError,'SUPPORT_SPAN_INVALID'):
            materialize(json.dumps(data),'A firm must retain records.')

    def test_embedded_modal_does_not_force_two_duties(self):
        text='A firm must consider circumstances of which it is or should be aware.'
        value=Candidate(source_quote=text,subject='A firm',modality='MUST',
            required_action='consider circumstances of which it is or should be aware.',confidence_score='0.5')
        verify(text,ExtractionOutput(status='EXTRACTED',obligations=[value]))

    def test_one_modal_can_govern_two_list_actions(self):
        text='A firm must: (1) retain records; (2) review controls.'
        values=[Candidate(source_quote=text,subject='A firm',modality='MUST',required_action=action,
            confidence_score='0.5') for action in ['retain records','review controls.']]
        verify(text,ExtractionOutput(status='EXTRACTED',obligations=values))

    def test_independent_modal_cannot_silently_disappear(self):
        text='A firm must retain records. A firm must review controls.'
        value=Candidate(source_quote=text,subject='A firm',modality='MUST',required_action='retain records.',confidence_score='0.5')
        with self.assertRaisesRegex(ValueError,'UNCOVERED_MODAL'):
            verify(text,ExtractionOutput(status='EXTRACTED',obligations=[value]))

    def test_optional_neighbor_does_not_become_a_condition(self):
        target=section('a','A firm must retain records.','7.1')
        neighbor=section('b','A firm should notify customers if fees change.','7.2')
        packet=retrieve(target,[target,neighbor])
        self.assertFalse(packet.items[0]['required'])
        self.assertEqual(extract(target['text'],RulesProvider(),packet).output.status,'EXTRACTED')

    def test_incomplete_required_chapter_is_not_called_complete(self):
        target=section('a','A firm must act under CONC 5D.3.','CONC 7.3.4')
        refs=[corpus_section(str(i),'Text of provision.','CONC 5D.3.'+str(i),'v2') for i in range(1,4)]
        packet=retrieve(target,[target],corpus=refs,max_items=1)
        self.assertTrue(packet.unresolved)

    def test_fully_inlined_direct_chapter_is_required_context(self):
        target=section('a','A firm must act under CONC 5D.3.','CONC 7.3.4')
        refs=[corpus_section(str(i),'Text of provision.','CONC 5D.3.'+str(i),'v2') for i in range(1,3)]
        packet=retrieve(target,[target],corpus=refs)
        self.assertFalse(packet.unresolved)
        self.assertTrue(all(item['required'] for item in packet.items))
        self.assertEqual(extract(target['text'],RulesProvider(),packet).reason,'CONTEXT_REQUIRES_LLM')

    def test_failure_has_bounded_diagnostic_and_only_one_repair(self):
        class Broken:
            name='fixture'; model_version='v1'
            def generate(self,text): return 'invalid-json'
            def repair(self,text,context,feedback):
                self.feedback=feedback
                return 'still invalid'
        provider=Broken()
        result=extract('A firm must act.',provider)
        self.assertEqual(result.attempts,2)
        self.assertEqual(len(result.diagnostics),2)
        self.assertEqual(result.diagnostics[0]['code'],'SCHEMA_INVALID')
        self.assertTrue(provider.feedback)

    def test_runtime_records_thinking_and_context_budget(self):
        first=runtime_manifest(OllamaProvider('m','d','http://localhost:11434',thinking=False))
        second=runtime_manifest(OllamaProvider('m','d','http://localhost:11434',thinking=True))
        self.assertNotEqual(first,second)
        self.assertIn('num_ctx',first)

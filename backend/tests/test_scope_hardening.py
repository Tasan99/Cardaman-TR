"""P3–P5: no live inference; source loss and inheritance are fault-injected."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from regchain.pilot.applicability import rule_chain
from regchain.extraction.providers import ContextBudgetError, ProviderFailure
from regchain.pilot.engine import judge_scope, model_answer, scope_compressor, propose, pipeline_settings
from regchain.evidence import canonical_bytes, make_event, verify_chain
from regchain.pilot.decision_trace import attempt_record, finish_trace
from regchain.extraction.quantities import evidence_safe
from regchain.pilot.entities import entity_gate
from regchain.pilot.schema import Company, ResolvedBasis
from regchain.pilot.scope_integrity import inheritance_evidence, local_assessment_evidence, local_scope


SECTION = {'id': 'provision', 'printed_label': 'Test rule 9', 'text': 'Banks must retain records.'}
SCOPES = {'s1': {'id': 'purpose', 'printed_label': 'Purpose', 'text': 'This rule governs record keeping.'},
          's2': {'id': 'actors', 'printed_label': 'Scope', 'text': 'This rule applies to banks.'}}
DUTY = {'subject': 'Banks', 'modality': 'MUST', 'required_action': 'retain records',
        'prohibited_action': None, 'conditions': [], 'exceptions': []}


def company(**changes):
    return Company(**{**dict(id='scope-test', name='Test company', version='1', synthetic=True, jurisdictions=['UK'],
                             activities=['banking'], licences=['bank licence'], products=['accounts'],
                             customer_types=['individuals'], description=''), **changes})


def answer(*, quote='Banks must retain records.', source='p0', condition='This rule applies to banks.', state='APPLIES'):
    return {'applicability': state, 'company_fact_keys': ['activities'],
            'scope_evidence': [{'source_id': source, 'quote': quote}],
            'basis': [{'company_fact': 'banking', 'regulatory_condition': condition, 'match': 'YES'}],
            'applicability_reason': 'The supplied actor and company activity match.', 'missing_information': []}


class Scripted:
    base_url = 'http://localhost:11434'

    def __init__(self, *answers):
        self.answers, self.sent = list(answers), []

    def _chat(self, prompt, payload, schema):
        self.sent.append(deepcopy(payload))
        return json.dumps(self.answers.pop(0))


def fake_chain(*, clear=False, obliged=False, actor=None, subject=None):
    gates = {'REGULATION_SUBJECT_SCOPE': subject or {'status': 'UNDETERMINED', 'clear': False, 'evidence': {}},
             'COMPANY_ENTITY': actor or {'status': 'UNDETERMINED', 'clear': False, 'evidence': {}}}
    return SimpleNamespace(by_name=gates.get, clear_match=clear, obliged_addressee=obliged)


class ScopeCompletenessTests(unittest.TestCase):
    def test_missing_profile_precedes_a_negative_actor_gate_and_never_calls_model(self):
        profile = company(licences=None)
        section = {**SECTION, 'text': 'The authority must retain records.'}
        duty = {**DUTY, 'subject': 'The authority'}
        gate = entity_gate(duty, section['text'], '', profile, section['printed_label'], 0)
        chain = rule_chain(profile, section, duty, section['text'], gate, None, False)
        self.assertEqual((chain.decision, chain.decided_by), ('UNKNOWN', 'PROFILE_INCOMPLETE'))
        provider = Scripted()
        value, basis, rule, notes, failure = judge_scope(provider, profile, section, duty, SCOPES)
        self.assertEqual((value.applicability, rule, provider.sent), ('UNKNOWN', 'PROFILE_INCOMPLETE', []))
        self.assertIsNone(failure)
        self.assertIn('Missing company field: licences', value.missing_information)
        self.assertEqual(notes.decision_trace['RAW_MODEL_DECISION']['status'], 'NOT_CALLED')

    def test_missing_source_does_not_become_model_guessed_applicability(self):
        provider = Scripted()
        value, basis, rule, notes, failure = judge_scope(provider, company(), {**SECTION, 'text': ''}, DUTY, SCOPES)
        self.assertEqual((value.applicability, rule, provider.sent), ('UNKNOWN', 'SOURCE_INCOMPLETE', []))
        self.assertIsNone(failure)

    def test_empty_licence_list_is_an_explicit_value_not_a_missing_field(self):
        provider = Scripted(answer())
        value, _, _, _, failure = judge_scope(provider, company(licences=[]), SECTION, DUTY, SCOPES)
        self.assertIsNone(failure)
        self.assertEqual((value.applicability, len(provider.sent)), ('APPLIES', 1))


class SourceRepairTests(unittest.TestCase):
    def test_removed_candidate_source_cannot_validate_quote_or_basis(self):
        provider = Scripted(answer(quote=SCOPES['s2']['text'], source='s2'), answer(quote=SCOPES['s2']['text'], source='s2'))

        def omit_scope(instance, prompt, payload, schema, compress):
            return instance, {**payload, 'scope': [payload['scope'][0]]}, {}

        with patch('regchain.pilot.engine.fit_call', side_effect=omit_scope):
            value, basis, rule, notes, failure = judge_scope(provider, company(), SECTION, DUTY, SCOPES, fitted=True)
        self.assertIsNone(value)
        self.assertFalse(basis)
        self.assertIn('PROPOSAL_INVALID', failure)
        self.assertIn('BASIS_INVALID', [n['code'] for n in notes])
        trace = notes.decision_trace
        self.assertEqual(trace['RAW_MODEL_DECISION']['decision'], 'APPLIES')
        self.assertIsNone(trace['VALIDATED_MODEL_DECISION']['decision'])
        self.assertEqual(trace['FINAL_PIPELINE_DECISION']['decision'], 'UNKNOWN')
        lineage = trace['source_lineage']
        self.assertEqual({r['source_id'] for r in lineage['candidate_source_set']}, {'s1', 's2', 'p0'})
        self.assertEqual({r['source_id'] for r in lineage['original_source_set']}, {'s1', 'p0'})
        self.assertEqual(lineage['final_validated_source_set'], [])

    def test_repair_cannot_silently_drop_previously_verified_source(self):
        provider = Scripted(answer(quote='An invented quote which the source does not contain.'))
        calls = []

        def lose_on_repair(instance, prompt, payload, schema, compress):
            calls.append(deepcopy(payload))
            return instance, ({**payload, 'scope': [payload['scope'][0]]} if len(calls) > 1 else payload), {}

        with patch('regchain.pilot.engine.fit_call', side_effect=lose_on_repair), \
                patch('regchain.pilot.engine.trimmed_retry', return_value=(None, [])):
            value, _, _, notes, failure = judge_scope(provider, company(), SECTION, DUTY, SCOPES, fitted=True)
        self.assertIsNone(value)
        self.assertEqual(len(provider.sent), 1)
        self.assertIn('ContextBudgetError', failure)
        self.assertTrue(any(n['code'] == 'BASIS_INSUFFICIENT' and n.get('source_ids') == ['s2'] for n in notes))
        self.assertIn('s2', notes.decision_trace['source_lineage']['original_verified_source_ids'])

    def test_repair_may_add_source_only_from_candidate_set(self):
        provider = Scripted(answer(quote='Invented quote', condition='No such source phrase.'),
                            answer(quote=SCOPES['s2']['text'], source='s2'))
        calls = []

        def restore_candidate(instance, prompt, payload, schema, compress):
            calls.append(payload)
            return instance, ({**payload, 'scope': payload['scope'][:1]} if len(calls) == 1 else payload), {}

        with patch('regchain.pilot.engine.fit_call', side_effect=restore_candidate):
            value, basis, _, notes, failure = judge_scope(provider, company(), SECTION, DUTY, SCOPES, fitted=True)
        self.assertIsNone(failure)
        self.assertEqual(value.applicability, 'APPLIES')
        self.assertEqual({b.source_id for b in basis}, {'s2'})
        self.assertTrue(any(n['code'] == 'BASIS_ADDED' and 's2' in n.get('source_ids', []) for n in notes))
        self.assertIn('BASIS_REPAIRED', [n['code'] for n in notes])
        self.assertEqual(notes.decision_trace['source_lineage']['final_validated_source_set'], ['s2'])

    def test_source_id_invention_is_rejected_even_if_quote_exists_elsewhere(self):
        provider = Scripted(answer(source='fabricated'), answer(source='fabricated'))
        value, _, _, notes, failure = judge_scope(provider, company(), SECTION, DUTY, SCOPES)
        self.assertIsNone(value)
        self.assertTrue(any('QUOTE_UNKNOWN_SOURCE' in n.get('reason_codes', []) for n in notes))
        self.assertEqual(len(provider.sent), 2)

    def test_repair_request_cannot_introduce_a_non_candidate_source(self):
        provider = Scripted()

        def invented_source(instance, prompt, payload, schema, compress):
            return instance, {**payload, 'scope': [*payload['scope'],
                                                  {'source_id': 'new', 'text': 'An outside source.', 'label': 'Invented'}]}, {}

        with patch('regchain.pilot.engine.fit_call', side_effect=invented_source):
            value, _, _, notes, failure = judge_scope(provider, company(), SECTION, DUTY, SCOPES, fitted=True)
        self.assertIsNone(value)
        self.assertEqual(provider.sent, [])
        self.assertTrue(any(n['code'] == 'BASIS_INVALID' and n.get('source_ids') == ['new'] for n in notes))

    def test_compressor_preserves_verified_scope_before_unverified_scope(self):
        payload = {'provision': {'source_id': 'p0', 'text': 'Duty'},
                   'scope': [{'source_id': 's1', 'text': 'x' * 2000}, {'source_id': 's2', 'text': 'Verified actor'}]}
        smaller = scope_compressor(payload, 200, protected={'s2'})
        self.assertEqual([r['source_id'] for r in smaller['scope']], ['s2'])
        self.assertEqual([r['source_id'] for r in payload['scope']], ['s1', 's2'])


class InheritanceTests(unittest.TestCase):
    def test_parent_generic_answer_does_not_authorize_a_different_named_actor(self):
        origin = local_scope({'subject': 'Companies'}, {'text': 'Companies must supply information.'})
        current = local_scope({'subject': 'Trustees'}, {'offset': 47, 'text': 'Trustees must retain records.'})
        decision = inheritance_evidence(origin, current, fake_chain(), [], None)
        self.assertFalse(decision['allowed'])
        self.assertEqual(decision['reason_code'], 'SCOPE_INCOMPLETE')
        self.assertEqual(decision['current_local_scope']['actor_quote'], 'Trustees')
        self.assertEqual(decision['current_local_scope']['actor_span'], [47, 55])

    def test_same_explicit_actor_with_same_scope_can_share(self):
        origin = local_scope({'subject': 'Banks'}, {'text': 'Banks must supply information.'})
        current = local_scope({'subject': 'Banks'}, {'text': 'Banks must retain records.'})
        self.assertTrue(inheritance_evidence(origin, current, fake_chain(), [], None)['allowed'])

    def test_changed_exception_is_not_resolved_by_a_generic_actor_basis(self):
        origin = local_scope({'subject': 'Banks'}, {'text': 'Banks must supply information.'})
        current = local_scope({'subject': 'Banks', 'exceptions': ['unless the branch is abroad']},
                              {'text': 'Banks must retain records unless the branch is abroad.'})
        basis = [ResolvedBasis(company_fact='banking', regulatory_condition='Banks', source_id='p0', match='YES')]
        decision = inheritance_evidence(origin, current, fake_chain(clear=True), basis, None)
        self.assertFalse(decision['allowed'])
        self.assertTrue(decision['exception_changed'])

    def test_explicit_licence_fact_is_recorded_on_an_obliged_local_actor(self):
        origin = local_scope({'subject': 'Companies'}, {'text': 'Companies must supply information.'})
        current = local_scope({'subject': 'Yükümlüler'}, {'text': 'Yükümlüler kayıt tutar.'})
        decision = inheritance_evidence(origin, current, fake_chain(obliged=True), [], None, 'licences: BDDK izni')
        self.assertTrue(decision['allowed'])
        self.assertEqual(decision['support'][0]['company_fact'], 'licences: BDDK izni')


class TraceCallIdentityTests(unittest.TestCase):
    class Logged(Scripted):
        num_predict = 99

        def __init__(self, task, old=0, capped=False):
            super().__init__(answer())
            self.task, self.capped = task, capped
            self.call_log = [{'task': task, 'request_hash': 'stale-request'} for _ in range(old)]

        def _chat(self, prompt, payload, schema):
            raw = super()._chat(prompt, payload, schema)
            self.call_log.append({'task': self.task, 'request_hash': 'fresh-request',
                                  'num_predict': 99, 'done_reason': 'stop'})
            if self.capped:
                self.call_log.pop(0)
            return raw

    def invoke(self, kind, provider):
        if kind == 'scope':
            _, _, _, notes, _ = judge_scope(provider, company(), SECTION, DUTY, SCOPES, fitted=True)
            return notes.decision_trace['attempts']
        trace = []
        model_answer(provider, 'test', {}, {}, json.loads, 'judge.verify', 'verify', trace=trace)
        return trace

    def test_selected_instance_with_shorter_independent_log_keeps_its_request_identity(self):
        for kind, task in [('scope', 'judge.applicability'), ('model', 'judge.verify')]:
            with self.subTest(kind=kind):
                provider, target = self.Logged(task, old=3), self.Logged(task)
                with patch('regchain.pilot.engine.fit_call', side_effect=lambda p, prompt, body, schema, compress:
                           (target, body, {'action': 'fits'})):
                    trace = self.invoke(kind, provider)
                self.assertEqual(trace[-1]['request_hash'], 'fresh-request')
                self.assertEqual(trace[-1]['status'], 'OK')
                self.assertEqual(trace[-1]['budget']['actual_output_cap'], 99)

    def test_fixed_length_call_log_still_identifies_a_new_request(self):
        for kind, task in [('scope', 'judge.applicability'), ('model', 'judge.verify')]:
            with self.subTest(kind=kind):
                provider = self.Logged(task, old=2, capped=True)
                with patch('regchain.pilot.engine.fit_call', side_effect=lambda p, prompt, body, schema, compress:
                           (p, body, {'action': 'fits'})):
                    trace = self.invoke(kind, provider)
                self.assertEqual(len(provider.call_log), 2)
                self.assertEqual(trace[-1]['request_hash'], 'fresh-request')
                self.assertEqual(trace[-1]['budget']['admission_decision'], 'ADMITTED')

    def test_admission_failure_never_borrows_a_previous_request_from_the_log(self):
        for kind, task in [('scope', 'judge.applicability'), ('model', 'judge.verify')]:
            with self.subTest(kind=kind):
                provider = self.Logged(task, old=2)
                with patch('regchain.pilot.engine.fit_call', side_effect=ContextBudgetError('test admission refusal')), \
                        patch('regchain.pilot.engine.trimmed_retry', return_value=(None, [])):
                    trace = self.invoke(kind, provider)
                self.assertEqual(provider.sent, [])
                self.assertIsNone(trace[-1]['request_hash'])
                self.assertEqual(trace[-1]['status'], 'CONTEXT_ADMISSION')
                self.assertEqual(trace[-1]['budget']['admission_decision'], 'REFUSED')


class TracePacketTests(unittest.TestCase):
    def test_rejected_float_answer_then_valid_repair_preserves_exact_raw_and_writes_evidence(self):
        original = {**answer(), 'confidence': 0.123456789}
        provider = Scripted(original, answer())
        proposal, _ = propose(provider, company(), SECTION, DUTY, list(SCOPES.values()), [], remediation=False,
                              settings=pipeline_settings(coverage_pipeline='v18', applicability_clear_match='model', relevance_screen='off'))
        self.assertEqual((len(provider.sent), proposal.applicability), (2, 'APPLIES'))
        trace = proposal.trace['decision_trace']
        rejected = trace['attempts'][0]
        self.assertEqual(rejected['status'], 'VALIDATION_REJECTED')
        self.assertEqual(rejected['raw']['confidence'], '0.123457')
        self.assertEqual(rejected['raw_text'], json.dumps(original))
        self.assertEqual(rejected['raw_output_hash'], sha256(rejected['raw_text'].encode('utf-8')).hexdigest())
        self.assertEqual(json.loads(rejected['raw_text'])['confidence'], 0.123456789)
        self.assertIn('parsed numeric JSON types may differ', trace['numeric_storage_encoding']['original_response'])
        payload = {'proposal': proposal.model_dump(mode='json')}
        canonical_bytes(payload)
        event = make_event(payload)
        self.assertTrue(verify_chain([event], event['event_hash'], 1))

    def test_nonfinite_raw_json_is_retained_but_never_becomes_a_parsed_prediction(self):
        for number in ('NaN', 'Infinity', '-Infinity', '1e999'):
            with self.subTest(number=number):
                raw = '{"applicability":"APPLIES","confidence":' + number + '}'
                attempt = attempt_record(raw, status='VALIDATION_REJECTED', kind='applicability')
                self.assertEqual(attempt['raw_text'], raw)
                self.assertEqual(attempt['raw_output_hash'], sha256(raw.encode('utf-8')).hexdigest())
                self.assertIsNone(attempt['raw'])
                self.assertIsNone(attempt['raw_decision'])
                trace = evidence_safe(finish_trace([attempt], {'applicability': 'UNKNOWN'}, kind='applicability'))
                event = make_event({'decision_trace': trace})
                self.assertTrue(verify_chain([event], event['event_hash'], 1))


class LocalScopeFallbackTests(unittest.TestCase):
    parent = 'Banks must retain records.'
    child = 'Banks must not disclose records except to auditors.'

    def run_pair(self, provider, *, child_subject='Banks', child_company=None, pipeline='v18'):
        child_text = self.child.replace('Banks', child_subject)
        section = {**SECTION, 'text': self.parent + ' ' + child_text}
        child_duty = {**DUTY, 'subject': child_subject, 'modality': 'MUST_NOT', 'required_action': None,
                      'prohibited_action': 'disclose records', 'exceptions': ['except to auditors']}
        settings = pipeline_settings(coverage_pipeline=pipeline, applicability_clear_match='model', relevance_screen='off')
        memo = {}
        first, _ = propose(provider, company(), section, DUTY, list(SCOPES.values()), [], remediation=False,
                           settings=settings, scope_memo=memo, clause={'offset': 0, 'text': self.parent}, obligation_id='parent')
        frozen_memo = deepcopy(memo)
        second, notes = propose(provider, child_company or company(), section, child_duty, list(SCOPES.values()), [], remediation=False,
                                settings=settings, scope_memo=memo, clause={'offset': len(self.parent)+1, 'text': child_text}, obligation_id='child')
        self.assertEqual(memo, frozen_memo, 'A local answer must not overwrite or mutate the provision memo')
        return first, second, notes

    def test_same_explicit_actor_with_changed_recipient_exception_gets_its_own_grounded_answer(self):
        provider = Scripted(answer(), answer(quote=self.child))
        first, second, notes = self.run_pair(provider)
        self.assertEqual((first.applicability, second.applicability, len(provider.sent)), ('APPLIES', 'APPLIES', 2))
        self.assertEqual(provider.sent[1]['obligation']['exceptions'], ['except to auditors'])
        self.assertFalse(provider.sent[1].get('sibling_obligations'))
        trace = second.trace['decision_trace']
        event = next(n for n in trace['gate_events'] if n['code'] == 'SCOPE_LOCAL_REJUDGE')
        self.assertFalse(event['rejected_parent']['inheritance']['allowed'])
        self.assertEqual(trace['attempts'][0]['raw']['scope_evidence'][0]['quote'], self.child)
        self.assertEqual(second.trace['scope_inheritance']['reason_code'], 'LOCAL_MODEL_REQUEST')
        self.assertEqual(second.scope_evidence[0].source_id, SECTION['id'])

    def test_local_failure_does_not_restore_the_parent_applies(self):
        class FailsAfterParent(Scripted):
            def _chat(self, prompt, payload, schema):
                if not self.answers:
                    self.sent.append(deepcopy(payload))
                    raise ProviderFailure('local scope unavailable')
                return super()._chat(prompt, payload, schema)
        provider = FailsAfterParent(answer())
        _, second, _ = self.run_pair(provider)
        self.assertEqual(second.applicability, 'UNKNOWN')
        self.assertGreaterEqual(len(provider.sent), 2)
        self.assertIsNone(second.trace['decision_trace']['VALIDATED_MODEL_DECISION']['decision'])

    def test_failed_local_answer_is_not_promoted_by_the_old_clear_match_rescue(self):
        class FailsAfterParent(Scripted):
            def _chat(self, prompt, payload, schema):
                if not self.answers:
                    raise ProviderFailure('local scope unavailable')
                return super()._chat(prompt, payload, schema)

        def clear_rules(*args, **kwargs):
            chain = rule_chain(*args, **kwargs)
            # Isolate the legacy clear-match rescue: a changed exception still
            # refuses reuse, even when the independent entity gates all match.
            chain.clear_match = True
            return chain

        with patch('regchain.pilot.engine.rule_chain', side_effect=clear_rules):
            _, second, notes = self.run_pair(FailsAfterParent(answer()))
        self.assertEqual(second.applicability, 'UNKNOWN')
        self.assertIn('SCOPE_LOCAL_REJUDGE', [n.get('code') for n in notes])
        self.assertNotIn('RULE_OVER_UNDECIDED_MODEL', [n.get('code') for n in notes])

    def test_v19_local_request_uses_current_child_context_and_fitted_source_validation(self):
        provider = Scripted(answer(), answer(quote=self.child))
        observed = []
        from regchain.pilot import engine
        real_judge = engine.judge_scope
        def traced(*args, **kwargs):
            observed.append(kwargs.get('fitted'))
            return real_judge(*args, **kwargs)
        with patch('regchain.pilot.engine.judge_scope', side_effect=traced):
            _, second, _ = self.run_pair(provider, pipeline='v19')
        self.assertEqual(observed, [True, True])
        self.assertEqual(second.applicability, 'APPLIES')
        self.assertEqual(second.trace['decision_trace']['source_lineage']['final_validated_source_set'], ['p0', 's2'])

    def test_different_actor_is_not_given_the_same_actor_local_retry(self):
        provider = Scripted(answer())
        _, second, notes = self.run_pair(provider, child_subject='Trustees')
        self.assertEqual(len(provider.sent), 1)
        self.assertNotEqual(second.applicability, 'APPLIES')
        self.assertNotIn('SCOPE_LOCAL_REJUDGE', [n.get('code') for n in notes])

    def test_missing_child_profile_is_withheld_without_a_retry(self):
        provider = Scripted(answer())
        _, second, _ = self.run_pair(provider, child_company=company(licences=None))
        self.assertEqual((len(provider.sent), second.applicability, second.applicability_rule), (1, 'UNKNOWN', 'PROFILE_INCOMPLETE'))

    def test_rejected_parent_float_trace_remains_canonical_in_local_child_packet(self):
        original = {**answer(), 'confidence': 0.123456789}
        provider = Scripted(original, answer(), answer(quote=self.child))
        _, second, _ = self.run_pair(provider)
        event = next(n for n in second.trace['decision_trace']['gate_events'] if n['code'] == 'SCOPE_LOCAL_REJUDGE')
        parent_attempt = event['rejected_parent']['decision_trace']['attempts'][0]
        self.assertEqual(parent_attempt['raw']['confidence'], '0.123457')
        self.assertEqual(parent_attempt['raw_text'], json.dumps(original))
        packet = make_event({'proposal': second.model_dump(mode='json')})
        self.assertTrue(verify_chain([packet], packet['event_hash'], 1))


class IndependentLocalScopeTests(unittest.TestCase):
    """A passive parent and a matched local actor need separate source-bound questions."""
    parent = '(1) Transfer mesajlarında gönderene ilişkin bilgiler bulunur.'
    receiving = '(2) Mesajı alan finansal kuruluş eksik bilgileri tamamlatır.'
    intermediary = ('(2) Gönderen finansal kuruluştan alıcı finansal kuruluşa uzanan mesaj zincirindeki bilgiler '
                    'transfere aracılık eden tüm finansal kuruluşlarca aktarılır.')
    exception = '(3) Bankaların kendi nam ve hesaplarına yaptıkları transferler birinci fıkra kapsamı dışındadır.'
    scopes = [{'id': 'purpose', 'printed_label': 'Amaç', 'text': 'Bu düzenleme kayıt usullerini belirler.'},
              {'id': 'listed', 'printed_label': 'Yükümlüler', 'heading_path': ['Yükümlüler'],
               'text': '(1) Yükümlüler şunlardır: a) Bankalar. b) Sigorta şirketleri.'}]

    def profile(self):
        return company(jurisdictions=['TR'], activities=['mevduat bankacılığı'],
                       licences=['BDDK mevduat bankası faaliyet izni'], products=['para transferleri'])

    def response(self, text, local=False):
        return {'applicability': 'APPLIES', 'company_fact_keys': ['licences'],
                'scope_evidence': [{'source_id': 'p0', 'quote': text}],
                'basis': [{'company_fact': 'BDDK mevduat bankası faaliyet izni',
                           'regulatory_condition': 'a) Bankalar.' if local else self.parent, 'match': 'YES'}],
                'applicability_reason': 'The stated bank licence is compared with the supplied source.',
                'missing_information': []}

    def run_pair(self, provider, child=None, child_profile=None):
        child = child or self.receiving
        section = {**SECTION, 'text': ' '.join([self.parent, child, self.exception])}
        parent_duty = {**DUTY, 'subject': 'Transfer mesajlarında gönderene ilişkin bilgiler',
                       'required_action': 'bulunur'}
        child_duty = {**DUTY, 'subject': 'Mesajı alan finansal kuruluş' if child == self.receiving else
                      'Gönderen finansal kuruluştan alıcı finansal kuruluşa uzanan mesaj zincirindeki bilgiler',
                      'required_action': child.split(' ', 1)[1]}
        settings = pipeline_settings(coverage_pipeline='v19', applicability_clear_match='model', relevance_screen='off')
        memo = {}
        first, _ = propose(provider, self.profile(), section, parent_duty, self.scopes, [], remediation=False,
                           settings=settings, scope_memo=memo, clause={'offset': 0, 'text': self.parent},
                           turkish_text=True, obligation_id='passive-parent')
        saved = deepcopy(memo)
        second, notes = propose(provider, child_profile or self.profile(), section, child_duty, self.scopes, [], remediation=False,
                                settings=settings, scope_memo=memo, clause={'offset': len(self.parent)+1, 'text': child},
                                turkish_text=True, obligation_id='local-child')
        self.assertEqual(memo, saved)
        return first, second, notes

    def test_changed_actor_with_independent_list_and_profile_match_gets_local_judgement(self):
        for child in (self.receiving, self.intermediary):
            with self.subTest(child=child):
                provider = Scripted(self.response(self.parent), self.response(child, local=True))
                first, second, notes = self.run_pair(provider, child)
                self.assertEqual((first.applicability, second.applicability, len(provider.sent)), ('APPLIES', 'APPLIES', 2))
                event = next(n for n in notes if n.get('code') == 'SCOPE_LOCAL_REJUDGE')
                self.assertEqual(event['local_assessment']['reason_code'], 'INDEPENDENT_LOCAL_ACTOR_MATCH')
                self.assertFalse(event['rejected_parent']['inheritance']['allowed'])
                self.assertEqual(second.trace['exemption_check']['status'], 'POSSIBLE', 'Rejudging does not erase an exception')
                self.assertIn('EXEMPTION_POSSIBLE', second.review_flags)
                self.assertEqual(provider.sent[-1]['obligation']['subject'], second.trace['scope_inheritance']['current_local_scope']['subject'])
                self.assertFalse(provider.sent[-1].get('sibling_obligations'))
                required = second.trace['decision_trace']['source_lineage']['required_local_assessment_evidence']
                self.assertIn({'source_id': 's2', 'quote': 'a) Bankalar.'}, required)
                self.assertIn({'source_id': 'p0', 'quote': child}, required)
                self.assertEqual(second.basis[0].source_id, 'listed')

    def test_failed_independent_local_judgement_does_not_restore_parent_answer(self):
        class FailedChild(Scripted):
            def _chat(self, prompt, payload, schema):
                if not self.answers:
                    self.sent.append(deepcopy(payload))
                    raise ProviderFailure('child answer unavailable')
                return super()._chat(prompt, payload, schema)
        provider = FailedChild(self.response(self.parent))
        _, second, notes = self.run_pair(provider)
        self.assertEqual(second.applicability, 'UNKNOWN')
        self.assertEqual(len(provider.sent), 2)
        self.assertIn('SCOPE_LOCAL_REJUDGE', [n.get('code') for n in notes])

    def test_missing_profile_still_blocks_independent_child_call(self):
        provider = Scripted(self.response(self.parent))
        profile = self.profile().model_copy(update={'licences': None})
        _, second, _ = self.run_pair(provider, child_profile=profile)
        self.assertEqual((len(provider.sent), second.applicability_rule), (1, 'PROFILE_INCOMPLETE'))

    def test_required_actor_source_or_local_clause_cannot_disappear_during_fit(self):
        for missing in ('s2', 'local_clause'):
            with self.subTest(missing=missing):
                provider = Scripted(self.response(self.parent))
                def remove_required(instance, prompt, payload, schema, compress):
                    if payload['obligation']['subject'].startswith('Mesajı alan'):
                        payload = deepcopy(payload)
                        if missing == 's2':
                            payload['scope'] = payload['scope'][:1]
                        else:
                            payload['provision']['text'] = self.parent
                    return instance, payload, {}
                with patch('regchain.pilot.engine.fit_call', side_effect=remove_required), \
                        patch('regchain.pilot.engine.trimmed_retry', return_value=(None, [])):
                    _, second, notes = self.run_pair(provider)
                self.assertEqual((len(provider.sent), second.applicability), (1, 'UNKNOWN'))
                self.assertIn('SOURCE_INCOMPLETE', [n.get('code') for n in notes])
                attempt = second.trace['decision_trace']['attempts'][0]
                self.assertEqual(attempt['status'], 'CONTEXT_ADMISSION')
                self.assertIsNone(attempt['request_hash'])

    def test_match_labels_without_source_and_profile_value_do_not_authorize_local_call(self):
        origin = local_scope({'subject': 'Messages'}, {'text': 'Messages include information.'})
        current = local_scope({'subject': 'Banks'}, {'text': 'Banks keep records.'})
        actor = {'status': 'MATCH', 'clear': True, 'evidence': {}}
        valid = {'source_id': 'scope', 'quote': 'Banks are covered.', 'company_field': 'licences', 'company_value': 'bank licence'}
        clause_gate = {'required_entities': [{'role': 'obliged_party', 'match': 'MATCH', 'text': 'Banks', 'type': 'BANK'}]}
        for evidence, source, profile in (({}, 'Banks are covered.', company()),
                                         (valid, 'Insurers are covered.', company()),
                                         (valid, 'Banks are covered.', company(licences=[]))):
            with self.subTest(evidence=evidence, source=source):
                chain = fake_chain(actor=actor, subject={**actor, 'evidence': evidence})
                chain.decision = 'OPEN'
                result = local_assessment_evidence(origin, current, chain, profile, clause_gate, {'scope': source})
                self.assertFalse(result['allowed'])

    def test_independent_actor_definition_is_bound_and_required_in_the_local_request(self):
        origin = local_scope({'subject': 'Messages'}, {'text': 'Messages include information.'})
        current = local_scope({'subject': 'Banks'}, {'text': 'Banks keep records.'})
        evidence = {'source_id': 'scope', 'quote': 'Banks are covered.', 'company_field': 'licences', 'company_value': 'bank licence'}
        definition = {'source_id': 'definition', 'quote': 'Financial institutions include banks.', 'category': 'FINANCIAL_GENERIC'}
        actor = {'status': 'MATCH', 'clear': True, 'evidence': {'definitions': [definition]}}
        chain = fake_chain(actor=actor, subject={'status': 'MATCH', 'clear': True, 'evidence': evidence})
        chain.decision = 'OPEN'
        gate = {'required_entities': [{'role': 'obliged_party', 'match': 'MATCH', 'text': 'Banks', 'type': 'BANK'}]}
        sources = {'scope': 'Banks are covered.', 'definition': definition['quote']}
        permission = local_assessment_evidence(origin, current, chain, company(), gate, sources)
        self.assertTrue(permission['allowed'])
        self.assertIn({'source_id': 'definition', 'quote': definition['quote']}, permission['required_evidence'])
        self.assertFalse(local_assessment_evidence(origin, current, chain, company(), gate, {'scope': sources['scope']})['allowed'])


if __name__ == '__main__':
    unittest.main()

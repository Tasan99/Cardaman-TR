"""v0.19 t6 (P4): the truncation ladder of the thinking verifier.

Measured on the t3/t4/t5 evidence runs (25 September 2026, 372 thinking verifier calls): 7 answers were cut at num_predict
(OUTPUT_TRUNCATED), all on ordinary prompts, while every answered call used at most 3,982 output tokens: a cut answer is a
runaway, not a slightly long one, and 4 of the 5 recovered retries ended in 681-1,487 tokens once the request changed. In
the t5 micro run one confirmation was cut at 8k and then, as the identical request, at 16k (490 s), and the row became a
PARTIAL whose only gap was that failure. The ladder now:
- a cut answer is asked again as the COMPACT request (the six decision fields, a short prompt, the same schema) on the
  SAME window, never as the identical request and never on the wide window first;
- only a cut compact answer goes to the 16k twin (adaptive mode), with its own fallback reason;
- after that no further call: the reason code VERIFIER_TRUNCATED, and the question's documented failure outcome
  (a confirmation leaves the fast answer unconfirmed, a possible conflict stays open for review), never a guess;
- a request that is answered at once is the 8k request as before, byte for byte.
Every duty and passage here is synthetic; every model is a scripted fake or a transport-level fake. No Ollama.
"""
import json
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes, digest
from regchain.evaluation import metrics
from regchain.extraction.providers import (AI_UNCACHED, AI_WINDOW, COMPACT_TRUNCATED, WINDOW_ACTIONS, claim_fit, compact_after_truncation,
                                           estimate_tokens, failure, request_hashes)
from regchain.extraction.structure import duty_payload, structure_of
from regchain.pilot import conflict as precheck
from regchain.pilot import engine
from regchain.pilot.engine import (COMPACT_QUESTIONS, COMPACT_SIGNALS, COMPACT_VERIFY_PROMPT, QUESTIONS, SUPPORT_ESCALATIONS, V19_PROMPT_HASH,
                                   V19_RETRY_CODES, VERIFIER_FAILED_FLAG, VERIFIER_TRUNCATED, VERIFY_PROMPT, compact_duty, compact_request,
                                   coverage_of_v19, model_answer, passages_v19, prompt_registry, pipeline_settings, verify_payload)
from test_v018_reliability import SCHEMA, fake_client
from test_v019_conflict_pipeline import V19, candidate, fast, rows, verdict
from test_v019_context import CLIENT, adaptive, answer, judge

# A synthetic duty whose exception relieves (the shape that ran away): below an amount the name is recorded, its check is
# optional. The stricter passage records every amount and checks it.
RELIEF_UNIT = "(2) Yükümlüler, onbin TL'nin altındaki nakit ödemelerde müşterinin adını kaydeder. Bu bilginin teyidi zorunlu değildir."
RELIEF_ACT = "onbin TL'nin altındaki nakit ödemelerde müşterinin adını kaydeder"
STRICTER = ('Şirketimizce alınan tüm nakit ödemelerde, tutarına bakılmaksızın, müşterinin adı kaydedilir ve bu bilginin doğruluğu teyit '
            'edilir.')
SKIPPED = "Onbin TL'nin altındaki nakit ödemelerde müşterinin adı kaydedilmez."
SHREDDED = 'Ödeme kayıtları beş yıl sonra imha edilir.'
RECEIPT = 'Nakit ödemelerde müşteriye makbuz verilmez.'
# A synthetic duty with an object element, for the six decision fields.
KEEP_UNIT = '(3) Yükümlüler, müşteri sözleşmesinin bir örneğini şube arşivinde saklar. Bu örneğin noter onaylı olması zorunlu değildir.'
KEEP_ACT = 'müşteri sözleşmesinin bir örneğini şube arşivinde saklar'


def made(unit, act):
    """(duty payload, labelled quantities) of a synthetic MUST duty whose second sentence is its exception."""
    value = candidate(unit, 'Yükümlüler', 'MUST', act).model_dump(mode='json')
    value['exceptions'] = [unit.split('. ')[1]]
    structure = structure_of(unit, value, 0)
    return duty_payload(value, structure), precheck.labelled_quantities(structure)


def relief():
    return made(RELIEF_UNIT, RELIEF_ACT)


def ids_of(duty):
    return [e['id'] for e in duty['elements']]


def cut():
    return failure('Provider output incomplete or truncated', 'output_truncated')


def supports(duty, quote):
    return {**verdict(relation='SUPPORTS', support=quote), 'covered_elements': ids_of(duty), 'missing_elements': []}


class Scripted(V19):
    """A judge with a window whose verifier answers are scripted in call order (an exception is raised). Every verifier
    call is recorded with its prompt, payload, window, cache setting, and the fit decision providers kept for exactly
    that request (what OllamaProvider writes into its call record)."""

    def __init__(self, num_ctx, num_predict, script=(), fast_answer=None, thinking=True):
        super().__init__(fast=fast_answer)
        self.num_ctx, self.num_predict, self.thinking, self.script, self.asked = num_ctx, num_predict, thinking, list(script), []

    def _chat(self, prompt, payload, schema):
        if 'conflict' not in schema['properties']:
            return super()._chat(prompt, payload, schema)
        hashes = request_hashes(prompt, payload, schema)
        fitted = claim_fit(self, *hashes) or {}
        self.asked.append({'prompt': prompt, 'payload': payload, 'num_ctx': self.num_ctx, 'uncached': AI_UNCACHED.get(), 'hashes': hashes,
                           'window_action': fitted.get('window_action'), 'reason': fitted.get('16k_fallback_reason'),
                           'estimate': estimate_tokens(prompt, payload, schema)})
        answer = self.script.pop(0) if self.script else verdict(relation='UNRELATED')
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer)


def pair(narrow_script, wide_script=(), fast_answer=None, mode='adaptive'):
    """An 8k thinking judge with its 16k twin (adaptive) or without using it (fixed)."""
    narrow = Scripted(8192, 4096, narrow_script, fast_answer)
    narrow.wide, narrow.ctx_mode = Scripted(16384, 6144, wide_script), mode
    narrow.wide.ctx_mode = mode
    return narrow


def gate(judge, payload, quantities, texts):
    """passages_v19 and the engine's aggregation (strong gate on, as analyze runs it)."""
    checks, _, results, _, _ = passages_v19(judge, payload, rows(*texts), frozenset(), quantities, 1, 's', 'o', False)
    canonical_bytes(results)                                     # packets forbid floats: the records must be canonical
    outcome = coverage_of_v19(checks, results, frozenset(), quantities, payload.get('elements'), strong_gate=True)
    return {c.source_id: c.relation for c in checks}, {r['source_id']: r for r in results}, outcome


def question_of(call):
    keys = {**{text: key for key, text in QUESTIONS.items()}, **{text: key for key, text in COMPACT_QUESTIONS.items()}}
    return keys.get(call['payload'].get('question'))


def calls(judge):
    """Every verifier call of the 8k judge and its twin, in order: (prompt, window, window action)."""
    wide = getattr(judge, 'wide', None)
    return [('compact' if c['prompt'] == COMPACT_VERIFY_PROMPT else 'full', c['num_ctx'], c['window_action'])
            for c in [*judge.asked, *(wide.asked if wide is not None else [])]]


class Isolated(unittest.TestCase):
    """Every test starts without fit decisions waiting from another test."""

    def setUp(self):
        token = AI_WINDOW.set(None)
        self.addCleanup(AI_WINDOW.reset, token)


class LadderOrderTests(Isolated):
    def test_a_cut_answer_is_asked_again_as_the_compact_request_on_the_same_window_first(self):
        payload, quantities = relief()
        judge = pair([cut(), supports(payload, STRICTER)], fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)))
        relations, results, (coverage, _, _, flags) = gate(judge, payload, quantities, [STRICTER])
        self.assertEqual(calls(judge), [('full', 8192, 'fits'), ('compact', 8192, 'compact_after_truncation')])
        self.assertEqual(judge.wide.asked, [])                                   # the 16k twin is never the first remedy
        self.assertEqual([question_of(c) for c in judge.asked], ['CONFIRM_COVERS', 'CONFIRM_COVERS'])
        self.assertEqual([c['uncached'] for c in judge.asked], [False, True])   # the retry is asked, not read from the cache
        self.assertEqual((relations['p1'], results['p1']['covers_confirmation'], coverage, flags), ('SUPPORTS', 'CONFIRMED', 'COVERS_TEXT', []))
        retry = [n for n in results['p1']['notes'] if n.get('retry')]
        self.assertEqual(len(retry), 1)
        self.assertIn('compact_after_truncation', retry[0]['retry'])
        self.assertNotIn(VERIFIER_TRUNCATED, [n.get('code') for n in results['p1']['notes']])

    def test_only_a_cut_compact_answer_goes_to_the_wide_window_with_its_own_reason(self):
        payload, quantities = relief()
        judge = pair([cut(), cut()], [supports(payload, STRICTER)], fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)))
        relations, results, (coverage, _, _, _) = gate(judge, payload, quantities, [STRICTER])
        self.assertEqual(calls(judge), [('full', 8192, 'fits'), ('compact', 8192, 'compact_after_truncation'),
                                        ('compact', 16384, 'fallback_large_after_truncation')])
        self.assertEqual(judge.wide.asked[0]['reason'], COMPACT_TRUNCATED)
        self.assertEqual(judge.wide.asked[0]['hashes'], judge.asked[1]['hashes'])          # the same compact request, widened
        self.assertTrue(judge.wide.asked[0]['uncached'])
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'COVERS_TEXT'))
        self.assertEqual([('compact_after_truncation' in n['retry'], 'fallback_large_after_truncation' in n['retry'])
                          for n in results['p1']['notes'] if n.get('retry')], [(True, False), (False, True)])

    def test_no_identical_request_is_resent_as_the_first_remedy(self):
        payload, quantities = relief()
        for mode in ('adaptive', 'fixed'):
            with self.subTest(mode=mode):
                judge = pair([cut(), cut()], [cut()], fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)), mode=mode)
                gate(judge, payload, quantities, [STRICTER])
                first, second = judge.asked[:2]
                self.assertEqual((first['prompt'], second['prompt']), (VERIFY_PROMPT, COMPACT_VERIFY_PROMPT))
                self.assertNotEqual(first['hashes'][:2], second['hashes'][:2])
                self.assertEqual(first['hashes'][2], second['hashes'][2])                  # the same answer schema: the same gates
                self.assertLess(second['estimate'], first['estimate'])
                self.assertEqual(second['num_ctx'], first['num_ctx'])

    def test_fixed_mode_climbs_to_the_compact_request_and_stops_there(self):
        payload, quantities = relief()
        judge = pair([cut(), cut()], [supports(payload, STRICTER)], fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)),
                     mode='fixed')
        _, results, _ = gate(judge, payload, quantities, [STRICTER])
        self.assertEqual(calls(judge), [('full', 8192, 'fits'), ('compact', 8192, 'compact_after_truncation')])
        self.assertEqual(judge.wide.asked, [])
        ended = [n for n in results['p1']['notes'] if n.get('code') == VERIFIER_TRUNCATED]
        self.assertEqual([(n['attempts'], n['window_actions']) for n in ended], [(2, ['fits', 'compact_after_truncation'])])

    def test_a_timeout_keeps_its_one_identical_retry(self):
        # Not a runaway: the service was slow. The one retry of V19_RETRY_CODES stays as it was (the same request, uncached).
        payload, quantities = relief()
        judge = pair([failure('slow', 'transport', 'TIMEOUT'), supports(payload, STRICTER)],
                     fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)))
        relations, _, _ = gate(judge, payload, quantities, [STRICTER])
        self.assertEqual(calls(judge), [('full', 8192, 'fits'), ('full', 8192, 'fits')])
        self.assertEqual(judge.asked[0]['hashes'], judge.asked[1]['hashes'])
        self.assertEqual(relations['p1'], 'SUPPORTS')


class TerminalStateTests(Isolated):
    def run_cut(self, fast_answer, texts=(STRICTER,)):
        payload, quantities = relief()
        judge = pair([cut(), cut(), cut()], [cut(), cut()], fast_answer=fast_answer)
        return judge, gate(judge, payload, quantities, list(texts))

    def test_after_the_wide_window_no_further_call_and_the_reason_code(self):
        payload, _ = relief()
        judge, (relations, results, (coverage, reason, _, flags)) = self.run_cut(fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)))
        self.assertEqual(len(judge.asked) + len(judge.wide.asked), 3)
        codes = [n.get('code') for n in results['p1']['notes']]
        self.assertEqual(codes[-2:], [VERIFIER_TRUNCATED, 'VERIFIER_FAILED'])
        ended = results['p1']['notes'][-2]
        self.assertEqual((ended['failure_code'], ended['attempts'], ended['window_actions']),
                         ('OUTPUT_TRUNCATED', 3, ['fits', 'compact_after_truncation', 'fallback_large_after_truncation']))
        self.assertEqual(results['p1']['notes'][-1], {'question': 'verify', 'code': 'VERIFIER_FAILED', 'failure_code': 'OUTPUT_TRUNCATED'})
        # A confirmation that never came leaves the fast support unconfirmed: the count is UNKNOWN for review, never PARTIAL.
        self.assertEqual((relations['p1'], results['p1']['covers_confirmation']), ('SUPPORTS', 'FAILED'))
        self.assertEqual((coverage, flags), ('UNKNOWN', ['COVERS_UNCONFIRMED', VERIFIER_FAILED_FLAG]))
        self.assertIn(VERIFIER_TRUNCATED, reason)
        self.assertIn('OUTPUT_TRUNCATED', reason)

    def test_a_possible_conflict_that_ran_away_stays_open_for_review(self):
        judge, (relations, results, (coverage, reason, _, flags)) = self.run_cut(fast('POSSIBLE_CONFLICT', SKIPPED), texts=(SKIPPED,))
        self.assertEqual([question_of(c) for c in [*judge.asked, *judge.wide.asked]], ['FAST_POSSIBLE_CONFLICT'] * 3)
        self.assertEqual((relations['p1'], results['p1']['uncertainty']), ('UNCLEAR', 'strong'))
        self.assertEqual((coverage, flags), ('UNKNOWN', ['POSSIBLE_CONFLICT_UNRESOLVED']))
        self.assertIn(VERIFIER_TRUNCATED, reason)
        self.assertIn(VERIFIER_TRUNCATED, results['p1']['reason'])
        self.assertIn('manual review required', results['p1']['reason'])

    def test_a_partial_confirmation_that_ran_away_keeps_the_fast_partial_unconfirmed(self):
        payload, _ = relief()
        judge, (relations, results, (coverage, _, _, flags)) = self.run_cut(
            fast('POSSIBLE_PARTIAL', STRICTER, covered=[i for i in ids_of(payload) if i != 'action'], missing=('action',)))
        self.assertEqual([question_of(c) for c in judge.asked], ['CONFIRM_PARTIAL', 'CONFIRM_PARTIAL'])
        self.assertEqual((relations['p1'], results['p1']['partial_confirmation'], coverage, flags),
                         ('PARTIAL', 'FAILED', 'PARTIAL', ['PARTIAL_UNCONFIRMED']))

    def test_the_same_failure_always_gives_the_same_outcome(self):
        payload, _ = relief()
        seen = {json.dumps(self.run_cut(fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)))[1][2], sort_keys=True) for _ in range(3)}
        self.assertEqual(len(seen), 1)


class CompactRequestTests(unittest.TestCase):
    def test_the_compact_payload_keeps_the_six_decision_fields(self):
        duty, _ = made(KEEP_UNIT, KEEP_ACT)
        shown = compact_duty(duty)
        self.assertEqual(set(shown), {'actor', 'polarity', 'action', 'object', 'scope', 'source_quote'})
        self.assertEqual((shown['actor'], shown['polarity']), ('obliged party', 'REQUIRED'))
        self.assertEqual(shown['action'], {'action': KEEP_ACT})
        self.assertEqual(shown['object'], {'object': 'müşteri sözleşmesinin bir örneğini şube arşivinde'})
        self.assertEqual(shown['scope'], {'subject': 'Yükümlüler', 'exception_1': 'Bu örneğin noter onaylı olması zorunlu değildir.'})
        self.assertEqual(shown['source_quote'], 'Yükümlüler, müşteri sözleşmesinin bir örneğini şube arşivinde saklar.')
        # Every element id once, so covered and missing still map; nothing the decision does not rest on.
        grouped = [i for key in ('action', 'object', 'scope') for i in shown[key]]
        self.assertEqual(sorted(grouped), sorted(ids_of(duty)))
        self.assertTrue(all(isinstance(v, str) for key in ('action', 'object', 'scope') for v in shown[key].values()))

    def test_a_prohibition_and_a_transfer_role_are_kept(self):
        duty = {'subject': 'Aracı kuruluşlar', 'modality': 'MUST_NOT', 'prohibited_action': 'gönderene ilişkin bilgileri mesajdan çıkaramaz',
                'source_sentence': 'Aracı kuruluşlar, aktarılan transfer mesajlarından gönderene ilişkin bilgileri çıkaramaz.',
                'elements': [{'id': 'action', 'kind': 'action', 'text': 'gönderene ilişkin bilgileri mesajdan çıkaramaz'},
                             {'id': 'prohibition', 'kind': 'prohibition', 'text': 'çıkaramaz'},
                             {'id': 'subject', 'kind': 'subject', 'text': 'Aracı kuruluşlar'}]}
        shown = compact_duty(duty)
        self.assertEqual(shown['polarity'], 'PROHIBITED')
        self.assertEqual(shown['action'], {'action': 'gönderene ilişkin bilgileri mesajdan çıkaramaz', 'prohibition': 'çıkaramaz'})
        self.assertEqual(shown['actor'], precheck.role_view(duty)['actor'])
        self.assertEqual({k: shown[k] for k in ('transfers', 'to') if k in shown}, {k: v for k, v in precheck.role_view(duty).items()
                                                                                    if k in ('transfers', 'to')})

    def test_the_compact_request_is_smaller_and_keeps_the_passage_as_sent(self):
        payload, quantities = relief()
        text = STRICTER + ' ' + SHREDDED
        signals, _, _ = engine.precheck_passage(payload, text, quantities)
        reading = {'row': {'source_id': 'p1', 'text': text}, 'duty': payload, 'signals': signals, 'quantities': quantities}
        schema = engine.verify_schema(ids_of(payload))
        for question in QUESTIONS:
            with self.subTest(question=question):
                full = verify_payload(reading, question)
                prompt, compact = compact_request(reading, question, full)
                self.assertEqual(prompt, COMPACT_VERIFY_PROMPT)
                self.assertLess(estimate_tokens(prompt, compact, schema), estimate_tokens(VERIFY_PROMPT, full, schema))
                self.assertEqual(compact['passage'], full['passage'])
                self.assertEqual(compact['question'], COMPACT_QUESTIONS[question])
                self.assertEqual(compact.get('quantity_check'), full.get('quantity_check'))
                self.assertNotIn('modality', compact['duty'])
                if question in SUPPORT_ESCALATIONS:
                    self.assertNotIn('automatic_signals', compact)
                else:
                    self.assertEqual(compact.get('automatic_signals', []), full['automatic_signals'][:COMPACT_SIGNALS])
        # A passage the first request sent compressed stays compressed.
        _, compact = compact_request(reading, 'CONFIRM_COVERS', {'passage': '[...] ' + SHREDDED})
        self.assertEqual(compact['passage'], '[...] ' + SHREDDED)
        self.assertEqual(set(COMPACT_QUESTIONS), set(QUESTIONS))

    def test_the_compact_prompt_is_registered_and_hashed(self):
        registry = prompt_registry(pipeline_settings(coverage_pipeline='v19'))
        self.assertEqual(registry['judge.verify.compact']['sha256'], digest(COMPACT_VERIFY_PROMPT))
        self.assertNotIn('judge.verify.compact', prompt_registry())                    # a v18 packet keeps its registry
        self.assertEqual(pipeline_settings(coverage_pipeline='v19')['coverage_prompts_sha256'], V19_PROMPT_HASH)
        self.assertLess(len(COMPACT_VERIFY_PROMPT.encode('utf-8')), len(VERIFY_PROMPT.encode('utf-8')))


class GateTests(Isolated):
    def test_a_compact_conflict_claim_passes_the_same_gates(self):
        payload, quantities = relief()
        # A claim on a sentence about another act (a receipt) is not a contradiction of this duty (the t5 and t6 gates).
        other = verdict(True, 'DIRECT_OPPOSITE', RECEIPT, 'UNRELATED')
        judge = pair([cut(), other], fast_answer=fast('POSSIBLE_CONFLICT', RECEIPT))
        relations, results, (coverage, _, _, _) = gate(judge, payload, quantities, [RECEIPT + ' ' + STRICTER])
        self.assertEqual(calls(judge)[1], ('compact', 8192, 'compact_after_truncation'))
        self.assertNotEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        self.assertTrue({'CONFLICT_NOT_ANCHORED', 'DIFFERENT_ACTION', 'ACTOR_ROLE_MISMATCH', *precheck.GATE_CODES}
                        & {n.get('code') for n in results['p1']['notes']})
        # The same answer shape on the sentence that skips the duty's own act below its amount stands as a contradiction.
        own = verdict(True, 'REQUIRED_ACTION_FORBIDDEN', SKIPPED, 'NOT_APPLICABLE')
        judge = pair([cut(), own], fast_answer=fast('POSSIBLE_CONFLICT', SKIPPED))
        relations, results, (coverage, _, _, _) = gate(judge, payload, quantities, [SKIPPED + ' ' + STRICTER])
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        self.assertIn(precheck.GATE_PASSED, [n.get('code') for n in results['p1']['notes']])


class DefaultPathTests(Isolated):
    def test_without_truncation_the_8k_request_is_unchanged(self):
        payload, quantities = relief()
        judge = pair([supports(payload, STRICTER)], fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)))
        relations, results, (coverage, _, _, _) = gate(judge, payload, quantities, [STRICTER])
        self.assertEqual(calls(judge), [('full', 8192, 'fits')])
        sent = judge.asked[0]
        reading = {'row': {'source_id': 'p1', 'text': STRICTER}, 'duty': payload, 'quantities': quantities,
                   'signals': engine.precheck_passage(payload, STRICTER, quantities)[0]}
        self.assertEqual((sent['prompt'], sent['payload']), (VERIFY_PROMPT, verify_payload(reading, 'CONFIRM_COVERS')))
        self.assertFalse(sent['uncached'])
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'COVERS_TEXT'))
        self.assertEqual([n for n in results['p1'].get('notes') or [] if n.get('question') == 'verify'], [])

    def test_a_caller_without_a_compact_request_keeps_the_v019_retry(self):
        # model_answer is shared: only the verifier passes `compact`; any other caller is retried as before.
        judge = pair([cut(), verdict(relation='UNRELATED')], [verdict(relation='UNRELATED')])
        value, notes, code = model_answer(judge, VERIFY_PROMPT, {'passage': 'x'}, engine.verify_schema(['action']), json.loads,
                                          'judge.verify', 'verify', V19_RETRY_CODES)
        self.assertEqual((code, calls(judge)), (None, [('full', 8192, 'fits'), ('full', 16384, 'fallback_large_after_truncation')]))
        self.assertIn('asked once more on the wide window', notes[0]['retry'])


class ProviderRecordTests(Isolated):
    def test_the_compact_rung_is_only_for_a_cut_answer_and_only_on_the_same_window(self):
        big = adaptive()
        truncated = cut()
        for instance in (big, big.quick, judge(JUDGE_NUM_CTX='8192')):                 # adaptive, no-thinking, fixed
            with self.subTest(num_ctx=instance.num_ctx, thinking=instance.thinking, mode=instance.ctx_mode):
                used, sent, info = compact_after_truncation(instance, 'p', {'passage': 'x' * 400}, SCHEMA, truncated)
                self.assertIs(used, instance)
                self.assertEqual((info['action'], info['num_ctx'], sent), ('compact_after_truncation', instance.num_ctx, {'passage': 'x' * 400}))
        for exc in (failure('slow', 'transport', 'TIMEOUT'), failure('empty', 'answer', 'EMPTY_RESPONSE')):
            self.assertIsNone(compact_after_truncation(big, 'p', {'passage': 'x'}, SCHEMA, exc))
        self.assertIsNone(compact_after_truncation(big, 'p', {'passage': 'x' * 40000}, SCHEMA, truncated))   # not admitted at 8k

    def test_the_call_records_name_each_rung_and_the_micro_report_can_count_them(self):
        big = adaptive()
        full, small = {'passage': 'x' * 3000, 'question': 'q'}, {'passage': 'x' * 1500, 'question': 'q'}
        with patch(CLIENT) as factory:
            client = fake_client(factory)
            client.stream.return_value.__enter__.return_value.iter_bytes.side_effect = [
                [answer(1300, '{', done_reason='length', eval_count=4096)], [answer(700, '{', done_reason='length', eval_count=4096)],
                [answer(700, '{"a": 2}', eval_count=900)]]
            with self.assertLogs('regchain.ai', 'WARNING'):
                value, notes, code = model_answer(big, 'the full prompt', full, SCHEMA, json.loads, 'judge.verify.support', 'verify',
                                                  V19_RETRY_CODES, compact=lambda sent: ('the compact prompt', small))
            options = [call.kwargs['json']['options'] for call in client.stream.call_args_list]
            prompts = [call.kwargs['json']['messages'][0]['content'].split('\n')[0] for call in client.stream.call_args_list]
        self.assertEqual((value, code), ({'a': 2}, None))
        self.assertEqual([(o['num_ctx'], o['num_predict']) for o in options], [(8192, 4096), (8192, 4096), (16384, 6144)])
        self.assertEqual(prompts, ['the full prompt', 'the compact prompt', 'the compact prompt'])
        records = big.call_log[-3:]
        self.assertEqual([(r['status'], r['window_action'], r['num_ctx']) for r in records],
                         [('PROVIDER_FAILURE', 'fits', 8192), ('PROVIDER_FAILURE', 'compact_after_truncation', 8192),
                          ('OK', 'fallback_large_after_truncation', 16384)])
        self.assertEqual((records[1].get('16k_fallback_reason'), records[2]['16k_fallback_reason']), (None, COMPACT_TRUNCATED))
        self.assertEqual(sum(1 for r in records if str(r['window_action']).startswith('fallback')), 1)        # one 16k retry
        self.assertEqual(sum(1 for r in records if 'compact' in str(r['window_action'])), 1)                   # one compact retry
        runtime = metrics.runtime(records)
        self.assertEqual((runtime['window_actions'], runtime['fallbacks']),
                         ({'fits': 1, 'compact_after_truncation': 1, 'fallback_large_after_truncation': 1}, 1))
        self.assertLess(WINDOW_ACTIONS.index('compact_after_truncation'), WINDOW_ACTIONS.index('fallback_large_after_truncation'))


class RecordTests(Isolated):
    def test_the_notes_are_digest_safe_and_name_only_known_codes(self):
        payload, quantities = relief()
        judge = pair([cut(), cut()], [cut()], fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=ids_of(payload)))
        _, results, _ = gate(judge, payload, quantities, [STRICTER])
        canonical_bytes(results)
        ladder = [n for n in results['p1']['notes'] if n.get('code') in ('ProviderFailure', VERIFIER_TRUNCATED, 'VERIFIER_FAILED')]
        self.assertEqual(len(ladder), 5)                                      # three cut answers, the reason code, the failure
        for note in ladder:
            for value in note.values():
                self.assertIsInstance(value, (str, int, list))
                self.assertNotIsInstance(value, bool)
                if isinstance(value, list):
                    self.assertTrue(all(isinstance(v, str) for v in value))
        actions = [a for n in results['p1']['notes'] for a in n.get('window_actions') or []]
        self.assertTrue(set(actions) <= set(WINDOW_ACTIONS))
        self.assertEqual(engine.failure_codes([results['p1']]), ['OUTPUT_TRUNCATED', VERIFIER_TRUNCATED])


if __name__ == '__main__':
    unittest.main()

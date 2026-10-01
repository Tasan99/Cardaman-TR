"""v0.19 short round 2, FIX 1: the COVERS confirmation gate and element-by-element acts.

Measured on the rep5 and indep5 smoke runs (24 September 2026): the 4b fast reading said SUPPORTS on passages
that do not state the duty (C03 md. 8(1), 9(1); independent I04 md. 4(2)) or state only part of it (I04 md. 8:
records kept, nothing about producing them on request), and aggregation turned each into COVERS_TEXT on its own.
The fast model alone never makes COVERS_TEXT now: the strong verifier confirms (question CONFIRM_COVERS), at
most twice per obligation, in the verifier stage. A coordinated action is split into act_1, act_2 ... elements.

Every model is a scripted fake (no Ollama).
"""
import dataclasses
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction.structure import coordinated_acts, duty_payload, structure_of
from regchain.pilot import conflict as precheck
from regchain.pilot import engine
from regchain.pilot.engine import (COVERS_CONFIRMATIONS, QUESTIONS, analyze, autonomous_gate, coverage_of_v19, passages_v19,
                                   pipeline_settings)
from test_pilot import company, policies, sections
from test_v019_conflict_pipeline import V19, candidate, fast, md26, md28, md46, verdict
from test_v019_short_round import Logged, switches

REPORT = "Şüpheli işlemler, şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde MASAK'a bildirilir."
REPORT_LATE = "Şüpheli işlemler uyum birimince değerlendirilir ve MASAK'a bildirilir."
KEEP = 'Kimlik tespitine ilişkin belge ve kayıtlar son işlem tarihinden itibaren sekiz yıl süreyle muhafaza edilir.'
# 5549 sayılı Kanun md. 8 (1), the shape of the independent I04 duty: one modal, two acts sharing "etmekle".
LAW_8 = ('(1) Yükümlüler, bu Kanunla getirilen yükümlülüklere ve işlemlerine ilişkin her türlü ortamdaki; belgeleri düzenleme '
         'tarihinden, defter ve kayıtları son kayıt tarihinden, kimlik tespitine ilişkin belgeleri ise son işlem tarihinden itibaren '
         'sekiz yıl süreyle muhafaza ve istenmesi halinde yetkililere ibraz etmekle yükümlüdür.')
LAW_8_ACTION = LAW_8[LAW_8.index('bu Kanunla'):LAW_8.index(' yükümlüdür')]


def rows(*texts):
    return [{'source_id': f'p{i}', 'text': text} for i, text in enumerate(texts, 1)]


def run(judge, payload, quantities, texts):
    """(relations, results by id, gated coverage outcome): the engine's own aggregation (strong_gate)."""
    checks, evidence, results, _, _ = passages_v19(judge, payload, rows(*texts), frozenset(), quantities, 1, 's', 'o', False)
    canonical_bytes(results)                                          # packets forbid floats
    outcome = coverage_of_v19(checks, results, frozenset(), quantities, payload.get('elements'), strong_gate=True)
    return {c.source_id: c.relation for c in checks}, {r['source_id']: r for r in results}, outcome


def law8():
    value = candidate(LAW_8, 'Yükümlüler', 'MUST', LAW_8_ACTION).model_dump(mode='json')
    structure = structure_of(LAW_8, value, 0)
    return duty_payload(value, structure), precheck.labelled_quantities(structure)


def confirm(relation, support, covered, missing=(), **more):
    return {**verdict(relation=relation, support=support, **more), 'covered_elements': list(covered), 'missing_elements': list(missing)}


class ConfirmationTests(unittest.TestCase):
    def test_a_fast_support_alone_is_confirmed_and_a_partial_answer_makes_partial(self):
        payload, quantities = md28()
        judge = V19(fast=fast('SUPPORTS', REPORT, covered=('action', 'deadline_1')),
                    verify=confirm('PARTIAL', REPORT, ['deadline_1'], ['action']))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [REPORT])
        calls = judge.kinds('verify')
        self.assertEqual([(c['task'], c['payload']['question']) for c in calls], [('judge.verify.support', QUESTIONS['CONFIRM_COVERS'])])
        self.assertIn('Check EVERY element id', calls[0]['payload']['question'])
        self.assertEqual((relations['p1'], coverage, flags), ('PARTIAL', 'PARTIAL', []))
        self.assertEqual(results['p1']['confirmation'], {'previous_relation': 'SUPPORTS', 'previous_escalation': 'NONE', 'outcome': 'PARTIAL'})
        self.assertEqual(results['p1']['escalation'], 'CONFIRM_COVERS')
        self.assertNotIn('COVERS_UNCONFIRMED', reason)

    def test_an_off_topic_support_confirmed_unrelated_is_no_evidence(self):
        # C03 md. 8(1)/9(1)-style: a duty with no number and a passage that only shares its words.
        payload, quantities = md26()
        off = 'Basitleştirilmiş tedbirler kapsamındaki müşterilerin sözleşmeleri elektronik ortamda imzalanır.'
        judge = V19(fast=fast('SUPPORTS', off, covered=('action',)),
                    verify=confirm('UNRELATED', '', [], ['action'], rationale='Signing contracts is another measure.'))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [off])
        self.assertEqual((results['p1']['fast']['label'], results['p1']['confirmation']['previous_escalation']), ('POSSIBLE_SUPPORT', 'NONE'))
        self.assertEqual((relations['p1'], results['p1']['covers_confirmation'], coverage), ('UNRELATED', 'UNRELATED', 'NO_EVIDENCE'))
        self.assertEqual(len(judge.kinds('verify')), 1)

    def test_a_confirmed_support_covers(self):
        payload, quantities = md28()
        judge = V19(fast=fast('SUPPORTS', REPORT, covered=('action', 'deadline_1')),
                    verify=confirm('SUPPORTS', REPORT, ['action', 'deadline_1']))
        relations, results, (coverage, _, _, flags) = run(judge, payload, quantities, [REPORT])
        self.assertEqual((relations['p1'], results['p1']['covers_confirmation'], coverage, flags), ('SUPPORTS', 'CONFIRMED', 'COVERS_TEXT', []))
        # A confirmed SUPPORTS that lists a missing element is the PARTIAL it describes.
        judge = V19(fast=fast('SUPPORTS', REPORT, covered=('action', 'deadline_1')),
                    verify=confirm('SUPPORTS', REPORT, ['action'], ['deadline_1']))
        self.assertEqual(run(judge, payload, quantities, [REPORT])[2][0], 'PARTIAL')

    def test_an_anchored_contradiction_found_by_the_confirmation_is_a_conflict(self):
        payload, quantities = md28()
        # Short round 3: the contradicting sentence must state a weaker number; a claim on the duty's own deadline
        # (REPORT: "en geç on iş günü") is STRICTER_THAN_REQUIRED, not a conflict (test_v019_stricter_partial).
        late = REPORT.replace('on iş günü', 'otuz gün')
        judge = V19(fast=fast('SUPPORTS', late, covered=('action', 'deadline_1')),
                    verify=verdict(True, 'DEADLINE_MISMATCH', late, 'NOT_APPLICABLE'))
        relations, _, (coverage, _, _, _) = run(judge, payload, quantities, [late])
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))

    def test_no_confirmation_when_the_verifier_already_read_the_passage_with_its_lists(self):
        payload, quantities = md28()
        judge = V19(fast=fast('POSSIBLE_CONFLICT', REPORT), verify=confirm('SUPPORTS', REPORT, ['action', 'deadline_1']))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [REPORT])
        self.assertEqual([c['task'] for c in judge.kinds('verify')], ['judge.verify'])          # the conflict question only
        self.assertEqual((results['p1']['escalation'], coverage), ('FAST_POSSIBLE_CONFLICT', 'COVERS_TEXT'))
        self.assertNotIn('covers_confirmation', results['p1'])
        # Nothing favourable costs no strong call at all; a fast PARTIAL the result rests on costs one (short round 3:
        # CONFIRM_PARTIAL), and a confirmed PARTIAL stands.
        for answer, expected, calls in ((fast('PARTIAL', REPORT_LATE, covered=('action',), missing=('deadline_1',)), 'PARTIAL', 1),
                                        (fast('IRRELEVANT', covered=()), 'NO_EVIDENCE', 0)):
            judge = V19(fast=answer, verify=confirm('PARTIAL', REPORT_LATE, ['action'], ['deadline_1']))
            self.assertEqual(run(judge, payload, quantities, ['Ofis girişinde kartlı geçiş sistemi kullanılır.'
                                                              if answer['label'] == 'IRRELEVANT' else REPORT_LATE])[2][0], expected)
            self.assertEqual(len(judge.kinds('verify')), calls)

    def test_at_most_two_confirmations_per_obligation_and_the_rest_stays_unconfirmed(self):
        payload, quantities = md28()
        texts = [REPORT, REPORT.replace('Şüpheli işlemler', 'Şüpheli görülen işlemler'), REPORT.replace('MASAK', 'Başkanlığa').replace("'a", '')]
        judge = V19(fast=lambda text: fast('SUPPORTS', text, covered=('action', 'deadline_1')),
                    verify=lambda text: confirm('PARTIAL', text, ['action'], ['deadline_1']))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, texts)
        self.assertEqual(COVERS_CONFIRMATIONS, 2)
        self.assertEqual([c['passage'] for c in judge.kinds('verify')], texts[:2])             # best-ranked first, then the next
        self.assertEqual((relations, coverage, flags), ({'p1': 'PARTIAL', 'p2': 'PARTIAL', 'p3': 'SUPPORTS'}, 'PARTIAL', ['COVERS_UNCONFIRMED']))
        self.assertIn('COVERS_UNCONFIRMED', reason)
        # The ungated count (the legacy aggregation) would still have called the third fast support a cover.
        checks = [engine.PolicyCheck(source_id=k, quote='q', relation=v) for k, v in relations.items()]
        self.assertEqual(coverage_of_v19(checks, list(results.values()), frozenset(), quantities, payload['elements'])[0], 'COVERS_TEXT')

    def test_a_failed_confirmation_leaves_the_fast_answer_unconfirmed(self):
        payload, quantities = md28()
        from regchain.extraction.providers import ProviderFailure
        judge = V19(fast=fast('SUPPORTS', REPORT, covered=('action', 'deadline_1')), verify=ProviderFailure('down'))
        relations, results, (coverage, _, _, flags) = run(judge, payload, quantities, [REPORT])
        # t6: the count is not settled (UNKNOWN, VERIFIER_FAILED); a model failure is not evidence of a gap (was PARTIAL).
        self.assertEqual((relations['p1'], results['p1']['covers_confirmation'], coverage, flags),
                         ('SUPPORTS', 'FAILED', 'UNKNOWN', ['COVERS_UNCONFIRMED', 'VERIFIER_FAILED']))


class CoordinatedActTests(unittest.TestCase):
    def test_coordinated_verb_clauses_become_act_elements(self):
        payload, _ = md46()
        acts = [e for e in payload['elements'] if e['kind'] == 'action']
        self.assertEqual([e['id'] for e in acts], ['act_1', 'act_2'])
        self.assertTrue(acts[0]['text'].endswith('sekiz yıl süre ile muhafaza etmek'))
        self.assertEqual(acts[1]['text'], 'istenmesi halinde yetkililere ibraz etmek')
        law, _ = law8()
        self.assertEqual([(e['id'], e['text'][-30:]) for e in law['elements'] if e['kind'] == 'action'],
                         [('act_1', LAW_8_ACTION[:LAW_8_ACTION.index(' ve istenmesi')][-30:]), ('act_2', 'istenmesi halinde yetkililere ibraz etmekle'[-30:])])
        for e in (*acts, *law['elements']):
            if e['kind'] == 'action':
                self.assertIn(e['text'], payload['required_action'] if e in acts else law['required_action'])     # exact substrings
        self.assertEqual(coordinated_acts('müşterinin kimlik bilgileri alınır ve belgelerin örnekleri beş yıl saklanır.'),
                         ['müşterinin kimlik bilgileri alınır', 'belgelerin örnekleri beş yıl saklanır'])
        self.assertEqual(coordinated_acts('retain records of each transaction for five years and produce them to the FCA on request'),
                         ['retain records of each transaction for five years', 'produce them to the FCA on request'])

    def test_one_verb_clause_keeps_the_action_element(self):
        for action in ("şüpheli işlemleri en geç on iş günü içinde MASAK'a bildirmek", 'belge ve kayıtları sekiz yıl saklamak',
                       'belge ve kayıtları saklamak veya ibraz etmek', 'kimlik tespit ve teyit edilir', 'keep records of payments and transfers',
                       'retain records or produce them'):
            with self.subTest(action=action):
                self.assertEqual(coordinated_acts(action), [])
        payload, _ = md28()
        self.assertEqual([e['id'] for e in payload['elements']][:1], ['action'])
        canonical_bytes(law8()[0])

    def test_the_act_of_producing_on_request_missing_is_partial(self):
        payload, quantities = law8()
        ids = [e['id'] for e in payload['elements']]
        self.assertEqual(ids, ['act_1', 'act_2', 'subject', 'deadline_1'])
        # The fast reading lists the ibraz act as missing: PARTIAL, confirmed by one strong call (short round 3).
        judge = V19(fast=fast('SUPPORTS', KEEP, covered=('act_1', 'deadline_1'), missing=('act_2',)),
                    verify=confirm('PARTIAL', KEEP, ['act_1', 'deadline_1'], ['act_2']))
        relations, _, (coverage, reason, _, _) = run(judge, payload, quantities, [KEEP])
        self.assertEqual((relations['p1'], coverage, [c['payload']['question'] for c in judge.kinds('verify')]),
                         ('PARTIAL', 'PARTIAL', [QUESTIONS['CONFIRM_PARTIAL']]))
        self.assertIn('ibraz', reason)
        # The fast reading claims it all (I04 md. 8): the confirmation checks each act and finds ibraz missing.
        judge = V19(fast=fast('SUPPORTS', KEEP, covered=('act_1', 'act_2', 'deadline_1')),
                    verify=confirm('SUPPORTS', KEEP, ['act_1', 'deadline_1'], ['act_2']))
        relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [KEEP])
        self.assertEqual((relations['p1'], results['p1']['covers_confirmation'], coverage), ('PARTIAL', 'PARTIAL', 'PARTIAL'))
        self.assertIn('istenmesi halinde yetkililere ibraz etmekle', reason)
        # md. 46(1) the same way.
        payload, quantities = md46()
        judge = V19(fast=fast('SUPPORTS', KEEP, covered=('act_1', 'act_2', 'deadline_1')),
                    verify=confirm('PARTIAL', KEEP, ['act_1', 'deadline_1'], ['act_2']))
        self.assertEqual(run(judge, payload, quantities, [KEEP])[2][0], 'PARTIAL')


def many_policies(*texts):
    chunks = [{'source_id': f'policy-{i}', 'policy_hash': 'd' * 64, 'filename': 'fixture.txt', 'locator': 'text_block', 'number': i,
               'start': 0, 'end': len(text), 'text': text} for i, text in enumerate(texts, 1)]
    return [{'filename': 'fixture.txt', 'raw_hash': 'd' * 64, 'bytes': 100, 'parser': 'fixture', 'chunks': chunks}]


class OrderAndFlagTests(unittest.TestCase):
    def test_the_confirmation_sits_in_the_verifier_stage_one_round_trip(self):
        log = []
        texts = ('All staff must retain records.', 'Records are retained by the operations team.', 'Staff retain customer records.')
        answers = {texts[0]: fast('POSSIBLE_CONFLICT', texts[0])}
        extraction = Logged('FAST', log, fast=lambda text: answers.get(text) or fast('SUPPORTS', text))
        judge = Logged('STRONG', log, fast=lambda text: self.fail('the judge was asked a fast question'),
                       verify=lambda text: confirm('PARTIAL', text, ['action'], []))
        payload = analyze(company(), many_policies(*texts), sections(), extraction, ['CONC 7.3.4'], judge=judge, categories=['Diğer'],
                          settings=pipeline_settings(coverage_pipeline='v19'))['events'][0]['payload']
        tasks = [(role, kind, task) for role, kind, task in log]
        labels = [i for i, (_, kind, _) in enumerate(tasks) if kind == 'label']
        strong = [i for i, (role, kind, _) in enumerate(tasks) if role == 'STRONG' and kind == 'conflict']
        self.assertEqual(payload['obligations'][0]['proposal']['coverage'], 'PARTIAL')
        # The conflict question on policy-1, then the confirmation of each fast support in turn (the cap: two).
        self.assertEqual([t for r, k, t in tasks if r == 'STRONG' and k == 'conflict'], ['judge.verify', 'judge.verify.support', 'judge.verify.support'])
        self.assertLess(max(labels), min(strong))                          # every fast reading before any verifier call
        self.assertEqual(switches(log), 2)                                   # one 4b→8b→4b round trip (the draft on the fast model)

    def test_rows_of_a_salvaged_extraction_unit_are_flagged_for_a_person(self):
        real = engine.extract

        def salvaged(*args, **kwargs):
            result = real(*args, **kwargs)
            return dataclasses.replace(result, diagnostics=(*result.diagnostics, {'code': 'GROUNDING_SALVAGED', 'detail': 'normalised span'}))

        judge = V19(fast=lambda text: fast('SUPPORTS', text), verify=lambda text: verdict(relation='SUPPORTS', support=text))
        settings = pipeline_settings(coverage_pipeline='v19')
        with patch('regchain.pilot.engine.extract', side_effect=salvaged):
            row = analyze(company(), policies(), sections(), judge, ['CONC 7.3.4'], settings=settings)['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['review_flags'][0], 'EXTRACTION_SALVAGED')
        self.assertIn('EXTRACTION_SALVAGED', autonomous_gate(row))
        self.assertTrue(row['proposal']['trace']['human_review'])
        plain = analyze(company(), policies(), sections(), judge, ['CONC 7.3.4'], settings=settings)['events'][0]['payload']['obligations'][0]
        self.assertNotIn('EXTRACTION_SALVAGED', plain['proposal']['review_flags'])


if __name__ == '__main__':
    unittest.main()

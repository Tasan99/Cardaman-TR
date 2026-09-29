"""v0.19 t6 (P2): the COVERS / PARTIAL boundary.

PARTIAL needs a gap some reading states: a critical element missing (act, object, prohibition, deadline or timing,
threshold, item), a number not met, a timing the passage's text does not state. Measured on the v0.19 t5 micro run
(25 September 2026):
- a timing without a number ("derhal") was met by the model's element list alone: the verifier listed it covered for a
  sentence that states no timing, and a gold PARTIAL row became COVERS_TEXT. A timing is now read in the text like a
  number: a listing counts only when that passage's own text states a timing that meets it;
- a verifier failure (OUTPUT_TRUNCATED twice) on the confirmation of the only covering passage made a PARTIAL whose
  only gap was COVERS_UNCONFIRMED. A model failure is not evidence of a gap: the count is UNKNOWN (VERIFIER_FAILED);
- when the verifier read a cover except for such a timing, the confirmation goes to the passage whose text states it.
Wording, an unrepeated subject, condition or exception, and the obliged party in other words stay no gap; a verifier's
own PARTIAL is never raised by the code (the recipient and unenumerated items are not elements, the verifier carries
them). Every duty and passage here is synthetic; every model is a scripted fake (tests/test_v019_conflict_pipeline.V19).
"""
import json
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction.providers import ProviderFailure, failure
from regchain.extraction.structure import duty_payload, structure_of
from regchain.pilot import conflict as precheck
from regchain.pilot import engine
from regchain.pilot.engine import (QUESTIONS, TIMING_NOT_IN_TEXT, TIMING_NOT_STATED, TIMING_STATED, VERIFIER_FAILED_FLAG, confirmation_failed,
                                   coverage_of_v19, failure_codes, passages_v19, stated_timings, timing_match)
from test_v019_conflict_pipeline import V19, candidate, fast, rows, verdict

# Synthetic duties, one per timing kind without a number.
NOW_UNIT = '(2) Yükümlüler, aracı kuruluştan alınan müşteri bilgilerini derhal müşteri dosyasına ekler.'
NOW_ACT = 'aracı kuruluştan alınan müşteri bilgilerini derhal müşteri dosyasına ekler'
BEFORE_UNIT = '(1) Şirket, hesap açılmadan önce müşterinin adres bilgisini doğrular.'
BEFORE_ACT = 'hesap açılmadan önce müşterinin adres bilgisini doğrular'
END_UNIT = '(4) Yükümlüler, dönem sonunda risk değerlendirme raporunu yönetim kuruluna sunar.'
END_ACT = 'dönem sonunda risk değerlendirme raporunu yönetim kuruluna sunar'
# Passages for the "derhal" duty: the act without a timing, the act with it, the timing for another act.
UNTIMED = 'Aracı kuruluştan alınan müşteri bilgileri müşteri dosyasına eklenir.'
TIMED = 'Aracı kuruluştan gelen müşteri bilgileri derhal müşteri dosyasına eklenir.'
OTHER_ACT = 'Aracı kuruluşun, talep edildiğinde belge örneklerini derhal göndereceğinden emin olunur.'
BOTH = ('action', 'deadline_1')


def made(unit, subject, act):
    """(duty payload, labelled quantities) of a synthetic duty, as analyze() hands them to the v19 judge."""
    value = candidate(unit, subject, 'MUST', act).model_dump(mode='json')
    structure = structure_of(unit, value, 0)
    return duty_payload(value, structure), precheck.labelled_quantities(structure)


def now():
    return made(NOW_UNIT, 'Yükümlüler', NOW_ACT)


def lists(relation, text, covered, missing=()):
    return {**verdict(relation=relation, support=text if relation in ('SUPPORTS', 'PARTIAL') else ''),
            'covered_elements': list(covered), 'missing_elements': list(missing)}


def by_text(answers, default=None):
    """A fake's answer per passage text (an exception is raised by the fake)."""
    return lambda text: answers.get(text, default)


def gate(judge, payload, quantities, texts):
    """passages_v19 and the engine's aggregation (strong gate on): (relations, results, (coverage, reason, control, flags))."""
    checks, _, results, _, _ = passages_v19(judge, payload, rows(*texts), frozenset(), quantities, 1, 's', 'o', False)
    canonical_bytes(results)                                     # packets forbid floats: the records must be canonical
    outcome = coverage_of_v19(checks, results, frozenset(), quantities, payload.get('elements'), strong_gate=True)
    return {c.source_id: c.relation for c in checks}, {r['source_id']: r for r in results}, outcome


def asked(judge):
    """(passage, question key) of every verifier call, in order."""
    keys = {text: key for key, text in QUESTIONS.items()}
    return [(c['passage'], keys.get(c['payload'].get('question'), c['payload'].get('question'))) for c in judge.kinds('verify')]


class TimingReadTests(unittest.TestCase):
    def test_the_timing_kinds_a_text_states(self):
        for text, kinds in (('Bilgiler derhal eklenir.', {'IMMEDIATE'}), ('Derhal eklenir.', {'IMMEDIATE'}), ('Gecikmeksizin iletilir.', {'IMMEDIATE'}),
                            ('Bilgiler derhâl eklenir.', {'IMMEDIATE'}), ('İşlem anında kaydedilir.', {'IMMEDIATE'}),
                            ('En kısa sürede iletilir.', {'IMMEDIATE'}), ('Hesap açılmadan önce doğrulanır.', {'BEFORE'}),
                            ('Hesap açılışı öncesinde doğrulanır.', {'BEFORE'}), ('Hesap açılmadan evvel doğrulanır.', {'BEFORE'}),
                            ('Adres önceden doğrulanır.', {'BEFORE'}), ('Adres doğrulanmadan, hesap açılmaz.', {'BEFORE'}),
                            ('Bilgiler beklemeden iletilir.', {'IMMEDIATE'}), ('Ay sonunda toplu olarak iletilir.', {'PERIOD_END'}),
                            ('En geç iki iş günü içinde iletilir.', set()), (UNTIMED, set()), ('Öncelikle risk değerlendirilir.', set()),
                            ('Bu durumda hesap açılmaz.', set()), ('Immediately reported to the authority.', {'IMMEDIATE'}),
                            ('Identity is verified before the account is opened.', {'BEFORE'})):
            with self.subTest(text=text):
                self.assertEqual(stated_timings(text), kinds)
        _, quantities = now()
        self.assertEqual(timing_match(quantities, TIMED), {'deadline_1': TIMING_STATED})
        self.assertEqual(timing_match(quantities, UNTIMED), {'deadline_1': TIMING_NOT_STATED})
        _, before = made(BEFORE_UNIT, 'Şirket', BEFORE_ACT)
        _, end = made(END_UNIT, 'Yükümlüler', END_ACT)
        self.assertEqual([(q['element'], q['kind']) for q in before + end], [('deadline_1', 'BEFORE'), ('deadline_1', 'PERIOD_END')])
        # An immediate timing meets a period end (it is sooner), never a "before"; a number never meets a timing without one.
        self.assertEqual(timing_match(end, 'Rapor derhal sunulur.'), {'deadline_1': TIMING_STATED})
        self.assertEqual(timing_match(before, 'Adres derhal doğrulanır.'), {'deadline_1': TIMING_NOT_STATED})
        self.assertEqual(timing_match(quantities, 'Bilgiler en geç iki iş günü içinde eklenir.'), {'deadline_1': TIMING_NOT_STATED})

    def test_a_timing_the_verifier_lists_for_a_sentence_without_one_is_partial(self):
        payload, quantities = now()
        judge = V19(fast=fast('SUPPORTS', UNTIMED, covered=BOTH), verify=lists('SUPPORTS', UNTIMED, ['action', 'subject', 'deadline_1']))
        relations, results, (coverage, reason, _, flags) = gate(judge, payload, quantities, [UNTIMED])
        # The passage keeps the verifier's reading; the count reads the timing in the text.
        self.assertEqual((relations['p1'], results['p1']['covered_elements'], results['p1']['timing_match']),
                         ('SUPPORTS', ['action', 'subject', 'deadline_1'], {'deadline_1': 'NOT_STATED'}))
        self.assertEqual((coverage, flags), ('PARTIAL', []))
        self.assertIn('TIMING_NOT_STATED', reason)
        self.assertIn(TIMING_NOT_IN_TEXT, reason)
        self.assertEqual(len(judge.kinds('verify')), 1)

    def test_the_same_act_with_the_timing_in_its_text_covers(self):
        cases = ((now(), [TIMED, TIMED.replace('Aracı kuruluştan gelen müşteri bilgileri derhal', 'Derhal, aracı kuruluştan gelen müşteri bilgileri'),
                          TIMED.replace('derhal', 'gecikmeksizin'), TIMED.replace('derhal', 'derhâl'), TIMED.replace('derhal', 'en kısa sürede')]),
                 (made(BEFORE_UNIT, 'Şirket', BEFORE_ACT), ['Müşterinin adres bilgisi hesap açılmadan önce doğrulanır.',
                                                           'Müşterinin adres bilgisi hesap açılışı öncesinde doğrulanır.']),
                 (made(END_UNIT, 'Yükümlüler', END_ACT), ['Risk değerlendirme raporu dönem sonunda yönetim kuruluna sunulur.',
                                                         'Risk değerlendirme raporu derhal yönetim kuruluna sunulur.']))
        for (payload, quantities), texts in cases:
            for text in texts:
                with self.subTest(text=text):
                    judge = V19(fast=fast('SUPPORTS', text, covered=BOTH), verify=lists('SUPPORTS', text, BOTH))
                    relations, results, (coverage, _, _, flags) = gate(judge, payload, quantities, [text])
                    self.assertEqual((relations['p1'], results['p1']['covers_confirmation'], coverage, flags),
                                     ('SUPPORTS', 'CONFIRMED', 'COVERS_TEXT', []))

    def test_another_timing_kind_or_a_numbered_deadline_does_not_meet_it(self):
        for (payload, quantities), text in ((made(BEFORE_UNIT, 'Şirket', BEFORE_ACT), 'Müşterinin adres bilgisi derhal doğrulanır.'),
                                            (now(), 'Aracı kuruluştan gelen müşteri bilgileri en geç iki iş günü içinde müşteri dosyasına eklenir.')):
            with self.subTest(text=text):
                judge = V19(fast=fast('SUPPORTS', text, covered=BOTH), verify=lists('SUPPORTS', text, BOTH))
                _, _, (coverage, reason, _, _) = gate(judge, payload, quantities, [text])
                self.assertEqual(coverage, 'PARTIAL')
                self.assertIn(TIMING_NOT_IN_TEXT, reason)

    def test_the_text_never_adds_a_timing_no_reading_lists(self):
        payload, quantities = now()
        judge = V19(fast=fast('SUPPORTS', TIMED, covered=BOTH), verify=lists('SUPPORTS', TIMED, ['action'], ['deadline_1']))
        relations, results, (coverage, reason, _, _) = gate(judge, payload, quantities, [TIMED])
        self.assertEqual((relations['p1'], results['p1']['timing_match'], coverage), ('PARTIAL', {'deadline_1': 'STATED'}, 'PARTIAL'))
        self.assertNotIn(TIMING_NOT_IN_TEXT, reason)

    def test_a_timing_stated_for_another_act_in_another_passage_does_not_complete_the_cover(self):
        # The act without its timing, and "derhal" in a passage about another act that the fast reading listed as the timing.
        payload, quantities = now()
        judge = V19(fast=by_text({UNTIMED: fast('SUPPORTS', UNTIMED, covered=BOTH), OTHER_ACT: fast('PARTIAL', OTHER_ACT, covered=BOTH)}),
                    verify=by_text({UNTIMED: lists('SUPPORTS', UNTIMED, BOTH)}))
        relations, results, (coverage, reason, _, _) = gate(judge, payload, quantities, [UNTIMED, OTHER_ACT])
        self.assertEqual(asked(judge), [(UNTIMED, 'CONFIRM_COVERS')])       # the other act's passage is never asked to complete it
        self.assertEqual((relations, coverage), ({'p1': 'SUPPORTS', 'p2': 'PARTIAL'}, 'PARTIAL'))
        self.assertIn('TIMING_NOT_STATED', reason)

    def test_the_timing_reading_is_digest_safe_and_only_for_a_timed_duty(self):
        payload, quantities = now()
        judge = V19(fast=fast('SUPPORTS', TIMED, covered=BOTH), verify=lists('SUPPORTS', TIMED, BOTH))
        _, results, _ = gate(judge, payload, quantities, [TIMED])
        self.assertEqual(json.loads(canonical_bytes(results['p1']['timing_match'])), {'deadline_1': 'STATED'})
        plain = {**payload, 'elements': [e for e in payload['elements'] if e['kind'] != 'deadline']}
        judge = V19(fast=fast('SUPPORTS', TIMED, covered=('action',)), verify=lists('SUPPORTS', TIMED, ['action']))
        _, results, (coverage, _, _, _) = gate(judge, plain, [], [TIMED])
        self.assertNotIn('timing_match', results['p1'])
        self.assertEqual(coverage, 'COVERS_TEXT')
        # A record not made by the reader (no timing_match) never meets a timing.
        self.assertFalse(engine.text_timed({'covered_elements': ['deadline_1']}, 'deadline_1'))


class TimingConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.payload, self.quantities = now()
        # p1 is read by the verifier (a possible conflict) and lists the timing its text does not state; p2 lists it and
        # does not state it either; p3 states it. Only p3 can make the timing verified.
        self.untimed_too = 'Aracı kuruluştan temin edilen müşteri bilgileri müşteri dosyasına eklenir.'
        self.fast = by_text({UNTIMED: fast('POSSIBLE_CONFLICT', UNTIMED), self.untimed_too: fast('SUPPORTS', self.untimed_too, covered=BOTH),
                             TIMED: fast('SUPPORTS', TIMED, covered=BOTH)})

    def run_with(self, third):
        judge = V19(fast=self.fast, verify=by_text({UNTIMED: lists('SUPPORTS', UNTIMED, BOTH), TIMED: third}))
        return judge, gate(judge, self.payload, self.quantities, [UNTIMED, self.untimed_too, TIMED])

    def test_the_confirmation_goes_to_the_passage_whose_text_states_the_timing(self):
        judge, (relations, results, (coverage, _, _, flags)) = self.run_with(lists('SUPPORTS', TIMED, BOTH))
        self.assertEqual(asked(judge), [(UNTIMED, 'FAST_POSSIBLE_CONFLICT'), (TIMED, 'CONFIRM_COVERS')])
        self.assertEqual((results['p3']['covers_confirmation'], coverage, flags), ('CONFIRMED', 'COVERS_TEXT', []))

    def test_a_rejected_or_failed_timing_confirmation_leaves_the_verified_partial(self):
        for third, relation in ((lists('UNRELATED', TIMED, [], ['action']), 'UNRELATED'), (ProviderFailure('down'), 'SUPPORTS')):
            with self.subTest(third=type(third).__name__):
                judge, (relations, _, (coverage, reason, _, flags)) = self.run_with(third)
                self.assertEqual([text for text, _ in asked(judge)][:2], [UNTIMED, TIMED])
                # The verified reading of p1 states the act without its timing: a gap a reading states, never COVERS_TEXT.
                self.assertEqual((relations['p3'], coverage, flags), (relation, 'PARTIAL', []))
                self.assertIn('TIMING_NOT_STATED', reason)

    def test_no_call_is_spent_when_no_passage_of_the_cover_states_the_timing(self):
        judge = V19(fast=self.fast, verify=by_text({UNTIMED: lists('SUPPORTS', UNTIMED, BOTH)}))
        relations, _, (coverage, _, _, _) = gate(judge, self.payload, self.quantities, [UNTIMED, self.untimed_too])
        self.assertEqual(asked(judge), [(UNTIMED, 'FAST_POSSIBLE_CONFLICT')])
        self.assertEqual((relations, coverage), ({'p1': 'SUPPORTS', 'p2': 'SUPPORTS'}, 'PARTIAL'))


class FailureTests(unittest.TestCase):
    def test_a_failed_covers_confirmation_with_nothing_verified_is_unknown_never_partial(self):
        payload, quantities = now()
        judge = V19(fast=fast('SUPPORTS', TIMED, covered=BOTH), verify=failure('runaway', 'output_truncated'))
        relations, results, (coverage, reason, _, flags) = gate(judge, payload, quantities, [TIMED])
        self.assertEqual(len(judge.kinds('verify')), 2)                     # asked once more (V19_RETRY_CODES), then failed
        self.assertEqual((relations['p1'], results['p1']['covers_confirmation']), ('SUPPORTS', 'FAILED'))
        self.assertEqual((coverage, flags), ('UNKNOWN', ['COVERS_UNCONFIRMED', VERIFIER_FAILED_FLAG]))
        self.assertTrue(reason.startswith('VERIFIER_FAILED: ') and 'OUTPUT_TRUNCATED' in reason and 'not evidence of a gap' in reason)

    def test_a_rejected_passage_beside_a_failed_confirmation_is_unknown(self):
        # Two fast supports: the verifier rejects the first, the confirmation of the second fails; a third reads part of it.
        payload, quantities = now()
        second = TIMED.replace('gelen', 'iletilen')
        part = 'Aracı kuruluştan alınan bilgiler için müşteri dosyası açılır.'
        judge = V19(fast=by_text({TIMED: fast('SUPPORTS', TIMED, covered=BOTH), second: fast('SUPPORTS', second, covered=BOTH),
                                  part: fast('PARTIAL', part, covered=('action',), missing=('deadline_1',))}),
                    verify=by_text({TIMED: lists('UNRELATED', TIMED, [], ['action']), second: ProviderFailure('down')}))
        relations, _, (coverage, reason, _, flags) = gate(judge, payload, quantities, [TIMED, second, part])
        self.assertEqual([key for _, key in asked(judge)], ['CONFIRM_COVERS', 'CONFIRM_COVERS'])
        self.assertEqual(relations, {'p1': 'UNRELATED', 'p2': 'SUPPORTS', 'p3': 'PARTIAL'})
        self.assertEqual((coverage, flags), ('UNKNOWN', ['COVERS_UNCONFIRMED', VERIFIER_FAILED_FLAG]))

    def test_a_verified_partial_beside_a_failed_confirmation_stays_partial(self):
        payload, quantities = now()
        judge = V19(fast=by_text({UNTIMED: fast('POSSIBLE_CONFLICT', UNTIMED), TIMED: fast('SUPPORTS', TIMED, covered=BOTH)}),
                    verify=by_text({UNTIMED: lists('PARTIAL', UNTIMED, ['action'], ['deadline_1']), TIMED: ProviderFailure('down')}))
        relations, results, (coverage, _, _, flags) = gate(judge, payload, quantities, [UNTIMED, TIMED])
        self.assertEqual((relations, results['p2']['covers_confirmation']), ({'p1': 'PARTIAL', 'p2': 'SUPPORTS'}, 'FAILED'))
        self.assertEqual((coverage, flags), ('PARTIAL', ['COVERS_UNCONFIRMED']))      # the verified reading states the gap

    def test_a_failed_partial_confirmation_keeps_the_fast_partial_its_reading_states(self):
        payload, quantities = now()
        judge = V19(fast=fast('PARTIAL', UNTIMED, covered=('action',), missing=('deadline_1',)), verify=ProviderFailure('down'))
        relations, results, (coverage, _, _, flags) = gate(judge, payload, quantities, [UNTIMED])
        self.assertEqual((relations['p1'], results['p1']['partial_confirmation'], coverage, flags),
                         ('PARTIAL', 'FAILED', 'PARTIAL', ['PARTIAL_UNCONFIRMED']))

    def test_the_failure_hook_names_the_codes_of_the_failed_confirmation(self):
        failed = {'source_id': 'p1', 'relation': 'SUPPORTS', 'covered_elements': list(BOTH), 'missing_elements': [], 'fast': {'label': 'POSSIBLE_SUPPORT'},
                  'timing_match': {'deadline_1': 'STATED'}, 'covers_confirmation': 'FAILED',
                  'confirmation': {'previous_relation': 'SUPPORTS', 'previous_escalation': 'NONE', 'outcome': 'FAILED'},
                  'notes': [{'question': 'verify', 'code': 'VERIFIER_RUNAWAY', 'failure_code': 'OUTPUT_TRUNCATED_TWICE'}]}
        self.assertTrue(confirmation_failed(failed))
        self.assertFalse(confirmation_failed({'confirmation': {'outcome': 'CONFIRMED'}}))
        self.assertEqual(failure_codes([failed]), [])                        # not a failure note (yet)
        payload, quantities = now()
        checks = [engine.PolicyCheck(source_id='p1', quote=TIMED, relation='SUPPORTS')]
        with patch.object(engine, 'VERIFIER_FAILURE_NOTES', ('VERIFIER_FAILED', 'VERIFIER_RUNAWAY')):
            coverage, reason, _, flags = coverage_of_v19(checks, [failed], frozenset(), quantities, payload['elements'], strong_gate=True)
        self.assertEqual((coverage, flags), ('UNKNOWN', ['COVERS_UNCONFIRMED', 'VERIFIER_FAILED']))
        self.assertIn('OUTPUT_TRUNCATED_TWICE', reason)


class BoundaryTests(unittest.TestCase):
    def test_an_unrepeated_condition_or_exception_is_no_gap(self):
        payload, quantities = now()
        payload = {**payload, 'elements': [*payload['elements'], {'id': 'condition_1', 'kind': 'condition', 'text': 'talep halinde'},
                                           {'id': 'exception_1', 'kind': 'exception', 'text': 'Kurumsal müşteriler hariç.'}]}
        judge = V19(fast=fast('SUPPORTS', TIMED, covered=BOTH, missing=('condition_1', 'exception_1')),
                    verify=lists('SUPPORTS', TIMED, BOTH, ['condition_1', 'exception_1']))
        relations, _, (coverage, _, _, _) = gate(judge, payload, quantities, [TIMED])
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'COVERS_TEXT'))

    def test_a_verifier_partial_is_never_raised_by_the_code(self):
        # Every listed critical element covered, the verifier's PARTIAL rests on what no element names (a named recipient,
        # unenumerated items): it stands.
        payload, quantities = now()
        judge = V19(fast=fast('SUPPORTS', TIMED, covered=BOTH), verify=lists('PARTIAL', TIMED, BOTH))
        relations, _, (coverage, _, _, _) = gate(judge, payload, quantities, [TIMED])
        self.assertEqual((relations['p1'], coverage), ('PARTIAL', 'PARTIAL'))

    def test_the_obliged_party_in_other_words_is_no_gap(self):
        payload, quantities = now()
        text = 'Şirketimiz, aracı kuruluştan gelen müşteri bilgilerini derhal müşteri dosyasına ekler.'
        judge = V19(fast=fast('SUPPORTS', text, covered=BOTH), verify=lists('SUPPORTS', text, BOTH, ['subject']))
        relations, results, (coverage, _, _, _) = gate(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'COVERS_TEXT'))
        self.assertIn('ROLE_EQUIVALENT_SUBJECT', [n['code'] for n in results['p1']['notes']])


if __name__ == '__main__':
    unittest.main()

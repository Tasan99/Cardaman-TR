"""v0.19 short round 3: a stricter number is not a conflict (problem 2) and a PARTIAL is confirmed (problem 4).

Measured on the t3 evidence runs (25 September 2026):
- I02 md. 8: "on yıl süreyle saklanır" against "sekiz yıl süreyle muhafaza" was called a THRESHOLD_MISMATCH by the
  verifier and the row became CONFLICT (label COVERS_TEXT). A deadline, period or amount the contradicting
  sentence states the same or stricter is now STRICTER_THAN_REQUIRED, never a conflict; a weaker one stays one.
- C03 md. 8(1), 9(1): a rule for registered companies was PARTIAL against a duty about associations and unions;
  I01 md. 3(1), I04 md. 8, I09 md. 18 and C12 md. 28(2) were decided by a fast PARTIAL no strong model read.
  The fast labels only propose now (POSSIBLE_SUPPORT / POSSIBLE_PARTIAL); the passages a PARTIAL rests on are
  confirmed by the strong verifier element by element (CONFIRM_PARTIAL), within the per-obligation cap shared
  with CONFIRM_COVERS, and never twice for one passage.

Every model is a scripted fake (no Ollama).
"""
import json
import unittest

from regchain.evidence import canonical_bytes
from regchain.extraction.quantities import compare, parse_quantities
from regchain.extraction.structure import SCOPE_KINDS, critical_ids, duty_payload, structure_of
from regchain.pilot import conflict as precheck
from regchain.pilot.engine import (CONFIRMATIONS, QUESTIONS, VERIFY_PROMPT, coverage_of_v19, escalation_of, fast_schema, parse_fast,
                                   passages_v19)
from regchain.pilot.schema import FAST_LABELS, ConflictVerdict, FastJudgement
from test_v019_conflict_pipeline import MONTH_END, NOISE, V19, candidate, duty, fast, md28, verdict
from test_v019_covers_gate import KEEP, law8

TEN_YEARS = KEEP.replace('sekiz yıl', 'on yıl')
FIVE_YEARS = KEEP.replace('sekiz yıl', 'beş yıl')
REPORT_FIVE = "Şüpheli işlemler, şüphenin oluştuğu tarihten itibaren en geç beş gün içinde MASAK'a bildirilir."
REPORT_TEN = "Şüpheli işlemler, şüphenin oluştuğu tarihten itibaren en geç on iş günü içinde MASAK'a bildirilir."
REPORT_BARE = "Şüpheli görülen işlemler uyum birimi aracılığıyla MASAK'a bildirilir."
SAME_DAY_THEN_MONTH_END = 'Personel, şüpheli bulduğu işlemi aynı gün içinde uyum birimine iletir. ' + MONTH_END
FLOOR_UNIT = '(1) Yükümlüler, tutarı yüzseksenbeşbin TL veya üzerinde olan işlemlerde müşterinin kimliğini tespit etmek zorundadır.'
CAP_UNIT = '(2) Yükümlüler, bir müşteriden en fazla elli bin TL nakit tahsilat kabul ederler.'


def q(text):
    return parse_quantities(text)[0]


def made(unit_text, subject, modality, action):
    value = candidate(unit_text, subject, modality, action).model_dump(mode='json')
    structure = structure_of(unit_text, value, 0)
    return duty_payload(value, structure), precheck.labelled_quantities(structure)


def floor_duty():
    return made(FLOOR_UNIT, 'Yükümlüler', 'MUST', FLOOR_UNIT[FLOOR_UNIT.index('tutarı'):FLOOR_UNIT.index(' zorundadır')])


def cap_duty():
    return made(CAP_UNIT, 'Yükümlüler', 'MUST', CAP_UNIT[CAP_UNIT.index('bir müşteriden'):CAP_UNIT.index('.')])


def associations():
    """Tedbirler md. 8(1): the identification of associations (subject 'Dernekler'), the C03 duty."""
    return duty('8', 1, 'Dernekler', 'MUST', lambda t: t[t.index('kimlik tespitinde'):t.index(' alınır') + len(' alınır')])


def rows(*texts):
    return [{'source_id': f'p{i}', 'text': text} for i, text in enumerate(texts, 1)]


def run(judge, payload, quantities, texts):
    """(relations, results by id, gated coverage outcome): the engine's own flow and aggregation (strong_gate)."""
    checks, _, results, _, _ = passages_v19(judge, payload, rows(*texts), frozenset(), quantities, 1, 's', 'o', False)
    canonical_bytes(results)                                          # packets forbid floats
    outcome = coverage_of_v19(checks, results, frozenset(), quantities, payload.get('elements'), strong_gate=True)
    return {c.source_id: c.relation for c in checks}, {r['source_id']: r for r in results}, outcome


def lists(answer, covered, missing=()):
    return {**answer, 'covered_elements': list(covered), 'missing_elements': list(missing)}


class QuantityDirectionTests(unittest.TestCase):
    def test_compare_reads_min_max_floor_and_cap_the_right_way(self):
        self.assertEqual(compare(q('en geç 10 gün içinde'), q('5 gün'), 'max'), 'STRICTER')
        self.assertEqual(compare(q('en geç 10 gün içinde'), q('15 gün'), 'max'), 'WEAKER')
        self.assertEqual(compare(q('en az 8 yıl'), q('10 yıl'), 'min'), 'STRICTER')
        self.assertEqual(compare(q('8 yıl'), q('5 yıl'), 'min'), 'WEAKER')
        self.assertEqual(compare(q('185.000 TL ve üzeri'), q('100.000 TL ve üzeri'), 'floor'), 'STRICTER')
        self.assertEqual(compare(q('185.000 TL ve üzeri'), q('250.000 TL ve üzeri'), 'floor'), 'WEAKER')
        self.assertEqual(compare(q('en fazla 1.000 TL'), q('500 TL'), 'cap'), 'STRICTER')
        self.assertEqual(compare(q('en fazla 1.000 TL'), q('2.000 TL'), 'cap'), 'WEAKER')
        self.assertEqual(compare(q('en fazla 1.000 TL'), q('500 TL'), 'max'), 'STRICTER')         # 'cap' is 'max' for amounts

    def test_at_the_same_number_the_comparators_decide(self):
        self.assertEqual(compare(q('185.000 TL ve üzeri'), q("185.000 TL'yi aşan"), 'floor'), 'WEAKER')    # misses the amount itself
        self.assertEqual(compare(q("185.000 TL'yi aşan"), q('185.000 TL ve üzeri'), 'floor'), 'STRICTER')
        self.assertEqual(compare(q('en az 8 yıl'), q('8 yıl'), 'min'), 'SAME')
        self.assertEqual(compare(q('en fazla 1.000 TL'), q("1.000 TL'nin altında"), 'cap'), 'STRICTER')     # below the cap itself
        self.assertEqual(compare(q("1.000 TL'nin altında"), q('en fazla 1.000 TL'), 'cap'), 'WEAKER')


class StricterIsNotConflictTests(unittest.TestCase):
    def test_eight_years_required_and_ten_kept_is_stricter_than_required_not_a_conflict(self):
        payload, quantities = law8()
        span = 'on yıl süreyle muhafaza edilir'
        judge = V19(fast=fast('SUPPORTS', TEN_YEARS, covered=('act_1', 'act_2', 'deadline_1')),
                    verify=lists(verdict(True, 'THRESHOLD_MISMATCH', span, 'PARTIAL'), ['act_1'], ['deadline_1']))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [TEN_YEARS])
        record = results['p1']
        self.assertEqual(record['quantity_match'], {'deadline_1': 'STRICTER'})
        self.assertNotEqual(relations['p1'], 'CONFLICTS')
        self.assertNotEqual(coverage, 'CONFLICT')
        self.assertEqual((record['verifier']['contradiction_type'], record['verifier']['claimed_contradiction_type']),
                         ('STRICTER_THAN_REQUIRED', 'THRESHOLD_MISMATCH'))
        self.assertEqual(record['verifier']['stricter_than_required'], {'deadline_1': 'STRICTER'})
        self.assertIn('deadline_1', record['covered_elements'])
        self.assertNotIn('deadline_1', record['missing_elements'])
        self.assertIn('STRICTER_THAN_REQUIRED', [n['code'] for n in record['notes']])
        # The verifier listed act_2 neither covered nor missing and the passage states it: with the claim rejected the
        # passage is PARTIAL (act_2 unlisted) - never a conflict. With act_2 listed covered it is the SUPPORTS it describes.
        self.assertEqual(relations['p1'], 'PARTIAL')
        judge = V19(fast=fast('SUPPORTS', TEN_YEARS, covered=('act_1', 'act_2', 'deadline_1')),
                    verify=lists(verdict(True, 'THRESHOLD_MISMATCH', span, 'PARTIAL'), ['act_1', 'act_2'], ['deadline_1']))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [TEN_YEARS])
        self.assertEqual((relations['p1'], coverage, flags), ('SUPPORTS', 'COVERS_TEXT', []))
        self.assertIn('stated the same or stricter', reason)

    def test_a_true_numeric_mismatch_eight_years_against_five_stays_a_conflict(self):
        payload, quantities = law8()
        span = 'beş yıl süreyle muhafaza edilir'
        judge = V19(fast=fast('PARTIAL', FIVE_YEARS, covered=('act_1',), missing=('deadline_1',)),
                    verify=lists(verdict(True, 'DEADLINE_MISMATCH', span, 'NOT_APPLICABLE'), ['deadline_1'], ['act_1', 'act_2']))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [FIVE_YEARS])
        self.assertEqual((relations['p1'], coverage, results['p1']['escalation']), ('CONFLICTS', 'CONFLICT', 'STRUCTURAL_SIGNAL'))
        self.assertEqual(results['p1']['verifier']['contradiction_type'], 'DEADLINE_MISMATCH')
        self.assertEqual(precheck.stricter_than_required(quantities, span), {})

    def test_a_ten_business_day_deadline_met_in_five_days_is_no_conflict(self):
        payload, quantities = md28()
        judge = V19(fast=fast('SUPPORTS', REPORT_FIVE, covered=('action', 'deadline_1')),
                    verify=lists(verdict(True, 'DEADLINE_MISMATCH', REPORT_FIVE, 'SUPPORTS', support=REPORT_FIVE), ['action'], []))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [REPORT_FIVE])
        self.assertEqual(results['p1']['quantity_match'], {'deadline_1': 'STRICTER'})
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'COVERS_TEXT'))
        self.assertEqual(results['p1']['verifier']['contradiction_type'], 'STRICTER_THAN_REQUIRED')
        # A later deadline is still the conflict it is.
        late = REPORT_FIVE.replace('beş gün', 'otuz gün')
        judge = V19(fast=fast('POSSIBLE_CONFLICT', late), verify=verdict(True, 'DEADLINE_MISMATCH', late, 'NOT_APPLICABLE'))
        self.assertEqual(run(judge, payload, quantities, [late])[2][0], 'CONFLICT')

    def test_only_the_quoted_sentence_counts_a_stricter_number_elsewhere_does_not_answer_it(self):
        # C13 md. 28(2): "aynı gün" (stricter) in the first sentence, the month-end batch in the quoted one: a conflict.
        payload, quantities = md28()
        self.assertEqual(precheck.quantity_match(quantities, SAME_DAY_THEN_MONTH_END), {'deadline_1': 'STRICTER'})
        judge = V19(fast=fast('PARTIAL', MONTH_END, missing=('deadline_1',)), verify=verdict(True, 'DEADLINE_MISMATCH', MONTH_END, 'UNRELATED'))
        relations, _, (coverage, _, _, _) = run(judge, payload, quantities, [SAME_DAY_THEN_MONTH_END])
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        # A sentence stating a stricter and a weaker number is not answered by the stricter one.
        self.assertEqual(precheck.stricter_than_required(quantities, 'en geç beş gün içinde, en geç otuz gün içinde bildirilir'), {})

    def test_a_minimum_threshold_set_lower_is_stricter(self):
        payload, quantities = floor_duty()
        self.assertEqual([(x['element'], x['direction']) for x in quantities], [('threshold_1', 'floor')])
        text = "Tutarı 100.000 TL ve üzerinde olan işlemlerde müşterinin kimliği tespit edilir."
        judge = V19(fast=fast('SUPPORTS', text, covered=('action', 'threshold_1')),
                    verify=lists(verdict(True, 'THRESHOLD_MISMATCH', text, 'SUPPORTS', support=text), ['action', 'threshold_1'], []))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage, results['p1']['verifier']['contradiction_type']), ('SUPPORTS', 'COVERS_TEXT', 'STRICTER_THAN_REQUIRED'))
        higher = text.replace('100.000', '250.000')
        judge = V19(fast=fast('POSSIBLE_CONFLICT', higher), verify=verdict(True, 'THRESHOLD_MISMATCH', higher, 'NOT_APPLICABLE'))
        self.assertEqual(run(judge, payload, quantities, [higher])[2][0], 'CONFLICT')

    def test_a_maximum_threshold_set_lower_is_stricter(self):
        payload, quantities = cap_duty()
        self.assertEqual([(x['element'], x['direction'], x['comparator']) for x in quantities], [('threshold_1', 'max', '<=')])
        self.assertEqual(precheck.stricter_than_required(quantities, 'Bir müşteriden en fazla yirmi bin TL nakit tahsilat kabul edilir.'),
                         {'threshold_1': 'STRICTER'})
        self.assertEqual(precheck.stricter_than_required(quantities, 'Bir müşteriden en fazla yüz bin TL nakit tahsilat kabul edilir.'), {})

    def test_the_precheck_never_escalates_a_stricter_number_on_its_own(self):
        payload, quantities = law8()
        signals, overlap, matched = (precheck.structural_signals(payload, TEN_YEARS, quantities), precheck.topic_overlap(payload, TEN_YEARS),
                                     precheck.quantity_match(quantities, TEN_YEARS))
        self.assertEqual([(s['type'], s['strength']) for s in signals], [('STRICTER_QUANTITY', 'weak')])
        answer = FastJudgement(label='POSSIBLE_SUPPORT', quote=TEN_YEARS, covered_elements=['act_1'], missing_elements=[], reason='r')
        self.assertIsNone(escalation_of(answer, signals, overlap, 0, True, payload))
        self.assertEqual(matched, {'deadline_1': 'STRICTER'})

    def test_the_verifier_is_told_the_deterministic_comparison(self):
        payload, quantities = law8()
        judge = V19(fast=fast('POSSIBLE_CONFLICT', TEN_YEARS), verify=verdict(relation='UNRELATED'))
        run(judge, payload, quantities, [TEN_YEARS])
        hint = judge.kinds('verify')[0]['payload']['quantity_check']
        self.assertEqual(hint, [{'element': 'deadline_1', 'duty_states': 'sekiz yıl süreyle', 'policy_states': 'on yıl', 'comparison': 'STRICTER'}])
        self.assertIn('quantity_check', VERIFY_PROMPT)
        # No number in the passage: no hint.
        judge = V19(fast=fast('POSSIBLE_CONFLICT', REPORT_BARE), verify=verdict(relation='UNRELATED'))
        run(judge, *md28(), [REPORT_BARE])
        self.assertNotIn('quantity_check', judge.kinds('verify')[0]['payload'])

    def test_a_verifier_that_names_stricter_than_required_has_found_no_conflict(self):
        from regchain.pilot.engine import parse_verdict
        raw = json.dumps({**verdict(True, 'STRICTER_THAN_REQUIRED', TEN_YEARS, 'SUPPORTS', support=TEN_YEARS), 'covered_elements': [],
                          'missing_elements': []})
        self.assertFalse(parse_verdict(raw, TEN_YEARS).conflict)
        self.assertIn('STRICTER_THAN_REQUIRED', ConflictVerdict.model_json_schema()['properties']['contradiction_type']['enum'])

    def test_turkish_unless_is_excepting_wording(self):
        text = ('Müşterilerimize ait veriler, mahkeme kararı bulunmadıkça hiçbir kamu kurumuna verilmez; Başkanlığın yazılı talepleri de '
                'bu kurala tabidir.')
        span = text[:text.index(';')]
        self.assertTrue(precheck.type_supported({}, 'EXEMPTION_ADDED', span, text))
        self.assertIn('EXEMPTION', [s['type'] for s in precheck.structural_signals({'modality': 'MUST'}, text)])


class FastLabelTests(unittest.TestCase):
    def test_the_fast_labels_only_propose_and_the_old_names_still_read(self):
        self.assertEqual(FAST_LABELS, ('IRRELEVANT', 'POSSIBLE_SUPPORT', 'POSSIBLE_PARTIAL', 'POSSIBLE_CONFLICT'))
        self.assertEqual(fast_schema(['action'])['properties']['label']['enum'], list(FAST_LABELS))
        for old, new in (('SUPPORTS', 'POSSIBLE_SUPPORT'), ('PARTIAL', 'POSSIBLE_PARTIAL'), ('IRRELEVANT', 'IRRELEVANT'),
                         ('POSSIBLE_CONFLICT', 'POSSIBLE_CONFLICT'), ('POSSIBLE_PARTIAL', 'POSSIBLE_PARTIAL')):
            raw = json.dumps(fast(old, REPORT_BARE if old != 'IRRELEVANT' else ''))
            self.assertEqual(parse_fast(raw, REPORT_BARE, ['action']).label, new)

    def test_a_scope_element_left_unstated_is_no_gap(self):
        # A fast support that does not repeat the subject still covers (confirmed); a missing critical element does not.
        payload, quantities = md28()
        judge = V19(fast=fast('POSSIBLE_SUPPORT', REPORT_TEN, covered=('action', 'deadline_1'), missing=('subject',)),
                    verify=lists(verdict(relation='SUPPORTS', support=REPORT_TEN), ['action', 'deadline_1', 'subject']))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [REPORT_TEN])
        self.assertEqual((results['p1']['fast']['label'], coverage), ('POSSIBLE_SUPPORT', 'COVERS_TEXT'))


class PartialConfirmationTests(unittest.TestCase):
    def test_a_missing_deadline_is_partial_confirmed_by_the_verifier(self):
        payload, quantities = md28()
        judge = V19(fast=fast('POSSIBLE_PARTIAL', REPORT_BARE, covered=('action',), missing=('deadline_1',)),
                    verify=lists(verdict(relation='PARTIAL', support=REPORT_BARE), ['action', 'subject'], ['deadline_1']))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [REPORT_BARE])
        calls = judge.kinds('verify')
        self.assertEqual([(c['task'], c['payload']['question']) for c in calls], [('judge.verify.support', QUESTIONS['CONFIRM_PARTIAL'])])
        self.assertEqual((relations['p1'], coverage, flags, results['p1']['partial_confirmation']), ('PARTIAL', 'PARTIAL', [], 'CONFIRMED'))
        self.assertIn('QUANTITY_NOT_STATED', reason)

    def test_a_missing_threshold_is_partial(self):
        payload, quantities = floor_duty()
        text = 'Müşterinin kimliği işlem öncesinde tespit edilir.'
        judge = V19(fast=fast('POSSIBLE_PARTIAL', text, covered=('action',), missing=('threshold_1',)),
                    verify=lists(verdict(relation='PARTIAL', support=text), ['action'], ['threshold_1']))
        relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage, results['p1']['quantity_match']), ('PARTIAL', 'PARTIAL', {'threshold_1': 'NOT_STATED'}))
        self.assertIn('yüzseksenbeşbin TL veya üzerinde', reason)

    def test_full_coverage_is_covers_text(self):
        payload, quantities = md28()
        judge = V19(fast=fast('POSSIBLE_SUPPORT', REPORT_TEN, covered=('action', 'deadline_1')),
                    verify=lists(verdict(relation='SUPPORTS', support=REPORT_TEN), ['action', 'deadline_1']))
        relations, results, (coverage, _, _, flags) = run(judge, payload, quantities, [REPORT_TEN])
        self.assertEqual((relations['p1'], coverage, flags, results['p1']['covers_confirmation']), ('SUPPORTS', 'COVERS_TEXT', [], 'CONFIRMED'))
        self.assertEqual(len(judge.kinds('verify')), 1)

    def test_a_rule_for_another_customer_group_is_irrelevant_never_partial(self):
        # C03 md. 8(1): the duty identifies associations; the passage identifies registered companies.
        payload, quantities = associations()
        self.assertEqual([(e['id'], e['text']) for e in payload['elements'] if e['kind'] == 'subject'], [('subject', 'Dernekler')])
        text = ('Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde; tüzel kişinin unvanı, ticaret sicil numarası, vergi kimlik '
                'numarası ve açık adresi alınır.')
        judge = V19(fast=fast('POSSIBLE_SUPPORT', text), verify=lists(verdict(relation='PARTIAL', support=text), ['action'], ['subject']))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('OTHER_SUBJECT', [n['code'] for n in results['p1']['notes']])
        self.assertIn('different group', QUESTIONS['CONFIRM_PARTIAL'] + VERIFY_PROMPT)

    def test_no_meaningful_overlap_is_no_evidence(self):
        payload, quantities = md28()
        judge = V19(fast=fast('IRRELEVANT', covered=()))
        relations, _, (coverage, _, _, _) = run(judge, payload, quantities, NOISE[:3])
        self.assertEqual((set(relations.values()), coverage, judge.kinds('verify')), ({'UNRELATED'}, 'NO_EVIDENCE', []))
        # A PARTIAL that states no critical element (the verifier lists only the gap) is not a partial statement.
        judge = V19(fast=fast('POSSIBLE_PARTIAL', REPORT_BARE, covered=('action',), missing=('deadline_1',)),
                    verify=lists(verdict(relation='PARTIAL', support=REPORT_BARE), ['subject'], ['action', 'deadline_1']))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [REPORT_BARE])
        self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertIn('NO_ELEMENT_STATED', [n['code'] for n in results['p1']['notes']])

    def test_no_passage_is_read_twice_by_the_verifier(self):
        payload, quantities = md28()
        # Read once as a possible conflict (no element lists in its answer): its reading is reused, never re-asked.
        judge = V19(fast=fast('POSSIBLE_CONFLICT', REPORT_TEN), verify=verdict(relation='SUPPORTS', support=REPORT_TEN))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [REPORT_TEN])
        self.assertEqual(([c['task'] for c in judge.kinds('verify')], relations['p1']), (['judge.verify'], 'SUPPORTS'))
        self.assertNotIn('confirmation', results['p1'])
        # A failed confirmation is not asked again either (the cap is not spent on the same passage twice).
        from regchain.extraction.providers import ProviderFailure
        judge = V19(fast=fast('POSSIBLE_PARTIAL', REPORT_BARE, covered=('action',), missing=('deadline_1',)), verify=ProviderFailure('down'))
        relations, results, (coverage, _, _, flags) = run(judge, payload, quantities, [REPORT_BARE])
        self.assertEqual(len({c['passage'] for c in judge.kinds('verify')}), 1)
        self.assertEqual((results['p1']['partial_confirmation'], coverage, flags), ('FAILED', 'PARTIAL', ['PARTIAL_UNCONFIRMED']))

    def test_the_cap_two_confirmations_per_obligation_then_review(self):
        payload, quantities = md28()
        texts = [REPORT_BARE, REPORT_BARE.replace('uyum birimi', 'uyum görevlisi'), REPORT_BARE.replace('Şüpheli görülen', 'Şüpheli')]
        judge = V19(fast=lambda text: fast('POSSIBLE_PARTIAL', text, covered=('action',), missing=('deadline_1',)),
                    verify=lambda text: lists(verdict(relation='UNRELATED', rationale='another measure'), [], ['action']))
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, texts)
        self.assertEqual(CONFIRMATIONS, 2)
        self.assertEqual([c['passage'] for c in judge.kinds('verify')], texts[:2])             # best-ranked first, then the next
        self.assertEqual(relations, {'p1': 'UNRELATED', 'p2': 'UNRELATED', 'p3': 'PARTIAL'})
        # The verifier found both it read about something else; the third fast PARTIAL is weaker evidence still.
        self.assertEqual((coverage, flags), ('NO_EVIDENCE', ['PARTIAL_UNCONFIRMED']))
        self.assertIn('listed for review', reason)
        # The cap is shared with the COVERS confirmation: two covers readings leave no PARTIAL reading.
        judge = V19(fast=lambda text: fast('POSSIBLE_SUPPORT', text, covered=('action', 'deadline_1')),
                    verify=lambda text: lists(verdict(relation='PARTIAL', support=text), ['action'], ['deadline_1']))
        run(judge, payload, quantities, [REPORT_TEN, REPORT_TEN.replace('Şüpheli işlemler', 'Şüpheli görülen işlemler'), REPORT_BARE])
        self.assertEqual(len(judge.kinds('verify')), CONFIRMATIONS)


class ScopeElementTests(unittest.TestCase):
    def test_subject_object_prohibition_and_conditions_are_elements_and_never_gaps(self):
        unit = '(4) Yükümlüler, müşterinin talebi halinde, belge ve kayıtları üçüncü kişilere açıklayamazlar.'
        payload, _ = made(unit, 'Yükümlüler', 'MUST_NOT', 'belge ve kayıtları üçüncü kişilere açıklayamazlar')
        kinds = {e['id']: e['kind'] for e in payload['elements']}
        self.assertEqual(kinds, {'action': 'action', 'subject': 'subject', 'object': 'object', 'prohibition': 'prohibition',
                                 'condition_1': 'condition'})
        for element in payload['elements']:
            self.assertIn(element['text'], unit)                                              # exact substrings
        self.assertEqual(critical_ids(payload['elements']), ['action', 'object', 'prohibition'])
        self.assertEqual(SCOPE_KINDS, ('subject', 'condition', 'exception'))
        canonical_bytes(payload)
        # Without a usable structure the legacy element list stands.
        self.assertEqual(duty_payload(candidate(unit, 'Yükümlüler', 'MUST_NOT', 'x').model_dump(mode='json'), None)['elements'],
                         [{'id': 'action', 'kind': 'action', 'text': 'x'}])


if __name__ == '__main__':
    unittest.main()

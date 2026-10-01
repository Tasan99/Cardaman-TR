"""v0.19 t7 (P2): the second reading (review) decides per candidate; one doubted candidate never empties the unit.

In the v019t6 micro run the model gave all three duties of Tedbirler Yönetmeliği md. 24/A(3), the labelled "... iade eder"
one among them. The answer went through a repair, the review answered one of its three decisions UNCERTAIN, and
pipeline.extract dropped the whole unit (SECOND_PASS_UNCERTAIN): the correct duty and the duty t5 had kept were lost, and
the sentence coverage step never ran. Now SUPPORTED keeps a candidate, UNSUPPORTED drops that candidate alone (and the
coverage step does not re-create it), UNCERTAIN keeps it with a low confidence and a review note. The unit ends only when
the review rejects every candidate or its decisions cannot be paired with the candidates, and then its record keeps every
decision and every withheld candidate. An all-SUPPORTED review changes nothing. t7 review 1 (R1): a candidate's fate depends on
its own decision only, so an UNCERTAIN candidate is kept (flagged) whatever its siblings got; a review with no SUPPORTED decision
no longer ends the unit (tests/test_v019_t7_review1.py).
"""
import json
import unittest
from hashlib import sha256
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction import pipeline
from regchain.extraction.grounding import verify
from regchain.extraction.pipeline import (ALL_REJECTED, ALL_UNCERTAIN, MISALIGNED, NO_DECISIONS, NONE_SUPPORTED, PER_CANDIDATE_CODE,
                                          REVIEW_CODE, REVIEW_OUTCOMES, REVIEW_UNCERTAIN_CONFIDENCE, apply_review, extract,
                                          second_reading)
from regchain.extraction.providers import REVIEW_PROMPT, OllamaProvider
from regchain.extraction.retrieval import ContextPacket
from regchain.extraction.schema import ExtractionOutput, SecondPass
from regchain.extraction.spans import COVERED_CODE, REVIEW_REJECTED, SOURCE_BOUND_OK, UNCOVERED_CODE, accepted_trace

from test_v019_extraction import action_of, answer, candidate, unit
from test_v019_t6_alignment import (FULL, PERSISTENT, RECEIVER, RECEIVING, REPORTS, REQUEST, RETURN, SHORT, records, sentence_of,
                                    sentences, trace_of)

MISSING = 'Bilgilerin tamamlanmaması halinde'
# A main duty and the duty that applies only when the main one cannot be met (a supporting, conditional duty).
FALLBACK = ('(4) Yükümlüler, şüpheli işlemleri gecikmeksizin Başkanlığa bildirir. '
            'Bildirimin yapılamaması halinde işlemi askıya alır.')
NOT_MADE = 'Bildirimin yapılamaması halinde'


class Unreviewed:
    """Fixed ExtractionOutput JSON answers, the first and then one per repair turn, with no second reading at all."""
    name, model_version = 'reviewed', 'fixture-v1'

    def __init__(self, decisions, *answers):
        self.decisions, self.answers, self.calls, self.reviewed = decisions, list(answers), 0, []

    def generate(self, text):
        value = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return value

    def repair(self, text, context, feedback):
        return self.generate(text)


class Reviewed(Unreviewed):
    """The same answers and a scripted second reading: `decisions` is what review answers, a list as OllamaProvider gives
    it, or a bool as the t6 test doubles do."""

    def review(self, text, output, context):
        self.reviewed.append([action_of(v) for v in output.obligations])
        return self.decisions


def transfer():
    """Tedbirler md. 24/A(3) and its three duties, one per sentence, in source order (the gate accepts them as they are)."""
    _, text = unit('24/A', 3)
    return text, [candidate(text, RECEIVING, 'MUST', REQUEST), candidate(text, RECEIVING, 'MUST', RETURN, conditions=[MISSING]),
                  candidate(text, RECEIVER, 'MUST', FULL, conditions=[PERSISTENT])]


def one_uncertain(count, position):
    return ['UNCERTAIN' if index == position else 'SUPPORTED' for index in range(count)]


def fields(result):
    return [(v.subject, action_of(v), v.conditions) for v in result.output.obligations]


def scores(result):
    return [v.model_dump(mode='json')['confidence_score'] for v in result.output.obligations]


def comparable(result):
    """A result as bytes, without the wall-clock and call-log fields of its records."""
    diagnostics = [{k: v for k, v in d.items() if k not in ('elapsed_ms', 'calls')} for d in result.diagnostics]
    return canonical_bytes({'output': result.output.model_dump(mode='json'), 'reason': result.reason, 'attempts': result.attempts,
                            'diagnostics': diagnostics})


class Checks(unittest.TestCase):
    def kept(self, text, result):
        """A unit the review let through: legal-review reason, exact source fields, digest-safe records and output."""
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        for value in result.output.obligations:
            for field in [value.subject, action_of(value), *value.conditions, *value.exceptions]:
                self.assertIn(field, text)
        canonical_bytes(list(result.diagnostics))
        canonical_bytes(result.output.model_dump(mode='json'))
        self.assertEqual(records(result, REVIEW_CODE), [])

    def ended(self, result, ending, decisions, withheld):
        """A unit the review ended: no candidate, reason SECOND_PASS_UNCERTAIN, and a terminal record that says why and
        keeps every decision and every withheld candidate."""
        self.assertEqual(result.reason, REVIEW_CODE)
        self.assertEqual(result.output, ExtractionOutput(status='INSUFFICIENT_EVIDENCE'))
        last = result.diagnostics[-1]
        self.assertEqual({k: last[k] for k in ('stage', 'code', 'review', 'decisions', 'candidates')},
                         {'stage': 'review', 'code': REVIEW_CODE, 'review': ending, 'decisions': decisions, 'candidates': len(withheld)})
        self.assertEqual([(w['subject'], w['action'], w['conditions']) for w in last['withheld']],
                         [(v.subject, action_of(v), v.conditions) for v in withheld])
        self.assertEqual([d for d in result.diagnostics if d.get('stage') == 'complete'], [])
        canonical_bytes(list(result.diagnostics))


class CandidateReviewTests(Checks):
    def test_one_uncertain_candidate_of_three_keeps_the_other_two_and_itself_flagged(self):
        text, three = transfer()
        for position in range(3):
            with self.subTest(uncertain=position):
                decisions = one_uncertain(3, position)
                result = extract(text, Reviewed(decisions, answer(*three)))
                self.kept(text, result)
                self.assertEqual(fields(result), [(v.subject, action_of(v), v.conditions) for v in three])
                self.assertEqual(scores(result), ['0.2500' if i == position else '0.5000' for i in range(3)])
                self.assertEqual(records(result, PER_CANDIDATE_CODE),
                                 [{'attempt': 1, 'stage': 'review', 'code': PER_CANDIDATE_CODE, 'decisions': decisions,
                                   'uncertain': [position], 'rejected': []}])
                self.assertEqual([entry.get('review') for entry in trace_of(result)], ['UNCERTAIN' if i == position else None for i in range(3)])
                verify(text, result.output)                                     # nothing dropped: the gate still accepts it
                # Each duty is still located in its own sentence, and the coverage step has nothing to add.
                self.assertEqual([sentence_of(text, v) for v in result.output.obligations],
                                 [sentences(text)[0][len('(3) '):], *sentences(text)[1:]])
                self.assertEqual(records(result, COVERED_CODE) + records(result, UNCOVERED_CODE), [])

    def test_a_single_uncertain_candidate_is_kept_flagged_and_only_rejecting_every_one_ends_the_unit(self):
        # t7 review 1 (R1): this test pinned the P2 unit-level endings REVIEW_ALL_UNCERTAIN and REVIEW_NONE_SUPPORTED, under which
        # an UNCERTAIN candidate was kept beside a SUPPORTED sibling and withheld beside a rejected or another doubted one (the
        # user's blocker "valid candidate lost due to an unrelated candidate's review"). Now UNCERTAIN always keeps its candidate,
        # flagged (0.2500, trace 'review'); only a review that rejects every candidate ends the unit, on record.
        _, text = unit('24/A', 3)
        only = candidate(text, RECEIVER, 'MUST', FULL, conditions=[PERSISTENT])
        result = extract(text, Reviewed(['UNCERTAIN'], answer(only)))
        self.kept(text, result)
        # The doubted candidate stays first, flagged; the sentence coverage step still fills the unit's other duty sentences.
        self.assertEqual((fields(result)[0], scores(result)[0]), ((RECEIVER, FULL, [PERSISTENT]), '0.2500'))
        self.assertEqual(pipeline.review_uncertain(result), [0])
        english = 'A firm must retain records.'
        value = candidate(english, 'A firm', 'MUST', 'retain records.')
        result = extract(english, Reviewed(['UNCERTAIN'], answer(value)))
        self.assertEqual((result.reason, scores(result)), ('CANDIDATE_REQUIRES_LEGAL_REVIEW', ['0.2500']))
        text, three = transfer()
        for decisions in (['UNCERTAIN'] * 3, ['UNCERTAIN', 'UNSUPPORTED', 'UNCERTAIN']):
            with self.subTest(decisions=decisions):
                result = extract(text, Reviewed(decisions, answer(*three)))
                self.kept(text, result)
                self.assertEqual([action_of(v) for v in result.output.obligations],
                                 [action_of(v) for v, d in zip(three, decisions) if d != 'UNSUPPORTED'])
                self.assertEqual(set(scores(result)), {'0.2500'})
        self.ended(extract(text, Reviewed(['UNSUPPORTED'] * 3, answer(*three))), ALL_REJECTED, ['UNSUPPORTED'] * 3, three)

    def test_an_unsupported_candidate_drops_alone_and_its_sentence_is_not_filled_again(self):
        text, three = transfer()
        first, second, third = sentences(text)
        result = extract(text, Reviewed(['SUPPORTED', 'UNSUPPORTED', 'SUPPORTED'], answer(*three)))
        self.kept(text, result)
        self.assertEqual(fields(result), [(RECEIVING, REQUEST, []), (RECEIVER, FULL, [PERSISTENT])])
        self.assertEqual(scores(result), ['0.5000', '0.5000'])
        [record] = records(result, PER_CANDIDATE_CODE)
        self.assertEqual((record['uncertain'], record['rejected']), ([], [{'candidate': 1, 'action': RETURN}]))
        self.assertEqual(trace_of(result), [{'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'exact'},
                                            {'candidate': 1, 'code': SOURCE_BOUND_OK, 'tier': 'exact', 'review': 'UNSUPPORTED'},
                                            {'candidate': 2, 'code': SOURCE_BOUND_OK, 'tier': 'exact'}])
        # The coverage step leaves the rejected duty's sentence without a duty, and says why.
        self.assertEqual(records(result, UNCOVERED_CODE)[0]['uncovered'], [{'sentence': 1, 'detail': REVIEW_REJECTED}])
        self.assertEqual([sentence_of(text, v) for v in result.output.obligations], [first[len('(3) '):], third])
        self.assertNotIn(second, [sentence_of(text, v) for v in result.output.obligations])
        verify(text, ExtractionOutput(status='EXTRACTED', obligations=[*result.output.obligations, three[1]]))

    def test_a_sentence_that_simply_had_no_candidate_is_still_filled_beside_a_rejection(self):
        text, (request, _, full) = transfer()
        result = extract(text, Reviewed(['SUPPORTED', 'UNSUPPORTED'], answer(request, full)))
        self.kept(text, result)
        # The third sentence's duty was rejected and stays out; the second sentence, which no candidate had, gets its own.
        self.assertEqual(fields(result), [(RECEIVING, REQUEST, []), (RECEIVING, RETURN, [MISSING])])
        [record] = records(result, COVERED_CODE)
        self.assertEqual([(f['sentence'], f['field'], f['source']) for f in record['fields']], [(1, 'candidate', RETURN)])
        self.assertEqual(record['uncovered'], [{'sentence': 2, 'detail': REVIEW_REJECTED}])
        self.assertEqual(trace_of(result), [
            {'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'exact'},
            {'candidate': 1, 'code': SOURCE_BOUND_OK, 'tier': 'exact', 'review': 'UNSUPPORTED'},
            {'candidate': 2, 'code': SOURCE_BOUND_OK, 'tier': 'structured', 'origin': 'sentence', 'sentence': 1, 'cross_sentence': ['subject']}])

    def test_a_doubted_supporting_duty_never_takes_the_main_duty_with_it(self):
        main = candidate(FALLBACK, 'Yükümlüler', 'MUST', 'şüpheli işlemleri gecikmeksizin Başkanlığa bildirir')
        support = candidate(FALLBACK, 'Yükümlüler', 'MUST', 'işlemi askıya alır', conditions=[NOT_MADE])
        verify(FALLBACK, ExtractionOutput(status='EXTRACTED', obligations=[main, support]))
        for decisions in (['SUPPORTED', 'UNCERTAIN'], ['SUPPORTED', 'UNSUPPORTED'], ['UNCERTAIN', 'SUPPORTED']):
            with self.subTest(decisions=decisions):
                result = extract(FALLBACK, Reviewed(decisions, answer(main, support)))
                self.kept(FALLBACK, result)
                kept = result.output.obligations[0]
                # The main duty keeps every field; only its confidence says when the review doubted it.
                self.assertEqual(kept.model_copy(update={'confidence_score': main.confidence_score}), main)
                self.assertEqual(scores(result)[0], '0.2500' if decisions[0] == 'UNCERTAIN' else '0.5000')
                if decisions[1] == 'UNSUPPORTED':
                    self.assertEqual(len(result.output.obligations), 1)
                    self.assertEqual(records(result, UNCOVERED_CODE)[0]['uncovered'], [{'sentence': 1, 'detail': REVIEW_REJECTED}])
                else:
                    # The supporting duty keeps its own condition.
                    self.assertEqual(fields(result)[1], ('Yükümlüler', 'işlemi askıya alır', [NOT_MADE]))

    def test_a_cross_sentence_dependency_survives_a_doubt_or_a_rejection_of_a_sibling(self):
        _, text = unit('24/A', 3)
        request = candidate(text, RECEIVING, 'MUST', REQUEST)
        # The third sentence's duty keeps the condition of the sentence before it (a supporting dependency).
        full = candidate(text, RECEIVER, 'MUST', FULL, conditions=[MISSING, PERSISTENT])
        for decisions in (['UNCERTAIN', 'SUPPORTED'], ['UNSUPPORTED', 'SUPPORTED']):
            with self.subTest(decisions=decisions):
                result = extract(text, Reviewed(decisions, answer(request, full)))
                self.kept(text, result)
                kept = next(v for v in result.output.obligations if action_of(v) == FULL)
                self.assertEqual(kept.conditions, [MISSING, PERSISTENT])
                entry = next(e for e in trace_of(result) if e['candidate'] == 1)
                self.assertEqual(entry, {'candidate': 1, 'code': SOURCE_BOUND_OK, 'tier': 'exact', 'cross_sentence': ['conditions']})
        # A doubted first duty still lends its subject to the sentence after it, which carries that subject over.
        result = extract(text, Reviewed(['UNCERTAIN', 'SUPPORTED'], answer(request, full)))
        self.assertEqual(fields(result)[2], (RECEIVING, RETURN, [MISSING]))
        self.assertEqual(trace_of(result)[2]['cross_sentence'], ['subject'])
        # A rejected one lends nothing: that sentence stays uncovered, on record, and no subject is invented for it.
        result = extract(text, Reviewed(['UNSUPPORTED', 'SUPPORTED'], answer(request, full)))
        self.assertEqual([action_of(v) for v in result.output.obligations], [FULL])
        self.assertEqual(records(result, UNCOVERED_CODE)[0]['uncovered'],
                         [{'sentence': 0, 'detail': REVIEW_REJECTED}, {'sentence': 1, 'detail': 'NO_SUBJECT'}])

    def test_decisions_that_cannot_be_paired_with_the_candidates_end_the_unit_on_record(self):
        text, three = transfer()
        for decisions in (['SUPPORTED', 'SUPPORTED'], ['SUPPORTED'] * 4, [], ['SUPPORTED', 'MAYBE', 'SUPPORTED']):
            with self.subTest(decisions=decisions):
                self.ended(extract(text, Reviewed(decisions, answer(*three))), MISALIGNED, decisions, three)
        # A review that answers one bool (the t6 test doubles) and says no: every candidate withheld, on record.
        self.ended(extract(text, Reviewed(False, answer(*three))), NO_DECISIONS, [], three)
        # t7 review 1: ALL_UNCERTAIN and NONE_SUPPORTED no longer end a unit (R1); a failed review does (REVIEW_FAILED, R4).
        self.assertEqual(set(REVIEW_OUTCOMES), {ALL_REJECTED, MISALIGNED, NO_DECISIONS, 'REVIEW_FAILED'})
        self.assertFalse({ALL_UNCERTAIN, NONE_SUPPORTED} & set(REVIEW_OUTCOMES))

    def test_a_rescued_answer_is_reviewed_per_candidate_too(self):
        # The v019t6 shape: three duties without their conditions, refused (MISSING_QUALIFIER) on every turn, then rescued
        # by the grounding ladder. Whichever single candidate the review doubts or rejects, the "iade eder" duty survives
        # unless it is the rejected one, and the others keep their ladder conditions.
        _, text = unit('24/A', 3)
        bare = answer(candidate(text, 'kripto varlık hizmet sağlayıcı', 'MUST', REQUEST),
                      candidate(text, 'kripto varlık hizmet sağlayıcı', 'MUST', RETURN),
                      candidate(text, RECEIVER, 'MUST', FULL))
        plans = [one_uncertain(3, p) for p in range(3)] + [['UNSUPPORTED' if i == p else 'SUPPORTED' for i in range(3)] for p in range(3)]
        for decisions in plans + [['SUPPORTED'] * 3]:
            with self.subTest(decisions=decisions):
                result = extract(text, Reviewed(decisions, bare))
                self.kept(text, result)
                self.assertEqual([r['steps'] for r in records(result, 'GROUNDING_SALVAGED')], [['qualifier_source']])
                actions = [action_of(v) for v in result.output.obligations]
                self.assertEqual(RETURN in actions, decisions[1] != 'UNSUPPORTED')
                self.assertEqual(len(actions), 3 - decisions.count('UNSUPPORTED'))
                conditions = {action_of(v): v.conditions for v in result.output.obligations}
                self.assertEqual({a: c for a, c in conditions.items() if c}, {a: c for a, c in {RETURN: [MISSING], FULL: [PERSISTENT]}.items()
                                                                              if a in conditions})


class UnchangedWhenSupportedTests(Checks):
    def scenarios(self):
        text, three = transfer()
        yield text, [answer(*three)]
        yield text, [answer(candidate(text, RECEIVER, 'MUST', SHORT))]                     # t5 ladder, then two new duties
        yield text, [answer(candidate(text, RECEIVER, 'MUST', FULL, conditions=[MISSING, PERSISTENT]))]
        yield REPORTS, [answer(candidate(REPORTS, 'yükümlüler', 'MUST', 'müşteriyle iş ilişkisinin sona erdirilmesi hususu göz önünde '
                                                  'bulundurulur', conditions=['İşlemin tekrarlanması halinde']))]
        english = 'A firm must retain records. The firm will notify the FCA of a breach.'
        yield english, [answer(candidate(english, 'A firm', 'MUST', 'retain records.'))]

    def test_an_all_supported_review_leaves_output_trace_and_records_exactly_as_in_t6(self):
        for text, answers in self.scenarios():
            with self.subTest(text=text[:30]):
                baseline = extract(text, Unreviewed(None, *answers))
                count = len(json.loads(answers[0])['obligations'])
                for decisions in (True, ['SUPPORTED'] * count):
                    provider = Reviewed(decisions, *answers)
                    result = extract(text, provider)
                    self.assertEqual(comparable(result), comparable(baseline))
                    self.assertEqual([d for d in result.diagnostics if d.get('stage') == 'review'], [])
                    self.assertTrue(provider.reviewed)

    def test_apply_review_is_a_no_op_without_a_doubt(self):
        text, three = transfer()
        output = ExtractionOutput(status='EXTRACTED', obligations=three)
        trace = accepted_trace(output)
        for decisions in (None, [], ['SUPPORTED'] * 3):
            got, kept_trace, rejected, record = apply_review(output, trace, decisions)
            self.assertIs(got, output)
            self.assertIs(kept_trace, trace)
            self.assertEqual((rejected, record), ([], None))

    def test_second_reading_reads_a_bool_review_as_before(self):
        text, three = transfer()
        output = ExtractionOutput(status='EXTRACTED', obligations=three)
        self.assertEqual(second_reading(Reviewed(True), text, output), (['SUPPORTED'] * 3, None))
        self.assertEqual(second_reading(Reviewed(False), text, output), ([], NO_DECISIONS))
        self.assertEqual(second_reading(Reviewed(('SUPPORTED', 'UNCERTAIN', 'SUPPORTED')), text, output),
                         (['SUPPORTED', 'UNCERTAIN', 'SUPPORTED'], None))

    def test_the_review_request_is_the_t6_one_and_its_answer_the_decisions(self):
        text, three = transfer()
        output = ExtractionOutput(status='EXTRACTED', obligations=three)
        context = ContextPacket(target_id='x', items=({'section_id': 's-1', 'text': 'Bağlam.', 'reason': 'neighbor'},), unresolved=(),
                                truncated=False, source_kind='CONSOLIDATED', quality_flags=())
        provider = OllamaProvider('fixture:latest', 'sha256:abc', 'http://localhost:11434')
        sent = []

        def chat(prompt, payload, schema):
            sent.append((prompt, payload, schema))
            return json.dumps({'decisions': ['SUPPORTED', 'UNCERTAIN', 'UNSUPPORTED']})

        with patch.object(provider, '_chat', side_effect=chat):
            self.assertEqual(provider.review(text, output, context), ['SUPPORTED', 'UNCERTAIN', 'UNSUPPORTED'])
        # The t6 request, built here the way t6 built it: same prompt, payload and schema, byte for byte.
        expected = {'source_paragraph': text,
                    'context': {'sources': [{'source_id': '1', 'text': 'Bağlam.', 'reason': 'neighbor', 'required': False, 'label': None}],
                                'dependency_gaps': []},
                    'candidates': [{'subject': c.subject, 'modality': c.modality, 'action': c.required_action or c.prohibited_action,
                                    'conditions': c.conditions, 'exceptions': c.exceptions, 'evidence': []} for c in three]}
        [(prompt, payload, schema)] = sent
        self.assertEqual(prompt, REVIEW_PROMPT)
        self.assertEqual(sha256(canonical_bytes(payload)).hexdigest(), sha256(canonical_bytes(expected)).hexdigest())
        self.assertEqual(schema, SecondPass.model_json_schema())

    def test_an_uncertain_candidate_carries_its_flag_into_the_packet_row(self):
        text, three = transfer()
        result = extract(text, Reviewed(one_uncertain(3, 1), answer(*three)))
        rows = [v.model_dump(mode='json') for v in result.output.obligations]
        self.assertEqual([row['confidence_score'] == REVIEW_UNCERTAIN_CONFIDENCE for row in rows], [False, True, False])
        canonical_bytes(rows)

    def test_the_review_step_still_never_breaks_on_coverage(self):
        text, three = transfer()
        with patch.object(pipeline, 'cover_sentences', side_effect=RuntimeError('boom')):
            result = extract(text, Reviewed(['SUPPORTED', 'UNSUPPORTED', 'SUPPORTED'], answer(*three)))
        self.assertEqual([action_of(v) for v in result.output.obligations], [REQUEST, FULL])
        self.assertEqual(records(result, UNCOVERED_CODE)[0]['detail'], 'COVERAGE_FAILED:RuntimeError')


if __name__ == '__main__':
    unittest.main()

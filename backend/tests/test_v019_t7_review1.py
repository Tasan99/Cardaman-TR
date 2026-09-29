"""v0.19 t7 review 1 (26 September 2026): the fixes of the first review round of t7, each pinned here.

Extraction (pipeline.py):
  R1  a candidate's fate is a function of its own review decision only: UNCERTAIN always keeps its candidate (flagged), whatever
      its siblings got; only a review that rejects every candidate (each by its own decision) ends the unit.
  R4  a failing review never escapes extract (the ladder rescue path ran it inside the except handler) and is never reported as
      GROUNDING_REJECTED: the unit ends on record (REVIEW_FAILED, the failure code, every withheld candidate).
  H1  a candidate the review kept as UNCERTAIN carries the review flag EXTRACTION_REVIEW_UNCERTAIN in the pilot (autonomous_gate
      escalates it), read from the review record's explicit positions (review_uncertain), never from confidence_score; no judge
      request changes.
  H2  the extraction PROMPT_HASH names the per-candidate review (grounding_version v3.5), so a v3.4 run or packet is not reused.
Coverage (conflict.py positive gate, engine.py):
  R6  a failed confirmation no longer brings back the favourable fast reading an answered one loses (UNRELATED, or UNCLEAR for an
      opposed effect); F8 the last CONFLICT_TYPE_UNSUPPORTED fallback is gated too.
  F1  a favourable fast reading of a prohibition on a sentence giving the prohibited act an opposed effect goes to the verifier
      (OPPOSED_EFFECT signal) instead of standing unread: the contradiction can be found.
  F2  SCOPE reads the groups whose data the act is about, not an addressee list of the parties that act.
  F3  OBJECT needs an unguarded statement about the object: negated or confidentiality wording keeps the reading.
  F4  RECIPIENT reads "X tarafına" as the dative of X and "şirketin yetkilileri" as that entity's officials.
  F7  the duty's own exception limits RECIPIENT only for the kinds of addressee it names itself.
Applicability (applicability.py): F5 the provision-level answer is withheld from an obliged-only clause only when the profile's own
business shows nothing that may make the company an obliged party.
Report (report.py): the t7 codes are named (additive).
Every model here is a fake answering by schema; no Ollama call.
"""
import inspect
import json
import unittest
from itertools import product
from unittest.mock import patch

from pydantic import ValidationError

from regchain.evidence import canonical_bytes, digest
from regchain.extraction import pipeline, service
from regchain.extraction.pipeline import (ALL_REJECTED, MISALIGNED, PER_CANDIDATE_CODE, REVIEW_CODE, REVIEW_FAILED, extract,
                                          review_uncertain)
from regchain.extraction.providers import ProviderFailure, failure, failure_code
from regchain.extraction.structure import LEGACY_KEYS, duty_payload
from regchain.extraction.schema import SecondPass
from regchain.pilot import applicability as rules
from regchain.pilot import conflict as precheck
from regchain.pilot import engine, impact
from regchain.pilot.engine import REVIEW_UNCERTAIN_FLAG, analyze, autonomous_gate, autonomous_review, pipeline_settings
from regchain.pilot.report import POSITIVE_GATES, REVIEW_FLAGS, TRACE_GATES, WITHHELD_BY, render
from regchain.pilot.schema import Company
from regchain.pilot.sources import load_sources, select_targets

import test_v019_t7_positive_gate as T
from test_pilot import FixtureProvider, company, policies, sections
from test_v019_conflict_pipeline import fast, verdict
from test_v019_extraction import action_of, answer, candidate, unit
from test_v019_stricter_partial import associations
from test_v019_t5_engine import codes, disclosure, plain, with_lists
from test_v019_t6_alignment import FULL, PERSISTENT, RECEIVER, REQUEST, RETURN, records
from test_v019_t7_candidate_review import Reviewed, transfer
from test_v019_t7_negative_control import CAFE, LABEL, ProvisionJudge
from test_v019_universal_duty import TEDBIRLER, independent_profile

DECISIONS = ('SUPPORTED', 'UNCERTAIN', 'UNSUPPORTED')


def bare():
    """The v019t6 md. 24/A(3) shape the gate refuses on every turn (no conditions) and the grounding ladder rescues."""
    _, text = unit('24/A', 3)
    return text, answer(candidate(text, 'kripto varlık hizmet sağlayıcı', 'MUST', REQUEST),
                        candidate(text, 'kripto varlık hizmet sağlayıcı', 'MUST', RETURN), candidate(text, RECEIVER, 'MUST', FULL))


class Failing(Reviewed):
    """A review that fails `times` times with `error` (an exception, or a callable raising one), then answers `decisions`."""

    def __init__(self, error, decisions, *answers, times=99):
        super().__init__(decisions, *answers)
        self.error, self.times, self.failed = error, times, 0

    def review(self, text, output, context):
        if self.failed < self.times:
            self.failed += 1
            if callable(self.error):
                self.error()
            raise self.error
        return super().review(text, output, context)


def malformed():
    SecondPass.model_validate_json('{"decisions": ["MAYBE"]}')


class CandidateFateTests(unittest.TestCase):
    """R1: one candidate's review decision never decides another candidate's fate."""

    def check(self, text, answers, three_actions, path):
        for decisions in map(list, product(DECISIONS, repeat=3)):
            with self.subTest(path=path, decisions=decisions):
                result = extract(text, Reviewed(decisions, answers))
                if decisions == ['UNSUPPORTED'] * 3:
                    # Every candidate rejected by its own decision: the unit ends, on record.
                    self.assertEqual((result.reason, result.diagnostics[-1]['review']), (REVIEW_CODE, ALL_REJECTED))
                    continue
                kept = [a for a, d in zip(three_actions, decisions) if d != 'UNSUPPORTED']
                self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
                self.assertEqual([action_of(v) for v in result.output.obligations][:len(kept)], kept)
                scores = {action_of(v): v.model_dump(mode='json')['confidence_score'] for v in result.output.obligations}
                self.assertEqual({a: scores[a] for a in kept},
                                 {a: '0.2500' if d == 'UNCERTAIN' else '0.5000' for a, d in zip(three_actions, decisions) if d != 'UNSUPPORTED'})
                # The explicit marker the pilot reads names exactly the doubted ones.
                self.assertEqual(review_uncertain(result), [i for i, d in enumerate(d for d in decisions if d != 'UNSUPPORTED') if d == 'UNCERTAIN'])

    def test_every_decision_plan_keeps_exactly_the_candidates_not_rejected_by_their_own_decision(self):
        text, three = transfer()
        self.check(text, answer(*three), [REQUEST, RETURN, FULL], 'direct')
        text, rescued = bare()
        self.check(text, rescued, [REQUEST, RETURN, FULL], 'ladder rescue')

    def test_the_reviewer_s_non_local_example_now_keeps_the_doubted_supporting_duty(self):
        # t7 review R1: [S, U] kept "işlemi askıya alır" and [R, U] withheld it; only candidate 0's decision differed.
        text = ('(4) Yükümlüler, şüpheli işlemleri gecikmeksizin Başkanlığa bildirir. '
                'Bildirimin yapılamaması halinde işlemi askıya alır.')
        main = candidate(text, 'Yükümlüler', 'MUST', 'şüpheli işlemleri gecikmeksizin Başkanlığa bildirir')
        support = candidate(text, 'Yükümlüler', 'MUST', 'işlemi askıya alır', conditions=['Bildirimin yapılamaması halinde'])
        for first in ('SUPPORTED', 'UNSUPPORTED', 'UNCERTAIN'):
            result = extract(text, Reviewed([first, 'UNCERTAIN'], answer(main, support)))
            self.assertIn('işlemi askıya alır', [action_of(v) for v in result.output.obligations], first)

    def test_a_non_list_answer_is_no_confirmation(self):
        text, three = transfer()
        for odd in ('UNCERTAIN', {'decisions': ['SUPPORTED'] * 3}):
            result = extract(text, Reviewed(odd, answer(*three)))
            self.assertEqual((result.reason, result.diagnostics[-1]['review']), (REVIEW_CODE, MISALIGNED))
        # The bool of the t6 test doubles keeps its meaning.
        self.assertEqual(extract(text, Reviewed(True, answer(*three))).reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')


class ReviewFailureTests(unittest.TestCase):
    """R4: a failing review ends the unit on record; it never escapes extract and is never a grounding reject."""

    def ended_on_failure(self, result, code, withheld):
        self.assertEqual(result.reason, REVIEW_CODE)
        last = result.diagnostics[-1]
        self.assertEqual((last['stage'], last['code'], last['review'], last['failure'], last['candidates']),
                         ('review', REVIEW_CODE, REVIEW_FAILED, code, withheld))
        self.assertEqual(len(last['withheld']), withheld)
        canonical_bytes(list(result.diagnostics))

    def test_a_failing_review_on_the_ladder_rescue_path_ends_the_unit_on_record(self):
        text, rescued = bare()
        for error, code in ((failure('cut', 'output_truncated'), 'OUTPUT_TRUNCATED'), (failure('slow', 'transport', 'TIMEOUT'), 'TIMEOUT'),
                            (malformed, 'MALFORMED_JSON'), (KeyError('s-9'), 'MALFORMED_JSON'), (ValueError('x'), failure_code(ValueError('x')))):
            with self.subTest(code=code, error=type(error).__name__):
                result = extract(text, Failing(error, ['SUPPORTED'] * 3, rescued))
                self.ended_on_failure(result, code, 3)
                self.assertTrue(records(result, 'GROUNDING_SALVAGED'))

    def test_an_unreadable_review_on_the_direct_path_is_no_grounding_reject(self):
        text, three = transfer()
        for error, code in ((malformed, 'MALFORMED_JSON'), (ValueError('x'), failure_code(ValueError('x'))), (KeyError('s-9'), 'MALFORMED_JSON')):
            with self.subTest(error=getattr(error, '__name__', type(error).__name__)):
                result = extract(text, Failing(error, ['SUPPORTED'] * 3, answer(*three)))
                self.ended_on_failure(result, code, 3)
                self.assertNotIn('GROUNDING_REJECTED', [d.get('code') for d in result.diagnostics])

    def test_a_provider_failure_on_the_direct_path_is_retried_as_before(self):
        text, three = transfer()
        result = extract(text, Failing(failure('cut', 'output_truncated'), ['SUPPORTED'] * 3, answer(*three), times=1))
        self.assertEqual((result.reason, result.attempts), ('CANDIDATE_REQUIRES_LEGAL_REVIEW', 2))
        self.assertEqual([(d['attempt'], d['stage'], d['code']) for d in result.diagnostics if d.get('code') == 'PROVIDER_FAILURE'],
                         [(1, 'review', 'PROVIDER_FAILURE')])
        with self.assertRaises(ProviderFailure):
            pipeline.read_review(Failing(failure('cut', 'output_truncated'), [], answer(*three)), text,
                                 pipeline.ExtractionOutput.model_validate_json(answer(*three)))


class ReviewFlagTests(unittest.TestCase):
    """H1 (R3): a duty the review kept as UNCERTAIN is flagged EXTRACTION_REVIEW_UNCERTAIN and escalated; no judge request changes."""
    TEXT = 'A firm must retain records. The firm must notify the FCA of a breach.'

    class Recording(FixtureProvider):
        def __init__(self):
            super().__init__()
            self.requests = []

        def _chat(self, prompt, payload, schema):
            self.requests.append(canonical_bytes({'prompt': prompt, 'payload': payload, 'schema': schema}))
            return super()._chat(prompt, payload, schema)

    def run_plan(self, plan):
        secs = sections()
        secs[1] = dict(secs[1], text=self.TEXT)
        real = engine.extract

        class Scripted:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, key):
                return getattr(self.inner, key)

            def review(self, text, output, context):
                return list(plan) if len(output.obligations) == len(plan) else ['SUPPORTED'] * len(output.obligations)

        provider = self.Recording()
        with patch('regchain.pilot.engine.extract', side_effect=lambda text, p, context=None: real(text, Scripted(p), context)):
            packet = analyze(company(), policies(), secs, provider, ['CONC 7.3.4'])
        return packet, provider

    def test_an_uncertain_duty_is_escalated_and_a_confirmed_one_is_not(self):
        packet, _ = self.run_plan(['SUPPORTED', 'UNCERTAIN'])
        rows = {action_of_row(r): r for r in packet['events'][0]['payload']['obligations']}
        doubted, confirmed = rows['notify the FCA of a breach.'], rows['retain records.']
        self.assertEqual(doubted['proposal']['review_flags'][0], REVIEW_UNCERTAIN_FLAG)
        self.assertEqual(autonomous_gate(doubted), 'Flagged for a person: ' + REVIEW_UNCERTAIN_FLAG)
        self.assertNotIn(REVIEW_UNCERTAIN_FLAG, confirmed['proposal']['review_flags'])
        _, summary = autonomous_review(packet)
        self.assertEqual((summary['decided'], summary['escalated']), (1, 1))
        packet, _ = self.run_plan(['SUPPORTED', 'SUPPORTED'])
        self.assertFalse(any(REVIEW_UNCERTAIN_FLAG in r['proposal']['review_flags'] for r in packet['events'][0]['payload']['obligations']))

    def test_no_judge_request_changes_with_the_flag(self):
        # The same review answer with and without the flag: every judge request is byte-identical (the flag is a review flag only).
        packet, flagged = self.run_plan(['SUPPORTED', 'UNCERTAIN'])
        with patch.object(engine, 'review_uncertain', return_value=[]):
            unflagged_packet, unflagged = self.run_plan(['SUPPORTED', 'UNCERTAIN'])
        self.assertEqual(flagged.requests, unflagged.requests)
        self.assertTrue(any(REVIEW_UNCERTAIN_FLAG in r['proposal']['review_flags'] for r in packet['events'][0]['payload']['obligations']))
        self.assertFalse(any(REVIEW_UNCERTAIN_FLAG in r['proposal']['review_flags'] for r in unflagged_packet['events'][0]['payload']['obligations']))
        # The v0.19 judge reads the duty through duty_payload, which never carries the review's confidence either.
        self.assertNotIn('confidence_score', LEGACY_KEYS)
        text, three = transfer()
        result = extract(text, Reviewed(['SUPPORTED', 'UNCERTAIN', 'SUPPORTED'], answer(*three)))
        values = [v.model_dump(mode='json') for v in result.output.obligations]
        self.assertEqual(duty_payload(values[1], None), duty_payload({**values[1], 'confidence_score': '0.5000'}, None))

    def test_the_flag_is_read_from_the_review_record_not_from_the_confidence(self):
        self.assertNotIn('confidence_score', inspect.getsource(pipeline.review_uncertain).split('"""')[-1])
        text, three = transfer()
        # A doubted candidate followed by the coverage step's fills: the record's position still names it.
        only = candidate(text, RECEIVER, 'MUST', FULL, conditions=[PERSISTENT])
        result = extract(text, Reviewed(['UNCERTAIN'], answer(only)))
        self.assertGreater(len(result.output.obligations), 1)
        self.assertEqual(review_uncertain(result), [0])
        self.assertEqual(action_of(result.output.obligations[0]), FULL)
        self.assertEqual(review_uncertain(extract(text, Reviewed(['SUPPORTED'] * 3, answer(*three)))), [])


def action_of_row(row):
    c = row['candidate']
    return c.get('required_action') or c.get('prohibited_action')


class ProvenanceTests(unittest.TestCase):
    """H2 (PROV-P2-1): the per-candidate review moves the extraction PROMPT_HASH, so a v3.4 run or packet is not reused."""

    def test_the_grounding_version_names_the_candidate_review_and_moves_the_hash(self):
        source = inspect.getsource(service)
        self.assertIn("'grounding_version':'scoped-evidence-v3.5-candidate-review'", source)
        self.assertNotIn("'grounding_version':'scoped-evidence-v3.4-sentence-cover'", source)
        parts = {'prompt': service.PROMPT, 'context_prompt': service.CONTEXT_PROMPT, 'review_prompt': service.REVIEW_PROMPT,
                 'schema_json': json.dumps(service.ExtractionOutput.model_json_schema(), sort_keys=True, separators=(',', ':'), allow_nan=False),
                 'retriever_version': service.RETRIEVER_VERSION, 'contract_version': service.CONTRACT_VERSION, 'example': service.EXAMPLE,
                 'model_schema_json': json.dumps(service.ModelOutput.model_json_schema(), sort_keys=True, separators=(',', ':'), allow_nan=False)}
        self.assertEqual(digest({**parts, 'grounding_version': 'scoped-evidence-v3.5-candidate-review'}), service.PROMPT_HASH)
        old = digest({**parts, 'grounding_version': 'scoped-evidence-v3.4-sentence-cover'})
        self.assertNotEqual(old, service.PROMPT_HASH)
        previous = {'company_hash': 'c', 'policies': [], 'extraction_prompt_hash': old, 'pilot_prompt_hash': 'p'}
        self.assertEqual(impact.reuse_blockers(previous, 'c', [], service.PROMPT_HASH, 'p'), ['prompts or judgement version changed'])


class FailedConfirmationTests(unittest.TestCase):
    """R6: a failed confirmation never brings back the favourable reading the gate rejects when the verifier answers."""
    INTERNAL = 'Şüpheli işlem bildirimleri uyum görevlisine iletilir.'
    FILED = 'Şüpheli işlem bildirimleri uyum biriminde dosyalanır.'

    def failed(self, payload, text, label, quantities=()):
        missing = ('prohibition',) if label == 'POSSIBLE_PARTIAL' and any(e['kind'] == 'prohibition' for e in payload['elements']) else ()
        covered = [e['id'] for e in payload['elements'] if e['id'] not in missing]
        answers = {text: fast(label, text, covered=covered, missing=missing)}
        out = {}
        for name, answer in (('truncated', failure('cut', 'output_truncated')), ('malformed', '{"conflict": tru'), ('failure', ProviderFailure('x'))):
            relations, results, (coverage, _, _, flags), _ = T.flow(payload, [text], answers, {text: answer}, quantities)
            out[name] = (relations['p1'], coverage, T.gate_codes(results['p1']))
        return out

    def test_an_opposed_effect_is_left_for_a_person_and_another_object_or_group_is_no_evidence(self):
        # The confirmation of an opposed-effect reading (the F1 escalation stubbed out, so the confirmation question is asked).
        with patch.object(precheck, 'opposed_effect_signal', return_value=None):
            for name, (relation, coverage, gate) in self.failed(disclosure(), self.INTERNAL, 'POSSIBLE_PARTIAL').items():
                self.assertEqual((relation, coverage, gate), ('UNCLEAR', 'UNKNOWN', ['POSITIVE_GATE_POLARITY']), name)
        # With F1 the same reading is asked as a possible conflict; its failure leaves it UNCLEAR too, never PARTIAL.
        for name, (relation, coverage, gate) in self.failed(disclosure(), self.INTERNAL, 'POSSIBLE_PARTIAL').items():
            self.assertEqual((relation, coverage), ('UNCLEAR', 'UNKNOWN'), name)
        for name, (relation, coverage, gate) in self.failed(disclosure(), self.FILED, 'POSSIBLE_PARTIAL').items():
            self.assertEqual((relation, coverage, gate), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_OBJECT']), name)
        payload, quantities = associations()
        for label in ('POSSIBLE_PARTIAL', 'POSSIBLE_SUPPORT'):
            for name, (relation, coverage, gate) in self.failed(payload, T.COMPANIES, label, quantities).items():
                self.assertEqual((relation, coverage, gate), ('UNRELATED', 'NO_EVIDENCE', ['POSITIVE_GATE_SCOPE']), (label, name))

    def test_a_reading_the_gate_passes_still_stands_unconfirmed(self):
        payload, quantities = T.md28()
        answers = {T.REPORT_BARE: fast('POSSIBLE_PARTIAL', T.REPORT_BARE, covered=('action',), missing=('deadline_1',))}
        relations, results, (coverage, _, _, flags), _ = T.flow(payload, [T.REPORT_BARE], answers,
                                                                {T.REPORT_BARE: failure('cut', 'output_truncated')}, quantities)
        self.assertEqual((relations['p1'], coverage, T.gate_codes(results['p1'])), ('PARTIAL', 'PARTIAL', []))
        self.assertIn('PARTIAL_UNCONFIRMED', flags)

    def test_the_last_unsupported_type_fallback_is_gated_too(self):
        # F8: a numeric contradiction type the sentence has no wording for, with no relation of its own.
        answers = {T.TO_CUSTOMER: fast('POSSIBLE_CONFLICT', T.TO_CUSTOMER)}
        for kind in ('DEADLINE_MISMATCH', 'THRESHOLD_MISMATCH'):
            claim = {T.TO_CUSTOMER: verdict(True, kind, T.TO_CUSTOMER, 'NOT_APPLICABLE')}
            relations, results, (coverage, _, _, _), _ = T.flow(T.REPORT, [T.TO_CUSTOMER], answers, claim)
            self.assertEqual((relations['p1'], coverage), ('UNRELATED', 'NO_EVIDENCE'), kind)
            self.assertEqual(codes(results['p1']), ['CONFLICT_TYPE_UNSUPPORTED', 'POSITIVE_GATE_RECIPIENT'])


class OpposedEffectTests(unittest.TestCase):
    """F1: a fast SUPPORTS / PARTIAL of a prohibition whose sentence performs the prohibited act is read by the verifier."""
    OPPOSED = 'Şüpheli işlem bildirimi yapıldığı bilgisi şube uyum sorumlusuna iletilir.'

    def flow(self, claim):
        payload = disclosure()
        answers = {T.CONFIDENTIAL: fast('POSSIBLE_SUPPORT', T.CONFIDENTIAL, covered=('action', 'subject', 'prohibition')),
                   self.OPPOSED: fast('POSSIBLE_PARTIAL', self.OPPOSED, covered=('action',), missing=('prohibition',))}
        verifier = {T.CONFIDENTIAL: with_lists(verdict(relation='SUPPORTS', support=T.CONFIDENTIAL), ['action', 'subject', 'prohibition']),
                    self.OPPOSED: claim}
        return T.flow(payload, [T.CONFIDENTIAL, self.OPPOSED], answers, verifier)

    def test_the_opposed_sentence_is_asked_and_its_contradiction_found(self):
        claim = verdict(True, 'PROHIBITED_ACTION_ALLOWED', self.OPPOSED, 'UNRELATED')
        relations, results, (coverage, _, _, _), judge = self.flow(claim)
        self.assertEqual((relations['p2'], coverage), ('CONFLICTS', 'CONFLICT'))
        self.assertEqual(results['p2']['escalation'], 'STRUCTURAL_SIGNAL')
        self.assertIn(precheck.OPPOSED_EFFECT, codes(results['p2']))
        asked = [c for c in judge.kinds('verify') if c['passage'] == self.OPPOSED]
        self.assertEqual([s['type'] for s in asked[0]['payload']['automatic_signals']], [precheck.OPPOSED_EFFECT])
        # Before this fix the fast reading stood unread beside the confirmed cover: the row was COVERS_TEXT, the contradiction missed.
        with patch.object(precheck, 'opposed_effect_signal', return_value=None):
            relations, results, (coverage, _, _, _), judge = self.flow(claim)
        self.assertEqual((relations['p2'], coverage), ('PARTIAL', 'COVERS_TEXT'))
        self.assertEqual([c['passage'] for c in judge.kinds('verify')], [T.CONFIDENTIAL])

    def test_a_favourable_answer_on_it_is_no_evidence_and_never_a_contradiction(self):
        support = with_lists(verdict(relation='PARTIAL', support=self.OPPOSED), ['action'], ['prohibition'])
        relations, results, (coverage, _, _, _), _ = self.flow(support)
        self.assertEqual((relations, coverage), ({'p1': 'SUPPORTS', 'p2': 'UNRELATED'}, 'COVERS_TEXT'))
        self.assertEqual(T.gate_codes(results['p2']), ['POSITIVE_GATE_POLARITY'])

    def test_only_an_unguarded_opposed_effect_on_a_prohibition_is_signalled(self):
        self.assertIsNotNone(precheck.opposed_effect_signal(disclosure(), self.OPPOSED, self.OPPOSED, ['prohibition']))
        for text in ('Şüpheli işlem bildiriminde bulunulduğu hiç kimseyle paylaşılmaz.',
                     'Bildirim bilgisi yalnızca denetim elemanlarına ve mahkemelere açıklanır.', T.CONFIDENTIAL):
            self.assertIsNone(precheck.opposed_effect_signal(disclosure(), text, text), text)
        # A requirement is not read here (its negations are the structural signals' job), nor is another recipient.
        self.assertIsNone(precheck.opposed_effect_signal(T.REPORT, T.TO_CUSTOMER, T.TO_CUSTOMER))
        self.assertIsNone(precheck.opposed_effect_signal(disclosure(), T.AUTHORITY_REPORT_QUOTE, T.AUTHORITY_REPORT))


class GateReadingTests(unittest.TestCase):
    """F2, F3, F4, F7: the positive gate's readings of scope, object and recipient."""
    PROVIDE = plain('Başkanlık ve denetim elemanları tarafından istenilecek her türlü bilgi ve belgeyi tam ve doğru olarak vermek',
                    subject='Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar')
    KEEP = plain('belge ve kayıtları sekiz yıl süre ile muhafaza etmek ve istenmesi halinde yetkililere ibraz etmek')

    def test_an_addressee_list_names_no_customer_group(self):
        self.assertEqual(precheck._scope_groups(self.PROVIDE), set())
        for text in ('Bireysel müşterilere ait bilgiler dahil tüm kayıtlar, Başkanlık istediğinde eksiksiz verilir.',
                     "Gerçek kişi müşterilerin bilgileri de MASAK'ın talebi üzerine gecikmeksizin verilir."):
            for verified in (False, True):
                self.assertEqual(precheck.support_gate(self.PROVIDE, text, text, (), verified)[0], '', text)
        # The groups whose data the act is about still read: identification of associations, of registered companies.
        payload, _ = associations()
        self.assertTrue(precheck._scope_groups(payload))
        self.assertEqual(precheck.support_gate(payload, T.COMPANIES, T.COMPANIES, ())[0], precheck.PG_SCOPE)
        genitive = plain('kimlik tespitinde adı, amacı ve adresi almak', subject='Sendika ve konfederasyonların')
        self.assertEqual(precheck._scope_groups(genitive), {'sendika'})
        # "gerçek ve tüzel kişi": both groups on either side.
        both = plain('kimlik tespitinde unvanı ve adresi almak', subject='Gerçek ve tüzel kişilerin')
        self.assertEqual(precheck._scope_groups(both), {'gerçek', 'tüzel'})
        self.assertEqual(precheck._party_groups(precheck.fold('Gerçek ve tüzel kişi müşterilerin kimlik tespitinde')), {'gerçek', 'tüzel'})

    def test_confidentiality_or_negated_wording_is_no_object_evidence(self):
        for text in ('Şüpheli işlem bildirimi yapıldığı bilgisi gizli tutulur.', 'Müşteri, hakkında bildirim yapıldığı konusunda bilgilendirilmez.',
                     'Bildirimde bulunulduğu hususunda müşteri uyarılmaz.', 'Şüpheli işlem bildirimlerinin gizliliği esastır.',
                     'Şüpheli işlem bildirimlerine ilişkin bilgiler gizlidir.', 'Bildirim yapıldığı müşteriye söylenmez.',
                     'STR filings are kept strictly confidential.'):
            self.assertEqual(precheck.support_gate(disclosure(), text, text, ['prohibition'])[0], '', text)
        # Another statement of the object, unguarded, still is.
        for text in (FailedConfirmationTests.FILED, 'Suspicious transactions are reported to the FIU.'):
            self.assertEqual(precheck.support_gate(disclosure(), text, text, ['prohibition'])[0], precheck.PG_OBJECT, text)

    def test_a_dative_of_an_institution_and_an_entity_s_officials_are_no_recipient_evidence(self):
        for text in ('Belge ve kayıtlar, istenmesi halinde talep eden kamu kurumu tarafına ibraz edilir.',
                     'Belge ve kayıtlar, istenmesi halinde denetim elemanları tarafına ibraz edilir.'):
            for verified in (False, True):
                self.assertEqual(precheck.support_gate(self.KEEP, text, text, (), verified)[0], '', text)
        self.assertEqual(precheck.support_gate(self.KEEP, 'Kayıtlar, talep halinde müşteri tarafına ibraz edilir.',
                                               'Kayıtlar, talep halinde müşteri tarafına ibraz edilir.', (), False)[0], precheck.PG_RECIPIENT)
        read = lambda text: [kind for _, kind in precheck._gate_recipients(precheck.fold(text))]
        self.assertEqual(read('müşterinin çalıştığı şirketin yetkililerine'), [])
        self.assertEqual(read('muhabir bankanın yetkililerine'), [])
        self.assertEqual(read('talep halinde yetkililere'), ['AUTHORITY'])
        self.assertEqual(read("MASAK'a ve işlemin tarafına"), ['AUTHORITY', 'THIRD_PARTY'])
        self.assertEqual(read('talep eden kamu kurumu tarafına'), [])
        # The t6 conflict gate's reading is untouched.
        self.assertEqual(precheck.recipient('şirketin yetkililerine'), 'AUTHORITY')

    def test_the_duty_s_own_exception_limits_only_the_addressees_it_names(self):
        cut = {**disclosure(), 'exceptions': ['şüpheli işlem bildiriminde bulunulduğuna veya bulunulacağına dair, yükümlülük denetimi ile '
                                              'görevlendirilen denetim elemanlarına ve yargılama sırasında mahkemelere verilen bilgiler dışında']}
        for text in ('Tüm şüpheli işlemler ilgili otoriteye raporlanır', "Şüpheli işlemler MASAK'a bildirilir."):
            self.assertTrue(precheck._own_exception(text, cut))
            self.assertEqual(precheck.support_gate(cut, text, text, (), False)[0], precheck.PG_RECIPIENT, text)
        for text in ('Bildirim bilgisi yalnızca denetim elemanlarına ve mahkemelere açıklanır.',
                     'Bildirim bilgisi denetim elemanlarına ve mahkemelere açıklanabilir.'):
            self.assertEqual(precheck.support_gate(cut, text, text, ['subject'])[0], '', text)

    def test_the_gate_s_version_moved(self):
        # Qualified-effect reading changes semantics and must invalidate old cache entries.
        self.assertEqual(precheck.POSITIVE_GATE_VERSION, 'positive-v3-qualified-effect')
        self.assertEqual(engine.V19_HASH_PARTS['positive_gate'], precheck.POSITIVE_GATE_VERSION)


class ObligedIdentityTests(unittest.TestCase):
    """F5: a clause addressed only to the obliged parties keeps the provision-level answer when the profile's own business may make
    the company one of them; the restaurant and the café of the negative control are still withheld."""
    ASSET = dict(id='alacak', name='Alacak Çözüm Ltd. Şti.', version='1', synthetic=True, jurisdictions=['TR'],
                 activities=['sorunlu alacak yönetimi'], licences=['BDDK izni'], products=['alacak portföyü satın alma'],
                 customer_types=['gerçek kişiler', 'tüzel kişiler'],
                 description='Bankalardan takipteki alacak portföyleri satın alıp tahsil eden şirket.')

    def rows(self, profile):
        tedbirler = load_sources(TEDBIRLER)[1]
        labels = select_targets(tedbirler, 'YONETMELIK', '200713012', 'all', ['31'])
        settings = pipeline_settings(applicability_clear_match='rule', relevance_screen='off', coverage_pipeline='v18')
        judge = ProvisionJudge('APPLIES')
        payload = analyze(profile, policies(), tedbirler, judge, labels, judge=judge, target_filter=['31'], settings=settings)['events'][0]['payload']
        return {o['proposal']['applicability_scope']['child_clause']: o['proposal'] for o in payload['obligations']}

    def test_a_supervised_company_the_list_cannot_place_keeps_the_answer(self):
        company = Company(**self.ASSET)
        self.assertEqual(rules.financial_identity(company), 'licences: BDDK izni')
        proposal = self.rows(company)['(3)']
        self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('APPLIES', 'MODEL'))
        self.assertNotIn('CHILD_ADDRESSEE', [g['gate'] for g in proposal['trace']['gates'] if g])

    def test_a_business_with_nothing_near_a_listed_kind_is_still_withheld(self):
        for profile in (Company(**CAFE), independent_profile('I08')):
            self.assertEqual(rules.financial_identity(profile), '')
            proposal = self.rows(profile)['(3)']
            self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('UNKNOWN', 'PROFILE_AMBIGUOUS'))
        self.assertEqual(rules.financial_identity(Company(**{**CAFE, 'activities': ['döviz alım satımı']})), 'activities: döviz alım satımı')
        self.assertEqual(rules.financial_identity(Company(**{**CAFE, 'licences': ['BDDK lisansı yoktur']})), '')


class ReportTests(unittest.TestCase):
    """Hook 3 and PROV-P5-2: the report names the t7 codes; a packet without them renders as before."""

    def test_the_withheld_clause_and_the_t7_codes_are_named(self):
        self.assertEqual(TRACE_GATES['CHILD_ADDRESSEE'], 'Alt bent muhatabı (yalnız yükümlüler)')
        self.assertIn('EXTRACTION_REVIEW_UNCERTAIN', REVIEW_FLAGS)
        self.assertTrue(set(precheck.POSITIVE_GATE_CODES) | {precheck.OPPOSED_EFFECT} <= set(POSITIVE_GATES))
        tedbirler = load_sources(TEDBIRLER)[1]
        labels = select_targets(tedbirler, 'YONETMELIK', '200713012', 'all', ['31'])
        settings = pipeline_settings(applicability_clear_match='rule', relevance_screen='off', coverage_pipeline='v18')
        judge = ProvisionJudge('APPLIES')
        packet = analyze(independent_profile('I08'), policies(), tedbirler, judge, labels, judge=judge, target_filter=['31'], settings=settings)
        html = render(packet)
        self.assertIn('Alt bent muhatabı (yalnız yükümlüler)', html)
        self.assertIn('OBLIGED_ADDRESSEE_UNDETERMINED', html)
        self.assertNotIn(WITHHELD_BY, render(analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])))


if __name__ == '__main__':
    unittest.main()

"""Autonomous decisions: the AI signs what passes the evidence gate and escalates the rest."""
import base64
import json
import time
import unittest
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
from regchain.evidence import verify_chain
from regchain.extraction.providers import RulesProvider
from regchain.pilot.engine import AUTONOMOUS_REVIEWER, analyze, autonomous_gate, autonomous_review
from regchain.pilot.report import render
from regchain.pilot.review import apply_review
from regchain.pilot.workspace import RunInput, Workspace, create_app
from test_pilot import FixtureProvider, company, policies, sections
from test_workspace import request_input, source_fixture


def many(count, conflict='Records are deleted every night.'):
    texts = [f'Staff must retain records in archive {i}.' for i in range(count)] + [conflict]
    return [dict(policies()[0], chunks=[dict(policies()[0]['chunks'][0], source_id=f'c{i:03d}', text=text) for i, text in enumerate(texts)])]


class Deleting(FixtureProvider):
    def passage(self, payload):
        if 'deleted' not in payload['passage']:
            return super().passage(payload)
        return {'relation': 'CONFLICTS', 'quote': 'Records are deleted every night.', 'reason': 'Fixture.'}


class Flaky(FixtureProvider):
    """Answers the support question differently on every reading of the same passage."""
    turn = 0

    def _chat(self, prompt, payload, schema):
        if 'supports' not in schema['properties']:
            return super()._chat(prompt, payload, schema)
        Flaky.turn += 1
        odd = Flaky.turn % 2 == 1
        return json.dumps({'supports': 'SUPPORTS' if odd else 'UNRELATED', 'quote': payload['passage'] if odd else '', 'reason': 'Fixture.'})


class GateTests(unittest.TestCase):
    def test_a_grounded_conflict_and_a_grounded_coverage_are_decided_by_the_ai(self):
        packet = analyze(company(), many(2), sections(), Deleting(), ['CONC 7.3.4'])
        review, summary = autonomous_review(packet)
        self.assertEqual((summary['decided'], summary['escalated']), (1, 0))
        self.assertEqual(review.reviewer, AUTONOMOUS_REVIEWER)
        decision = review.decisions[0]
        self.assertEqual((decision.extraction, decision.applicability, decision.coverage), ('ACCEPT', 'APPLIES', 'CONFLICT'))
        self.assertTrue(decision.rationale.startswith('Otonom AI kararı'))
        reviewed = apply_review(packet, review)
        self.assertTrue(verify_chain(reviewed['events'], reviewed['head'], 2))
        self.assertEqual(reviewed['events'][-1]['payload']['outcomes'][0]['status'], 'REVIEWED_POLICY_GAP')
        self.assertIn('Kaydedilmiş otonom AI kararı', render(reviewed))
        covered = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        review, summary = autonomous_review(covered)
        self.assertEqual(summary['decided'], 1)
        self.assertEqual(apply_review(covered, review)['events'][-1]['payload']['outcomes'][0]['status'], 'REVIEWED_POLICY_COVERAGE')

    def test_an_unclear_passage_or_an_unknown_applicability_is_escalated_not_guessed(self):
        class Unsure(FixtureProvider):
            def passage(self, payload):
                return {'relation': 'UNCLEAR', 'quote': '', 'reason': 'Fixture cannot decide.'}
        packet = analyze(company(), policies(), sections(), Unsure(), ['CONC 7.3.4'])
        row = packet['events'][0]['payload']['obligations'][0]
        self.assertIn('unclear or disputed', autonomous_gate(row))
        review, summary = autonomous_review(packet)
        self.assertEqual((summary['decided'], summary['escalated']), (0, 1))
        self.assertEqual(review.decisions[0].coverage, 'UNKNOWN')
        self.assertTrue(review.decisions[0].rationale.startswith('Otonom kapıdan geçmedi'))
        self.assertEqual(apply_review(packet, review)['events'][-1]['payload']['outcomes'][0]['status'], 'POLICY_REVIEW_REQUIRED')
        profile = company()
        profile.licences = None                                          # applicability must stay UNKNOWN
        packet = analyze(profile, policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        self.assertIn('Applicability could not be proposed', autonomous_gate(packet['events'][0]['payload']['obligations'][0]))
        self.assertEqual(apply_review(packet, autonomous_review(packet)[0])['events'][-1]['payload']['outcomes'][0]['status'],
                         'APPLICABILITY_REVIEW_REQUIRED')

    def test_repeated_readings_must_agree_or_the_passage_is_escalated(self):
        Flaky.turn = 0
        row = analyze(company(), policies(), sections(), Flaky(), ['CONC 7.3.4'], votes=2)['events'][0]['payload']['obligations'][0]
        check = row['proposal']['policy_checks'][0]
        self.assertEqual(check['relation'], 'UNCLEAR')
        self.assertEqual(row['diagnostics'][-1]['results'][0]['votes'], ['SUPPORTS', 'UNRELATED'])
        self.assertIn('disagreed', row['diagnostics'][-1]['results'][0]['reason'])
        self.assertEqual(row['proposal']['coverage'], 'UNKNOWN')
        self.assertIsNotNone(autonomous_gate(row))
        steady = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'], votes=3)['events'][0]['payload']
        self.assertEqual(steady['obligations'][0]['diagnostics'][-1]['results'][0]['votes'], ['SUPPORTS']*3)
        self.assertEqual(steady['obligations'][0]['proposal']['coverage'], 'COVERS_TEXT')
        self.assertEqual(steady['judge_runtime']['votes'], 3)


class WorkspaceAutonomyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name)/'workspace', provider_factory=lambda _: FixtureProvider(), source_fetcher=source_fixture)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767', client=('127.0.0.1', 51000),
                                 headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def finish(self, **changes):
        run_id = self.client.post('/api/runs', json=request_input(**changes)).json()['id']
        for _ in range(400):
            row = self.workspace.metadata(run_id)
            if row['state'] in ('COMPLETED', 'FAILED') and not self.workspace.active:
                return row
            time.sleep(.01)
        self.fail('Local fixture job did not finish')

    def test_the_autonomous_run_records_the_ai_review_as_a_separate_verifiable_event(self):
        row = self.finish(provider='ollama', decision_mode='autonomous', judge_votes=2)
        self.assertEqual(row['state'], 'COMPLETED', row['error'])
        self.assertEqual((row['decision_mode'], row['judge_votes']), ('autonomous', 2))
        self.assertEqual(row['autonomy']['decided'] + row['autonomy']['escalated'], row['candidates'])
        self.assertEqual(len(row['reviews']), 1)
        self.assertEqual(row['reviews'][0]['reviewer'], AUTONOMOUS_REVIEWER)
        verified = self.client.get('/api/runs/'+row['id']+'/verify', params={'review_id': row['autonomy']['review_id']})
        self.assertEqual(verified.status_code, 200, verified.text)
        report = self.client.get('/api/runs/'+row['id']+'/report', params={'review_id': row['autonomy']['review_id']}).text
        self.assertIn('otonom AI kararı', report)
        # The original analysis is untouched: a person can still review it separately.
        self.assertEqual(self.client.get('/api/runs/'+row['id']+'/packet').json()['count'], 1)
        plain = self.finish(provider='ollama')
        self.assertNotIn('autonomy', plain)
        self.assertEqual(plain['reviews'], [])
        page = self.client.get('/').text
        for needle in ('id="decisionMode"', 'name="judge_votes"', 'Otonom AI kararı'):
            self.assertIn(needle, page)

    def test_votes_are_bounded(self):
        for bad in (0, 4):
            self.assertEqual(self.client.post('/api/runs', json=request_input(judge_votes=bad)).status_code, 422)
        self.assertEqual(RunInput.model_validate(request_input()).judge_votes, 1)


if __name__ == '__main__':
    unittest.main()

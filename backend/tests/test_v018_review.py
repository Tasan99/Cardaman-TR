"""v0.18 human review: the applicability decision trace in the three-column card, and the audit of what
the reviewer did with the AI proposal (who, when, which action, why, and the untouched AI proposal)."""
import copy
import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from regchain.evidence import make_event, verify_chain
from regchain.extraction.providers import RulesProvider
from regchain.pilot.report import render
from regchain.pilot.review import apply_review
from regchain.pilot.schema import Review
from regchain.pilot.workspace import Workspace, create_app
from test_pilot import packet, review_input
from test_workspace import request_input, source_fixture, uncertain_review


GATE_LABELS = ('Regülasyon yükümlü kapsamı', 'Şirket varlık türü', 'Yargı alanı', 'Müşteri türü', 'Alt bent', 'İstisna', 'Model', 'Nihai karar')
# The keys the design contract adds to proposal.trace in v0.18; a v0.17 packet has none of them.
V018_KEYS = ('gates', 'decided_by', 'subject_gate', 'jurisdiction_gate', 'exemption_check', 'model_gate', 'applies_trace', 'human_review')


def gate(name, status, clear=True, reason='', **extra):
    return {'gate': name, 'status': status, 'clear': clear, 'reason': reason, **extra}


# Shaped as applicability.gate() writes it: the extras (matched_items ...) sit under 'evidence', and a
# matched subject gate quotes the list item it names.
SUBJECT = gate('REGULATION_SUBJECT_SCOPE', 'MATCH', reason='Consumer credit lending is named among the obliged parties.',
               evidence={'source_label': 'CONC 7.1.1', 'source_id': 's-scope', 'quote': 'consumer credit lending',
                         'company_field': 'activities', 'company_value': 'consumer credit lending', 'list_item': 'consumer credit lending',
                         'matched_items': ['(a)']})
CLEAR_MATCH = {
    'gates': [SUBJECT,
              gate('COMPANY_ENTITY', 'NOT_RESTRICTED', reason='The clause names no obliged entity type.'),
              gate('JURISDICTION', 'NOT_RESTRICTED', reason='The rule names no jurisdiction.'),
              gate('CUSTOMER_ENTITY', 'NOT_RESTRICTED', reason='No counterparty restriction.'),
              gate('CHILD_CLAUSE', 'NOT_RESTRICTED', reason='Whole provision.',
                   evidence={'child_clause': 'whole provision', 'quote': 'This chapter applies to consumer credit lending.'}),
              gate('EXEMPTION', 'NONE', reason='No exemption wording <b>found</b>.'),
              gate('MODEL', 'NOT_ASKED', reason='Every gate was a clear match; decided by rule.', mode='rule'),
              gate('FINAL_AGGREGATOR', 'APPLIES', reason='All deterministic gates matched.')],
    'decided_by': 'RULE_CLEAR_MATCH', 'subject_gate': SUBJECT,
    'model_gate': gate('MODEL', 'NOT_ASKED', reason='Every gate was a clear match; decided by rule.', mode='rule'),
    'applies_trace': {'regulation_subject_basis': 'CONC 7.1.1', 'company_matching_field': 'activities', 'jurisdiction_basis': None,
                      'child_clause_match': 'whole provision', 'exclusion_check': 'none found', 'final_reasoning': 'rule',
                      'complete': False, 'missing': ['jurisdiction_basis']},
    'human_review': True}
RULED_OUT = {
    'gates': [gate('REGULATION_SUBJECT_SCOPE', 'MISMATCH', reason='The company is not in the obliged-party list.',
                   evidence={'source_label': 'CONC 7.1.1', 'source_id': 's-scope', 'company_field': 'activities',
                             'company_value': 'activities: software development'}),
              *[gate(name, 'NOT_ASKED', clear=False, reason='Not reached.') for name in
                ('COMPANY_ENTITY', 'JURISDICTION', 'CUSTOMER_ENTITY', 'CHILD_CLAUSE', 'EXEMPTION', 'MODEL')],
              gate('FINAL_AGGREGATOR', 'DOES_NOT_APPLY', reason='A clear mismatch settles the row.')],
    'decided_by': 'SUBJECT_SCOPE_GATE',
    'applies_trace': {'complete': False, 'missing': ['regulation_subject_basis']}, 'human_review': False}


def resealed(value, payload):
    event = make_event(payload)
    return {'format': value['format'], 'events': [event], 'head': event['event_hash'], 'count': 1}


def with_trace(value, trace, flags, rule, **changes):
    """The fixture packet with a v0.18 trace written into its proposal, sealed again so the chain verifies."""
    payload = copy.deepcopy(value['events'][0]['payload'])
    proposal = payload['obligations'][0]['proposal']
    proposal.update(trace={**proposal['trace'], **trace}, review_flags=flags, applicability_rule=rule, **changes)
    return resealed(value, payload)


def v017(value, flags=()):
    """The same packet as v0.17 wrote it, whatever the engine now adds: no v0.18 trace keys."""
    payload = copy.deepcopy(value['events'][0]['payload'])
    for row in payload['obligations']:
        row['proposal']['trace'] = {k: v for k, v in (row['proposal'].get('trace') or {}).items() if k not in V018_KEYS}
        row['proposal']['review_flags'] = list(flags)
    return resealed(value, payload)


def right_column(page):
    return page.split('<h4>AI önerisi ve karar</h4>', 1)[1].split('</section>', 1)[0]


class DecisionTraceCardTests(unittest.TestCase):
    def test_the_card_lists_every_gate_in_order_with_status_evidence_decider_and_flag(self):
        page = render(with_trace(packet(), CLEAR_MATCH, ['APPLIES_TRACE_INCOMPLETE'], 'RULE_CLEAR_MATCH'))
        column = right_column(page)
        self.assertIn('<summary>Karar izi</summary>', column)
        positions = [column.index(f'<b>{text}</b>') for text in GATE_LABELS]
        self.assertEqual(positions, sorted(positions))
        for text in ('eşleşiyor · MATCH', 'sınırlama yok · NOT_RESTRICTED', 'uygulanıyor · APPLIES', '<span class="muted">kesin</span>',
                     'mod: rule', 'liste maddesi: “consumer credit lending” · eşleşen: (a) · kaynak: CONC 7.1.1',
                     'şirket: activities = “consumer credit lending”', '“This chapter applies to consumer credit lending.”',
                     '<b>Karar veren:</b> kural: bütün kapılar açıkça eşleşti (RULE_CLEAR_MATCH)',
                     '<b>İnsan incelemesi gerekli:</b> APPLIES izinde kritik alan eksik (APPLIES_TRACE_INCOMPLETE)',
                     'yargı alanı dayanağı (jurisdiction_basis)'):
            self.assertIn(text, column, text)
        # The list item the subject gate also quotes is printed once, beside the company value.
        self.assertEqual(column.count('“consumer credit lending”'), 2)
        # Gate text is data: escaped like every other document string.
        self.assertIn('No exemption wording &lt;b&gt;found&lt;/b&gt;.', column)
        self.assertNotIn('<b>found</b>', page)
        # The trace sits above the decision; the four actions, the reason box and the entity-gate table stay.
        self.assertLess(column.index('Karar izi'), column.index('data-field="action"'))
        for text in ('value="APPROVE"', 'value="REJECT"', 'value="OVERRIDE"', 'value="NEEDS_EVIDENCE"', 'data-field="override_reason"'):
            self.assertIn(text, column, text)
        self.assertIn('Uygulanabilirlik dayanağı — bent düzeyi (6 alan)', page)

    def test_a_clear_mismatch_names_the_gate_that_ruled_the_duty_out(self):
        page = render(with_trace(packet(), RULED_OUT, [], 'SUBJECT_SCOPE_GATE', applicability='DOES_NOT_APPLY', coverage_assessed=False))
        column = right_column(page)
        self.assertIn('<span class="badge no">eşleşmiyor · MISMATCH</span>', column)
        self.assertIn('<span class="badge no">uygulanmıyor · DOES_NOT_APPLY</span>', column)
        self.assertIn('sorulmadı · NOT_ASKED', column)
        self.assertIn('(SUBJECT_SCOPE_GATE)', column)
        # A gate that quotes nothing still names the list it read and the profile value it compared.
        self.assertIn('şirket: activities = “activities: software development” · kaynak: CONC 7.1.1', column)
        # No flag, and an incomplete APPLIES trace means nothing on a DOES_NOT_APPLY row.
        self.assertNotIn('İnsan incelemesi gerekli', column)
        self.assertNotIn('APPLIES izi eksik', column)

    def test_the_engine_trace_renders_as_it_is_written(self):
        # The fixture packet as engine.propose now writes it, not a hand-built trace.
        value = packet()
        proposal = value['events'][0]['payload']['obligations'][0]['proposal']
        column = right_column(render(value))
        positions = [column.index(f'<b>{text}</b>') for text in GATE_LABELS]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(column.count('<li>'), len(proposal['trace']['gates']))
        self.assertIn('<b>Karar veren:</b> ', column)
        self.assertIn(f'({proposal["trace"]["decided_by"]})', column)
        self.assertIn(f'mod: {proposal["trace"]["model_gate"]["mode"]}', column)

    def test_a_v017_packet_shows_no_trace_and_keeps_every_earlier_string(self):
        value = v017(packet())
        page = render(value)
        self.assertNotIn('Karar izi', page)
        self.assertNotIn('class="trace"', page)
        self.assertNotIn('Karar veren', page)
        for text in ('class="tri"', '<h4>Regülasyon</h4>', '<h4>Policy kanıtı</h4>', '<h4>AI önerisi ve karar</h4>', 'data-field="action"',
                     'value="APPROVE"', 'value="REJECT"', 'value="OVERRIDE"', 'value="NEEDS_EVIDENCE"', 'data-field="override_reason"',
                     '"proposals":', 'Uygulanabilirlik dayanağı — bent düzeyi (6 alan)', 'Bölüm özeti'):
            self.assertIn(text, page, text)
        review = review_input(value)
        review['decisions'][0].update(action='OVERRIDE', override_reason='Policy saklama süresini belirtmiyor.', coverage='PARTIAL')
        saved = render(apply_review(value, Review.model_validate(review)))
        for text in ('Denetim kaydı', 'override gerekçesi', 'Kaydedilmiş insan incelemesi'):
            self.assertIn(text, saved, text)
        self.assertNotIn('Karar izi', saved)
        self.assertNotIn('AI kararını veren', saved)


class ReviewAuditTests(unittest.TestCase):
    def setUp(self):
        self.value = with_trace(packet(), CLEAR_MATCH, ['APPLIES_TRACE_INCOMPLETE'], 'RULE_CLEAR_MATCH')

    def test_override_needs_a_reason_and_the_audit_keeps_the_ai_proposal_and_its_trace(self):
        review = review_input(self.value)
        review['decisions'][0].update(action='OVERRIDE', coverage='PARTIAL')
        for blank in ('', '   '):
            review['decisions'][0]['override_reason'] = blank
            with self.assertRaisesRegex(ValueError, 'override_reason'):
                Review.model_validate(review)
        review['decisions'][0]['override_reason'] = 'Policy metni saklama süresini belirtmiyor.'
        reviewed = apply_review(self.value, Review.model_validate(review))
        entry = reviewed['events'][-1]['payload']['audit'][0]
        self.assertEqual((entry['reviewer'], entry['reviewer_role'], entry['action'], entry['override_reason'], entry['changed_fields']),
                         ('Fixture reviewer', 'Synthetic test actor', 'OVERRIDE', 'Policy metni saklama süresini belirtmiyor.', ['coverage']))
        self.assertIsNotNone(datetime.fromisoformat(entry['at']).tzinfo)
        self.assertEqual(entry['at'], reviewed['events'][-1]['payload']['created_at'])
        self.assertEqual(entry['ai_proposal'], {'applicability': 'APPLIES', 'coverage': 'COVERS_TEXT'})
        self.assertEqual(entry['human_decision']['coverage'], 'PARTIAL')
        self.assertEqual(entry['ai_trace'], {'decided_by': 'RULE_CLEAR_MATCH', 'review_flags': ['APPLIES_TRACE_INCOMPLETE']})
        # The AI proposal itself is never edited: the review is a second event on the same chain.
        self.assertEqual(reviewed['events'][0], self.value['events'][0])
        self.assertTrue(verify_chain(reviewed['events'], reviewed['head'], 2))
        page = render(reviewed)
        self.assertIn('AI kararını veren: RULE_CLEAR_MATCH · AI işaretleri: APPLIES_TRACE_INCOMPLETE', page)
        self.assertIn('<summary>Karar izi</summary>', page)                          # the saved view keeps the trace
        self.assertNotIn('<select data-field=', page)

    def test_approve_keeps_the_ai_proposal_and_a_changed_verdict_is_refused(self):
        approve = review_input(self.value)
        approve['decisions'][0]['action'] = 'APPROVE'
        entry = apply_review(self.value, Review.model_validate(approve))['events'][-1]['payload']['audit'][0]
        self.assertEqual((entry['action'], entry['changed_fields'], entry['override_reason']), ('APPROVE', [], None))
        self.assertEqual(entry['ai_proposal'], {'applicability': 'APPLIES', 'coverage': 'COVERS_TEXT'})
        self.assertEqual((entry['human_decision']['applicability'], entry['human_decision']['coverage']), ('APPLIES', 'COVERS_TEXT'))
        self.assertEqual(entry['ai_trace']['decided_by'], 'RULE_CLEAR_MATCH')
        approve['decisions'][0]['coverage'] = 'PARTIAL'
        with self.assertRaisesRegex(ValueError, 'OVERRIDE'):
            apply_review(self.value, Review.model_validate(approve))

    def test_a_v017_proposal_records_ai_trace_only_when_it_carries_flags(self):
        plain = v017(packet())
        entry = apply_review(plain, Review.model_validate(review_input(plain)))['events'][-1]['payload']['audit'][0]
        self.assertNotIn('ai_trace', entry)
        self.assertEqual({'reviewer', 'reviewer_role', 'at', 'action', 'ai_proposal', 'override_reason'} - set(entry), set())
        flagged = v017(packet(), ['RULE_MODEL_DISAGREEMENT'])
        entry = apply_review(flagged, Review.model_validate(review_input(flagged)))['events'][-1]['payload']['audit'][0]
        self.assertEqual(entry['ai_trace'], {'decided_by': None, 'review_flags': ['RULE_MODEL_DISAGREEMENT']})


class ReviewEndToEndTests(unittest.TestCase):
    """Through the workspace API, as the hosted review page posts it."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source_fixture(root/'cached')
        self.workspace = Workspace(root/'workspace', root/'cached', provider_factory=lambda _: RulesProvider(), source_fetcher=source_fixture)
        self.client = TestClient(create_app(self.workspace, 'test-only-token'), base_url='http://127.0.0.1:8767',
                                 client=('127.0.0.1', 51000), headers={'Authorization': 'Bearer test-only-token'})
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_an_override_without_a_reason_is_refused_and_one_with_a_reason_is_audited(self):
        run_id = self.client.post('/api/runs', json=request_input()).json()['id']
        for _ in range(300):
            if self.workspace.metadata(run_id)['state'] in ('COMPLETED', 'FAILED', 'INTERRUPTED') and not self.workspace.active:
                break
            time.sleep(.01)
        self.assertEqual(self.workspace.metadata(run_id)['state'], 'COMPLETED')
        value = self.client.get(f'/api/runs/{run_id}/packet').json()
        review = uncertain_review(value)
        for decision in review['decisions']:
            decision.update(action='OVERRIDE', override_reason='')
        refused = self.client.post(f'/api/runs/{run_id}/reviews', json=review)
        self.assertEqual(refused.status_code, 422)
        self.assertIn('override_reason', json.dumps(refused.json(), ensure_ascii=False))
        self.assertEqual(self.workspace.metadata(run_id).get('reviews', []), [])
        for decision in review['decisions']:
            decision['override_reason'] = 'Kaynak metin bu şirket türünü açıkça dışarıda bırakıyor.'
        saved = self.client.post(f'/api/runs/{run_id}/reviews', json=review)
        self.assertEqual(saved.status_code, 200, saved.text)
        reviewed = self.client.get(f'/api/runs/{run_id}/packet', params={'review_id': saved.json()['id']}).json()
        proposals = {o['id']: o['proposal'] for o in value['events'][0]['payload']['obligations']}
        audit = reviewed['events'][-1]['payload']['audit']
        self.assertEqual(sorted(entry['obligation_id'] for entry in audit), sorted(proposals))
        self.assertTrue(audit)
        for entry in audit:
            self.assertEqual((entry['reviewer'], entry['action'], entry['override_reason']),
                             ('Automated test only', 'OVERRIDE', 'Kaynak metin bu şirket türünü açıkça dışarıda bırakıyor.'))
            self.assertTrue(entry['at'])
            ai = proposals[entry['obligation_id']]
            self.assertEqual(entry['ai_proposal'], {'applicability': ai['applicability'], 'coverage': ai['coverage']})
            # ai_trace is written exactly when the proposal carries a decided_by or a flag, and copies them.
            self.assertEqual(entry.get('ai_trace', {}).get('decided_by'), ai['trace'].get('decided_by'))
            self.assertEqual(entry.get('ai_trace', {}).get('review_flags', []), ai.get('review_flags', []))
        self.assertEqual(reviewed['events'][0], value['events'][0])
        page = self.client.get(f'/api/runs/{run_id}/report', params={'review_id': saved.json()['id']}).text
        self.assertIn('Denetim kaydı', page)
        self.assertIn('override gerekçesi: Kaynak metin bu şirket türünü açıkça dışarıda bırakıyor.', page)


if __name__ == '__main__':
    unittest.main()

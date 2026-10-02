"""QDMS export (tr/qdms.py): every gap row becomes a change request that waits for a person; only an approval bound to
the row as it was reviewed makes its actions ready for the QDMS. Registers and the board decision here are SYNTHETIC."""
import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from regchain.tr import qdms
from regchain.tr.adjudicate import assess_profile
from regchain.tr.compare import load_register
from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles

REGISTRY = Registry.load()
STORE = CorpusStore()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
BREWER = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
REGISTER = load_register(BREWER.profile_id)
OBLIGATIONS = extract_regulation('TR:KANUN:4250', REGISTRY, STORE, articles=['6'])[1]
REPORT, ASSESSMENTS, _ = assess_profile(BREWER, OBLIGATIONS, REGISTER, REGISTRY, STORE)
WHEN = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def export(include_covered=False, store=STORE):
    return qdms.export_rows(BREWER, REGISTER, REGISTRY, {'BEVERAGE_ALCOHOL_TR': REPORT}, {'BEVERAGE_ALCOHOL_TR': ASSESSMENTS},
                            include_covered, generated_at=WHEN, store=store)


def entry(row, decision, fingerprint=None):
    return {'row_id': row.row_id, 'fingerprint': fingerprint or row.fingerprint, 'decision': decision, 'approver': 'uyum.sorumlusu',
            'decided_at': '2026-10-02T13:00:00+00:00', 'note': 'test'}


class ExportTests(unittest.TestCase):
    def test_every_row_is_a_draft_that_waits_for_a_person(self):
        result = export()
        self.assertTrue(result.rows)
        self.assertEqual(result.summary['by_approval'], {'PENDING': len(result.rows)})
        self.assertEqual(result.summary['ready_for_qdms'], 0)
        self.assertEqual(qdms.ready_actions(result), [])
        for row in result.rows:
            self.assertTrue(row.human_approval.required)
            self.assertTrue(all(a.state == 'DRAFT' for a in row.required_actions))
            self.assertTrue(row.provision['quote'])
            self.assertTrue(row.provision['version_id'], 'the row names the stored version it was read from')
            self.assertEqual(row.provision['version_id'], STORE.head('TR:KANUN:4250').version_id)
            self.assertTrue(row.entity['entity_ids'], row.row_id)
            self.assertEqual(row.entity['entity_name'], '; '.join(BREWER.entity(e).name for e in row.entity['entity_ids']))
            if row.entity['target_level'] == 'PRODUCT':
                self.assertEqual(row.entity['entity_ids'], [e.entity_id for e in BREWER.legal_entities
                                                           if row.entity['target_id'] in e.product_ids])
            self.assertTrue(row.impacted_process['departments'])

    def test_a_contradiction_asks_for_the_conflicting_statement_to_be_revised(self):
        contradicted = [r for r in export().rows if r.policy_status['mapping_status'] == 'CONTRADICTED']
        self.assertTrue(contradicted)
        for row in contradicted:
            change = [a for a in row.required_actions if a.change == 'REVISE_CONFLICTING_STATEMENT']
            self.assertTrue(change, row.row_id)
            for action in change:
                self.assertEqual(action.qdms_type, 'DOCUMENT_CHANGE_REQUEST')
                self.assertEqual(action.priority, 'HIGH')
                self.assertIn(action.target_document, REGISTER.documents and {d.document_id for d in REGISTER.documents})
                conflicting = {s['passage_id'] for s in row.reason['statements'] if s['relation'] == 'CONFLICTS'}
                self.assertIn(action.target, conflicting)

    def test_covered_automatic_rows_are_left_out_unless_asked(self):
        short, full = export(), export(include_covered=True)
        covered = [r for r in REPORT.rows if r.mapping.status == 'COVERED']
        self.assertTrue(covered)
        self.assertNotIn('COVERED', short.summary['by_policy_status'])
        self.assertEqual(full.summary['by_policy_status'].get('COVERED'), len(covered))
        self.assertEqual(len(full.rows), len(REPORT.rows))

    def test_an_open_decision_leads_with_a_review_task(self):
        row, assessment = next((r, a) for r, a in zip(REPORT.rows, ASSESSMENTS) if r.mapping.status == 'NOT_COVERED')
        opened = assessment.model_copy(update={'decision': 'REVIEW_REQUIRED', 'review_reasons': ['MODEL_PROPOSES_COVERED'],
                                               'proposal': 'COVERS_TEXT'})
        qrow = qdms.export_row(row, opened, BREWER, REGISTER, REGISTRY)
        self.assertEqual((qrow.required_actions[0].qdms_type, qrow.required_actions[0].change), ('COMPLIANCE_REVIEW_TASK', 'DECIDE_COVERAGE'))
        self.assertEqual(qrow.reason['model_proposal'], 'COVERS_TEXT')
        self.assertEqual(qrow.reason['decision'], 'REVIEW_REQUIRED')
        plain = qdms.export_row(row, assessment, BREWER, REGISTER, REGISTRY)
        self.assertNotEqual(plain.fingerprint, qrow.fingerprint)


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.export = export()
        self.rows = self.export.rows

    def apply(self, entries):
        return qdms.apply_approvals(self.export, qdms.Approvals.model_validate(
            {'format': qdms.APPROVALS_FORMAT, 'profile_id': BREWER.profile_id, 'entries': entries}))

    def test_only_approved_rows_reach_the_qdms(self):
        approved, rejected, returned = self.rows[0], self.rows[1], self.rows[2]
        result, problems = self.apply([entry(approved, 'APPROVED'), entry(rejected, 'REJECTED'), entry(returned, 'RETURNED')])
        self.assertEqual(problems, [])
        by_id = {r.row_id: r for r in result.rows}
        self.assertEqual({a.state for a in by_id[approved.row_id].required_actions}, {'READY_FOR_QDMS'})
        self.assertEqual({a.state for a in by_id[rejected.row_id].required_actions}, {'CANCELLED'})
        self.assertEqual({a.state for a in by_id[returned.row_id].required_actions}, {'DRAFT'})
        self.assertEqual(by_id[approved.row_id].human_approval.approver, 'uyum.sorumlusu')
        ready = qdms.ready_actions(result)
        self.assertEqual({a['row_id'] for a in ready}, {approved.row_id})
        self.assertEqual(len(ready), len(approved.required_actions))
        self.assertEqual(result.summary['by_approval']['PENDING'], len(self.rows) - 3)

    def test_an_approval_of_a_row_that_changed_since_is_refused(self):
        row = self.rows[0]
        result, problems = self.apply([entry(row, 'APPROVED', fingerprint='0' * 64)])
        self.assertEqual([p['problem'] for p in problems], ['STALE_APPROVAL'])
        self.assertEqual(qdms.ready_actions(result), [])
        self.assertEqual(next(r for r in result.rows if r.row_id == row.row_id).human_approval.status, 'PENDING')
        # a new version of the regulation's text changes the fingerprint of every row read from it

        class NewVersion:
            def head(self, regulation_id):
                return type('V', (), {'version_id': 'TR:KANUN:4250@yeni'})()
        newer = export(store=NewVersion())
        self.assertNotEqual(newer.rows[0].fingerprint, row.fingerprint)
        refreshed, problems = qdms.apply_approvals(newer, qdms.Approvals.model_validate(
            {'format': qdms.APPROVALS_FORMAT, 'profile_id': BREWER.profile_id, 'entries': [entry(row, 'APPROVED')]}))
        self.assertEqual([p['problem'] for p in problems], ['STALE_APPROVAL'])

    def test_unknown_rows_double_decisions_and_another_profile_are_refused(self):
        row = self.rows[0]
        _, problems = self.apply([{**entry(row, 'APPROVED'), 'row_id': 'yok:yok'}, entry(row, 'APPROVED'), entry(row, 'REJECTED')])
        self.assertEqual(sorted(p['problem'] for p in problems), ['DECIDED_TWICE', 'UNKNOWN_ROW'])
        with self.assertRaises(ValueError):
            qdms.apply_approvals(self.export, qdms.Approvals(format=qdms.APPROVALS_FORMAT, profile_id='baska', entries=[]))
        with self.assertRaises(ValueError):                      # an approval names who decided
            qdms.ApprovalEntry.model_validate({**entry(row, 'APPROVED'), 'approver': ''})


class FileTests(unittest.TestCase):
    def test_the_files_round_trip_and_a_blank_template_decides_nothing(self):
        result = export()
        with tempfile.TemporaryDirectory() as directory:
            paths = qdms.write(result, Path(directory))
            self.assertEqual(qdms.load_export(paths['json']).rows, result.rows)
            raw = paths['csv'].read_bytes()
            self.assertTrue(raw.startswith(b'\xef\xbb\xbf'))   # a spreadsheet opens it as UTF-8
            with paths['csv'].open(encoding='utf-8-sig', newline='') as handle:
                lines = list(csv.DictReader(handle, delimiter=';'))
            self.assertEqual(len(lines), sum(max(1, len(r.required_actions)) for r in result.rows))
            self.assertEqual(set(lines[0]), set(qdms.CSV_COLUMNS))
            self.assertEqual({l['action_state'] for l in lines if l['action_type']}, {'DRAFT'})
            self.assertEqual(json.loads(paths['ready'].read_text(encoding='utf-8')), [])
            template = json.loads(paths['template'].read_text(encoding='utf-8'))
            self.assertEqual(len(template['entries']), len(result.rows))
            self.assertEqual(qdms.load_approvals(paths['template']).entries, [])
            template['entries'][0].update({'decision': 'APPROVED', 'approver': 'kalite.muduru', 'decided_at': '2026-10-02T14:00:00+00:00'})
            filled = Path(directory) / 'filled.json'
            filled.write_text(json.dumps(template, ensure_ascii=False), encoding='utf-8')
            approved, problems = qdms.apply_approvals(result, qdms.load_approvals(filled))
            self.assertEqual(problems, [])
            self.assertEqual({a['row_id'] for a in qdms.ready_actions(approved)}, {result.rows[0].row_id})


class DecisionLayerExportTests(unittest.TestCase):
    def test_a_board_decision_exports_as_its_own_layer_and_every_row_waits_for_review(self):
        from test_tr_decisions import DECISION_ID, TEXT, record_for
        from regchain.tr.decisions import DecisionStore, LayeredStore, assess_decision, decision_meta, registry_with
        profile = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
        register = load_register(profile.profile_id)
        with tempfile.TemporaryDirectory() as directory:
            decisions = DecisionStore(Path(directory))
            decisions.write(record_for(TEXT), TEXT)
            record = decisions.record(DECISION_ID)
            report, assessments, _ = assess_decision(record, profile, REGISTRY, STORE, decisions, register, named=['NONALC-SALES'])
            layer = qdms.export_rows(profile, register, registry_with(REGISTRY, [decision_meta(record)]), {DECISION_ID: report},
                                     {DECISION_ID: assessments}, True, 'DECISION', WHEN, LayeredStore(STORE, decisions))
        self.assertEqual(len(layer.rows), len(report.rows))
        for row in layer.rows:
            self.assertEqual((row.source_layer, row.entity['entity_id']), ('DECISION', 'NONALC-SALES'))
            self.assertEqual(row.provision['binding_status'], 'DECISION_PRECEDENT')
            self.assertEqual(row.provision['version_id'], record.version_id)
            self.assertEqual(row.reason['decision'], 'REVIEW_REQUIRED')
            self.assertEqual(row.required_actions[0].qdms_type, 'COMPLIANCE_REVIEW_TASK')
        merged = qdms.merge(qdms.export_rows(profile, register, REGISTRY, {}, {}, generated_at=WHEN), layer)
        self.assertEqual(merged.summary['by_layer'], {'DECISION': len(layer.rows)})
        with self.assertRaises(ValueError):
            qdms.merge(export(), layer)                          # another profile


if __name__ == '__main__':
    unittest.main()

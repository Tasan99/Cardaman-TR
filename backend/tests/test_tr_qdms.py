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
            self.assertEqual(len(template['entries']), len(result.rows) + len(result.review_tasks))
            self.assertEqual(qdms.load_approvals(paths['template']).entries, [])
            template['entries'][0].update({'decision': 'APPROVED', 'approver': 'kalite.muduru', 'decided_at': '2026-10-02T14:00:00+00:00'})
            filled = Path(directory) / 'filled.json'
            filled.write_text(json.dumps(template, ensure_ascii=False), encoding='utf-8')
            approved, problems = qdms.apply_approvals(result, qdms.load_approvals(filled))
            self.assertEqual(problems, [])
            self.assertEqual({a['row_id'] for a in qdms.ready_actions(approved)}, {result.rows[0].row_id})


class UnknownApplicabilityTests(unittest.TestCase):
    """A duty whose applicability is UNKNOWN is neither lost nor turned into a policy action: routing -> report ->
    analysis statistics -> QDMS review section -> CSV -> approval, the same records all the way."""

    def test_every_unknown_decision_reaches_the_report_and_the_export_as_a_review(self):
        from regchain.tr.extraction import route
        unknown = {(o.obligation_id, d.target_id) for o in OBLIGATIONS for d in route(o, BREWER, REGISTRY, STORE)[0] if d.status == 'UNKNOWN'}
        self.assertTrue(unknown, 'the fixture needs an UNKNOWN decision')
        self.assertEqual({(r.obligation_id, r.target_id) for r in REPORT.applicability_reviews}, unknown)
        self.assertEqual(REPORT.summary['applicability_unknown'], len(unknown))
        result = export()
        self.assertEqual({r.row_id for r in result.applicability_reviews}, {f'APPLICABILITY:{o}:{t}' for o, t in unknown})
        self.assertEqual(result.summary['applicability_reviews'], len(unknown))
        # kept apart from the policy rows: no policy status, no coverage, no document change
        policy_keys = {r.row_id for r in result.rows}
        self.assertFalse(policy_keys & {f'{o}:{t}' for o, t in unknown})
        for item in result.applicability_reviews:
            self.assertEqual(item.applicability['status'], 'UNKNOWN')
            self.assertFalse(item.coverage_assessed)
            self.assertTrue(item.entity['entity_ids'])
            self.assertTrue(item.provision['regulation_id'] and item.provision['provision_ref'] and item.provision['quote'])
            self.assertEqual(item.provision['version_id'], STORE.head(item.provision['regulation_id']).version_id)
            self.assertTrue(item.applicability['reason_codes'])
            self.assertIn(item.uncertainty['review_type'], ('COMPLETE_PROFILE_FACT', 'RESOLVE_PROFILE_CONFLICT', 'CLARIFY_REGULATORY_SCOPE',
                                                            'CHECK_SOURCE_GROUNDING'))
            if item.uncertainty['review_type'] == 'COMPLETE_PROFILE_FACT':
                self.assertTrue(item.uncertainty['missing_facts'], item.row_id)        # the missing fact is named
                for fact in item.uncertainty['missing_facts']:
                    if fact['gate'] not in ('PRODUCT_SCOPE', 'EXCEPTIONS'):
                        # what the duty requires and whether the profile's list is complete, not only the gate's name
                        self.assertIn('required', fact, item.row_id)
                        self.assertIn('complete', fact, item.row_id)
            self.assertTrue(item.uncertainty['source_evidence']['quote'])
            self.assertEqual(item.required_actions, [])                         # grouped: the action is the task's
            self.assertTrue(item.task_id)
            self.assertEqual(item.human_approval.status, 'PENDING')
        task_actions = [a for task in result.review_tasks for a in task.required_actions]
        self.assertTrue(task_actions)
        self.assertEqual({a.qdms_type for a in task_actions} - {'PROFILE_DATA_REQUEST', 'APPLICABILITY_REVIEW_TASK'}, set())
        self.assertEqual({a.state for a in task_actions}, {'DRAFT'})
        self.assertEqual(qdms.ready_actions(result), [])

    def test_the_review_section_is_written_and_approved_through_its_tasks(self):
        result = export()
        item = result.applicability_reviews[0]
        task = next(t for t in result.review_tasks if t.task_id == item.task_id)
        with tempfile.TemporaryDirectory() as directory:
            paths = qdms.write(result, Path(directory))
            with paths['reviews_csv'].open(encoding='utf-8-sig', newline='') as handle:
                lines = list(csv.DictReader(handle, delimiter=';'))
            self.assertEqual(len(lines), len(result.applicability_reviews))
            self.assertEqual({l['applicability'] for l in lines}, {'UNKNOWN'})
            self.assertEqual({l['task_id'] for l in lines}, {t.task_id for t in result.review_tasks})
            with paths['tasks_csv'].open(encoding='utf-8-sig', newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle, delimiter=';'))), len(result.review_tasks))
            loaded = qdms.load_export(paths['json'])
            self.assertEqual((loaded.applicability_reviews, loaded.review_tasks), (result.applicability_reviews, result.review_tasks))
        # a child record is decided through its task, never alone
        _, problems = qdms.apply_approvals(result, qdms.Approvals.model_validate(
            {'format': qdms.APPROVALS_FORMAT, 'profile_id': BREWER.profile_id, 'entries': [entry(item, 'APPROVED')]}))
        self.assertEqual([p['problem'] for p in problems], ['APPROVE_THE_TASK'])
        approved, problems = qdms.apply_approvals(result, qdms.Approvals.model_validate(
            {'format': qdms.APPROVALS_FORMAT, 'profile_id': BREWER.profile_id,
             'entries': [{'row_id': task.task_id, 'fingerprint': task.fingerprint, 'decision': 'APPROVED', 'approver': 'hukuk.sorumlusu',
                          'decided_at': '2026-10-04T10:00:00+00:00', 'note': 'test'}]}))
        self.assertEqual(problems, [])
        ready = qdms.ready_actions(approved)
        self.assertEqual([a['row_id'] for a in ready], [task.task_id])
        self.assertIn(ready[0]['qdms_type'], ('PROFILE_DATA_REQUEST', 'APPLICABILITY_REVIEW_TASK'))
        children = [c for c in approved.applicability_reviews if c.task_id == task.task_id]
        self.assertEqual({c.human_approval.status for c in children}, {'APPROVED'})
        self.assertTrue(all(c.human_approval.note.startswith(f'via {task.task_id}') for c in children))
        others = [c for c in approved.applicability_reviews if c.task_id != task.task_id]
        self.assertEqual({c.human_approval.status for c in others} - {'PENDING'}, set())

    def test_the_analysis_statistics_count_them(self):
        from regchain.tr.adjudicate import assess_profile as assess
        _, _, stats = assess(BREWER, OBLIGATIONS, REGISTER, REGISTRY, STORE)
        self.assertEqual(stats['applicability_unknown'], len(REPORT.applicability_reviews))
        self.assertEqual(sum(stats['applicability_unknown_by_type'].values()), stats['applicability_unknown'])


class GroupingTests(unittest.TestCase):
    """The UNKNOWN records under parent review tasks (step 4 of the 3 October plan): one task per source question of one
    article, or per company fact of one target; every record a child with its own decision and evidence."""

    def setUp(self):
        self.result = export()
        self.tasks = {t.task_id: t for t in self.result.review_tasks}

    def test_every_record_is_the_child_of_exactly_one_task(self):
        children = [c for t in self.result.review_tasks for c in t.child_ids]
        self.assertEqual(sorted(children), sorted(r.row_id for r in self.result.applicability_reviews))
        self.assertEqual(len(children), len(set(children)))
        for record in self.result.applicability_reviews:
            self.assertIn(record.row_id, self.tasks[record.task_id].child_ids)
        self.assertLessEqual(len(self.tasks), sum(t.topics for t in self.tasks.values()))
        self.assertLessEqual(sum(t.topics for t in self.tasks.values()), len(self.result.applicability_reviews))
        self.assertEqual(self.result.summary['review_tasks'], len(self.tasks))

    def test_a_task_never_mixes_causes_departments_or_subjects(self):
        by_id = {r.row_id: r for r in self.result.applicability_reviews}
        for task in self.tasks.values():
            records = [by_id[c] for c in task.child_ids]
            self.assertEqual({r.uncertainty['review_type'] for r in records}, {task.review_type})
            self.assertEqual({qdms.review_cause(r) for r in records}, {task.cause})
            if task.review_type == 'CLARIFY_REGULATORY_SCOPE':
                self.assertEqual(len({r.provision['provision_ref'].split('/')[0] for r in records}), 1)   # one article
            else:
                self.assertEqual(len({r.entity['target_id'] for r in records}), 1)                       # one target
            self.assertEqual(len(task.required_actions), 1)
            self.assertEqual(task.due_date, None)

    def test_the_children_keep_their_decision_and_evidence(self):
        ungrouped = {f'APPLICABILITY:{r.review_id}': r for r in REPORT.applicability_reviews}
        for record in self.result.applicability_reviews:
            source = ungrouped[record.row_id]
            self.assertEqual(record.applicability['reason_codes'], source.reason_codes)
            self.assertEqual(record.uncertainty['missing_facts'], source.missing_facts)
            self.assertEqual(record.uncertainty['company_evidence'], source.company_evidence)
            self.assertEqual(record.uncertainty['source_evidence'], source.source_evidence)
            self.assertEqual(record.provision['quote'], source.quote)

    def test_a_changed_child_makes_its_task_approval_stale(self):
        task = next(iter(self.tasks.values()))
        record = next(r for r in self.result.applicability_reviews if r.task_id == task.task_id)
        changed = record.model_copy(update={'fingerprint': '0' * 64})
        regrouped, tasks = qdms.group_reviews([changed if r.row_id == record.row_id else r for r in self.result.applicability_reviews])
        self.assertNotEqual(next(t for t in tasks if t.task_id == task.task_id).fingerprint, task.fingerprint)


class LedgerTests(unittest.TestCase):
    """Every analysis row is accounted for: exported, merged with the same duty and target from another pack, left out as
    covered with no action, or an applicability review. The counts reconcile per pack."""

    def test_two_packs_reading_the_same_duty_on_the_same_target_give_one_row(self):
        reports = {'PACK_A': REPORT, 'PACK_B': REPORT}
        result = qdms.export_rows(BREWER, REGISTER, REGISTRY, reports, {'PACK_A': ASSESSMENTS, 'PACK_B': ASSESSMENTS}, generated_at=WHEN, store=STORE)
        single = export()
        self.assertEqual([r.row_id for r in result.rows], [r.row_id for r in single.rows])
        self.assertTrue(all(r.packs == ['PACK_A', 'PACK_B'] for r in result.rows))
        self.assertEqual(len({r.row_id for r in result.rows}), len(result.rows))
        self.assertEqual(len({r.row_id for r in result.applicability_reviews}), len(result.applicability_reviews))
        for pack in reports:
            entries = [e for e in result.ledger if e['pack'] == pack]
            policy = [e for e in entries if not e['row_id'].startswith('APPLICABILITY:')]
            self.assertEqual(len(policy), len(REPORT.rows), pack)                   # every analysis row, once
            self.assertEqual(len(entries) - len(policy), len(REPORT.applicability_reviews), pack)
        outcomes = result.summary['ledger']
        covered_auto = sum(r.mapping.status == 'COVERED' and a.decision == 'AUTO' and not r.actions for r, a in zip(REPORT.rows, ASSESSMENTS))
        self.assertEqual(outcomes['EXCLUDED_COVERED_AUTO'], 2 * covered_auto)
        self.assertEqual(outcomes['EXPORTED'], len(result.rows))
        self.assertEqual(outcomes['APPLICABILITY_REVIEW'], len(result.applicability_reviews))
        self.assertEqual(outcomes['MERGED_SAME_DUTY_AND_TARGET'], len(result.rows) + len(result.applicability_reviews))
        self.assertEqual(sum(outcomes.values()), 2 * (len(REPORT.rows) + len(REPORT.applicability_reviews)))


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

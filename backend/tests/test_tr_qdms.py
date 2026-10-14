"""QDMS export (tr/qdms.py): every gap row becomes a change request that waits for a person; only an approval bound to
the row as it was reviewed makes its actions ready for the QDMS. Registers and the board decision here are SYNTHETIC."""
import contextlib
import copy
import csv
import io
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


def task_entry(task, decision):
    return {'row_id': task.task_id, 'fingerprint': task.fingerprint, 'decision': decision, 'approver': 'hukuk.sorumlusu',
            'decided_at': '2026-10-04T10:00:00+00:00', 'note': 'test'}


def approvals(*entries):
    return qdms.Approvals.model_validate({'format': qdms.APPROVALS_FORMAT, 'profile_id': BREWER.profile_id, 'entries': list(entries)})


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

    def test_an_approval_does_not_survive_a_change_of_who_must_act(self):
        """A product duty is the business of every entity that handles the product: when the profile gives the product to
        one more entity, the approval given before names a set of companies the reviewer never saw."""
        row = next(r for r in self.rows if r.entity['target_level'] == 'PRODUCT')
        product = row.entity['target_id']
        outsider = next(e for e in BREWER.legal_entities if product not in e.product_ids)
        widened = BREWER.model_copy(update={'legal_entities': [
            e.model_copy(update={'product_ids': [*e.product_ids, product]}) if e.entity_id == outsider.entity_id else e
            for e in BREWER.legal_entities]})
        newer = qdms.export_rows(widened, REGISTER, REGISTRY, {'BEVERAGE_ALCOHOL_TR': REPORT}, {'BEVERAGE_ALCOHOL_TR': ASSESSMENTS},
                                 generated_at=WHEN, store=STORE)
        moved = next(r for r in newer.rows if r.row_id == row.row_id)
        self.assertEqual(set(moved.entity['entity_ids']), {*row.entity['entity_ids'], outsider.entity_id})
        result, problems = qdms.apply_approvals(newer, qdms.Approvals.model_validate(
            {'format': qdms.APPROVALS_FORMAT, 'profile_id': BREWER.profile_id, 'entries': [entry(row, 'APPROVED')]}))
        self.assertEqual([p['problem'] for p in problems], ['STALE_APPROVAL'])
        self.assertEqual(qdms.ready_actions(result), [])
        # a review record of the product, and through it its task, changes with the entities too
        review = REPORT.applicability_reviews[0].model_copy(update={'level': 'PRODUCT', 'target_id': product, 'entity_id': None})
        before, after = (qdms.applicability_item(review, profile, REGISTRY, 'P') for profile in (BREWER, widened))
        self.assertNotEqual(before.entity['entity_ids'], after.entity['entity_ids'])
        self.assertNotEqual(before.fingerprint, after.fingerprint)
        self.assertNotEqual(qdms.group_reviews([before])[1][0].fingerprint, qdms.group_reviews([after])[1][0].fingerprint)

    def test_an_approval_does_not_survive_a_change_of_the_evidence_it_rests_on(self):
        row = next(r for r in self.rows if r.evidence['evidence'])
        evidence_id = row.evidence['evidence'][0]['evidence_id']
        register = REGISTER.model_copy(update={'evidence': [e.model_copy(update={'in_period': False}) if e.evidence_id == evidence_id else e
                                                            for e in REGISTER.evidence]})
        newer = qdms.export_rows(BREWER, register, REGISTRY, {'BEVERAGE_ALCOHOL_TR': REPORT}, {'BEVERAGE_ALCOHOL_TR': ASSESSMENTS},
                                 generated_at=WHEN, store=STORE)
        moved = next(r for r in newer.rows if r.row_id == row.row_id)
        self.assertFalse(next(e for e in moved.evidence['evidence'] if e['evidence_id'] == evidence_id)['in_period'])
        _, problems = qdms.apply_approvals(newer, qdms.Approvals.model_validate(
            {'format': qdms.APPROVALS_FORMAT, 'profile_id': BREWER.profile_id, 'entries': [entry(row, 'APPROVED')]}))
        self.assertEqual([p['problem'] for p in problems], ['STALE_APPROVAL'])
        # the register's period alone is not evidence: a row that rests on none keeps its approval
        bare = next(r for r in self.rows if not r.evidence['evidence'])
        relabelled = qdms.export_rows(BREWER, REGISTER.model_copy(update={'period': '2099Q1'}), REGISTRY, {'BEVERAGE_ALCOHOL_TR': REPORT},
                                      {'BEVERAGE_ALCOHOL_TR': ASSESSMENTS}, generated_at=WHEN, store=STORE)
        self.assertEqual(next(r for r in relabelled.rows if r.row_id == bare.row_id).fingerprint, bare.fingerprint)

    def test_unknown_rows_double_decisions_and_another_profile_are_refused(self):
        row = self.rows[0]
        _, problems = self.apply([{**entry(row, 'APPROVED'), 'row_id': 'yok:yok'}, entry(row, 'APPROVED'),
                                  {**entry(row, 'APPROVED'), 'approver': 'baska.biri'}])
        self.assertEqual(sorted(p['problem'] for p in problems), ['DECIDED_TWICE', 'UNKNOWN_ROW'])
        with self.assertRaises(ValueError):
            qdms.apply_approvals(self.export, qdms.Approvals(format=qdms.APPROVALS_FORMAT, profile_id='baska', entries=[]))
        with self.assertRaises(ValueError):                      # an approval names who decided
            qdms.ApprovalEntry.model_validate({**entry(row, 'APPROVED'), 'approver': ''})


class ApprovalGateTests(unittest.TestCase):
    """`qdms approve` takes nothing on trust from export.json, a plain file anyone can edit between export and approve:
    every row, record and task fingerprint is recomputed from the content beside it, and every decision comes from an
    approvals file given to the run, never from an approval or action state already written in the export."""

    def setUp(self):
        self.export = export()
        self.data = self.export.model_dump(mode='json')

    def test_states_written_into_export_json_make_nothing_ready(self):
        rows, tasks = self.data['rows'], self.data['review_tasks']
        approved = {'required': True, 'status': 'APPROVED', 'approver': 'kimse', 'decided_at': '2026-10-12T10:00:00+00:00', 'note': ''}
        rows[0]['human_approval'] = dict(approved)
        for action in rows[0]['required_actions']:
            action['state'] = 'READY_FOR_QDMS'
        rows[1]['required_actions'][0]['state'] = 'READY_FOR_QDMS'          # READY under a PENDING approval
        tasks[0]['human_approval'] = dict(approved)
        tasks[0]['required_actions'][0]['state'] = 'READY_FOR_QDMS'
        result, problems = qdms.apply_approvals(qdms.QdmsExport.model_validate(self.data), approvals())
        self.assertEqual(qdms.ready_actions(result), [])
        self.assertEqual(result.summary['ready_for_qdms'], 0)
        self.assertEqual(result.summary['by_approval'], {'PENDING': len(result.rows)})
        self.assertEqual({a.state for r in [*result.rows, *result.review_tasks] for a in r.required_actions}, {'DRAFT'})
        self.assertEqual({c.human_approval.status for c in result.applicability_reviews}, {'PENDING'})
        self.assertEqual(sorted((p['row_id'], p['problem']) for p in problems),
                         sorted([(rows[0]['row_id'], 'DECISION_NOT_IN_APPROVALS'), (rows[1]['row_id'], 'DECISION_NOT_IN_APPROVALS'),
                                 (tasks[0]['task_id'], 'DECISION_NOT_IN_APPROVALS')]))

    def test_a_row_edited_after_export_keeps_no_approval(self):
        """The reviewer approved the row as exported; export.json was then edited with the stored fingerprint kept."""
        row = self.export.rows[0]

        def inject(r):
            r['required_actions'].append({**r['required_actions'][-1], 'action_id': r['required_actions'][-1]['action_id'] + '-x',
                                          'qdms_type': 'NONCONFORMITY', 'change': 'OPEN_NONCONFORMITY', 'department': 'SALES'})
        edits = {'an action added': inject,
                 'an action retargeted': lambda r: r['required_actions'][0].update({'department': 'SALES'}),
                 'the duty text': lambda r: r['provision'].update({'quote': 'baska bir metin'}),
                 'the reasoning': lambda r: r['reason'].update({'statements': [*r['reason']['statements'], {
                     'passage_id': 'P-X', 'document_id': 'D-X', 'relation': 'SUPPORTS', 'reasons': [], 'quote': 'eklenen ifade'}]}),
                 'who must act': lambda r: r['entity'].update({'entity_ids': [*r['entity']['entity_ids'], 'ALC-INT-MALT']}),
                 'the evidence': lambda r: r['evidence'].update({'evidence': [{'evidence_id': 'EV-X', 'control_id': 'CTL-X', 'evidence_type': 'X',
                                                                              'period': '2026Q3', 'in_period': True}]})}
        for name, edit in edits.items():
            with self.subTest(name):
                data = copy.deepcopy(self.data)
                edit(data['rows'][0])
                result, problems = qdms.apply_approvals(qdms.QdmsExport.model_validate(data), approvals(entry(row, 'APPROVED')))
                self.assertEqual([(p['row_id'], p['problem']) for p in problems], [(row.row_id, 'EXPORT_MODIFIED')])
                self.assertEqual(qdms.ready_actions(result), [])
                self.assertEqual(result.rows[0].human_approval.status, 'PENDING')
        with self.subTest('a row copied'):                     # its approved actions would reach the QDMS twice
            data = copy.deepcopy(self.data)
            data['rows'].append(copy.deepcopy(data['rows'][0]))
            result, problems = qdms.apply_approvals(qdms.QdmsExport.model_validate(data), approvals(entry(row, 'APPROVED')))
            self.assertEqual([(p['row_id'], p['problem']) for p in problems], [(row.row_id, 'EXPORT_MODIFIED')])
            self.assertEqual(qdms.ready_actions(result), [])

    def test_an_export_of_other_rules_is_refused_whole(self):
        """Fingerprints are recomputed with today's rules: an export written under other rules is exported again, not
        reported row by row as modified."""
        data = {**self.data, 'rules_version': 'tr-qdms-v4'}
        with self.assertRaises(ValueError) as refused:
            qdms.apply_approvals(qdms.QdmsExport.model_validate(data), approvals(entry(self.export.rows[0], 'APPROVED')))
        self.assertIn('tr-qdms-v4', str(refused.exception))
        self.assertIn(qdms.QDMS_RULES_VERSION, str(refused.exception))

    def test_a_review_record_or_task_edited_after_export_keeps_no_approval(self):
        task, other = self.export.review_tasks[0], self.export.review_tasks[1]
        index = next(i for i, c in enumerate(self.data['applicability_reviews']) if c['task_id'] == task.task_id)
        self.assertTrue(self.data['applicability_reviews'][index]['uncertainty']['company_evidence'])
        edits = {'a record\'s company evidence': lambda d: d['applicability_reviews'][index]['uncertainty'].update({'company_evidence': []}),
                 'a record\'s missing facts': lambda d: d['applicability_reviews'][index]['uncertainty'].update({
                     'missing_facts': [*d['applicability_reviews'][index]['uncertainty']['missing_facts'], {'gate': 'ACTIVITY_CLASS', 'status': 'UNKNOWN'}]}),
                 'a record\'s entity': lambda d: d['applicability_reviews'][index]['entity'].update({'entity_ids': ['ALC-INT-MALT']}),
                 'a record moved to another task': lambda d: d['applicability_reviews'][index].update({'task_id': other.task_id}),
                 'the task\'s action': lambda d: d['review_tasks'][0]['required_actions'][0].update({'qdms_type': 'NONCONFORMITY'}),
                 'the task\'s question': lambda d: d['review_tasks'][0].update({'question_tr': 'baska bir soru'}),
                 'the task\'s entities': lambda d: d['review_tasks'][0].update({'entities': ['ALC-INT-MALT']})}
        for name, edit in edits.items():
            with self.subTest(name):
                data = copy.deepcopy(self.data)
                edit(data)
                result, problems = qdms.apply_approvals(qdms.QdmsExport.model_validate(data), approvals(task_entry(task, 'APPROVED')))
                self.assertIn((task.task_id, 'EXPORT_MODIFIED'), [(p['row_id'], p['problem']) for p in problems])
                self.assertEqual({p['problem'] for p in problems}, {'EXPORT_MODIFIED'})
                self.assertEqual(qdms.ready_actions(result), [])
                self.assertEqual({c.human_approval.status for c in result.applicability_reviews}, {'PENDING'})

    def test_an_approve_output_is_approved_again_only_from_its_files(self):
        first, second = self.export.rows[0], self.export.rows[1]
        earlier, later = approvals(entry(first, 'APPROVED')), approvals(entry(second, 'REJECTED'))
        once, problems = qdms.apply_approvals(self.export, earlier)
        self.assertEqual(problems, [])
        with tempfile.TemporaryDirectory() as directory:
            reread = qdms.load_export(qdms.write(once, Path(directory))['json'])
        # the earlier decision is written in export.json, but no file of this run makes it: it is not taken over
        alone, problems = qdms.apply_approvals(reread, later)
        self.assertEqual([(p['row_id'], p['problem']) for p in problems], [(first.row_id, 'DECISION_NOT_IN_APPROVALS')])
        self.assertEqual(qdms.ready_actions(alone), [])
        self.assertEqual(next(r for r in alone.rows if r.row_id == first.row_id).human_approval.status, 'PENDING')
        # given every file, both decisions apply
        both, problems = qdms.apply_approvals(reread, [earlier, later])
        self.assertEqual(problems, [])
        self.assertEqual({a['row_id'] for a in qdms.ready_actions(both)}, {first.row_id})
        self.assertEqual(next(r for r in both.rows if r.row_id == second.row_id).human_approval.status, 'REJECTED')

    def test_one_file_that_decides_a_row_two_ways_is_refused_whole(self):
        row, task, other = self.export.rows[0], self.export.review_tasks[0], self.export.rows[1]
        for first, second in (('APPROVED', 'REJECTED'), ('REJECTED', 'APPROVED'), ('RETURNED', 'APPROVED')):
            for one, two in ((entry(row, first), entry(row, second)), (task_entry(task, first), task_entry(task, second))):
                with self.subTest(first=first, second=second, id=one['row_id']):
                    with self.assertRaises(ValueError) as refused:
                        qdms.apply_approvals(self.export, approvals(one, entry(other, 'APPROVED'), two))
                    self.assertIn(one['row_id'], str(refused.exception))
                    self.assertNotIn(other.row_id, str(refused.exception))
        # an exact duplicate counts once; the same decision twice by different people decides nothing
        result, problems = qdms.apply_approvals(self.export, approvals(entry(row, 'APPROVED'), entry(row, 'APPROVED')))
        self.assertEqual(problems, [])
        self.assertEqual({a['row_id'] for a in qdms.ready_actions(result)}, {row.row_id})
        result, problems = qdms.apply_approvals(self.export, approvals(entry(row, 'APPROVED'), {**entry(row, 'APPROVED'), 'approver': 'baska'}))
        self.assertEqual([(p['row_id'], p['problem']) for p in problems], [(row.row_id, 'DECIDED_TWICE')])
        self.assertEqual(qdms.ready_actions(result), [])
        self.assertEqual(result.rows[0].human_approval.status, 'PENDING')

    def test_a_later_file_overrides_a_decision_only_with_a_newer_date_and_keeps_it(self):
        row = self.export.rows[0]
        rejected = approvals({**entry(row, 'REJECTED'), 'approver': 'hukuk', 'decided_at': '2026-10-12T10:00:00+03:00'})
        older = approvals({**entry(row, 'APPROVED'), 'approver': 'satis', 'decided_at': '2026-10-01T09:00:00+03:00'})
        newer = approvals({**entry(row, 'APPROVED'), 'approver': 'hukuk', 'decided_at': '2026-10-13T09:00:00+03:00'})
        result, problems = qdms.apply_approvals(self.export, [rejected, older])
        self.assertEqual(problems, [])
        approval = result.rows[0].human_approval
        self.assertEqual((approval.status, approval.approver), ('REJECTED', 'hukuk'))
        self.assertEqual([(h.status, h.approver) for h in approval.history], [('APPROVED', 'satis')])
        self.assertEqual(qdms.ready_actions(result), [])
        result, problems = qdms.apply_approvals(self.export, [rejected, newer])
        self.assertEqual(problems, [])
        approval = result.rows[0].human_approval
        self.assertEqual((approval.status, [(h.status, h.approver) for h in approval.history]), ('APPROVED', [('REJECTED', 'hukuk')]))
        self.assertEqual({a['row_id'] for a in qdms.ready_actions(result)}, {row.row_id})
        # the rejection written in an approve output is not overwritten by an older approval given alone
        once, _ = qdms.apply_approvals(self.export, rejected)
        chained, problems = qdms.apply_approvals(once, older)
        self.assertEqual([(p['row_id'], p['problem']) for p in problems], [(row.row_id, 'DECISION_NOT_IN_APPROVALS')])
        self.assertEqual(chained.rows[0].human_approval.status, 'PENDING')
        self.assertEqual(qdms.ready_actions(chained), [])
        # two files that decide the row two ways at the same moment: neither is later, nothing is decided
        tied = approvals({**entry(row, 'APPROVED'), 'approver': 'satis', 'decided_at': '2026-10-12T10:00:00+03:00'})
        result, problems = qdms.apply_approvals(self.export, [rejected, tied])
        self.assertEqual([(p['row_id'], p['problem']) for p in problems], [(row.row_id, 'DECIDED_TWICE')])
        self.assertEqual(result.rows[0].human_approval.status, 'PENDING')
        self.assertEqual(qdms.ready_actions(result), [])


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

    def test_an_approvals_file_with_a_bom_is_read_and_a_bad_one_is_named(self):
        result = export()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = json.loads(qdms.write(result, root / 'export')['template'].read_text(encoding='utf-8'))
            template['entries'][0].update({'decision': 'APPROVED', 'approver': 'kalite.muduru', 'decided_at': '2026-10-02T14:00:00+00:00'})
            (root / 'bom.json').write_text(json.dumps(template, ensure_ascii=False), encoding='utf-8-sig')   # Windows editors add one
            self.assertEqual([e.row_id for e in qdms.load_approvals(root / 'bom.json').entries], [result.rows[0].row_id])
            # a bad entry is named by its place in the file and its row, not by its place among the filled entries
            position = len(template['entries']) - 1
            bad = copy.deepcopy(template)
            bad['entries'][position].update({'decision': 'ONAYLANDI', 'approver': 'kalite.muduru', 'decided_at': '2026-10-02T14:00:00+00:00'})
            (root / 'bad.json').write_text(json.dumps(bad, ensure_ascii=False), encoding='utf-8')
            undated = copy.deepcopy(template)
            undated['entries'][position].update({'decision': 'APPROVED', 'approver': 'kalite.muduru', 'decided_at': '12.10.2026'})
            (root / 'undated.json').write_text(json.dumps(undated, ensure_ascii=False), encoding='utf-8')
            # decisions are ordered by their time, so a time without its UTC offset cannot be placed
            naive = copy.deepcopy(template)
            naive['entries'][position].update({'decision': 'APPROVED', 'approver': 'kalite.muduru', 'decided_at': '2026-10-02T14:00:00'})
            (root / 'naive.json').write_text(json.dumps(naive, ensure_ascii=False), encoding='utf-8')
            for name in ('bad.json', 'undated.json', 'naive.json'):
                with self.subTest(name), self.assertRaises(ValueError) as named:
                    qdms.load_approvals(root / name)
                self.assertIn(f'entries[{position}]', str(named.exception))
                self.assertIn(template['entries'][position]['row_id'], str(named.exception))
            template['entries'][0]['approver'] = 'kalite.müdürü'
            (root / 'cp1254.json').write_bytes(json.dumps(template, ensure_ascii=False).encode('cp1254'))
            (root / 'broken.json').write_text('{"format": ', encoding='utf-8')
            for name, expected in (('cp1254.json', 'UTF-8'), ('broken.json', 'JSON')):
                with self.subTest(name), self.assertRaises(ValueError) as unreadable:
                    qdms.load_approvals(root / name)
                self.assertIn(expected, str(unreadable.exception))
                self.assertIn(name, str(unreadable.exception))


class ApproveCommandTests(unittest.TestCase):
    """`qdms approve` on files: a refused approvals file is a message and exit 2 with nothing written, not a traceback."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.result = export()
        cls.paths = qdms.write(cls.result, cls.root / 'export')
        cls.template = json.loads(cls.paths['template'].read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def filled(self, *decisions):
        template = copy.deepcopy(self.template)
        for index, decision in decisions:
            template['entries'][index].update({'decision': decision, 'approver': 'kalite.muduru', 'decided_at': '2026-10-02T14:00:00+00:00'})
        return template

    def approve(self, name, *files, export_json=None):
        from regchain.tr.__main__ import main
        arguments = ['qdms', 'approve', '--export', str(export_json or self.paths['json']), '--out', str(self.root / f'{name}-out')]
        for index, content in enumerate(files):
            path = self.root / f'{name}-{index}.json'
            path.write_bytes(content)
            arguments += ['--approvals', str(path)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue(), self.root / f'{name}-out'

    def test_a_refused_approvals_file_exits_2_and_writes_nothing(self):
        conflicting = self.filled((0, 'APPROVED'), (1, 'APPROVED'))
        conflicting['entries'].append({**conflicting['entries'][0], 'decision': 'REJECTED'})
        bad = self.filled((0, 'APPROVED'), (len(self.template['entries']) - 1, 'ONAYLANDI'))
        legacy = self.filled((0, 'APPROVED'))
        legacy['entries'][0]['approver'] = 'kalite.müdürü'
        extra = {**self.filled((0, 'APPROVED')), 'yorum': 'ek alan'}
        cases = {'conflicting': (json.dumps(conflicting).encode('utf-8'), self.template['entries'][0]['row_id']),
                 'extra-key': (json.dumps(extra).encode('utf-8'), 'yorum'),
                 'blank': (json.dumps(self.template).encode('utf-8'), 'no decision'),
                 'broken': (b'{"format": ', 'JSON'),
                 'cp1254': (json.dumps(legacy, ensure_ascii=False).encode('cp1254'), 'UTF-8'),
                 'bad-entry': (json.dumps(bad).encode('utf-8'), f"entries[{len(self.template['entries']) - 1}]")}
        for name, (content, said) in cases.items():
            with self.subTest(name):
                code, printed, message, out = self.approve(name, content)
                self.assertEqual(code, 2)
                self.assertIn(said, message)
                self.assertNotIn('Traceback', message)
                self.assertFalse(out.exists())

    def test_approvals_files_with_and_without_a_bom_are_applied_together(self):
        first, second = self.filled((0, 'APPROVED')), self.filled((1, 'REJECTED'))
        code, printed, message, out = self.approve('two', json.dumps(first).encode('utf-8-sig'), json.dumps(second).encode('utf-8'))
        self.assertEqual((code, message), (0, ''))
        self.assertEqual(json.loads(printed)['decisions_read'], 2)
        ready = json.loads((out / 'ready-actions.json').read_text(encoding='utf-8'))
        self.assertEqual({a['row_id'] for a in ready}, {self.template['entries'][0]['row_id']})
        self.assertEqual(json.loads((out / 'approval-problems.json').read_text(encoding='utf-8')), [])

    def test_a_hand_edited_export_json_releases_only_what_the_approvals_file_decides(self):
        """The review's repro: export.json edited to show a row approved and ready (approver 'kimse'); the approvals file
        decides another row. Only that row reaches ready-actions.json, and the edit is reported."""
        data = json.loads(self.paths['json'].read_text(encoding='utf-8'))
        forged = data['rows'][2]
        forged['human_approval'] = {'required': True, 'status': 'APPROVED', 'approver': 'kimse', 'decided_at': '2026-10-12T10:00:00+00:00', 'note': ''}
        for action in forged['required_actions']:
            action['state'] = 'READY_FOR_QDMS'
        edited = self.root / 'edited-export.json'
        edited.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        code, printed, message, out = self.approve('edited', json.dumps(self.filled((0, 'APPROVED'))).encode('utf-8'), export_json=edited)
        self.assertEqual((code, message), (1, ''))
        ready = json.loads((out / 'ready-actions.json').read_text(encoding='utf-8'))
        self.assertEqual({a['row_id'] for a in ready}, {self.template['entries'][0]['row_id']})
        self.assertNotIn('kimse', {a['approver'] for a in ready})
        problems = json.loads((out / 'approval-problems.json').read_text(encoding='utf-8'))
        self.assertEqual([(p['row_id'], p['problem']) for p in problems], [(forged['row_id'], 'DECISION_NOT_IN_APPROVALS')])
        self.assertEqual(json.loads(printed)['summary']['by_approval'].get('APPROVED'), 1)


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

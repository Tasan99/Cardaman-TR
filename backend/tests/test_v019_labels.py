"""v0.19 human review loop: pending label changes with a hash-chained audit log, and the expert review package.

A label change is proposed (PENDING), approved by a second person (moved to the reviewed file as
ACCEPTED, HUMAN_REVIEWED) or rejected, and every step is chained in <dataset>.audit.jsonl; a strict
overlay ignores an ACCEPTED entry the log did not approve. The dataset file is never written: every
test checks its sha256 before and after. The ledger lives in a temporary directory (--dir), so the
committed evaluation/reviewed_labels files are only read here.
"""
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
import zipfile
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree

from regchain.evaluation.cli import main
from regchain.evaluation.expert_export import (ALL_COLUMNS, COLUMNS, parse_decision, read_sheet, read_xlsx, sheet_changes,
                                               write_expert_review, write_xlsx)
from regchain.evaluation.harness import load_dataset, run
from regchain.evaluation.labels import apply_reviewed
from regchain.evaluation.pending import GENESIS, PENDING_FORMAT, Ledger, chain_problems, import_review, read_audit
from regchain.pilot.engine import AUTONOMOUS_REVIEWER

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / 'evaluation' / 'datasets' / 'tr-aml-v1.json'
COMMITTED = REPO / 'evaluation' / 'reviewed_labels'
SHEET_NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def quiet(argv):
    """(exit code, printed output) of the evaluation CLI."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue() + err.getvalue()


C08_COVERAGE = {'case_id': 'C08', 'article': '6', 'clause': '1', 'field': 'coverage', 'to': 'PARTIAL',
                'reason': 'md. 6(1) also asks for the phone number; aml-exact.md section 2 does not list it.'}


class LedgerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.dir = self.root / 'reviewed_labels'
        self.before = sha256(DATASET.read_bytes()).hexdigest()
        self.ledger = Ledger(DATASET, self.dir)

    def tearDown(self):
        self.assertEqual(sha256(DATASET.read_bytes()).hexdigest(), self.before, 'the dataset file must never be written')

    def propose(self, **change):
        return self.ledger.propose([{**C08_COVERAGE, **change}], 'Analyst A', 'label auditor', 'label-audit')[0]

    def test_propose_then_approve_moves_the_entry_to_the_reviewed_file_with_approver_metadata(self):
        entry = self.propose()
        self.assertEqual((entry['status'], entry['from'], entry['to'], entry['label_source_before']), ('PENDING', 'COVERS_TEXT', 'PARTIAL', 'RULE_DERIVED'))
        self.assertEqual((entry['clause'], entry['action_keywords']), ('(1)', ['kimlik tespitinde']))     # pinned to the one target
        pending = json.loads(self.ledger.pending_path.read_text(encoding='utf-8'))
        self.assertEqual((pending['format'], [e['entry_id'] for e in pending['entries']]), (PENDING_FORMAT, [entry['entry_id']]))
        self.assertFalse(self.ledger.reviewed_path.exists())                      # a proposal changes no label
        moved = self.ledger.approve([entry['entry_id']], 'Approver B', 'AML lawyer', note='checked against the clause')[0]
        reviewed = json.loads(self.ledger.reviewed_path.read_text(encoding='utf-8'))
        stored = reviewed['entries'][0]
        self.assertEqual(stored, moved)
        self.assertEqual((stored['status'], stored['approved_by'], stored['reviewer'], stored['label_source_after'], stored['entry_id']),
                         ('ACCEPTED', 'Approver B', 'Approver B', 'HUMAN_REVIEWED', entry['entry_id']))
        self.assertTrue(stored['approved_at'] and not stored['self_approved'] and stored['approval_note'])
        events = read_audit(self.ledger.audit_path)
        self.assertEqual([e['action'] for e in events], ['PROPOSED', 'APPROVED'])
        self.assertEqual(stored['pending_sha256'], events[0]['entry_sha256'])
        self.assertEqual((events[0]['prev_hash'], events[1]['prev_hash']), (GENESIS, events[0]['hash']))
        self.assertEqual(json.loads(self.ledger.pending_path.read_text(encoding='utf-8'))['entries'], [])   # moved, not copied
        report = self.ledger.verify()
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual((report['approved'], report['pending'], report['events']), (1, 0, 2))
        overlaid, overlay = apply_reviewed(load_dataset(DATASET), self.ledger.reviewed_path, DATASET)
        self.assertTrue(overlay['strict'])                                       # the log exists, so strict by default
        self.assertEqual([(a['label_source_before'], a['label_source_after']) for a in overlay['applied']], [('RULE_DERIVED', 'HUMAN_REVIEWED')])
        c08 = [e for e in {c.case_id: c for c in overlaid.cases}['C08'].expected_obligations if e.article == '6'][0]
        self.assertEqual((c08.coverage, c08.label_sources), ('PARTIAL', {'coverage': 'HUMAN_REVIEWED'}))

    def test_a_proposal_must_name_one_expectation_its_current_label_and_a_new_valid_one(self):
        refused = {'case': dict(case_id='C99'), 'stale from': dict(**{'from': 'PARTIAL'}), 'same label': dict(to='COVERS_TEXT'),
                   'invalid value': dict(to='MOSTLY'), 'field': dict(field='notes'), 'no reason': dict(reason='  '),
                   'no match': dict(article='99')}
        for name, change in refused.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.propose(**change)
        with self.assertRaises(ValueError):
            self.ledger.propose([C08_COVERAGE], '', 'x')                          # an anonymous proposal
        self.assertFalse(self.ledger.pending_path.exists() or self.ledger.audit_path.exists())   # nothing refused was written
        self.propose()
        with self.assertRaisesRegex(ValueError, 'already pending'):
            self.propose()
        conflict = self.ledger.propose([{**C08_COVERAGE, 'field': 'conflict', 'to': 'true', 'from': 'false'}], 'Analyst A')[0]
        self.assertEqual((conflict['from'], conflict['to']), (False, True))

    def test_four_eyes_refuses_self_approval_unless_it_is_allowed_and_recorded(self):
        entry = self.propose()
        with self.assertRaisesRegex(ValueError, 'four-eyes'):
            self.ledger.approve([entry['entry_id']], ' analyst a ')
        self.assertFalse(self.ledger.reviewed_path.exists())
        moved = self.ledger.approve([entry['entry_id']], 'Analyst A', allow_self_approval=True)[0]
        self.assertTrue(moved['self_approved'])
        report = self.ledger.verify()
        self.assertTrue(report['ok'], report['problems'])
        self.assertTrue(any('own proposer' in w for w in report['warnings']))

    def test_reject_and_withdraw_keep_the_entry_with_the_decision_and_never_apply_it(self):
        rejected, withdrawn = self.propose(), self.propose(field='conflict', to='true')
        self.ledger.reject([rejected['entry_id']], 'Approver B', 'The missing items are optional ("varsa").')
        self.ledger.withdraw([withdrawn['entry_id']], 'Analyst A', 'Proposed by mistake.')
        entries = {e['entry_id']: e for e in self.ledger.pending()['entries']}
        self.assertEqual((entries[rejected['entry_id']]['status'], entries[rejected['entry_id']]['decided_by']), ('REJECTED', 'Approver B'))
        self.assertEqual(entries[withdrawn['entry_id']]['status'], 'WITHDRAWN')
        with self.assertRaisesRegex(ValueError, 'not PENDING'):
            self.ledger.approve([rejected['entry_id']], 'Approver C')
        self.assertEqual([e['action'] for e in read_audit(self.ledger.audit_path)], ['PROPOSED', 'PROPOSED', 'REJECTED', 'WITHDRAWN'])
        report = self.ledger.verify()
        self.assertTrue(report['ok'], report['problems'])
        self.assertEqual((report['rejected'], report['withdrawn'], report['approved']), (1, 1, 0))
        self.assertFalse(self.ledger.reviewed_path.exists())

    def test_verify_finds_an_edited_event_a_cut_tail_and_entries_edited_after_the_fact(self):
        entry = self.propose()
        self.ledger.approve([entry['entry_id']], 'Approver B')
        self.propose(field='conflict', to='true')
        self.assertTrue(self.ledger.verify()['ok'])
        lines = self.ledger.audit_path.read_text(encoding='utf-8').splitlines()
        original = '\n'.join(lines) + '\n'

        edited = json.loads(lines[0])
        edited['actor'] = 'Somebody else'
        self.ledger.audit_path.write_text('\n'.join([json.dumps(edited), *lines[1:]]) + '\n', encoding='utf-8')
        self.assertTrue(any('hash does not match' in p for p in self.ledger.verify()['problems']))
        self.assertTrue(chain_problems(read_audit(self.ledger.audit_path)))
        with self.assertRaisesRegex(ValueError, 'fails verification'):             # no new step on a broken log
            self.propose(field='applicability', to='UNKNOWN')

        self.ledger.audit_path.write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
        self.assertTrue(any('lines cut' in p for p in self.ledger.verify()['problems']))

        self.ledger.audit_path.write_text(original, encoding='utf-8')
        book = json.loads(self.ledger.reviewed_path.read_text(encoding='utf-8'))
        book['entries'][0]['to'] = 'NO_EVIDENCE'                                   # edited after approval
        self.ledger.reviewed_path.write_text(json.dumps(book), encoding='utf-8')
        self.assertTrue(any('edited after approval' in p for p in self.ledger.verify()['problems']))
        _, overlay = apply_reviewed(load_dataset(DATASET), self.ledger.reviewed_path, DATASET)
        self.assertEqual((overlay['applied'], [s['reason'] for s in overlay['skipped']]),
                         ([], ['the entry differs from the one approved (edited after approval)']))

    def test_strict_overlay_ignores_accepted_entries_without_an_approval_event(self):
        self.propose()                                                           # the flow is in use: the log exists
        typed = {'case_id': 'C09', 'article': '7', 'clause': '(1)', 'field': 'coverage', 'from': 'COVERS_TEXT', 'to': 'PARTIAL',
                 'reviewer': 'someone', 'reviewed_at': '2026-09-24', 'reason': 'typed by hand', 'status': 'ACCEPTED'}
        book = {**json.loads((COMMITTED / 'tr-aml-v1.reviewed.json').read_text(encoding='utf-8')), 'entries': [typed]}
        self.ledger.reviewed_path.write_text(json.dumps(book), encoding='utf-8')
        _, overlay = apply_reviewed(load_dataset(DATASET), self.ledger.reviewed_path, DATASET)
        self.assertEqual((overlay['strict'], overlay['applied'], [s['reason'] for s in overlay['skipped']]), (True, [], ['no approval record']))
        _, legacy = apply_reviewed(load_dataset(DATASET), self.ledger.reviewed_path, DATASET, strict=False)
        self.assertEqual(len(legacy['applied']), 1)                              # the v0.18 reading, only when asked for
        report = self.ledger.verify()
        self.assertTrue(report['ok'], report['problems'])                        # a hand-typed entry is a warning, never approved
        self.assertEqual(report['unaudited_reviewed'], 1)

    def test_the_committed_ledger_verifies(self):
        if not (COMMITTED / 'tr-aml-v1.audit.jsonl').is_file():
            self.skipTest('no committed audit log yet')
        report = Ledger(DATASET, COMMITTED).verify()
        self.assertTrue(report['ok'], report['problems'])

    # Review finding 4: a step on a log that no longer matches the head the pending file recorded.
    def test_a_cut_tail_stops_every_new_step_until_it_is_repaired(self):
        rejected = self.propose()
        self.ledger.reject([rejected['entry_id']], 'Bob', 'The missing items are optional.')
        lines = self.ledger.audit_path.read_text(encoding='utf-8').splitlines()
        self.ledger.audit_path.write_text(lines[0] + '\n', encoding='utf-8')     # the REJECTED line removed ...
        pending = json.loads(self.ledger.pending_path.read_text(encoding='utf-8'))
        for entry in pending['entries']:                                           # ... and the entry put back to PENDING
            entry['status'] = 'PENDING'
            for key in ('decided_by', 'decider_role', 'decided_at', 'decision_reason'):
                entry.pop(key)
        self.ledger.pending_path.write_text(json.dumps(pending, ensure_ascii=False), encoding='utf-8')
        self.assertTrue(any('lines cut' in p for p in self.ledger.verify()['problems']))
        steps = {'approve': lambda: self.ledger.approve([rejected['entry_id']], 'Carol'),
                 'propose': lambda: self.propose(field='conflict', to='true'),
                 'withdraw': lambda: self.ledger.withdraw([rejected['entry_id']], 'Analyst A', 'x')}
        for name, step in steps.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'records the log at 2 events'):
                step()
        self.assertFalse(self.ledger.reviewed_path.exists())
        self.assertEqual(len(read_audit(self.ledger.audit_path)), 1)             # nothing was appended

    # Review finding 7: "from" is the label as the approved entries left it, and one field has one open change.
    def test_a_change_starts_from_the_label_as_approved_entries_left_it(self):
        first = self.propose()
        self.ledger.approve([first['entry_id']], 'Carol')
        with self.assertRaisesRegex(ValueError, "the label is now 'PARTIAL', the proposal changes 'COVERS_TEXT'"):
            self.propose(to='NO_EVIDENCE', **{'from': 'COVERS_TEXT'})
        with self.assertRaisesRegex(ValueError, 'already is'):
            self.propose()                                                        # approved already: nothing to change
        follow = self.propose(to='NO_EVIDENCE')
        self.assertEqual((follow['from'], follow['label_source_before']), ('PARTIAL', 'HUMAN_REVIEWED'))
        with self.assertRaisesRegex(ValueError, 'already pending'):
            self.propose(to='UNKNOWN')                                            # a competing change of the same field
        self.ledger.approve([follow['entry_id']], 'Erin')
        overlaid, overlay = apply_reviewed(load_dataset(DATASET), self.ledger.reviewed_path, DATASET)
        c08 = [e for e in {c.case_id: c for c in overlaid.cases}['C08'].expected_obligations if e.article == '6'][0]
        self.assertEqual((c08.coverage, len(overlay['applied']), overlay['skipped']), ('NO_EVIDENCE', 2, []))
        self.assertTrue(self.ledger.verify()['ok'])

    def test_verify_flags_an_approved_change_the_overlay_no_longer_applies(self):
        folder = self.root / 'evaluation' / 'datasets'
        folder.mkdir(parents=True)
        copy = folder / 'tr-aml-v1.json'
        shutil.copyfile(DATASET, copy)
        ledger = Ledger(copy, self.dir)
        entry = ledger.propose([C08_COVERAGE], 'Analyst A')[0]
        ledger.approve([entry['entry_id']], 'Approver B')
        self.assertTrue(ledger.verify()['ok'])
        data = json.loads(copy.read_text(encoding='utf-8'))                       # a new dataset version moves the label under it
        target = next(e for c in data['cases'] if c['case_id'] == 'C08' for e in c['expected_obligations'] if e['article'] == '6' and e['clause'] == '(1)')
        target['coverage'] = 'NO_EVIDENCE'
        copy.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        report = Ledger(copy, self.dir).verify()
        self.assertFalse(report['ok'])
        self.assertTrue(any(entry['entry_id'] in p and 'not applied' in p for p in report['problems']), report['problems'])
        target['coverage'] = 'PARTIAL'                                            # the change is already in the dataset: a warning only
        copy.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        report = Ledger(copy, self.dir).verify()
        self.assertTrue(report['ok'], report['problems'])
        self.assertTrue(any('already' in w for w in report['warnings']))


class ReviewImportTests(unittest.TestCase):
    """import-review (pilot OVERRIDE decisions and filled expert sheets) and the expert export, on a rules-mode run."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.before = sha256(DATASET.read_bytes()).hexdigest()
        with contextlib.redirect_stdout(io.StringIO()):
            cls.run_dir, _, _ = run(DATASET, cls.root / 'runs', provider='rules', retrieval='lexical', cases=['C08', 'C09'], label='review')
        cls.results = {r['case_id']: r for r in json.loads((cls.run_dir / 'results.json').read_text(encoding='utf-8'))}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(dir=self.root))

    def tearDown(self):
        self.assertEqual(sha256(DATASET.read_bytes()).hexdigest(), self.before, 'the dataset file must never be written')

    def review(self, reviewer='Ayşe Hukukçu', action='OVERRIDE'):
        result = self.results['C08']
        matched = {i['prediction']: i for i in result['scored'] if i['matched']}
        by_article = {matched[p['key']]['expected']['article'] + matched[p['key']]['expected']['clause']: p
                      for p in result['predictions'] if p['key'] in matched}
        decision = lambda prediction, **fields: {'obligation_id': prediction['obligation_id'], 'extraction': 'ACCEPT', 'company_fact_keys': [],
                                                 'scope_evidence': [], 'policy_evidence': [], 'rationale': 'Read against the clause.', **fields}
        return {'format': 'regchain-pilot-review-v1', 'analysis_head': result['head'], 'reviewer': reviewer, 'reviewer_role': 'AML lawyer',
                'decisions': [decision(by_article['6(1)'], applicability='APPLIES', coverage='PARTIAL', action=action,
                                       override_reason='Phone number missing from the policy.' if action == 'OVERRIDE' else ''),
                              decision(by_article['7(1)'], applicability='DOES_NOT_APPLY', coverage='NO_EVIDENCE', action='APPROVE')]}

    def test_import_review_turns_override_decisions_into_pending_proposals(self):
        path = self.dir / 'review.json'
        path.write_text(json.dumps(self.review(), ensure_ascii=False), encoding='utf-8')
        code, output = quiet(['labels', 'import-review', '--dataset', str(DATASET), '--dir', str(self.dir), '--run', str(self.run_dir),
                              '--review', str(path)])
        self.assertEqual(code, 0, output)
        entries = Ledger(DATASET, self.dir).pending()['entries']
        self.assertEqual([(e['case_id'], e['article'], e['field'], e['from'], e['to'], e['status']) for e in entries],
                         [('C08', '6', 'coverage', 'COVERS_TEXT', 'PARTIAL', 'PENDING')])           # the APPROVE decision proposes nothing
        entry = entries[0]
        self.assertEqual((entry['origin'], entry['proposed_by'], entry['proposer_role']), ('pilot-review', 'Ayşe Hukukçu', 'AML lawyer'))
        self.assertIn('Phone number missing', entry['reason'])
        self.assertEqual(entry['evidence']['analysis_head'], self.results['C08']['head'])
        self.assertFalse((self.dir / 'tr-aml-v1.reviewed.json').exists())
        autonomous = self.dir / 'auto.json'
        autonomous.write_text(json.dumps(self.review(reviewer=AUTONOMOUS_REVIEWER)), encoding='utf-8')
        outcome = import_review(DATASET, self.run_dir, autonomous, self.dir)
        self.assertEqual(outcome['proposed'], [])                               # an AI review is not a person's decision
        stranger = self.dir / 'other.json'
        stranger.write_text(json.dumps({**self.review(), 'analysis_head': 'f' * 64}), encoding='utf-8')
        code, output = quiet(['labels', 'import-review', '--dataset', str(DATASET), '--dir', str(self.dir), '--run', str(self.run_dir),
                              '--review', str(stranger)])
        self.assertEqual(code, 1)
        self.assertIn('analysis head', output)

    # Review finding 3: an override that a duty does not apply takes back the coverage and conflict the label asserted.
    def test_a_does_not_apply_or_unknown_override_also_proposes_the_coverage_and_conflict_it_implies(self):
        for applicability, coverage in (('DOES_NOT_APPLY', 'NOT_ASSESSED'), ('UNKNOWN', 'ANY')):
            with self.subTest(applicability):
                folder = Path(tempfile.mkdtemp(dir=self.root))
                review = self.review()
                review['decisions'][0].update(applicability=applicability, coverage='NO_EVIDENCE')
                path = folder / 'review.json'
                path.write_text(json.dumps(review, ensure_ascii=False), encoding='utf-8')
                outcome = import_review(DATASET, self.run_dir, path, folder)
                self.assertEqual([(e['field'], e['from'], e['to']) for e in outcome['proposed']],
                                 [('applicability', 'APPLIES', applicability), ('coverage', 'COVERS_TEXT', coverage), ('conflict', False, None)])
                self.assertEqual(outcome['skipped'], [])

    # Review finding 12: the package goes to an outside reviewer; no local absolute path goes with it.
    def test_the_expert_package_holds_no_absolute_local_path(self):
        out = self.dir / 'expert'
        summary = write_expert_review(self.run_dir, out, DATASET)
        bundle = json.loads((out / 'expert_review.json').read_text(encoding='utf-8'))
        self.assertEqual((bundle['run'], bundle['run_name'], bundle['dataset']['path']),
                         (self.run_dir.name, self.run_dir.name, 'evaluation/datasets/tr-aml-v1.json'))
        readme = (out / 'README.md').read_text(encoding='utf-8')
        self.assertIn('--dataset ../evaluation/datasets/tr-aml-v1.json', readme)
        texts = {'README.md': readme, 'expert_review.json': (out / 'expert_review.json').read_text(encoding='utf-8'),
                 'expert_review.csv': (out / 'expert_review.csv').read_text(encoding='utf-8-sig'),
                 'README sheet': json.dumps(read_xlsx(out / 'expert_review.xlsx', sheet='README'), ensure_ascii=False)}
        local = {str(REPO), REPO.as_posix(), json.dumps(str(REPO))[1:-1], str(self.root), self.root.as_posix(), json.dumps(str(self.root))[1:-1]}
        for name, text in texts.items():
            for path in local:
                self.assertNotIn(path, text, name)
        self.assertEqual(Path(summary['dataset']), DATASET)                     # the operator's console still gets the local file

    def test_the_cli_runs_the_whole_flow_and_verify_fails_on_a_tampered_log(self):
        base = ['--dataset', str(DATASET), '--dir', str(self.dir)]
        code, output = quiet(['labels', 'propose', *base, '--case', 'C09', '--article', '7', '--clause', '1', '--field', 'coverage', '--to', 'PARTIAL',
                              '--reason', 'Phone number and signature sample missing.', '--actor', 'Analyst A', '--origin', 'label-audit',
                              '--run', str(self.run_dir), '--evidence-json', '{"annotation": "C09 md.7(1)"}'])
        self.assertEqual(code, 0, output)
        entry_id = json.loads(output)['proposed'][0]
        entry = Ledger(DATASET, self.dir).pending()['entries'][0]
        self.assertEqual((entry['evidence']['annotation'], entry['evidence']['run']), ('C09 md.7(1)', self.run_dir.name))
        self.assertTrue(entry['evidence']['regulation_quote'].startswith('(1)'))
        code, output = quiet(['labels', 'pending', *base])
        self.assertEqual(code, 0)
        self.assertIn(entry_id, output)
        self.assertEqual(quiet(['labels', 'approve', *base, '--ids', entry_id, '--approver', 'Analyst A'])[0], 1)   # four-eyes
        self.assertEqual(quiet(['labels', 'approve', *base, '--ids', entry_id[:8], '--approver', 'Approver B'])[0], 0)
        code, output = quiet(['labels', 'verify', *base])
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output)['approved'], 1)
        audit = self.dir / 'tr-aml-v1.audit.jsonl'
        audit.write_text(audit.read_text(encoding='utf-8').replace('Approver B', 'Approver C'), encoding='utf-8')
        self.assertEqual(quiet(['labels', 'verify', *base])[0], 1)

    def test_the_expert_export_writes_csv_json_and_an_xlsx_excel_can_open(self):
        out = self.dir / 'expert'
        code, output = quiet(['expert-export', '--run', str(self.run_dir), '--dataset', str(DATASET), '--out', str(out)])
        self.assertEqual(code, 0, output)
        raw = (out / 'expert_review.csv').read_bytes()
        self.assertTrue(raw.startswith(b'\xef\xbb\xbf'))                          # UTF-8 byte-order mark for Excel
        header = next(csv.reader(io.StringIO(raw.decode('utf-8-sig')), delimiter=';'))
        self.assertEqual(header[:len(COLUMNS)], list(COLUMNS))
        rows = read_sheet(out / 'expert_review.csv')
        bundle = json.loads((out / 'expert_review.json').read_text(encoding='utf-8'))
        self.assertEqual((bundle['format'], len(bundle['rows'])), ('cardaman-expert-review-v1', len(rows)))
        dataset = load_dataset(DATASET)
        expected = sum(len(c.expected_obligations) for c in dataset.cases if c.case_id in ('C08', 'C09'))
        extras = sum(len(self.results[c]['extras']) for c in ('C08', 'C09'))
        self.assertEqual(len(rows), expected + extras)                           # every expectation, plus every unclaimed duty
        self.assertEqual(sum(r['row_kind'] == 'UNEXPECTED_CANDIDATE' for r in rows), extras)
        self.assertTrue(all(r['reviewer_decision'] == r['reviewer_reason'] == r['reviewer_confidence'] == '' for r in rows))
        c08 = next(r for r in rows if r['case'] == 'C08' and r['article/clause'] == 'md.6(1)')
        self.assertEqual((c08['expected_coverage'], c08['label_source'], c08['row_id']), ('COVERS_TEXT', 'RULE_DERIVED', 'C08|6|1|kimlik tespitinde'))
        self.assertTrue(c08['regulation_text'].startswith('(1)') and c08['company_profile'].startswith('CepÖdeme'))
        with zipfile.ZipFile(out / 'expert_review.xlsx') as archive:
            self.assertIsNone(archive.testzip())
            names = archive.namelist()
            for part in ('[Content_Types].xml', '_rels/.rels', 'xl/workbook.xml', 'xl/_rels/workbook.xml.rels', 'xl/styles.xml',
                         'xl/worksheets/sheet1.xml', 'xl/worksheets/sheet2.xml'):
                self.assertIn(part, names)
            parsed = {name: ElementTree.fromstring(archive.read(name)) for name in names}       # every part is well-formed XML
        sheets = [s.get('name') for s in parsed['xl/workbook.xml'].iter(SHEET_NS + 'sheet')]
        self.assertEqual(sheets, ['review', 'README'])
        types = {o.get('PartName') for o in parsed['[Content_Types].xml']}
        self.assertTrue({'/xl/workbook.xml', '/xl/worksheets/sheet1.xml', '/xl/worksheets/sheet2.xml', '/xl/styles.xml'} <= types)
        first = next(parsed['xl/worksheets/sheet1.xml'].iter(SHEET_NS + 'row'))
        self.assertEqual([''.join(t.text or '' for t in c.iter(SHEET_NS + 't')) for c in first], list(ALL_COLUMNS))
        self.assertEqual(read_xlsx(out / 'expert_review.xlsx'), rows)             # the workbook holds the same rows as the CSV
        readme = (out / 'README.md').read_text(encoding='utf-8')
        self.assertIn('reviewer_decision', readme)
        self.assertIn('labels import-review', readme)

    def test_xlsx_escapes_xml_and_drops_characters_xml_cannot_hold(self):
        path = self.dir / 'tricky.xlsx'
        tricky = {'case': 'C<1>&"x"', 'obligation': 'a\x0bb\nline two', 'reviewer_reason': ''}
        write_xlsx(path, [tricky], columns=['case', 'obligation', 'reviewer_reason'], readme_lines=['<README> & notes'])
        self.assertEqual(read_xlsx(path), [{'case': 'C<1>&"x"', 'obligation': 'ab\nline two', 'reviewer_reason': ''}])
        self.assertEqual(read_xlsx(path, sheet='README'), [{'README': '<README> & notes'}])

    def test_a_filled_sheet_comes_back_as_pending_proposals_only(self):
        out = self.dir / 'expert'
        write_expert_review(self.run_dir, out, DATASET)
        rows = read_sheet(out / 'expert_review.csv')
        pick = lambda case, where: next(r for r in rows if r['case'] == case and r['article/clause'] == where)
        pick('C08', 'md.6(1)').update(reviewer_decision='SET coverage=PARTIAL', reviewer_reason='Telefon numarası politikada yok.',
                                      reviewer_confidence='Yüksek')
        pick('C09', 'md.7(1)').update(reviewer_decision='Etiket doğru', reviewer_reason='Tam karşılık.')
        pick('C09', 'md.7(4)').update(reviewer_decision='SET coverage=PARTIAL')                 # no reason: nothing proposed
        write_xlsx(out / 'filled.xlsx', rows)
        code, output = quiet(['labels', 'import-review', '--dataset', str(DATASET), '--dir', str(self.dir), '--sheet', str(out / 'filled.xlsx'),
                              '--actor', 'Ayşe Hukukçu', '--role', 'AML lawyer'])
        self.assertEqual(code, 0, output)
        summary = json.loads(output)
        self.assertEqual((len(summary['proposed']), summary['label_confirmed']), (1, 1))
        self.assertTrue(any('reviewer_reason is empty' in s['reason'] for s in summary['skipped']))
        entry = Ledger(DATASET, self.dir).pending()['entries'][0]
        self.assertEqual((entry['case_id'], entry['field'], entry['from'], entry['to'], entry['origin'], entry['status']),
                         ('C08', 'coverage', 'COVERS_TEXT', 'PARTIAL', 'expert-review', 'PENDING'))
        self.assertEqual((entry['evidence']['reviewer_confidence'], entry['evidence']['sheet']), ('HIGH', 'filled.xlsx'))
        self.assertFalse((self.dir / 'tr-aml-v1.reviewed.json').exists())

    def test_decisions_and_the_ai_correct_shortcut(self):
        self.assertEqual(parse_decision('düzelt applicability=DOES_NOT_APPLY; conflict=false'),
                         ('SET', {'applicability': 'DOES_NOT_APPLY', 'conflict': 'false'}))
        self.assertEqual(parse_decision('coverage=PARTIAL'), ('SET', {'coverage': 'PARTIAL'}))
        self.assertEqual(parse_decision('Emin değilim')[0], 'UNSURE')
        with self.assertRaises(ValueError):
            parse_decision('SET notes=x')
        row = {'row_id': 'C13|5|2|işlem yapılmadan önce tamamlanır', 'reviewer_decision': 'AI_CORRECT', 'reviewer_reason': 'The skip is a conflict.',
               'expected_applicability': 'APPLIES', 'expected_coverage': 'COVERS_TEXT', 'expected_conflict': 'false',
               'ai_applicability': 'APPLIES', 'ai_coverage': 'CONFLICT', 'ai_conflict': 'true'}
        changes, note = sheet_changes(row)
        self.assertIsNone(note)
        self.assertEqual([(c['field'], c['from'], c['to']) for c in changes], [('coverage', 'COVERS_TEXT', 'CONFLICT'), ('conflict', 'false', 'true')])
        self.assertEqual(changes[0]['action_keywords'], ['işlem yapılmadan önce tamamlanır'])
        self.assertEqual(sheet_changes({**row, 'row_id': 'C13|+|YONETMELIK:200713012/md.5/2/abc'})[0], [])     # an unexpected candidate
        self.assertEqual(sheet_changes({**row, 'reviewer_decision': ''}), ([], None))
        # A sheet re-saved by a Turkish Excel: true/false became DOĞRU/YANLIŞ, and a plain CSV save is cp1254, not UTF-8.
        excel = {**row, 'reviewer_decision': 'AI doğru', 'expected_conflict': 'YANLIŞ', 'ai_conflict': 'DOĞRU'}
        self.assertEqual([(c['field'], c['to']) for c in sheet_changes(excel)[0]], [('coverage', 'CONFLICT'), ('conflict', 'true')])
        path = self.dir / 'ansi.csv'
        path.write_bytes('row_id;reviewer_reason\nC13|5|2|x;Şüpheli işlem çelişkisi\n'.encode('cp1254'))
        self.assertEqual(read_sheet(path), [{'row_id': 'C13|5|2|x', 'reviewer_reason': 'Şüpheli işlem çelişkisi'}])


if __name__ == '__main__':
    unittest.main()

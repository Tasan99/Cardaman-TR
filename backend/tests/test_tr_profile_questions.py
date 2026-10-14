"""Profile questions (tr/questions.py): a company fact missing from the profile is asked once, the answer is applied with
its basis, and only the evaluations it affects run again. Profiles and registers are SYNTHETIC; the answers here are
test answers, not company facts."""
import io
import unittest

from regchain.tr import questions as q
from regchain.tr.adjudicate import assess_profile
from regchain.tr.compare import load_register
from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles
from regchain.tr.routing import evaluate_scope
from test_tr_routing import by_target, profile, scope

REGISTRY = Registry.load()
STORE = CorpusStore()
BOTTLER = load_pilot_profiles(REGISTRY.vocabulary)['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
REGISTER = load_register(BOTTLER.profile_id)
OBLIGATIONS = [o for article in ('17', '20') for o in extract_regulation('TR:YONETMELIK:INSANI_TUKETIM_SULAR', REGISTRY, STORE, articles=[article])[1]]
REPORT, ASSESSMENTS, _ = assess_profile(BOTTLER, OBLIGATIONS, REGISTER, REGISTRY, STORE)


def answer(question, present, complete=True, by='test.reviewer', basis='TEST ANSWER on a synthetic profile'):
    return q.Answer(question_id=question.question_id, present=present, list_complete=complete, answered_by=by,
                    answered_at='2026-10-03T10:00:00+03:00', basis=basis)


class DimensionCompletenessTests(unittest.TestCase):
    def test_a_dimension_stated_complete_decides_while_the_others_stay_open(self):
        # SALES: profile_complete False; its activities stated complete, its licences not
        prof = profile(lambda d: d['legal_entities'][1].update({'activities_complete': True}))
        decision = by_target(evaluate_scope(scope('LEGAL_ENTITY', activity_classes=['PRODUCTION']), prof, REGISTRY))['SALES']
        self.assertEqual((decision.status, decision.reason_codes), ('DOES_NOT_APPLY', ['ACTIVITY_MISMATCH']))
        decision = by_target(evaluate_scope(scope('LEGAL_ENTITY', license_classes=['ALCOHOL_PRODUCTION_PERMIT']), prof, REGISTRY))['SALES']
        self.assertEqual(decision.status, 'UNKNOWN')

    def test_an_explicit_open_dimension_is_open_on_a_complete_profile(self):
        prof = profile(lambda d: d['legal_entities'][0].update({'activities_complete': False}))   # PROD: profile_complete True
        decision = by_target(evaluate_scope(scope('LEGAL_ENTITY', activity_classes=['RETAIL_SALE']), prof, REGISTRY))['PROD']
        self.assertEqual(decision.status, 'UNKNOWN')


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.questions = q.profile_questions(BOTTLER, REPORT.applicability_reviews, REGISTRY.vocabulary)
        self.brand = next(x for x in self.questions if x.target_id == 'NONALC-BRAND' and x.gate == 'ACTIVITY_CLASS')

    def test_one_question_per_entity_and_missing_fact(self):
        self.assertEqual(len({x.question_id for x in self.questions}), len(self.questions))
        blocked = [r for r in REPORT.applicability_reviews if r.target_id == 'NONALC-BRAND'
                   and any(f['gate'] == 'ACTIVITY_CLASS' for f in r.missing_facts)]
        self.assertTrue(blocked)
        self.assertEqual(sorted(self.brand.blocked_reviews), sorted(r.review_id for r in blocked))
        wanted = {v for r in blocked for f in r.missing_facts if f['gate'] == 'ACTIVITY_CLASS' for v in f['required']}
        stated = set(BOTTLER.entity('NONALC-BRAND').activity_classes)
        self.assertEqual({a['value'] for a in self.brand.asked}, wanted - stated)
        self.assertIn('NONALC-BRAND', self.brand.prompt_tr)
        # a source question is not a profile question
        self.assertFalse([x for x in self.questions if x.gate == 'EXCEPTIONS'])

    def test_an_answer_is_applied_with_its_basis_and_checked(self):
        value = self.brand.asked[0]['value']
        new, applied = q.apply_answers(BOTTLER, [answer(self.brand, [value])], self.questions, REGISTRY.vocabulary)
        entity = new.entity('NONALC-BRAND')
        self.assertIn(value, entity.activity_classes)
        self.assertTrue(entity.activities_complete)
        self.assertIn('test.reviewer', entity.facts_basis['activity_classes'])
        self.assertFalse(BOTTLER.entity('NONALC-BRAND').activities_complete)          # the pilot profile is not changed
        self.assertEqual(applied[0]['question_id'], self.brand.question_id)
        with self.assertRaises(q.AnswerError):                                         # not in the vocabulary
            q.apply_answers(BOTTLER, [answer(self.brand, ['NOT_A_CLASS'])], self.questions, REGISTRY.vocabulary)
        with self.assertRaises(q.AnswerError):                                         # no such question
            q.apply_answers(BOTTLER, [answer(self.brand, []).model_copy(update={'question_id': 'X'})], self.questions, REGISTRY.vocabulary)
        with self.assertRaises(ValueError):                                            # who answered is required
            answer(self.brand, [], by='')

    def test_questions_of_another_profile_are_refused_naming_the_profile(self):
        other = {p.profile_id: p for p in load_pilot_profiles(REGISTRY.vocabulary).values()}['tr-bev-pilot-alcohol-integrated']
        with self.assertRaises(q.AnswerError) as caught:                               # NONALC-BRAND is not an entity of it
            q.apply_answers(other, [answer(self.brand, [])], self.questions, REGISTRY.vocabulary)
        self.assertIn('tr-bev-pilot-alcohol-integrated', str(caught.exception))


class ReassessTests(unittest.TestCase):
    def test_only_the_affected_evaluations_run_again(self):
        questions = q.profile_questions(BOTTLER, REPORT.applicability_reviews, REGISTRY.vocabulary)
        brand = next(x for x in questions if x.target_id == 'NONALC-BRAND' and x.gate == 'ACTIVITY_CLASS')
        new, _ = q.apply_answers(BOTTLER, [answer(brand, [])], questions, REGISTRY.vocabulary)   # none of them, list complete
        result = q.reassess(BOTTLER, new, OBLIGATIONS, REGISTER, REGISTRY, STORE)
        self.assertEqual(result['affected_entities'], ['NONALC-BRAND'])
        self.assertTrue(result['changes'])
        for change in result['changes']:
            self.assertEqual(change['entity_id'], 'NONALC-BRAND')
            self.assertEqual(change['before'], 'UNKNOWN')
            self.assertIn(change['after'], ('DOES_NOT_APPLY', 'UNKNOWN', 'APPLIES', 'PARTIAL'))
        resolved = {c['review_id'] for c in result['changes'] if c['after'] != 'UNKNOWN'}
        self.assertTrue(resolved <= set(brand.blocked_reviews))
        # nothing about the other entities is evaluated again
        self.assertTrue(result['evaluated'])
        self.assertTrue(all(k.endswith(':NONALC-BRAND') for k in result['evaluated']))
        # a duty that now applies is assessed against the register
        for row in result['rows']:
            self.assertEqual(row['target_id'], 'NONALC-BRAND')


class QuestionnaireTests(unittest.TestCase):
    """The company fills the questions in a questionnaire (CSV); the filled file is read back as answers."""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.questions = q.profile_questions(BOTTLER, REPORT.applicability_reviews, REGISTRY.vocabulary)
        self.brand = next(x for x in self.questions if x.target_id == 'NONALC-BRAND' and x.gate == 'ACTIVITY_CLASS')
        self.directory = tempfile.TemporaryDirectory()
        self.sheet = Path(self.directory.name) / 'questionnaire.csv'
        self.lines = q.write_questionnaire(self.questions, self.sheet)

    def tearDown(self):
        self.directory.cleanup()

    def rows(self):
        import csv
        with self.sheet.open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle, delimiter=';'))

    def fill(self, rows):
        import csv
        with self.sheet.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=q.QUESTIONNAIRE_COLUMNS, delimiter=';')
            writer.writeheader()
            writer.writerows(rows)

    def test_one_line_per_value_and_one_completeness_line_per_question(self):
        rows = self.rows()
        self.assertEqual(len(rows), self.lines)
        self.assertEqual(len(rows), sum(len(x.asked) + 1 for x in self.questions))
        self.assertEqual(sum(r['satir'] == 'LISTE_TAM' for r in rows), len(self.questions))
        self.assertEqual(q.read_questionnaire(self.sheet, self.questions), [])            # a blank questionnaire answers nothing

    def test_a_filled_questionnaire_is_read_back_as_answers(self):
        rows = self.rows()
        first = self.brand.asked[0]['value']
        for row in rows:
            if row['question_id'] == self.brand.question_id:
                if row['satir'] == 'DEGER':
                    row['cevap'] = 'E' if row['deger_kodu'] == first else 'H'
                else:
                    row.update({'cevap': 'E', 'cevaplayan': 'marka.uyum', 'dayanak': 'faaliyet belgesi 2026', 'tarih': '2026-10-04'})
        self.fill(rows)
        answers = q.read_questionnaire(self.sheet, self.questions)
        self.assertEqual([(a.question_id, a.present, a.list_complete, a.answered_by, a.basis) for a in answers],
                         [(self.brand.question_id, [first], True, 'marka.uyum', 'faaliyet belgesi 2026')])

    def test_an_answer_without_who_or_basis_or_with_another_mark_is_refused(self):
        rows = self.rows()
        brand_rows = [r for r in rows if r['question_id'] == self.brand.question_id]
        brand_rows[0]['cevap'] = 'E'
        self.fill(rows)
        with self.assertRaises(q.AnswerError):
            q.read_questionnaire(self.sheet, self.questions)                              # no cevaplayan / dayanak
        brand_rows[0]['cevap'] = 'belki'
        brand_rows[-1].update({'cevaplayan': 'x', 'dayanak': 'y'})
        self.fill(rows)
        with self.assertRaises(q.AnswerError):
            q.read_questionnaire(self.sheet, self.questions)                              # neither E nor H

    # The sheet comes back from Excel on a Turkish Windows, or from another spreadsheet program.

    def filled(self):
        """The brand question answered: its first value E, the others H, the list complete; who and basis in Turkish."""
        rows = self.rows()
        first = self.brand.asked[0]['value']
        for row in rows:
            if row['question_id'] == self.brand.question_id:
                if row['satir'] == 'DEGER':
                    row['cevap'] = 'E' if row['deger_kodu'] == first else 'H'
                else:
                    row.update({'cevap': 'E', 'cevaplayan': 'Ayşe Yılmaz (Uyum Müdürü)', 'dayanak': 'faaliyet belgesi; şirket sözleşmesi',
                                'tarih': '2026-10-04'})
        return rows

    def expected(self):
        return [(self.brand.question_id, [self.brand.asked[0]['value']], True, 'Ayşe Yılmaz (Uyum Müdürü)', '2026-10-04',
                 'faaliyet belgesi; şirket sözleşmesi')]

    def save(self, rows, encoding='utf-8-sig', delimiter=';', columns=None):
        """The rows as a program saves them: its encoding, its separator, the columns it kept."""
        import csv
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=columns or q.QUESTIONNAIRE_COLUMNS, delimiter=delimiter, extrasaction='ignore',
                                lineterminator='\r\n')
        writer.writeheader()
        writer.writerows(rows)
        self.sheet.write_bytes(text.getvalue().encode(encoding))

    def read(self):
        return [(a.question_id, a.present, a.list_complete, a.answered_by, a.answered_at, a.basis)
                for a in q.read_questionnaire(self.sheet, self.questions)]

    def test_an_excel_save_on_a_turkish_windows_is_read_like_the_written_sheet(self):
        # Excel 'CSV (virgülle ayrılmış)' on a Turkish Windows: Windows-1254 with ';' (',' on another list separator);
        # 'CSV UTF-8': UTF-8 with a BOM; 'Unicode metin': UTF-16 with a BOM and TABs; Google Sheets, LibreOffice: UTF-8 with ','
        rows, expected = self.filled(), self.expected()
        for encoding, delimiter in [('utf-8-sig', ';'), ('utf-8', ';'), ('cp1254', ';'), ('cp1254', ','), ('utf-16', '\t'),
                                    ('utf-8', ','), ('utf-8-sig', '\t')]:
            with self.subTest(encoding=encoding, delimiter=delimiter):
                self.save(rows, encoding, delimiter)
                self.assertEqual(self.read(), expected)

    def test_a_file_that_is_no_known_text_is_refused_saying_how_to_save_it(self):
        self.save(self.filled(), 'cp1254')
        self.sheet.write_bytes(self.sheet.read_bytes() + b'\x81')            # 0x81: neither UTF-8 nor a Windows-1254 letter
        with self.assertRaises(q.AnswerError) as caught:
            q.read_questionnaire(self.sheet, self.questions)
        self.assertIn('CSV UTF-8', str(caught.exception))
        self.sheet.write_bytes(b'PK\x03\x04' + bytes(range(256)))               # an Excel workbook (.xlsx is a zip file)
        with self.assertRaises(q.AnswerError) as caught:
            q.read_questionnaire(self.sheet, self.questions)
        self.assertIn('CSV UTF-8', str(caught.exception))

    def test_a_missing_column_is_refused_naming_it_and_the_columns_found(self):
        rows = self.filled()
        for missing in (['question_id'], ['satir'], ['deger_kodu'], ['cevap'], ['cevaplayan', 'dayanak']):
            with self.subTest(missing=missing):
                kept = [c for c in q.QUESTIONNAIRE_COLUMNS if c not in missing]
                self.save(rows, columns=kept)
                with self.assertRaises(q.AnswerError) as caught:
                    q.read_questionnaire(self.sheet, self.questions)
                self.assertIn(f"missing columns: {', '.join(missing)} (found: {', '.join(kept)})", str(caught.exception))

    def test_lines_excel_leaves_blank_and_a_retyped_header_do_not_change_the_answers(self):
        rows, expected = self.filled(), self.expected()
        blank = dict.fromkeys(q.QUESTIONNAIRE_COLUMNS, '')
        self.save([*rows, blank, blank])                                        # ';;;;;;;;;' lines below the data
        self.assertEqual(self.read(), expected)
        self.save(rows)
        text = self.sheet.read_bytes().decode('utf-8-sig')
        self.sheet.write_bytes(text.replace(';cevap;', '; Cevap ;', 1).encode('utf-8-sig'))
        self.assertEqual(self.read(), expected)

    def test_e_and_h_are_read_in_their_turkish_spellings(self):
        expected = self.expected()
        for yes, no in [('e', 'h'), ('Evet', 'Hayır'), ('evet', 'hayır'), ('EVET', 'HAYIR'), ('EVET', 'HAYİR'), (' E ', 'hayir')]:
            with self.subTest(yes=yes, no=no):
                rows = self.filled()
                for row in rows:
                    row['cevap'] = {'E': yes, 'H': no}.get(row['cevap'], row['cevap'])
                self.save(rows)
                self.assertEqual(self.read(), expected)

    def test_a_value_left_blank_is_still_not_named_by_the_answer(self):
        # unchanged by the import fixes: blank is not an answer, and with the list stated complete the value is absent
        rows = self.filled()
        for row in rows:
            if row['question_id'] == self.brand.question_id and row['satir'] == 'DEGER' and row['cevap'] == 'H':
                row['cevap'] = ''
        self.save(rows, 'cp1254')
        self.assertEqual(self.read(), self.expected())

    def test_contradictory_duplicate_rows_are_refused_naming_their_lines(self):
        rows = self.filled()
        lines = [n for n, row in enumerate(rows, 2) if row['question_id'] == self.brand.question_id]
        value, complete = rows[lines[0] - 2], rows[lines[-1] - 2]
        copied = len(rows) + 2
        self.save([*rows, {**value, 'cevap': 'H'}])                              # the same value E on one line, H on another
        with self.assertRaises(q.AnswerError) as caught:
            q.read_questionnaire(self.sheet, self.questions)
        self.assertIn(value['deger_kodu'], str(caught.exception))
        self.assertRegex(str(caught.exception), rf'\bline {lines[0]}\b')
        self.assertRegex(str(caught.exception), rf'\bline {copied}\b')
        self.save([*rows, {**complete, 'cevap': 'H', 'cevaplayan': 'Mehmet Demir', 'dayanak': 'sözlü teyit'}])   # two completeness answers
        with self.assertRaises(q.AnswerError) as caught:
            q.read_questionnaire(self.sheet, self.questions)
        self.assertRegex(str(caught.exception), rf'\bline {lines[-1]}\b')
        self.assertRegex(str(caught.exception), rf'\bline {copied}\b')

    def test_an_exact_duplicate_row_is_read_once(self):
        rows = self.filled()
        brand = [row for row in rows if row['question_id'] == self.brand.question_id]
        blank_copy = {**brand[-1], 'cevap': '', 'cevaplayan': '', 'dayanak': '', 'tarih': ''}
        self.save([*rows, dict(brand[0]), dict(brand[-1]), {**brand[0], 'cevap': ''}, blank_copy])   # copied as they are, or left blank
        self.assertEqual(self.read(), self.expected())

    def test_a_line_kind_other_than_DEGER_or_LISTE_TAM_is_refused_with_its_line(self):
        rows = self.filled()
        number = next(n for n, row in enumerate(rows, 2) if row['question_id'] == self.brand.question_id)
        rows[number - 2]['satir'] = 'DEĞER'
        self.save(rows)
        with self.assertRaises(q.AnswerError) as caught:
            q.read_questionnaire(self.sheet, self.questions)
        self.assertRegex(str(caught.exception), rf'\bline {number}\b')


class InteractiveTests(unittest.TestCase):
    def test_answers_from_the_terminal(self):
        questions = q.profile_questions(BOTTLER, REPORT.applicability_reviews, REGISTRY.vocabulary)
        brand = next(x for x in questions if x.target_id == 'NONALC-BRAND' and x.gate == 'ACTIVITY_CLASS')
        others = [x for x in questions if x is not brand]
        # who answers, then: brand -> item 1, list complete, basis; every other question skipped
        feed = ['uyum.sorumlusu', '1', 'e', 'faaliyet listesi teyit edildi'] + ['s'] * len(others)
        lines = iter(feed)
        out = io.StringIO()
        answers = q.ask_interactively([brand, *others], input_fn=lambda prompt='': (out.write(prompt), next(lines))[1], output=out)
        self.assertEqual(len(answers), 1)
        self.assertEqual((answers[0].question_id, answers[0].present, answers[0].list_complete, answers[0].answered_by),
                         (brand.question_id, [brand.asked[0]['value']], True, 'uyum.sorumlusu'))
        self.assertIn('NONALC-BRAND', out.getvalue())


class CommandLineTests(unittest.TestCase):
    """`python -m regchain.tr questions / answer` as a company runs them: a refused input is one line on stderr with exit
    code 2 (no traceback), a filled questionnaire is never overwritten, and the output is UTF-8 whatever the code page."""

    def setUp(self):
        import json
        import tempfile
        from pathlib import Path
        self.directory = tempfile.TemporaryDirectory()
        self.folder = Path(self.directory.name)
        questions = q.profile_questions(BOTTLER, REPORT.applicability_reviews, REGISTRY.vocabulary)
        self.questions = self.folder / 'questions.json'
        self.questions.write_text(json.dumps([x.model_dump(mode='json') for x in questions], ensure_ascii=False), encoding='utf-8')
        self.sheet = self.folder / 'questionnaire.csv'
        q.write_questionnaire(questions, self.sheet)

    def tearDown(self):
        self.directory.cleanup()

    def cli(self, *argv):
        import contextlib
        from regchain.tr.__main__ import main
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main([str(a) for a in argv])
        return code, err.getvalue()

    def sheet_text(self, mark):
        """The written questionnaire with its first question answered `mark` on its first value, as text with ';'."""
        import csv
        with self.sheet.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle, delimiter=';'))
        rows[0]['cevap'] = mark
        rows[next(n for n, r in enumerate(rows) if r['satir'] == 'LISTE_TAM')].update({'cevaplayan': 'Ayşe Yılmaz', 'dayanak': 'faaliyet belgesi'})
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=q.QUESTIONNAIRE_COLUMNS, delimiter=';', lineterminator='\r\n')
        writer.writeheader()
        writer.writerows(rows)
        return text.getvalue()

    def test_questions_does_not_overwrite_a_questionnaire_unless_forced(self):
        from unittest import mock
        filled = 'question_id;satir;cevap;cevaplayan\r\nX;LISTE_TAM;E;Ayşe Yılmaz\r\n'.encode('utf-8-sig')
        self.sheet.write_bytes(filled)
        self.questions.unlink()
        command = ['questions', '--profile', BOTTLER.profile_id, '--out', self.questions]
        with mock.patch('regchain.tr.engines.assess_by_engine', return_value={'reports': {}}) as assess:
            code, err = self.cli(*command)
            self.assertEqual(code, 2)
            self.assertEqual(len(err.strip().splitlines()), 1, err)
            self.assertIn(str(self.sheet), err)
            self.assertIn('--force', err)
            self.assertEqual(self.sheet.read_bytes(), filled)                  # the company's answers are kept
            self.assertFalse(self.questions.exists())
            assess.assert_not_called()                                         # refused before the assessment runs
            self.sheet.unlink()
            self.questions.write_text('[]', encoding='utf-8')                  # questions.json alone is not overwritten either
            self.assertEqual(self.cli(*command)[0], 2)
            self.assertFalse(self.sheet.exists())
            self.sheet.write_bytes(filled)
            code, err = self.cli(*command, '--force')
            self.assertEqual((code, err), (0, ''))
            self.assertNotEqual(self.sheet.read_bytes(), filled)

    def test_answer_refuses_a_bad_input_in_one_line_with_exit_code_2(self):
        marked = self.folder / 'excel.csv'
        marked.write_bytes(self.sheet_text('Var').encode('cp1254'))            # Excel 'CSV (virgülle ayrılmış)', Turkish Windows
        unicode_text = self.folder / 'excel.txt'
        unicode_text.write_bytes(self.sheet_text('Var').replace(';', '\t').encode('utf-16'))   # Excel 'Unicode metin'
        answered = self.folder / 'answered.csv'
        answered.write_bytes(self.sheet_text('E').encode('cp1254'))
        workbook = self.folder / 'questionnaire.xlsx'
        workbook.write_bytes(b'PK\x03\x04' + bytes(range(256)))
        lines = self.sheet_text('E').split('\r\n')
        twice = self.folder / 'twice.csv'                                      # its first value answered E, then again H below
        twice.write_bytes('\r\n'.join([*lines[:-1], lines[1].replace(';E;', ';H;', 1), '']).encode('utf-8-sig'))
        broken = self.folder / 'profile.json'
        broken.write_text('{"profile_id": "company"}', encoding='utf-8')     # a profile file that is not a whole profile
        other = 'tr-bev-pilot-alcohol-integrated'
        cases = [('a mark other than E/H', BOTTLER.profile_id, self.questions, marked, "'Var'"),
                 ('the same, saved as Unicode text', BOTTLER.profile_id, self.questions, unicode_text, "'Var'"),
                 ('an Excel workbook', BOTTLER.profile_id, self.questions, workbook, 'CSV UTF-8'),
                 ('a value answered E and H', BOTTLER.profile_id, self.questions, twice, 'contradictory lines'),
                 ('no questions file', BOTTLER.profile_id, self.folder / 'none.json', answered, 'none.json'),
                 ('a questionnaire given as the questions', BOTTLER.profile_id, self.sheet, answered, 'questionnaire.csv'),
                 ('a profile file that is no profile', broken, self.questions, answered, 'validation error'),
                 ('questions of another profile', other, self.questions, answered, other)]
        for number, (name, profile_id, questions, answers, expected) in enumerate(cases):
            with self.subTest(name):
                out = self.folder / f'out-{number}'
                code, err = self.cli('answer', '--profile', profile_id, '--questions', questions, '--answers', answers, '--out', out)
                self.assertEqual(code, 2, err)
                self.assertEqual(len(err.strip().splitlines()), 1, err)
                self.assertTrue(err.startswith('answer: '), err)
                self.assertIn(expected, err)
                self.assertFalse(out.exists())

    def test_a_programming_error_in_answer_keeps_its_traceback(self):
        from unittest import mock
        answered = self.folder / 'answered.csv'
        answered.write_bytes(self.sheet_text('E').encode('utf-8-sig'))
        with mock.patch.object(q, 'apply_answers', side_effect=KeyError('a bug, not an input')):
            with self.assertRaises(KeyError):                                  # not turned into a refused input
                self.cli('answer', '--profile', BOTTLER.profile_id, '--questions', self.questions, '--answers', answered,
                         '--out', self.folder / 'out')

    def test_an_answers_file_saved_with_a_bom_is_read(self):
        import json
        brand = next(x for x in q.profile_questions(BOTTLER, REPORT.applicability_reviews, REGISTRY.vocabulary)
                     if x.target_id == 'NONALC-BRAND' and x.gate == 'ACTIVITY_CLASS')
        answers = [answer(brand, []).model_dump(mode='json')]
        path = self.folder / 'answers.json'
        path.write_bytes(json.dumps(answers, ensure_ascii=False).encode('utf-8-sig'))      # Notepad's 'UTF-8 with BOM'
        self.assertEqual([a.model_dump(mode='json') for a in q.load_list(path, q.Answer, 'answers')], answers)

    def test_output_redirected_to_a_file_is_utf8_without_python_utf8_mode(self):
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path
        source = str(Path(q.__file__).resolve().parents[2])
        environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONUTF8', 'PYTHONIOENCODING')}
        for name, extra in [('the code page of this machine', {}), ('a Turkish Windows code page', {'PYTHONIOENCODING': 'cp1254'})]:
            with self.subTest(name):
                path = self.folder / f'status-{len(extra)}.json'
                with path.open('wb') as handle:
                    done = subprocess.run([sys.executable, '-m', 'regchain.tr', 'corpus', 'status'], stdout=handle, stderr=subprocess.PIPE,
                                          env={**environment, **extra, 'PYTHONPATH': source}, timeout=300)
                self.assertEqual(done.returncode, 0, done.stderr.decode('utf-8', 'replace')[-400:])
                text = path.read_bytes().decode('utf-8')
                self.assertIn('İ', text)
                self.assertTrue(json.loads(text))


if __name__ == '__main__':
    unittest.main()

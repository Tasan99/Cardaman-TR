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


if __name__ == '__main__':
    unittest.main()

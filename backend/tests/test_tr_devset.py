"""BEVERAGE_TR_DEV_V2: developer labels scored as INDICATIVE. The DEV split is the regression set of the wording rules
(it must stay right); the HOLDOUT split and the policy comparison are pinned as measured, misses included."""
import unittest

from regchain.evaluation.tasks import MIN_TARGET
from regchain.tr import ai, devset
from regchain.tr.corpus import CorpusStore
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles

REGISTRY = Registry.load()
STORE = CorpusStore()
DATASET = devset.load()
RESULT = devset.evaluate(DATASET, REGISTRY, STORE)


def task(group, name, split=None):
    block = RESULT[group] if split is None else RESULT[group][split]
    return block['tasks'][name]


class DatasetTests(unittest.TestCase):
    def test_the_labels_are_a_developers_and_say_so(self):
        self.assertEqual((DATASET['label_status'], DATASET['dataset_id']), ('DEVELOPMENT', 'BEVERAGE_TR_DEV_V2'))
        self.assertIn('not expert gold', DATASET['note'])
        self.assertTrue(DATASET['label_changes'])                                   # corrections after a first scoring are on record
        splits = {c['split'] for c in DATASET['clauses']}
        self.assertEqual(splits, {'DEV', 'HOLDOUT'})
        self.assertGreaterEqual(sum(1 for c in DATASET['clauses'] if c['split'] == 'HOLDOUT'), MIN_TARGET)

    def test_every_label_points_at_a_clause_of_the_stored_version_it_was_made_on(self):
        readers = devset.Readers(DATASET, REGISTRY, STORE)
        self.assertEqual({rid: STORE.head(rid).version_id for rid in DATASET['corpus_versions']}, DATASET['corpus_versions'])
        for clause in DATASET['clauses']:
            self.assertIn(clause['ref'], readers.frames)
        profiles = {p.profile_id: p for p in load_pilot_profiles(REGISTRY.vocabulary).values()}
        for case in DATASET['applicability']:
            for side in case['sides']:
                profile = profiles[side['profile_id']]
                known = {e.entity_id for e in profile.legal_entities} | {p.product_id for p in profile.products} \
                    | {a.activity_id for a in profile.activities} | {f.facility_id for f in profile.facilities}
                self.assertIn(side['target_id'], known, case['case_id'])

    def test_no_brand_name_is_in_the_dataset(self):
        text = devset.DATASET.read_text(encoding='utf-8').lower()
        for name in ('efes', 'tuborg', 'coca', 'pepsi', 'red bull'):
            self.assertNotIn(name, text)


class ClauseTaskTests(unittest.TestCase):
    def test_every_report_is_indicative_whatever_its_numbers(self):
        reports = [*RESULT['clauses']['DEV']['tasks'].values(), *RESULT['clauses']['HOLDOUT']['tasks'].values(),
                   *RESULT['applicability']['tasks'].values(), *RESULT['coverage']['tasks'].values()]
        self.assertEqual({r['claim'] for r in reports}, {'INDICATIVE'})
        self.assertEqual(RESULT['claim'], 'INDICATIVE')
        self.assertEqual(task('applicability', 'APPLICABILITY/rule')['sample_status'], 'INSUFFICIENT_SAMPLE')     # 97 sides
        self.assertEqual(task('coverage', 'POLICY_COVERAGE/rule')['sample_status'], 'INSUFFICIENT_SAMPLE')

    def test_the_dev_clauses_are_the_regression_set_of_the_wording_rules(self):
        for name in ('OBLIGATION_EXTRACTION/rule', 'ADDRESSEE_MATCH/rule', 'EXCEPTION_DETECTION/rule'):
            report = task('clauses', name, 'DEV')
            self.assertEqual(report['accuracy'], 1.0, name)
        self.assertGreaterEqual(task('clauses', 'OBLIGATION_EXTRACTION/rule', 'DEV')['n'], 280)
        self.assertGreaterEqual(task('clauses', 'ADDRESSEE_MATCH/rule', 'DEV')['n'], 200)

    def test_the_holdout_is_reported_as_measured(self):
        # Labelled before any reader was run on these clauses; no rule was changed after the first scoring (105 of 112).
        report = task('clauses', 'OBLIGATION_EXTRACTION/rule', 'HOLDOUT')
        self.assertEqual(report['n'], 112)
        self.assertGreaterEqual(report['accuracy'], 0.93)
        misses = [w for w in RESULT['wrong'] if w.get('split') == 'HOLDOUT']
        self.assertLessEqual(len(misses), 7)
        self.assertEqual({w['layer'] for w in misses}, {'VALIDATOR_PIPELINE'})     # a rule reader's miss is a wording rule's

    def test_applicability_and_its_contrast_pairs(self):
        report = task('applicability', 'APPLICABILITY/rule')
        self.assertEqual((report['n'], report['accuracy'], report['overconfident'], report['unsupported_conclusions']), (97, 1.0, 0, 0))
        pairs = RESULT['applicability']['pairs']
        self.assertEqual((pairs['consistency'], pairs['contrast_detected']), (1.0, pairs['contrast_pairs']))
        self.assertGreaterEqual(pairs['contrast_pairs'], 70)
        self.assertEqual(set(RESULT['applicability']['by_family']),
                         {'ACTIVITY_MATCH', 'ACTOR_MATCH', 'EXCEPTION', 'FACILITY_MATCH', 'PRODUCT_ATTRIBUTE', 'PRODUCT_MATCH'})

    def test_the_rule_comparer_misses_paraphrase_and_invents_no_conflict(self):
        report = task('coverage', 'POLICY_COVERAGE/rule')
        self.assertEqual(report['n'], 44)
        self.assertGreaterEqual(report['accuracy'], 0.72)
        self.assertLess(report['accuracy'], 1.0)                                    # the known limit stays visible
        conflict = report['per_class']['CONTRADICTED']
        self.assertEqual((conflict['fp'], conflict['fn']), (0, 0))
        misses = [w for w in RESULT['wrong'] if w['task'] == 'POLICY_COVERAGE']
        self.assertEqual({w['layer'] for w in misses}, {'VALIDATOR_PIPELINE'})
        self.assertIn('RULE_MISSED_STATEMENT', {w['code'] for w in misses})

    def test_every_wrong_result_names_one_of_the_five_layers(self):
        self.assertTrue(RESULT['wrong'])
        self.assertLessEqual({w['layer'] for w in RESULT['wrong']}, set(ai.LAYERS))
        self.assertTrue(all(w['code'] and w['reader'] for w in RESULT['wrong']))
        self.assertEqual(sum(RESULT['wrong_by_layer']['rule'].values()), len(RESULT['wrong']))


class ModelReadingScoreTests(unittest.TestCase):
    def test_a_recorded_model_reading_is_scored_next_to_the_rule_reader(self):
        readers = devset.Readers(DATASET, REGISTRY, STORE)
        right = {'refs': ['Kanun 4250 md. 6/f.5/c.3'], 'status': 'EXTRACTED', 'reason': 'CANDIDATE_REQUIRES_LEGAL_REVIEW',
                 'candidates': [{'modality': 'MUST_NOT'}], 'diagnostics': [], 'input': ''}
        lost = {'refs': ['Kanun 4250 md. 6/f.8/c.1'], 'status': 'INSUFFICIENT_EVIDENCE', 'reason': 'UNRESOLVED_CROSS_REFERENCE',
                'candidates': [], 'diagnostics': [], 'input': ''}
        invented = {'refs': ['Kanun 4250 md. 6/f.8/c.4'], 'status': 'EXTRACTED', 'reason': 'CANDIDATE_REQUIRES_LEGAL_REVIEW',
                    'candidates': [{'modality': 'MUST', 'source_quote': 'x'}], 'diagnostics': [], 'input': 'y'}
        scored = devset.score_clauses(DATASET, readers, [right, lost, invented], 'DEV')
        report = scored['tasks']['OBLIGATION_EXTRACTION/model']
        self.assertEqual((report['n'], report['claim']), (3, 'INDICATIVE'))
        self.assertEqual(report['sample_status'], 'INSUFFICIENT_SAMPLE')
        wrong = {w['case']: (w['layer'], w['code']) for w in scored['wrong'] if w['reader'] == 'model'}
        self.assertEqual(wrong, {'Kanun 4250 md. 6/f.8/c.1': ('RETRIEVAL', 'REFERENCE_NOT_SUPPLIED'),
                                 'Kanun 4250 md. 6/f.8/c.4': ('MODEL', 'NON_DUTY_READ_AS_DUTY')})
        self.assertEqual(scored['rule_reader_by_model_support']['CONFIRMED'], [1, 0])


if __name__ == '__main__':
    unittest.main()

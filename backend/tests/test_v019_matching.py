"""v0.19 FIX 4: expectations are paired with extracted duties by identity, not by one label word in the action."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from regchain.evaluation import harness as h
from regchain.evaluation.identifiers import BASES, basis_counts, duty_fingerprint, match_predictions, pair_predictions, stems
from regchain.evaluation.schema import ExpectedObligation
from regchain.evaluation.taxonomy import load_results

ROOT = Path(__file__).resolve().parents[2]
V018_RUN = ROOT / 'evaluation' / 'runs' / '20260924-121921-v018-final-ollama'
INDEP_RUN = ROOT / 'evaluation' / 'runs' / '20260924-200143-v019s-indep5-adaptive-ollama'
INDEP = ROOT / 'evaluation' / 'independent' / 'independent-v1.json'


def label(article='4', clause='(1)', keywords=(), **fields):
    return ExpectedObligation(article=article, clause=clause, action_keywords=list(keywords), applicability='APPLIES', **fields)


def duty(article='4', clause='1', action='', subject='Yükümlüler', modality='MUST', conditions=(), exceptions=(), deadline=None, key=None):
    return {'key': key or f'{article}/{clause}/{action[:20]}', 'obligation_id': key or action[:20], 'article': article, 'clause': clause,
            'subject': subject, 'action': action, 'modality': modality, 'conditions': list(conditions), 'exceptions': list(exceptions),
            'deadline': deadline}


class IdentityMatchingTests(unittest.TestCase):
    def test_a_label_word_in_the_condition_matches_the_duty(self):
        reported = duty(action='bu işlemlerin Başkanlığa bildirilmesi', conditions=['işlemde bir şüphe bulunması halinde'])
        (match,), extras = pair_predictions([label(keywords=['şüphe'])], [reported])
        self.assertIs(match['prediction'], reported)
        self.assertEqual((match['basis'], match['candidates'], match['stems'], match['stems_found']), ('CLAUSE_UNIQUE', 1, 1, 1))
        self.assertEqual(extras, [])
        # The deadline, the exceptions and the subject are the duty's own words as well.
        for field, value in (('deadline', 'şüphenin öğrenildiği tarihten itibaren on gün'), ('exceptions', ['şüphe halinde hariç']),
                             ('subject', 'Şüpheli işlem bildirimi')):
            other = {**duty(action='Başkanlığa bildirilmesi', subject='Yükümlüler'), field: value}
            self.assertIs(pair_predictions([label(keywords=['şüphe'])], [other])[0][0]['prediction'], other, field)

    def test_sibling_duties_of_one_clause_are_told_apart_by_their_own_fields(self):
        keep = duty(action='kayıtları sekiz yıl muhafaza etmek', key='keep')
        report = duty(action='Başkanlığa bildirmek', conditions=['bir şüphe bulunması halinde'], key='report')
        matches, extras = pair_predictions([label(keywords=['şüphe']), label(keywords=['muhafaza etmek'])], [keep, report])
        self.assertEqual([m['prediction']['key'] for m in matches], ['report', 'keep'])
        self.assertEqual([m['basis'] for m in matches], ['FINGERPRINT', 'CLAUSE_UNIQUE'])     # 'keep' is alone once 'report' is taken
        self.assertEqual(extras, [])
        # The polarity a label word implies ("uygulayamazlar" = MUST_NOT) decides between duties that both carry the word.
        must = duty(action='basitleştirilmiş tedbirleri uygulamak', key='must')
        must_not = duty(action='basitleştirilmiş tedbirleri uygulamamak', modality='MUST_NOT', key='must_not')
        (match,), _ = pair_predictions([label(keywords=['uygulayamazlar'])], [must, must_not])
        self.assertEqual((match['prediction']['key'], match['basis'], match['polarity']), ('must_not', 'FINGERPRINT', 1))

    def test_a_sub_paragraph_text_shared_by_siblings_does_not_name_either(self):
        shared = 'Şüphe halinde yükümlüler kayıtları saklar ve Başkanlığa bildirir.'
        first = {**duty(action='kayıtları saklamak', key='a'), 'source_quote': shared, 'clause_text': shared}
        second = {**duty(action='Başkanlığa bildirmek', key='b'), 'source_quote': shared, 'clause_text': shared}
        (match,), extras = pair_predictions([label(keywords=['şüphe'])], [first, second])
        self.assertIsNone(match['prediction'])
        self.assertEqual((match['basis'], match['candidates']), ('AMBIGUOUS', 2))
        self.assertEqual(len(extras), 2)

    def test_single_paragraph_article_pairs_empty_clause_with_paragraph_one(self):
        whole = duty(article='8', clause='', action='belgeleri sekiz yıl muhafaza etmek')
        (match,), _ = pair_predictions([label(article='8', clause='(1)', keywords=['muhafaza'])], [whole])
        self.assertEqual((match['prediction'], match['basis']), (whole, 'CLAUSE_UNIQUE'))
        (match,), _ = pair_predictions([label(article='8', clause='', keywords=['muhafaza'])], [duty(article='8', clause='1', action='muhafaza etmek')])
        self.assertEqual(match['basis'], 'CLAUSE_UNIQUE')
        # Once anything in the case names another paragraph of the article, '' and (1) are no longer the same paragraph.
        second = duty(article='8', clause='2', action='yetkililere ibraz etmek')
        (match, _), _ = pair_predictions([label(article='8', clause='(1)', keywords=['muhafaza']), label(article='8', clause='(2)', keywords=['ibraz'])],
                                         [whole, second])
        self.assertEqual((match['prediction'], match['basis']), (None, 'NO_CANDIDATE'))

    def test_ambiguous_and_missing_duties_stay_unmatched(self):
        # The only duty of the clause shares the object but not the act: it is another duty, not this one.
        (match,), extras = pair_predictions([label(article='7', clause='', keywords=['destroy records'])],
                                            [duty(article='7', clause='', action='retain records for five years')])
        self.assertEqual((match['prediction'], match['basis'], match['stems'], match['stems_found']), (None, 'AMBIGUOUS', 2, None))
        self.assertEqual(len(extras), 1)
        (match,), _ = pair_predictions([label(keywords=['şüphe'])], [duty(article='5', action='şüphe halinde bildirmek')])
        self.assertEqual(match['basis'], 'NO_CANDIDATE')
        # A label with no words names no duty: the first duty of its clause, as before v0.19.
        matches, _ = pair_predictions([label(), label()], [duty(action='a', key='x'), duty(action='b', key='y')])
        self.assertEqual([(m['prediction']['key'], m['basis']) for m in matches], [('x', 'CLAUSE_ORDER'), ('y', 'CLAUSE_UNIQUE')])

    def test_each_prediction_is_claimed_once_in_dataset_order(self):
        only = duty(action='Başkanlığa bildirmek', conditions=['şüphe halinde'])
        matches, extras = pair_predictions([label(keywords=['şüphe']), label(keywords=['bildirmek'])], [only])
        self.assertEqual([(m['prediction'] is only, m['basis']) for m in matches], [(True, 'CLAUSE_UNIQUE'), (False, 'NO_CANDIDATE')])
        self.assertEqual(extras, [])
        pairs, _ = match_predictions([label(keywords=['bildir']), label(keywords=['bildir'])],
                                     [duty(action='bildirmek', key='x'), duty(action='bildirilir', key='y')])
        self.assertEqual([p['key'] for _, p in pairs], ['x', 'y'])

    def test_the_fingerprint_is_stable_across_wording_and_names_article_clause_and_modality(self):
        first = duty_fingerprint('4', '(1)', 'MUST', 'Bu işlemlerin Başkanlığa bildirilmesi.')
        self.assertEqual(first, duty_fingerprint('4', '1', 'MUST', 'başkanlığa   bu işlemlerin bildirilmesi'))
        self.assertTrue(first.startswith('4/1/MUST/'))
        self.assertNotEqual(first, duty_fingerprint('4', '1', 'MUST_NOT', 'Bu işlemlerin Başkanlığa bildirilmesi.'))
        self.assertNotEqual(first, duty_fingerprint('4', '2', 'MUST', 'Bu işlemlerin Başkanlığa bildirilmesi.'))
        self.assertEqual(stems('kimliğini tespit etmek'), ['kimli', 'tespi'])
        self.assertEqual(stems('tedbirleri almak'), ['tedbi'])

    def test_every_scored_row_records_its_basis_and_the_metrics_count_them(self):
        from regchain.evaluation.schema import FORMAT, Dataset, EvaluationCase
        case = EvaluationCase(case_id='M01', title='m', categories=[], jurisdiction='TR', regulation_id='KANUN:5549', regulation_fixture='.',
                              target_sections=['4'], company_profile={'name': 'X'},
                              policy_documents=['p.md'], expected_obligations=[label(keywords=['şüphe']), label(article='9', keywords=['ibraz'])])
        prediction = {**duty(action='Başkanlığa bildirmek', conditions=['şüphe halinde']), 'applicability': 'APPLIES', 'coverage': 'COVERS_TEXT',
                      'coverage_assessed': True, 'conflict': False, 'applicability_rule': None, 'ranks': {}, 'policy_evidence': [],
                      'remediation': False, 'draft_status': None}
        scored, extras = h.score_case(case, {'policies': []}, [prediction])
        self.assertEqual([item['match_basis'] for item in scored], ['CLAUSE_UNIQUE', 'NO_CANDIDATE'])
        # problem 1 E-F: the match also names the label's sentences, the candidates from them and the paired duty's fingerprint;
        # the unscored row says why it is missing (this payload records no provision text: NOT_RECORDED).
        self.assertEqual(scored[0]['match'], {'candidates': 1, 'stems': 1, 'stems_found': 1, 'polarity': 0, 'obligation_id': prediction['obligation_id'],
                                              'label_sentences': None, 'from_sentence': None, 'fingerprint': None})
        self.assertEqual((scored[1]['missing_reason'], scored[1]['missing_detail']), ('EXTRACTION_MISSED', 'NOT_RECORDED'))
        json.dumps(scored)                                                              # plain JSON: names and integers
        dataset = Dataset(format=FORMAT, dataset_id='t', version='1', cases=[case])
        metrics = h.aggregate(dataset, [{'case_id': 'M01', 'predictions': [prediction], 'scored': scored, 'extras': extras, 'calls': [],
                                         'wall_seconds': 1.0, 'error': None}])
        self.assertEqual(metrics['matching'], {'bases': {'CLAUSE_UNIQUE': 1, 'NO_CANDIDATE': 1}, 'rows': 2, 'expected_rows': 2, 'required_rows': 2,
                                               'scored_rows': 1, 'scored_required': 1, 'missing_rows': 1, 'missing_required': 1,
                                               'missing_by_reason': {'EXTRACTION_MISSED': 1, 'GROUNDING_REJECTED': 0, 'MATCHING_FAILED': 0},
                                               'missing_required_by_reason': {'EXTRACTION_MISSED': 1, 'GROUNDING_REJECTED': 0, 'MATCHING_FAILED': 0},
                                               'missing_details': {'EXTRACTION_MISSED': {'NOT_RECORDED': 1}}})
        self.assertEqual((metrics['extraction']['found_required'], metrics['extraction']['expected_required']), (1, 2))
        self.assertEqual(basis_counts([{'matched': True}]), {'bases': {'NOT_RECORDED': 1}, 'rows': 1})     # a row scored before v0.19
        self.assertTrue(set(metrics['matching']['bases']) <= set(BASES))


def snapshot(folder: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(folder.rglob('*')) if p.is_file()}


@unittest.skipUnless((V018_RUN / 'results.json').is_file(), 'the v0.18 final run is not on this machine')
class LegacyRunTests(unittest.TestCase):
    def test_on_the_v018_run_only_rows_the_old_matcher_left_unmatched_change(self):
        """tr-aml-v1 on 20260924-121921-v018-final-ollama: every row the v0.18 matcher paired keeps its duty."""
        dataset = h.load_dataset(ROOT / 'evaluation' / 'datasets' / 'tr-aml-v1.json')
        stored = {r['case_id']: r for r in load_results(V018_RUN)[0]}
        kept = changed = 0
        for case in dataset.cases:
            result, packet = stored.get(case.case_id), V018_RUN / 'packets' / f'{case.case_id}.json'
            if not result or result.get('error') or not packet.is_file():
                continue
            payload = json.loads(packet.read_text(encoding='utf-8'))['events'][0]['payload']
            scored, _ = h.score_case(case, payload, h.prediction_rows(payload, case.regulation_id))
            self.assertEqual(len(scored), len(result['scored']), case.case_id)
            for old, new in zip(result['scored'], scored):
                if old['matched']:
                    self.assertEqual(new['prediction'], old['prediction'], f'{case.case_id} md.{old["expected"]["article"]}{old["expected"]["clause"]}')
                    kept += 1
                elif new['matched']:
                    changed += 1
        self.assertEqual((kept, changed), (92, 1))                                      # C28 md.42(2): the label word is in the subject

    def test_rematch_scores_both_matchers_alike_and_writes_nothing(self):
        before = snapshot(V018_RUN)
        _, stored = h.score_subset(V018_RUN, ['C28', 'C12'])
        _, again = h.score_subset(V018_RUN, ['C28', 'C12'], rematch=True)
        self.assertEqual(snapshot(V018_RUN), before)
        self.assertEqual(stored['matching']['bases'], {'NOT_RECORDED': stored['matching']['rows']})
        self.assertEqual((stored['extraction']['matched'], stored['extraction']['unexpected']), (6, 1))
        self.assertEqual((again['extraction']['matched'], again['extraction']['unexpected']), (7, 0))
        self.assertTrue(again['subset']['rematched'])


@unittest.skipUnless((INDEP_RUN / 'results.json').is_file() and INDEP.is_file(), 'the v0.19 independent run is not on this machine')
class IndependentRunTests(unittest.TestCase):
    def test_a_duty_whose_label_word_is_in_its_condition_is_scored(self):
        _, before = h.score_subset(INDEP_RUN, None, INDEP)
        _, after = h.score_subset(INDEP_RUN, None, INDEP, rematch=True)
        self.assertEqual((before['extraction']['found_required'], after['extraction']['found_required']), (12, 15))
        self.assertEqual((before['conflict']['tp'], after['conflict']['tp']), (3, 4))

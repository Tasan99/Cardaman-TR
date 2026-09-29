"""The v0.19 coverage-error analysis: which scored rows are errors, how a packet row is read, and the
automatic categories, on tiny synthetic rows and one synthetic run directory (no model, no network).
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from regchain.evaluation.cli import main
from regchain.evaluation.coverage_errors import (CATEGORIES, auto_category, category_rules, clause_text, duty_truncation, error_kinds,
                                                 evidence_view, first_judgement, label_view, load_annotations, passage_rows, settle,
                                                 verifier_view)

EVIDENCE = 'bütün tutarlar için uygulanabilir'


def passage(sid, relation='UNRELATED', status='JUDGED', holds=(), verifier=None, truncation=None, rank=1, control=False, quote=''):
    """One record shaped like coverage_errors.passage_rows output."""
    return {'source_id': sid * 8, 'status': status, 'relation': relation, 'screen': 'SCREENED_OUT' if status == 'SCREENED_OUT' else 'NO',
            'reason': f'{sid} reason', 'text': f'{sid} text', 'expected_evidence': list(holds), 'verifier': verifier, 'truncation': truncation,
            'control_row': control, 'cited': False, 'retrieval': {'rank': rank},
            'first_judgement': {'contradicts': 'YES' if verifier else 'NO', 'contradicts_quote': quote, 'support': relation, 'support_by': 'quick'}}


def row(expected, actual, passages, needles=(), in_packet=True, truncated=False, disputed=False):
    return {'expected': {'coverage': expected[0], 'conflict': expected[1]}, 'actual': {'coverage': actual[0], 'conflict': actual[1]},
            'passages': passages,
            'evidence': {'needles': [{'text': n, 'in_packet': in_packet, 'passages': []} for n in needles], 'any_in_packet': in_packet},
            'duty_truncation': {'truncated': truncated, 'duty_words': 2, 'sentence_words': 25, 'ratio': 0.08},
            'label': {'disputed': disputed, 'dispute_reasons': ['taxonomy.DISPUTED_LABELS: open'] if disputed else []}}


def primary(r):
    return category_rules(r)[0][0]


def categories(r):
    return [c for c, _ in category_rules(r)]


class ErrorSelectionTests(unittest.TestCase):
    def test_only_assessed_coverage_mismatches_and_conflict_flips_are_errors(self):
        self.assertEqual(error_kinds({'coverage': ('COVERS_TEXT', 'PARTIAL'), 'conflict': (False, False)}), ['coverage_mismatch'])
        self.assertEqual(error_kinds({'coverage': ('CONFLICT', 'NO_EVIDENCE'), 'conflict': (True, False)}),
                         ['coverage_mismatch', 'conflict_false_negative'])
        self.assertEqual(error_kinds({'coverage': ('COVERS_TEXT', 'CONFLICT'), 'conflict': (False, True)}),
                         ['coverage_mismatch', 'conflict_false_positive'])
        # NOT_ASSESSED is counted apart by the harness (coverage.not_assessed_*), not as a coverage error.
        self.assertEqual(error_kinds({'coverage': ('NOT_ASSESSED', 'UNKNOWN')}), [])
        self.assertEqual(error_kinds({'coverage': ('PARTIAL', 'PARTIAL'), 'conflict': (False, False)}), [])


class CategoryTests(unittest.TestCase):
    def test_withdrawn_contradiction_on_the_expected_passage_is_a_verifier_overrule(self):
        withdrawn = {'code': 'CONFLICT_WITHDRAWN', 'by': 'model', 'detail': 'NOT_A_CONTRADICTION: an exception', 'quote': 'x atlanır'}
        r = row(('CONFLICT', True), ('NO_EVIDENCE', False), [passage('a'), passage('b', holds=[EVIDENCE], verifier=withdrawn, rank=4)], [EVIDENCE])
        self.assertEqual(primary(r), 'VERIFIER_OVERRULE_ERROR')
        # The same row without a withdrawal is a plain miss.
        r['passages'][1]['verifier'] = None
        self.assertEqual(primary(r), 'CONFLICT_MISSED')

    def test_screened_out_evidence_is_a_prefilter_false_negative_and_absent_evidence_a_retrieval_miss(self):
        screened = row(('COVERS_TEXT', False), ('NO_EVIDENCE', False), [passage('a'), passage('b', status='SCREENED_OUT', holds=['kanıt'])], ['kanıt'])
        self.assertEqual(primary(screened), 'PREFILTER_FALSE_NEGATIVE')
        missed = row(('COVERS_TEXT', False), ('NO_EVIDENCE', False), [passage('a')], ['kanıt'])
        self.assertEqual(primary(missed), 'RETRIEVAL_MISS')
        nowhere = row(('COVERS_TEXT', False), ('NO_EVIDENCE', False), [passage('a')], ['kanıt'], in_packet=False)
        self.assertEqual(primary(nowhere), 'LABEL_AMBIGUITY')

    def test_truncated_judgement_that_left_the_row_unknown_is_context_truncation(self):
        cut = passage('a', relation='UNCLEAR', holds=['kanıt'], truncation='judge.supports: Provider output incomplete or truncated')
        r = row(('COVERS_TEXT', False), ('UNKNOWN', False), [cut, passage('b', relation='SUPPORTS', rank=2)], ['kanıt'])
        self.assertEqual(primary(r), 'CONTEXT_TRUNCATION')
        # A SUPPORTS passage outweighed by the UNCLEAR one is also recorded as an aggregation effect.
        self.assertIn('AGGREGATION_ERROR', categories(r))

    def test_false_conflict_on_a_truncated_duty_names_the_extraction_first(self):
        conflict = passage('b', relation='CONFLICTS', verifier={'code': 'CONFLICT_CONFIRMED', 'by': 'model', 'detail': '', 'quote': ''})
        r = row(('COVERS_TEXT', False), ('CONFLICT', True), [passage('a', holds=['tedbir uygulanmaz']), conflict], ['tedbir uygulanmaz'], truncated=True)
        auto = auto_category(r)
        self.assertEqual(auto['primary'], 'OBLIGATION_EXTRACTION_TRUNCATION')
        self.assertEqual(auto['secondary'][:3], ['FALSE_CONFLICT', 'SEMANTIC_MATCH_MISSED', 'IRRELEVANT_PASSAGE_SELECTED'])
        r['duty_truncation']['truncated'] = False
        self.assertEqual(primary(r), 'FALSE_CONFLICT')

    def test_partial_under_calls_and_over_calls(self):
        r = row(('COVERS_TEXT', False), ('PARTIAL', False), [passage('a', relation='PARTIAL', holds=['k'])], ['k'], disputed=True)
        auto = auto_category(r)
        self.assertEqual(auto['primary'], 'COVERS_AS_PARTIAL')
        self.assertIn('LABEL_AMBIGUITY', auto['secondary'])
        self.assertEqual(primary(row(('PARTIAL', False), ('COVERS_TEXT', False), [passage('a', relation='SUPPORTS', holds=['k'])], ['k'])),
                         'PARTIAL_AS_COVERS')
        self.assertEqual(primary(row(('PARTIAL', False), ('NO_EVIDENCE', False), [passage('a', holds=['k'])], ['k'])), 'SEMANTIC_MATCH_MISSED')
        self.assertEqual(primary(row(('NO_EVIDENCE', False), ('PARTIAL', False), [passage('a', relation='PARTIAL')])), 'IRRELEVANT_PASSAGE_SELECTED')

    def test_a_control_row_carrying_the_needed_relation_is_an_aggregation_effect(self):
        r = row(('COVERS_TEXT', False), ('NO_EVIDENCE', False), [passage('a', relation='SUPPORTS', control=True)])
        self.assertEqual(primary(r), 'AGGREGATION_ERROR')

    def test_every_category_named_by_a_rule_is_in_the_taxonomy(self):
        rows = [row(('CONFLICT', True), ('NO_EVIDENCE', False), [passage('a')]), row(('COVERS_TEXT', False), ('UNKNOWN', False), [passage('a')]),
                row(('NO_EVIDENCE', False), ('CONFLICT', True), [passage('a', relation='CONFLICTS')])]
        for r in rows:
            self.assertTrue(set(categories(r)) <= set(CATEGORIES), categories(r))
        self.assertEqual(primary(rows[1]), 'OTHER')


class ReadingTests(unittest.TestCase):
    def test_duty_truncation_compares_the_duty_with_its_clause_sentence(self):
        clause = ('(2) Yükümlüler işlem nedeniyle aklama veya terörün finansmanı riskinin oluşabileceği durumlarda, basitleştirilmiş '
                  'tedbirleri uygulayamazlar ve işlemin 27 nci maddeye göre şüpheli işlem olabileceğini göz önünde bulundururlar.')
        cut = duty_truncation({'subject': 'Yükümlüler', 'prohibited_action': 'uygulayamazlar'}, clause)
        self.assertTrue(cut['truncated'])
        self.assertLess(cut['ratio'], 0.2)
        whole = duty_truncation({'subject': 'Kimlik tespiti', 'required_action': 'iş ilişkisi tesisinden veya işlem yapılmadan önce tamamlanır.'},
                                '(2) Kimlik tespiti, iş ilişkisi tesisinden veya işlem yapılmadan önce tamamlanır.')
        self.assertFalse(whole['truncated'])
        # "T.C. kimlik" does not end a sentence; a full stop before a capital does.
        self.assertEqual(duty_truncation({'required_action': 'T.C. kimlik numarası alınır'}, 'Adı ve T.C. kimlik numarası alınır.')['sentence'],
                         'Adı ve T.C. kimlik numarası alınır.')
        self.assertEqual(duty_truncation({'required_action': 'ibraz etmek'}, 'Belgeler saklanır. Yetkililere ibraz etmek zorunludur.')['sentence'],
                         'Yetkililere ibraz etmek zorunludur.')

    def test_first_judgement_and_verifier_read_the_recorded_answers(self):
        withdrawn = {'source_id': 's', 'relation': 'UNRELATED', 'screen': 'WITHDRAWN', 'reason': 'r',
                     'notes': [{'question': 'confirm', 'code': 'CONFLICT_WITHDRAWN', 'detail': 'NOT_A_CONTRADICTION: exception', 'quote': 'atlanır'}]}
        calls = [{'task': 'judge.contradicts', 'thinking': True}, {'task': 'judge.supports', 'thinking': False}]
        judgement = first_judgement(withdrawn, calls=calls)
        self.assertEqual((judgement['contradicts'], judgement['contradicts_quote'], judgement['support'], judgement['support_by']),
                         ('YES', 'atlanır', 'UNRELATED', 'quick'))
        self.assertEqual(verifier_view(withdrawn)['by'], 'model')
        by_rule = {**withdrawn, 'notes': [{'code': 'CONFLICT_WITHDRAWN', 'detail': 'Rule: restates the prohibition', 'quote': 'q'}]}
        self.assertEqual(verifier_view(by_rule)['by'], 'rule')
        confirmed = {'relation': 'CONFLICTS', 'screen': 'YES', 'reason': 'x [Confirmed on a second reading: the duty requires "a".]'}
        self.assertEqual(verifier_view(confirmed)['code'], 'CONFLICT_CONFIRMED')
        self.assertEqual(first_judgement(confirmed, 'quoted')['contradicts_quote'], 'quoted')
        self.assertEqual(first_judgement({'screen': 'SCREENED_OUT', 'relation': 'UNRELATED'})['contradicts'], 'NOT_ASKED')

    def test_passages_and_expected_evidence_statuses_come_from_the_packet_row(self):
        payload, obligation = synthetic_packet()
        needles = [EVIDENCE, 'yok böyle metin', 'ayrı pasaj', 'sekiz yıl saklanır']
        rows = passage_rows(payload, obligation, needles)
        self.assertEqual([p['status'] for p in rows], ['JUDGED', 'SCREENED_OUT', 'FILTERED'])
        self.assertEqual(rows[0]['expected_evidence'], [EVIDENCE])
        self.assertTrue(rows[0]['truncation'] is None and rows[0]['verifier']['code'] == 'CONFLICT_WITHDRAWN')
        view = evidence_view(payload, obligation, needles, rows)
        self.assertEqual([[p['status'] for p in n['passages']] for n in view['needles']],
                         [['JUDGED:UNRELATED'], [], ['FILTERED'], ['NOT_RETRIEVED']])
        self.assertTrue(view['judged_contains'])
        self.assertEqual(clause_text(payload, obligation), '(2) Kimlik tespiti, işlem yapılmadan önce tamamlanır.')

    def test_disputed_labels_flag_ambiguity_but_run_agreement_does_not(self):
        disputed = label_view('C12', {'article': '5', 'clause': '(2)', 'notes': ''}, ['ALL_RUNS_DISAGREE (coverage): every run says X.'])
        self.assertTrue(disputed['disputed'])
        self.assertEqual(disputed['review_reasons'], ['ALL_RUNS_DISAGREE (coverage): every run says X.'])
        agreed = label_view('C99', {'article': '5', 'clause': '(2)', 'notes': ''}, ['ALL_RUNS_DISAGREE (coverage): every run says X.'])
        self.assertFalse(agreed['disputed'])


class AnnotationTests(unittest.TestCase):
    def test_manual_annotation_overrides_and_keeps_the_automatic_reading(self):
        r = row(('COVERS_TEXT', False), ('PARTIAL', False), [passage('a', relation='PARTIAL', holds=['k'])], ['k'])
        settled = settle(r, {'primary': 'LABEL_AMBIGUITY', 'secondary': ['COVERS_AS_PARTIAL'], 'root_cause': 'The policy omits items.'})
        self.assertEqual((settled['auto']['primary'], settled['final']['primary'], settled['final']['source']),
                         ('COVERS_AS_PARTIAL', 'LABEL_AMBIGUITY', 'manual'))
        self.assertEqual(settled['final']['fixable_by'], 'label')
        plain = settle(row(('CONFLICT', True), ('NO_EVIDENCE', False), [passage('a')]))
        self.assertEqual((plain['final']['source'], plain['final']['fixable_by']), ('auto', 'pipeline'))

    def test_an_unknown_category_in_the_annotations_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.json'
            path.write_text(json.dumps({'annotations': {'C01 md.5(2)': {'primary': 'MODEL_HALLUCINATION'}}}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_annotations(path)
            path.write_text(json.dumps({'C01 md.5(2)': {'primary': 'FALSE_CONFLICT'}}), encoding='utf-8')
            self.assertEqual(load_annotations(path)['C01 md.5(2)']['primary'], 'FALSE_CONFLICT')


def synthetic_packet():
    """A packet payload with one md. 5(2) duty: one judged passage (a withdrawn YES), one screened out, one filtered."""
    text = '(1) Yükümlüler kimliği tespit etmek zorundadır. (2) Kimlik tespiti, işlem yapılmadan önce tamamlanır.'
    ids = {'a': 'a' * 64, 'b': 'b' * 64, 'c': 'c' * 64}
    chunks = [{'source_id': ids['a'], 'text': f'Kimlik tespiti atlanır. Bu istisna kampanyada {EVIDENCE}.', 'locator': 'text_block'},
              {'source_id': ids['b'], 'text': 'Parolalar doksan günde değişir.', 'locator': 'text_block'},
              {'source_id': ids['c'], 'text': 'Başlık — ayrı pasaj', 'locator': 'text_block'},
              {'source_id': 'd' * 64, 'text': 'Kayıtlar sekiz yıl saklanır.', 'locator': 'text_block'}]
    obligation = {
        'id': 'o' * 64, 'source_label': 'Yönetmelik 200713012 md. 5', 'signals': [],
        'candidate': {'subject': 'Kimlik tespiti', 'modality': 'MUST', 'required_action': 'işlem yapılmadan önce tamamlanır.', 'prohibited_action': None,
                      'conditions': [], 'exceptions': [], 'deadline': None, 'source_quote': '(2) Kimlik tespiti, işlem yapılmadan önce tamamlanır.'},
        'proposal': {'applicability': 'APPLIES', 'coverage': 'NO_EVIDENCE', 'coverage_assessed': True, 'coverage_reason': 'None of the 2 passages.',
                     'applicability_rule': 'RULE_CLEAR_MATCH', 'policy_checks': [{'source_id': ids['a'], 'relation': 'UNRELATED', 'quote': 'x'},
                                                                              {'source_id': ids['b'], 'relation': 'UNRELATED', 'quote': 'y'}],
                     'policy_evidence': [], 'filtered_passages': [{'source_id': ids['c'], 'reason': 'STRUCTURAL_CRUMB'}],
                     'applicability_scope': {'child_clause': '(2)', 'child_offset': 48, 'child_excerpt': '(2) Kimlik'}, 'trace': {}},
        'retrieved_policy_ids': [ids['a'], ids['b']], 'judged_policy_ids': [ids['a'], ids['b']],
        'evidence_signals': [{'source_id': ids['a'], 'rank': 1, 'similarity': '0.6', 'lexical_score': 1, 'rrf': '0.03', 'rerank_score': None},
                             {'source_id': ids['b'], 'rank': 2, 'similarity': '0.4', 'lexical_score': 0, 'rrf': '0.01', 'rerank_score': None}],
        'diagnostics': [{'stage': 'passages', 'judged': 2, 'filtered': [], 'results': [
            {'source_id': ids['a'], 'relation': 'UNRELATED', 'screen': 'WITHDRAWN', 'reason': 'An exception, not the duty.',
             'notes': [{'question': 'confirm', 'code': 'CONFLICT_WITHDRAWN', 'detail': 'NOT_A_CONTRADICTION: an exception', 'quote': 'Kimlik tespiti atlanır.'}]},
            {'source_id': ids['b'], 'relation': 'UNRELATED', 'screen': 'SCREENED_OUT', 'reason': 'Screened out.'}]}]}
    payload = {'obligations': [obligation], 'policies': [{'filename': 'p.md', 'chunks': chunks}],
               'cases': [{'source': {'printed_label': 'Yönetmelik 200713012 md. 5', 'text': text},
                          'units': [{'offset': 0, 'chars': 47}, {'offset': 48, 'chars': len(text) - 48}]}]}
    return payload, obligation


class RunTests(unittest.TestCase):
    def test_the_cli_re_pairs_a_run_and_writes_both_reports(self):
        payload, _ = synthetic_packet()
        case = {'case_id': 'T01', 'title': 't', 'categories': [], 'jurisdiction': 'TR', 'regulation_id': 'YONETMELIK:200713012',
                'regulation_fixture': 'x', 'target_sections': ['5'], 'company_profile': {}, 'policy_documents': ['p.md'],
                'expected_obligations': [{'article': '5', 'clause': '(2)', 'action_keywords': ['önce tamamlanır'], 'applicability': 'APPLIES',
                                          'coverage': 'CONFLICT', 'conflict': True, 'evidence': [EVIDENCE]},
                                         {'article': '5', 'clause': '(1)', 'applicability': 'APPLIES', 'coverage': 'NOT_ASSESSED', 'required': False}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'datasets' / 'd.json'
            run_dir = root / 'runs' / 'r1'
            (run_dir / 'packets').mkdir(parents=True)
            dataset.parent.mkdir()
            dataset.write_text(json.dumps({'format': 'cardaman-evaluation-dataset-v1', 'dataset_id': 'd', 'version': '1', 'cases': [case]}),
                               encoding='utf-8')
            (run_dir / 'manifest.json').write_text(json.dumps({'dataset': {'id': 'd', 'sha256': 'x'}, 'mode': {'cases': ['T01']}}), encoding='utf-8')
            (run_dir / 'packets' / 'T01.json').write_text(json.dumps({'events': [{'payload': payload}]}), encoding='utf-8')
            out_json, out_md = root / 'e.json', root / 'e.md'
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(['coverage-errors', '--run', str(run_dir), '--out-json', str(out_json), '--out-md', str(out_md)])
            self.assertEqual(code, 0)
            report = json.loads(out_json.read_text(encoding='utf-8'))
            self.assertEqual(report['dataset']['found_by'], 'manifest dataset id (evaluation/datasets)')
            self.assertEqual(len(report['rows']), 1)
            only = report['rows'][0]
            self.assertEqual((only['key'], only['kinds'], only['auto']['primary']),
                             ('T01 md.5(2)', ['coverage_mismatch', 'conflict_false_negative'], 'VERIFIER_OVERRULE_ERROR'))
            self.assertIn('VERIFIER_OVERRULE_ERROR', out_md.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()

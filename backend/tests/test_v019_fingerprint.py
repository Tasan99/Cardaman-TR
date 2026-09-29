"""v0.19 problem 1 E-F: a stable obligation fingerprint, matching by the source sentence, and a reason for every missing row.

E: an expectation is located in the regulation text (the sentences of its clause that carry its words) and a duty by the
sentence it came from; after the identity pass, a label left unmatched is paired with the unclaimed candidate of its clause
that came from its sentence (SOURCE_SPAN). F: every expectation without a scored row says why (EXTRACTION_MISSED,
GROUNDING_REJECTED, MATCHING_FAILED + a detail code), counted in metrics['matching'] and shown in the reports.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from regchain.evaluation import harness as h
from regchain.evaluation import metrics as m
from regchain.evaluation.coverage_errors import analyse_run, markdown
from regchain.evaluation.identifiers import (BASES, duty_span, label_sentences, missing_reason, obligation_fingerprint, pair_predictions,
                                             source_index, unit_outcome)
from regchain.evaluation.manifest import build_manifest
from regchain.evaluation.report import missing_section, render_markdown
from regchain.evaluation.schema import FORMAT, Dataset, EvaluationCase, ExpectedObligation
from regchain.evaluation.taxonomy import load_results
from regchain.pilot.engine import analyze, split_units
from test_pilot import FixtureProvider, company, policies, sections

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'evaluation' / 'runs'
V018_RUN = RUNS / '20260924-121921-v018-final-ollama'
EVIDENCE = [(RUNS / '20260925-012420-v019t3-indep5-ollama', ROOT / 'evaluation' / 'independent' / 'independent-v1.json'),
            (RUNS / '20260925-015904-v019t3-heldout8-ollama', ROOT / 'evaluation' / 'independent' / 'independent-v1.json'),
            (RUNS / '20260925-004450-v019t3-rep8-ollama', ROOT / 'evaluation' / 'datasets' / 'tr-aml-v1.json')]

# A Turkish article in the shape of Kanun 5549 md. 4: the label word "şüphe" sits in the condition of (1), not in its action.
REPORTING = ('(1) Yükümlüler nezdinde yapılan işlemlere konu malvarlığının suçtan kaynaklandığına dair bir şüphe bulunması halinde, '
             'bu işlemlerin Başkanlığa bildirilmesi zorunludur. (2) Yükümlüler, bildirimde bulunulduğunu işlemin taraflarına açıklayamazlar.')
# One paragraph, three sentences, three acts: ask for the missing data, return the transfer, weigh ending the relationship.
TRANSFER = ('(1) Transfer mesajlarında gönderenin adı yer alır. (2) Kayıtlar saklanır. (3) Eksik bilgi içeren mesajı alan kuruluş, mesajı gönderen '
            'kuruluştan eksik bilgilerin tamamlanmasını talep eder. Bilgiler tamamlanmazsa transfer iade edilir. Eksiklik sürekli hale gelirse '
            'alıcı kuruluş tarafından iş ilişkisine son verilmesi değerlendirilir.')
# An English rule whose label word is in the subject's relative clause, which the model leaves out of the duty.
ABROAD = ('A firm dealing with a customer who is resident in a different jurisdiction must ensure that it takes appropriate account of '
          'any differences in law and court procedure. [Note: paragraph 2.3 of DCG]')


def label(article='4', clause='(1)', keywords=(), required=True, **fields):
    return ExpectedObligation(article=article, clause=clause, action_keywords=list(keywords), applicability='APPLIES', required=required, **fields)


def provision(printed_label, text, outcomes=None, reason='CANDIDATE_REQUIRES_LEGAL_REVIEW', candidates=1):
    """A packet case: its units (split as the engine splits them) and what extraction made of each.

    ``outcomes`` = {paragraph number: (kind, reason, candidates)}; a paragraph left out was never sent to a model (no modal wording).
    Without ``outcomes`` the provision was read whole and records ``reason`` on the case itself.
    """
    units = split_units(text)
    number = lambda unit: unit[1:unit.index(')')] if unit.startswith('(') else ''
    classification = [{'offset': offset, 'chars': len(unit), 'kind': (outcomes or {}).get(number(unit), ('OTHER',))[0]} for offset, unit in units]
    case = {'source': {'printed_label': printed_label, 'text': text}, 'classification': classification, 'reason': reason,
            'output': {'obligations': [{}] * candidates}}
    if outcomes is not None:
        case['units'] = [{'offset': offset, 'chars': len(unit), 'kind': outcomes[number(unit)][0], 'reason': outcomes[number(unit)][1],
                          'candidates': outcomes[number(unit)][2]} for offset, unit in units if number(unit) in outcomes]
    return case


def duty(payload, printed_label, clause, action, sentence=None, key=None, modality='MUST', conditions=(), subject='Yükümlüler', structure=True):
    """A prediction row as harness.prediction_rows writes it, located in the packet's provision text.

    ``sentence`` is the text the structure-v1 source span covers (a v0.19 row); structure=False leaves it out (a v0.18 row,
    located by its action).
    """
    case = next(c for c in payload['cases'] if c['source']['printed_label'] == printed_label)
    text = case['source']['text']
    unit = next(((offset, u) for offset, u in split_units(text) if u.startswith(f'({clause})')), (0, text))
    row = {'candidate': {'required_action': action}, 'multipart': {'unit_offset': unit[0]}}
    if structure:
        start = unit[1].index(sentence)
        row['structure'] = {'unit_offset': unit[0], 'source_span': {'text': sentence, 'start': start, 'end': start + len(sentence)}}
    article = printed_label.split(' md. ')[-1] if ' md. ' in printed_label else printed_label.split(' ', 1)[1]
    sources = source_index(payload)
    span = duty_span(row, sources.get(article))
    return {'key': key or f'{article}/{clause}/{action[:20]}', 'obligation_id': key or action[:20], 'article': article, 'clause': clause,
            'subject': subject, 'action': action, 'modality': modality, 'conditions': list(conditions), 'exceptions': [], 'deadline': None,
            'source_span': span, 'fingerprint': obligation_fingerprint('KANUN:5549', article, clause, action, None,
                                                                       text[span['start']:span['end']] if span else None)}


def reporting_packet():
    return {'cases': [provision('Kanun 5549 md. 4', REPORTING, {'1': ('OBLIGATION', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 1),
                                                                 '2': ('PROHIBITION', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 1)})]}


def transfer_packet():
    return {'cases': [provision('Yönetmelik 200713012 md. 24', TRANSFER, {'3': ('OBLIGATION', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 2),
                                                                           '2': ('OBLIGATION', 'GROUNDING_REJECTED', 0)})]}


def no_floats(value) -> bool:
    if isinstance(value, float):
        return False
    if isinstance(value, dict):
        return all(no_floats(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(no_floats(v) for v in value)
    return True


class SourceSpanMatchingTests(unittest.TestCase):
    def test_a_label_word_in_the_condition_matches_by_the_duty_or_by_its_sentence(self):
        payload = reporting_packet()
        sentence = REPORTING[4:REPORTING.index(' (2)')]
        # The model kept the condition: the duty's own words carry the label (FIX 4, unchanged).
        kept = duty(payload, 'Kanun 5549 md. 4', '1', 'bu işlemlerin Başkanlığa bildirilmesi', sentence, conditions=['bir şüphe bulunması halinde'])
        (match,), _ = pair_predictions([label(keywords=['şüphe'])], [kept], source_index(payload))
        self.assertEqual((match['prediction'], match['basis']), (kept, 'CLAUSE_UNIQUE'))
        # The model dropped it: the duty still came from the sentence that holds the label's word.
        dropped = duty(payload, 'Kanun 5549 md. 4', '1', 'bu işlemlerin Başkanlığa bildirilmesi', sentence)
        (match,), extras = pair_predictions([label(keywords=['şüphe'])], [dropped], source_index(payload))
        self.assertEqual((match['prediction'], match['basis'], match['label_sentences'], match['from_sentence']), (dropped, 'SOURCE_SPAN', 1, 1))
        self.assertEqual((match['missing_reason'], extras), (None, []))
        # Without the provision text nothing can be told, as before: AMBIGUOUS and unmatched.
        (match,), _ = pair_predictions([label(keywords=['şüphe'])], [dropped])
        self.assertEqual((match['prediction'], match['basis']), (None, 'AMBIGUOUS'))
        # A duty of the other paragraph never pairs by sentence: the clause decides the candidates first (here the model
        # returned nothing for (1)).
        payload = {'cases': [provision('Kanun 5549 md. 4', REPORTING, {'1': ('OBLIGATION', 'NO_EXPLICIT_OBLIGATION', 0),
                                                                        '2': ('PROHIBITION', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 1)})]}
        other = duty(payload, 'Kanun 5549 md. 4', '2', 'bildirimde bulunulduğunu işlemin taraflarına açıklamamak',
                     REPORTING[REPORTING.index('(2)') + 4:], modality='MUST_NOT')
        (match,), _ = pair_predictions([label(keywords=['şüphe'])], [other], source_index(payload))
        self.assertEqual((match['basis'], match['missing_reason'], match['missing_detail']), ('NO_CANDIDATE', 'EXTRACTION_MISSED', 'MODEL_RETURNED_NONE'))

    def test_a_paraphrased_action_pairs_by_its_source_sentence(self):
        payload = {'cases': [provision('CONC 7.8.1', ABROAD)]}
        sentence = ABROAD[:ABROAD.index(' [Note')]
        # v0.19 row: structure-v1 recorded the sentence; the model's action is its own paraphrase.
        paraphrased = duty(payload, 'CONC 7.8.1', '', 'take account of legal differences', sentence, subject='A firm')
        (match,), _ = pair_predictions([label(article='7.8.1', clause='', keywords=['jurisdiction'])], [paraphrased], source_index(payload))
        self.assertEqual((match['prediction'], match['basis']), (paraphrased, 'SOURCE_SPAN'))
        self.assertEqual(paraphrased['source_span']['origin'], 'STRUCTURE')
        # v0.18 row (no structure): the action is found in the provision text and its sentence is the span.
        located = duty(payload, 'CONC 7.8.1', '', 'ensure that it takes appropriate account of any differences in law', structure=False, subject='A firm')
        self.assertEqual(located['source_span'], {'start': 0, 'end': len(sentence), 'origin': 'ACTION'})
        (match,), _ = pair_predictions([label(article='7.8.1', clause='', keywords=['jurisdiction'])], [located], source_index(payload))
        self.assertEqual(match['basis'], 'SOURCE_SPAN')
        # A label whose words are in no sentence of the clause is not paired by the sentence rule ("destroy records").
        (match,), _ = pair_predictions([label(article='7.8.1', clause='', keywords=['destroy records'])], [located], source_index(payload))
        self.assertEqual((match['prediction'], match['basis'], match['missing_reason'], match['missing_detail']),
                         (None, 'OTHER_SENTENCE', 'MATCHING_FAILED', 'LABEL_WORDS_NOT_IN_CLAUSE'))

    def test_sibling_duties_of_one_clause_are_separated_by_their_source_sentence(self):
        payload = transfer_packet()
        first, _, third = [s + '.' for s in TRANSFER[TRANSFER.index('(3)') + 4:].rstrip('.').split('. ')]
        ask = duty(payload, 'Yönetmelik 200713012 md. 24', '3', 'eksik bilgilerin tamamlanmasını talep eder', first, key='ask', subject='kuruluş')
        end = duty(payload, 'Yönetmelik 200713012 md. 24', '3', 'iş ilişkisine son verilmesi değerlendirilir', third, key='end', subject='alıcı kuruluş')
        sources = source_index(payload)
        matches, extras = pair_predictions([label(article='24', clause='(3)', keywords=['gönderen']), label(article='24', clause='(3)', keywords=['iade']),
                                            label(article='24', clause='(3)', keywords=['sürekli'])], [ask, end], sources)
        self.assertEqual([(m['prediction'] or {}).get('key') for m in matches], ['ask', None, 'end'])
        self.assertEqual([m['basis'] for m in matches], ['SOURCE_SPAN', 'OTHER_SENTENCE', 'SOURCE_SPAN'])
        self.assertEqual((matches[0]['candidates'], matches[0]['from_sentence'], matches[2]['candidates']), (2, 1, 1))
        # The "iade" duty (the second sentence) was never extracted: the clause's duties come from other sentences.
        self.assertEqual((matches[1]['missing_reason'], matches[1]['missing_detail']), ('MATCHING_FAILED', 'LABEL_SENTENCE_NOT_EXTRACTED'))
        self.assertEqual(extras, [])
        # Two duties cut from the same sentence: AMBIGUOUS unless something of their own separates them (here the polarity).
        twin = duty(payload, 'Yönetmelik 200713012 md. 24', '3', 'eksik bilgileri istemek', first, key='twin', subject='kuruluş')
        (match,), _ = pair_predictions([label(article='24', clause='(3)', keywords=['gönderen'])], [ask, twin], sources)
        self.assertEqual((match['prediction'], match['basis'], match['from_sentence']), (None, 'AMBIGUOUS', 2))
        self.assertEqual((match['missing_reason'], match['missing_detail']), ('MATCHING_FAILED', 'AMBIGUOUS'))
        refuse = {**twin, 'modality': 'MUST_NOT', 'key': 'refuse'}
        (match,), _ = pair_predictions([label(article='24', clause='(3)', keywords=['göndermez'])], [ask, refuse], sources)
        self.assertEqual((match['prediction']['key'], match['basis']), ('refuse', 'SOURCE_SPAN'))
        # The fingerprint names the sentence too: two duties worded alike from different sentences keep different fingerprints.
        alike = duty(payload, 'Yönetmelik 200713012 md. 24', '3', 'eksik bilgilerin tamamlanmasını talep eder', third, key='alike')
        self.assertNotEqual(ask['fingerprint'], alike['fingerprint'])
        self.assertEqual(ask['fingerprint'].rsplit('/', 2)[0], alike['fingerprint'].rsplit('/', 2)[0])

    def test_the_sentence_pass_never_changes_a_pair_the_identity_pass_made(self):
        payload = transfer_packet()
        first = TRANSFER[TRANSFER.index('(3)') + 4:].split('. ')[0] + '.'
        # The second label's word is in the first duty's own condition: pass 1 gives it that duty although the
        # first label (earlier in dataset order) shares its sentence; the first label then has nothing left.
        own = duty(payload, 'Yönetmelik 200713012 md. 24', '3', 'eksik bilgilerin tamamlanmasını talep eder', first, key='own',
                   conditions=['eksik bilgi içeren mesajı alması halinde'])
        expected = [label(article='24', clause='(3)', keywords=['gönderen']), label(article='24', clause='(3)', keywords=['eksik'])]
        without, _ = pair_predictions(expected, [own])
        with_sources, _ = pair_predictions(expected, [own], source_index(payload))
        self.assertEqual([m['basis'] for m in without], ['AMBIGUOUS', 'CLAUSE_UNIQUE'])
        self.assertEqual([m['basis'] for m in with_sources], ['NO_CANDIDATE', 'CLAUSE_UNIQUE'])
        self.assertIs(with_sources[1]['prediction'], own)
        self.assertEqual((with_sources[0]['missing_reason'], with_sources[0]['missing_detail']), ('MATCHING_FAILED', 'CLAIMED_BY_ANOTHER_LABEL'))

    def test_the_fingerprint_is_stable_and_names_regulation_provision_clause_stems_and_sentence(self):
        one = obligation_fingerprint('KANUN:5549', '4', '(1)', 'Bu işlemlerin Başkanlığa bildirilmesi.', None, 'Şüphe halinde bildirilmesi zorunludur.')
        self.assertEqual(one, obligation_fingerprint('KANUN:5549', '4', '1', 'başkanlığa  bu işlemlerin bildirilmesi', None,
                                                     'şüphe halinde   bildirilmesi zorunludur'))
        self.assertTrue(one.startswith('KANUN:5549/md.4/1/'))
        self.assertEqual(len(one.split('/')), 5)
        self.assertNotEqual(one, obligation_fingerprint('KANUN:5549', '4', '2', 'Bu işlemlerin Başkanlığa bildirilmesi.', None,
                                                        'Şüphe halinde bildirilmesi zorunludur.'))
        self.assertTrue(obligation_fingerprint('CONC:7', '7.8.1', '', 'ensure').endswith('/nospan'))


class MissingReasonTests(unittest.TestCase):
    ARTICLE = ('(1) Kuruluşlar müşterinin kimliğini teyit edebilir. (2) Müşteri kimliği sözleşmede yer alır. (3) Defterler denetime hazır '
               'bulundurulur. (4) Bilgiler talep üzerine verilir. (5) Kayıtlar sekiz yıl saklanır. (6) Bu madde yatırım fonlarına uygulanmaz.')

    def packet(self):
        return {'cases': [provision('Yönetmelik 200713012 md. 21', self.ARTICLE,
                                    {'1': ('PERMISSION', 'CLASSIFIED_PERMISSION', 0), '3': ('OBLIGATION', 'GROUNDING_REJECTED', 0),
                                     '4': ('OBLIGATION', 'NO_EXPLICIT_OBLIGATION', 0), '5': ('OBLIGATION', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 1),
                                     '6': ('EXEMPTION', 'CLASSIFIED_EXEMPTION', 0)}),
                          provision('Kanun 5549 md. 15', 'Bu Kanunun uygulanmasında gerçekleştirilen işlemler kayda alınır.', reason='GROUNDING_REJECTED',
                                    candidates=0)]}

    def test_every_missing_row_says_why_extraction_grounding_or_matching(self):
        payload = self.packet()
        kept = duty(payload, 'Yönetmelik 200713012 md. 21', '5', 'Kayıtlar sekiz yıl saklanır', 'Kayıtlar sekiz yıl saklanır.', key='keep')
        labels = [label(article='21', clause='(1)', keywords=['teyit']), label(article='21', clause='(2)', keywords=['sözleşme']),
                  label(article='21', clause='(3)', keywords=['defter']), label(article='21', clause='(4)', keywords=['talep']),
                  label(article='21', clause='(5)', keywords=['sekiz']), label(article='21', clause='(5)', keywords=['kayıt']),
                  label(article='21', clause='(6)', keywords=['fon']), label(article='15', clause='', keywords=['kayda']),
                  label(article='16', clause='', keywords=['bildir'])]
        matches, _ = pair_predictions(labels, [kept], source_index(payload))
        self.assertEqual([(m['missing_reason'], m['missing_detail']) for m in matches], [
            ('EXTRACTION_MISSED', 'CLASSIFIED_PERMISSION'),        # the unit was classified as a permission: no model read it
            ('EXTRACTION_MISSED', 'NO_MODAL'),                     # no modal wording: never sent to a model
            ('GROUNDING_REJECTED', 'GROUNDING_REJECTED'),          # the model proposed a duty; the evidence gate refused it
            ('EXTRACTION_MISSED', 'MODEL_RETURNED_NONE'),          # the model answered with no obligation
            (None, None),                                          # scored (CLAUSE_UNIQUE)
            ('MATCHING_FAILED', 'CLAIMED_BY_ANOTHER_LABEL'),       # the clause's one duty went to the label before it
            ('EXTRACTION_MISSED', 'CLASSIFIED_EXEMPTION'),
            ('GROUNDING_REJECTED', 'GROUNDING_REJECTED'),          # a provision read whole, rejected whole (reason on the case)
            ('EXTRACTION_MISSED', 'NOT_ANALYSED')])                # the provision is not in the packet
        self.assertEqual(matches[4]['basis'], 'CLAUSE_UNIQUE')
        # The remaining unit outcomes: a model call that failed, the model's own INSUFFICIENT_EVIDENCE, a candidate under another number.
        unit = {'worked': True, 'candidates': 0}
        self.assertEqual(unit_outcome({**unit, 'reason': 'PROVIDER_FAILURE'}), ('EXTRACTION_MISSED', 'PROVIDER_FAILURE'))
        self.assertEqual(unit_outcome({**unit, 'reason': 'INSUFFICIENT_EVIDENCE'}), ('GROUNDING_REJECTED', 'INSUFFICIENT_EVIDENCE'))
        self.assertEqual(unit_outcome({**unit, 'reason': 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 'candidates': 1}), ('MATCHING_FAILED', 'CLAUSE_MISMATCH'))
        misnumbered = {**kept, 'clause': '4', 'key': 'misnumbered'}
        (match,), _ = pair_predictions([labels[4]], [misnumbered], source_index(payload))
        self.assertEqual((match['missing_reason'], match['missing_detail']), ('MATCHING_FAILED', 'CLAUSE_MISMATCH'))
        self.assertEqual(missing_reason({**match, 'basis': 'NO_CANDIDATE'}, [], None, set()), ('EXTRACTION_MISSED', 'NOT_RECORDED'))

    def test_the_counts_the_report_and_the_error_report_show_the_reasons(self):
        payload = self.packet()
        kept = duty(payload, 'Yönetmelik 200713012 md. 21', '5', 'Kayıtlar sekiz yıl saklanır', 'Kayıtlar sekiz yıl saklanır.', key='keep')
        scored = [{'expected': label(article='21', clause='(5)', keywords=['sekiz']).model_dump(), 'matched': True, 'match_basis': 'CLAUSE_UNIQUE'},
                  {'expected': label(article='21', clause='(3)', keywords=['defter']).model_dump(), 'matched': False,
                   'missing_reason': 'GROUNDING_REJECTED', 'missing_detail': 'GROUNDING_REJECTED'},
                  {'expected': label(article='21', clause='(2)', keywords=['sözleşme'], required=False).model_dump(), 'matched': False,
                   'missing_reason': 'EXTRACTION_MISSED', 'missing_detail': 'NO_MODAL'},
                  {'expected': label(article='21', clause='(4)', keywords=['talep']).model_dump(), 'matched': False}]      # scored before v0.19
        counts = m.missing_rows(scored)
        self.assertEqual({k: counts[k] for k in ('expected_rows', 'required_rows', 'scored_rows', 'scored_required', 'missing_rows', 'missing_required')},
                         {'expected_rows': 4, 'required_rows': 3, 'scored_rows': 1, 'scored_required': 1, 'missing_rows': 3, 'missing_required': 2})
        self.assertEqual(counts['missing_by_reason'], {'EXTRACTION_MISSED': 1, 'GROUNDING_REJECTED': 1, 'MATCHING_FAILED': 0, 'NOT_RECORDED': 1})
        self.assertEqual(counts['missing_required_by_reason'], {'EXTRACTION_MISSED': 0, 'GROUNDING_REJECTED': 1, 'MATCHING_FAILED': 0, 'NOT_RECORDED': 1})
        self.assertEqual(counts['missing_details'], {'EXTRACTION_MISSED': {'NO_MODAL': 1}, 'GROUNDING_REJECTED': {'GROUNDING_REJECTED': 1},
                                                     'NOT_RECORDED': {'NOT_RECORDED': 1}})
        section = '\n'.join(missing_section({'matching': counts}, [{'case_id': 'M01', 'scored': scored}]))
        self.assertIn('## Missing rows', section)
        self.assertIn('| M01 | md.21(3) | defter | yes | GROUNDING_REJECTED | GROUNDING_REJECTED |', section)
        self.assertEqual(missing_section({'matching': counts}, [{'case_id': 'M01', 'scored': scored[:1]}]), [])
        self.assertTrue(no_floats(counts) and no_floats(kept))

    def test_a_run_scored_end_to_end_names_its_missing_rows_everywhere(self):
        """analyze() on the CONC fixture, a run folder, score_subset, the report and the coverage-error report."""
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        payload = json.loads(json.dumps(packet['events'][0]['payload']))
        case = EvaluationCase(case_id='F01', title='fixture', categories=[], jurisdiction='UK', regulation_id='CONC:7', regulation_fixture='.',
                              target_sections=['7.3'], company_profile=company().model_dump(), policy_documents=['policy.txt'],
                              expected_obligations=[ExpectedObligation(article='7.3.4', action_keywords=['retain'], applicability='APPLIES',
                                                                       coverage='COVERS_TEXT'),
                                                    ExpectedObligation(article='7.3.4', action_keywords=['destroy records'], applicability='APPLIES',
                                                                       coverage='NO_EVIDENCE'),
                                                    ExpectedObligation(article='7.9.9', action_keywords=['notify'], applicability='APPLIES',
                                                                       coverage='NO_EVIDENCE', required=False)])
        predictions = h.prediction_rows(payload, case.regulation_id)
        self.assertEqual(predictions[0]['source_span'], {'start': 0, 'end': len('A firm must retain records.'), 'origin': 'STRUCTURE'})
        self.assertTrue(predictions[0]['fingerprint'].startswith('CONC:7/md.7.3.4/*/'))
        scored, extras = h.score_case(case, payload, predictions)
        self.assertEqual([(i['match_basis'], i.get('missing_reason'), i.get('missing_detail')) for i in scored],
                         [('CLAUSE_UNIQUE', None, None), ('NO_CANDIDATE', 'MATCHING_FAILED', 'CLAIMED_BY_ANOTHER_LABEL'),
                          ('NO_CANDIDATE', 'EXTRACTION_MISSED', 'NOT_ANALYSED')])
        self.assertTrue(no_floats(scored) and no_floats(predictions[0]['source_span']))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / 'datasets' / 'f.json'
            dataset_path.parent.mkdir()
            dataset_path.write_text(Dataset(format=FORMAT, dataset_id='f', version='1', cases=[case]).model_dump_json(), encoding='utf-8')
            dataset = h.load_dataset(dataset_path)
            run_dir = root / 'runs' / 'r'
            (run_dir / 'packets').mkdir(parents=True)
            (run_dir / 'packets' / 'F01.json').write_text(json.dumps(packet, ensure_ascii=False), encoding='utf-8')
            result = {'case_id': 'F01', 'title': 'fixture', 'categories': [], 'error': None, 'predictions': predictions, 'scored': scored,
                      'extras': extras, 'extraction': [], 'timings': {}, 'wall_seconds': 1.0}
            (run_dir / 'results.json').write_text(json.dumps([result], ensure_ascii=False), encoding='utf-8')
            manifest = build_manifest(dataset, {'provider': 'rules', 'retrieval': 'lexical', 'reranker': False, 'reranker_status': {'status': 'off'},
                                                'cases': ['F01'], 'overrides': {}}, dataset_path=dataset_path)
            (run_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
            _, metrics = h.score_subset(run_dir, None, dataset_path, rematch=True)
            report = render_markdown(manifest, metrics, [{**result, 'calls': []}])
            errors = analyse_run(run_dir, dataset_path)
        self.assertEqual({k: metrics['matching'][k] for k in ('required_rows', 'scored_rows', 'missing_rows', 'missing_by_reason')},
                         {'required_rows': 2, 'scored_rows': 1, 'missing_rows': 2,
                          'missing_by_reason': {'EXTRACTION_MISSED': 1, 'GROUNDING_REJECTED': 0, 'MATCHING_FAILED': 1}})
        self.assertIn('| Label rows: required / scored / missing (required missing) | 2 / 1 / 2 (1) of 3 expected |', report)
        self.assertIn('| Missing rows: extraction missed / grounding rejected / matching failed | 1 (0) / 0 (0) / 1 (1) (required in brackets) |', report)
        self.assertIn('**missing: MATCHING_FAILED (CLAIMED_BY_ANOTHER_LABEL)**', report)
        self.assertIn('| F01 | md.7.9.9 | notify | no | EXTRACTION_MISSED | NOT_ANALYSED | NO_CANDIDATE |', report)
        extraction_misses = [x for x in metrics['mismatches'] if x['kind'] == 'extraction_miss']
        self.assertEqual([(x['missing_reason'], x['missing_detail']) for x in extraction_misses], [('MATCHING_FAILED', 'CLAIMED_BY_ANOTHER_LABEL')])
        self.assertEqual([(u['missing_reason'], u['missing_detail']) for u in errors['unscored']],
                         [('MATCHING_FAILED', 'CLAIMED_BY_ANOTHER_LABEL'), ('EXTRACTION_MISSED', 'NOT_ANALYSED')])
        self.assertEqual(errors['unscored_by_reason'], {'MATCHING_FAILED': 1, 'EXTRACTION_MISSED': 1})
        self.assertIn('Eksik nedeni: MATCHING_FAILED 1, EXTRACTION_MISSED 1', markdown(errors))
        self.assertTrue(set(i['match_basis'] for i in scored) <= set(BASES))


def rows_of(run_dir, dataset_path):
    dataset = h.load_dataset(dataset_path)
    stored = {r['case_id']: r for r in load_results(run_dir)[0]}
    for case in dataset.cases:
        packet = run_dir / 'packets' / f'{case.case_id}.json'
        if case.case_id in stored and not stored[case.case_id].get('error') and packet.is_file():
            payload = json.loads(packet.read_text(encoding='utf-8'))['events'][0]['payload']
            yield case, stored[case.case_id], payload, h.prediction_rows(payload, case.regulation_id)


@unittest.skipUnless(all((run / 'results.json').is_file() for run, _ in EVIDENCE) and (V018_RUN / 'results.json').is_file(),
                     'the evidence runs are not on this machine')
class EvidenceRunTests(unittest.TestCase):
    def test_no_previously_matched_row_changes_its_prediction(self):
        """v0.18 final and the three t3 evidence runs: every row the identity pass pairs, and every row the run itself scored
        as matched, keeps its prediction; only rows unmatched before gain one: I08 md.31(1) and I12 CONC 7.8.1 by their
        source sentence, and C28 md.42(2), which the v0.18 run's own (keyword) matcher left unmatched and FIX 4 pairs."""
        new = []
        for run_dir, dataset_path in [*EVIDENCE, (V018_RUN, ROOT / 'evaluation' / 'datasets' / 'tr-aml-v1.json')]:
            for case, stored, payload, predictions in rows_of(run_dir, dataset_path):
                before, _ = pair_predictions(case.expected_obligations, predictions)
                after, _ = h.pair_case(case.expected_obligations, predictions, payload)
                scored, _ = h.score_case(case, payload, predictions)
                for old, now, run_row, item in zip(before, after, stored['scored'], scored):
                    where = f'{case.case_id} md.{old["expected"].article}{old["expected"].clause}'
                    if old['prediction'] is not None:
                        self.assertIs(now['prediction'], old['prediction'], where)
                    if run_row['matched']:
                        self.assertEqual(item['prediction'], run_row['prediction'], where)
                    elif item['matched']:
                        new.append((case.case_id.split('-')[0], f'md.{old["expected"].article}{old["expected"].clause}', item['match_basis']))
                    self.assertTrue(no_floats(item), where)
        self.assertEqual(sorted(new), [('C28', 'md.42(2)', 'CLAUSE_UNIQUE'), ('I08', 'md.31(1)', 'SOURCE_SPAN'), ('I12', 'md.7.8.1', 'SOURCE_SPAN')])

    def test_the_missing_rows_of_the_independent_runs_by_reason(self):
        reasons = {}
        for run_dir, dataset_path in EVIDENCE[:2]:
            for case, _, payload, predictions in rows_of(run_dir, dataset_path):
                for item in h.score_case(case, payload, predictions)[0]:
                    if not item['matched']:
                        reasons[(case.case_id.split('-')[0], item['expected']['article'], item['expected']['clause'])] = (
                            item['missing_reason'], item['missing_detail'])
        self.assertEqual(sum(1 for r, _ in reasons.values() if r == 'EXTRACTION_MISSED'), 10)
        self.assertEqual(reasons[('I08', '15', '')], ('GROUNDING_REJECTED', 'GROUNDING_REJECTED'))
        self.assertEqual(reasons[('I08', '31', '(3)')], ('GROUNDING_REJECTED', 'GROUNDING_REJECTED'))
        # md. 24/A(3) "iade": the clause's one duty is the third sentence's; the labelled second sentence was never extracted.
        self.assertEqual(reasons[('I06', '24/A', '(3)')], ('MATCHING_FAILED', 'LABEL_SENTENCE_NOT_EXTRACTED'))
        self.assertEqual(reasons[('I13', '21', '(5)')], ('EXTRACTION_MISSED', 'CLASSIFIED_EXEMPTION'))
        self.assertEqual(reasons[('I09', '22', '(1)')], ('EXTRACTION_MISSED', 'NO_MODAL'))


if __name__ == '__main__':
    unittest.main()

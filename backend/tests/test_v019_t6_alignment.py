"""v0.19 t6 (P3): every duty sentence of a unit gets a duty of its own, and a duty is never another sentence's.

Tedbirler Yönetmeliği md. 24/A(3) reads "... eksik olan bilgilerin tamamlanmasını talep eder. Bilgilerin tamamlanmaması
halinde söz konusu kripto varlık transferini iade eder. ... hususları göz önünde bulundurulur." In the v019t3, v019t4
and v019t5 runs the model gave a candidate for the third sentence only. The gate accepted it (the first two sentences
end in v0.19 soft markers, which verify never demands), and the labelled "iade" duty was scored OTHER_SENTENCE:
no candidate came from its sentence. spans.cover_sentences now builds each missing duty from its own sentence after
the answer is accepted. It cuts an action that runs over a full stop back to the sentence of its marker, and it
records a subject or qualifier taken from another sentence. identifiers.duty_span names a duty by the sentence of
its marker.
"""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction import pipeline
from regchain.extraction.contract import materialize
from regchain.extraction.grounding import verify
from regchain.extraction.pipeline import extract
from regchain.extraction.providers import RulesProvider
from regchain.extraction.retrieval import ContextPacket
from regchain.extraction.schema import ExtractionOutput
from regchain.extraction.spans import (COVERED_CODE, OTHER_SENTENCE, SOURCE_BOUND_OK, UNCOVERED_CODE, cover_sentences, sentence_subject,
                                       unit_sentences)
from regchain.extraction.structure import structure_of
from regchain.evaluation import harness as h
from regchain.evaluation.identifiers import UNIT_NUMBER, article_of, duty_span, pair_predictions, source_index
from regchain.pilot.engine import split_units

from test_v019_extraction import Scripted, action_of, answer, candidate, unit
from test_v019_fingerprint import TRANSFER, label, provision

RECEIVER = 'alıcı kripto varlık hizmet sağlayıcı'
# The recorded v019t5 answer for md. 24/A(3): the third sentence's duty, stopped short of its marker, no condition.
SHORT = ('gönderici kripto varlık hizmet sağlayıcıdan gelen transferlerin reddedilmesi veya söz konusu kripto varlık hizmet '
         'sağlayıcıyla yapılan işlemlerin sınırlandırılması ya da iş ilişkisine son verilmesi')
FULL = SHORT + ' hususları göz önünde bulundurulur'
PERSISTENT = ('Gönderilen mesajların sürekli olarak eksik bilgiler içermesi ve bu bilgilerin talep edildiği halde tamamlanmaması '
              'halinde')
RECEIVING = 'Birinci ve ikinci fıkrada belirtilen bilgileri içermeyen kripto varlık transfer mesajı alan kripto varlık hizmet sağlayıcı'
REQUEST = 'bu mesajı gönderen kripto varlık hizmet sağlayıcıdan eksik olan bilgilerin tamamlanmasını talep eder'
RETURN = 'söz konusu kripto varlık transferini iade eder'
# The same shape in other words: act, act on a failure, weigh ending the relationship.
REPORTS = ('(2) Yükümlüler, şüpheli işlemleri gecikmeksizin Başkanlığa bildirir. Bildirimin yapılamaması halinde işlemi askıya alır. '
           'İşlemin tekrarlanması halinde, yükümlüler tarafından; müşteriyle iş ilişkisinin sona erdirilmesi hususu göz önünde bulundurulur.')
RUNS = Path(__file__).resolve().parents[2] / 'evaluation' / 'runs'
MICRO_RUNS = [RUNS / '20260925-153620-v019t5-micro-indep-ollama', RUNS / '20260925-162639-v019t5-micro-tr-ollama']


def records(result, code):
    return [d for d in result.diagnostics if d.get('code') == code]


def trace_of(result):
    return next(d['grounding'] for d in reversed(result.diagnostics) if d.get('stage') == 'complete')


def sentence_of(text, value):
    """The source sentence structure-v1 records for a duty (what the harness locates it by)."""
    span = structure_of(text, value.model_dump(mode='json'))['source_span']
    return text[span['start']:span['end']]


def sentences(text):
    return [text[a:b].strip() for a, b in unit_sentences(text)]


class Checks(unittest.TestCase):
    def clean(self, text, result):
        """Every stored field is a source substring, the answer passes the gate, and the record is digest-safe."""
        self.assertEqual(result.reason, 'CANDIDATE_REQUIRES_LEGAL_REVIEW')
        for value in result.output.obligations:
            for field in [value.subject, action_of(value), *value.conditions, *value.exceptions]:
                self.assertIn(field, text)
        verify(text, result.output)
        canonical_bytes(list(result.diagnostics))


class SentenceCoverageTests(Checks):
    def test_a_three_sentence_sub_paragraph_gives_one_duty_per_duty_sentence(self):
        _, text = unit('24/A', 3)
        first, second, third = sentences(text)
        result = extract(text, Scripted(answer(candidate(text, RECEIVER, 'MUST', SHORT))))
        self.clean(text, result)
        model, request, give_back = result.output.obligations
        # The model's duty comes first and is exactly what the t5 ladder made of it.
        self.assertEqual((model.subject, model.required_action, model.conditions), (RECEIVER, FULL, [PERSISTENT]))
        self.assertEqual([r['steps'] for r in records(result, 'GROUNDING_SALVAGED')], [['action', 'qualifier_source']])
        # One duty per sentence, each located in its own sentence.
        self.assertEqual([sentence_of(text, v) for v in (model, request, give_back)],
                         [third, first[len('(3) '):], second])
        self.assertEqual((request.subject, request.required_action, request.conditions), (RECEIVING, REQUEST, []))
        # The second sentence has no subject of its own: it carries the first one's over, and its own condition.
        self.assertEqual((give_back.subject, give_back.required_action, give_back.conditions),
                         (RECEIVING, RETURN, ['Bilgilerin tamamlanmaması halinde']))
        [record] = records(result, COVERED_CODE)
        self.assertEqual([(f['sentence'], f['field'], f['source']) for f in record['fields']],
                         [(0, 'candidate', REQUEST), (1, 'candidate', RETURN)])
        self.assertEqual(trace_of(result), [
            {'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'structured'},
            {'candidate': 1, 'code': SOURCE_BOUND_OK, 'tier': 'structured', 'origin': 'sentence', 'sentence': 0},
            {'candidate': 2, 'code': SOURCE_BOUND_OK, 'tier': 'structured', 'origin': 'sentence', 'sentence': 1, 'cross_sentence': ['subject']}])

    def test_the_same_holds_for_any_paragraph_of_that_shape(self):
        first, second, third = sentences(REPORTS)
        model = candidate(REPORTS, 'yükümlüler', 'MUST', 'müşteriyle iş ilişkisinin sona erdirilmesi hususu göz önünde bulundurulur',
                          conditions=['İşlemin tekrarlanması halinde'])
        result = extract(REPORTS, Scripted(answer(model)))
        self.clean(REPORTS, result)
        self.assertEqual([(v.subject, action_of(v), v.conditions) for v in result.output.obligations[1:]],
                         [('Yükümlüler', 'şüpheli işlemleri gecikmeksizin Başkanlığa bildirir', []),
                          ('Yükümlüler', 'işlemi askıya alır', ['Bildirimin yapılamaması halinde'])])
        self.assertEqual([sentence_of(REPORTS, v) for v in result.output.obligations], [third, first[len('(2) '):], second])

    def test_a_condition_from_the_previous_sentence_stays_attached(self):
        _, text = unit('24/A', 3)
        model = candidate(text, RECEIVER, 'MUST', FULL, conditions=['Bilgilerin tamamlanmaması halinde', PERSISTENT])
        result = extract(text, Scripted(answer(model)))
        self.clean(text, result)
        kept = result.output.obligations[0]
        self.assertEqual((kept.required_action, kept.conditions), (FULL, ['Bilgilerin tamamlanmaması halinde', PERSISTENT]))
        self.assertEqual(trace_of(result)[0], {'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'exact', 'cross_sentence': ['conditions']})
        # The sentence the condition came from still gets its own duty.
        self.assertEqual([action_of(v) for v in result.output.obligations[1:]], [REQUEST, RETURN])

    def test_a_duty_shifted_over_a_full_stop_is_cut_back_to_its_own_sentence(self):
        _, text = unit('24/A', 3)
        shifted = text[text.index(RETURN):text.index(FULL) + len(FULL)]          # "... iade eder. Gönderilen ... bulundurulur"
        result = extract(text, Scripted(answer(candidate(text, RECEIVER, 'MUST', shifted, conditions=[PERSISTENT]))))
        self.clean(text, result)
        cut, request, give_back = result.output.obligations
        self.assertEqual(cut.required_action, FULL)
        self.assertEqual([action_of(v) for v in (request, give_back)], [REQUEST, RETURN])
        [record] = records(result, COVERED_CODE)
        self.assertEqual({k: record['fields'][0][k] for k in ('field', 'model', 'source', 'sentence', 'detail')},
                         {'field': 'required_action', 'model': shifted, 'source': FULL, 'sentence': 2, 'detail': OTHER_SENTENCE})
        self.assertEqual(trace_of(result)[0], {'candidate': 0, 'code': SOURCE_BOUND_OK, 'tier': 'structured', 'sentence': 2})
        # No duty's source sentence holds two sentences any more.
        for value in result.output.obligations:
            self.assertEqual(len(sentences(sentence_of(text, value))), 1)

    def test_a_sentence_that_has_its_duty_gets_no_second_one(self):
        _, text = unit('24/A', 3)
        whole = answer(candidate(text, RECEIVING, 'MUST', REQUEST),
                       candidate(text, RECEIVING, 'MUST', RETURN, conditions=['Bilgilerin tamamlanmaması halinde']),
                       candidate(text, RECEIVER, 'MUST', FULL, conditions=[PERSISTENT]))
        result = extract(text, Scripted(whole))
        self.clean(text, result)
        self.assertEqual(result.output, ExtractionOutput.model_validate_json(whole))
        self.assertEqual([d['code'] for d in result.diagnostics], ['CANDIDATE_REQUIRES_LEGAL_REVIEW'])
        # The rules baseline reads one duty per sentence already: nothing is added to it.
        rules = extract(text, RulesProvider())
        self.assertEqual(rules.output, ExtractionOutput.model_validate_json(RulesProvider().generate(text)))
        self.assertEqual(records(rules, COVERED_CODE) + records(rules, UNCOVERED_CODE), [])

    def test_a_duty_sentence_with_no_subject_to_take_is_recorded_not_invented(self):
        text = ('(4) Bilgilerin tamamlanmaması halinde söz konusu transferi iade eder. '
                'Yükümlüler, şüpheli işlemleri Başkanlığa bildirmek zorundadır.')
        model = answer(candidate(text, 'Yükümlüler', 'MUST', 'şüpheli işlemleri Başkanlığa bildirmek'))
        result = extract(text, Scripted(model))
        self.clean(text, result)
        self.assertEqual(result.output, ExtractionOutput.model_validate_json(model))
        self.assertEqual(records(result, UNCOVERED_CODE)[0]['uncovered'], [{'sentence': 0, 'detail': 'NO_SUBJECT'}])

    def test_english_is_left_to_the_gate(self):
        text = 'A firm must retain records. The firm will notify the FCA of a breach.'
        result = extract(text, Scripted(answer(candidate(text, 'A firm', 'MUST', 'retain records.'))))
        self.assertEqual(len(result.output.obligations), 1)
        self.assertEqual([d['code'] for d in result.diagnostics], ['CANDIDATE_REQUIRES_LEGAL_REVIEW'])

    def test_a_subject_is_the_sentence_head_before_its_comma_not_a_list_or_a_condition(self):
        _, text = unit('24/A', 3)
        first, second, third = unit_sentences(text)
        self.assertEqual(text[slice(*sentence_subject(text, first[0], first[1]))], RECEIVING)
        self.assertIsNone(sentence_subject(text, second[0], second[1]))           # carried over from the sentence before
        self.assertIsNone(sentence_subject(text, third[0], third[1]))             # a condition, then "... tarafından;" (no comma)
        listing = 'Derneğin adı, amacı, kütük numarası ve adres bilgileri, dernek tüzüğü üzerinden teyit edilir.'
        self.assertIsNone(sentence_subject(listing, 0, len(listing)))
        own = 'İşlemin tekrarlanması halinde, yükümlüler, iş ilişkisini sona erdirir.'
        self.assertEqual(own[slice(*sentence_subject(own, 0, len(own)))], 'yükümlüler')

    def test_the_step_never_breaks_extraction(self):
        _, text = unit('24/A', 3)
        with patch.object(pipeline, 'cover_sentences', side_effect=RuntimeError('boom')):
            result = extract(text, Scripted(answer(candidate(text, RECEIVER, 'MUST', FULL, conditions=[PERSISTENT]))))
        self.assertEqual([action_of(v) for v in result.output.obligations], [FULL])
        self.assertEqual(records(result, UNCOVERED_CODE)[0]['detail'], 'COVERAGE_FAILED:RuntimeError')

    def test_the_t5_ladder_answers_come_out_as_before(self):
        # md. 46(1), 21(2), 24/A(1), 31(1): the t5 answers; their other sentences state no duty, so nothing is added.
        from test_v019_t5_grounding import MD46_DUTY, MD46_NOT_A_DUTY, TypoTests
        cases = [(unit('46', 1)[1], [('Yükümlüler', MD46_DUTY), ('Yükümlü nezdindeki hesaplarla ilgili', MD46_NOT_A_DUTY)], ['drop', 'qualifier']),
                 (unit('31', 1)[1], [(TypoTests.MD31_SUBJECT, TypoTests.MD31_TAIL.replace('istenilen', 'isterilen'))], ['semantic', 'span_repair'])]
        text = unit('24/A', 1)[1]
        first = text[text.index('tarafından aracılık'):text.index(' zorunlu olup')]
        cases.append((text, [('Kripto varlık hizmet sağlayıcılar', first.replace(' cüzdan adresinin bulunmadığı durumda işlemle ilgili '
                                                                                  'referans numarasına,', ''))], ['overlap']))
        for text, values, steps in cases:
            with self.subTest(text=text[:30]):
                result = extract(text, Scripted(answer(*[candidate(text, s, 'MUST', a) for s, a in values])))
                self.assertEqual([r['steps'] for r in records(result, 'GROUNDING_SALVAGED')], [steps])
                self.assertEqual(records(result, COVERED_CODE) + records(result, UNCOVERED_CODE), [])
                self.assertEqual(len(result.output.obligations), 1)
                self.assertNotIn('cross_sentence', json.dumps(trace_of(result)))


def transfer_rows(result, payload, clause='3'):
    """Harness-style prediction rows for the duties extracted from one paragraph of the TRANSFER provision."""
    offset, text = next((o, u) for o, u in split_units(TRANSFER) if u.startswith(f'({clause})'))
    sources = source_index(payload)
    actions = [action_of(v) for v in result.output.obligations]
    rows = []
    for index, value in enumerate(result.output.obligations):
        structure = structure_of(text, value.model_dump(mode='json'), offset, siblings=[a for j, a in enumerate(actions) if j != index])
        span = duty_span({'structure': structure, 'candidate': value.model_dump(mode='json')}, sources.get('24'))
        rows.append({'key': f'd{index}', 'obligation_id': f'd{index}', 'article': '24', 'clause': clause, 'subject': value.subject,
                     'action': action_of(value), 'modality': value.modality, 'conditions': value.conditions, 'exceptions': value.exceptions,
                     'deadline': None, 'source_span': span})
    return rows


class OtherSentenceMatchingTests(unittest.TestCase):
    LABELS = [label(article='24', clause='(3)', keywords=['gönderen']), label(article='24', clause='(3)', keywords=['iade']),
              label(article='24', clause='(3)', keywords=['sürekli'])]

    def packet(self):
        return {'cases': [provision('Yönetmelik 200713012 md. 24', TRANSFER, {'3': ('OBLIGATION', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', 1)})]}

    def test_a_label_sentence_with_no_duty_of_its_own_stays_other_sentence_until_its_duty_is_extracted(self):
        payload = self.packet()
        _, text = next((o, u) for o, u in split_units(TRANSFER) if u.startswith('(3)'))
        model = answer(candidate(text, 'alıcı kuruluş', 'MUST', 'iş ilişkisine son verilmesi değerlendirilir'))
        before = transfer_rows(SimpleNamespace(output=ExtractionOutput.model_validate_json(model)), payload)
        matches, _ = pair_predictions(self.LABELS, before, source_index(payload))
        self.assertEqual([m['basis'] for m in matches], ['OTHER_SENTENCE', 'OTHER_SENTENCE', 'SOURCE_SPAN'])
        self.assertEqual((matches[1]['missing_reason'], matches[1]['missing_detail']), ('MATCHING_FAILED', 'LABEL_SENTENCE_NOT_EXTRACTED'))
        # The same answer through the pipeline: each sentence's label pairs with the duty of that sentence.
        result = extract(text, Scripted(model))
        after = transfer_rows(result, payload)
        matches, extras = pair_predictions(self.LABELS, after, source_index(payload))
        self.assertEqual([(m['prediction'] or {}).get('action') for m in matches],
                         ['mesajı gönderen kuruluştan eksik bilgilerin tamamlanmasını talep eder', 'Bilgiler tamamlanmazsa transfer iade edilir',
                          action_of(result.output.obligations[0])])
        self.assertEqual([m['basis'] for m in matches], ['FINGERPRINT', 'FINGERPRINT', 'SOURCE_SPAN'])
        self.assertEqual(extras, [])

    def test_a_structure_span_run_on_into_the_next_sentence_names_the_sentence_of_the_marker(self):
        payload = self.packet()
        offset, text = next((o, u) for o, u in split_units(TRANSFER) if u.startswith('(3)'))
        first, second, _ = sentences(text)
        first = first[len('(3) '):]
        run_on = text[text.index(first):text.index(second) + len(second)]
        marker = text.index('talep eder') + len('talep ')
        structure = {'unit_offset': offset, 'source_span': {'text': run_on, 'start': text.index(run_on), 'end': text.index(run_on) + len(run_on)},
                     'modal': {'start': marker, 'end': marker + len('eder')}}
        row = {'structure': structure, 'candidate': {'required_action': 'eksik bilgilerin tamamlanmasını talep eder.'}}
        source = source_index(payload)['24']
        span = duty_span(row, source)
        self.assertEqual(TRANSFER[span['start']:span['end']], first)
        ask = {'key': 'ask', 'article': '24', 'clause': '3', 'subject': 'kuruluş', 'action': 'eksik bilgilerin tamamlanmasını talep eder.',
               'modality': 'MUST', 'conditions': [], 'exceptions': [], 'deadline': None, 'source_span': span}
        (match,), _ = pair_predictions([self.LABELS[1]], [ask], source_index(payload))
        self.assertEqual((match['prediction'], match['basis'], match['missing_detail']), (None, 'OTHER_SENTENCE', 'LABEL_SENTENCE_NOT_EXTRACTED'))
        # Without the marker the recorded span is all there is to go by, and it reaches the "iade" sentence (the t5 reading).
        del structure['modal']
        (match,), _ = pair_predictions([self.LABELS[1]], [{**ask, 'source_span': duty_span(row, source)}], source_index(payload))
        self.assertEqual((match['prediction']['key'], match['basis']), ('ask', 'SOURCE_SPAN'))

    def test_a_one_sentence_span_is_left_as_recorded(self):
        payload = self.packet()
        offset, text = next((o, u) for o, u in split_units(TRANSFER) if u.startswith('(3)'))
        first = sentences(text)[0][len('(3) '):]
        marker = text.index('talep eder') + len('talep ')
        row = {'structure': {'unit_offset': offset, 'source_span': {'text': first, 'start': text.index(first), 'end': text.index(first) + len(first)},
                             'modal': {'start': marker, 'end': marker + 4}}, 'candidate': {'required_action': 'talep eder'}}
        span = duty_span(row, source_index(payload)['24'])
        self.assertEqual((span['start'], span['end'], span['origin']), (offset + text.index(first), offset + text.index(first) + len(first), 'STRUCTURE'))


# --- the recorded v019t5 micro answers, replayed through the pipeline (no model) -----------------------------------

TERMINAL_CODES = ('GROUNDING_REJECTED', 'SECOND_PASS_UNCERTAIN', 'CONTEXT_BUDGET_EXCEEDED')


def unit_records(diagnostics):
    """The diagnostics of one provision, split per worked unit (each unit ends with a terminal record)."""
    out, current = [], []
    for record in diagnostics:
        current.append(record)
        if record.get('stage') in ('complete', 'classification') or record.get('code') in TERMINAL_CODES:
            out.append(current)
            current = []
    return out + ([current] if current else [])


class Recorded:
    """The recorded answers of one unit, in order; the review passes (every replayed unit passed it in the run)."""
    name, model_version = 'recorded', 'v019t5'

    def __init__(self, answers, text, context):
        self.answers, self.text, self.context, self.calls = answers, text, context, 0

    def generate_with_context(self, text, context):
        raw = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        self.last_raw = raw
        return materialize(raw, self.text, self.context).model_dump_json()

    def repair(self, text, context, feedback):
        return self.generate_with_context(text, context)

    def review(self, text, output, context):
        return True


def replay(run):
    """[(case, label, unit offset, recorded rows, recorded unit records, result)] for every extracted unit of a run."""
    out = []
    for path in sorted((run / 'packets').glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))['events'][0]['payload']
        for case in payload.get('cases') or []:
            text, name = case['source']['text'], case['source']['printed_label']
            worked = case.get('units') or [{'offset': 0, 'chars': len(text), 'reason': case['reason']}]
            context = case.get('context') or {}
            context = ContextPacket(target_id='x', items=tuple(context['items']), unresolved=(), truncated=False, source_kind='CONSOLIDATED',
                                    quality_flags=(), dependency_gaps=tuple(context.get('dependency_gaps', []))) if context.get('items') else None
            for entry, found in zip(worked, unit_records(case['diagnostics'])):
                if found[-1].get('stage') != 'complete':
                    continue
                piece = text[entry['offset']:entry['offset'] + entry['chars']]
                rows = [o for o in payload['obligations'] if o['source_label'] == name and (o.get('multipart') or {}).get('unit_offset', 0) == entry['offset']]
                answers = [d['response_excerpt'] for d in found if d.get('code') == 'EVIDENCE_INVALID']
                if not any(d.get('code') == 'EVIDENCE_INVALID' and d.get('attempt') == found[-1]['attempt'] for d in found):
                    answers.append(json.dumps({'status': 'EXTRACTED', 'obligations': [
                        {'subject': o['candidate']['subject'], 'modality': o['candidate']['modality'], 'action': action_of_row(o),
                         'conditions': o['candidate']['conditions'], 'exceptions': o['candidate']['exceptions'], 'evidence': []} for o in rows]}))
                out.append((path.stem, name, entry['offset'], rows, found, extract(piece, Recorded(answers, piece, context), context)))
    return out


def action_of_row(row):
    return row['candidate']['required_action'] or row['candidate']['prohibited_action']


@unittest.skipUnless(all((run / 'results.json').is_file() for run in MICRO_RUNS), 'the v019t5 micro runs are not on this machine')
class RecordedMicroAnswerTests(unittest.TestCase):
    def test_only_the_24A_3_unit_gains_duties_and_the_t5_units_keep_their_trace(self):
        changed, t5_units = [], 0
        for run in MICRO_RUNS:
            for case, name, offset, rows, found, result in replay(run):
                recorded = [(o['candidate']['subject'], action_of_row(o), o['candidate']['conditions']) for o in rows]
                now = [(v.subject, action_of(v), v.conditions) for v in result.output.obligations]
                self.assertEqual(now[:len(recorded)], recorded, f'{case} {name} {offset}')     # the model's duties are untouched
                if now != recorded:
                    changed.append((case[:3], name.rsplit(' ', 1)[-1], offset, [a for _, a, _ in now[len(recorded):]]))
                steps = [d['steps'] for d in found if d.get('code') == 'GROUNDING_SALVAGED']
                self.assertEqual([d['steps'] for d in records(result, 'GROUNDING_SALVAGED')], steps, f'{case} {name} {offset}')
                if steps:
                    t5_units += 1
                    self.assertEqual([{k: v for k, v in e.items() if k != 'cross_sentence'} for e in trace_of(result) if e.get('origin') != 'sentence'],
                                     found[-1]['grounding'], f'{case} {name} {offset}')
        self.assertEqual(changed, [('I06', '24/A', 1111, [REQUEST, RETURN])])
        self.assertGreaterEqual(t5_units, 6)                    # 46(1) twice, 24/A(1), 24/A(3), 31(1), 21(2), ...

    def test_the_24A_3_label_now_pairs_with_the_duty_of_its_sentence(self):
        run = MICRO_RUNS[0]
        dataset = h.load_dataset(Path(__file__).resolve().parents[2] / 'evaluation' / 'independent' / 'independent-v1.json')
        case = next(c for c in dataset.cases if c.case_id.startswith('I06'))
        payload = json.loads((run / 'packets' / f'{case.case_id}.json').read_text(encoding='utf-8'))['events'][0]['payload']
        sources = source_index(payload)
        predictions = []
        for _, name, offset, _, _, result in replay(run):
            if not name.endswith('md. 24/A'):
                continue
            piece = next(u for o, u in split_units(sources['24/A']['text']) if o == offset)
            number = UNIT_NUMBER.match(piece)
            for index, value in enumerate(result.output.obligations):
                structure = structure_of(piece, value.model_dump(mode='json'), offset)
                predictions.append({'key': f'{offset}/{index}', 'article': article_of(name), 'clause': number.group(1) if number else '',
                                    'subject': value.subject, 'action': action_of(value), 'modality': value.modality,
                                    'conditions': value.conditions, 'exceptions': value.exceptions, 'deadline': None,
                                    'source_span': duty_span({'structure': structure, 'candidate': value.model_dump(mode='json')}, sources['24/A'])})
        wanted = [e for e in case.expected_obligations if e.article == '24/A' and e.clause == '(3)']
        (match,), _ = pair_predictions(wanted, predictions, sources)
        self.assertEqual((match['basis'], match['prediction']['action']), ('FINGERPRINT', RETURN))
        span = match['prediction']['source_span']
        self.assertTrue(sources['24/A']['text'][span['start']:span['end']].startswith('Bilgilerin tamamlanmaması halinde'))


if __name__ == '__main__':
    unittest.main()

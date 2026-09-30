"""Reading units, the two readers and the layer of a disagreement. No model is called: the model readings are small
records written the way live.read_units writes them, and the one run through the extraction pipeline uses the
rules provider."""
import json
import tempfile
import unittest
from pathlib import Path

from regchain.model_router import ModelRouter
from regchain.tr import ai
from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation
from regchain.tr.live import load_readings, read_units, select_units, unit_sections, units_of
from regchain.tr.packs import Registry

REGISTRY = Registry.load()
STORE = CorpusStore()


def section(regulation_id, number):
    return next(s for s in STORE.sections(regulation_id) if s['paragraph_number'] == number)


def record(unit, status, reason, candidates=(), diagnostics=()):
    return {**unit, 'status': status, 'reason': reason, 'attempts': 1, 'candidates': list(candidates), 'diagnostics': list(diagnostics)}


def candidate(modality, quote):
    return {'modality': modality, 'source_quote': quote, 'subject': '', 'required_action': None, 'prohibited_action': None}


class UnitTests(unittest.TestCase):
    def test_a_statute_with_unnumbered_paragraphs_is_read_sentence_by_sentence(self):
        # Kanun 4250 md. 6 went to the model as one unit of 4,171 characters (run 20260930-beverage-live-2: no obligation).
        article = section('TR:KANUN:4250', '6')
        units = units_of(article, 'TR:KANUN:4250')
        self.assertEqual((len(units), {u['mode'] for u in units}), (26, {'SENTENCE'}))
        self.assertTrue(all(article['text'][u['start']:u['end']] == u['input'] for u in units))
        self.assertLess(max(len(u['input']) for u in units), 420)
        first = units[0]
        self.assertEqual((first['refs'], first['input']), (['Kanun 4250 md. 6/f.1/c.1'],
                         'Alkollü içkilerin her ne surette olursa olsun reklamı ve tüketicilere yönelik tanıtımı yapılamaz.'))

    def test_an_item_without_a_predicate_of_its_own_is_read_with_its_lead_in_and_closing_line(self):
        article = section('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', '10')
        units = units_of(article, 'TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM')
        stadium = next(u for u in units if u['refs'] == ['Yönetmelik 14646 md. 10/f.3/b.c'])
        self.assertEqual(stadium['mode'], 'ITEM')
        self.assertEqual(stadium['input'], 'Aşağıda sayılan yerlerde faaliyet gösteren işyerlerinde alkollü içki satışı veya sunumu '
                                           'yapılamaz: Spor müsabakası yapılan stadyum ve kapalı spor salonlarında.')
        self.assertEqual(' '.join(article['text'][a:b] for a, b in stadium['segments']), stadium['input'])   # exact spans, joined
        waste = section('TR:YONETMELIK:AMBALAJ_ATIKLARI', '10')
        register = next(u for u in units_of(waste, 'TR:YONETMELIK:AMBALAJ_ATIKLARI') if u['refs'] == ['Yönetmelik 38745 md. 10/f.1/b.ç'])
        self.assertEqual((register['mode'], len(register['segments'])), ('ITEM', 3))
        self.assertTrue(register['input'].startswith('Piyasaya sürenler; Ambalaj Bilgi Sistemine'))
        self.assertTrue(register['input'].endswith('vermekle, yükümlüdürler.'))    # the closing line carries the duty word

    def test_a_bent_that_is_a_sentence_is_read_without_the_lead_in_that_carries_its_own_duty_word(self):
        # "... aşağıdaki kurallara da uyulmalıdır: a) ... olmalıdır." read as one text has two duty markers; the answer that
        # covered the bent was rejected for the lead-in's (UNCOVERED_MODAL on every bent of Tebliğ 11376 md. 11).
        units = units_of(section('TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER', '11'), 'TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER')
        name = next(u for u in units if u['refs'] == ['Tebliğ 11376 md. 11/f.1/b.a/c.1'])
        self.assertEqual(name['input'], 'Ürünlerin satış isimleri tanımlarda yer aldığı biçimde olmalıdır.')
        self.assertIn('uyulmalıdır', name['lead'])
        self.assertNotIn('uyulmalıdır', name['input'])

    def test_unit_provisions_for_the_engine_keep_the_article_and_its_number(self):
        sections, labels, by_label = unit_sections(STORE.sections('TR:KANUN:4250'), 'TR:KANUN:4250', ['6'])
        self.assertEqual(len(labels), 26)
        self.assertEqual(sum(1 for s in sections if s['printed_label'] == 'Kanun 4250 md. 6'), 1)
        unit = next(s for s in sections if s['printed_label'] == 'Kanun 4250 md. 6/f.5/c.3')
        self.assertEqual((unit['paragraph_number'], unit['locator_kind']), ('6', 'mevzuat_birim'))   # no answer to "6 ncı madde"
        self.assertEqual(unit['text'], 'Alkollü içkiler, 22:00 ila 06:00 saatleri arasında perakende olarak satılamaz.')
        self.assertEqual(len({s['id'] for s in sections}), len(sections))

    def test_a_read_run_classifies_before_it_asks_and_records_every_unit(self):
        units = select_units({'TR:KANUN:4250': ['9']}, REGISTRY, STORE)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'readings.jsonl'
            summary = read_units(units, ModelRouter('rules').extraction_provider(), STORE, path)
            header, records = load_readings(path)
        self.assertEqual((summary['units'], header['format'], len(records)), (len(units), 'cardaman-tr-clause-readings/2', len(units)))
        by_ref = {r['refs'][0]: r for r in records}
        exempt = by_ref['Kanun 4250 md. 9/f.2/c.2']                                 # "... turizm belgeli işletmeler için uygulanmaz."
        self.assertEqual((exempt['status'], exempt['reason'], exempt['attempts']), ('NOT_EXTRACTED', 'CLASSIFIED_EXEMPTION', 0))
        licence = by_ref['Kanun 4250 md. 9/f.1/c.2']
        self.assertEqual(licence['classification']['kind'], 'OBLIGATION')
        self.assertIsNotNone(licence['context'])


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.units = {u['refs'][0]: u for u in units_of(section('TR:KANUN:4250', '6'), 'TR:KANUN:4250')}
        self.frames = {f.ref: f for f in extract_regulation('TR:KANUN:4250', REGISTRY, STORE, articles=['6'])[0]}

    def reading(self, ref, *args, **kwargs):
        return ai.compare_readings([self.frames[ref]], [record(self.units[ref], *args, **kwargs)])[0]

    def test_both_readers_on_the_same_duty(self):
        ref = 'Kanun 4250 md. 6/f.5/c.3'
        both = self.reading(ref, 'EXTRACTED', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', [candidate('MUST_NOT', self.units[ref]['input'])])
        self.assertEqual((both.agreement, both.verdict, both.support, both.diagnosis), ('BOTH_DUTY', 'PROHIBITION', 'CONFIRMED', None))

    def test_a_duty_left_out_is_a_model_failure_and_a_partial_confirmation(self):
        ref = 'Kanun 4250 md. 6/f.5/c.1'                                            # "... satılamaz, ... oyun ve bahse konu edilemez."
        gate = {'attempt': 2, 'stage': 'grounding', 'code': 'EVIDENCE_INVALID',
                'detail': 'UNCOVERED_MODAL: explicit modal evidence is not covered by any action',
                'response_excerpt': json.dumps({'status': 'EXTRACTED', 'obligations': [{'subject': 'Alkollü içkiler', 'modality': 'MUST_NOT', 'action': 'satılamaz'}]})}
        left = self.reading(ref, 'INSUFFICIENT_EVIDENCE', 'GROUNDING_REJECTED', diagnostics=[gate])
        self.assertEqual((left.agreement, left.verdict, left.support), ('RULE_ONLY', 'PROHIBITION', 'PARTLY_CONFIRMED'))
        self.assertEqual((left.diagnosis.layer, left.diagnosis.code), ('MODEL', 'DUTY_LEFT_OUT'))
        self.assertFalse(left.review_required)

    def test_an_action_that_is_no_span_of_the_unit_is_a_grounding_failure(self):
        ref = 'Kanun 4250 md. 6/f.10'
        gate = {'attempt': 2, 'stage': 'grounding', 'code': 'EVIDENCE_INVALID', 'detail': 'ACTION_ALIGNMENT: subject, modality and action must align with source',
                'response_excerpt': json.dumps({'status': 'EXTRACTED', 'obligations': [{'subject': 'İçerdiği alkol miktarı', 'modality': 'MUST',
                                                                                     'action': 'alcohol completely removed from the product'}]})}
        invented = self.reading(ref, 'INSUFFICIENT_EVIDENCE', 'GROUNDING_REJECTED', diagnostics=[gate])
        self.assertEqual((invented.diagnosis.layer, invented.diagnosis.code), ('GROUNDING', 'ACTION_NOT_IN_SOURCE'))
        copied = dict(gate, response_excerpt=json.dumps({'status': 'EXTRACTED', 'obligations': [{'subject': 'x', 'modality': 'MUST', 'action': 'şekilde yazılır'}]}))
        misaligned = self.reading(ref, 'INSUFFICIENT_EVIDENCE', 'GROUNDING_REJECTED', diagnostics=[copied])
        self.assertEqual((misaligned.diagnosis.layer, misaligned.diagnosis.code), ('MODEL', 'ACTION_MISALIGNED'))

    def test_what_the_pipeline_withheld_is_not_held_against_the_model(self):
        ref = 'Kanun 4250 md. 6/f.8/c.1'
        unresolved = self.reading(ref, 'INSUFFICIENT_EVIDENCE', 'UNRESOLVED_CROSS_REFERENCE')
        self.assertEqual((unresolved.diagnosis.layer, unresolved.support), ('RETRIEVAL', 'RULE_ONLY'))
        no_marker = self.reading(ref, 'NOT_EXTRACTED', 'NO_DUTY_MARKER')
        self.assertEqual((no_marker.diagnosis.layer, no_marker.diagnosis.code), ('VALIDATOR_PIPELINE', 'GATE_MARKER_GAP'))
        classed = ai.compare_readings([self.frames[ref]], [dict(record(self.units[ref], 'NOT_EXTRACTED', 'CLASSIFIED_PERMISSION'),
                                                                 classification={'kind': 'PERMISSION', 'marker': 'x', 'basis': 'wording'})])[0]
        self.assertEqual((classed.diagnosis.layer, classed.diagnosis.code), ('VALIDATOR_PIPELINE', 'PRECLASSIFIED_NOT_A_DUTY'))
        denied = self.reading(ref, 'NO_EXPLICIT_OBLIGATION', 'NO_EXPLICIT_OBLIGATION')
        self.assertEqual((denied.diagnosis.layer, denied.support, denied.review_required), ('MODEL', 'CONTESTED', True))

    def test_a_duty_only_the_model_sees_goes_to_a_person(self):
        ref = 'Kanun 4250 md. 6/f.8/c.4'                                            # "... Kurumu tarafından belirlenir."
        only = self.reading(ref, 'EXTRACTED', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', [candidate('MUST', self.units[ref]['input'])])
        self.assertEqual((only.rule_kind, only.agreement, only.verdict, only.support), ('DELEGATION', 'MODEL_ONLY', 'NONE', 'MODEL_ONLY'))
        self.assertEqual((only.diagnosis.layer, only.diagnosis.code, only.review_required), ('MODEL', 'NON_DUTY_READ_AS_DUTY', True))

    def test_the_summary_counts_layers_and_names_no_clause_unread(self):
        refs = ['Kanun 4250 md. 6/f.5/c.3', 'Kanun 4250 md. 6/f.8/c.1', 'Kanun 4250 md. 6/f.1/c.3']
        records = [record(self.units[refs[0]], 'EXTRACTED', 'CANDIDATE_REQUIRES_LEGAL_REVIEW', [candidate('MUST_NOT', '')]),
                   record(self.units[refs[1]], 'INSUFFICIENT_EVIDENCE', 'UNRESOLVED_CROSS_REFERENCE')]
        summary = ai.reading_summary(ai.compare_readings([self.frames[r] for r in refs], records))
        self.assertEqual((summary['clauses'], summary['read'], summary['agreement']['NOT_READ']), (3, 2, 1))
        self.assertEqual(summary['by_layer'], {'RETRIEVAL': 1, 'MODEL': 0, 'GROUNDING': 0, 'SCOPE': 0, 'VALIDATOR_PIPELINE': 0})
        self.assertEqual(summary['agreement_rate'], 0.5)


class EngineJoinTests(unittest.TestCase):
    def test_rows_of_a_unit_run_are_joined_by_the_clauses_the_unit_names(self):
        obligations = extract_regulation('TR:KANUN:4250', REGISTRY, STORE, articles=['6'])[1]
        text = 'Alkollü içkiler, 22:00 ila 06:00 saatleri arasında perakende olarak satılamaz.'
        row = {'source_label': 'Kanun 4250 md. 6/f.5/c.3', 'candidate': {'source_quote': text, 'prohibited_action': 'perakende olarak satılamaz'},
               'proposal': {'coverage': 'CONFLICT', 'policy_checks': [{'relation': 'CONFLICTS', 'quote': '2. ... 23:00 saatine kadar yapılır.'}]}}
        run = {'units': {'Kanun 4250 md. 6/f.5/c.3': {'refs': ['Kanun 4250 md. 6/f.5/c.3']}},
               'packet': {'events': [{'payload': {'obligations': [row], 'cases': []}}]}}
        joined = ai.engine_rows(run, obligations)
        self.assertEqual(list(joined), ['Kanun 4250 md. 6/f.5/c.3'])
        self.assertEqual(ai.engine_coverage(joined['Kanun 4250 md. 6/f.5/c.3']), 'CONTRADICTED')
        whole = {'units': None, 'packet': {'events': [{'payload': {'obligations': [dict(row, source_label='Kanun 4250 md. 6')], 'cases': []}}]}}
        self.assertEqual(list(ai.engine_rows(whole, obligations)), ['Kanun 4250 md. 6/f.5/c.3'])       # joined by quote

    def test_a_provision_that_went_in_as_one_large_unit_is_a_pipeline_failure(self):
        case = {'source': {'printed_label': 'Kanun 4250 md. 6', 'text': 'x' * 4171}, 'reason': 'GROUNDING_REJECTED',
                'output': {'status': 'INSUFFICIENT_EVIDENCE', 'obligations': []}, 'classification': [{'kind': 'OBLIGATION', 'marker': 'konulur'}],
                'diagnostics': [{'code': 'EVIDENCE_INVALID', 'detail': 'ACTION_ALIGNMENT: subject, modality and action must align with source'}]}
        found = ai.diagnose_engine_coverage(['COVERED'], [], case)
        self.assertEqual((found.layer, found.code), ('VALIDATOR_PIPELINE', 'UNIT_TOO_LARGE'))
        self.assertEqual(ai.diagnose_engine_coverage(['COVERED'], [], None).layer, 'RETRIEVAL')

    def test_coverage_diagnoses(self):
        row = {'proposal': {'coverage': 'COVERS_TEXT', 'policy_checks': [{'relation': 'SUPPORTS', 'quote': '1. Okul kantinlerinde ve spor tesislerinde ...'}]}}
        over = ai.diagnose_engine_coverage(['PARTIALLY_COVERED'], [row], None, decisive=['1. Okul kantinlerinde ve spor tesislerinde ...'])
        self.assertEqual((over.layer, over.code), ('MODEL', 'SUPPORT_OVERSTATED'))
        unread = ai.diagnose_engine_coverage(['PARTIALLY_COVERED'], [row], None, decisive=['7. Başka bir cümle.'])
        self.assertEqual((unread.layer, unread.code), ('RETRIEVAL', 'DECISIVE_PASSAGE_NOT_JUDGED'))
        conflict = {'proposal': {'coverage': 'CONFLICT', 'policy_checks': [{'relation': 'CONFLICTS', 'quote': '3. ... numune dağıtımı yapılabilir.'}]}}
        self.assertEqual(ai.diagnose_engine_coverage(['PARTIALLY_COVERED'], [conflict], None).code, 'FALSE_CONFLICT')
        self.assertEqual(ai.diagnose_rule_coverage(['COVERED'], 'NOT_COVERED', ['NO_RELATED_STATEMENT']).layer, 'VALIDATOR_PIPELINE')


if __name__ == '__main__':
    unittest.main()

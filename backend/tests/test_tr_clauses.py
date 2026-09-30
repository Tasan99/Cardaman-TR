"""Clause segmentation and clause frames, on the stored official texts (no network, no model).

The expectations are a developer's reading of the quoted wording, written down so a change in the rules shows;
they are not a legal opinion and not expert gold.
"""
import unittest

from regchain.tr.clauses import Clause, blank_notes, clause_at, sentence_spans, split_clauses
from regchain.tr.corpus import CorpusStore
from regchain.tr.facts import alcoholic_from, source_facts
from regchain.tr.frames import fold, frames_of, lexicon

STORE = CorpusStore()


def section(regulation_id, label):
    return STORE.section(regulation_id, label)


def frames(regulation_id, label):
    return {f.ref.split('/', 1)[1]: f for f in frames_of(section(regulation_id, label), regulation_id)}


def synthetic(text, lines=None, label='Tebliğ 99001 md. 1', heading=''):
    """A synthetic provision (a rule fixture, not law)."""
    spans, offset = [], 0
    for line in lines or [text]:
        start = text.index(line, offset)
        spans.append({'kind': 'line', 'start': start, 'end': start + len(line)})
        offset = start + len(line)
    return {'text': text, 'printed_label': label, 'heading_path': ['', '', heading], 'lines': spans, 'quality_flags': []}


class SegmentationTests(unittest.TestCase):
    def test_every_clause_is_an_exact_span_of_the_stored_article(self):
        for regulation_id in ('TR:KANUN:4250', 'TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'TR:YONETMELIK:AMBALAJ_ATIKLARI',
                              'TR:YONETMELIK:TGK_ETIKETLEME'):
            for row in STORE.sections(regulation_id):
                if 'DELETED_PROVISION' in row['quality_flags']:
                    continue
                refs = set()
                for clause in split_clauses(row):
                    self.assertEqual(row['text'][clause.start:clause.end], clause.text, clause.ref)
                    self.assertEqual(len(clause.reading), len(clause.text))
                    if clause.chapeau:
                        self.assertEqual(row['text'][clause.chapeau_start:clause.chapeau_start + len(clause.chapeau)], clause.chapeau)
                    self.assertNotIn(clause.ref, refs, clause.ref)
                    refs.add(clause.ref)

    def test_an_unnumbered_statute_is_cut_by_its_source_paragraphs_and_sentences(self):
        clauses = split_clauses(section('TR:KANUN:4250', 'Kanun 4250 md. 6'))
        night = clause_at(clauses, 'Kanun 4250 md. 6/f.5/c.3')       # "beşinci fıkrasının üçüncü cümlesi" (md. 7/f)
        self.assertEqual(night.text, 'Alkollü içkiler, 22:00 ila 06:00 saatleri arasında perakende olarak satılamaz.')
        self.assertEqual(max(c.fikra for c in clauses), 11)
        self.assertFalse(night.numbered)
        self.assertTrue(clause_at(clauses, 'Kanun 4250 md. 6/f.11/c.2').text.startswith('Öğrenci yurtları'))

    def test_a_bent_carries_the_lead_in_and_the_closing_predicate_of_its_fikra(self):
        clauses = split_clauses(section('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 10'))
        item = clause_at(clauses, 'Yönetmelik 38745 md. 10/f.1/b.ç')
        self.assertEqual(item.chapeau, '(1) Piyasaya sürenler;')
        self.assertEqual(item.closing, 'yükümlüdürler.')
        self.assertTrue(item.text.startswith('ç) Ambalaj Bilgi Sistemine'))
        self.assertTrue(item.composed().endswith('yükümlüdürler.'))

    def test_a_dash_list_and_its_predicate_stay_inside_their_bent(self):
        clauses = split_clauses(section('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 5'))
        limits = clause_at(clauses, 'Tebliğ 23706 md. 5/f.1/b.c')
        self.assertTrue(limits.text.endswith('-Taurin 800 mg/L den fazla olamaz.'))
        self.assertEqual(clause_at(clauses, 'Tebliğ 23706 md. 5/f.1/b.ç/c.1').text, 'ç) Enerji içeceklerine bileşen olarak etil alkol ilave edilmez.')

    def test_a_quoted_label_warning_is_not_cut_into_sentences(self):
        clauses = split_clauses(section('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 12'))
        warning = clause_at(clauses, 'Tebliğ 23706 md. 12/f.1/b.b')
        self.assertIn('Günlük 500 ml’den fazla tüketilmesi tavsiye edilmez.”', warning.text)
        self.assertIsNone(warning.sentence)

    def test_an_editorial_note_neither_ends_a_sentence_nor_is_read_as_wording(self):
        text = '(2) (Değişik:RG-18/9/2013-28769) Örnek içecek satılamaz. (Ek cümle:11/6/2026-7584/2 md.) Örnek içecek sunulamaz.'
        spans = sentence_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(len(blank_notes(text)), len(text))
        self.assertNotIn('Değişik', blank_notes(text))

    def test_a_repealed_item_that_keeps_only_its_marker_is_not_a_clause(self):
        text = ('(1) Satıcılar aşağıdaki belgeleri almak zorundadır: a) Örnek satış belgesi. c) (Mülga: RG-3/12/2013-28840) '
                'ç) Örnek sunum belgesi.')
        row = synthetic(text, ['(1) Satıcılar aşağıdaki belgeleri almak zorundadır:', 'a) Örnek satış belgesi.',
                               'c) (Mülga: RG-3/12/2013-28840)', 'ç) Örnek sunum belgesi.'])
        self.assertEqual([c.bent for c in split_clauses(row)], ['a', 'ç'])


class FrameTests(unittest.TestCase):
    def test_folding_keeps_every_offset(self):
        text = 'İSPİRTO ve IŞIK, Âlâ'
        self.assertEqual(len(fold(text)), len(text))
        self.assertEqual(fold(text), 'ispirto ve ışık, ala')

    def test_the_advertising_ban_is_an_impersonal_prohibition_with_its_fair_exception(self):
        ban = frames('TR:KANUN:4250', 'Kanun 4250 md. 6')['f.1/c.1']
        self.assertEqual((ban.kind, ban.modality, ban.passive), ('PROHIBITION', 'MUST_NOT', True))
        self.assertEqual(ban.actors, [])
        self.assertEqual([m.id for m in ban.activities], ['ADVERTISING', 'PRODUCT_PROMOTION'])
        self.assertEqual([m.id for m in ban.products], ['ALCOHOLIC_BEVERAGE'])
        self.assertEqual(ban.topic, 'ADVERTISING')
        self.assertEqual([(e.effect, e.source_ref) for e in ban.exceptions], [('PERMITS', 'Kanun 4250 md. 6/f.1/c.3')])
        self.assertTrue(ban.exceptions[0].quote.startswith('Ancak, münhasıran alkollü içkilerin uluslararası düzeyde'))

    def test_a_named_addressee_is_read_with_its_activities(self):
        gifts = frames('TR:KANUN:4250', 'Kanun 4250 md. 6')['f.2']
        self.assertEqual([a.id for a in gifts.actors], ['PRODUCER', 'IMPORTER', 'MARKETER'])
        self.assertFalse(gifts.passive)
        self.assertEqual(gifts.kind, 'PROHIBITION')
        waste = frames('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 10')['f.1/b.ç']
        self.assertEqual([(a.id, a.where) for a in waste.actors], [('PLACER_ON_MARKET', 'CHAPEAU')])
        self.assertEqual((waste.kind, waste.marker), ('OBLIGATION', 'yükümlüdürler (closing)'))
        agentive = frames('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 10')['f.2/c.1']
        self.assertEqual([(a.id, a.note) for a in agentive.actors], [('PLACER_ON_MARKET', 'agentive')])

    def test_every_element_is_an_exact_span_of_the_article(self):
        for regulation_id, label in (('TR:KANUN:4250', 'Kanun 4250 md. 6'), ('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 11'),
                                     ('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 11'),
                                     ('TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER', 'Tebliğ 11376 md. 11')):
            text = section(regulation_id, label)['text']
            for frame in frames_of(section(regulation_id, label), regulation_id):
                self.assertEqual(text[frame.start:frame.end], frame.text)
                for group in ('actors', 'activities', 'products', 'facilities', 'places', 'counterparties'):
                    for mention in getattr(frame, group):
                        if mention.where != 'PREVIOUS':
                            self.assertEqual(text[mention.start:mention.end], mention.text, (frame.ref, group))
                for item in (*frame.conditions, *frame.quantities):
                    self.assertEqual(text[item.start:item.end], item.quote if hasattr(item, 'quote') else item.text, frame.ref)
                for rule in frame.exceptions:
                    self.assertEqual(text[rule.start:rule.end], rule.quote, frame.ref)

    def test_an_inline_exception_with_a_product_property_is_evaluable(self):
        warning = frames('TR:KANUN:4250', 'Kanun 4250 md. 6')['f.8/c.1']
        self.assertEqual((warning.kind, warning.topic), ('OBLIGATION', 'LABELLING'))
        self.assertEqual([(e.effect, e.about, e.predicate) for e in warning.exceptions],
                         [('EXEMPTS', 'PRODUCT', {'fact': 'product.export_only', 'op': 'eq', 'value': True})])
        self.assertEqual(warning.exceptions[0].quote, 'İhraç amaçlı üretilenler hariç olmak üzere')

    def test_an_exception_sentence_reaches_the_duties_it_names(self):
        all_frames = frames('TR:KANUN:4250', 'Kanun 4250 md. 6')
        brand, fermented = all_frames['f.9/c.1'], all_frames['f.9/c.2']                    # "... bu fıkra hükmü uygulanmaz"
        self.assertEqual(all_frames['f.9/c.3'].kind, 'EXCEPTION')
        for frame in (brand, fermented):
            self.assertEqual([e.source_ref for e in frame.exceptions], ['Kanun 4250 md. 6/f.9/c.3'])
        minors = all_frames['f.4/c.1']                                                    # "... bu hükmün dışındadır"
        self.assertEqual([e.source_ref for e in minors.exceptions], ['Kanun 4250 md. 6/f.4/c.2'])
        self.assertEqual(all_frames['f.3'].exceptions, [])

    def test_bu_urunler_means_the_products_of_the_sentence_before(self):
        post = frames('TR:KANUN:4250', 'Kanun 4250 md. 6')['f.5/c.2']
        self.assertEqual([(m.id, m.where) for m in post.products], [('ALCOHOLIC_BEVERAGE', 'PREVIOUS')])
        self.assertEqual(post.product_basis, 'CLAUSE')

    def test_limits_are_read_with_unit_direction_and_attribute(self):
        energy = frames('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 5')
        self.assertEqual([(q.attribute, q.value, q.unit, q.comparator, q.role) for q in energy['f.1/b.b'].quantities],
                         [('caffeine_mg_per_l', 150.0, 'mg/L', 'le', 'LIMIT')])
        self.assertEqual([(q.attribute, q.value, q.comparator) for q in energy['f.1/b.c'].quantities],
                         [('inositol_mg_per_l', 100.0, 'le'), ('glucuronolactone_mg_per_l', 20.0, 'le'), ('taurine_mg_per_l', 800.0, 'le')])
        self.assertEqual([(q.attribute, q.value, q.unit, q.comparator) for q in energy['f.1/b.ç/c.2'].quantities],
                         [('ethanol_g_per_l', 3.0, 'g/L', 'le')])
        distance = frames('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 10')['f.5/c.1']
        self.assertEqual([(q.attribute, q.value, q.unit, q.comparator) for q in distance.quantities], [('distance_m', 100.0, 'm', 'ge')])

    def test_a_product_property_in_the_wording_is_an_evaluable_condition(self):
        strength = frames('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9')['f.1/b.i']
        self.assertEqual([c.predicate for c in strength.conditions if c.predicate],
                         [{'fact': 'product.abv_percent', 'op': 'gt', 'value': 1.2}])
        caffeine = frames('TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER', 'Tebliğ 11376 md. 11')['f.1/b.h/c.1']
        self.assertEqual([c.predicate for c in caffeine.conditions if c.predicate],
                         [{'fact': 'product.caffeine_mg_per_l', 'op': 'gt', 'value': 1.0}])
        self.assertTrue(caffeine.conditions[0].quote.startswith('Kafein miktarı 1,0 mg/L’den fazla olan ürünlerde'))

    def test_places_and_the_other_party_are_circumstances_not_profile_predicates(self):
        energy = frames('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 11')
        self.assertEqual([m.id for m in energy['f.2'].places], ['SPORTS_VENUE', 'SCHOOL', 'HEALTH_FACILITY'])
        self.assertTrue(all(c.predicate is None for c in energy['f.2'].conditions))
        self.assertEqual([(c.kind, c.quote) for c in energy['f.3'].conditions], [('COUNTERPARTY', 'On sekiz yaşından küçüklere')])
        night = frames('TR:KANUN:4250', 'Kanun 4250 md. 6')['f.5/c.3']
        self.assertEqual([(c.kind, c.quote) for c in night.conditions], [('TIME', '22:00 ila 06:00 saatleri arasında')])

    def test_text_that_carries_no_duty_for_a_company_is_classed_not_extracted(self):
        energy = {f.ref.split(' md. ', 1)[1]: f for label in ('Tebliğ 23706 md. 2', 'Tebliğ 23706 md. 4', 'Tebliğ 23706 md. 5',
                                                              'Tebliğ 23706 md. 11', 'Tebliğ 23706 md. 17')
                  for f in frames_of(section('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', label), 'TR:TEBLIG:TGK_ENERJI_ICECEKLERI')}
        self.assertEqual(energy['2/f.1'].kind, 'SCOPE')
        self.assertEqual(energy['4/f.1/b.a'].kind, 'DEFINITION')
        self.assertEqual(energy['5/f.1/b.a'].kind, 'OTHER')              # "... kapsamında değerlendirilmez": a legal reading
        self.assertEqual(energy['5/f.1/b.f'].kind, 'PERMISSION')
        self.assertEqual(energy['11/f.1'].kind, 'REFERENCE')
        self.assertEqual(energy['17/f.1'].kind, 'ENFORCEMENT')
        law = frames('TR:KANUN:4250', 'Kanun 4250 md. 6')
        self.assertEqual(law['f.8/c.4'].kind, 'DELEGATION')              # "... Kurumu tarafından belirlenir"
        self.assertEqual(law['f.1/c.6'].kind, 'PERMISSION')
        marking = frames('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 15')
        self.assertEqual(marking['f.1/c.2'].kind, 'OTHER')               # "Bu işaretleme, ... gösterir.": a description
        self.assertEqual(marking['f.4/c.1'].kind, 'DELEGATION')

    def test_a_repealed_article_has_no_frames(self):
        repealed = next(s for s in STORE.sections('TR:KANUN:4250') if 'DELETED_PROVISION' in s['quality_flags'])
        self.assertEqual(frames_of(repealed, 'TR:KANUN:4250'), [])

    def test_the_lexicon_compiles_and_names_only_vocabulary_classes(self):
        from regchain.tr.packs import Registry
        vocabulary = Registry.load().vocabulary
        data = lexicon().data
        for actor in data['actors']:
            self.assertEqual(vocabulary.unknown('activity_classes', actor['activities']), [], actor['id'])
            self.assertEqual(vocabulary.unknown('entity_classes', actor['entities']), [], actor['id'])
        for product in data['products']:
            self.assertEqual(vocabulary.unknown('product_classes', product['classes']), [], product['id'])
        for activity in data['activities']:
            if activity['class'] not in ('SALE', 'LABELLING', 'EMPLOYMENT'):
                self.assertEqual(vocabulary.unknown('activity_classes', [activity['class']]), [], activity['class'])
        self.assertEqual(vocabulary.unknown('activity_classes', data['sale_activities']), [])


class SourceFactTests(unittest.TestCase):
    def test_every_source_fact_quotes_the_stored_text(self):
        for name, fact in source_facts().items():
            with self.subTest(name):
                label = fact['provision_ref'].split('/')[0]
                text = STORE.section(fact['regulation_id'], label)['text']
                self.assertIn(fact['quote'], text)
                clause = clause_at(split_clauses(STORE.section(fact['regulation_id'], label)), fact['provision_ref'])
                self.assertIn(fact['quote'], clause.text)
                for other in fact.get('also', []):
                    self.assertIn(other['quote'], STORE.section(other['regulation_id'], other['provision_label'])['text'])

    def test_the_alcohol_threshold_is_the_one_the_text_states(self):
        self.assertEqual(alcoholic_from(), 0.5)


if __name__ == '__main__':
    unittest.main()

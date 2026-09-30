"""Wording and scope rules corrected against the labelled clauses of BEVERAGE_TR_DEV_V2 (30 September 2026).

Each test names the clause of the stored official text the rule was wrong on and the layer the miss sat in. The
expectations are a developer's reading (INDICATIVE); the corpus is read from disk and no model is called.
"""
import unittest

from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation, ground, route
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles, to_company

REGISTRY = Registry.load()
STORE = CorpusStore()
PROFILES = {p.profile_id: p for p in load_pilot_profiles(REGISTRY.vocabulary).values()}
BREWER = PROFILES['tr-bev-pilot-alcohol-integrated']
BOTTLER = PROFILES['tr-bev-pilot-non-alcohol-bottler']
_CACHE = {}


def read(regulation_id):
    if regulation_id not in _CACHE:
        frames, obligations = extract_regulation(regulation_id, REGISTRY, STORE)
        _CACHE[regulation_id] = ({f.ref: f for f in frames}, obligations)
    return _CACHE[regulation_id]


def frame(regulation_id, ref):
    return read(regulation_id)[0][ref]


def duties(regulation_id, ref):
    return [o for o in read(regulation_id)[1] if o.provision_ref == ref]


def status(regulation_id, ref, profile):
    out = {}
    for obligation in duties(regulation_id, ref):
        for decision in route(obligation, profile, REGISTRY, STORE)[0]:
            out[decision.target_id] = decision
    return out


class ClauseKindTests(unittest.TestCase):
    def test_a_list_item_that_ends_in_a_plural_noun_takes_the_duty_of_its_lead_in(self):
        # "c) Ek-1'de yer alan ... Madde veya Ürünler." and "ğ) Sosyal amaçlı yardımlar," ended in a word the aorist pattern
        # matched; the items were read as statements and dropped from the list (validator / pipeline: wording rule).
        allergens = frame('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9/f.1/b.c')
        self.assertEqual((allergens.kind, allergens.marker), ('OBLIGATION', 'zorunludur (chapeau)'))
        aid = frame('TR:YONETMELIK:TICARI_REKLAM', 'Yönetmelik 20435 md. 7/f.5/b.ğ')
        self.assertEqual((aid.kind, aid.modality), ('PROHIBITION', 'MUST_NOT'))

    def test_a_sentence_names_its_own_addressee_inside_a_list(self):
        # "Yaş konusunda tereddüde düşülmesi halinde satıcı, ... kimlik belgesi talep eder." sits under a lead-in about
        # workplaces; the lead-in's noun hid the seller and the duty was read as a description.
        ask = frame('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 6/f.1/b.f/c.2')
        self.assertEqual(ask.kind, 'OBLIGATION')
        self.assertEqual([(a.id, a.where, a.note) for a in ask.actors], [('SELLER', 'CLAUSE', 'nominative')])
        [duty] = duties('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 6/f.1/b.f/c.2')
        self.assertEqual(duty.basis, 'ACTOR_EXPLICIT')
        self.assertIn('RETAIL_SALE', duty.scope.activity_classes)

    def test_a_pronoun_subject_takes_the_parties_of_the_sentence_before(self):
        # "Bu yerler ... sorumlulukları yerine getirirler.": the places the sentence before called sales points.
        places = frame('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 11/f.2/c.2')
        self.assertEqual(places.kind, 'OBLIGATION')
        self.assertEqual([(a.id, a.where) for a in places.actors], [('SALES_POINT', 'PREVIOUS')])

    def test_an_authority_that_is_the_subject_of_the_sentence_does_its_own_task(self):
        for regulation_id, ref in (('TR:KANUN:4250', 'Kanun 4250 md. 9/f.1/c.3'),                       # "Belediye veya il özel idaresi, ... alır."
                                   ('TR:YONETMELIK:INSANI_TUKETIM_SULAR', 'Yönetmelik 7510 md. 34/f.4/b.b/c.3')):   # "... Valilikçe onaylanır."
            self.assertEqual(frame(regulation_id, ref).kind, 'DELEGATION', ref)
        # "Kurum tarafından adına dağıtım yetki belgesi düzenlenen firmalar, ... alırlar": the firms are the subject.
        self.assertEqual(frame('TR:YONETMELIK:ALKOL_IC_DIS_TICARET', 'Yönetmelik 6203 md. 12/f.2').kind, 'OBLIGATION')

    def test_statements_that_only_look_like_duties(self):
        self.assertEqual(frame('TR:KANUN:4250', 'Kanun 4250 md. 9/f.3').kind, 'OTHER')                  # when the distance condition is assessed
        self.assertEqual(frame('TR:TEBLIG:TGK_BIRA', 'Tebliğ 10474 md. 5/f.1/b.a').kind, 'OTHER')       # "... dörde ayrılır:"
        self.assertEqual(frame('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 7/f.4').kind, 'REFERENCE')
        # "... aynı renkte olma şartı aranır" states the condition itself and stays a duty.
        self.assertEqual(frame('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 17/f.3/c.2').kind, 'OBLIGATION')
        # What the consumer may not do with the right of withdrawal is no duty of the seller.
        self.assertEqual(frame('TR:KANUN:6502', 'Kanun 6502 md. 18/f.3/c.1').kind, 'OTHER')


class ScopeTests(unittest.TestCase):
    def test_the_name_of_an_authority_names_no_product(self):
        # "Tütün ve Alkol Piyasası Düzenleme Kurumundan satış belgesi almak isteyenlerin ...": the clause was dropped as
        # tobacco-only because of the first word of the authority's name (scope).
        [licence] = duties('TR:KANUN:4250', 'Kanun 4250 md. 9/f.1/c.1')
        self.assertEqual(licence.basis, 'ACTOR_EXPLICIT')
        self.assertIn('RETAIL_SALE', licence.scope.activity_classes)

    def test_coordinated_parties_and_the_party_a_duty_is_owed_to(self):
        [report] = duties('TR:YONETMELIK:ALKOL_IC_DIS_TICARET', 'Yönetmelik 6203 md. 15/f.2')            # "Üretici ve ithalatçı firmalar ..."
        self.assertLessEqual({'PRODUCTION', 'IMPORT'}, set(report.scope.activity_classes))
        [supply] = duties('TR:YONETMELIK:ALKOL_IC_DIS_TICARET', 'Yönetmelik 6203 md. 9/f.2')             # "... perakende satıcıların taleplerini ..."
        self.assertNotIn('RETAIL_SALE', supply.scope.activity_classes)
        [through] = duties('TR:YONETMELIK:ALKOL_IC_DIS_TICARET', 'Yönetmelik 6203 md. 9/f.1')            # "... toptan satıcılar eliyle yürütürler"
        self.assertNotIn('WHOLESALE', through.scope.activity_classes)
        [placed] = duties('TR:YONETMELIK:ALKOL_IC_DIS_TICARET', 'Yönetmelik 6203 md. 9/f.3')             # "üreticileri veya ithalatçıları tarafından"
        self.assertLessEqual({'PRODUCTION', 'IMPORT'}, set(placed.scope.activity_classes))

    def test_a_contract_supplier_is_not_every_producer(self):
        [register] = duties('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 10/f.5/b.a')
        self.assertEqual((register.scope.entity_classes, register.scope.activity_classes), (['CONTRACT_SUPPLIER'], []))
        routed = status('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 10/f.5/b.a', BREWER)
        self.assertEqual(routed['ALC-INT-BREWING'].status, 'DOES_NOT_APPLY')

    def test_the_operator_of_a_by_law_is_the_party_its_catalogue_names(self):
        [measures] = duties('TR:YONETMELIK:INSANI_TUKETIM_SULAR', 'Yönetmelik 7510 md. 34/f.2/c.1')      # "... işletmeci tarafından alınır"
        self.assertEqual((measures.basis, measures.scope.activity_classes), ('ACTOR_EXPLICIT', ['WATER_EXTRACTION', 'BOTTLING']))
        self.assertIn('ACTOR_IS_REGULATION_PARTY', measures.flags)

    def test_a_distance_is_a_rule_about_the_place_of_sale_not_a_property_of_the_product(self):
        [distance] = duties('TR:KANUN:4250', 'Kanun 4250 md. 9/f.2/c.1')
        self.assertEqual((distance.basis, distance.scope.level), ('GOVERNING_ACTIVITY', 'LEGAL_ENTITY'))
        self.assertIn('PLACE_OF_ACTIVITY', distance.flags)
        routed = status('TR:KANUN:4250', 'Kanun 4250 md. 9/f.2/c.1', BREWER)
        self.assertEqual((routed['ALC-INT-SALES'].status, routed['ALC-INT-BREWING'].status), ('PARTIAL', 'DOES_NOT_APPLY'))

    def test_a_date_and_information_are_not_the_acts_their_words_name(self):
        [dates] = duties('TR:YONETMELIK:INSANI_TUKETIM_SULAR', 'Yönetmelik 7510 md. 34/f.1/c.2')         # "İmal ve son kullanma tarihi ... yazılır"
        self.assertEqual((dates.basis, dates.scope.level, dates.topic), ('PRODUCT_PROPERTY', 'PRODUCT', 'LABELLING'))
        [given] = duties('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 19/f.2/c.2')              # "... bilgiler de ... sunulur"
        self.assertEqual((given.scope.level, given.scope.activity_classes), ('PRODUCT', []))

    def test_the_company_the_engine_reads_states_its_customers(self):
        sales = to_company(BREWER, 'ALC-INT-SALES', REGISTRY.vocabulary)
        self.assertTrue(any('tüketiciler' in c for c in sales.customer_types))
        self.assertEqual(to_company(BREWER, 'ALC-INT-HOLDING', REGISTRY.vocabulary).customer_types, [])     # complete, sells to no one
        importer = PROFILES['tr-bev-pilot-alcohol-import']                           # an incomplete entity that states no channel
        self.assertIsNone(to_company(importer, 'ALC-IMP-MARKETING', REGISTRY.vocabulary).customer_types)


class ExceptionTests(unittest.TestCase):
    def test_an_exception_that_cites_its_bents_lifts_them_for_the_products_its_condition_selects(self):
        # md. 19/4: "... 9 uncu maddenin birinci fıkrasının (b) ve (j) bentlerinde atıf yapılan bilgilerin hacmen % 1,2'den
        # fazla alkol içeren içeceklerde verilmesi zorunlu değildir." The ingredients list was routed to every beer (scope).
        [ingredients] = duties('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9/f.1/b.b')
        lifted = [e for e in ingredients.scope.exceptions if e.get('source_ref') == 'Yönetmelik 23282 md. 19/f.4']
        self.assertEqual([(e['fact'], e['op'], e['value']) for e in lifted], [('product.abv_percent', 'gt', 1.2)])
        self.assertTrue(ground(ingredients, STORE)['grounded'])                      # the quote is checked against md. 19
        routed = status('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9/f.1/b.b', BREWER)
        self.assertEqual((routed['ALC-INT-LAGER'].status, routed['ALC-INT-LOW'].status, routed['ALC-INT-AF'].status),
                         ('DOES_NOT_APPLY', 'APPLIES', 'APPLIES'))
        self.assertIn('EXCEPTION_APPLIES', routed['ALC-INT-LAGER'].reason_codes)
        self.assertEqual(status('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9/f.1/b.b', BOTTLER)['NONALC-COLA'].status, 'APPLIES')
        [name] = duties('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9/f.1/b.a')               # not cited: not lifted
        self.assertFalse([e for e in name.scope.exceptions if e.get('source_ref') == 'Yönetmelik 23282 md. 19/f.4'])

    def test_the_reservation_of_a_lead_in_qualifies_every_item(self):
        [name] = duties('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9/f.1/b.a')
        self.assertTrue(any('istisnalar saklı kalmak kaydıyla' in e['quote'] for e in name.scope.exceptions))
        self.assertIn('EXCEPTION_UNEVALUATED', name.flags)

    def test_sentences_that_qualify_the_duty_before_them(self):
        transport = frame('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 6/f.1/b.d/c.2')
        self.assertEqual([(e.effect, e.source_ref) for e in transport.exceptions], [('NARROWS', 'Yönetmelik 14646 md. 6/f.1/b.d/c.3')])
        signs = frame('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.5')                   # "... servis amaçlı materyallerde ... kullanılabilir."
        self.assertEqual([(e.effect, e.source_ref) for e in signs.exceptions], [('PERMITS', 'Kanun 4250 md. 6/f.1/c.6')])
        label = frame('TR:YONETMELIK:INSANI_TUKETIM_SULAR', 'Yönetmelik 7510 md. 34/f.2/c.2')            # export waters, sixth paragraph
        self.assertEqual([(e.source_ref, e.predicate) for e in label.exceptions],
                         [('Yönetmelik 7510 md. 34/f.6', {'fact': 'product.export_only', 'op': 'eq', 'value': True})])

    def test_besides_is_not_except(self):
        # "Bu Yönetmelik kapsamındaki ürünler dışında, farklı ürünlerin de satıldığı işyerlerinde; ...": no exception.
        self.assertEqual(frame('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 6/f.1/b.d/c.1').exceptions, [])


if __name__ == '__main__':
    unittest.main()

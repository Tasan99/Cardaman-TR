"""Clause-level obligations, their extracted scope and their routing on the synthetic pilot profiles.

Every expectation is a developer's reading of the quoted clause against a synthetic profile: an INDICATIVE contrast,
not expert gold. The corpus is read from disk; nothing here calls a model or the network.
"""
import unittest

from regchain.tr.core import ObligationScope
from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation, ground, regulation_scope, route
from regchain.tr.frames import regulation_frames
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles
from regchain.tr.routing import evaluate_predicate

REGISTRY = Registry.load()
STORE = CorpusStore()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
INTEGRATED = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
IMPORTER = PROFILES['ALCOHOL_GROUP_TUBORG_TYPE']
BOTTLER = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
_CACHE = {}


def obligations(regulation_id):
    if regulation_id not in _CACHE:
        _CACHE[regulation_id] = extract_regulation(regulation_id, REGISTRY, STORE)[1]
    return _CACHE[regulation_id]


def obligation(regulation_id, ref, group=None):
    found = [o for o in obligations(regulation_id) if o.provision_ref == ref and (group is None or o.group == group)]
    assert len(found) == 1, (ref, [o.group for o in found])
    return found[0]


def routed(regulation_id, ref, profile, group=None):
    decisions, _ = route(obligation(regulation_id, ref, group), profile, REGISTRY, STORE)
    return {d.target_id: d for d in decisions}


def status(decisions):
    return {k: d.status for k, d in decisions.items()}


class RegulationScopeTests(unittest.TestCase):
    def scope(self, regulation_id):
        return regulation_scope(regulation_frames(STORE.sections(regulation_id), regulation_id),
                                REGISTRY.regulations[regulation_id], REGISTRY.vocabulary)

    def test_the_scope_article_names_what_a_communique_covers_and_leaves_out(self):
        energy = self.scope('TR:TEBLIG:TGK_ENERJI_ICECEKLERI')         # "enerji içeceklerini kapsar, sporcu içeceklerini kapsamaz"
        self.assertEqual((energy.product_classes, energy.excluded_product_classes, energy.basis),
                         (['ENERGY_DRINK'], ['SPORTS_DRINK'], 'SCOPE_ARTICLE'))
        beer = self.scope('TR:TEBLIG:TGK_BIRA')                        # "birayı kapsar. Malt içeceğini kapsamaz."
        self.assertEqual(beer.excluded_product_classes, ['MALT_BEVERAGE'])
        self.assertIn('BEER', beer.product_classes)
        water = self.scope('TR:YONETMELIK:INSANI_TUKETIM_SULAR')
        self.assertEqual((water.product_classes, water.excluded_product_classes), (['PACKAGED_WATER'], ['NATURAL_MINERAL_WATER']))
        self.assertEqual(self.scope('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM').alcohol_scope, 'ALCOHOLIC')

    def test_a_horizontal_text_has_no_product_scope(self):
        waste = self.scope('TR:YONETMELIK:AMBALAJ_ATIKLARI')
        self.assertEqual((waste.product_classes, waste.alcohol_scope, waste.basis), ([], 'ANY', 'NONE'))


class ScopeDerivationTests(unittest.TestCase):
    def test_every_extracted_scope_is_a_valid_resolved_scope_that_quotes_its_clause(self):
        for regulation_id in ('TR:KANUN:4250', 'TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'TR:YONETMELIK:AMBALAJ_ATIKLARI',
                              'TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER', 'TR:YONETMELIK:TGK_ETIKETLEME'):
            found = obligations(regulation_id)
            self.assertTrue(found)
            ids = [o.scope.scope_id for o in found]
            self.assertEqual(len(ids), len(set(ids)), regulation_id)
            for item in found:
                self.assertIsInstance(item.scope, ObligationScope)
                self.assertEqual((item.scope.origin, item.scope.provision_status), ('EXTRACTED', 'RESOLVED'))
                self.assertEqual(item.scope.provision_ref, item.provision_ref)
                self.assertEqual(item.scope.quote, item.frame.text)
                self.assertTrue(ground(item, STORE)['grounded'], item.provision_ref)

    def test_an_impersonal_advertising_ban_lands_on_the_activity(self):
        ban = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.1')
        self.assertEqual((ban.basis, ban.scope.level, ban.scope.alcohol_scope), ('GOVERNING_ACTIVITY', 'ACTIVITY', 'ALCOHOLIC'))
        self.assertEqual(ban.scope.activity_classes, ['ADVERTISING', 'DIGITAL_MARKETING', 'PRODUCT_PROMOTION'])
        self.assertIn('EXCEPTION_UNEVALUATED', ban.flags)                # the trade-fair sentence: recorded, not decided
        self.assertEqual([e['effect'] for e in ban.scope.exceptions], ['PERMITS'])

    def test_a_named_addressee_lands_on_the_entities_that_do_what_it_names(self):
        gifts = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.2')
        self.assertEqual((gifts.basis, gifts.scope.level), ('ACTOR_EXPLICIT', 'LEGAL_ENTITY'))
        self.assertTrue({'PRODUCTION', 'IMPORT', 'DISTRIBUTION'} <= set(gifts.scope.activity_classes))
        waste = obligation('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 10/f.1/b.ç')
        self.assertEqual(waste.scope.activity_classes, ['PLACING_ON_MARKET', 'PRODUCTION', 'BOTTLING', 'IMPORT'])
        makers = obligation('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 9/f.1/b.ç')
        self.assertEqual((makers.scope.entity_classes, makers.scope.activity_classes), (['PACKAGING_PRODUCER'], []))

    def test_the_kind_of_sale_the_clause_names_is_the_kind_it_binds(self):
        night = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.5/c.3')      # "... perakende olarak satılamaz"
        self.assertEqual(night.scope.activity_classes, ['RETAIL_SALE'])
        minors = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.3')         # "... satılamaz veya sunulamaz"
        self.assertEqual(set(minors.scope.activity_classes), {'WHOLESALE', 'RETAIL_SALE', 'ON_PREMISE_SERVICE', 'ECOMMERCE_SALE'})

    def test_employment_is_the_act_and_the_activities_are_where_it_is_forbidden(self):
        minors = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.4/c.1')
        self.assertEqual(minors.group, 'EMPLOYMENT')
        self.assertIn('PRODUCTION', minors.scope.activity_classes)
        self.assertIn('RETAIL_SALE', minors.scope.activity_classes)

    def test_a_composition_limit_lands_on_the_product_class_of_the_clause(self):
        caffeine = obligation('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 5/f.1/b.b')
        self.assertEqual((caffeine.basis, caffeine.scope.level, caffeine.scope.product_classes),
                         ('PRODUCT_PROPERTY', 'PRODUCT', ['ENERGY_DRINK']))
        warning = obligation('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 12/f.1/b.b')   # "Bu Tebliğ kapsamındaki ürünlerin ..."
        self.assertEqual((warning.scope.product_classes, warning.scope.excluded_product_classes), (['ENERGY_DRINK'], ['SPORTS_DRINK']))
        self.assertIn('PRODUCT_FROM_REGULATION_SCOPE', warning.flags)

    def test_one_sentence_with_two_acts_gives_two_scopes(self):
        groups = {o.group: o for o in obligations('TR:TEBLIG:TGK_ENERJI_ICECEKLERI') if o.provision_ref == 'Tebliğ 23706 md. 11/f.2'}
        self.assertEqual(set(groups), {'MARKETING', 'SUPPLY'})             # "piyasaya arz edilmez ve reklamı yapılmaz"
        self.assertEqual((groups['MARKETING'].scope.level, groups['SUPPLY'].scope.level), ('ACTIVITY', 'LEGAL_ENTITY'))
        self.assertEqual(groups['SUPPLY'].scope.activity_classes, ['PLACING_ON_MARKET'])

    def test_a_list_item_is_addressed_by_the_lead_in_of_its_fikra(self):
        places = obligation('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 10/f.3/b.c')   # "c) ... stadyum ..."
        self.assertEqual((places.kind, places.group, places.scope.alcohol_scope), ('PROHIBITION', 'SALES', 'ALCOHOLIC'))
        label = obligation('TR:YONETMELIK:TGK_ETIKETLEME', 'Yönetmelik 23282 md. 9/f.1/b.f')     # "f) Özel muhafaza ... koşulları."
        self.assertEqual((label.scope.level, label.scope.activity_classes), ('PRODUCT', []))
        self.assertIn('LEVEL_FROM_CATALOGUE_DEFAULT', label.flags)

    def test_tobacco_only_clauses_are_outside_the_beverage_packs(self):
        refs = {o.provision_ref for o in obligations('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM')}
        self.assertNotIn('Yönetmelik 14646 md. 5/f.1/b.a', refs)           # "Tütün mamullerinin perakende satışını ..."
        self.assertIn('Yönetmelik 14646 md. 5/f.1/b.b', refs)              # "Alkollü içkilerin perakende satışını ..."

    def test_a_clause_the_rules_cannot_address_is_unclear_not_guessed(self):
        tv = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.7')       # series and films on television
        self.assertEqual((tv.basis, tv.scope.scope_status), ('UNCLEAR', 'UNCLEAR'))
        brands = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.9/c.1')   # brand signs of alcoholic and of other drinks: both sides
        self.assertEqual((brands.scope.alcohol_scope, brands.scope.product_classes), ('ANY', []))

    def test_a_drink_made_from_an_alcoholic_one_is_the_product_not_both_categories(self):
        # "alkollü içki kategorisindeki ürünlerin işlenmesi sonucunda, elde edilen alkolsüz içkilerde; ... yazılır": the
        # clause names the alcoholic category as what the product is made from. The lexicon maps the phrase to the one
        # de-alcoholised class of the vocabulary (a developer's mapping, flagged in the lexicon note).
        derived = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.10')
        self.assertEqual((derived.basis, derived.scope.level, derived.scope.product_classes), ('PRODUCT_PROPERTY', 'PRODUCT', ['ALCOHOL_FREE_BEER']))
        self.assertNotIn('PRODUCT_SCOPE_MIXED', derived.flags)
        decisions = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.10', INTEGRATED)
        self.assertEqual((decisions['ALC-INT-AF'].status, decisions['ALC-INT-LAGER'].status), ('APPLIES', 'DOES_NOT_APPLY'))
        by_law = obligation('TR:YONETMELIK:ALKOL_IC_DIS_TICARET', 'Yönetmelik 6203 md. 13/f.13/c.1')
        self.assertEqual(by_law.scope.product_classes, ['ALCOHOL_FREE_BEER'])


class RoutingContrastTests(unittest.TestCase):
    def test_the_advertising_ban_reaches_alcohol_advertising_and_nothing_of_the_soft_drink_group(self):
        ref = 'Kanun 4250 md. 6/f.1/c.1'
        brewer = routed('TR:KANUN:4250', ref, INTEGRATED)
        self.assertEqual(status(brewer), {'ALC-INT-ACT-BEER-ADS': 'APPLIES', 'ALC-INT-ACT-SPONSORSHIP': 'DOES_NOT_APPLY',
                                          'ALC-INT-ACT-AF-ADS': 'DOES_NOT_APPLY', 'ALC-INT-ACT-TRADE-PROMO': 'DOES_NOT_APPLY',
                                          'ALC-INT-ACT-DIGITAL': 'APPLIES', 'ALC-INT-ACT-AF-BRAND': 'DOES_NOT_APPLY',
                                          'ALC-INT-ACT-FESTIVAL': 'DOES_NOT_APPLY'})
        self.assertEqual(brewer['ALC-INT-ACT-AF-ADS'].reason_codes, ['ALCOHOL_SCOPE_MISMATCH'])       # 0.3 % vol
        self.assertIn('EXCEPTION_NOT_EVALUATED', brewer['ALC-INT-ACT-BEER-ADS'].reason_codes)
        self.assertEqual(set(status(routed('TR:KANUN:4250', ref, BOTTLER)).values()), {'DOES_NOT_APPLY'})

    def test_the_night_sale_ban_reaches_the_retailer_and_not_the_wholesaler(self):
        decisions = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.5/c.3', INTEGRATED)
        self.assertEqual(decisions['ALC-INT-SALES'].status, 'PARTIAL')             # beer yes, the alcohol-free beer no
        self.assertEqual(set(decisions['ALC-INT-SALES'].applies_to_products), {'ALC-INT-LAGER', 'ALC-INT-DARK', 'ALC-INT-LOW'})
        self.assertEqual((decisions['ALC-INT-DISTRIBUTION'].status, decisions['ALC-INT-DISTRIBUTION'].reason_codes),
                         ('DOES_NOT_APPLY', ['ACTIVITY_MISMATCH']))
        self.assertEqual(decisions['ALC-INT-EXPORT'].status, 'UNKNOWN')            # an incomplete profile is not a NO

    def test_the_warning_label_duty_skips_the_export_beer_by_the_clause_exception(self):
        decisions = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.8/c.1', INTEGRATED)
        self.assertEqual(status(decisions), {'ALC-INT-LAGER': 'APPLIES', 'ALC-INT-DARK': 'APPLIES', 'ALC-INT-LOW': 'APPLIES',
                                             'ALC-INT-AF': 'DOES_NOT_APPLY', 'ALC-INT-LAGER-EXPORT': 'DOES_NOT_APPLY'})
        self.assertEqual(decisions['ALC-INT-LAGER-EXPORT'].reason_codes, ['EXCEPTION_APPLIES'])
        self.assertEqual(decisions['ALC-INT-AF'].reason_codes, ['ALCOHOL_SCOPE_MISMATCH'])
        # The importer's wine does not say whether it is made for export: the duty stays, the exception is flagged.
        wine = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.8/c.1', IMPORTER)['ALC-IMP-WINE']
        self.assertEqual(wine.status, 'APPLIES')
        self.assertIn('EXCEPTION_POSSIBLE', wine.reason_codes)
        self.assertTrue(wine.review_required)

    def test_the_label_strength_rule_follows_the_stated_strength_and_stops_at_the_threshold(self):
        ref = 'Yönetmelik 23282 md. 9/f.1/b.i'                                    # "hacmen % 1,2’den fazla alkol içeren içeceklerde"
        brewer = routed('TR:YONETMELIK:TGK_ETIKETLEME', ref, INTEGRATED)
        self.assertEqual(status(brewer), {'ALC-INT-LAGER': 'APPLIES', 'ALC-INT-DARK': 'APPLIES', 'ALC-INT-LOW': 'DOES_NOT_APPLY',
                                          'ALC-INT-AF': 'DOES_NOT_APPLY', 'ALC-INT-LAGER-EXPORT': 'APPLIES'})
        self.assertEqual(brewer['ALC-INT-LOW'].reason_codes, ['CONDITION_NOT_MET'])   # exactly 1.2 is not "more than 1.2"
        importer = routed('TR:YONETMELIK:TGK_ETIKETLEME', ref, IMPORTER)
        self.assertEqual((importer['ALC-IMP-WINE'].status, importer['ALC-IMP-WINE'].reason_codes), ('UNKNOWN', ['PROFILE_INCOMPLETE']))
        self.assertEqual(importer['ALC-IMP-WHISKY'].status, 'APPLIES')
        self.assertEqual(set(status(routed('TR:YONETMELIK:TGK_ETIKETLEME', ref, BOTTLER)).values()) - {'UNKNOWN'}, {'DOES_NOT_APPLY'})

    def test_the_caffeine_limit_reaches_the_energy_drinks_only(self):
        decisions = routed('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 5/f.1/b.b', BOTTLER)
        self.assertEqual({k for k, d in decisions.items() if d.status == 'APPLIES'}, {'NONALC-ENERGY', 'NONALC-ENERGY-MAX'})
        self.assertEqual(decisions['NONALC-COLA'].reason_codes, ['PRODUCT_MISMATCH'])
        self.assertEqual(set(status(routed('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 5/f.1/b.b', INTEGRATED)).values()),
                         {'DOES_NOT_APPLY'})

    def test_the_caffeine_label_rule_needs_the_caffeine_content(self):
        decisions = routed('TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER', 'Tebliğ 11376 md. 11/f.1/b.h/c.1', BOTTLER)
        self.assertEqual(decisions['NONALC-COLA'].status, 'APPLIES')                # 96 mg/L stated
        self.assertIn('CONDITION_MET', decisions['NONALC-COLA'].reason_codes)
        self.assertEqual(decisions['NONALC-ICED-TEA'].status, 'UNKNOWN')            # caffeine not stated
        self.assertEqual(decisions['NONALC-WATER'].status, 'DOES_NOT_APPLY')        # not a product of this communiqué
        self.assertEqual(decisions['NONALC-ENERGY'].status, 'DOES_NOT_APPLY')

    def test_packaging_duties_reach_the_party_that_places_on_the_market_in_both_sectors(self):
        ref = 'Yönetmelik 38745 md. 10/f.1/b.ç'
        soft, beer = routed('TR:YONETMELIK:AMBALAJ_ATIKLARI', ref, BOTTLER), routed('TR:YONETMELIK:AMBALAJ_ATIKLARI', ref, INTEGRATED)
        self.assertEqual((soft['NONALC-BOTTLING'].status, soft['NONALC-DISTRIBUTION'].status), ('APPLIES', 'DOES_NOT_APPLY'))
        self.assertEqual((beer['ALC-INT-BREWING'].status, beer['ALC-INT-DISTRIBUTION'].status), ('APPLIES', 'DOES_NOT_APPLY'))
        makers = routed('TR:YONETMELIK:AMBALAJ_ATIKLARI', 'Yönetmelik 38745 md. 9/f.1/b.ç', BOTTLER)
        self.assertEqual(makers['NONALC-BOTTLING'].reason_codes, ['ENTITY_MISMATCH'])   # a bottler is not a packaging producer

    def test_the_energy_drink_marketing_ban_lands_on_the_energy_drink_advertising(self):
        decisions = routed('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 11/f.2', BOTTLER, 'MARKETING')
        self.assertEqual(decisions['NONALC-ACT-ENERGY-ONLINE-ADS'].status, 'APPLIES')
        self.assertEqual(decisions['NONALC-ACT-COLA-ADS'].status, 'DOES_NOT_APPLY')
        placing = routed('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 11/f.2', BOTTLER, 'SUPPLY')
        self.assertEqual((placing['NONALC-BOTTLING'].status, placing['NONALC-DISTRIBUTION'].status), ('PARTIAL', 'DOES_NOT_APPLY'))

    def test_an_unclear_clause_answers_unknown_on_every_target(self):
        decisions = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.7', INTEGRATED)
        self.assertEqual({(d.status, tuple(d.reason_codes)) for d in decisions.values()}, {('UNKNOWN', ('REGULATORY_SCOPE_UNCLEAR',))})

    def test_a_completeness_flip_turns_a_no_into_unknown_never_the_reverse(self):
        data = INTEGRATED.model_dump()
        next(e for e in data['legal_entities'] if e['entity_id'] == 'ALC-INT-DISTRIBUTION')['profile_complete'] = False
        from regchain.tr.profile import load_profile
        open_profile = load_profile(data, REGISTRY.vocabulary)
        before = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.5/c.3', INTEGRATED)['ALC-INT-DISTRIBUTION']
        after = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.5/c.3', open_profile)['ALC-INT-DISTRIBUTION']
        self.assertEqual((before.status, after.status), ('DOES_NOT_APPLY', 'UNKNOWN'))


class DecisionAuditTests(unittest.TestCase):
    def test_an_extracted_decision_cites_the_clause_and_keeps_the_stages_apart(self):
        decision = routed('TR:KANUN:4250', 'Kanun 4250 md. 6/f.8/c.1', INTEGRATED)['ALC-INT-LAGER']
        self.assertEqual((decision.basis, decision.provision_status, decision.provision_ref),
                         ('EXTRACTED', 'RESOLVED', 'Kanun 4250 md. 6/f.8/c.1'))
        self.assertTrue(decision.creates_obligation)
        self.assertIsNone(decision.audit.raw)                                    # no model was asked
        self.assertEqual(decision.audit.parsed['kind'], 'OBLIGATION')
        self.assertEqual(decision.audit.parsed['exceptions'], ['İhraç amaçlı üretilenler hariç olmak üzere'])
        self.assertTrue(decision.audit.validator['grounded'])
        self.assertEqual(decision.audit.final['status'], 'APPLIES')

    def test_a_quote_that_is_not_in_the_stored_text_fails_the_validator_and_decides_nothing(self):
        original = obligation('TR:KANUN:4250', 'Kanun 4250 md. 6/f.8/c.1')
        frame = original.frame.model_copy(update={'text': original.frame.text.replace('uyarı mesajları', 'uyarı etiketleri')})
        tampered = original.model_copy(update={'frame': frame})
        self.assertFalse(ground(tampered, STORE)['grounded'])
        decisions, rollups = route(tampered, INTEGRATED, REGISTRY, STORE)
        self.assertEqual({(d.status, tuple(d.reason_codes)) for d in decisions + rollups}, {('UNKNOWN', ('GROUNDING_FAILED',))})
        self.assertTrue(all(d.review_required for d in decisions))


class PredicateTests(unittest.TestCase):
    def test_predicates_are_three_valued(self):
        lager, low, wine = INTEGRATED.product('ALC-INT-LAGER'), INTEGRATED.product('ALC-INT-LOW'), IMPORTER.product('ALC-IMP-WINE')
        over = {'fact': 'product.abv_percent', 'op': 'gt', 'value': 1.2}
        self.assertEqual([evaluate_predicate(over, p) for p in (lager, low, wine)], ['YES', 'NO', 'UNKNOWN'])
        export = {'fact': 'product.export_only', 'op': 'eq', 'value': True}
        self.assertEqual([evaluate_predicate(export, p) for p in (lager, INTEGRATED.product('ALC-INT-LAGER-EXPORT'), wine)],
                         ['NO', 'YES', 'UNKNOWN'])
        self.assertEqual(evaluate_predicate({'fact': 'product.class', 'op': 'in', 'value': ['BEER']}, lager), 'YES')
        self.assertEqual(evaluate_predicate({'fact': 'entity.turnover', 'op': 'gt', 'value': 1}, lager), 'UNKNOWN')

    def test_conditions_and_quotes_belong_to_extracted_scopes_only(self):
        base = dict(scope_id='SAMPLE', regulation_id='TR:KANUN:4250', topic='X', description_tr='x', level='PRODUCT',
                    provision_ref='Kanun 4250 md. 6', provision_status='RESOLVED')
        with self.assertRaises(ValueError):
            ObligationScope.model_validate(base | {'conditions': [{'fact': 'product.abv_percent', 'op': 'gt', 'value': 1}]})
        with self.assertRaises(ValueError):
            ObligationScope.model_validate(base | {'origin': 'EXTRACTED'})        # an extracted scope quotes its clause
        ObligationScope.model_validate(base | {'origin': 'EXTRACTED', 'quote': 'x'})


class ChannelAndPackagingTests(unittest.TestCase):
    """A duty written for a sales channel binds the entities that sell through it; a product duty written for a packaging
    reaches the products in that packaging (sales_channels / product_attributes of the scope, 2 October 2026)."""

    def test_a_duty_about_small_bottles_in_grocers_names_its_channels(self):
        from regchain.tr.extraction import extract_regulation
        frames, obligations = extract_regulation('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', REGISTRY, STORE, articles=['22'])
        small = next(o for o in obligations if o.provision_ref == 'Yönetmelik 14646 md. 22/f.4/c.1')
        self.assertEqual(small.scope.sales_channels, ['TRADITIONAL_RETAIL', 'MODERN_RETAIL'])
        self.assertIn('SALES_CHANNEL_NAMED', small.flags)
        # "toptan satış belgesi", "Kurumun internet sitesinde": a channel word in a licence's name or a web address is no channel
        for obligation in obligations:
            if obligation.provision_ref in ('Yönetmelik 14646 md. 22/f.1/c.1', 'Yönetmelik 14646 md. 22/f.2/c.1'):
                self.assertEqual(obligation.scope.sales_channels, [], obligation.provision_ref)

    def test_a_channel_duty_binds_the_entities_that_sell_through_it(self):
        from regchain.tr.extraction import extract_regulation, route
        profile = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
        obligations = extract_regulation('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', REGISTRY, STORE, articles=['22'])[1]
        small = next(o for o in obligations if o.provision_ref == 'Yönetmelik 14646 md. 22/f.4/c.1')
        decisions = {d.target_id: d for d in route(small, profile, REGISTRY, STORE)[0]}
        selling = [e for e in profile.legal_entities if set(e.sales_channels) & {'TRADITIONAL_RETAIL', 'MODERN_RETAIL'}]
        self.assertTrue(selling)
        for entity in selling:                                  # PARTIAL where the product list is only partly in scope
            self.assertIn(decisions[entity.entity_id].status, ('APPLIES', 'PARTIAL'), entity.entity_id)
            self.assertIn('SALES_CHANNEL_MATCH', decisions[entity.entity_id].reason_codes)
            self.assertIn('SALES_CHANNEL', [g['gate'] for g in decisions[entity.entity_id].gates])
        silent = [e for e in profile.legal_entities if not e.sales_channels and e.profile_complete and e.entity_id in decisions]
        for entity in silent:
            self.assertEqual(decisions[entity.entity_id].status, 'DOES_NOT_APPLY', entity.entity_id)
            self.assertIn('SALES_CHANNEL_MISMATCH', decisions[entity.entity_id].reason_codes)
        # channels not stated: not NO
        open_profile = profile.model_copy(update={'legal_entities': [e.model_copy(update={'sales_channels': [], 'profile_complete': False})
                                                                      for e in profile.legal_entities]})
        wanted = {e.entity_id for e in selling}
        for decision in route(small, open_profile, REGISTRY, STORE)[0]:
            if decision.target_id in wanted:
                self.assertEqual(decision.status, 'UNKNOWN', decision.target_id)
                self.assertIn(('SALES_CHANNEL', 'UNDETERMINED'), [(g['gate'], g['status']) for g in decision.gates])

    def test_a_product_duty_about_glass_packaging_reaches_the_products_in_glass(self):
        from regchain.tr.extraction import extract_regulation, route
        obligations = extract_regulation('TR:YONETMELIK:INSANI_TUKETIM_SULAR', REGISTRY, STORE, articles=['34'])[1]
        glass = next(o for o in obligations if o.provision_ref == 'Yönetmelik 7510 md. 34/f.4/b.c')
        self.assertEqual((glass.scope.level, glass.scope.product_attributes), ('PRODUCT', ['GLASS_PACKAGING']))
        profile = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
        water = next(p for p in profile.products if p.product_class == 'PACKAGED_WATER')
        in_glass = water.model_copy(update={'tags': [*water.tags, 'GLASS_PACKAGING'], 'tags_complete': True})
        in_pet = water.model_copy(update={'tags': [t for t in water.tags if t != 'GLASS_PACKAGING'] + ['PET_PACKAGING'], 'tags_complete': True})
        unknown = water.model_copy(update={'tags': [], 'tags_complete': False})
        for product, expected in ((in_glass, 'APPLIES'), (in_pet, 'DOES_NOT_APPLY'), (unknown, 'UNKNOWN')):
            trial = profile.model_copy(update={'products': [product if p.product_id == water.product_id else p for p in profile.products]})
            decision = next(d for d in route(glass, trial, REGISTRY, STORE)[0] if d.target_id == water.product_id)
            self.assertEqual(decision.status, expected, expected)
        # a duty of the party that names a packaging is not narrowed by it
        waste = extract_regulation('TR:YONETMELIK:AMBALAJ_ATIKLARI', REGISTRY, STORE, articles=['9'])[1]
        deposit = next(o for o in waste if o.provision_ref == 'Yönetmelik 38745 md. 9/f.1/b.ç')
        self.assertEqual((deposit.scope.level, deposit.scope.product_attributes), ('LEGAL_ENTITY', []))


if __name__ == '__main__':
    unittest.main()

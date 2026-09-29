"""Alcohol vs non-alcohol contrast on the representative pilot profiles (Cardaman TR, beverage milestone 4).

The profiles are synthetic: they reproduce the shape of an integrated brewer, a brewer that also
imports spirits and wine, and a soft-drink bottler with water, juice and energy lines. Nothing in
them is a real company's data, and the routing never reads the representative type."""
import json
import unittest

from regchain.pilot.applicability import company_kinds
from regchain.tr.packs import Registry
from regchain.tr.profile import PILOT_PROFILES, PolicyScope, load_pilot_profiles, policy_covers, to_company
from regchain.tr.routing import evaluate_scope, resolve, rollup

REGISTRY = Registry.load()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
INTEGRATED = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
IMPORTER = PROFILES['ALCOHOL_GROUP_TUBORG_TYPE']
BOTTLER = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']


def targets(profile, scope_id):
    return {d.target_id: d for d in evaluate_scope(REGISTRY.scopes[scope_id], profile, REGISTRY)}


def rolled(profile, scope_id):
    scope = REGISTRY.scopes[scope_id]
    return {(d.level, d.target_id): d for d in rollup(scope, evaluate_scope(scope, profile, REGISTRY), profile, REGISTRY)}


def status(decisions):
    return {k: d.status for k, d in decisions.items()}


class PilotFixtureTests(unittest.TestCase):
    def test_three_synthetic_representative_profiles_load(self):
        self.assertEqual(set(PROFILES), {'ALCOHOL_GROUP_EFES_TYPE', 'ALCOHOL_GROUP_TUBORG_TYPE',
                                         'NON_ALCOHOL_GROUP_COCA_COLA_TYPE'})
        for profile in PROFILES.values():
            self.assertTrue(profile.synthetic)
            self.assertIn('Sentetik temsili test profili', profile.disclaimer)
            self.assertIn('Gerçek bir şirketin', profile.disclaimer)

    def test_each_group_has_production_distribution_and_sales_entities_facilities_and_products(self):
        for kind, profile in PROFILES.items():
            classes = [set(e.entity_classes) for e in profile.legal_entities]
            self.assertTrue(any(c & {'PRODUCER', 'BOTTLER'} for c in classes), kind)
            self.assertTrue(any('DISTRIBUTOR' in c for c in classes), kind)
            self.assertTrue(any(c & {'WHOLESALER', 'RETAILER', 'ONLINE_SELLER'} for c in classes), kind)
            self.assertGreaterEqual(len(profile.facilities), 3, kind)
            self.assertGreaterEqual(len(profile.products), 3, kind)

    def test_no_real_brand_name_appears_in_the_profile_data(self):
        for path in PILOT_PROFILES.glob('*.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            data.pop('representative_type')
            text = json.dumps(data, ensure_ascii=False).lower()
            for brand in ('efes', 'tuborg', 'coca', 'pepsi', 'carlsberg', 'fanta', 'sprite'):
                self.assertNotIn(brand, text, f'{brand} in {path.name}')

    def test_every_entity_projects_to_the_legacy_company_without_triggering_the_aml_gates(self):
        for profile in PROFILES.values():
            for entity in profile.legal_entities:
                self.assertEqual(company_kinds(to_company(profile, entity.entity_id, REGISTRY.vocabulary)), [],
                                 entity.entity_id)


class PackSelectionContrastTests(unittest.TestCase):
    def selections(self, profile):
        return {s.pack_id: s.status for s in resolve(profile, REGISTRY).selections}

    def test_the_integrated_brewer_with_an_alcohol_free_line_falls_under_both_packs(self):
        self.assertEqual(self.selections(INTEGRATED), {'BEVERAGE_ALCOHOL_TR': 'SELECTED', 'BEVERAGE_NON_ALCOHOL_TR': 'SELECTED'})

    def test_the_importer_is_alcohol_only_and_the_bottler_non_alcohol_only(self):
        self.assertEqual(self.selections(IMPORTER), {'BEVERAGE_ALCOHOL_TR': 'SELECTED', 'BEVERAGE_NON_ALCOHOL_TR': 'NOT_SELECTED'})
        self.assertEqual(self.selections(BOTTLER), {'BEVERAGE_ALCOHOL_TR': 'NOT_SELECTED', 'BEVERAGE_NON_ALCOHOL_TR': 'SELECTED'})

    def test_the_bottler_gets_no_alcohol_decision_and_sees_the_alcohol_scopes_as_out_of_scope(self):
        result = resolve(BOTTLER, REGISTRY, include_unselected=True)
        self.assertFalse([d for d in result.decisions if d.scope_id.startswith('ALC_')])
        out = {d.scope_id: d.reason_codes for d in result.out_of_scope}
        self.assertEqual(out['ALC_ADVERTISING_PROMOTION'], ['PACK_NOT_SELECTED'])
        self.assertEqual(out['ALC_SALES_PRESENTATION'], ['PACK_NOT_SELECTED'])


class AlcoholAdvertisingContrastTests(unittest.TestCase):
    def test_alcohol_advertising_lands_on_the_alcohol_activities_only(self):
        decisions = targets(INTEGRATED, 'ALC_ADVERTISING_PROMOTION')
        self.assertEqual(status(decisions), {'ALC-INT-ACT-BEER-ADS': 'APPLIES', 'ALC-INT-ACT-SPONSORSHIP': 'APPLIES',
                                             'ALC-INT-ACT-AF-ADS': 'DOES_NOT_APPLY', 'ALC-INT-ACT-TRADE-PROMO': 'APPLIES'})
        self.assertEqual(decisions['ALC-INT-ACT-AF-ADS'].reason_codes, ['ALCOHOL_SCOPE_MISMATCH'])

    def test_the_alcohol_group_is_partially_in_scope_and_the_non_alcohol_group_not_at_all(self):
        alcohol = rolled(INTEGRATED, 'ALC_ADVERTISING_PROMOTION')
        self.assertEqual(alcohol[('GROUP', 'GRP-ALC-INT')].status, 'PARTIAL')
        self.assertEqual(alcohol[('LEGAL_ENTITY', 'ALC-INT-HOLDING')].status, 'PARTIAL')
        self.assertEqual(alcohol[('LEGAL_ENTITY', 'ALC-INT-DISTRIBUTION')].status, 'APPLIES')
        self.assertEqual(alcohol[('LEGAL_ENTITY', 'ALC-INT-BREWING')].reason_codes, ['NO_TARGET_AT_LEVEL'])
        self.assertEqual(alcohol[('LEGAL_ENTITY', 'ALC-INT-EXPORT')].status, 'UNKNOWN')
        soft = rolled(BOTTLER, 'ALC_ADVERTISING_PROMOTION')
        self.assertEqual(soft[('GROUP', 'GRP-NONALC')].status, 'DOES_NOT_APPLY')
        for decision in targets(BOTTLER, 'ALC_ADVERTISING_PROMOTION').values():
            self.assertEqual(decision.reason_codes, ['ALCOHOL_SCOPE_MISMATCH'])

    def test_the_general_advertising_rule_still_reaches_the_non_alcohol_advertising(self):
        decisions = targets(BOTTLER, 'COMMERCIAL_ADVERTISING_GENERAL')
        self.assertEqual(status(decisions), {'NONALC-ACT-COLA-ADS': 'APPLIES', 'NONALC-ACT-SPONSORSHIP': 'DOES_NOT_APPLY',
                                             'NONALC-ACT-ENERGY-ONLINE-ADS': 'APPLIES'})
        self.assertEqual(targets(INTEGRATED, 'COMMERCIAL_ADVERTISING_GENERAL')['ALC-INT-ACT-AF-ADS'].status, 'APPLIES')


class AlcoholSalesContrastTests(unittest.TestCase):
    def test_the_alcohol_sales_rule_lands_on_the_selling_entity_not_the_producer(self):
        decisions = targets(IMPORTER, 'ALC_SALES_PRESENTATION')
        self.assertEqual((decisions['ALC-IMP-TRADING'].status, decisions['ALC-IMP-TRADING'].reason_codes),
                         ('APPLIES', ['ACTIVITY_MATCH', 'ALCOHOL_SCOPE_MATCH']))
        self.assertEqual((decisions['ALC-IMP-BREWING'].status, decisions['ALC-IMP-BREWING'].reason_codes),
                         ('DOES_NOT_APPLY', ['ACTIVITY_MISMATCH']))
        self.assertEqual((decisions['ALC-IMP-MARKETING'].status, decisions['ALC-IMP-MARKETING'].reason_codes),
                         ('UNKNOWN', ['PROFILE_INCOMPLETE']))

    def test_a_seller_of_alcoholic_and_alcohol_free_beer_is_partially_in_scope(self):
        decision = targets(INTEGRATED, 'ALC_SALES_PRESENTATION')['ALC-INT-DISTRIBUTION']
        self.assertEqual(decision.status, 'PARTIAL')
        self.assertEqual(set(decision.applies_to_products), {'ALC-INT-LAGER', 'ALC-INT-DARK'})

    def test_online_sale_is_a_distance_sale_for_both_but_an_alcohol_online_sale_only_for_alcohol(self):
        self.assertEqual(targets(BOTTLER, 'DISTANCE_SALES')['NONALC-ONLINE'].status, 'APPLIES')
        self.assertEqual(targets(BOTTLER, 'ALC_ONLINE_SALES')['NONALC-ONLINE'].reason_codes, ['ALCOHOL_SCOPE_MISMATCH'])
        self.assertEqual(targets(BOTTLER, 'DISTANCE_SALES')['NONALC-BOTTLING'].status, 'DOES_NOT_APPLY')
        self.assertEqual(targets(BOTTLER, 'DISTANCE_SALES')['NONALC-BRAND'].status, 'UNKNOWN')

    def test_permit_holder_duties_follow_the_held_licence(self):
        decisions = targets(IMPORTER, 'ALC_PERMIT_HOLDER_DUTIES')
        self.assertEqual(status(decisions), {'ALC-IMP-BREWING': 'APPLIES', 'ALC-IMP-TRADING': 'APPLIES',
                                             'ALC-IMP-MARKETING': 'UNKNOWN'})


class ProductContrastTests(unittest.TestCase):
    def test_the_energy_drink_rule_applies_to_the_energy_drink_and_not_to_the_water(self):
        decisions = targets(BOTTLER, 'ENERGY_DRINK_PRODUCT')
        self.assertEqual((decisions['NONALC-ENERGY'].status, decisions['NONALC-ENERGY'].reason_codes),
                         ('APPLIES', ['PRODUCT_MATCH', 'NON_ALCOHOL_SCOPE_MATCH']))
        self.assertEqual((decisions['NONALC-WATER'].status, decisions['NONALC-WATER'].reason_codes),
                         ('DOES_NOT_APPLY', ['PRODUCT_MISMATCH']))
        self.assertEqual(rolled(BOTTLER, 'ENERGY_DRINK_PRODUCT')[('GROUP', 'GRP-NONALC')].status, 'PARTIAL')

    def test_the_water_rule_applies_to_the_water_and_not_to_beer(self):
        self.assertEqual(targets(BOTTLER, 'PACKAGED_WATER_PRODUCT')['NONALC-WATER'].status, 'APPLIES')
        beer = targets(INTEGRATED, 'PACKAGED_WATER_PRODUCT')
        self.assertEqual(beer['ALC-INT-LAGER'].status, 'DOES_NOT_APPLY')
        self.assertIn('PRODUCT_MISMATCH', beer['ALC-INT-LAGER'].reason_codes)
        self.assertEqual(rolled(INTEGRATED, 'PACKAGED_WATER_PRODUCT')[('GROUP', 'GRP-ALC-INT')].status, 'DOES_NOT_APPLY')

    def test_the_alcohol_label_rule_covers_alcoholic_products_and_skips_the_alcohol_free_beer(self):
        decisions = targets(INTEGRATED, 'ALC_LABEL_WARNINGS')
        self.assertEqual(status(decisions), {'ALC-INT-LAGER': 'APPLIES', 'ALC-INT-DARK': 'APPLIES', 'ALC-INT-AF': 'DOES_NOT_APPLY'})
        self.assertEqual(status(targets(IMPORTER, 'ALC_LABEL_WARNINGS')),
                         {p.product_id: 'APPLIES' for p in IMPORTER.products})

    def test_general_food_labelling_reaches_every_product_of_both_sectors(self):
        for profile in PROFILES.values():
            self.assertEqual(set(status(targets(profile, 'FOOD_LABELLING_GENERAL')).values()), {'APPLIES'})

    def test_claims_follow_the_product_tags_and_stay_unknown_where_tags_are_incomplete(self):
        decisions = targets(BOTTLER, 'NUTRITION_HEALTH_CLAIMS')
        self.assertEqual(decisions['NONALC-COLA-ZERO'].status, 'APPLIES')
        self.assertEqual(decisions['NONALC-COLA'].status, 'DOES_NOT_APPLY')
        self.assertEqual(decisions['NONALC-JUICE'].status, 'UNKNOWN')


class FacilityContrastTests(unittest.TestCase):
    def test_the_food_facility_rule_lands_on_production_plants_not_on_warehouses(self):
        self.assertEqual(status(targets(BOTTLER, 'FOOD_PRODUCTION_FACILITY')),
                         {'NONALC-PLANT-1': 'APPLIES', 'NONALC-PLANT-2': 'APPLIES', 'NONALC-WATER-PLANT': 'APPLIES',
                          'NONALC-SPRING': 'DOES_NOT_APPLY', 'NONALC-WAREHOUSE': 'DOES_NOT_APPLY', 'NONALC-DC': 'DOES_NOT_APPLY'})
        self.assertEqual(status(targets(INTEGRATED, 'FOOD_PRODUCTION_FACILITY')),
                         {'ALC-INT-BREWERY-1': 'APPLIES', 'ALC-INT-BREWERY-2': 'APPLIES', 'ALC-INT-MALTING': 'APPLIES',
                          'ALC-INT-DC': 'DOES_NOT_APPLY', 'ALC-INT-HQ': 'DOES_NOT_APPLY'})

    def test_the_alcohol_facility_rule_reads_the_brewery_and_what_it_brews(self):
        decisions = targets(INTEGRATED, 'ALC_PRODUCTION_FACILITY')
        self.assertEqual(decisions['ALC-INT-BREWERY-1'].status, 'PARTIAL')
        self.assertEqual(set(decisions['ALC-INT-BREWERY-1'].applies_to_products), {'ALC-INT-LAGER', 'ALC-INT-DARK'})
        self.assertEqual(decisions['ALC-INT-BREWERY-2'].status, 'APPLIES')
        self.assertEqual(decisions['ALC-INT-MALTING'].status, 'DOES_NOT_APPLY')
        entity = rolled(INTEGRATED, 'ALC_PRODUCTION_FACILITY')
        self.assertEqual(entity[('LEGAL_ENTITY', 'ALC-INT-BREWING')].status, 'PARTIAL')
        self.assertEqual(entity[('LEGAL_ENTITY', 'ALC-INT-DISTRIBUTION')].status, 'DOES_NOT_APPLY')

    def test_the_water_facility_rule_reaches_the_water_plant_and_the_source_only(self):
        decisions = targets(BOTTLER, 'WATER_BOTTLING_FACILITY')
        self.assertEqual({k for k, d in decisions.items() if d.status == 'APPLIES'}, {'NONALC-WATER-PLANT', 'NONALC-SPRING'})


class UnclearScopeTests(unittest.TestCase):
    def test_scopes_whose_addressee_is_unread_stay_unknown_everywhere(self):
        for profile in PROFILES.values():
            for decision in targets(profile, 'EXCISE_TAX_BEVERAGES').values():
                self.assertEqual((decision.status, decision.reason_codes), ('UNKNOWN', ['REGULATORY_SCOPE_UNCLEAR']))
        self.assertEqual(set(status(targets(IMPORTER, 'ALC_SPIRIT_MONOPOLY')).values()), {'UNKNOWN'})

    def test_no_pilot_decision_is_grounded_beyond_pack_metadata_yet(self):
        for profile in PROFILES.values():
            result = resolve(profile, REGISTRY)
            self.assertTrue(result.decisions)
            for decision in result.decisions + result.rollups:
                self.assertEqual((decision.basis, decision.provision_status), ('PACK_METADATA', 'UNRESOLVED'))


class PolicyInheritanceContrastTests(unittest.TestCase):
    def test_a_group_marketing_policy_reaches_the_marketing_entities_only(self):
        policy = PolicyScope(policy_id='GROUP-MARKETING', owner_level='GROUP',
                             activity_classes=['ADVERTISING', 'SPONSORSHIP', 'PROMOTION'])
        covers = {e.entity_id: policy_covers(policy, INTEGRATED, 'LEGAL_ENTITY', e.entity_id)
                  for e in INTEGRATED.legal_entities}
        self.assertEqual(covers['ALC-INT-HOLDING'], {'covers': 'YES', 'reason': 'GROUP_POLICY_INHERITED'})
        self.assertEqual(covers['ALC-INT-DISTRIBUTION']['covers'], 'YES')
        self.assertEqual(covers['ALC-INT-BREWING'], {'covers': 'NO', 'reason': 'POLICY_ACTIVITY_MISMATCH'})
        self.assertEqual(covers['ALC-INT-EXPORT'], {'covers': 'UNKNOWN', 'reason': 'POLICY_ACTIVITY_UNSTATED'})

    def test_an_entity_policy_does_not_cover_the_entity_the_sales_duty_lands_on(self):
        production = PolicyScope(policy_id='BREWING-QA', owner_level='LEGAL_ENTITY', entity_id='ALC-INT-BREWING')
        sales = targets(INTEGRATED, 'ALC_SALES_PRESENTATION')
        subject = [k for k, d in sales.items() if d.status in ('APPLIES', 'PARTIAL')]
        self.assertEqual(subject, ['ALC-INT-DISTRIBUTION'])
        self.assertEqual(policy_covers(production, INTEGRATED, 'LEGAL_ENTITY', 'ALC-INT-DISTRIBUTION'),
                         {'covers': 'NO', 'reason': 'OTHER_ENTITY_POLICY'})
        self.assertEqual(policy_covers(production, INTEGRATED, 'FACILITY', 'ALC-INT-BREWERY-1'),
                         {'covers': 'YES', 'reason': 'ENTITY_POLICY_INHERITED'})
        self.assertEqual(policy_covers(production, INTEGRATED, 'GROUP', 'GRP-ALC-INT'),
                         {'covers': 'NO', 'reason': 'NARROWER_THAN_TARGET'})

    def test_a_non_inherited_group_policy_stays_at_the_group(self):
        policy = PolicyScope(policy_id='GROUP-CHARTER', owner_level='GROUP', inherits=False)
        self.assertEqual(policy_covers(policy, BOTTLER, 'GROUP', 'GRP-NONALC')['covers'], 'YES')
        self.assertEqual(policy_covers(policy, BOTTLER, 'LEGAL_ENTITY', 'NONALC-BOTTLING'),
                         {'covers': 'NO', 'reason': 'GROUP_POLICY_NOT_INHERITED'})


if __name__ == '__main__':
    unittest.main()

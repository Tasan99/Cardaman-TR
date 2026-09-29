"""Enterprise profile graph, its legacy projection and policy scopes (Cardaman TR, beverage milestone 2)."""
import copy
import unittest

from regchain.pilot.applicability import company_kinds
from regchain.pilot.schema import Company
from regchain.tr.packs import Registry
from regchain.tr.profile import (EnterpriseProfile, PolicyScope, ProfileError, load_profile, policy_covers, product_alcohol,
                                 to_company)

VOCABULARY = Registry.load().vocabulary


def sample():
    return {
        'format': 'cardaman-tr-enterprise-profile/1', 'profile_id': 'sample-group', 'version': '1', 'synthetic': True,
        'representative_type': 'TEST', 'disclaimer': 'Sentetik test profili.', 'jurisdiction': 'TR',
        'group': {'group_id': 'G', 'name': 'Örnek Grup'},
        'legal_entities': [
            {'entity_id': 'PROD', 'name': 'Örnek Üretim A.Ş.', 'entity_classes': ['PRODUCER'],
             'activity_classes': ['PRODUCTION', 'PLACING_ON_MARKET'], 'product_ids': ['P-BEER', 'P-WATER'],
             'licenses': [{'license_class': 'ALCOHOL_PRODUCTION_PERMIT', 'status': 'HELD', 'facility_id': 'F-BREW'}],
             'sales_channels': [], 'profile_complete': True},
            {'entity_id': 'SALES', 'name': 'Örnek Satış A.Ş.', 'entity_classes': ['WHOLESALER', 'RETAILER'],
             'activity_classes': ['WHOLESALE', 'RETAIL_SALE', 'ADVERTISING'], 'product_ids': ['P-BEER', 'P-WATER'],
             'licenses': [], 'sales_channels': ['MODERN_RETAIL'], 'profile_complete': False},
        ],
        'facilities': [
            {'facility_id': 'F-BREW', 'name': 'Tesis 1', 'entity_id': 'PROD', 'facility_classes': ['BREWERY'],
             'activity_classes': ['PRODUCTION'], 'product_ids': ['P-BEER'], 'activities_complete': True},
            {'facility_id': 'F-DEPOT', 'name': 'Depo', 'entity_id': 'SALES', 'facility_classes': ['WAREHOUSE'],
             'activity_classes': ['WAREHOUSING'], 'product_ids': ['P-BEER', 'P-WATER'], 'activities_complete': True},
        ],
        'products': [
            {'product_id': 'P-BEER', 'name': 'Örnek Bira', 'product_class': 'BEER', 'tags': ['GLASS_PACKAGING'], 'tags_complete': True},
            {'product_id': 'P-WATER', 'name': 'Örnek Su', 'product_class': 'PACKAGED_WATER', 'tags': [], 'tags_complete': False},
        ],
        'activities': [
            {'activity_id': 'A-ADV', 'entity_id': 'SALES', 'activity_class': 'ADVERTISING', 'product_ids': ['P-BEER']},
        ],
        'products_complete': True,
    }


class ProfileStructureTests(unittest.TestCase):
    def assert_rejected(self, mutate, needle):
        data = copy.deepcopy(sample())
        mutate(data)
        with self.assertRaises(ProfileError) as caught:
            load_profile(data, VOCABULARY)
        self.assertIn(needle, str(caught.exception))

    def test_the_five_levels_load(self):
        profile = load_profile(sample(), VOCABULARY)
        self.assertIsInstance(profile, EnterpriseProfile)
        self.assertEqual(profile.group.group_id, 'G')
        self.assertEqual([e.entity_id for e in profile.legal_entities], ['PROD', 'SALES'])
        self.assertEqual([f.facility_id for f in profile.facilities_of('PROD')], ['F-BREW'])
        self.assertEqual([a.activity_id for a in profile.activities_of('SALES')], ['A-ADV'])

    def test_ids_are_unique_across_the_graph(self):
        self.assert_rejected(lambda d: d['facilities'][1].update(facility_id='F-BREW'), 'F-BREW')
        self.assert_rejected(lambda d: d['products'][1].update(product_id='PROD'), 'PROD')

    def test_references_must_resolve_inside_the_group(self):
        self.assert_rejected(lambda d: d['facilities'][0].update(entity_id='NOPE'), 'NOPE')
        self.assert_rejected(lambda d: d['legal_entities'][0]['product_ids'].append('P-GHOST'), 'P-GHOST')
        self.assert_rejected(lambda d: d['activities'][0].update(entity_id='NOPE'), 'NOPE')

    def test_a_facility_holds_only_products_its_entity_handles(self):
        self.assert_rejected(lambda d: d['legal_entities'][0].update(product_ids=['P-WATER']), 'P-BEER')

    def test_an_activity_must_be_one_its_entity_carries_out_at_its_own_facility(self):
        self.assert_rejected(lambda d: d['activities'][0].update(activity_class='SPONSORSHIP'), 'SPONSORSHIP')
        self.assert_rejected(lambda d: d['activities'][0].update(facility_id='F-BREW'), 'F-BREW')

    def test_a_licence_can_name_only_its_own_entity_facility(self):
        self.assert_rejected(lambda d: d['legal_entities'][0]['licenses'][0].update(facility_id='F-DEPOT'), 'F-DEPOT')

    def test_a_synthetic_profile_says_so(self):
        self.assert_rejected(lambda d: d.update(disclaimer=''), 'disclaimer')

    def test_vocabulary_terms_are_checked(self):
        self.assert_rejected(lambda d: d['products'][0].update(product_class='LAGER'), 'LAGER')
        self.assert_rejected(lambda d: d['facilities'][0]['facility_classes'].append('BREWRY'), 'BREWRY')
        self.assert_rejected(lambda d: d['legal_entities'][1]['sales_channels'].append('TV'), 'TV')
        self.assert_rejected(lambda d: d['products'][0]['tags'].append('ORGANIC'), 'ORGANIC')

    def test_the_alcohol_category_follows_the_product_class(self):
        profile = load_profile(sample(), VOCABULARY)
        self.assertEqual(product_alcohol(profile.product('P-BEER'), VOCABULARY), 'ALCOHOLIC')
        self.assertEqual(product_alcohol(profile.product('P-WATER'), VOCABULARY), 'NON_ALCOHOLIC')

    def test_a_low_alcohol_class_without_abv_is_unknown_not_alcoholic_by_name(self):
        data = sample()
        data['products'].append({'product_id': 'P-LOW', 'name': 'Düşük alkollü bira', 'product_class': 'LOW_ALCOHOL_BEER'})
        data['legal_entities'][0]['product_ids'].append('P-LOW')
        profile = load_profile(data, VOCABULARY)
        self.assertEqual(product_alcohol(profile.product('P-LOW'), VOCABULARY), 'UNKNOWN')
        data['products'][-1]['attributes'] = {'abv_percent': {'value': 0.5, 'status': 'STATED'}}
        self.assertEqual(product_alcohol(load_profile(data, VOCABULARY).product('P-LOW'), VOCABULARY), 'ALCOHOLIC')
        data['products'][-1]['attributes'] = {'abv_percent': {'value': 0, 'status': 'STATED'}}
        self.assertEqual(product_alcohol(load_profile(data, VOCABULARY).product('P-LOW'), VOCABULARY), 'NON_ALCOHOLIC')

    def test_entity_and_product_and_facility_facts_do_not_leak(self):
        profile = load_profile(sample(), VOCABULARY)
        self.assertEqual(profile.entity('PROD').activity_classes, ['PRODUCTION', 'PLACING_ON_MARKET'])
        self.assertNotIn('PRODUCTION', profile.entity('SALES').activity_classes)
        self.assertEqual(profile.facility('F-BREW').product_ids, ['P-BEER'])
        self.assertNotIn('P-WATER', profile.facility('F-BREW').product_ids)
        self.assertEqual(profile.product('P-BEER').tags, ['GLASS_PACKAGING'])
        self.assertEqual(profile.product('P-WATER').tags, [])
        self.assertNotEqual(profile.product('P-BEER').tags_complete, profile.product('P-WATER').tags_complete)


class LegacyProjectionTests(unittest.TestCase):
    def test_each_legal_entity_projects_to_the_existing_company_schema(self):
        profile = load_profile(sample(), VOCABULARY)
        company = to_company(profile, 'PROD', VOCABULARY)
        self.assertIsInstance(company, Company)
        self.assertEqual(company.id, 'sample-group/PROD')
        self.assertTrue(company.synthetic)
        self.assertEqual(company.jurisdictions, ['Türkiye'])
        self.assertIn('üretim', company.activities)
        self.assertIn('alkollü içki üretim izni', company.licences)
        self.assertEqual(len(company.products), 2)

    def test_an_unstated_list_stays_unknown_and_a_complete_empty_list_stays_empty(self):
        profile = load_profile(sample(), VOCABULARY)
        self.assertIsNone(to_company(profile, 'SALES', VOCABULARY).licences)
        data = sample()
        data['legal_entities'][0]['licenses'] = []
        self.assertEqual(to_company(load_profile(data, VOCABULARY), 'PROD', VOCABULARY).licences, [])

    def test_the_existing_aml_obliged_party_reading_finds_no_financial_kind_in_a_beverage_entity(self):
        profile = load_profile(sample(), VOCABULARY)
        for entity in profile.legal_entities:
            self.assertEqual(company_kinds(to_company(profile, entity.entity_id, VOCABULARY)), [], entity.entity_id)


class PolicyScopeTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile(sample(), VOCABULARY)

    def covers(self, policy, level, target):
        return policy_covers(PolicyScope.model_validate(policy), self.profile, level, target)

    def test_a_group_policy_reaches_every_entity_and_facility_unless_it_is_group_only(self):
        policy = {'policy_id': 'GP', 'owner_level': 'GROUP'}
        for level, target in (('GROUP', 'G'), ('LEGAL_ENTITY', 'PROD'), ('LEGAL_ENTITY', 'SALES'), ('FACILITY', 'F-DEPOT')):
            self.assertEqual(self.covers(policy, level, target)['covers'], 'YES', target)
        closed = policy | {'inherits': False}
        self.assertEqual(self.covers(closed, 'LEGAL_ENTITY', 'SALES'), {'covers': 'NO', 'reason': 'GROUP_POLICY_NOT_INHERITED'})

    def test_an_entity_policy_covers_its_own_entity_and_facilities_only(self):
        policy = {'policy_id': 'EP', 'owner_level': 'LEGAL_ENTITY', 'entity_id': 'PROD'}
        self.assertEqual(self.covers(policy, 'LEGAL_ENTITY', 'PROD')['covers'], 'YES')
        self.assertEqual(self.covers(policy, 'FACILITY', 'F-BREW')['covers'], 'YES')
        self.assertEqual(self.covers(policy, 'LEGAL_ENTITY', 'SALES'), {'covers': 'NO', 'reason': 'OTHER_ENTITY_POLICY'})
        self.assertEqual(self.covers(policy, 'FACILITY', 'F-DEPOT'), {'covers': 'NO', 'reason': 'OTHER_ENTITY_POLICY'})
        self.assertEqual(self.covers(policy, 'GROUP', 'G'), {'covers': 'NO', 'reason': 'NARROWER_THAN_TARGET'})

    def test_a_production_policy_does_not_cover_the_sales_entity_even_from_group_level(self):
        policy = {'policy_id': 'GP-PROD', 'owner_level': 'GROUP', 'activity_classes': ['PRODUCTION']}
        self.assertEqual(self.covers(policy, 'LEGAL_ENTITY', 'PROD')['covers'], 'YES')
        self.assertEqual(self.covers(policy, 'LEGAL_ENTITY', 'SALES'), {'covers': 'UNKNOWN', 'reason': 'POLICY_ACTIVITY_UNSTATED'})
        data = sample()
        data['legal_entities'][1]['profile_complete'] = True
        profile = load_profile(data, VOCABULARY)
        self.assertEqual(policy_covers(PolicyScope.model_validate(policy), profile, 'LEGAL_ENTITY', 'SALES'),
                         {'covers': 'NO', 'reason': 'POLICY_ACTIVITY_MISMATCH'})

    def test_a_facility_policy_stays_at_its_facility(self):
        policy = {'policy_id': 'FP', 'owner_level': 'FACILITY', 'facility_id': 'F-BREW'}
        self.assertEqual(self.covers(policy, 'FACILITY', 'F-BREW')['covers'], 'YES')
        self.assertEqual(self.covers(policy, 'LEGAL_ENTITY', 'PROD'), {'covers': 'NO', 'reason': 'NARROWER_THAN_TARGET'})
        self.assertEqual(self.covers(policy, 'FACILITY', 'F-DEPOT'), {'covers': 'NO', 'reason': 'OTHER_FACILITY_POLICY'})

    def test_policy_owner_fields_follow_the_level(self):
        for broken in ({'policy_id': 'X', 'owner_level': 'GROUP', 'entity_id': 'PROD'},
                       {'policy_id': 'X', 'owner_level': 'LEGAL_ENTITY'},
                       {'policy_id': 'X', 'owner_level': 'FACILITY', 'entity_id': 'PROD'}):
            with self.assertRaises(ValueError):
                PolicyScope.model_validate(broken)


if __name__ == '__main__':
    unittest.main()

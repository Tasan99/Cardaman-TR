"""Pack selection and applicability routing (Cardaman TR, beverage milestone 3)."""
import copy
import re
import unittest
from pathlib import Path

from regchain.tr.core import ObligationScope
from regchain.tr.packs import DATA, Registry, load_json
from regchain.tr.profile import load_profile
from regchain.tr.routing import evaluate_scope, resolve, rollup, select_packs
from test_tr_profile import sample

REGISTRY = Registry.load()
VOCABULARY = REGISTRY.vocabulary


def profile(mutate=None):
    data = copy.deepcopy(sample())
    if mutate:
        mutate(data)
    return load_profile(data, VOCABULARY)


def scope(level, **fields):
    base = dict(scope_id='TEST_SCOPE', regulation_id='TR:YONETMELIK:TGK_ETIKETLEME', topic='TEST', description_tr='test',
                level=level, provision_ref=None, provision_status='UNRESOLVED')
    return ObligationScope.model_validate(base | fields)


def by_target(decisions):
    return {d.target_id: d for d in decisions}


class PackSelectionTests(unittest.TestCase):
    def selections(self, prof):
        return {s.pack_id: s for s in select_packs(prof, REGISTRY)}

    def test_a_group_with_an_alcoholic_and_a_non_alcoholic_product_selects_both_packs(self):
        selected = self.selections(profile())
        self.assertEqual(set(selected), {'BEVERAGE_ALCOHOL_TR', 'BEVERAGE_NON_ALCOHOL_TR'})
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].status, 'SELECTED')
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].product_ids, ['P-BEER'])
        self.assertIn('ALCOHOL_SCOPE_MATCH', selected['BEVERAGE_ALCOHOL_TR'].reason_codes)
        self.assertEqual(selected['BEVERAGE_NON_ALCOHOL_TR'].product_ids, ['P-WATER'])
        self.assertIn('NON_ALCOHOL_SCOPE_MATCH', selected['BEVERAGE_NON_ALCOHOL_TR'].reason_codes)

    def test_a_complete_product_list_without_alcohol_rules_the_alcohol_pack_out(self):
        def water_only(d):
            d['products'] = [p for p in d['products'] if p['product_id'] == 'P-WATER']
            for record in d['legal_entities'] + d['facilities'] + d['activities']:
                record['product_ids'] = [p for p in record['product_ids'] if p == 'P-WATER']
            d['legal_entities'][0]['licenses'] = []
            d['activities'] = []
        selected = self.selections(profile(water_only))
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].status, 'NOT_SELECTED')
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].reason_codes, ['PRODUCT_MISMATCH'])

    def test_an_incomplete_product_list_leaves_the_pack_unknown_not_excluded(self):
        def water_only_open(d):
            d['products'] = [p for p in d['products'] if p['product_id'] == 'P-WATER']
            for record in d['legal_entities'] + d['facilities'] + d['activities']:
                record['product_ids'] = [p for p in record['product_ids'] if p == 'P-WATER']
            d['legal_entities'][0]['licenses'] = []
            d['activities'] = []
            d['products_complete'] = False
        selected = self.selections(profile(water_only_open))
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].status, 'UNKNOWN')
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].reason_codes, ['PROFILE_INCOMPLETE'])

    def test_a_licence_only_one_pack_uses_selects_that_pack_without_products(self):
        def importer(d):
            d['products'] = [p for p in d['products'] if p['product_id'] == 'P-WATER']
            for record in d['legal_entities'] + d['facilities'] + d['activities']:
                record['product_ids'] = [p for p in record['product_ids'] if p == 'P-WATER']
            d['legal_entities'][0]['licenses'] = [{'license_class': 'ALCOHOL_IMPORT_PERMIT', 'status': 'HELD'}]
            d['activities'] = []
        selected = self.selections(profile(importer))
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].status, 'SELECTED')
        self.assertEqual(selected['BEVERAGE_ALCOHOL_TR'].reason_codes, ['LICENSE_MATCH'])

    def test_a_module_is_never_selected_on_its_own(self):
        self.assertNotIn('TR_FOOD_BEVERAGE_COMMON', self.selections(profile()))


class DimensionTests(unittest.TestCase):
    def test_a_stated_activity_matches_and_a_complete_profile_without_it_mismatches(self):
        decisions = by_target(evaluate_scope(scope('LEGAL_ENTITY', activity_classes=['PRODUCTION']), profile(), REGISTRY))
        self.assertEqual((decisions['PROD'].status, decisions['PROD'].reason_codes), ('APPLIES', ['ACTIVITY_MATCH']))
        self.assertEqual((decisions['SALES'].status, decisions['SALES'].reason_codes), ('UNKNOWN', ['PROFILE_INCOMPLETE']))
        complete = profile(lambda d: d['legal_entities'][1].update(profile_complete=True))
        decisions = by_target(evaluate_scope(scope('LEGAL_ENTITY', activity_classes=['PRODUCTION']), complete, REGISTRY))
        self.assertEqual((decisions['SALES'].status, decisions['SALES'].reason_codes), ('DOES_NOT_APPLY', ['ACTIVITY_MISMATCH']))

    def test_a_licence_matches_only_when_held(self):
        permit = scope('LEGAL_ENTITY', license_classes=['ALCOHOL_PRODUCTION_PERMIT'])
        self.assertEqual(by_target(evaluate_scope(permit, profile(), REGISTRY))['PROD'].reason_codes, ['LICENSE_MATCH'])
        applied = profile(lambda d: d['legal_entities'][0]['licenses'][0].update(status='APPLIED'))
        self.assertEqual(by_target(evaluate_scope(permit, applied, REGISTRY))['PROD'].status, 'DOES_NOT_APPLY')
        unknown = profile(lambda d: d['legal_entities'][0]['licenses'][0].update(status='UNKNOWN'))
        self.assertEqual(by_target(evaluate_scope(permit, unknown, REGISTRY))['PROD'].status, 'UNKNOWN')

    def test_a_product_level_duty_follows_the_product_class_and_the_alcohol_category(self):
        alcohol = by_target(evaluate_scope(scope('PRODUCT', regulation_id='TR:KANUN:4733', alcohol_scope='ALCOHOLIC'),
                                           profile(), REGISTRY))
        self.assertEqual((alcohol['P-BEER'].status, alcohol['P-BEER'].reason_codes), ('APPLIES', ['ALCOHOL_SCOPE_MATCH']))
        self.assertEqual((alcohol['P-WATER'].status, alcohol['P-WATER'].reason_codes),
                         ('DOES_NOT_APPLY', ['ALCOHOL_SCOPE_MISMATCH']))
        water = by_target(evaluate_scope(scope('PRODUCT', product_classes=['PACKAGED_WATER']), profile(), REGISTRY))
        self.assertEqual(water['P-WATER'].reason_codes, ['PRODUCT_MATCH'])
        self.assertEqual(water['P-BEER'].reason_codes, ['PRODUCT_MISMATCH'])

    def test_a_product_attribute_is_unknown_until_the_tags_are_complete(self):
        claims = by_target(evaluate_scope(scope('PRODUCT', product_attributes=['HEALTH_CLAIM']), profile(), REGISTRY))
        self.assertEqual(claims['P-BEER'].status, 'DOES_NOT_APPLY')
        self.assertEqual((claims['P-WATER'].status, claims['P-WATER'].reason_codes), ('UNKNOWN', ['PROFILE_INCOMPLETE']))

    def test_an_entity_handling_some_in_scope_products_gets_a_partial_answer_naming_them(self):
        decision = by_target(evaluate_scope(scope('LEGAL_ENTITY', regulation_id='TR:KANUN:4733', alcohol_scope='ALCOHOLIC',
                                                  activity_classes=['WHOLESALE']), profile(), REGISTRY))['SALES']
        self.assertEqual(decision.status, 'PARTIAL')
        self.assertEqual(decision.applies_to_products, ['P-BEER'])
        self.assertIn('PARTIAL_PRODUCT_SCOPE', decision.reason_codes)

    def test_an_entity_with_no_products_in_a_complete_list_is_out_of_an_alcohol_scope(self):
        def no_products(d):
            d['legal_entities'][1]['product_ids'] = []
            d['facilities'][1]['product_ids'] = []
            d['activities'][0]['product_ids'] = []
        decision = by_target(evaluate_scope(scope('LEGAL_ENTITY', regulation_id='TR:KANUN:4733', alcohol_scope='ALCOHOLIC'),
                                            profile(no_products), REGISTRY))['SALES']
        self.assertEqual((decision.status, decision.reason_codes), ('DOES_NOT_APPLY', ['ALCOHOL_SCOPE_MISMATCH']))

    def test_an_unclear_scope_always_answers_unknown(self):
        decisions = evaluate_scope(scope('LEGAL_ENTITY', scope_status='UNCLEAR', activity_classes=['PRODUCTION']),
                                   profile(), REGISTRY)
        for decision in decisions:
            self.assertEqual(decision.status, 'UNKNOWN')
            self.assertIn('REGULATORY_SCOPE_UNCLEAR', decision.reason_codes)

    def test_decisions_carry_gate_records_in_the_existing_trace_shape_and_their_grounding(self):
        decision = by_target(evaluate_scope(scope('FACILITY', facility_classes=['BREWERY']), profile(), REGISTRY))['F-BREW']
        self.assertEqual(decision.status, 'APPLIES')
        self.assertEqual(set(decision.gates[0]), {'gate', 'status', 'clear', 'reason', 'evidence'})
        self.assertEqual((decision.gates[0]['gate'], decision.gates[0]['status']), ('FACILITY_CLASS', 'MATCH'))
        self.assertEqual((decision.basis, decision.provision_status), ('PACK_METADATA', 'UNRESOLVED'))
        self.assertTrue(decision.creates_obligation)

    def test_an_activity_level_duty_reads_the_activity_and_its_own_products(self):
        decisions = by_target(evaluate_scope(scope('ACTIVITY', regulation_id='TR:KANUN:4733', alcohol_scope='ALCOHOLIC',
                                                   activity_classes=['ADVERTISING']), profile(), REGISTRY))
        self.assertEqual(decisions['A-ADV'].status, 'APPLIES')
        self.assertEqual(decisions['A-ADV'].entity_id, 'SALES')
        water_ad = profile(lambda d: d['activities'][0].update(product_ids=['P-WATER']))
        decisions = by_target(evaluate_scope(scope('ACTIVITY', regulation_id='TR:KANUN:4733', alcohol_scope='ALCOHOLIC',
                                                   activity_classes=['ADVERTISING']), water_ad, REGISTRY))
        self.assertEqual((decisions['A-ADV'].status, decisions['A-ADV'].reason_codes),
                         ('DOES_NOT_APPLY', ['ALCOHOL_SCOPE_MISMATCH']))


class RollupTests(unittest.TestCase):
    def test_a_facility_duty_rolls_up_to_its_entity_and_to_the_group_without_spreading(self):
        target = scope('FACILITY', facility_classes=['BREWERY'])
        decisions = evaluate_scope(target, profile(), REGISTRY)
        rolled = {(d.level, d.target_id): d for d in rollup(target, decisions, profile(), REGISTRY)}
        self.assertEqual(rolled[('LEGAL_ENTITY', 'PROD')].status, 'APPLIES')
        self.assertEqual(rolled[('LEGAL_ENTITY', 'SALES')].status, 'DOES_NOT_APPLY')
        self.assertEqual(rolled[('GROUP', 'G')].status, 'PARTIAL')
        self.assertIn('PARTIAL_CHILD_SCOPE', rolled[('GROUP', 'G')].reason_codes)

    def test_an_unknown_child_keeps_the_rollup_unknown_rather_than_does_not_apply(self):
        target = scope('LEGAL_ENTITY', activity_classes=['PRODUCTION'], scope_id='ONLY_UNKNOWN')
        decisions = [d for d in evaluate_scope(target, profile(), REGISTRY) if d.target_id == 'SALES']
        rolled = {(d.level, d.target_id): d for d in rollup(target, decisions, profile(), REGISTRY)}
        self.assertEqual(rolled[('GROUP', 'G')].status, 'UNKNOWN')


class ResolutionTests(unittest.TestCase):
    def test_every_routed_scope_comes_from_a_selected_pack_and_shared_scopes_name_both(self):
        result = resolve(profile(), REGISTRY)
        packs = {s.pack_id: s.status for s in result.selections}
        routed = {d.scope_id for d in result.decisions}
        self.assertTrue(routed)
        for decision in result.decisions:
            self.assertTrue(all(packs[p] in ('SELECTED', 'UNKNOWN') for p in decision.packs), decision.scope_id)
        labelling = next(d for d in result.decisions if d.scope_id == 'FOOD_LABELLING_GENERAL')
        self.assertEqual(labelling.packs, ['BEVERAGE_ALCOHOL_TR', 'BEVERAGE_NON_ALCOHOL_TR'])
        group = {d.scope_id: d for d in result.rollups if d.level == 'GROUP'}
        self.assertEqual(set(group), routed)

    def test_scopes_of_an_unselected_pack_are_reported_out_of_scope_not_silently_dropped(self):
        def water_only(d):
            d['products'] = [p for p in d['products'] if p['product_id'] == 'P-WATER']
            for record in d['legal_entities'] + d['facilities'] + d['activities']:
                record['product_ids'] = [p for p in record['product_ids'] if p == 'P-WATER']
            d['legal_entities'][0]['licenses'] = []
            d['activities'] = []
        result = resolve(profile(water_only), REGISTRY, include_unselected=True)
        out = {d.scope_id: d for d in result.out_of_scope}
        self.assertIn('ALC_SALES_PRESENTATION', out)
        self.assertEqual((out['ALC_SALES_PRESENTATION'].status, out['ALC_SALES_PRESENTATION'].reason_codes),
                         ('DOES_NOT_APPLY', ['PACK_NOT_SELECTED']))
        self.assertNotIn('ALC_SALES_PRESENTATION', {d.scope_id for d in result.decisions})

    def test_guidance_is_routed_but_never_creates_an_obligation(self):
        parts = {'vocabulary': load_json(DATA / 'vocabulary.json'), 'regulations': load_json(DATA / 'regulations.json'),
                 'scopes': load_json(DATA / 'scopes.json'),
                 'packs': [load_json(p) for p in sorted((DATA / 'packs').glob('*.json'))]}
        parts['regulations']['regulations'].append({
            'regulation_id': 'TR:RESMI_KILAVUZ:TEST_WATER_GUIDE', 'title': 'Test kılavuzu', 'jurisdiction': 'TR',
            'regulator': 'SAGLIK', 'regulation_type': 'RESMI_KILAVUZ', 'binding_status': 'OFFICIAL_GUIDANCE', 'number': None,
            'effective_date': None, 'effective_status': 'UNKNOWN', 'metadata_status': 'UNVERIFIED',
            'verification_note': 'test only', 'interprets': ['TR:YONETMELIK:INSANI_TUKETIM_SULAR'], 'sector_tags': ['WATER']})
        parts['scopes']['scopes'].append({'scope_id': 'TEST_WATER_GUIDANCE', 'regulation_id': 'TR:RESMI_KILAVUZ:TEST_WATER_GUIDE',
                                          'topic': 'T', 'description_tr': 't', 'level': 'PRODUCT', 'provision_ref': None,
                                          'provision_status': 'UNRESOLVED', 'alcohol_scope': 'NON_ALCOHOLIC',
                                          'product_classes': ['PACKAGED_WATER']})
        next(p for p in parts['packs'] if p['pack_id'] == 'BEVERAGE_NON_ALCOHOL_TR')['regulation_ids'].append(
            'TR:RESMI_KILAVUZ:TEST_WATER_GUIDE')
        registry = Registry.from_data(parts['vocabulary'], parts['regulations'], parts['scopes'], parts['packs'])
        decision = by_target(evaluate_scope(registry.scopes['TEST_WATER_GUIDANCE'], profile(), registry))['P-WATER']
        self.assertEqual(decision.status, 'APPLIES')
        self.assertFalse(decision.creates_obligation)
        self.assertIn('GUIDANCE_ONLY', decision.reason_codes)


class NoBrandLogicTests(unittest.TestCase):
    def test_no_brand_name_appears_in_the_tr_code(self):
        source = Path(__file__).resolve().parents[1] / 'src' / 'regchain' / 'tr'
        for path in source.glob('*.py'):
            text = path.read_text(encoding='utf-8').lower()
            for brand in ('efes', 'tuborg', 'coca', 'pepsi', 'carlsberg'):
                self.assertIsNone(re.search(brand, text), f'{brand} in {path.name}')


if __name__ == '__main__':
    unittest.main()

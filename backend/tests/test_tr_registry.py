"""CARDAMAN_TR_CORE metadata and the regulation pack registry (Cardaman TR, beverage milestone 1)."""
import copy
import json
import unittest
from datetime import date

from pydantic import ValidationError

from regchain.tr.core import (BINDING_OF, MVP_LAYERS, ObligationScope, RegulationMeta, RegulationVersion,
                              version_chain)
from regchain.tr.packs import DATA, Registry, RegistryError, load_json


def data():
    return {'vocabulary': load_json(DATA / 'vocabulary.json'),
            'regulations': load_json(DATA / 'regulations.json')['regulations'],
            'scopes': load_json(DATA / 'scopes.json')['scopes'],
            'packs': [load_json(path) for path in sorted((DATA / 'packs').glob('*.json'))]}


def build(parts):
    return Registry.from_data(parts['vocabulary'], {'regulations': parts['regulations']}, {'scopes': parts['scopes']},
                              parts['packs'])


def regulation(**overrides):
    base = dict(regulation_id='TR:YONETMELIK:SAMPLE', title='Örnek yönetmelik', jurisdiction='TR', regulator='TICARET',
                regulation_type='YONETMELIK', binding_status='BINDING', number=None, effective_date=None,
                effective_status='UNKNOWN', metadata_status='UNVERIFIED', verification_note='test entry',
                sector_tags=['BEVERAGE'])
    return base | overrides


class CoreMetadataTests(unittest.TestCase):
    def test_every_regulation_type_has_one_source_class_and_the_mvp_excludes_precedent(self):
        self.assertEqual(set(BINDING_OF.values()), {'BINDING', 'OFFICIAL_GUIDANCE', 'DECISION_PRECEDENT'})
        for kind in ('KANUN', 'YONETMELIK', 'TEBLIG'):
            self.assertEqual(BINDING_OF[kind], 'BINDING')
        self.assertEqual(BINDING_OF['KURUL_KARARI'], 'DECISION_PRECEDENT')
        self.assertEqual(MVP_LAYERS, ('BINDING', 'OFFICIAL_GUIDANCE'))

    def test_binding_status_must_follow_the_regulation_type(self):
        RegulationMeta.model_validate(regulation())
        with self.assertRaises(ValidationError):
            RegulationMeta.model_validate(regulation(binding_status='OFFICIAL_GUIDANCE'))
        with self.assertRaises(ValidationError):
            RegulationMeta.model_validate(regulation(regulation_id='TR:TEBLIG:SAMPLE'))

    def test_guidance_and_precedent_must_name_the_binding_text_they_read(self):
        guidance = regulation(regulation_id='TR:RESMI_REHBER:SAMPLE', regulation_type='RESMI_REHBER',
                              binding_status='OFFICIAL_GUIDANCE')
        with self.assertRaises(ValidationError):
            RegulationMeta.model_validate(guidance)
        RegulationMeta.model_validate(guidance | {'interprets': ['TR:YONETMELIK:SAMPLE']})
        with self.assertRaises(ValidationError):
            RegulationMeta.model_validate(regulation(interprets=['TR:KANUN:1']))

    def test_jurisdiction_is_tr_only(self):
        with self.assertRaises(ValidationError):
            RegulationMeta.model_validate(regulation(jurisdiction='UK'))

    def test_versions_form_one_chain_with_hashes_and_dates(self):
        v1 = dict(regulation_id='TR:YONETMELIK:SAMPLE', version_id='v1', publication_date=date(2020, 1, 1),
                  effective_date=date(2020, 6, 1), source_hash='a' * 64, previous_version_id=None)
        v2 = v1 | dict(version_id='v2', publication_date=date(2024, 1, 1), effective_date=None, source_hash=None,
                       previous_version_id='v1')
        meta = RegulationMeta.model_validate(regulation(versions=[v1, v2]))
        self.assertEqual([v.version_id for v in version_chain(meta)], ['v1', 'v2'])
        for broken in ([v1 | {'previous_version_id': 'v0'}], [v1, v2 | {'previous_version_id': None}],
                       [v1, v2 | {'regulation_id': 'TR:YONETMELIK:OTHER'}], [v1 | {'source_hash': 'not-a-hash'}]):
            with self.assertRaises(ValidationError):
                RegulationMeta.model_validate(regulation(versions=broken))
        with self.assertRaises(ValidationError):
            RegulationVersion.model_validate(v2 | {'previous_version_id': 'v2'})

    def test_scope_dimensions_follow_the_level(self):
        base = dict(scope_id='S1', regulation_id='TR:YONETMELIK:SAMPLE', topic='T', description_tr='d', level='PRODUCT',
                    provision_ref=None, provision_status='UNRESOLVED', product_classes=['BEER'])
        ObligationScope.model_validate(base)
        for extra in ({'facility_classes': ['BREWERY']}, {'entity_classes': ['PRODUCER']}, {'activity_classes': ['PRODUCTION']},
                      {'license_classes': ['FOOD_BUSINESS_REGISTRATION']}):
            with self.assertRaises(ValidationError):
                ObligationScope.model_validate(base | extra)
        with self.assertRaises(ValidationError):
            ObligationScope.model_validate(base | {'provision_ref': 'md. 5'})
        with self.assertRaises(ValidationError):
            ObligationScope.model_validate(base | {'provision_status': 'RESOLVED'})
        ObligationScope.model_validate(base | {'provision_ref': 'md. 5', 'provision_status': 'RESOLVED'})


class ShippedRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = Registry.load(DATA)

    def test_the_two_beverage_packs_and_the_shared_module_are_registered(self):
        packs = {p.pack_id: p for p in self.registry.packs}
        self.assertEqual(set(packs), {'BEVERAGE_ALCOHOL_TR', 'BEVERAGE_NON_ALCOHOL_TR', 'TR_FOOD_BEVERAGE_COMMON'})
        self.assertEqual(packs['TR_FOOD_BEVERAGE_COMMON'].kind, 'MODULE')
        for pack_id in ('BEVERAGE_ALCOHOL_TR', 'BEVERAGE_NON_ALCOHOL_TR'):
            pack = packs[pack_id]
            self.assertEqual((pack.kind, pack.jurisdiction, pack.sector), ('PACK', 'TR', 'BEVERAGE'))
            self.assertIn('TR_FOOD_BEVERAGE_COMMON', pack.includes)
            for field in ('pack_id', 'jurisdiction', 'sector', 'subsector', 'regulators', 'regulation_ids', 'product_classes',
                          'activity_classes', 'facility_classes', 'entity_classes', 'license_classes', 'status', 'version'):
                self.assertTrue(getattr(pack, field), (pack_id, field))
        self.assertEqual(packs['BEVERAGE_ALCOHOL_TR'].alcohol_scope, 'ALCOHOLIC')
        self.assertEqual(packs['BEVERAGE_NON_ALCOHOL_TR'].alcohol_scope, 'NON_ALCOHOLIC')

    def test_packs_share_one_core_module_instead_of_copying_it(self):
        common = set(self.registry.pack('TR_FOOD_BEVERAGE_COMMON').regulation_ids)
        alcohol = set(self.registry.pack('BEVERAGE_ALCOHOL_TR').regulation_ids)
        non_alcohol = set(self.registry.pack('BEVERAGE_NON_ALCOHOL_TR').regulation_ids)
        self.assertFalse(common & alcohol)
        self.assertFalse(common & non_alcohol)
        self.assertLessEqual(common, set(self.registry.regulation_ids('BEVERAGE_ALCOHOL_TR')))
        self.assertLessEqual(common, set(self.registry.regulation_ids('BEVERAGE_NON_ALCOHOL_TR')))

    def test_pack_product_classes_agree_with_the_pack_alcohol_scope(self):
        vocabulary = self.registry.vocabulary
        for pack_id, category in (('BEVERAGE_ALCOHOL_TR', 'ALCOHOLIC'), ('BEVERAGE_NON_ALCOHOL_TR', 'NON_ALCOHOLIC')):
            for product_class in self.registry.pack(pack_id).product_classes:
                self.assertEqual(vocabulary.alcohol_of(product_class), category, product_class)

    def test_mvp_packs_carry_binding_and_guidance_layers_only(self):
        for pack in self.registry.packs:
            self.assertNotIn('DECISION_PRECEDENT', pack.source_layers)
        for reg in self.registry.regulations.values():
            self.assertNotEqual(reg.binding_status, 'DECISION_PRECEDENT', reg.regulation_id)

    def test_every_catalogue_entry_says_whether_its_identifiers_were_verified(self):
        for reg in self.registry.regulations.values():
            self.assertIn(reg.metadata_status, ('VERIFIED', 'UNVERIFIED'))
            if reg.metadata_status == 'UNVERIFIED':
                self.assertTrue(reg.verification_note, reg.regulation_id)
            if reg.effective_date is None:
                self.assertEqual(reg.effective_status, 'UNKNOWN', reg.regulation_id)

    def test_scopes_resolve_to_the_packs_that_carry_their_regulation(self):
        alcohol = {s.scope_id for s in self.registry.scopes_of('BEVERAGE_ALCOHOL_TR')}
        non_alcohol = {s.scope_id for s in self.registry.scopes_of('BEVERAGE_NON_ALCOHOL_TR')}
        common = {s.scope_id for s in self.registry.scopes_of('TR_FOOD_BEVERAGE_COMMON')}
        self.assertTrue(common)
        self.assertLessEqual(common, alcohol & non_alcohol)
        for scope in self.registry.scopes.values():
            if scope.alcohol_scope == 'ALCOHOLIC':
                self.assertNotIn(scope.scope_id, non_alcohol)
            if scope.alcohol_scope == 'NON_ALCOHOLIC':
                self.assertNotIn(scope.scope_id, alcohol)

    def test_no_brand_name_is_part_of_the_registry(self):
        text = json.dumps(data(), ensure_ascii=False).lower()
        for brand in ('efes', 'tuborg', 'coca', 'pepsi', 'carlsberg', 'anadolu grubu'):
            self.assertNotIn(brand, text)


class RegistryValidationTests(unittest.TestCase):
    def assert_rejected(self, mutate, needle):
        parts = copy.deepcopy(data())
        mutate(parts)
        with self.assertRaises(RegistryError) as caught:
            build(parts)
        self.assertIn(needle, str(caught.exception))

    def pack(self, parts, pack_id):
        return next(p for p in parts['packs'] if p['pack_id'] == pack_id)

    def test_the_shipped_data_builds(self):
        build(data())

    def test_unknown_vocabulary_terms_are_rejected_not_ignored(self):
        self.assert_rejected(lambda p: self.pack(p, 'BEVERAGE_ALCOHOL_TR')['facility_classes'].append('BREWREY'), 'BREWREY')
        self.assert_rejected(lambda p: p['scopes'][0]['activity_classes'].append('ADVERTISNG')
                             if p['scopes'][0]['level'] != 'PRODUCT' else p['scopes'][0]['product_classes'].append('ADVERTISNG'),
                             'ADVERTISNG')

    def test_a_pack_cannot_name_a_regulation_the_catalogue_lacks(self):
        self.assert_rejected(lambda p: self.pack(p, 'BEVERAGE_NON_ALCOHOL_TR')['regulation_ids'].append('TR:KANUN:0000'),
                             'TR:KANUN:0000')

    def test_a_regulation_may_belong_to_one_pack_or_module_only(self):
        def duplicate(p):
            common = self.pack(p, 'TR_FOOD_BEVERAGE_COMMON')['regulation_ids'][0]
            self.pack(p, 'BEVERAGE_ALCOHOL_TR')['regulation_ids'].append(common)
        self.assert_rejected(duplicate, 'more than one')

    def test_an_alcohol_pack_cannot_list_a_non_alcoholic_product_class(self):
        self.assert_rejected(lambda p: self.pack(p, 'BEVERAGE_ALCOHOL_TR')['product_classes'].append('PACKAGED_WATER'),
                             'PACKAGED_WATER')

    def test_an_active_pack_cannot_enable_the_second_phase_precedent_layer(self):
        self.assert_rejected(lambda p: self.pack(p, 'BEVERAGE_ALCOHOL_TR')['source_layers'].append('DECISION_PRECEDENT'),
                             'DECISION_PRECEDENT')

    def test_includes_must_be_modules_and_acyclic(self):
        self.assert_rejected(lambda p: self.pack(p, 'BEVERAGE_ALCOHOL_TR')['includes'].append('BEVERAGE_NON_ALCOHOL_TR'), 'MODULE')
        self.assert_rejected(lambda p: self.pack(p, 'TR_FOOD_BEVERAGE_COMMON')['includes'].append('TR_FOOD_BEVERAGE_COMMON'), 'cycle')

    def test_a_scope_must_point_to_a_catalogued_regulation(self):
        def orphan(p):
            p['scopes'][0]['regulation_id'] = 'TR:YONETMELIK:MISSING'
        self.assert_rejected(orphan, 'TR:YONETMELIK:MISSING')

    def test_an_alcohol_scope_cannot_sit_in_a_non_alcohol_pack(self):
        def misplaced(p):
            scope = next(s for s in p['scopes'] if s['alcohol_scope'] == 'ALCOHOLIC')
            regulation_id = scope['regulation_id']
            for pack in p['packs']:
                if regulation_id in pack['regulation_ids']:
                    pack['regulation_ids'].remove(regulation_id)
            self.pack(p, 'BEVERAGE_NON_ALCOHOL_TR')['regulation_ids'].append(regulation_id)
        self.assert_rejected(misplaced, 'ALCOHOLIC')

    def test_guidance_must_interpret_binding_text_carried_by_the_same_pack(self):
        def guidance(p):
            binding = self.pack(p, 'BEVERAGE_NON_ALCOHOL_TR')['regulation_ids'][0]
            p['regulations'].append(regulation(regulation_id='TR:RESMI_KILAVUZ:SAMPLE', regulation_type='RESMI_KILAVUZ',
                                               binding_status='OFFICIAL_GUIDANCE', regulator='TARIM_ORMAN',
                                               interprets=[binding], verification_note='test'))
            self.pack(p, 'BEVERAGE_ALCOHOL_TR')['regulation_ids'].append('TR:RESMI_KILAVUZ:SAMPLE')
        self.assert_rejected(guidance, 'interprets')

    def test_a_pack_must_list_the_regulators_of_its_own_regulations(self):
        def drop(p):
            pack = self.pack(p, 'BEVERAGE_ALCOHOL_TR')
            pack['regulators'] = [r for r in pack['regulators'] if r != 'TARIM_ORMAN']
        self.assert_rejected(drop, 'TARIM_ORMAN')


if __name__ == '__main__':
    unittest.main()

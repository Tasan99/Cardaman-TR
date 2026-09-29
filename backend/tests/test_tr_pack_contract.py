"""Pack discovery, the contract every discovered pack meets, and the core/pack boundary (Cardaman TR step 2)."""
import copy
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from regchain.tr.packs import DATA, Registry, RegistryError, load_json, merge_vocabularies, pack_roots

SRC = Path(__file__).resolve().parents[1] / 'src' / 'regchain'
REGISTRY = Registry.discover()
BRANDS = ('efes', 'tuborg', 'coca', 'pepsi', 'carlsberg', 'fanta', 'sprite')


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def food_root(self, vocabulary=None, regulation_id='TR:YONETMELIK:TEST_FOOD'):
        write(self.root / 'vocabulary.json', vocabulary or {
            'format': 'cardaman-tr-vocabulary/1', 'regulators': {'TARIM_ORMAN': 'Tarım ve Orman Bakanlığı'},
            'sector_tags': {'FOOD': 'gıda'}, 'product_classes': {'BAKERY': {'label_tr': 'Unlu mamul', 'alcohol': 'NON_ALCOHOLIC'}},
            'facility_classes': {'BAKERY_PLANT': 'Unlu mamul tesisi'}})
        write(self.root / 'regulations.json', {'format': 'cardaman-tr-regulations/1', 'regulations': [{
            'regulation_id': regulation_id, 'title': 'Test gıda yönetmeliği', 'jurisdiction': 'TR',
            'regulator': 'TARIM_ORMAN', 'regulation_type': 'YONETMELIK', 'binding_status': 'BINDING', 'number': None,
            'effective_date': None, 'effective_status': 'UNKNOWN', 'metadata_status': 'UNVERIFIED',
            'verification_note': 'test only', 'sector_tags': ['FOOD']}]})
        write(self.root / 'scopes.json', {'format': 'cardaman-tr-scopes/1', 'scopes': [{
            'scope_id': 'TEST_BAKERY_FACILITY', 'regulation_id': regulation_id, 'topic': 'T', 'description_tr': 't',
            'level': 'FACILITY', 'provision_ref': None, 'provision_status': 'UNRESOLVED', 'facility_classes': ['BAKERY_PLANT']}]})
        write(self.root / 'packs' / 'food_test.json', {
            'format': 'cardaman-tr-pack/1', 'pack_id': 'FOOD_TEST_TR', 'kind': 'PACK', 'title': 'Test food pack',
            'jurisdiction': 'TR', 'sector': 'FOOD', 'subsector': 'BAKERY', 'alcohol_scope': 'NON_ALCOHOLIC',
            'regulators': ['TARIM_ORMAN', 'CEVRE_SEHIRCILIK', 'TICARET', 'HAZINE_MALIYE'], 'regulation_ids': [regulation_id], 'includes': ['TR_FOOD_BEVERAGE_COMMON'],
            'product_classes': ['BAKERY'], 'facility_classes': ['BAKERY_PLANT'], 'status': 'ACTIVE', 'version': '0.1.0'})
        return self.root

    def test_the_builtin_root_comes_first_and_registered_sector_roots_are_added(self):
        self.assertEqual(pack_roots()[0], DATA)
        builtin = {p.pack_id for p in Registry.load(DATA).packs}
        discovered = {p.pack_id for p in REGISTRY.packs}
        self.assertTrue(builtin <= discovered)
        self.assertGreaterEqual(len(pack_roots()), 2)

    def test_a_second_root_adds_a_pack_that_reuses_a_shared_module_without_touching_core(self):
        registry = Registry.discover([DATA, self.food_root()])
        self.assertIn('FOOD_TEST_TR', {p.pack_id for p in registry.selectable_packs()})
        self.assertIn('TR:YONETMELIK:TGK_ETIKETLEME', registry.regulation_ids('FOOD_TEST_TR'))
        self.assertIn('TEST_BAKERY_FACILITY', {s.scope_id for s in registry.scopes_of('FOOD_TEST_TR')})
        self.assertEqual(registry.owner('TR:YONETMELIK:TGK_ETIKETLEME'), 'TR_FOOD_BEVERAGE_COMMON')

    def test_two_roots_may_not_define_one_term_differently(self):
        vocabulary = {'format': 'cardaman-tr-vocabulary/1', 'regulators': {'TARIM_ORMAN': 'Başka bir ad'}}
        with self.assertRaisesRegex(RegistryError, 'regulators.TARIM_ORMAN'):
            Registry.discover([DATA, self.food_root(vocabulary=vocabulary)])

    def test_two_roots_may_not_own_one_regulation(self):
        with self.assertRaisesRegex(RegistryError, 'duplicate regulation TR:YONETMELIK:TGK_ETIKETLEME'):
            Registry.discover([DATA, self.food_root(regulation_id='TR:YONETMELIK:TGK_ETIKETLEME')])

    def test_merging_keeps_identical_definitions_once(self):
        base = load_json(DATA / 'vocabulary.json')
        merged = merge_vocabularies([base, copy.deepcopy(base)])
        self.assertEqual(merged['product_classes'], base['product_classes'])


class PackContractTests(unittest.TestCase):
    def test_every_selectable_pack_can_be_selected_by_what_a_company_handles(self):
        for pack in REGISTRY.selectable_packs():
            self.assertTrue(pack.product_classes or pack.license_classes, pack.pack_id)

    def test_every_pack_carries_regulations_and_every_carried_scope_resolves(self):
        for pack in REGISTRY.packs:
            carried = set(REGISTRY.regulation_ids(pack.pack_id))
            self.assertTrue(carried, pack.pack_id)
            for scope in REGISTRY.scopes_of(pack.pack_id):
                self.assertIn(scope.regulation_id, carried)

    def test_every_regulation_has_one_owner_inside_the_closure_of_each_pack_carrying_it(self):
        for pack in REGISTRY.packs:
            closure = set(REGISTRY.closure(pack.pack_id))
            for regulation_id in REGISTRY.regulation_ids(pack.pack_id):
                self.assertIn(REGISTRY.owner(regulation_id), closure)

    def test_an_active_pack_does_not_enable_second_phase_sources(self):
        for pack in REGISTRY.packs:
            if pack.status == 'ACTIVE':
                self.assertNotIn('DECISION_PRECEDENT', pack.source_layers, pack.pack_id)


class BoundaryTests(unittest.TestCase):
    CORE = ('pilot', 'extraction', 'ingestion', 'evaluation', 'platform')

    def sources(self, *parts):
        for part in parts:
            yield from (SRC / part).rglob('*.py') if (SRC / part).is_dir() else [SRC / part]

    def test_the_engine_does_not_import_the_pack_layer(self):
        for path in self.sources(*self.CORE, 'model_router.py', 'api.py'):
            text = path.read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'regchain\.tr\b|from \.\.?tr\b|from \.\.tr\.', text), path.name)

    def test_no_core_code_names_a_pack_or_module(self):
        """Only a pack's own package and the compatibility layer (regchain.compat, which registers the default) may name it."""
        ids = [p.pack_id for p in REGISTRY.packs]
        for path in SRC.rglob('*.py'):
            if path.relative_to(SRC).parts[0] == 'packs' or path == SRC / 'compat.py':
                continue
            text = path.read_text(encoding='utf-8')
            for pack_id in ids:
                self.assertNotIn(pack_id, text, f'{pack_id} in {path.relative_to(SRC)}')

    def test_no_code_names_a_brand(self):
        for path in SRC.rglob('*.py'):
            text = path.read_text(encoding='utf-8').lower()
            for brand in BRANDS:
                self.assertIsNone(re.search(rf'\b{brand}', text), f'{brand} in {path.relative_to(SRC)}')


if __name__ == '__main__':
    unittest.main()

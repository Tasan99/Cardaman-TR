"""FINANCIAL_SERVICES_TR behind the sector seam: the default analysis is byte-identical, the core names no pack."""
import json
import re
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from regchain import compat, sector
from regchain.pilot import engine
from regchain.pilot.engine import analyze
from regchain.pilot.sources import load_sources, select_targets
from regchain.tr.packs import Registry
from test_phase15 import TurkishRules, anadolu
from test_pilot import policies

SRC = Path(__file__).resolve().parents[1] / 'src' / 'regchain'
FIXTURES = Path(__file__).with_name('fixtures')
CORE = ('pilot', 'extraction', 'ingestion', 'evaluation', 'platform', 'model_router.py', 'api.py', 'evidence.py', 'assessment.py',
        'extraction_api.py', 'regulatory_api.py', 'migrate.py')


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 30, 0, 0, 0, tzinfo=tz or timezone.utc)


def tedbirler_sections():
    temp = tempfile.mkdtemp()
    root = Path(temp) / 'sources'
    root.mkdir()
    snapshot = json.loads(FIXTURES.joinpath('tedbirler-snapshot.json').read_text(encoding='utf-8'))
    shutil.copy(FIXTURES / 'tedbirler-200713012.html', root / snapshot['sources'][0]['raw_file'])
    (root / 'snapshot.json').write_text(json.dumps(snapshot), encoding='utf-8')
    _, sections = load_sources(root)
    shutil.rmtree(temp, ignore_errors=True)
    return sections


def run(pack, sections, labels) -> bytes:
    """One analysis with the clock frozen, serialised as the packet is written."""
    with patch.object(engine.time, 'monotonic', return_value=1000.0), patch.object(engine, 'datetime', FrozenDatetime):
        packet = analyze(anadolu(), policies(), sections, TurkishRules(), labels, pack=pack)
    return json.dumps(packet, ensure_ascii=False, default=str).encode('utf-8')


class ByteIdenticalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sections = tedbirler_sections()
        cls.labels = select_targets(cls.sections, 'YONETMELIK', '200713012', 'all', ['3', '4', '5', '8'])

    def test_the_default_analysis_is_deterministic_under_a_frozen_clock(self):
        self.assertEqual(run(None, self.sections, self.labels), run(None, self.sections, self.labels))

    def test_naming_the_financial_pack_is_byte_identical_to_the_default(self):
        default = run(None, self.sections, self.labels)
        explicit = run('FINANCIAL_SERVICES_TR', self.sections, self.labels)
        self.assertGreater(len(default), 1000)
        self.assertTrue(json.loads(default)['events'][0]['payload']['obligations'], 'the fixture run produced no obligations')
        self.assertEqual(default, explicit)

    def test_an_unregistered_pack_is_refused_not_silently_analysed_with_the_default(self):
        with self.assertRaisesRegex(ValueError, 'OTHER_PACK'):
            analyze(anadolu(), policies(), self.sections, TurkishRules(), self.labels, pack='OTHER_PACK')


class ResolutionTests(unittest.TestCase):
    def test_the_default_resolves_to_the_financial_pack_through_the_compatibility_layer_only(self):
        self.assertEqual(compat.DEFAULT_PACK_ID, 'FINANCIAL_SERVICES_TR')
        self.assertEqual(sector.default_pack_id(), compat.DEFAULT_PACK_ID)
        self.assertEqual(sector.knowledge().pack_id, compat.DEFAULT_PACK_ID)
        self.assertEqual(sector.knowledge('FINANCIAL_SERVICES_TR'), sector.knowledge())

    def test_the_financial_pack_is_discoverable_and_contract_valid(self):
        registry = Registry.discover()
        pack = registry.pack('FINANCIAL_SERVICES_TR')
        self.assertEqual((pack.kind, pack.status), ('PACK', 'ACTIVE'))
        self.assertIn(pack, registry.selectable_packs())
        self.assertEqual(set(registry.regulation_ids(pack.pack_id)), {'TR:KANUN:5549', 'TR:YONETMELIK:TEDBIRLER_200713012'})
        self.assertEqual(registry.vocabulary.regulators['MASAK'], 'Mali Suçları Araştırma Kurulu Başkanlığı')
        self.assertEqual(registry.vocabulary.regulators['TARIM_ORMAN'], 'Tarım ve Orman Bakanlığı')   # the beverage root is still there

    def test_the_engine_tables_come_from_the_pack_module(self):
        from regchain.packs.financial_services_tr import KNOWLEDGE
        from regchain.pilot import applicability, conflict, entities
        self.assertIs(applicability.SECTOR, KNOWLEDGE)
        self.assertIs(entities.SECTOR, KNOWLEDGE)
        self.assertIs(conflict.SECTOR, KNOWLEDGE)
        self.assertEqual(applicability.LABELS['BANK'], 'banka')
        self.assertEqual(engine.DEFAULT_CATEGORIES[0], 'Müşterinin tanınması ve kabulü')


class CoreBoundaryTests(unittest.TestCase):
    def core_files(self):
        for part in CORE:
            target = SRC / part
            yield from (target.rglob('*.py') if target.is_dir() else [target])

    def test_the_core_does_not_import_the_financial_pack_or_the_compatibility_layer(self):
        for path in self.core_files():
            text = path.read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'regchain\.packs\b|from \.\.?packs\b|regchain\.compat\b|from \.\.?compat\b', text),
                              path.relative_to(SRC))

    def test_no_generic_engine_module_hard_codes_the_financial_pack_id(self):
        for path in self.core_files():
            self.assertNotIn('FINANCIAL_SERVICES_TR', path.read_text(encoding='utf-8'), path.relative_to(SRC))
        self.assertNotIn('FINANCIAL_SERVICES_TR', (SRC / 'sector.py').read_text(encoding='utf-8'))
        self.assertNotIn('FINANCIAL_SERVICES_TR', (SRC / 'tr' / 'packs.py').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()

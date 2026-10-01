"""Sector engines over shared expert services (engines.py): which packs read a profile, that every row of a run says
which pack read it, and that a recorded run replays with no model. The registers are synthetic."""
import unittest

from regchain.tr.compare import load_register
from regchain.tr.corpus import CorpusStore
from regchain.tr.engines import ENGINES_VERSION, ExpertServices, SectorEngine, assess_by_engine, engines_for
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles

REGISTRY = Registry.load()
STORE = CorpusStore()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
SERVICES = ExpertServices(REGISTRY, STORE)


class EngineSelectionTests(unittest.TestCase):
    def test_a_profile_is_read_by_the_packs_it_falls_under(self):
        brewer = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
        bottler = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
        brewer_packs = {engine.pack_id: selection.status for engine, selection in engines_for(brewer, SERVICES)}
        bottler_packs = {engine.pack_id: selection.status for engine, selection in engines_for(bottler, SERVICES)}
        self.assertIn('BEVERAGE_ALCOHOL_TR', brewer_packs)
        self.assertIn('BEVERAGE_NON_ALCOHOL_TR', bottler_packs)
        self.assertNotIn('BEVERAGE_ALCOHOL_TR', bottler_packs)               # a bottler of soft drinks is not read by the alcohol engine
        self.assertEqual(set(brewer_packs.values()) | set(bottler_packs.values()), {'SELECTED'})

    def test_an_engine_reads_the_regulations_its_pack_carries_and_no_other(self):
        engine = SectorEngine('BEVERAGE_ALCOHOL_TR', SERVICES)
        self.assertTrue(engine.regulation_ids)
        self.assertLessEqual(set(engine.regulation_ids), set(REGISTRY.regulation_ids('BEVERAGE_ALCOHOL_TR')))
        self.assertIn('TR:KANUN:4250', engine.regulation_ids)
        other = SectorEngine('BEVERAGE_NON_ALCOHOL_TR', SERVICES)
        self.assertNotIn('TR:KANUN:4250', other.regulation_ids)
        self.assertEqual({o.regulation_id for o in engine.obligations()} - set(engine.regulation_ids), set())


class EngineRunTests(unittest.TestCase):
    def test_a_run_keeps_one_record_per_engine_with_its_statistics(self):
        profile = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
        result = assess_by_engine(profile, SERVICES, load_register(profile.profile_id))
        self.assertEqual(result['version'], ENGINES_VERSION)
        self.assertEqual([run.pack_id for run in result['engines']], list(result['reports']))
        for run in result['engines']:
            report = result['reports'][run.pack_id]
            self.assertEqual(run.statistics['rows'], len(report.rows))
            self.assertEqual(len(result['assessments'][run.pack_id]), len(report.rows))
            self.assertGreater(run.obligations, 0)
            self.assertEqual(run.statistics['adjudications'], 0)                   # no model in these services
            self.assertEqual(run.statistics['rows_not_verified'], 0)              # every row rests on exact spans and its gates
            self.assertEqual(run.selection.status, 'SELECTED')
        rows = [r for report in result['reports'].values() for r in report.rows]
        self.assertTrue(all(r.regulation_id in set(REGISTRY.regulation_ids(pack)) for pack, report in result['reports'].items() for r in report.rows))
        self.assertTrue(rows)

    def test_recorded_services_replay_without_a_model(self):
        from regchain.tr import devset
        recorded = devset.TR_DATA / 'evaluation' / 'recorded' / '20261001'
        services = ExpertServices.recorded(recorded / 'similarities.json', [recorded / 'adjudicate-dev.jsonl'], REGISTRY, STORE)
        self.assertIsNone(services.adjudicator.provider)
        self.assertTrue(services.adjudicator.recorded)
        self.assertTrue(services.table.pairs)


if __name__ == '__main__':
    unittest.main()

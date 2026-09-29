"""Policy/control/evidence mapping and stored-source version impact (Cardaman TR steps 15–16)."""
import unittest

from regchain.tr.change import AffectedTarget, SourceVersion, attach_targets, compare_versions
from regchain.tr.mapping import Control, EvidenceItem, map_obligation, mapping_status


class MappingTests(unittest.TestCase):
    def test_a_weaker_retention_period_is_contradicted(self):
        # Engine already judged CONFLICT (five years vs three years). Mapping does not re-read the wording.
        status, reasons = mapping_status('CONFLICT', True, True)
        self.assertEqual((status, reasons), ('CONTRADICTED', ['POLICY_CONFLICT']))

    def test_a_generic_records_system_is_not_covered(self):
        # "company operates a records-management system" is NO_EVIDENCE or PARTIAL, never COVERS_TEXT.
        self.assertEqual(mapping_status('NO_EVIDENCE', False, False)[0], 'NOT_COVERED')
        self.assertEqual(mapping_status('PARTIAL', False, False)[0], 'PARTIALLY_COVERED')
        self.assertNotEqual(mapping_status('NO_EVIDENCE', False, False)[0], 'COVERED')

    def test_covers_text_still_needs_a_control_and_in_period_evidence(self):
        self.assertEqual(mapping_status('COVERS_TEXT', False, False),
                         ('PARTIALLY_COVERED', ['POLICY_COVERS', 'CONTROL_MISSING']))
        self.assertEqual(mapping_status('COVERS_TEXT', True, False),
                         ('PARTIALLY_COVERED', ['POLICY_COVERS', 'EVIDENCE_MISSING']))
        self.assertEqual(mapping_status('COVERS_TEXT', True, True)[0], 'COVERED')

    def test_unknown_coverage_stays_unknown(self):
        self.assertEqual(mapping_status('UNKNOWN', True, True)[0], 'UNKNOWN')

    def test_map_obligation_records_the_chain(self):
        control = Control(control_id='CTL-LABEL', description='Legal and Quality approval before label release',
                          owner_department='QUALITY', documents=['SOP-LABEL-01'], evidence_types=['APPROVED_LABEL_ARTWORK'])
        evidence = EvidenceItem(evidence_id='EV-1', control_id='CTL-LABEL', evidence_type='APPROVED_LABEL_ARTWORK',
                                period='2026-Q3', content_hash='b' * 64, in_period=True)
        mapped = map_obligation('OBL-1', 'COVERS_TEXT', [control], [evidence], departments=['QUALITY', 'LEGAL'],
                                documents=['SOP-LABEL-01'])
        self.assertEqual(mapped.status, 'COVERED')
        self.assertEqual(mapped.control_ids, ['CTL-LABEL'])
        self.assertEqual(mapped.evidence_ids, ['EV-1'])


class ChangeTests(unittest.TestCase):
    def versions(self, old_hash='a' * 64, new_hash='b' * 64):
        old = SourceVersion(regulation_id='TR:TEBLIG:TGK_ENERJI_ICECEKLERI', version_id='v1', content_hash=old_hash)
        new = SourceVersion(regulation_id='TR:TEBLIG:TGK_ENERJI_ICECEKLERI', version_id='v2', content_hash=new_hash,
                            previous_version_id='v1')
        return old, new

    def test_identical_hashes_are_not_a_change(self):
        old, new = self.versions('a' * 64, 'a' * 64)
        record = compare_versions(old, new)
        self.assertFalse(record.hash_changed)
        self.assertEqual(record.provision_changes, [])

    def test_a_hash_change_names_the_version_pair(self):
        record = compare_versions(*self.versions())
        self.assertTrue(record.hash_changed)
        self.assertEqual(record.from_version, 'v1')
        self.assertEqual(record.to_version, 'v2')
        self.assertEqual(record.provision_changes[0].kind, 'MODIFIED')

    def test_a_change_without_a_reachable_target_is_reported_not_dropped(self):
        record = attach_targets(compare_versions(*self.versions()), [])
        self.assertEqual(record.affected[0].delta, 'NO_TARGET_AFFECTED')

    def test_affected_products_and_facilities_travel_with_the_obligation(self):
        record = attach_targets(compare_versions(*self.versions()), [
            AffectedTarget(obligation_id='ENERGY_DRINK_PRODUCT', delta='CHANGED_REQUIREMENT',
                           entity_ids=['NONALC-BOTTLING'], facility_ids=['NONALC-PLANT-1'],
                           product_ids=['NONALC-ENERGY'], document_ids=['SPEC-ENERGY-LABEL'],
                           departments=['QUALITY'])])
        self.assertEqual(record.affected[0].product_ids, ['NONALC-ENERGY'])
        self.assertEqual(record.affected[0].facility_ids, ['NONALC-PLANT-1'])
        self.assertEqual(record.affected[0].departments, ['QUALITY'])


if __name__ == '__main__':
    unittest.main()

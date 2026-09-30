"""Regulatory change at clause level: changed clause -> changed obligation -> affected product, entity, facility,
policy and control.

The amended versions are synthetic (tr_fixtures.amend re-renders the stored articles with edited lines in a temporary
store); they test the change logic and are not real amendments. The amendment-note tests read the packaged official
texts, whose notes are real.
"""
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from regchain.tr.amendments import clause_amendments, notes_in
from regchain.tr.change import clause_changes, impact, note_changes, version_changes
from regchain.tr.clauses import split_clauses
from regchain.tr.compare import load_register
from regchain.tr.corpus import CorpusStore
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles
from tr_fixtures import amend, seed

REGISTRY = Registry.load()
PACKAGED = CorpusStore()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
BREWER = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
BOTTLER = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
ENERGY = 'TR:TEBLIG:TGK_ENERJI_ICECEKLERI'
EDITS = {'Tebliğ 23706 md. 5': [('150 mg/L', '120 mg/L')],
         'Tebliğ 23706 md. 11': [('Spor tesislerinde, okul kantinlerinde ve hastanelerde',
                                  'Spor tesislerinde, okul kantinlerinde, öğrenci yurtlarında ve hastanelerde')],
         'Tebliğ 23706 md. 13': [(' Ancak ‘enerji’ için yapılabilen beslenme beyanı ifadeleri enerji içeceği için yapılamaz.', '')]}
ADDITIONS = {'Tebliğ 23706 md. 11': ['(5) Enerji içecekleri otomatik satış makineleri ile satılamaz.']}


class AmendmentNoteTests(unittest.TestCase):
    def test_a_parenthesis_may_hold_two_dated_notes(self):
        found = notes_in('(Mülga: 11/1/2001-4619/5 md.; Yeniden düzenleme: 24/5/2013-6487/2 md.) Reklamı yapılamaz.')
        self.assertEqual([(n.action, n.date, n.instrument) for n in found],
                         [('REPEALED', date(2001, 1, 11), '4619/5 md'), ('REENACTED', date(2013, 5, 24), '6487/2 md')])
        gazette = notes_in('(2) (Değişik:RG-18/9/2013-28769) Meskûn mahaller hariç ...')[0]
        self.assertEqual((gazette.action, gazette.unit, gazette.date, gazette.instrument), ('CHANGED', 'UNSPECIFIED', date(2013, 9, 18), '28769'))
        self.assertEqual(notes_in('(Ek cümle:11/6/2026-7584/2 md.) X')[0].unit, 'SENTENCE')
        self.assertEqual(notes_in('Bir (örnek) parantez.'), [])

    def test_a_sentence_note_reaches_its_sentence_and_the_article_note_every_clause(self):
        section = PACKAGED.section('TR:KANUN:4250', 'Kanun 4250 md. 6')
        notes = clause_amendments(section)
        recent = {ref.split('/', 1)[1]: [(n.action, n.unit) for n in found if n.date >= date(2026, 1, 1)] for ref, found in notes.items()}
        self.assertEqual({ref: value for ref, value in recent.items() if value},
                         {'f.1/c.4': [('CHANGED', 'SENTENCE')], 'f.1/c.5': [('ADDED', 'SENTENCE')], 'f.9/c.2': [('ADDED', 'SENTENCE')]})
        for found in notes.values():                                      # the 2013 re-enactment stands over the whole article
            self.assertIn(('REENACTED', date(2013, 5, 24)), [(n.action, n.date) for n in found])

    def test_a_note_at_the_head_of_a_fikra_reaches_its_bents(self):
        section = PACKAGED.section('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 10')
        notes = clause_amendments(section)
        for bent in ('a', 'b', 'c', 'ç', 'd'):
            self.assertEqual([(n.action, n.date) for n in notes[f'Yönetmelik 14646 md. 10/f.3/b.{bent}']], [('CHANGED', date(2023, 9, 20))])
        self.assertEqual(notes['Yönetmelik 14646 md. 10/f.1'], [])
        self.assertEqual({n.date for ref, found in notes.items() if '/f.5/' in ref for n in found}, {date(2013, 9, 18)})


class VersionDiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.store = CorpusStore(Path(cls.directory.name))
        cls.real = seed(cls.store, REGISTRY, ENERGY)
        cls.same = amend(cls.store, REGISTRY, ENERGY, when=datetime(2026, 10, 1, tzinfo=timezone.utc))
        cls.amended = amend(cls.store, REGISTRY, ENERGY, EDITS, ADDITIONS)
        cls.record, cls.old, cls.new = version_changes(ENERGY, cls.same.version_id, cls.amended.version_id, REGISTRY, cls.store)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_a_re_rendered_text_has_no_changed_clause(self):
        self.assertEqual(clause_changes(self.store.sections(ENERGY, self.real.version_id),
                                        self.store.sections(ENERGY, self.same.version_id)), [])
        # Re-rendering the stored articles reproduces the parsed text exactly, so the store keeps one version.
        self.assertEqual(self.same.version_id, self.real.version_id)
        self.assertTrue(self.amended.synthetic and not self.real.synthetic)

    def test_the_changed_clauses_are_named_with_old_and_new_wording(self):
        changes = {c.provision_ref: c for c in self.record.provision_changes}
        self.assertEqual({ref: c.kind for ref, c in changes.items()},
                         {'Tebliğ 23706 md. 5/f.1/b.b': 'MODIFIED', 'Tebliğ 23706 md. 11/f.2': 'MODIFIED',
                          'Tebliğ 23706 md. 11/f.5': 'ADDED', 'Tebliğ 23706 md. 13/f.1/c.2': 'REMOVED'})
        limit = changes['Tebliğ 23706 md. 5/f.1/b.b']
        self.assertIn('150 mg/L', limit.old_text)
        self.assertIn('120 mg/L', limit.new_text)
        self.assertNotEqual(limit.old_hash, limit.new_hash)
        self.assertEqual(limit.basis, 'VERSION_DIFF')
        self.assertTrue(self.record.hash_changed and self.record.synthetic)

    def test_each_changed_clause_says_what_changed_in_the_duty(self):
        duties = {(d.provision_ref, d.group): d for d in self.record.obligation_changes}
        limit = duties[('Tebliğ 23706 md. 5/f.1/b.b', 'PRODUCT')]
        self.assertEqual((limit.kind, limit.impact_types), ('MODIFIED', ['LIMIT_TIGHTENED']))
        self.assertEqual(limit.element_changes, [{'element': 'limit', 'attribute': 'caffeine_mg_per_l', 'comparator': 'le',
                                                  'old': 150.0, 'new': 120.0, 'direction': 'TIGHTENED'}])
        self.assertEqual(duties[('Tebliğ 23706 md. 11/f.2', 'MARKETING')].impact_types, ['PLACES_WIDENED'])
        self.assertEqual(duties[('Tebliğ 23706 md. 11/f.5', 'SALES')].impact_types, ['NEW_DUTY'])
        self.assertEqual(duties[('Tebliğ 23706 md. 13/f.1/c.2', 'PRODUCT')].impact_types, ['DUTY_REMOVED'])
        by_clause = {c.provision_ref: c.impact_types for c in self.record.provision_changes}
        self.assertEqual(by_clause['Tebliğ 23706 md. 5/f.1/b.b'], ['LIMIT_TIGHTENED'])

    def test_the_reverse_change_relaxes_the_limit(self):
        record, _, _ = version_changes(ENERGY, self.amended.version_id, self.same.version_id, REGISTRY, self.store)
        kinds = {(d.provision_ref, d.group): d.impact_types for d in record.obligation_changes}
        self.assertEqual(kinds[('Tebliğ 23706 md. 5/f.1/b.b', 'PRODUCT')], ['LIMIT_RELAXED'])
        self.assertEqual(kinds[('Tebliğ 23706 md. 11/f.5', 'SALES')], ['DUTY_REMOVED'])
        self.assertEqual(kinds[('Tebliğ 23706 md. 13/f.1/c.2', 'PRODUCT')], ['NEW_DUTY'])

    def test_the_impact_names_products_facilities_entities_documents_and_controls(self):
        result = impact(self.record, self.old, self.new, BOTTLER, REGISTRY, self.store, load_register(BOTTLER.profile_id))
        rows = {(a.provision_ref, a.target_id): a for a in result.affected}
        energy = rows[('Tebliğ 23706 md. 5/f.1/b.b', 'NONALC-ENERGY')]
        self.assertEqual((energy.decision_before, energy.decision_after, energy.delta), ('APPLIES', 'APPLIES', 'CHANGED_REQUIREMENT'))
        self.assertEqual(energy.product_ids, ['NONALC-ENERGY'])
        self.assertIn('NONALC-PLANT-1', energy.facility_ids)                     # the plant that fills the energy drink
        self.assertNotIn('NONALC-WATER-PLANT', energy.facility_ids)
        self.assertEqual((energy.document_ids, energy.mapping_after), (['SPEC-ENERGY-01'], 'CONTRADICTED'))
        self.assertEqual(energy.control_ids, ['CTL-ENERGY-BATCH', 'CTL-LABEL-APPROVAL'])
        self.assertIn('RND', energy.departments)
        actions = [(a['action'], a['target']) for a in energy.actions]
        self.assertIn(('UPDATE_DOCUMENT', 'SPEC-ENERGY-01#1'), actions)
        # 150 mg/L was within the old limit and is above the new one: the change itself creates the finding.
        self.assertIn(('PRODUCT_NONCONFORMITY', 'NONALC-ENERGY'), actions)

    def test_a_new_clause_newly_applies_and_a_removed_one_no_longer_applies(self):
        result = impact(self.record, self.old, self.new, BOTTLER, REGISTRY, self.store, load_register(BOTTLER.profile_id))
        rows = {(a.provision_ref, a.target_id): a for a in result.affected}
        vending = rows[('Tebliğ 23706 md. 11/f.5', 'NONALC-ONLINE')]
        self.assertEqual((vending.decision_before, vending.delta, vending.mapping_after), (None, 'NEWLY_APPLIES', 'NOT_COVERED'))
        self.assertEqual(vending.actions[0]['action'], 'ADD_POLICY_STATEMENT')
        claim = rows[('Tebliğ 23706 md. 13/f.1/c.2', 'NONALC-ENERGY')]
        self.assertEqual((claim.decision_before, claim.decision_after, claim.delta), ('APPLIES', None, 'NO_LONGER_APPLIES'))
        self.assertEqual(result.summary['targets_by_delta']['NO_LONGER_APPLIES'], 2)
        self.assertEqual(result.summary['products'], ['NONALC-ENERGY', 'NONALC-ENERGY-MAX'])
        self.assertTrue({'NONALC-BOTTLING', 'NONALC-SALES', 'NONALC-ONLINE'} <= set(result.summary['entities']))

    def test_an_undecided_target_is_unknown_and_names_nothing_as_affected(self):
        result = impact(self.record, self.old, self.new, BOTTLER, REGISTRY, self.store, load_register(BOTTLER.profile_id))
        brand = next(a for a in result.affected if a.target_id == 'NONALC-BRAND' and a.provision_ref == 'Tebliğ 23706 md. 11/f.5')
        self.assertEqual((brand.delta, brand.product_ids, brand.facility_ids), ('UNKNOWN', [], []))     # an incomplete entity profile

    def test_a_change_that_reaches_no_target_is_reported_not_dropped(self):
        result = impact(self.record, self.old, self.new, BREWER, REGISTRY, self.store, load_register(BREWER.profile_id))
        self.assertEqual({a.delta for a in result.affected}, {'NO_TARGET_AFFECTED'})
        self.assertEqual(len(result.affected), len(self.record.obligation_changes))
        self.assertEqual(result.summary['products'], [])

    def test_a_sentence_inserted_before_a_clause_does_not_change_the_clause_after_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CorpusStore(Path(directory))
            seed(store, REGISTRY, ENERGY)
            base = amend(store, REGISTRY, ENERGY, when=datetime(2026, 10, 1, tzinfo=timezone.utc))
            moved = amend(store, REGISTRY, ENERGY, {'Tebliğ 23706 md. 11': [
                ('(4) Bu Tebliğin', '(4) Enerji içecekleri çocuk menülerinde yer alamaz. Bu Tebliğin')]})
            changes = clause_changes(store.sections(ENERGY, base.version_id), store.sections(ENERGY, moved.version_id))
            self.assertEqual([(c.kind, c.provision_ref) for c in changes], [('ADDED', 'Tebliğ 23706 md. 11/f.4/c.1')])

    def test_the_same_text_twice_changes_nothing(self):
        record, _, _ = version_changes(ENERGY, self.same.version_id, self.same.version_id, REGISTRY, self.store)
        self.assertFalse(record.hash_changed)
        self.assertEqual((record.provision_changes, record.obligation_changes), ([], []))
        result = impact(record, [], [], BOTTLER, REGISTRY, self.store)
        self.assertEqual([a.delta for a in result.affected], ['UNCHANGED'])


class AmendmentNoteChangeTests(unittest.TestCase):
    """Kanun 4250 md. 6 as amended by Kanun 7584 on 11 June 2026, read from the notes of the stored text."""

    @classmethod
    def setUpClass(cls):
        cls.record, cls.old, cls.new = note_changes('TR:KANUN:4250', date(2026, 1, 1), REGISTRY, PACKAGED)

    def test_the_notes_name_the_clauses_an_instrument_changed_since_a_date(self):
        changes = {c.provision_ref: c for c in self.record.provision_changes}
        self.assertEqual(changes['Kanun 4250 md. 6/f.1/c.4'].kind, 'MODIFIED')
        self.assertEqual(changes['Kanun 4250 md. 6/f.1/c.5'].kind, 'ADDED')
        self.assertEqual(changes['Kanun 4250 md. 6/f.9/c.2'].kind, 'ADDED')
        for change in changes.values():
            self.assertEqual((change.basis, change.amended_on), ('AMENDMENT_NOTE', date(2026, 6, 11)))
            self.assertTrue(change.instrument.startswith('7584'))
            self.assertIsNone(change.old_text)                                  # the old wording is not in the text
        self.assertNotIn('Kanun 4250 md. 6/f.1/c.1', changes)                   # the 2013 advertising ban did not change
        self.assertFalse(self.record.synthetic)
        self.assertEqual(note_changes('TR:KANUN:4250', date(2026, 7, 1), REGISTRY, PACKAGED)[0].provision_changes, [])

    def test_the_changed_sponsorship_sentence_reaches_the_group_and_its_marketing_policy(self):
        result = impact(self.record, self.old, self.new, BREWER, REGISTRY, PACKAGED, load_register(BREWER.profile_id))
        rows = {(a.provision_ref, a.target_id): a for a in result.affected}
        holding = rows[('Kanun 4250 md. 6/f.1/c.4', 'ALC-INT-HOLDING')]
        self.assertEqual((holding.delta, holding.mapping_after, holding.document_ids), ('CHANGED_REQUIREMENT', 'CONTRADICTED', ['POL-MKT-01']))
        self.assertEqual([(a['action'], a['target'], a['department']) for a in holding.actions], [('UPDATE_DOCUMENT', 'POL-MKT-01#2', 'MARKETING')])
        brewing = rows[('Kanun 4250 md. 6/f.1/c.4', 'ALC-INT-BREWING')]
        self.assertEqual(brewing.facility_ids, ['ALC-INT-BREWERY-1', 'ALC-INT-BREWERY-2'])
        self.assertNotIn('ALC-INT-AF', brewing.product_ids)                     # the alcohol-free beer is not an alcoholic drink
        added = rows[('Kanun 4250 md. 6/f.9/c.2', 'ALC-INT-ACT-AF-BRAND')]
        self.assertEqual((added.delta, added.mapping_after), ('NEWLY_APPLIES', 'NOT_COVERED'))

    def test_a_changed_clause_the_rules_cannot_address_is_one_review_row_for_the_group(self):
        result = impact(self.record, self.old, self.new, BREWER, REGISTRY, PACKAGED, load_register(BREWER.profile_id))
        rows = [a for a in result.affected if a.provision_ref == 'Kanun 4250 md. Geçici 2/f.1']
        self.assertEqual([(a.level, a.target_id, a.delta) for a in rows], [('GROUP', 'GRP-ALC-INT', 'UNKNOWN')])
        self.assertEqual(rows[0].actions[0]['why'], ['REGULATORY_SCOPE_UNCLEAR'])

    def test_the_soft_drink_group_is_not_reached_by_the_alcohol_amendment(self):
        result = impact(self.record, self.old, self.new, BOTTLER, REGISTRY, PACKAGED, load_register(BOTTLER.profile_id))
        self.assertTrue(all(a.delta in ('NO_TARGET_AFFECTED', 'UNKNOWN') for a in result.affected))
        self.assertEqual((result.summary['products'], result.summary['documents']), ([], []))


if __name__ == '__main__':
    unittest.main()

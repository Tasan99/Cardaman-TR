"""Linking an amendment instrument to the consolidated provisions it cites.

Fixtures imitate the structure of a real FCA instrument. They are not FCA law, and the
assertions are about traceability, never about whether an obligation legally changed.
"""
import unittest
from datetime import date
from hashlib import sha256

from regchain.evidence import digest
from regchain.extraction.amendments import (MATCHER_VERSION, analyse, annex_books,
                                            annex_of, commencement_date)

PREAMBLE = ('CONSUMER CREDIT AND MORTGAGES (TAILORED SUPPORT) INSTRUMENT Powers exercised A. '
            'The Financial Conduct Authority makes this instrument. Commencement C. This '
            'instrument comes into force on 4 November 2024. Citation F. This instrument may '
            'be cited as the Fixture Instrument 2024.')


def paragraph(sid, text, label=None, annex='Annex D', kind='AMENDMENT', **values):
    values.setdefault('section_number', annex or 'preamble')
    values.setdefault('heading_path', [annex] if annex else [])
    values.setdefault('paragraph_number', label or sid)
    return dict(id=sid, version_id='instrument', text=text, printed_label=label,
                source_kind=kind, content_hash=sha256(text.encode()).hexdigest(), **values)


def provision(sid, text, label, effective, legal_type='RULE', version='FCA-HB-CONC-7-3'):
    book, numeric = label.split(' ', 1)
    return dict(id=sid, version_id=version, text=text, printed_label=label,
                section_number=f"{book} {'.'.join(numeric.split('.')[:2])}",
                paragraph_number=numeric, heading_path=[book], source_kind='CONSOLIDATED',
                legal_type=legal_type, effective_from=effective, quality_flags=[],
                external_key=version, version_hash='0' * 64,
                content_hash=sha256(text.encode()).hexdigest())


INSTRUMENT = [
    paragraph('p0', PREAMBLE, annex=None, kind='COMMENTARY'),
    paragraph('p1', 'Annex C Amendments to the Mortgages and Home Finance: Conduct of '
                    'Business sourcebook (MCOB)', annex='Annex C'),
    paragraph('p2', '13.3.1 R A firm must establish a policy.', '13.3.1', annex='Annex C'),
    paragraph('p3', 'Annex D Amendments to the Consumer Credit sourcebook (CONC)', annex='Annex D'),
    paragraph('p4', '7.3 Treatment of customers in arrears', '7.3', annex='Annex D'),
    paragraph('p5', '7.3.4 R A firm must treat customers with forbearance.', '7.3.4', annex='Annex D'),
    paragraph('p6', '7.3.2 G A firm should pay due regard.', '7.3.2', annex='Annex D'),
]

CORPUS = [
    provision('c1', 'A firm must establish a policy.', 'MCOB 13.3.1', date(2024, 11, 4),
              version='FCA-HB-MCOB-13-3'),
    provision('c2', 'A firm must treat customers with forbearance.', 'CONC 7.3.4', date(2024, 11, 4)),
    provision('c3', 'A firm should pay due regard.', 'CONC 7.3.2', date(2026, 6, 26), 'GUIDANCE'),
]


class InstrumentReadingTests(unittest.TestCase):
    def test_commencement_is_read_from_the_instrument_clause(self):
        self.assertEqual(commencement_date(INSTRUMENT), date(2024, 11, 4))

    def test_commentary_about_timing_is_not_a_commencement_clause(self):
        prose = [paragraph('x', '1.22 The rules come into force on 4 November 2024.',
                           annex=None, kind='COMMENTARY')]
        self.assertIsNone(commencement_date(prose))

    def test_annex_headings_name_their_own_sourcebook(self):
        self.assertEqual(annex_books(INSTRUMENT), {'C': 'MCOB', 'D': 'CONC'})
        self.assertEqual(annex_books([paragraph('g', 'Annex A Amendments to the Glossary of '
                                                     'definitions', annex='Annex A')]), {'A': 'GLOSSARY'})

    def test_annex_of_reads_the_paragraph_heading(self):
        self.assertEqual(annex_of(INSTRUMENT[2]), 'C')
        self.assertIsNone(annex_of(INSTRUMENT[0]))


class AmendmentLinkTests(unittest.TestCase):
    def setUp(self):
        self.analysis = analyse(INSTRUMENT, CORPUS)

    def test_locator_is_qualified_by_its_annex_sourcebook(self):
        self.assertEqual(sorted(link['locator'] for link in self.analysis.links),
                         ['CONC 7.3.2', 'CONC 7.3.4', 'MCOB 13.3.1'])

    def test_matching_commencement_date_corroborates_the_link(self):
        link = next(l for l in self.analysis.links if l['locator'] == 'CONC 7.3.4')
        self.assertEqual(link['date_agreement'], 'MATCH')
        self.assertEqual(link['consolidated_effective_from'], '2024-11-04')
        self.assertEqual(link['legal_type'], 'RULE')

    def test_a_later_effective_date_is_reported_as_superseded(self):
        link = next(l for l in self.analysis.links if l['locator'] == 'CONC 7.3.2')
        self.assertEqual(link['date_agreement'], 'SUPERSEDED')

    def test_an_earlier_effective_date_is_reported_as_predating(self):
        corpus = [provision('c9', 'Old text.', 'CONC 7.3.4', date(2014, 4, 1))]
        link = analyse(INSTRUMENT, corpus).links[0]
        self.assertEqual(link['date_agreement'], 'PREDATES')

    def test_without_a_commencement_clause_no_date_claim_is_made(self):
        instrument = [p for p in INSTRUMENT if p['id'] != 'p0']
        self.assertIsNone(analyse(instrument, CORPUS).commencement)
        self.assertEqual({l['date_agreement'] for l in analyse(instrument, CORPUS).links}, {'UNKNOWN'})

    def test_section_heading_inside_an_annex_is_not_linked(self):
        gap = next(g for g in self.analysis.unmatched if g['printed_label'] == '7.3')
        self.assertEqual(gap['reason'], 'SECTION_HEADING_NOT_A_PROVISION')

    def test_uningested_chapter_is_reported_with_its_chapter(self):
        instrument = INSTRUMENT + [paragraph('p7', '7.9.1 R A firm must report.', '7.9.1')]
        gap = next(g for g in analyse(instrument, CORPUS).unmatched if g.get('locator') == 'CONC 7.9.1')
        self.assertEqual((gap['reason'], gap['chapter']), ('NOT_IN_CONSOLIDATED_CORPUS', 'CONC 7.9'))

    def test_an_ambiguous_locator_is_not_linked_to_either_reading(self):
        instrument = [INSTRUMENT[0], INSTRUMENT[3], paragraph('p8', '7.3.5G G Text.', '7.3.5G')]
        corpus = [provision('c4', 'Sub-lettered.', 'CONC 7.3.5G', date(2024, 11, 4)),
                  provision('c5', 'Base guidance.', 'CONC 7.3.5', date(2024, 11, 4), 'GUIDANCE')]
        analysis = analyse(instrument, corpus)
        self.assertEqual(analysis.links, ())
        self.assertEqual(analysis.unmatched[0]['reason'], 'AMBIGUOUS_IN_CONSOLIDATED_CORPUS')

    def test_a_type_letter_in_the_instrument_locator_still_matches(self):
        instrument = [INSTRUMENT[0], INSTRUMENT[3], paragraph('p9', '7.3.4R R Text.', '7.3.4R')]
        self.assertEqual(analyse(instrument, CORPUS).links[0]['locator'], 'CONC 7.3.4R')
        self.assertEqual(analyse(instrument, CORPUS).links[0]['printed_label'], 'CONC 7.3.4')

    def test_glossary_amendments_are_recorded_but_not_linked_to_provisions(self):
        instrument = [INSTRUMENT[0],
                      paragraph('g0', 'Annex A Amendments to the Glossary of definitions', annex='Annex A'),
                      paragraph('g1', '1.1.1 priority debt means rent.', '1.1.1', annex='Annex A')]
        analysis = analyse(instrument, CORPUS)
        self.assertEqual(analysis.links, ())
        self.assertEqual(analysis.unmatched[0]['reason'], 'GLOSSARY_AMENDMENT_NOT_LINKED')

    def test_commentary_paragraphs_are_never_linked(self):
        instrument = [INSTRUMENT[0], INSTRUMENT[3],
                      paragraph('c', '7.3.4 R A firm must act.', '7.3.4', kind='COMMENTARY')]
        self.assertEqual(analyse(instrument, CORPUS).links, ())

    def test_manifest_is_canonically_hashable_for_the_audit_record(self):
        # A date or float leaking into the manifest would break the evidence format.
        self.assertRegex(digest(self.analysis.manifest()), r'^[0-9a-f]{64}$')

    def test_manifest_records_the_matcher_and_every_gap(self):
        manifest = self.analysis.manifest()
        self.assertEqual(manifest['matcher_version'], MATCHER_VERSION)
        self.assertEqual(manifest['commencement'], '2024-11-04')
        self.assertEqual(manifest['annexes'], {'C': 'MCOB', 'D': 'CONC'})
        self.assertEqual(len(manifest['links']) + len(manifest['unmatched']),
                         sum(1 for p in INSTRUMENT if p['source_kind'] == 'AMENDMENT' and p['printed_label']))


if __name__ == '__main__':
    unittest.main()

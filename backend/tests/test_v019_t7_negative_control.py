"""v0.19 t7 (26 September 2026): the applicability negative control. A clause addressed only to the obliged parties does not
inherit a favourable provision-level answer that may rest on another sub-paragraph addressed to everyone, when the regulation's
obliged-party list does not place the company (applicability.withheld_answer, engine.propose).

Measured on the v019t6 live micro run (independent I08, a restaurant chain): Tedbirler md. 31(1) binds "Kamu kurum ve
kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar", md. 31(3) binds "Yükümlüler" only (gold
DOES_NOT_APPLY). The list could not place the company: its description says it takes card payments ("ödeme", near a listed
kind). The model was asked about md. 31 once and answered APPLIES on (1)'s addressee list ("TR" against "gerçek ve tüzel
kişiler"), and (3) carried that APPLIES (APPLICABILITY_SHARED, applicability_rule MODEL). In t5 the same row was UNKNOWN only
because every scope quote was refused; the t6 quote check (P0) accepted the quote and exposed the propagation. The clause is now
UNKNOWN (PROFILE_AMBIGUOUS, CHILD_ADDRESSEE gate): the rules cannot tell whether the company is an obliged party, and neither
APPLIES nor DOES_NOT_APPLY may rest on that silence. A company the list rules out keeps its rule DOES_NOT_APPLY, an obliged
company keeps its APPLIES, and the sub-paragraphs addressed to everyone keep theirs. The judge is a fake answering by schema.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from regchain.evidence import canonical_bytes
from regchain.pilot import applicability as rules
from regchain.pilot.applicability import OBLIGED_ADDRESSEE_UNDETERMINED, obliged_list, rule_chain, withheld_answer
from regchain.pilot.engine import analyze, pipeline_settings, split_units
from regchain.pilot.entities import entity_gate
from regchain.pilot.schema import Company
from regchain.pilot.sources import application_rows, load_sources, select_targets
from test_pilot import policies
from test_v018_applicability import ASSOCIATION, BANK, SOFTWARE, Scripted, profile
from test_v019_universal_duty import TEDBIRLER, independent_profile, paragraph, section, statuses

# Tedbirler md. 31(1)'s addressee list: the basis of the provision-level APPLIES in the live run.
UNIVERSAL = 'Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar'
LABEL = 'Yönetmelik 200713012 md. 31'
# A company the list cannot place: no listed kind, no word of a kind the list does not oblige, and a word near a listed kind
# (it is paid by card). Not the fixture's profile: the same shape, another business.
CAFE = dict(id='kafe', name='Köşe Kahve Ltd. Şti.', version='1', synthetic=True, jurisdictions=['TR'], activities=['kafe işletmeciliği'],
            licences=['işyeri açma ruhsatı'], products=['kahve', 'pasta'], customer_types=['gerçek kişiler'],
            description='Üç şubeli kahve dükkânı; siparişlerin ödemesini nakit veya kartla alır.')
# Obliged parties of md. 4(1) among the independent cases: insurance, payment institutions, a bank, a crypto provider, factoring.
OBLIGED_CASES = ('I01', 'I02', 'I04', 'I05', 'I06', 'I13')


class ProvisionJudge(Scripted):
    """The live md. 31 answer: `state` on (1)'s addressee list (a YES pairing the profile's jurisdiction with it), its scope quote
    the opening of (1) closed with the cut mark "..." (the form the P0 quote check accepts) or copied exactly. `listed`, a list
    item of md. 4(1), adds a YES on the obliged-party list: the model says the company is one of the obliged parties. `match`
    is the pair's answer on (1)'s list (an UNKNOWN with a YES is stored as POSSIBLY_APPLIES by rule)."""

    def __init__(self, state='APPLIES', quote='cut', listed=None, match='YES'):
        super().__init__(state)
        self.quote, self.listed, self.match = quote, listed, match

    def scope(self, payload):
        company = payload['company']
        fact = (company.get('jurisdictions') or company.get('activities'))[0]
        text = payload['provision']['text']
        opening = text[:text.index('belge') + len('belge')]
        basis = [{'company_fact': fact, 'regulatory_condition': UNIVERSAL, 'match': self.match}]
        if self.listed:
            basis.append({'company_fact': company['activities'][0], 'regulatory_condition': self.listed, 'match': 'YES'})
        return {'applicability': self.applicability, 'company_fact_keys': ['jurisdictions', 'activities'],
                'scope_evidence': [{'source_id': 'p0', 'quote': opening + ('...' if self.quote == 'cut' else '')}], 'basis': basis,
                'applicability_reason': f'Fixture answers {self.applicability}: the provision binds every legal person.',
                'missing_information': []}


class NegativeControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tedbirler = load_sources(TEDBIRLER)[1]
        cls.md31 = section(cls.tedbirler, LABEL)
        cls.scope_rows = application_rows(cls.tedbirler, 'YONETMELIK', '200713012')
        cls.listed = obliged_list(cls.scope_rows)
        cls.restaurant = independent_profile('I08')

    def rows(self, company, judge, clear_match='rule'):
        labels = select_targets(self.tedbirler, 'YONETMELIK', '200713012', 'all', ['31'])
        settings = pipeline_settings(applicability_clear_match=clear_match, relevance_screen='off', coverage_pipeline='v18')
        packet = analyze(company, policies(), self.tedbirler, judge, labels, judge=judge, target_filter=['31'], settings=settings)
        payload = packet['events'][0]['payload']
        rows = {}
        for o in payload['obligations']:
            rows.setdefault(o['proposal']['applicability_scope']['child_clause'], o)
        return payload, rows

    def assert_withheld(self, row, provision_state):
        proposal = row['proposal']
        self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('UNKNOWN', 'PROFILE_AMBIGUOUS'))
        gates = statuses(proposal)
        self.assertEqual([g['gate'] for g in proposal['trace']['gates']][-3:], ['MODEL', 'CHILD_ADDRESSEE', 'FINAL_AGGREGATOR'])
        self.assertEqual((gates['MODEL']['status'], gates['CHILD_ADDRESSEE']['status'], gates['FINAL_AGGREGATOR']['status']),
                         (provision_state, 'UNDETERMINED', 'UNKNOWN'))
        evidence = gates['CHILD_ADDRESSEE']['evidence']
        self.assertEqual((evidence['reason_code'], evidence['subject'], evidence['provision_state']),
                         (OBLIGED_ADDRESSEE_UNDETERMINED, 'Yükümlüler', provision_state))
        self.assertEqual(evidence['source_id'], self.listed.source_id)
        self.assertIn(evidence['universal_addressee'], self.md31['text'])                    # an exact quote of the provision
        self.assertFalse(gates['CHILD_ADDRESSEE']['clear'])
        self.assertEqual((proposal['applicability_scope']['provision_state'], proposal['applicability_scope']['final']),
                         (provision_state, 'UNKNOWN'))
        self.assertEqual(proposal['trace']['decided_by'], 'PROFILE_AMBIGUOUS')
        self.assertIn(OBLIGED_ADDRESSEE_UNDETERMINED, [d.get('code') for d in row['diagnostics']])
        self.assertTrue(any('obliged parties' in item and self.listed.source_label in item for item in proposal['missing_information']))
        self.assertIn('not inherited', proposal['applicability_reason'])
        # The chain and the model's answer are recorded as they were read; only the CHILD_ADDRESSEE gate is added.
        self.assertEqual({k: gates[k]['status'] for k in ('REGULATION_SUBJECT_SCOPE', 'COMPANY_ENTITY', 'CHILD_CLAUSE')},
                         {'REGULATION_SUBJECT_SCOPE': 'UNDETERMINED', 'COMPANY_ENTITY': 'MATCH', 'CHILD_CLAUSE': 'NOT_RESTRICTED'})
        self.assertTrue(proposal['coverage_assessed'])                                       # the policy is still read for the reviewer

    def test_the_negative_control_does_not_inherit_the_provision_level_applies(self):
        for mode in ('rule', 'model'):
            with self.subTest(clear_match=mode):
                judge = ProvisionJudge('APPLIES')
                payload, rows = self.rows(self.restaurant, judge, mode)
                self.assertEqual(judge.scoped, [LABEL])                                          # one provision-level question
                self.assert_withheld(rows['(3)'], 'APPLIES')
                self.assertIn('APPLICABILITY_SHARED', [d.get('code') for d in rows['(3)']['diagnostics']])
                # The sub-paragraphs addressed to everyone keep the answer.
                for clause in ('(1)', '(2)'):
                    proposal = rows[clause]['proposal']
                    self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('APPLIES', 'MODEL'), clause)
                    self.assertNotIn('CHILD_ADDRESSEE', statuses(proposal))
                self.assertEqual(statuses(rows['(1)']['proposal'])['COMPANY_ENTITY']['evidence']['reason_code'], 'UNIVERSAL_ADDRESSEE')
                canonical_bytes(payload)                                                         # digest-safe: strings and ints only
                self.assertTrue(all(isinstance(v, str) for v in statuses(rows['(3)']['proposal'])['CHILD_ADDRESSEE']['evidence'].values()))

    def test_a_p0_quote_acceptance_cannot_turn_the_negative_control_into_applies(self):
        for quote in ('cut', 'exact'):
            for state, recorded in (('APPLIES', 'APPLIES'), ('POSSIBLY_APPLIES', 'POSSIBLY_APPLIES'), ('UNKNOWN', 'POSSIBLY_APPLIES')):
                with self.subTest(quote=quote, state=state):
                    _, rows = self.rows(self.restaurant, ProvisionJudge(state, quote))
                    codes = [d.get('code') for d in rows['(1)']['diagnostics']]
                    self.assertEqual('QUOTE_FULL_SOURCE_OK' in codes, quote == 'cut', codes)     # the P0 reading accepted the quote
                    self.assertNotIn('PROPOSAL_INVALID', codes)
                    self.assert_withheld(rows['(3)'], recorded)                                  # an UNKNOWN with a YES is POSSIBLY_APPLIES
        # An UNKNOWN or a DOES_NOT_APPLY is not touched: the gate only withholds a favourable answer.
        for state, judge in (('UNKNOWN', ProvisionJudge('UNKNOWN', match='UNCLEAR')),
                             ('DOES_NOT_APPLY', Scripted('DOES_NOT_APPLY', no_condition='Yükümlüler yerinde yapılacak denetimler kapsamında'))):
            with self.subTest(state=state):
                _, rows = self.rows(self.restaurant, judge)
                self.assertEqual(statuses(rows['(3)']['proposal'])['MODEL']['status'], state)
                self.assertNotIn('CHILD_ADDRESSEE', statuses(rows['(3)']['proposal']))
                self.assertNotEqual(rows['(3)']['proposal']['applicability'], 'APPLIES')

    def test_the_same_clause_applies_to_an_obliged_company(self):
        companies = [('BANK', profile(**BANK)), ('PAYMENT', profile())]
        companies += [(prefix, independent_profile(prefix)) for prefix in OBLIGED_CASES]
        for name, company in companies:
            with self.subTest(company=name):
                _, rows = self.rows(company, ProvisionJudge('APPLIES'))
                proposal = rows['(3)']['proposal']
                self.assertEqual(statuses(proposal)['REGULATION_SUBJECT_SCOPE']['status'], 'MATCH')
                self.assertEqual(proposal['applicability'], 'APPLIES')
                self.assertNotIn('CHILD_ADDRESSEE', statuses(proposal))
                self.assertNotIn(OBLIGED_ADDRESSEE_UNDETERMINED, json.dumps(rows['(3)']['diagnostics']))

    def test_a_company_the_list_rules_out_keeps_its_rule_does_not_apply(self):
        for changes in (SOFTWARE, ASSOCIATION):
            with self.subTest(company=changes['id']):
                _, rows = self.rows(profile(**changes), ProvisionJudge('APPLIES'))
                proposal = rows['(3)']['proposal']
                self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('DOES_NOT_APPLY', 'SUBJECT_SCOPE_GATE'))
                self.assertEqual(statuses(proposal)['MODEL']['status'], 'NOT_ASKED')
                self.assertEqual(rows['(1)']['proposal']['applicability'], 'APPLIES')         # the universal duty still binds it

    def test_a_model_that_reads_the_company_on_the_obliged_party_list_keeps_its_answer(self):
        """The shared answer is withheld because it may rest on (1); a verified YES on the list says the company is an obliged
        party, so the answer is about this clause too (the model decides what the rules leave open, as before)."""
        item = next(i['text'] for i in self.listed.items if i['text'].startswith('a) Bankalar'))
        _, rows = self.rows(self.restaurant, ProvisionJudge('APPLIES', listed=item))
        proposal = rows['(3)']['proposal']
        self.assertTrue(any(b['match'] == 'YES' and b['source_id'] == self.listed.source_id for b in proposal['basis']))
        self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('APPLIES', 'MODEL'))
        self.assertNotIn('CHILD_ADDRESSEE', statuses(proposal))

    def chain(self, text, company, subject, clause=None):
        section_row = {'id': 'x', 'printed_label': 'Yönetmelik 200713012 md. 99', 'heading_path': ['YÖNETMELİK', 'BEŞİNCİ BÖLÜM', 'Bilgi'],
                       'text': text}
        clause = clause or text
        candidate = {'subject': subject, 'source_quote': clause}
        gate = entity_gate(candidate, clause, 'Bilgi', company, section_row['printed_label'])
        return rule_chain(company, section_row, candidate, clause, gate, self.listed, True)

    def test_the_rule_reads_only_an_obliged_clause_beside_a_sub_paragraph_addressed_to_everyone(self):
        cafe = Company(**CAFE)
        provision = ('(1) Herkes, Başkanlıkça istenen bilgi ve belgeleri vermekle yükümlüdür. '
                     '(2) Yükümlüler, kayıtlarını denetime hazır bulundurmak zorundadır.')
        units = split_units(provision)
        second = units[1][1]
        chain = self.chain(provision, cafe, 'Yükümlüler', second)
        self.assertEqual((chain.decision, chain.obliged_addressee, chain.gates[0]['status']), ('OPEN', True, 'UNDETERMINED'))
        basis = [SimpleNamespace(match='YES', source_id='x')]
        record = withheld_answer(chain, units, cafe, 'APPLIES', basis, self.listed)
        self.assertEqual((record['gate'], record['status'], record['evidence']['universal_addressee']), ('CHILD_ADDRESSEE', 'UNDETERMINED', 'Herkes'))
        self.assertIsNotNone(withheld_answer(chain, units, cafe, 'POSSIBLY_APPLIES', basis, self.listed))
        for state in ('DOES_NOT_APPLY', 'UNKNOWN'):
            self.assertIsNone(withheld_answer(chain, units, cafe, state, basis, self.listed), state)
        # No sub-paragraph addressed to everyone: the provision-level answer is about this clause and stands.
        alone = '(1) Yükümlüler, kayıtlarını denetime hazır bulundurmak zorundadır. (2) Kayıtlar sekiz yıl saklanır.'
        self.assertIsNone(withheld_answer(self.chain(alone, cafe, 'Yükümlüler', split_units(alone)[0][1]), split_units(alone), cafe,
                                          'APPLIES', basis, self.listed))
        # A verified YES on the list itself: the answer says the company is an obliged party.
        self.assertIsNone(withheld_answer(chain, units, cafe, 'APPLIES', [SimpleNamespace(match='YES', source_id=self.listed.source_id)],
                                          self.listed))
        # No list read (NOT_RESTRICTED): nothing to withhold.
        unlisted = self.chain(provision, cafe, 'Yükümlüler', second)
        self.assertIsNone(withheld_answer(unlisted, units, cafe, 'APPLIES', basis, None))
        # A bank is on the list (MATCH); an association is ruled out by rule before any model (DOES_NOT_APPLY).
        self.assertIsNone(withheld_answer(self.chain(provision, profile(**BANK), 'Yükümlüler', second), units, profile(**BANK), 'APPLIES',
                                          basis, self.listed))
        ruled_out = self.chain(provision, profile(**ASSOCIATION), 'Yükümlüler', second)
        self.assertEqual((ruled_out.decision, ruled_out.decided_by), ('DOES_NOT_APPLY', 'SUBJECT_SCOPE_GATE'))
        # The sub-paragraph addressed to everyone, and one addressed to "those asked", are not obliged-only clauses.
        first = self.chain(provision, cafe, 'Herkes', units[0][1])
        self.assertFalse(first.obliged_addressee)
        self.assertIsNone(withheld_answer(first, units, cafe, 'APPLIES', basis, self.listed))
        asked = self.chain(provision, cafe, 'Kendisinden talepte bulunulanlar',
                           '(2) Kendisinden talepte bulunulanlar, bilgi ve belge vermekten kaçınamazlar.')
        self.assertFalse(asked.obliged_addressee)

    def test_rows_the_rule_does_not_read_are_recorded_byte_for_byte_as_before(self):
        from regchain.pilot.scope_integrity import inheritance_evidence

        def before_inheritance_guard(*args, **kwargs):
            record = inheritance_evidence(*args, **kwargs)
            # Disable the newer independent guard only on the same obliged-only
            # scope addressed by withheld_answer; other new safeguards remain.
            if args[2].obliged_addressee:
                record['allowed'] = True
            return record

        judge = lambda: ProvisionJudge('APPLIES')
        for name, company in (('BANK', profile(**BANK)), ('I08', self.restaurant), ('SOFTWARE', profile(**SOFTWARE))):
            with self.subTest(company=name):
                _, now = self.rows(company, judge())
                # Reconstruct the pre-fix counterfactual: the later generic scope
                # guard independently blocks the same unsafe inheritance now.
                with mock.patch('regchain.pilot.engine.withheld_answer', return_value=None), \
                        mock.patch('regchain.pilot.engine.inheritance_evidence', side_effect=before_inheritance_guard):
                    _, before = self.rows(company, judge())
                for clause, row in now.items():
                    if name == 'I08' and clause == '(3)':
                        # Only the CHILD_ADDRESSEE gate is added; the chain and the model's answer read as before.
                        self.assertEqual(row['proposal']['trace']['gates'][:7], before[clause]['proposal']['trace']['gates'][:7])
                        self.assertEqual(before[clause]['proposal']['applicability'], 'APPLIES')         # the live t6 failure
                        continue
                    self.assertEqual(json.dumps(row['proposal'], sort_keys=True), json.dumps(before[clause]['proposal'], sort_keys=True), clause)

    def test_the_generic_addressee_reading_is_marked_on_the_chain_only(self):
        """The addressee gate's record keeps its words; the mark lives on the Chain, so no packet changes where the rule does not fire."""
        clause = paragraph(self.md31['text'], 3)
        candidate = {'subject': 'Yükümlüler', 'source_quote': clause}
        gate = entity_gate(candidate, clause, 'Bilgi', self.restaurant, LABEL)
        chain = rule_chain(self.restaurant, self.md31, candidate, clause, gate, self.listed, True)
        self.assertEqual(chain.by_name('COMPANY_ENTITY'), {'gate': 'COMPANY_ENTITY', 'status': 'MATCH', 'clear': True,
                                                           'reason': 'The duty is addressed to the obliged parties in general.',
                                                           'evidence': {'subject': 'Yükümlüler'}})
        self.assertEqual(rules.OBLIGED_GENERAL, chain.by_name('COMPANY_ENTITY')['reason'])
        self.assertTrue(chain.obliged_addressee)


if __name__ == '__main__':
    unittest.main()

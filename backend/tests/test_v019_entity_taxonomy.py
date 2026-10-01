"""v0.19 problem 3: the entity taxonomy of Turkish AML obliged parties.

Measured on independent-v1 I07 (a jeweller; run 20260925-015904-v019t3-heldout8-ollama): Tedbirler md. 25(1)
"Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler, riskli ülkelerde yerleşik gerçek ve tüzel
kişiler, tüzel kişiliği olmayan teşekküller ve bu ülkelerin vatandaşları ile girecekleri iş ilişkilerine ...
özel dikkat göstermek ... zorundadır" came back DOES_NOT_APPLY by the ENTITY_GATE. The subject was read as
"finansal kuruluş" alone (UNDETERMINED: the six-kind company table did not know a jeweller), the second
addressee was not read, and the counterparties were read as "tüzel kişiliği olmayan" alone ("gerçek ve tüzel
kişiler" was not recognised and the legal entity it names was suppressed by the "tüzel kişi" inside "tüzel
kişiliği olmayan"): a MISMATCH for a jeweller serving individuals and businesses.

The general class: a clause names its obliged parties by a specific kind ("Bankalar") or by a category the
regulation defines by the letters of its obliged-party list (md. 3(1)(f) "Finansal kuruluş" = md. 4(1)(a)–(h),
(m), (ü); md. 3(1)(m) "Finansal olmayan belirli iş ve meslekler" = md. 4(1)(k), (n), (s), (ş), (t), (u)), often
several of them at once, and the company's kind must be read from its profile with the same taxonomy.
"""
import json
import unittest
from pathlib import Path

from regchain.evidence import canonical_bytes
from regchain.pilot.applicability import (DNFBP, FINANCIAL_INSTITUTIONS, addressee_gate, category_definitions, decision_evidence,
                                          obliged_list, rule_chain)
from regchain.pilot.entities import entity_gate
from regchain.pilot.schema import Company
from regchain.pilot.sources import application_rows, load_sources
from test_v018_applicability import BANK, firm, profile

REPO = Path(__file__).resolve().parents[2]
TEDBIRLER = REPO / 'evaluation' / 'fixtures' / 'regulations' / 'tedbirler-200713012'
INDEPENDENT = REPO / 'evaluation' / 'independent' / 'independent-v1.json'

JEWELLER = firm('Sentetik Kuyumculuk Ltd. Şti.', ['altın ve mücevher perakende alım satımı'], customers=('gerçek kişiler', 'esnaf'))
EXCHANGE = firm('Sentetik Döviz A.Ş.', ['döviz alım satımı'], ['TCMB yetkili müessese izni'], customers=('gerçek kişiler', 'esnaf'))
REAL_ESTATE = firm('Sentetik Gayrimenkul A.Ş.', ['gayrimenkul alım satımı'], customers=('gerçek kişiler', 'esnaf'))
CAR_DEALER = firm('Sentetik Otomotiv A.Ş.', ['ikinci el otomobil ticareti'], customers=('gerçek kişiler', 'esnaf'))
FACTORING = firm('Sentetik Faktoring A.Ş.', ['faktoring'], ['BDDK faktoring şirketi faaliyet izni'])
EMONEY = firm('Sentetik E-Para A.Ş.', ['elektronik para ihracı'], ['elektronik para kuruluşu faaliyet izni'], customers=('gerçek kişiler',))
HOLDING = firm('Sentetik Holding A.Ş.', ['iştirak yönetimi'], customers=('grup şirketleri',))


def independent_profile(prefix):
    cases = json.loads(INDEPENDENT.read_text(encoding='utf-8'))['cases']
    return Company(**next(c for c in cases if c['case_id'].startswith(prefix))['company_profile'])


def paragraph(text, number):
    start = text.index(f'({number}) ')
    following = text.find(f'({number + 1}) ', start)
    return text[start:following if following > 0 else len(text)].strip()


def floats(value):
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(floats(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(floats(v) for v in value)
    return False


class EntityTaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sections = load_sources(TEDBIRLER)[1]
        cls.by_number = {s['printed_label'].split(' md. ')[1]: s for s in cls.sections}
        cls.scope_rows = application_rows(cls.sections, 'YONETMELIK', '200713012')
        cls.static = obliged_list(cls.scope_rows)                              # md. 1 and md. 4: the definitions are not in reach
        # The same list with the definitions article (md. 3) among the scope rows: the regulation's own words decide.
        cls.defined = obliged_list(sorted([*cls.scope_rows, cls.by_number['3']], key=lambda s: s['ordinal']))
        cls.md25 = paragraph(cls.by_number['25']['text'], 1)
        cls.md21 = {n: paragraph(cls.by_number['21']['text'], n) for n in (1, 3)}

    def run_gates(self, company, text, subject, number='99', listed=None):
        section = self.by_number.get(number) or {'id': 'x', 'printed_label': f'Yönetmelik 200713012 md. {number}',
                                                  'heading_path': ['YÖNETMELİK', 'ÜÇÜNCÜ BÖLÜM', 'Test'], 'text': text}
        candidate = {'subject': subject, 'source_quote': text}
        gate = entity_gate(candidate, text, 'Test', company, section['printed_label'])
        chain = rule_chain(company, section, candidate, text, gate, listed or self.static, True)
        return gate, chain

    def status(self, chain, name):
        return chain.by_name(name)['status']

    def test_i07_md_25_1_binds_the_jeweller(self):
        jeweller = independent_profile('I07')
        gate, chain = self.run_gates(jeweller, self.md25, 'Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler', '25')
        self.assertEqual(gate['match'], 'MATCH')
        self.assertEqual(gate['company_entity_types'], ['PRECIOUS_METALS'])
        rows = {(r['role'], r['type']): r['match'] for r in gate['required_entities']}
        self.assertEqual(rows[('obliged_party', 'DNFBP_GENERIC')], 'MATCH')
        self.assertEqual(rows[('obliged_party', 'FINANCIAL_GENERIC')], 'MISMATCH')         # one of two addressees is enough
        self.assertEqual(rows[('counterparty', 'INDIVIDUAL')], 'MATCH')                     # "gerçek ve tüzel kişiler" is read
        self.assertEqual(rows[('counterparty', 'LEGAL_ENTITY')], 'MATCH')                   # ... and not hidden by "tüzel kişiliği olmayan"
        self.assertEqual(rows[('counterparty', 'UNINCORPORATED')], 'MISMATCH')
        self.assertEqual((chain.decision, chain.clear_match), ('OPEN', True))
        self.assertEqual({g['gate']: g['status'] for g in chain.gates},
                         {'REGULATION_SUBJECT_SCOPE': 'MATCH', 'COMPANY_ENTITY': 'MATCH', 'JURISDICTION': 'MATCH', 'CUSTOMER_ENTITY': 'MATCH',
                          'CHILD_CLAUSE': 'MATCH', 'EXEMPTION': 'NONE'})

    def test_financial_institutions_and_dnfbps_clause_matches_jeweller_exchange_office_and_real_estate(self):
        subject = 'Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler'
        for company, kind in ((JEWELLER, 'PRECIOUS_METALS'), (EXCHANGE, 'EXCHANGE_OFFICE'), (REAL_ESTATE, 'REAL_ESTATE')):
            for listed in (self.static, self.defined):
                with self.subTest(company=company.name, definitions=bool(listed.categories)):
                    gate, chain = self.run_gates(company, self.md25, subject, '25', listed)
                    self.assertIn(kind, gate['company_entity_types'])
                    self.assertEqual(gate['match'], 'MATCH')
                    self.assertEqual(self.status(chain, 'COMPANY_ENTITY'), 'MATCH')
                    self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')

    def test_a_car_dealer_is_neither_category_by_the_regulation_s_definitions_and_open_without_them(self):
        """md. 4(1)(o) is obliged, but md. 3(1)(m) names (k), (n), (s), (ş), (t), (u) only: not a DNFBP by the text."""
        subject = 'Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler'
        gate, chain = self.run_gates(CAR_DEALER, self.md25, subject, '25', self.static)
        rows = {r['type']: r['match'] for r in gate['required_entities'] if r['role'] == 'obliged_party'}
        self.assertEqual(rows, {'FINANCIAL_GENERIC': 'MISMATCH', 'DNFBP_GENERIC': 'UNDETERMINED'})
        self.assertEqual(gate['match'], 'UNDETERMINED')
        self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')                               # the static taxonomy cannot tell
        gate, chain = self.run_gates(CAR_DEALER, self.md25, subject, '25', self.defined)
        self.assertEqual((chain.decision, chain.decided_by), ('DOES_NOT_APPLY', 'ADDRESSEE_GATE'))
        record = chain.by_name('COMPANY_ENTITY')
        definitions = self.by_number['3']
        self.assertEqual(record['evidence']['membership'], 'DEFINITION')
        cited = {d['category']: d for d in record['evidence']['definitions']}
        self.assertEqual(set(cited), {'FINANCIAL_GENERIC', 'DNFBP_GENERIC'})
        self.assertEqual(cited['DNFBP_GENERIC']['items'], ['k)', 'n)', 's)', 'ş)', 't)', 'u)'])
        _, quotes = decision_evidence(chain, CAR_DEALER, self.by_number['25'], self.md25, self.defined)
        for d in cited.values():
            self.assertEqual(d['source_id'], definitions['id'])
            self.assertIn(d['quote'], definitions['text'])                                  # an exact quote of md. 3
            self.assertIn((definitions['id'], d['quote']), quotes)
        for source_id, quote in quotes:
            self.assertIn(quote, next(s['text'] for s in self.sections if s['id'] == source_id))

    def test_a_dnfbp_only_clause(self):
        text = '(1) Finansal olmayan belirli iş ve meslekler, müşterilerinin kimliğini tespit etmek zorundadır.'
        subject = 'Finansal olmayan belirli iş ve meslekler'
        for company in (JEWELLER, REAL_ESTATE):
            with self.subTest(company=company.name):
                gate, chain = self.run_gates(company, text, subject)
                self.assertEqual(gate['match'], 'MATCH')
                self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')
        for company in (profile(**BANK), EXCHANGE):                                         # financial institutions are not DNFBPs
            with self.subTest(company=company.name):
                gate, chain = self.run_gates(company, text, subject)
                self.assertEqual(gate['match'], 'MISMATCH')
                self.assertEqual((chain.decision, chain.decided_by), ('DOES_NOT_APPLY', 'ENTITY_GATE'))

    def test_bank_only_and_payment_only_clauses(self):
        gate, chain = self.run_gates(profile(), '(1) Bankalar, muhabir ilişkilerinde ek tedbir almak zorundadır.', 'Bankalar')
        self.assertEqual(([(r['type'], r['match']) for r in gate['required_entities'] if r['role'] == 'obliged_party'], chain.decision,
                          chain.decided_by),
                         ([('BANK', 'MISMATCH')], 'DOES_NOT_APPLY', 'ENTITY_GATE'))
        gate, chain = self.run_gates(profile(**BANK), '(1) Ödeme kuruluşları, üye işyeri kayıtlarını tutmak zorundadır.', 'Ödeme kuruluşları')
        self.assertEqual(([(r['type'], r['match']) for r in gate['required_entities']], chain.decision, chain.decided_by),
                         ([('PAYMENT_INSTITUTION', 'MISMATCH')], 'DOES_NOT_APPLY', 'ENTITY_GATE'))
        # Two kinds named together are alternatives: an e-money institution is addressed (v0.16.1 read the first one only).
        text = '(1) Ödeme kuruluşları ile elektronik para kuruluşları, fon kaynağını kaydetmek zorundadır.'
        gate, chain = self.run_gates(EMONEY, text, 'Ödeme kuruluşları ile elektronik para kuruluşları')
        self.assertEqual({r['type']: r['match'] for r in gate['required_entities']}, {'PAYMENT_INSTITUTION': 'MISMATCH', 'EMONEY_INSTITUTION': 'MATCH'})
        self.assertEqual(gate['match'], 'MATCH')
        self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')

    def test_financial_institution_clause_against_factoring_and_a_jeweller(self):
        factoring, jeweller = independent_profile('I13'), independent_profile('I07')
        for number, subject in ((1, 'Finansal kuruluşlar'), (3, 'Üçüncü tarafa güvenerek iş ilişkisi tesis eden veya işlem yapan finansal kuruluş')):
            with self.subTest(clause=number):
                gate, chain = self.run_gates(factoring, self.md21[number], subject, '21')
                self.assertEqual((gate['match'], self.status(chain, 'COMPANY_ENTITY')), ('MATCH', 'MATCH'))
                self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')
                gate, chain = self.run_gates(jeweller, self.md21[number], subject, '21')
                self.assertEqual([(r['type'], r['match']) for r in gate['required_entities'] if r['role'] == 'obliged_party'],
                                 [('FINANCIAL_GENERIC', 'MISMATCH')])
                self.assertEqual((chain.decision, chain.decided_by), ('DOES_NOT_APPLY', 'ENTITY_GATE'))
                self.assertIn('finansal kuruluş', gate['reason'])
        gate, _ = self.run_gates(FACTORING, self.md21[1], 'Finansal kuruluşlar', '21')
        self.assertEqual(gate['match'], 'MATCH')

    def test_an_unknown_or_open_kind_is_never_ruled_out(self):
        texts = ((self.md25, 'Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler', '25'), (self.md21[1], 'Finansal kuruluşlar', '21'),
                 ('(1) Finansal olmayan belirli iş ve meslekler, kayıt tutmak zorundadır.', 'Finansal olmayan belirli iş ve meslekler', '99'),
                 ('(1) Bankalar, muhabir ilişkilerinde ek tedbir almak zorundadır.', 'Bankalar', '99'))
        asset_manager = firm('Sentetik Varlık Yönetim A.Ş.', ['varlık yönetim şirketi; takipteki alacakların satın alınması'])
        for company in (HOLDING, asset_manager):
            for text, subject, number in texts[:3] if company is asset_manager else texts:
                with self.subTest(company=company.name, subject=subject):
                    gate, chain = self.run_gates(company, text, subject, number)
                    self.assertNotEqual(gate['match'], 'MISMATCH')
                    self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')
        self.assertEqual(entity_gate({'subject': 'Bankalar'}, texts[3][0], '', HOLDING, 'x')['required_entities'][0]['match'], 'UNDETERMINED')
        # A member kind named only in passing ("ürünlerini ... banka kanalıyla satar") is not a clear match, nor a mismatch.
        insurer = independent_profile('I01')
        gate = entity_gate({'subject': 'Bankalar'}, texts[3][0], '', insurer, 'x')
        self.assertEqual(gate['required_entities'][0]['match'], 'UNDETERMINED')

    def test_the_regulation_s_definitions_are_read_by_their_letters(self):
        self.assertEqual(self.static.categories, {})
        financial, dnfbp = self.defined.categories['FINANCIAL_GENERIC'], self.defined.categories['DNFBP_GENERIC']
        self.assertEqual(financial.markers, ('a)', 'b)', 'c)', 'ç)', 'd)', 'e)', 'f)', 'g)', 'ğ)', 'h)', 'm)', 'ü)'))
        self.assertEqual(dnfbp.markers, ('k)', 'n)', 's)', 'ş)', 't)', 'u)'))
        self.assertEqual(financial.members, FINANCIAL_INSTITUTIONS)                         # the static reading is the same definition
        self.assertEqual(dnfbp.members, DNFBP)
        self.assertIn('PRECIOUS_METALS', financial.non_members)                             # (k) is not (m)
        self.assertNotIn('POST_CARGO', financial.non_members | financial.members)           # PTT: "bankacılık faaliyetleriyle sınırlı"
        self.assertTrue({'VEHICLE_DEALER', 'SPORTS_CLUB', 'BANK'} <= dnfbp.non_members)
        for member in (financial, dnfbp):
            self.assertIn(member.quote, self.by_number['3']['text'])
        items = {i['marker']: i['kinds'] for i in self.static.items}
        self.assertEqual((items['k)'], items['m)']), (['PRECIOUS_METALS'], ['PRECIOUS_METALS_INTERMEDIARY']))
        # A definition that points at another article, or at letters the list does not have, is not read.
        for body in ('Bu Yönetmeliğin 5 inci maddesinin birinci fıkrasının (a) ila (h) bentlerinde sayılan yükümlüleri',
                     'Bu Yönetmeliğin 4 üncü maddesinin birinci fıkrasının (a) ila (x) bentlerinde sayılan yükümlüleri',
                     'finansal hizmet veren kuruluşları'):
            row = {'id': 'd', 'printed_label': 'Yönetmelik 200713012 md. 3', 'heading_path': ['Tanımlar'],
                   'text': f'(1) Bu Yönetmelikte geçen; a) Finansal kuruluş: {body}, b) Kanun: 5549 sayılı Kanunu, ifade eder.'}
            with self.subTest(body=body):
                self.assertEqual(category_definitions([row], self.static), {})

    def test_a_precious_metals_intermediary_is_a_financial_institution_and_a_jeweller_is_not_one(self):
        intermediary = firm('Sentetik Kıymetli Madenler Aracı Kuruluşu A.Ş.', ['kıymetli madenler aracı kuruluşu faaliyetleri'])
        gate, _ = self.run_gates(intermediary, self.md21[1], 'Finansal kuruluşlar', '21')
        self.assertEqual(gate['match'], 'MATCH')
        section = {'heading_path': ['YÖNETMELİK', 'ÜÇÜNCÜ BÖLÜM', 'Test'], 'text': ''}
        subject = 'Kıymetli madenler aracı kuruluşları'
        clause_gate = entity_gate({'subject': subject}, f'(1) {subject}, kayıt tutar.', 'Test', JEWELLER, 'x')
        self.assertEqual(addressee_gate(section, {'subject': subject}, f'(1) {subject}, kayıt tutar.', JEWELLER, clause_gate)['status'], 'MISMATCH')

    def test_a_qualifier_must_match_as_well(self):
        text = '(1) Merkezi yurt dışında bulunan bankalar, Türkiye’deki şubelerinin kayıtlarını ayrıca saklamak zorundadır.'
        subject = 'Merkezi yurt dışında bulunan bankalar'
        gate, _ = self.run_gates(profile(**BANK), text, subject)
        self.assertEqual((gate['match'], [(r['type'], r['match']) for r in gate['required_entities']]),
                         ('MISMATCH', [('FOREIGN_HQ_OBLIGED', 'MISMATCH'), ('BANK', 'MATCH')]))
        branch = profile(licences=['Merkezi yurt dışında bulunan ödeme kuruluşunun Türkiye şubesi izni'])
        gate, _ = self.run_gates(branch, text, subject)
        self.assertEqual((gate['match'], [(r['type'], r['match']) for r in gate['required_entities']]),
                         ('MISMATCH', [('FOREIGN_HQ_OBLIGED', 'MATCH'), ('BANK', 'MISMATCH')]))

    def test_counterparty_lists(self):
        individuals = firm('Sentetik Kuyumculuk Ltd. Şti.', ['kuyumculuk'], customers=('yalnızca gerçek kişiler',))
        text = '(1) Yükümlüler, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan teşekküllerle girecekleri iş ilişkilerine dikkat etmek zorundadır.'
        gate = entity_gate({'subject': 'Yükümlüler'}, text, '', individuals, 'x')
        self.assertEqual({r['type']: r['match'] for r in gate['required_entities']},
                         {'UNINCORPORATED': 'MISMATCH', 'INDIVIDUAL': 'MATCH', 'LEGAL_ENTITY': 'MISMATCH'})
        self.assertEqual(gate['match'], 'MATCH')
        gate = entity_gate({'subject': 'Yükümlüler'}, '(1) Yükümlüler, tüzel kişiliği olmayan teşekküllerin kimlik tespitinde belge alır.', '',
                           individuals, 'x')
        self.assertEqual([(r['type'], r['match']) for r in gate['required_entities']], [('UNINCORPORATED', 'MISMATCH')])
        self.assertEqual(gate['match'], 'MISMATCH')

    def test_every_new_field_is_digest_safe(self):
        gate, chain = self.run_gates(CAR_DEALER, self.md25, 'Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler', '25', self.defined)
        for value in (gate, chain.gates):
            self.assertFalse(floats(value))
            canonical_bytes(value)


if __name__ == '__main__':
    unittest.main()

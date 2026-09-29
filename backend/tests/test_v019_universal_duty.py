"""v0.19 FIX 3: a duty addressed to everyone is not ruled out by the entity, subject-scope or addressee gates.

Measured on independent-v1 I03 (a software house, not an obliged party): Kanun 5549 md. 7(1) "Kamu kurum
ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar, ... vermek ... yükümlüdür"
came back DOES_NOT_APPLY by the ENTITY_GATE, which read the duty's addressees ("tüzel kişiliği olmayan",
"kamu kurum") as counterparties the profile lacks. Tedbirler md. 31(1) has the same list (I08, a
restaurant), and its extracted subject "Başkanlık ve denetim elemanları tarafından istenilecek ..." made
the addressee gate read an authority. The general class: a clause whose opening addressee list binds
everyone ("herkes", "hiç kimse", "any person") or every legal person ("gerçek ve tüzel kişiler").
"""
import json
import unittest
from pathlib import Path
from unittest import mock

from regchain.evaluation.harness import run
from regchain.extraction.rules import RulesProvider
from regchain.pilot.applicability import addressee_gate, obliged_list, rule_chain, subject_gate
from regchain.pilot.engine import analyze
from regchain.pilot.entities import entity_gate, universal_addressee, universal_covering
from regchain.pilot.schema import Company
from regchain.pilot.sources import application_rows, load_sources, select_targets
from test_pilot import policies
from test_v018_applicability import BANK, SOFTWARE, Scripted, profile

REPO = Path(__file__).resolve().parents[2]
TEDBIRLER = REPO / 'evaluation' / 'fixtures' / 'regulations' / 'tedbirler-200713012'
KANUN = REPO / 'evaluation' / 'independent' / 'fixtures' / 'regulations' / 'kanun-5549'
INDEPENDENT = REPO / 'evaluation' / 'independent' / 'independent-v1.json'
GOLDEN = REPO / 'evaluation' / 'datasets' / 'tr-aml-v1.json'


def independent_profile(prefix):
    cases = json.loads(INDEPENDENT.read_text(encoding='utf-8'))['cases']
    return Company(**next(c for c in cases if c['case_id'].startswith(prefix))['company_profile'])


def section(sections, label):
    return next(s for s in sections if s['printed_label'] == label)


def paragraph(text, number):
    """Sub-paragraph (n) of an article's text, as the clause text the gates read."""
    start = text.index(f'({number}) ')
    following = text.find(f'({number + 1}) ', start)
    return text[start:following if following > 0 else len(text)].strip()


def statuses(proposal):
    return {g['gate']: g for g in proposal['trace']['gates'] if g}


class UniversalAddresseeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kanun = load_sources(KANUN)[1]
        cls.tedbirler = load_sources(TEDBIRLER)[1]
        cls.md7 = section(cls.kanun, 'Kanun 5549 md. 7')['text']
        cls.md31 = section(cls.tedbirler, 'Yönetmelik 200713012 md. 31')['text']
        cls.tedbirler_list = obliged_list(application_rows(cls.tedbirler, 'YONETMELIK', '200713012'))

    def analyze(self, company, sections, module, chapter, article, judge=None):
        labels = select_targets(sections, module, chapter, 'all', [article])
        provider = judge or RulesProvider()
        payload = analyze(company, policies(), sections, provider, labels, judge=provider, target_filter=[article])['events'][0]['payload']
        return {o['proposal']['applicability_scope']['child_clause']: o for o in payload['obligations']}

    def test_the_opening_addressee_list_is_read_only_when_it_binds_everyone_or_every_legal_person(self):
        universal = {
            paragraph(self.md7, 1): 'Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar',
            paragraph(self.md31, 1): 'Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar',
            '(1) Hiç kimse, şüpheli işlem bildiriminde bulunulduğunu üçüncü kişilere açıklayamaz.': 'Hiç kimse',
            '(2) Herkes, Başkanlıkça istenen bilgileri vermekle yükümlüdür.': 'Herkes',
            'Any person must provide the information the regulator requests.': 'Any person',
            'No person may disclose that a report has been made.': 'No person',
            'All persons shall keep the records for five years.': 'All persons'}
        for text, listed in universal.items():
            with self.subTest(text=text[:60]):
                self.assertEqual(universal_addressee(text)['text'], listed)
        self.assertIn('LEGAL_PERSON', universal_addressee(paragraph(self.md7, 1))['scope'])
        not_universal = (
            paragraph(self.md31, 3),                                                              # "Yükümlüler ... defter"
            '(1) Yükümlüler, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan teşekküller için kayıt tutmak zorundadır.',
            '(1) Bankalar, muhabir ilişkilerinde ek tedbir almak zorundadır.',
            '(1) Finansal kuruluşlar ile finansal olmayan belirli iş ve meslekler, riskli ülkelerde yerleşik gerçek ve tüzel kişiler '
            'ile girecekleri iş ilişkilerine özel dikkat göstermek zorundadır.',
            '(1) Bu Kanun gereğince yükümlülüklerini yerine getiren gerçek ve tüzel kişiler hiçbir şekilde sorumlu tutulamaz.',
            '(1) Gerçek ve tüzel kişiler adına işlem yapanlar kimliklerini bildirmek zorundadır.',   # a postposition: not the subject
            '(1) Tüzel kişiler, yükümlüler tarafından kaydedilir.',                                  # no duty verb: a passive
            '(1) Yükümlüler, işleme taraf olanlar dahil hiç kimseye açıklayamazlar.',               # "hiç kimseye": the object
            '(1) Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde unvan alınır.',
            '(1) Gerçek kişiler bilgi vermek zorundadır.',                                           # binds no company
            '(1) Kamu kurum ve kuruluşları bilgi vermek zorundadır.',
            'Any person who is a firm must keep records.',
            'A firm must ensure that all persons visiting a customer act in accordance with CONC 7.')
        for text in not_universal:
            with self.subTest(text=text[:60]):
                self.assertIsNone(universal_addressee(text))
        # The whole article as one clause: the list governs its own sentence, not md. 31(3).
        self.assertIsNone(universal_addressee(self.md31, paragraph(self.md31, 3)))
        self.assertIsNotNone(universal_addressee(self.md31, paragraph(self.md31, 1)))

    def test_kanun_5549_md_7_1_binds_a_company_that_is_not_an_obliged_party(self):
        software = independent_profile('I03')
        gate = entity_gate({'subject': 'Kamu kurum ve kuruluşları'}, paragraph(self.md7, 1), 'Bilgi ve belge verme', software, 'Kanun 5549 md. 7')
        self.assertEqual((gate['match'], gate['required_entities']), ('NOT_RESTRICTED', []))
        self.assertIn('UNIVERSAL_ADDRESSEE', gate['reason'])
        rows = self.analyze(software, self.kanun, 'KANUN', '5549', '7')
        proposal = rows['(1)']['proposal']
        self.assertNotEqual(proposal['applicability'], 'DOES_NOT_APPLY')
        self.assertEqual(proposal['applicability_scope']['match'], 'NOT_RESTRICTED')
        gates = statuses(proposal)
        self.assertEqual({name: gates[name]['status'] for name in ('REGULATION_SUBJECT_SCOPE', 'COMPANY_ENTITY', 'CUSTOMER_ENTITY', 'CHILD_CLAUSE')},
                         {'REGULATION_SUBJECT_SCOPE': 'NOT_RESTRICTED', 'COMPANY_ENTITY': 'MATCH', 'CUSTOMER_ENTITY': 'NOT_RESTRICTED',
                          'CHILD_CLAUSE': 'NOT_RESTRICTED'})
        for name in ('REGULATION_SUBJECT_SCOPE', 'COMPANY_ENTITY', 'CHILD_CLAUSE'):
            self.assertEqual(gates[name]['evidence']['reason_code'], 'UNIVERSAL_ADDRESSEE', name)
            self.assertIn(gates[name]['evidence']['universal_addressee'], self.md7)            # an exact quote of the clause
        # The judge is asked and its APPLIES stands.
        judge = Scripted('APPLIES')
        proposal = self.analyze(software, self.kanun, 'KANUN', '5549', '7', judge)['(1)']['proposal']
        self.assertEqual(proposal['applicability'], 'APPLIES')
        self.assertIn('Kanun 5549 md. 7', judge.scoped)

    def test_tedbirler_md_31_1_is_universal_and_md_31_3_stays_with_the_obliged_parties(self):
        software = profile(**SOFTWARE)
        self.assertEqual(subject_gate(software, self.tedbirler_list)['status'], 'MISMATCH')      # not an obliged party
        rows = self.analyze(software, self.tedbirler, 'YONETMELIK', '200713012', '31')
        first, third = rows['(1)']['proposal'], rows['(3)']['proposal']
        self.assertNotEqual(first['applicability'], 'DOES_NOT_APPLY')
        self.assertEqual({k: statuses(first)[k]['status'] for k in ('REGULATION_SUBJECT_SCOPE', 'COMPANY_ENTITY', 'CHILD_CLAUSE')},
                         {'REGULATION_SUBJECT_SCOPE': 'NOT_RESTRICTED', 'COMPANY_ENTITY': 'MATCH', 'CHILD_CLAUSE': 'NOT_RESTRICTED'})
        self.assertEqual((third['applicability'], third['applicability_rule']), ('DOES_NOT_APPLY', 'SUBJECT_SCOPE_GATE'))
        self.assertNotIn('UNIVERSAL_ADDRESSEE', json.dumps(third['trace'], ensure_ascii=False))
        # A bank is bound too: the participle's agent "Başkanlık ve denetim elemanları tarafından" is not the addressee.
        bank = self.analyze(profile(**BANK), self.tedbirler, 'YONETMELIK', '200713012', '31')['(1)']['proposal']
        self.assertNotEqual(bank['applicability'], 'DOES_NOT_APPLY')
        self.assertEqual(statuses(bank)['REGULATION_SUBJECT_SCOPE']['status'], 'MATCH')          # a list match is kept as it is

    def chain(self, text, company, subject='', label='Yönetmelik 200713012 md. 99'):
        section = {'id': 'x', 'printed_label': label, 'heading_path': ['YÖNETMELİK', 'BEŞİNCİ BÖLÜM', 'Bilgi'], 'text': text}
        candidate = {'subject': subject, 'source_quote': text}
        gate = entity_gate(candidate, text, 'Bilgi', company, label)
        return gate, rule_chain(company, section, candidate, text, gate, self.tedbirler_list, True)

    def test_a_hic_kimse_prohibition_binds_a_non_obliged_company_and_a_yukumluler_one_does_not(self):
        software = profile(**SOFTWARE)
        gate, chain = self.chain('(1) Hiç kimse, şüpheli işlem bildiriminde bulunulduğunu üçüncü kişilere açıklayamaz.', software, 'Hiç kimse')
        self.assertEqual(gate['match'], 'NOT_RESTRICTED')
        self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')
        self.assertEqual({chain.by_name(n)['evidence']['reason_code'] for n in ('REGULATION_SUBJECT_SCOPE', 'COMPANY_ENTITY', 'CHILD_CLAUSE')},
                         {'UNIVERSAL_ADDRESSEE'})
        _, chain = self.chain('(1) Yükümlüler, işleme taraf olanlar dahil hiç kimseye açıklayamazlar.', software, 'Yükümlüler')
        self.assertEqual((chain.decision, chain.decided_by), ('DOES_NOT_APPLY', 'SUBJECT_SCOPE_GATE'))
        # Only the addressee list is set aside: a counterparty the rest of the clause names is still read.
        gate, chain = self.chain('(1) Hiç kimse, tüzel kişiliği olmayan teşekküllere ilişkin bildirimleri üçüncü kişilere açıklayamaz.',
                                 software, 'Hiç kimse')
        self.assertEqual([(r['type'], r['match']) for r in gate['required_entities']], [('UNINCORPORATED', 'MISMATCH')])
        self.assertEqual((chain.decision, chain.decided_by), ('DOES_NOT_APPLY', 'ENTITY_GATE'))

    def test_english_any_person(self):
        software = profile(**SOFTWARE)
        text = 'Any person must provide the information that the regulator requests.'
        section = {'id': 'x', 'printed_label': 'CONC 9.9.9', 'heading_path': ['CONC 9'], 'text': text}
        gate = entity_gate({'subject': 'Any person'}, text, '', software, 'CONC 9.9.9')
        self.assertEqual(gate['match'], 'NOT_RESTRICTED')
        addressee = addressee_gate(section, {'subject': 'Any person'}, text, software, gate)
        self.assertEqual((addressee['status'], addressee['evidence']['reason_code']), ('MATCH', 'UNIVERSAL_ADDRESSEE'))
        chain = rule_chain(software, section, {'subject': 'Any person'}, text, gate, None, False)
        self.assertNotEqual(chain.decision, 'DOES_NOT_APPLY')
        self.assertIsNone(universal_covering({'subject': 'A firm'}, 'A firm must keep records.', software))

    def test_a_clause_addressed_only_to_obliged_parties_reads_as_before(self):
        software, bank = profile(**SOFTWARE), profile(**BANK)
        texts = (paragraph(self.md31, 3), '(1) Yükümlüler, derneklerin kimlik tespitinde kütük numarasını alır.',
                 '(1) Bankalar, muhabir ilişkilerinde ek tedbir almak zorundadır.',
                 '(1) Finansal kuruluşlar, yurt dışında yerleşik müşterileri için ek tedbir almak zorundadır.',
                 '(1) Yükümlüler, işleme taraf olanlar dahil hiç kimseye açıklayamazlar.')
        for text in texts:
            for company in (software, bank):
                with self.subTest(text=text[:50], company=company.id):
                    now = self.chain(text, company, text.split(',')[0][4:])
                    with mock.patch('regchain.pilot.entities.universal_addressee', return_value=None):
                        before = self.chain(text, company, text.split(',')[0][4:])
                    self.assertEqual(now[0], before[0])
                    self.assertEqual((now[1].gates, now[1].decision, now[1].decided_by), (before[1].gates, before[1].decision, before[1].decided_by))

    def test_tr_aml_v1_rules_mode_applicability_is_unchanged(self):
        """The golden set holds no universal duty: every row reads as it did without the rule."""
        import contextlib
        import io
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            now_dir, _, now = run(GOLDEN, Path(tmp) / 'now', provider='rules', retrieval='lexical', label='now')
            with mock.patch('regchain.pilot.entities.universal_addressee', return_value=None):
                off_dir, _, off = run(GOLDEN, Path(tmp) / 'off', provider='rules', retrieval='lexical', label='off')
            rows = lambda d: {(c['case_id'], p['key']): (p['applicability'], p['applicability_rule'], p['entity_gate'], json.dumps(p['gate_status']))
                              for c in json.loads((d / 'results.json').read_text(encoding='utf-8')) for p in c['predictions']}
            self.assertEqual(rows(now_dir), rows(off_dir))
            self.assertEqual(now['applicability']['accuracy'], off['applicability']['accuracy'])
            self.assertNotIn('UNIVERSAL_ADDRESSEE', (now_dir / 'results.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()

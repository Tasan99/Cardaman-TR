"""v0.18 regressions: rule-first applicability (the regulation's obliged-party list, the duty's
addressee, exemption wording, the profile's silence) and the relevance screen.

Measured on the v0.17 golden run (tr-aml-v1, 30 cases, 92 scored duties): 22 applicability errors,
19 of them decidable by rule — a foreign foundation's office, an association, a foundation and a
software company got APPLIES or UNKNOWN for the general duties of Tedbirler Yönetmeliği although
md. 4 does not list them; a bank serving associations got UNKNOWN for md. 8; customs duties of
md. 42 came back POSSIBLY_APPLIES. The reasoning "does it contradict?" question on every passage
cost half of the model time.
"""
import json
import tempfile
import unittest
from pathlib import Path

from regchain.ingestion.mevzuat import parse_mevzuat, text_url
from regchain.pilot.applicability import exemption_check, obliged_list, subject_gate
from regchain.pilot.engine import SCREEN_ALWAYS_JUDGE, analyze, autonomous_gate, coverage_of, pipeline_settings
from regchain.pilot.schema import Company, PolicyCheck
from regchain.pilot.sources import application_rows, load_sources, save_sources, select_targets
from test_phase15 import TurkishRules
from test_phase16 import demopay, madde
from test_pilot import FixtureProvider, company, policies, sections
from test_turkiye import download, p

URL = text_url('21', '200713012')
FIXTURES = Path(__file__).with_name('fixtures')


def regulation_html():
    """A by-law in the shape of Tedbirler Yönetmeliği with the parts v0.18 reads: a closed obliged-party
    list (md. 4), a generic duty (md. 3), associations (md. 8), a bank-only duty (md. 10), a crypto-only
    duty (md. 11), a duty that exempts payment institutions in its own words (md. 12) and in a sibling
    sub-paragraph (md. 13), and a customs part (md. 42)."""
    return ('<html><head><meta http-equiv=Content-Type content="text/html; charset=Windows-1254"></head><body lang=TR><div class=WordSection1>'
            + p('TEDBİRLER HAKKINDA YÖNETMELİK (TEST v0.18)', center=True, bold=True)
            + p('Resmî Gazete Tarihi : 09/01/2008 Resmî Gazete Sayısı : 26751')
            + p('BİRİNCİ BÖLÜM', center=True, bold=True) + p('Amaç, Kapsam ve Tanımlar', center=True, bold=True)
            + p('Amaç ve kapsam', bold=True) + madde(1, '(1) Bu Yönetmeliğin amacı, suç gelirlerinin aklanmasının önlenmesine ilişkin yükümlülükleri düzenlemektir.')
            + p('İKİNCİ BÖLÜM', center=True, bold=True) + p('Yükümlüler', center=True, bold=True)
            + p('Kimlik tespiti', bold=True) + madde(3, '(1) Yükümlüler, müşterilerinin kimliğini tespit etmek zorundadır.')
            + p('Yükümlü', bold=True) + madde(4, '(1) Kanunun uygulanmasında yükümlü, aşağıda sayılanlar ile bunların şubeleridir: a) Bankalar. '
                                              'b) Ödeme kuruluşları ile elektronik para kuruluşları. c) Kripto varlık hizmet sağlayıcılar. '
                                              '(2) Merkezi yurt dışında bulunan yükümlünün Türkiye’deki şubeleri de yükümlü sayılır.')
            + p('ÜÇÜNCÜ BÖLÜM', center=True, bold=True) + p('Müşterinin Tanınması', center=True, bold=True)
            + p('Dernek ve vakıflarda kimlik tespiti', bold=True) + madde(8, '(1) Derneklerin kimlik tespitinde derneğin adı ve kütük numarası alınır.')
            + p('Bankalarda muhabir ilişkisi', bold=True) + madde(10, '(1) Bankalar, muhabir ilişkilerinde ek tedbir almak zorundadır.')
            + p('Kripto varlık transferleri', bold=True)
            + madde(11, '(1) Kripto varlık hizmet sağlayıcılar, transferlerde gönderen bilgisini almak zorundadır.')
            + p('Tutar sınırı', bold=True)
            + madde(12, '(1) Yükümlüler, tek seferlik işlemlerde tutar sınırını izlemek zorundadır; bu fıkra ödeme kuruluşları hakkında uygulanmaz.')
            + p('Kayıt güncelleme', bold=True)
            + madde(13, '(1) Yükümlüler, müşteri kayıtlarını yılda bir güncellemek zorundadır. (2) Birinci fıkra, ödeme kuruluşları hakkında uygulanmaz.')
            + p('YEDİNCİ BÖLÜM', center=True, bold=True) + p('Gümrük İdaresine Yapılacak Açıklama', center=True, bold=True)
            + p('Tutanak düzenlenmesi', bold=True) + madde(42, '(1) Kendisinden açıklama talep edilen yolcu ile ilgili tespitler bir tutanağa bağlanır.')
            + p('SEKİZİNCİ BÖLÜM', center=True, bold=True) + p('Muhafaza', center=True, bold=True)
            + p('Muhafaza ve ibraz yükümlülüğü', bold=True) + madde(46, '(1) Yükümlüler belgeleri sekiz yıl süre ile muhafaza etmek zorundadır.')
            + '</div></body></html>')


def profile(**changes):
    return Company(**{**demopay().model_dump(), **changes})


ASSOCIATION = dict(id='dernek', name='Yeşil Vadi Doğa Derneği', activities=['dernek faaliyetleri, bağış toplama'], licences=[],
                   products=['üyelik aidatı'], customer_types=['üyeler ve bağışçılar'],
                   description='Dernekler Kanunu kapsamında kurulmuş sentetik dernek; finansal hizmet sunmaz.')
SOFTWARE = dict(id='yazilim', name='Anadolu Yazılım Ltd. Şti.', activities=['yazılım geliştirme'], licences=[], products=['yazılım lisansı'],
                customer_types=['kurumsal müşteriler'], description='Finansal hizmet sunmayan sentetik yazılım şirketi.')
BANK = dict(id='banka', name='Örnek Bank A.Ş.', activities=['bankacılık, mevduat kabulü'], licences=['mevduat bankası faaliyet izni'],
            products=['mevduat'], customer_types=['tüm müşteri türleri: bireysel, ticari, dernek ve vakıflar'],
            description='Her tür müşteriye hizmet veren sentetik banka.')


class Scripted(TurkishRules):
    """The Turkish fixture judge with a scripted applicability answer and counters for every question."""

    def __init__(self, applicability='APPLIES', no_condition=None, screen=None):
        self.applicability, self.no_condition, self.screen = applicability, no_condition, screen
        self.scoped, self.asked, self.screened = [], [], []

    def _chat(self, prompt, payload, schema):
        fields = schema['properties']
        if 'items' in fields:
            self.screened.append(payload)
            if self.screen is None:
                raise ValueError('no screen scripted')
            return json.dumps({'items': [{'id': row['id'], 'topic': self.screen(row['text'])} for row in payload['passages']]})
        if 'contradicts' in fields:
            self.asked.append(payload['passage'])
        if 'applicability' in fields:
            self.scoped.append(payload['source_label'])
        return super()._chat(prompt, payload, schema)

    def scope(self, payload):
        value = FixtureProvider.scope(self, payload)
        if self.applicability == 'APPLIES' and self.no_condition is None:
            return value
        condition = self.no_condition or payload['scope'][0]['text'][:100]
        match = {'UNKNOWN': 'YES', 'POSSIBLY_APPLIES': 'YES'}.get(self.applicability, 'NO')
        basis = [{'company_fact': value['basis'][0]['company_fact'], 'regulatory_condition': condition, 'match': match}]
        return {**value, 'applicability': self.applicability, 'basis': basis, 'applicability_reason': f'Fixture answers {self.applicability}.'}


class RuleFirstApplicabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        save_sources(Path(self.temp.name) / 'reg', [download(regulation_html(), URL)], mevzuat=('21', '200713012', '5'))
        self.sections = load_sources(Path(self.temp.name) / 'reg')[1]

    def labels(self, *numbers):
        return select_targets(self.sections, 'YONETMELIK', '200713012', 'all', list(numbers))

    def rows(self, company, provider, *numbers, **settings):
        payload = analyze(company, policies(), self.sections, provider, self.labels(*numbers),
                          settings=pipeline_settings(**settings) if settings else None)['events'][0]['payload']
        return payload, {(o['source_label'].split(' md. ')[1], o['proposal']['applicability_scope']['child_clause']): o for o in payload['obligations']}

    def test_global_subject_scope_mismatch_settles_general_duties_without_any_model_call(self):
        for changes in (ASSOCIATION, SOFTWARE):
            with self.subTest(company=changes['id']):
                judge = Scripted()                                                       # would say APPLIES if it were asked
                payload, rows = self.rows(profile(**changes), judge, '3', '46')
                for key in (('3', '(1)'), ('46', '(1)')):
                    proposal = rows[key]['proposal']
                    self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('DOES_NOT_APPLY', 'SUBJECT_SCOPE_GATE'), key)
                    self.assertFalse(proposal['coverage_assessed'])
                    self.assertIn('not one of the obliged parties', proposal['coverage_reason'])
                    self.assertEqual(proposal['trace']['decided_by'], 'SUBJECT_SCOPE_GATE')
                    self.assertEqual(proposal['trace']['subject_gate']['status'], 'MISMATCH')
                    self.assertEqual(proposal['scope_evidence'][0]['source_id'], proposal['trace']['subject_gate']['evidence']['source_id'])
                    self.assertIn(proposal['scope_evidence'][0]['quote'], next(s['text'] for s in self.sections if s['id'] == proposal['scope_evidence'][0]['source_id']))
                self.assertEqual((judge.scoped, judge.asked), ([], []))                  # deterministic DNA: the model is never asked
                self.assertEqual(payload['obliged_parties']['items'], 3)

    def test_deterministic_does_not_apply_cannot_be_overridden_by_the_model(self):
        judge = Scripted('APPLIES')
        _, rows = self.rows(profile(**ASSOCIATION), judge, '3')
        self.assertEqual(rows[('3', '(1)')]['proposal']['applicability'], 'DOES_NOT_APPLY')
        self.assertEqual(judge.scoped, [])
        gates = [g['gate'] for g in rows[('3', '(1)')]['proposal']['trace']['gates']]
        self.assertEqual(gates, ['REGULATION_SUBJECT_SCOPE', 'COMPANY_ENTITY', 'JURISDICTION', 'CUSTOMER_ENTITY', 'CHILD_CLAUSE', 'EXEMPTION', 'MODEL',
                                 'FINAL_AGGREGATOR'])
        self.assertEqual(rows[('3', '(1)')]['proposal']['trace']['gates'][6]['status'], 'NOT_ASKED')

    def test_child_entity_mismatch_payment_institution_against_association_and_bank_only_clauses(self):
        judge = Scripted()
        _, rows = self.rows(profile(), judge, '8', '10', '11')
        for key, gate in ((('8', '(1)'), 'CUSTOMER_ENTITY'), (('10', '(1)'), 'COMPANY_ENTITY'), (('11', '(1)'), 'COMPANY_ENTITY')):
            proposal = rows[key]['proposal']
            self.assertEqual(proposal['applicability'], 'DOES_NOT_APPLY', key)
            status = {g['gate']: g['status'] for g in proposal['trace']['gates']}
            self.assertEqual(status[gate], 'MISMATCH', key)
        self.assertEqual(rows[('8', '(1)')]['proposal']['applicability_rule'], 'ENTITY_GATE')          # the v0.16.1 clause gate keeps its name
        self.assertEqual(rows[('10', '(1)')]['proposal']['applicability_rule'], 'ENTITY_GATE')
        self.assertEqual(rows[('11', '(1)')]['proposal']['applicability_rule'], 'ADDRESSEE_GATE')      # crypto providers: a kind the payment firm is not
        self.assertEqual(judge.scoped, [])

    def test_bank_against_association_clause_is_a_clear_match_and_an_undecided_model_cannot_leave_it_unknown(self):
        for answer in ('UNKNOWN', 'POSSIBLY_APPLIES'):
            with self.subTest(model=answer):
                judge = Scripted(answer)
                _, rows = self.rows(profile(**BANK), judge, '8')
                proposal = rows[('8', '(1)')]['proposal']
                self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('APPLIES', 'RULE_CLEAR_MATCH'))
                self.assertEqual(proposal['trace']['model_decision'], answer if answer == 'POSSIBLY_APPLIES' else 'POSSIBLY_APPLIES')
                self.assertEqual(proposal['trace']['applies_trace']['complete'], True, proposal['trace']['applies_trace'])
                self.assertIn('a) Bankalar', proposal['scope_evidence'][0]['quote'])
                self.assertEqual(proposal['company_fact_keys'][0], 'licences')
                self.assertEqual(proposal['review_flags'], [])
                self.assertIsNone(autonomous_gate(rows[('8', '(1)')]))

    def test_a_clear_match_needs_no_model_call_when_the_rule_decides(self):
        judge = Scripted()
        _, rows = self.rows(profile(**BANK), judge, '3', '8', applicability_clear_match='rule')
        for key in (('3', '(1)'), ('8', '(1)')):
            proposal = rows[key]['proposal']
            self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('APPLIES', 'RULE_CLEAR_MATCH'), key)
            self.assertEqual(proposal['trace']['model_gate']['mode'], 'not_asked')
            self.assertEqual(proposal['basis'][0]['match'], 'YES')
            applies = proposal['trace']['applies_trace']
            self.assertTrue(applies['complete'], applies)
            for field in ('regulation_subject_basis', 'company_matching_field', 'jurisdiction_basis', 'child_clause_match', 'exclusion_check',
                          'final_reasoning'):
                self.assertTrue(applies[field], field)
        self.assertEqual(judge.scoped, [])                                               # no applicability call at all
        self.assertTrue(judge.asked)                                                     # the policy is still read

    def test_a_model_does_not_apply_against_a_clear_match_is_kept_and_flagged_and_an_explicit_exclusion_is_never_lifted(self):
        judge = Scripted('DOES_NOT_APPLY')
        _, rows = self.rows(profile(**BANK), judge, '3')
        proposal = rows[('3', '(1)')]['proposal']
        self.assertEqual((proposal['applicability'], proposal['review_flags']), ('DOES_NOT_APPLY', ['RULE_MODEL_DISAGREEMENT']))
        self.assertIn('Flagged for a person', autonomous_gate(rows[('3', '(1)')]))
        # The model quotes an explicit exclusion for an APPLIES: refused, then UNKNOWN — the rule does not lift it.
        judge = Scripted('APPLIES', no_condition='bunların şubeleridir')
        judge.scope = lambda payload: {**FixtureProvider.scope(judge, payload), 'basis': [
            {'company_fact': 'bankacılık, mevduat kabulü', 'regulatory_condition': 'a) Bankalar.', 'match': 'YES'},
            {'company_fact': 'bankacılık, mevduat kabulü', 'regulatory_condition': 'bu fıkra ödeme kuruluşları hakkında uygulanmaz', 'match': 'NO'}]}
        _, rows = self.rows(profile(**BANK), judge, '12')
        proposal = rows[('12', '(1)')]['proposal']
        self.assertNotEqual(proposal['applicability'], 'APPLIES')
        self.assertIn('EXEMPTION_POSSIBLE', proposal['review_flags'])

    def test_explicit_exemption_in_the_clause_settles_and_one_in_a_sibling_sub_paragraph_goes_to_a_person(self):
        judge = Scripted()
        _, rows = self.rows(profile(), judge, '12', '13')
        exempt = rows[('12', '(1)')]['proposal']
        self.assertEqual((exempt['applicability'], exempt['applicability_rule']), ('DOES_NOT_APPLY', 'EXEMPTION_GATE'))
        self.assertIn('uygulanmaz', exempt['trace']['exemption_check']['evidence']['quote'])
        sibling = rows[('13', '(1)')]['proposal']
        self.assertEqual(sibling['trace']['exemption_check']['status'], 'POSSIBLE')
        self.assertEqual(sibling['applicability'], 'APPLIES')                            # the model decides a possible exemption...
        self.assertIn('EXEMPTION_POSSIBLE', sibling['review_flags'])                     # ...and a person confirms it
        self.assertEqual(judge.scoped, ['Yönetmelik 200713012 md. 13'])
        self.assertEqual(exemption_check('Bankalar hariç yükümlüler kayıt tutar.', 'Bankalar hariç yükümlüler kayıt tutar.', profile())['status'], 'NONE')

    def test_a_duty_addressed_to_customs_is_not_the_company_s(self):
        judge = Scripted()
        _, rows = self.rows(profile(), judge, '42')
        proposal = rows[('42', '(1)')]['proposal']
        self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('DOES_NOT_APPLY', 'ADDRESSEE_GATE'))
        self.assertIn('Gümrük İdaresine', proposal['applicability_reason'])
        self.assertEqual(judge.scoped, [])

    def test_an_ambiguous_profile_is_unknown_without_a_model_call_and_a_generic_customer_base_cannot_rest_a_does_not_apply(self):
        judge = Scripted()
        _, rows = self.rows(profile(name='Nova Finans Teknoloji', licences=None, customer_types=None, activities=['finansal teknoloji hizmetleri'],
                                    products=['mobil uygulama'], description='Lisans durumu belirtilmemiş sentetik fintech.'), judge, '3')
        proposal = rows[('3', '(1)')]['proposal']
        self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('UNKNOWN', 'PROFILE_INCOMPLETE'))
        self.assertIn('Missing company field: licences', proposal['missing_information'])
        self.assertEqual(judge.scoped, [])
        self.assertTrue(proposal['coverage_assessed'])                                   # the policy is still read for the reviewer
        judge = Scripted('DOES_NOT_APPLY')
        _, rows = self.rows(profile(customer_types=['tüzel kişi müşteriler']), judge, '8')
        proposal = rows[('8', '(1)')]['proposal']
        self.assertEqual((proposal['applicability'], proposal['applicability_rule']), ('UNKNOWN', 'PROFILE_AMBIGUOUS'))
        self.assertEqual(proposal['trace']['model_decision'], 'DOES_NOT_APPLY')

    def test_a_harmless_no_does_not_veto_and_an_applies_on_an_undetermined_clause_goes_to_a_person(self):
        judge = Scripted('APPLIES', no_condition=None)
        judge.scope = lambda payload: {**FixtureProvider.scope(judge, payload), 'basis': [
            {'company_fact': 'ödeme hizmetleri, elektronik para, dijital cüzdan', 'regulatory_condition': 'b) Ödeme kuruluşları ile elektronik para kuruluşları.', 'match': 'YES'},
            {'company_fact': 'ödeme hizmetleri, elektronik para, dijital cüzdan', 'regulatory_condition': 'a) Bankalar.', 'match': 'NO'}]}
        _, rows = self.rows(profile(customer_types=['tüzel kişi müşteriler']), judge, '3', '8')
        self.assertEqual(rows[('3', '(1)')]['proposal']['applicability'], 'APPLIES')          # "a) Bankalar" NO is a list item, not an exclusion
        undetermined = rows[('8', '(1)')]['proposal']
        self.assertEqual(undetermined['applicability'], 'APPLIES')
        self.assertIn('APPLIES_TRACE_INCOMPLETE', undetermined['review_flags'])
        self.assertIn('child_clause_match', undetermined['trace']['applies_trace']['missing'])

    def test_a_jurisdiction_outside_the_regulator_s_is_shown_and_sent_to_a_person_not_settled(self):
        judge = Scripted()
        _, rows = self.rows(profile(jurisdictions=['United Kingdom']), judge, '3')
        proposal = rows[('3', '(1)')]['proposal']
        self.assertEqual(proposal['trace']['jurisdiction_gate']['status'], 'MISMATCH')
        self.assertEqual(proposal['applicability'], 'APPLIES')
        self.assertIn('JURISDICTION_MISMATCH', proposal['review_flags'])

    def test_fca_chapters_have_no_obliged_list_and_keep_the_v017_behaviour(self):
        packet = analyze(company(), policies(), sections(), FixtureProvider(), ['CONC 7.3.4'])
        proposal = packet['events'][0]['payload']['obligations'][0]['proposal']
        self.assertEqual((proposal['applicability'], proposal['applicability_rule'], proposal['review_flags']), ('APPLIES', 'MODEL', []))
        self.assertEqual(proposal['trace']['subject_gate']['status'], 'NOT_RESTRICTED')
        self.assertTrue(proposal['trace']['applies_trace']['complete'])


class ObligedListTests(unittest.TestCase):
    def test_the_retained_tedbirler_list_is_read_item_by_item_and_profiles_are_placed_on_it(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        html = FIXTURES.joinpath('tedbirler-200713012.html').read_text(encoding='utf-8')
        save_sources(Path(temp.name) / 'reg', [download(html, URL)], mevzuat=('21', '200713012', '5'))
        rows = application_rows(load_sources(Path(temp.name) / 'reg')[1], 'YONETMELIK', '200713012')
        listed = obliged_list(rows)
        self.assertEqual((listed.source_label, len(listed.items), listed.unrecognised, listed.open_ended), ('Yönetmelik 200713012 md. 4', 29, [], False))
        self.assertEqual({i['marker']: i['kinds'] for i in listed.items}['e)'], ['PAYMENT_INSTITUTION', 'EMONEY_INSTITUTION'])
        self.assertEqual({i['marker']: i['kinds'] for i in listed.items}['b)'], ['CARD_ISSUER'])        # "Bankalar dışında" is not a bank
        placed = {name: subject_gate(profile(**changes), listed) for name, changes in
                  (('payment', {}), ('bank', BANK), ('association', ASSOCIATION), ('software', SOFTWARE),
                   ('fintech', dict(name='Nova Finans Teknoloji', activities=['finansal teknoloji hizmetleri'], licences=None, customer_types=None,
                                    products=['mobil uygulama'], description='Lisans durumu ve müşteri kitlesi belirtilmemiş sentetik fintech.')))}
        self.assertEqual({k: (v['status'], v['clear']) for k, v in placed.items()},
                         {'payment': ('MATCH', True), 'bank': ('MATCH', True), 'association': ('MISMATCH', True), 'software': ('MISMATCH', True),
                          'fintech': ('UNDETERMINED', False)})

    def test_a_profile_near_a_listed_kind_is_left_to_the_model_not_ruled_out(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        save_sources(Path(temp.name) / 'reg', [download(regulation_html(), URL)], mevzuat=('21', '200713012', '5'))
        listed = obliged_list(application_rows(load_sources(Path(temp.name) / 'reg')[1], 'YONETMELIK', '200713012'))
        for activity in ('ödeme sistemleri altyapısı', 'finansal teknoloji çözümleri', 'yatırım danışmanlığı', 'dijital cüzdan uygulaması'):
            with self.subTest(activity=activity):
                gate = subject_gate(profile(**{**SOFTWARE, 'activities': [activity], 'description': 'Sentetik şirket.'}), listed)
                self.assertEqual((gate['status'], gate['clear']), ('UNDETERMINED', False))
        self.assertEqual(subject_gate(profile(**SOFTWARE), listed)['status'], 'MISMATCH')      # says nothing near a listed kind

    def test_an_open_ended_definition_never_rules_a_company_out(self):
        from test_phase15 import anadolu, turkish_sections
        listed = obliged_list(application_rows(turkish_sections(), 'KANUN', '5549'))
        self.assertTrue(listed.financial_catch_all)
        self.assertEqual((subject_gate(anadolu(), listed)['status'], subject_gate(anadolu(), listed)['clear']), ('MATCH', False))
        self.assertEqual(subject_gate(Company(**{**anadolu().model_dump(), **SOFTWARE}), listed)['status'], 'UNDETERMINED')
        self.assertIsNone(obliged_list(sections()[:1]))                                  # FCA: no list


def tedbirler_list():
    temp = tempfile.mkdtemp()
    html = FIXTURES.joinpath('tedbirler-200713012.html').read_text(encoding='utf-8')
    save_sources(Path(temp) / 'reg', [download(html, URL)], mevzuat=('21', '200713012', '5'))
    return obliged_list(application_rows(load_sources(Path(temp) / 'reg')[1], 'YONETMELIK', '200713012'))


def firm(name='Sentetik Şirket A.Ş.', activities=(), licences=(), description=None, products=(), customers=('kurumsal müşteriler',)):
    return Company(id='x', name=name, version='1', synthetic=True, jurisdictions=['Türkiye'], activities=list(activities),
                   licences=list(licences), products=list(products), customer_types=list(customers), description=description)


class AdversarialReviewTests(unittest.TestCase):
    """Every case the adversarial review of the first rule-first version (24 September 2026) reproduced."""

    @classmethod
    def setUpClass(cls):
        cls.listed = tedbirler_list()

    def test_obliged_kinds_in_other_words_are_never_ruled_out(self):
        for profile_ in (firm('Anka Menkul Değerler A.Ş.', ['sermaye piyasası işlemleri, hisse senedi alım satımına aracılık'], ['SPK geniş yetkili lisans']),
                         firm(activities=['SMMM bürosu']), firm(activities=['sarraflık']), firm(activities=['külçe ve ziynet ticareti']),
                         firm(activities=['dijital varlık alım satım platformu']), firm(activities=['Bitcoin ve Ethereum alım satım platformu']),
                         firm(activities=['ikinci el otomobil ticareti']), firm(activities=['yat ve tekne alım satımı']),
                         firm(activities=['sanat galerisi, tablo satışı']), firm(activities=['iddaa bayiliği']),
                         firm(activities=['sanal POS ve tahsilat altyapısı'], licences=['TCMB lisansı']),
                         firm(activities=['odeme kurulusu hizmetleri']), firm(activities=['doviz burosu']),
                         firm(activities=['kiymetli maden alim satimi']), firm(activities=['tasinmaz alim satimi']),
                         firm('ABC LIFE INSURANCE CO.', ['life insurance']), firm(activities=['virtual asset service provider'])):
            with self.subTest(profile=profile_.name + ' / ' + ', '.join(profile_.activities)):
                self.assertNotEqual(subject_gate(profile_, self.listed)['status'], 'MISMATCH')

    def test_a_denied_kind_is_not_a_match_and_negation_stops_at_its_own_clause(self):
        for description in ('Şirket ödeme hizmeti sunmamaktadır.', 'Şirketin ödeme kuruluşu lisansı bulunmamaktadır.',
                            'Şirket ödeme kuruluşu olarak faaliyet göstermemektedir.'):
            with self.subTest(description=description):
                gate = subject_gate(firm(activities=['yazılım geliştirme'], description=description), self.listed)
                self.assertFalse(gate['status'] == 'MATCH' and gate['clear'], gate)
        self.assertEqual(subject_gate(firm(activities=['elektronik para ihracı; kredi vermez']), self.listed)['status'], 'MATCH')
        self.assertEqual(subject_gate(firm(activities=['online betting and casino operations']), self.listed)['status'], 'MATCH')

    def test_an_incidental_word_is_never_a_clear_match(self):
        for activities, description in ((['yazılım geliştirme'], 'Çalışanlarına özel sağlık sigortası sağlar.'),
                                        (['yazılım geliştirme'], 'Maaşlar banka hesabına ödenir.'), (['bankacılık yazılımları geliştirme'], None),
                                        (['muhasebe yazılımı'], None), (['sigorta eksperliği'], None), (['operasyonel leasing, araç kiralama'], None),
                                        (['üretim'], 'Ürünler kargo ile gönderilir.'), (['e-ticaret ile giyim satışı'], None)):
            with self.subTest(activities=activities, description=description):
                gate = subject_gate(firm(activities=activities, description=description), self.listed)
                self.assertFalse(gate['status'] == 'MATCH' and gate['clear'], gate)

    def test_a_profile_that_states_nothing_is_not_ruled_out(self):
        self.assertEqual(subject_gate(firm('Örnek Holding A.Ş.', customers=()), self.listed)['status'], 'UNDETERMINED')

    def test_the_most_specific_list_item_is_cited(self):
        self.assertTrue(subject_gate(firm(activities=['kuyumculuk, kıymetli maden alım satımı']), self.listed)['evidence']['list_item'].startswith('k)'))
        self.assertTrue(subject_gate(firm('Birikim Tasarruf Finansman A.Ş.', ['tasarruf finansman şirketi']), self.listed)['evidence']['list_item'].startswith('(v)'))
        limited = subject_gate(firm(activities=['elektronik ticaret aracı hizmet sağlayıcı, pazaryeri']), self.listed)
        self.assertEqual((limited['status'], limited['clear']), ('MATCH', False))           # md. 4(1)(y) is limited to part of the business

    def test_the_subject_decides_the_addressee_before_a_mention_of_yukumluler(self):
        from regchain.pilot.applicability import addressee_gate
        section = {'heading_path': ['YÖNETMELİK', 'ALTINCI BÖLÜM Diğer', 'Bilgi'], 'text': ''}
        clause_gate = {'required_entities': [], 'company_customer_families': []}
        bank = profile(**BANK)
        for subject in ('Başkanlık', 'Denetim yetkisini haiz olanlar', 'Kripto varlık hizmet sağlayıcısı olan yükümlüler'):
            with self.subTest(subject=subject):
                gate = addressee_gate(section, {'subject': subject}, f'(2) {subject}, yükümlülerden bilgi ister.', bank, clause_gate)
                self.assertEqual(gate['status'], 'MISMATCH')
        self.assertEqual(addressee_gate(section, {'subject': 'Yükümlüler'}, '(1) Yükümlüler kayıt tutar.', bank, clause_gate)['status'], 'MATCH')
        self.assertEqual(addressee_gate(section, {'subject': 'A firm'}, 'A firm must retain records.', bank, clause_gate)['status'], 'MATCH')

    def test_a_customer_or_transaction_carve_out_does_not_exempt_the_company(self):
        demo = profile()
        clause = '(3) Yükümlüler bilgi alır; müşterinin ödeme kuruluşu veya elektronik para kuruluşu olması halinde bu fıkra uygulanmaz.'
        self.assertEqual(exemption_check(clause, clause, demo)['status'], 'POSSIBLE')
        clause = '(3) Yükümlüler bilgi alır; bankalar arasında kendi nam ve hesaplarına yapılan transferler bu fıkra kapsamı dışındadır.'
        self.assertEqual(exemption_check(clause, clause, profile(**BANK))['status'], 'POSSIBLE')

    def test_jurisdictions_are_read_case_and_accent_free_and_kktc_is_not_turkiye(self):
        from regchain.pilot.applicability import jurisdiction_gate
        for value, turkish, status in (('Turkiye', True, 'MATCH'), ('TÜRKİYE', True, 'MATCH'), ('KKTC (Kuzey Kıbrıs Türk Cumhuriyeti)', True, 'MISMATCH'),
                                       ('UNITED KINGDOM', False, 'MATCH')):
            with self.subTest(value=value):
                self.assertEqual(jurisdiction_gate(profile(jurisdictions=[value]), turkish)['status'], status)


class SecondReviewTests(unittest.TestCase):
    """Every case the second adversarial review (24 September 2026) reproduced."""

    @classmethod
    def setUpClass(cls):
        cls.listed = tedbirler_list()

    def test_exception_wording_near_the_company_s_kind_is_never_an_exemption_of_the_company(self):
        bank = firm(activities=['bankacılık, mevduat kabulü'], products=['hayat sigortası ürünleri (acente olarak)'])
        payment = firm(activities=['ödeme hizmetleri'])
        for text, company in (('(2) Bankalar, resmî tatil günleri hariç olmak üzere, bildirimleri ertesi iş günü yapar.', bank),
                              ('(1) Ödeme kuruluşları, mesai saatleri dışında gerçekleşen işlemleri de kayda alır.', payment),
                              ('A payment institution must, except in an emergency, notify the FCA.', firm(activities=['payment services'])),
                              ('Bu fıkra, sigorta ve emeklilik şirketleri hakkında uygulanmaz.', bank)):
            with self.subTest(text=text):
                self.assertNotEqual(exemption_check(text, text, company)['status'], 'EXEMPT')
        for text in ('(1) Yükümlüler kayıt tutar; bu fıkra ödeme kuruluşları hakkında uygulanmaz.', 'Ödeme kuruluşları bu madde hükmünden muaftır.'):
            self.assertEqual(exemption_check(text, text, payment)['status'], 'EXEMPT')

    def test_obliged_non_financial_kinds_are_never_ruled_out(self):
        for name, activities in (('Kartal Jimnastik Kulübü Derneği', ['amatör spor faaliyetleri']), ('Sentetik A.Ş.', ['futbol takımı ve medya hakları']),
                                 ('Sentetik A.Ş.', ['turizm acenteliği ve kambiyo işlemleri']),
                                 ('Sentetik A.Ş.', ['inşaat taahhüt; konut projeleri geliştirme ve daire satışı']),
                                 ('Sentetik A.Ş.', ['motosiklet ve scooter perakende satış mağazası']), ('Sentetik A.Ş.', ['gemi imalatı ve satışı']),
                                 ('Sentetik A.Ş.', ['gümüş ve pırlanta takı perakende satışı']), ('Sentetik A.Ş.', ['tarihi eser ve koleksiyon ürünleri satışı']),
                                 ('Sentetik A.Ş.', ['Spor Toto bayiliği']), ('Sentetik A.Ş.', ['çevrim içi pazar yeri işletmeciliği; yazılım geliştirme'])):
            with self.subTest(activities=activities):
                self.assertNotEqual(subject_gate(firm(name, activities), self.listed)['status'], 'MISMATCH')

    def test_a_kind_split_from_its_negation_or_its_object_by_ve_or_ile_is_not_a_clear_match(self):
        for activities, licences in ((['ödeme kuruluşları ve elektronik para kuruluşları için yazılım geliştirme'], []),
                                     (['software for banks and payment institutions'], []), (['sigorta şirketleri ve bankalara danışmanlık'], []),
                                     (['yazılım'], ['ödeme kuruluşu veya elektronik para kuruluşu lisansı bulunmamaktadır']),
                                     (['the company is not a bank or payment institution'], []), (['neither a bank nor an e-money institution'], []),
                                     (['banka dışı finansman hizmetleri'], []), (['ödeme kuruluşu lisansı başvurusu süreçte'], []),
                                     (['finansal kiralama'], ['BDDK (Bankacılık Düzenleme ve Denetleme Kurumu) faaliyet izni'])):
            with self.subTest(activities=activities, licences=licences):
                gate = subject_gate(firm(activities=activities, licences=licences), self.listed)
                self.assertFalse(gate['status'] == 'MATCH' and gate['clear'] and gate['evidence'].get('company_value', '').lower().startswith(('bddk', 'ödeme kuruluşu veya')), gate)
                if 'finansal kiralama' not in activities:
                    self.assertFalse(gate['status'] == 'MATCH' and gate['clear'], gate)
        cargo = subject_gate(firm(activities=['sigortasız kargo taşımacılığı']), self.listed)
        self.assertTrue(cargo['evidence']['list_item'].startswith('i)'), cargo)

    def test_a_genuine_contradiction_is_never_withdrawn_as_a_restatement(self):
        from regchain.pilot.engine import restates_prohibition
        simplified = {'modality': 'MUST_NOT', 'prohibited_action': 'basitleştirilmiş tedbirleri uygulayamazlar', 'conditions': []}
        anonymous = {'modality': 'MUST_NOT', 'prohibited_action': 'isimsiz veya hayali isimlere hesap açamazlar', 'conditions': []}
        for duty, quote in ((simplified, 'İşlem şüpheli görünse dahi basitleştirilmiş tedbirlerden vazgeçilmez.'),
                            (simplified, 'Basitleştirilmiş tedbirlerin uygulanması engellenmez.'),
                            (anonymous, '50.000 TL üzerindeki işlemlerde isimsiz veya hayali isimlere hesap açılmaz.'),
                            ({'modality': 'MUST_NOT', 'prohibited_action': 'open anonymous accounts', 'conditions': []},
                             'The Bank does not prohibit opening anonymous accounts for low-risk customers.')):
            with self.subTest(quote=quote):
                self.assertFalse(restates_prohibition(duty, quote))

    def test_a_withdrawn_contradiction_is_a_judgement_not_a_failure(self):
        class Alleging(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                if 'contradicts' in schema['properties']:
                    return json.dumps({'contradicts': 'YES', 'quote': payload['passage'], 'reason': 'Alleged.'})
                return super()._chat(prompt, payload, schema)
        text = 'Aklama riski oluşabilecek durumlarda basitleştirilmiş tedbir uygulanmaz.'
        single = [{'filename': 'p.txt', 'raw_hash': 'd' * 64, 'bytes': 80, 'parser': 'fixture', 'chunks': [
            {'source_id': 'p1', 'policy_hash': 'd' * 64, 'filename': 'p.txt', 'locator': 'text_block', 'number': 1, 'start': 0, 'end': len(text), 'text': text}]}]
        from regchain.pilot.engine import propose
        duty = {'subject': 'Yükümlüler', 'modality': 'MUST_NOT', 'required_action': None, 'prohibited_action': 'basitleştirilmiş tedbirleri uygulayamazlar',
                'conditions': ['aklama riskinin oluşabileceği durumlarda'], 'exceptions': []}
        proposal, _ = propose(Alleging(), company(), sections()[1], duty, sections()[:1], single[0]['chunks'])
        self.assertFalse(proposal.coverage_reason.startswith('Analysis unavailable'), proposal.coverage_reason)

    def test_a_kind_read_from_the_name_is_cited_by_a_stated_field(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        save_sources(Path(temp.name) / 'reg', [download(regulation_html(), URL)], mevzuat=('21', '200713012', '5'))
        sections_ = load_sources(Path(temp.name) / 'reg')[1]
        from regchain.pilot.engine import autonomous_review
        from regchain.pilot.review import apply_review
        for changes in (dict(name='Nova Bank A.Ş.', activities=['mevduat ve kredi'], licences=[], description='Sentetik.'),
                        dict(name='Yeşil Vadi Dernek', activities=['doğa koruma projeleri'], licences=[], description='Sentetik.')):
            with self.subTest(name=changes['name']):
                packet = analyze(profile(**changes), policies(), sections_, Scripted(), select_targets(sections_, 'YONETMELIK', '200713012', 'all', ['3']),
                                 settings=pipeline_settings(applicability_clear_match='rule'))
                for row in packet['events'][0]['payload']['obligations']:
                    self.assertNotIn('name', row['proposal']['company_fact_keys'])
                apply_review(packet, autonomous_review(packet)[0])


class RuleDecisionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        save_sources(Path(self.temp.name) / 'reg', [download(regulation_html(), URL)], mevzuat=('21', '200713012', '5'))
        self.sections = load_sources(Path(self.temp.name) / 'reg')[1]

    def test_a_rule_does_not_apply_carries_a_profile_field_and_an_exact_quote_and_the_autonomous_review_validates(self):
        from regchain.pilot.engine import autonomous_review
        from regchain.pilot.review import apply_review
        for changes, numbers in ((ASSOCIATION, ['3', '46']), ({}, ['8', '10', '11', '12', '42'])):
            with self.subTest(company=changes.get('id', 'demopay')):
                packet = analyze(profile(**changes), policies(), self.sections, Scripted(),
                                 select_targets(self.sections, 'YONETMELIK', '200713012', 'all', numbers))
                payload = packet['events'][0]['payload']
                sources = {s['id']: s['text'] for s in [*payload['scope_sources'], *(c['source'] for c in payload['cases'])]}
                for row in payload['obligations']:
                    proposal = row['proposal']
                    if proposal['applicability'] != 'DOES_NOT_APPLY':
                        continue
                    self.assertTrue(proposal['company_fact_keys'], row['source_label'])
                    for quote in proposal['scope_evidence']:
                        self.assertIn(quote['quote'], sources[quote['source_id']])
                review, summary = autonomous_review(packet)
                apply_review(packet, review)                                              # used to raise for rule-decided rows


class RelevanceScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        save_sources(Path(self.temp.name) / 'reg', [download(regulation_html(), URL)], mevzuat=('21', '200713012', '5'))
        self.sections = load_sources(Path(self.temp.name) / 'reg')[1]
        texts = ['Yükümlü sıfatıyla Şirket belgeleri sekiz yıl saklar.', 'Kayıtlar sekiz yıl muhafaza edilir ve istenirse ibraz edilir.',
                 'Arşiv birimi belge saklama takvimini yönetir.', 'Parolalar en az on iki karakter olur.', 'Yıllık izin bir hafta önce bildirilir.',
                 'Müşteri kayıtları beş yıl saklanır ve sonra imha edilir.']
        chunks = [{'source_id': f'p{i}', 'policy_hash': 'd' * 64, 'filename': 'politika.txt', 'locator': 'text_block', 'number': i,
                   'start': 0, 'end': len(t), 'text': t} for i, t in enumerate(texts, 1)]
        self.policies = [{'filename': 'politika.txt', 'raw_hash': 'd' * 64, 'bytes': 100, 'parser': 'fixture', 'chunks': chunks}]
        self.labels = select_targets(self.sections, 'YONETMELIK', '200713012', 'all', ['46'])

    def test_passages_screened_out_are_recorded_unrelated_and_never_put_to_the_judge(self):
        judge = Scripted(screen=lambda text: 'OTHER' if 'Parola' in text or 'izin' in text else 'SAME_SUBJECT')
        payload = analyze(profile(), self.policies, self.sections, judge, self.labels,
                          settings=pipeline_settings(relevance_screen='on'))['events'][0]['payload']
        row = payload['obligations'][0]
        relations = {c['source_id']: c['relation'] for c in row['proposal']['policy_checks']}
        self.assertEqual(len(relations), 6)
        screened = [r for r in next(d for d in row['diagnostics'] if d.get('stage') == 'passages')['results'] if r.get('screen') == 'SCREENED_OUT']
        self.assertEqual(len(screened), 2)
        self.assertTrue(all(relations[r['source_id']] == 'UNRELATED' for r in screened))
        self.assertEqual(len(judge.asked), 4)                                            # six passages, two set aside
        self.assertEqual(len(judge.screened), 1)                                         # one screen call per duty
        self.assertEqual(len(judge.screened[0]['passages']), 6 - SCREEN_ALWAYS_JUDGE)   # the best-ranked are never screened
        self.assertEqual(payload['pipeline_settings']['relevance_screen'], 'on')
        self.assertIsNotNone(payload['pipeline_settings']['relevance_prompt_sha256'])
        self.assertIn('relevance_screen', payload['timings'])

    def test_a_passage_that_skips_or_exempts_is_never_screened_out(self):
        skipping = dict(self.policies[0]['chunks'][4], source_id='p9', text='Kampanya döneminde kayıt adımı atlanır; bu istisna yöneticinin onayıyla uygulanır.')
        policies_ = [dict(self.policies[0], chunks=[*self.policies[0]['chunks'], skipping])]
        judge = Scripted(screen=lambda text: 'OTHER')
        analyze(profile(), policies_, self.sections, judge, self.labels, settings=pipeline_settings(relevance_screen='on'))
        self.assertIn(skipping['text'], judge.asked)                                    # judged in full although the screen said OTHER
        self.assertNotIn(skipping['text'], [p['text'] for p in judge.screened[0]['passages']])

    def test_a_failed_screen_sets_nothing_aside(self):
        judge = Scripted(screen=None)                                                    # the screen answer is refused
        payload = analyze(profile(), self.policies, self.sections, judge, self.labels,
                          settings=pipeline_settings(relevance_screen='on'))['events'][0]['payload']
        row = payload['obligations'][0]
        self.assertEqual(len(judge.asked), 6)
        self.assertIn('SCREEN_UNAVAILABLE', [d.get('code') for d in row['diagnostics']])

    def test_the_screen_is_off_by_default_and_a_settings_change_blocks_carrying_rows_forward(self):
        judge = Scripted(screen=lambda text: 'OTHER')
        first = analyze(profile(), self.policies, self.sections, judge, self.labels)
        self.assertEqual((judge.screened, len(judge.asked)), ([], 6))
        payload = first['events'][0]['payload']
        again = analyze(profile(), self.policies, self.sections, Scripted(screen=lambda text: 'SAME_SUBJECT'), self.labels, previous=[],
                        previous_head=first['head'], previous_payload=payload, settings=pipeline_settings(relevance_screen='on'))
        self.assertIn('pipeline settings changed', again['events'][0]['payload']['impact']['reuse_blocked_reason'])


class RestatedProhibitionTests(unittest.TestCase):
    """C14 of the golden set (and a real policy in v0.16): a sentence repeating the duty's own prohibition was a CONFLICT."""

    def judged(self, quote, duty):
        confirms = []

        class Alleging(FixtureProvider):
            def _chat(self, prompt, payload, schema):
                fields = schema['properties']
                if 'contradicts' in fields:
                    return json.dumps({'contradicts': 'YES', 'quote': quote, 'reason': 'Alleged.'})
                if 'verdict' in fields:
                    confirms.append(payload)
                    return json.dumps({'verdict': 'CONTRADICTS', 'duty_requires': 'd', 'passage_instructs': 'p', 'reason': 'r'})
                return super()._chat(prompt, payload, schema)
        from regchain.pilot.engine import judge_passage
        row = {'source_id': 'p1', 'text': quote}
        return judge_passage(Alleging(), duty, row), confirms

    def test_a_restated_prohibition_is_withdrawn_by_rule_and_a_permission_still_goes_to_the_second_reading(self):
        duty = {'subject': 'Yükümlüler', 'modality': 'MUST_NOT', 'required_action': None, 'prohibited_action': 'basitleştirilmiş tedbirleri uygulayamazlar',
                'conditions': ['işlem nedeniyle aklama veya terörün finansmanı riskinin oluşabileceği durumlarda'], 'exceptions': []}
        (judgement, notes, screen), confirms = self.judged('Aklama veya terörün finansmanı riski oluşabilecek durumlarda basitleştirilmiş tedbir '
                                                           'uygulanmaz ve işlem şüpheli işlem değerlendirmesine konu edilir.', duty)
        self.assertNotEqual(judgement.relation, 'CONFLICTS')
        self.assertEqual((screen, confirms), ('WITHDRAWN', []))                         # no second reasoning call was needed
        self.assertIn('restates the duty', next(n for n in notes if n.get('code') == 'CONFLICT_WITHDRAWN')['detail'])
        (judgement, notes, screen), confirms = self.judged('Kampanya dönemlerinde basitleştirilmiş tedbirler bütün müşterilere uygulanabilir.', duty)
        self.assertEqual(judgement.relation, 'CONFLICTS')
        self.assertEqual(len(confirms), 1)
        positive = dict(duty, modality='MUST', required_action='kimlik tespitini işlemden önce tamamlamak', prohibited_action=None)
        (judgement, _, _), confirms = self.judged('Kimlik tespiti yapılmaz.', positive)       # a MUST duty is never "restated" by a negation
        self.assertEqual((judgement.relation, len(confirms)), ('CONFLICTS', 1))


class ContextOverflowRecoveryTests(unittest.TestCase):
    """CONTEXT_BUDGET_EXCEEDED used to end the duty "Analysis unavailable"; now the long texts are clipped once and asked again."""

    def test_an_overlong_passage_and_an_overlong_provision_are_clipped_and_asked_again(self):
        from regchain.extraction.providers import ContextBudgetError, fits_budget
        from regchain.pilot.engine import ask, judge_scope
        from regchain.pilot.schema import Screen
        seen = []

        class Tight(FixtureProvider):
            num_predict, num_ctx = 4096, 16384

            def _chat(self, prompt, payload, schema):
                if not fits_budget(prompt, payload, schema, self.num_predict, self.num_ctx):
                    raise ContextBudgetError('CONTEXT_BUDGET_EXCEEDED: prompt and reserved output exceed admission budget')
                seen.append(payload)
                return super()._chat(prompt, payload, schema)
        long_text = ('Yükümlüler, müşterilerinin kimliğini işlem yapılmadan önce tespit eder. ' * 700).strip()
        row = {'source_id': 'p1', 'text': long_text}
        duty = {'subject': 'Yükümlüler', 'modality': 'MUST', 'required_action': 'kimlik tespit etmek', 'prohibited_action': None,
                'conditions': [], 'exceptions': []}
        answer, notes = ask(Tight(), 'You check a passage.', Screen, 'contradicts', ('YES',), duty, row)
        self.assertIsNotNone(answer)
        self.assertIn('trimmed payload', notes[0]['retry'])
        self.assertLess(len(seen[-1]['passage']), len(long_text))
        section = dict(sections()[1], text=long_text, id='s-long')
        scope, basis, rule, diagnostics, failure = judge_scope(Tight(), company(), section, duty, {'s1': sections()[0]})
        self.assertIsNone(failure)
        self.assertIn('trimmed payload', next(d for d in diagnostics if d.get('code') == 'ContextBudgetError')['retry'])


class JudgeWindowTests(unittest.TestCase):
    def test_a_narrow_judge_window_keeps_a_full_window_twin_for_applicability_on_the_same_log(self):
        import os
        from unittest.mock import patch
        from regchain.extraction.providers import NUM_CTX, OllamaProvider, configured_judge
        base = OllamaProvider('qwen3:4b', 'a' * 64, 'http://localhost:11434', 120)
        env = {'JUDGE_MODEL': 'qwen3:8b', 'JUDGE_MODEL_DIGEST': 'b' * 64, 'JUDGE_THINKING': 'on'}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop('JUDGE_NUM_CTX', None)
            judge = configured_judge(base)
            self.assertEqual((judge.num_ctx, getattr(judge, 'wide', None)), (NUM_CTX, None))          # the v0.17 default
        with patch.dict(os.environ, {**env, 'JUDGE_NUM_CTX': '8192'}):
            judge = configured_judge(base)
        self.assertEqual((judge.num_ctx, judge.quick.num_ctx, judge.wide.num_ctx, judge.quick.wide.num_ctx), (8192, 8192, NUM_CTX, NUM_CTX))
        self.assertLessEqual(judge.num_predict, 4096)
        self.assertIs(judge.wide.call_log, judge.call_log)                                   # the harness and workspace see its calls
        self.assertIs(judge.quick.wide.cache, judge.quick.cache)
        self.assertEqual((judge.wide.thinking, judge.quick.wide.thinking), (True, False))
        self.assertEqual(judge.runtime_manifest()['wide_num_ctx'], NUM_CTX)


class CoverageAggregationTests(unittest.TestCase):
    def test_strong_support_beside_irrelevant_and_one_uncertain_passage_is_not_unknown(self):
        checks = lambda *relations: [PolicyCheck(source_id=f'p{i}', quote='q', relation=r) for i, r in enumerate(relations)]
        self.assertEqual(coverage_of(checks('SUPPORTS', 'SUPPORTS', 'UNRELATED', 'UNRELATED', 'UNCLEAR'))[0], 'COVERS_TEXT')
        self.assertEqual(coverage_of(checks('SUPPORTS', 'UNRELATED', 'UNRELATED', 'UNRELATED'))[0], 'COVERS_TEXT')
        self.assertEqual(coverage_of(checks('SUPPORTS', 'SUPPORTS', 'CONFLICTS'))[0], 'CONFLICT')      # an explicit contradiction is apart


if __name__ == '__main__':
    unittest.main()

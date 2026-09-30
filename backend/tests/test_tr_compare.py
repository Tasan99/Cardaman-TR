"""Obligation -> policy -> control -> evidence -> gap on the synthetic pilot registers (no model, no network).

The registers are synthetic test data. The expectations say what the deterministic statement reader does on them;
where a developer's own reading differs, the evaluation set (BEVERAGE_TR_DEV_V2) records it as a miss, not this file.
"""
import unittest

from regchain.tr.compare import (GapRow, Passage, combine_coverage, compare_obligation, compare_profile, coverage_of, distinctive,
                                 documents_in_force, load_register, ownership, passage_frames, relate)
from regchain.tr.corpus import CorpusStore
from regchain.tr.extraction import extract_regulation, route
from regchain.tr.mapping import mapping_status
from regchain.tr.packs import Registry
from regchain.tr.profile import load_pilot_profiles

REGISTRY = Registry.load()
STORE = CorpusStore()
PROFILES = load_pilot_profiles(REGISTRY.vocabulary)
BREWER = PROFILES['ALCOHOL_GROUP_EFES_TYPE']
BOTTLER = PROFILES['NON_ALCOHOL_GROUP_COCA_COLA_TYPE']
BREWER_REGISTER = load_register(BREWER.profile_id)
BOTTLER_REGISTER = load_register(BOTTLER.profile_id)
_OBLIGATIONS = {}


def obligation(regulation_id, ref, group=None):
    if regulation_id not in _OBLIGATIONS:
        _OBLIGATIONS[regulation_id] = extract_regulation(regulation_id, REGISTRY, STORE)[1]
    found = [o for o in _OBLIGATIONS[regulation_id] if o.provision_ref == ref and (group is None or o.group == group)]
    assert len(found) == 1, ref
    return found[0]


def row(profile, register, regulation_id, ref, target_id, group=None, second=None) -> GapRow:
    item = obligation(regulation_id, ref, group)
    decisions, _ = route(item, profile, REGISTRY, STORE)
    decision = next(d for d in decisions if d.target_id == target_id)
    assert decision.status in ('APPLIES', 'PARTIAL'), (ref, target_id, decision.status)
    return compare_obligation(item, decision, profile, register, REGISTRY, second)


def statement(text, document_id='DOC-TEST', number=1) -> Passage:
    return Passage(passage_id=f'{document_id}#{number}', document_id=document_id, number=number, text=text, start=0, end=len(text))


def reading(regulation_id, ref, text, group=None):
    return relate(obligation(regulation_id, ref, group), statement(text), REGISTRY.vocabulary)


class RegisterTests(unittest.TestCase):
    def test_the_registers_are_synthetic_and_every_statement_is_an_exact_span(self):
        for profile, register in ((BREWER, BREWER_REGISTER), (BOTTLER, BOTTLER_REGISTER)):
            self.assertTrue(register.synthetic)
            self.assertIn('Gerçek bir şirketin', register.disclaimer)
            self.assertGreaterEqual(len(register.documents), 5)
            self.assertTrue(all(len(d.content_hash) == 64 for d in register.documents))
            from regchain.tr.compare import REGISTERS
            for document in register.documents:
                text = (REGISTERS / profile.profile_id / document.file).read_text(encoding='utf-8').replace('\r\n', '\n')
                own = [p for p in register.passages if p.document_id == document.document_id]
                self.assertTrue(own, document.document_id)
                for passage in own:
                    self.assertEqual(text[passage.start:passage.end], passage.text)
                self.assertNotIn('Sentetik test belgesidir', ' '.join(p.text for p in own))     # headers are not statements

    def test_no_real_brand_name_appears_in_the_registers(self):
        for register in (BREWER_REGISTER, BOTTLER_REGISTER):
            text = ' '.join(p.text for p in register.passages).lower() + ' '.join(d.title for d in register.documents).lower()
            for brand in ('efes', 'tuborg', 'coca', 'pepsi', 'carlsberg', 'fanta', 'sprite', 'red bull', 'burn'):
                self.assertNotIn(brand, text)

    def test_a_semicolon_separates_two_statements_with_their_own_polarity(self):
        frames = passage_frames(statement('Alkollü içkiler on sekiz yaşını doldurmamış kişilere satılamaz ve sunulamaz; '
                                          'yaş konusunda tereddüt halinde kimlik belgesi istenir.'))
        self.assertEqual([f.modality for f in frames], ['MUST_NOT', 'MUST'])

    def test_ownership_names_a_department_for_every_topic_the_frames_can_give(self):
        topics = ownership()['topics']
        for topic in ('ADVERTISING', 'BRANDING', 'LABELLING', 'COMPOSITION', 'CLAIMS', 'AGE_RESTRICTION', 'SALE', 'ECOMMERCE',
                      'LICENSING', 'EMPLOYMENT', 'PACKAGING_WASTE', 'PACKAGING', 'MARKET_PLACEMENT', 'PRODUCTION', 'LOGISTICS',
                      'HYGIENE', 'GENERAL'):
            self.assertTrue(topics[topic], topic)
            self.assertTrue(set(topics[topic]) <= set(ownership()['departments']))


class StatementRelationTests(unittest.TestCase):
    AD_BAN = ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.1/c.1')
    CAFFEINE = ('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 5/f.1/b.b')
    NIGHT = ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.5/c.3')

    def test_the_same_act_polarity_and_product_supports(self):
        result = reading(*self.AD_BAN, 'Alkollü içkilerin reklamı ve tüketicilere yönelik tanıtımı yapılamaz.')
        self.assertEqual((result.relation, result.reasons), ('SUPPORTS', ['ACT_AND_POLARITY_MATCH']))

    def test_shared_words_without_the_same_act_are_not_support(self):
        # The words of the ban appear; the statement is about an approval step for another product.
        result = reading(*self.AD_BAN, 'Alkolsüz bira için hazırlanan reklam ve tanıtım çalışmaları yayından önce Hukuk onayına sunulur.')
        self.assertEqual((result.relation, result.reasons), ('UNRELATED', ['OTHER_PRODUCT']))
        other = reading(*self.NIGHT, 'Alkollü içkiler otomatik satış makineleri ile satılamaz.')
        self.assertEqual(other.relation, 'UNRELATED')                       # another rule about selling alcohol

    def test_a_statement_that_allows_an_absolute_prohibition_conflicts(self):
        gifts = ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.2')
        result = reading(*gifts, 'Alkollü içkiler, yalnızca tadım etkinliklerinde bedelsiz olarak dağıtılabilir.')
        self.assertEqual((result.relation, result.reasons), ('CONFLICTS', ['POLICY_PERMITS_PROHIBITED_ACT']))

    def test_a_weaker_limit_conflicts_and_a_stricter_one_supports(self):
        weaker = reading(*self.CAFFEINE, 'Enerji içeceğinde toplam kafein miktarı 320 mg/L’den fazla olamaz.')
        self.assertEqual((weaker.relation, weaker.reasons), ('CONFLICTS', ['LIMIT_WEAKER']))
        self.assertEqual(weaker.detail['quantities'][0], {'attribute': 'caffeine_mg_per_l', 'duty': 150.0, 'policy': 320.0,
                                                          'unit': 'mg/L', 'relation': 'WEAKER'})
        same = reading(*self.CAFFEINE, 'Enerji içeceğinde toplam kafein miktarı 150 mg/L’den fazla olamaz.')
        stricter = reading(*self.CAFFEINE, 'Enerji içeceğinde toplam kafein miktarı 120 mg/L’den fazla olamaz.')
        self.assertEqual((same.relation, stricter.relation), ('SUPPORTS', 'SUPPORTS'))
        self.assertIn('LIMIT_SAME_OR_STRICTER', stricter.reasons)

    def test_a_number_that_selects_products_is_not_a_limit(self):
        # "1,0 mg/L’den fazla olan ürünlerde" names the products a label rule covers; it does not cap caffeine.
        cap = ('TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER', 'Tebliğ 11376 md. 5/f.1/b.c')
        result = reading(*cap, 'Kafein miktarı 1,0 mg/L’den fazla olan ürünlerde “Kafein içerir” ifadesi ürün adının altında yer alır.')
        self.assertNotEqual(result.relation, 'SUPPORTS')
        self.assertIn('LIMIT_MISSING:caffeine_mg_per_l', result.reasons)

    def test_a_sale_inside_the_forbidden_hours_conflicts_and_outside_does_not(self):
        late = reading(*self.NIGHT, 'Satış noktasında alkollü içkilerin perakende satışı 23:00 saatine kadar yapılır.')
        self.assertEqual((late.relation, late.reasons), ('CONFLICTS', ['TIME_WINDOW_VIOLATED']))
        early = reading(*self.NIGHT, 'Satış noktasında alkollü içkilerin perakende satışı 21:30 saatine kadar yapılır.')
        self.assertEqual(early.relation, 'UNRELATED')
        restated = reading(*self.NIGHT, 'Alkollü içkiler 22:00 ila 06:00 saatleri arasında perakende olarak satılamaz.')
        self.assertEqual(restated.relation, 'SUPPORTS')

    def test_a_missing_place_or_predicate_is_partial_with_the_missing_part_named(self):
        venues = ('TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 11/f.2')
        result = reading(*venues, 'Okul kantinlerinde ve spor tesislerinde enerji içecekleri piyasaya arz edilmez ve reklamı yapılmaz.',
                         group='MARKETING')
        self.assertEqual((result.relation, result.reasons), ('PARTIAL', ['PLACE_MISSING:HEALTH_FACILITY']))
        vending = ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.5/c.1')
        half = reading(*vending, 'Alkollü içkiler otomatik satış makineleri ile satılamaz.')
        self.assertEqual((half.relation, half.reasons), ('PARTIAL', ['PREDICATE_NOT_STATED']))

    def test_a_clause_that_names_both_sides_needs_a_statement_about_both(self):
        brand = ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.9/c.1')
        crossing = reading(*brand, 'Alkollü içki markalarımızın adı, amblemi ve logosu alkolsüz bira ambalajında ve tanıtımında kullanılabilir.')
        self.assertEqual(crossing.relation, 'CONFLICTS')
        one_side = reading(*brand, 'Alkollü içki markalarımızın logoları sponsor olunan konser ve festivallerde kullanılabilir.')
        self.assertEqual((one_side.relation, one_side.reasons), ('UNRELATED', ['OTHER_PRODUCT']))

    def test_an_exception_the_duty_does_not_have_makes_support_partial(self):
        minors = ('TR:KANUN:4250', 'Kanun 4250 md. 6/f.3')
        result = reading(*minors, 'Alkollü içkiler on sekiz yaşını doldurmamış kişilere satılamaz ve sunulamaz. '
                                  'Aile gözetiminde sunum bu kuralın dışındadır.')
        self.assertEqual(result.relation, 'PARTIAL')
        self.assertIn('POLICY_ADDS_EXCEPTION', result.reasons)

    def test_distinctive_wording_leaves_out_the_product_words(self):
        night = obligation(*self.NIGHT).frame
        self.assertNotIn('alkol', distinctive(night))
        self.assertIn('perak', distinctive(night))

    def test_coverage_is_counted_conflict_first(self):
        mk = lambda relation: reading(*self.AD_BAN, 'x').model_copy(update={'relation': relation, 'reasons': ['R']})
        self.assertEqual(coverage_of([mk('SUPPORTS'), mk('CONFLICTS')])[0], 'CONFLICT')
        self.assertEqual(coverage_of([mk('PARTIAL'), mk('SUPPORTS')])[0], 'COVERS_TEXT')
        self.assertEqual(coverage_of([mk('PARTIAL'), mk('UNCLEAR')])[0], 'PARTIAL')
        self.assertEqual(coverage_of([mk('UNCLEAR')])[0], 'UNKNOWN')
        self.assertEqual(coverage_of([])[0], 'NO_EVIDENCE')


class BrewerGapTests(unittest.TestCase):
    LAW = 'TR:KANUN:4250'

    def gap(self, ref, target, regulation_id=None, group=None):
        return row(BREWER, BREWER_REGISTER, regulation_id or self.LAW, ref, target, group)

    def test_covered_needs_the_statement_a_control_and_in_period_evidence(self):
        ads = self.gap('Kanun 4250 md. 6/f.1/c.1', 'ALC-INT-ACT-BEER-ADS')
        self.assertEqual((ads.mapping.status, ads.mapping.reasons), ('COVERED', ['POLICY_COVERS', 'CONTROL_MAPPED', 'EVIDENCE_PRESENT']))
        self.assertEqual((ads.mapping.document_ids, ads.mapping.control_ids, ads.mapping.evidence_ids),
                         (['POL-MKT-01'], ['CTL-MKT-APPROVAL'], ['EV-MKT-2026Q3']))
        self.assertIsNone(ads.gap)
        self.assertEqual(ads.actions, [])

    def test_a_statement_that_allows_the_forbidden_act_is_contradicted_whatever_else_exists(self):
        sponsor = self.gap('Kanun 4250 md. 6/f.1/c.4', 'ALC-INT-HOLDING')
        self.assertEqual(sponsor.mapping.status, 'CONTRADICTED')
        self.assertEqual([(a['action'], a['target'], a['department']) for a in sponsor.actions],
                         [('UPDATE_DOCUMENT', 'POL-MKT-01#2', 'MARKETING')])
        self.assertTrue(sponsor.review_required)
        night = self.gap('Kanun 4250 md. 6/f.5/c.3', 'ALC-INT-SALES')
        self.assertEqual((night.mapping.status, night.coverage_reasons), ('CONTRADICTED', ['TIME_WINDOW_VIOLATED']))
        brand = self.gap('Kanun 4250 md. 6/f.9/c.1', 'ALC-INT-ACT-AF-BRAND')
        self.assertEqual((brand.mapping.status, brand.actions[0]['target']), ('CONTRADICTED', 'POL-MKT-01#6'))

    def test_stale_evidence_and_a_missing_control_are_partial_not_covered(self):
        minors = self.gap('Kanun 4250 md. 6/f.3', 'ALC-INT-SALES')
        self.assertEqual((minors.document_coverage, minors.mapping.status, minors.mapping.reasons),
                         ('COVERS_TEXT', 'PARTIALLY_COVERED', ['POLICY_COVERS', 'EVIDENCE_MISSING']))       # the log is from 2025-Q4
        self.assertEqual(minors.actions, [{'action': 'COLLECT_EVIDENCE', 'target': 'CTL-AGE-CHECK', 'department': 'SALES',
                                           'why': ['EVIDENCE_MISSING']}])
        vending = self.gap('Kanun 4250 md. 6/f.5/c.1', 'ALC-INT-SALES')
        self.assertEqual((vending.document_coverage, vending.mapping.status), ('PARTIAL', 'PARTIALLY_COVERED'))
        self.assertEqual(vending.actions[0]['action'], 'COMPLETE_STATEMENT')

    def test_a_document_of_another_entity_does_not_cover_the_duty(self):
        # The sales procedure belongs to the sales company; the distribution company sells too and has nothing written.
        distribution = self.gap('Kanun 4250 md. 6/f.3', 'ALC-INT-DISTRIBUTION')
        self.assertNotIn('SOP-SALES-02', distribution.documents_in_force)
        self.assertEqual((distribution.mapping.status, distribution.actions[0]['action']), ('NOT_COVERED', 'ADD_POLICY_STATEMENT'))

    def test_the_label_duty_is_covered_for_the_domestic_beers_and_does_not_reach_the_export_beer(self):
        lager = self.gap('Kanun 4250 md. 6/f.8/c.1', 'ALC-INT-LAGER')
        self.assertEqual((lager.mapping.status, lager.mapping.document_ids), ('COVERED', ['SPEC-LABEL-03']))
        decisions, _ = route(obligation(self.LAW, 'Kanun 4250 md. 6/f.8/c.1'), BREWER, REGISTRY, STORE)
        self.assertEqual(next(d for d in decisions if d.target_id == 'ALC-INT-LAGER-EXPORT').status, 'DOES_NOT_APPLY')
        strength = self.gap('Yönetmelik 23282 md. 9/f.1/b.i', 'ALC-INT-LAGER', 'TR:YONETMELIK:TGK_ETIKETLEME')
        self.assertEqual(strength.mapping.status, 'COVERED')

    def test_a_statement_that_carries_half_of_a_duty_is_partial(self):
        packaging = self.gap('Yönetmelik 38745 md. 10/f.1/b.ç', 'ALC-INT-BREWING', 'TR:YONETMELIK:AMBALAJ_ATIKLARI')
        self.assertEqual((packaging.document_coverage, packaging.coverage_reasons), ('PARTIAL', ['WORDING_PARTLY_MATCHED']))
        self.assertEqual(packaging.mapping.departments, ['ENVIRONMENT', 'FINANCE'])


class BottlerGapTests(unittest.TestCase):
    ENERGY = 'TR:TEBLIG:TGK_ENERJI_ICECEKLERI'

    def gap(self, ref, target, regulation_id=None, group=None):
        return row(BOTTLER, BOTTLER_REGISTER, regulation_id or self.ENERGY, ref, target, group)

    def test_a_specification_with_a_weaker_caffeine_limit_is_contradicted(self):
        limit = self.gap('Tebliğ 23706 md. 5/f.1/b.b', 'NONALC-ENERGY')
        self.assertEqual((limit.mapping.status, limit.coverage_reasons), ('CONTRADICTED', ['LIMIT_WEAKER']))
        self.assertEqual([(f.product_id, f.stated, f.limit, f.result) for f in limit.product_findings],
                         [('NONALC-ENERGY', 150.0, 150.0, 'WITHIN_LIMIT')])
        self.assertEqual(limit.actions[0], {'action': 'UPDATE_DOCUMENT', 'target': 'SPEC-ENERGY-01#1', 'department': 'RND',
                                            'why': ['LIMIT_WEAKER']})

    def test_a_stated_product_value_above_the_limit_is_reported_whatever_the_policy_says(self):
        planned = self.gap('Tebliğ 23706 md. 5/f.1/b.b', 'NONALC-ENERGY-MAX')
        self.assertEqual([(f.attribute, f.stated, f.limit, f.result) for f in planned.product_findings],
                         [('caffeine_mg_per_l', 320.0, 150.0, 'EXCEEDS_LIMIT')])
        self.assertIn('PRODUCT_NONCONFORMITY', [a['action'] for a in planned.actions])
        self.assertEqual(planned.mapping.status, 'NOT_COVERED')            # the bottler's specification is not the importer's

    def test_partial_names_the_limits_and_places_the_statement_leaves_out(self):
        other = self.gap('Tebliğ 23706 md. 5/f.1/b.c', 'NONALC-ENERGY')
        self.assertEqual(other.coverage_reasons, ['LIMIT_MISSING:glucuronolactone_mg_per_l', 'LIMIT_MISSING:inositol_mg_per_l'])
        venues = self.gap('Tebliğ 23706 md. 11/f.2', 'NONALC-ACT-ENERGY-ONLINE-ADS', group='MARKETING')
        self.assertEqual((venues.mapping.status, venues.coverage_reasons), ('PARTIALLY_COVERED', ['PLACE_MISSING:HEALTH_FACILITY']))

    def test_covered_rows_of_the_soft_drink_lines(self):
        warning = self.gap('Tebliğ 23706 md. 12/f.1/b.c', 'NONALC-ENERGY')
        self.assertEqual(warning.mapping.status, 'COVERED')
        cola = self.gap('Tebliğ 11376 md. 5/f.1/b.c', 'NONALC-COLA', 'TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER')
        self.assertEqual((cola.mapping.status, cola.product_findings[0].result), ('COVERED', 'WITHIN_LIMIT'))
        waste = self.gap('Yönetmelik 38745 md. 10/f.1/b.ç', 'NONALC-BOTTLING', 'TR:YONETMELIK:AMBALAJ_ATIKLARI')
        self.assertEqual((waste.mapping.status, waste.mapping.control_ids), ('COVERED', ['CTL-GEKAP-DECLARATION']))

    def test_a_specification_covers_only_the_products_it_names(self):
        cola = self.gap('Yönetmelik 23282 md. 9/f.1/b.a', 'NONALC-COLA', 'TR:YONETMELIK:TGK_ETIKETLEME')
        water = self.gap('Yönetmelik 23282 md. 9/f.1/b.a', 'NONALC-WATER', 'TR:YONETMELIK:TGK_ETIKETLEME')
        self.assertEqual((cola.mapping.status, water.mapping.status), ('COVERED', 'NOT_COVERED'))
        self.assertNotIn('SPEC-LABEL-02', water.documents_in_force)

    def test_the_profile_report_counts_every_status_and_who_has_to_act(self):
        selected = [o for rid, articles in ((self.ENERGY, ['5', '11', '12']), ('TR:TEBLIG:TGK_ALKOLSUZ_ICECEKLER', ['5', '11']))
                    for o in extract_regulation(rid, REGISTRY, STORE, articles=articles)[1]]
        report = compare_profile(BOTTLER, selected, BOTTLER_REGISTER, REGISTRY, STORE)
        self.assertEqual(sum(report.summary['by_status'].values()), report.summary['rows'])
        self.assertTrue({'COVERED', 'PARTIALLY_COVERED', 'NOT_COVERED', 'CONTRADICTED'} <= set(report.summary['by_status']))
        self.assertGreaterEqual(report.summary['product_nonconformities'], 2)       # caffeine and taurine of the planned import
        self.assertTrue(report.register_synthetic)
        for item in report.rows:
            self.assertIn(item.applicability, ('APPLIES', 'PARTIAL'))
            self.assertEqual(item.mapping.status, mapping_status(item.document_coverage, bool(item.mapping.control_ids),
                                                                 any(e.in_period for e in BOTTLER_REGISTER.evidence
                                                                     if e.evidence_id in item.mapping.evidence_ids))[0])
            for seen in item.readings:
                self.assertEqual(next(p.text for p in BOTTLER_REGISTER.passages if p.passage_id == seen.passage_id), seen.quote)


class WholeCorpusTests(unittest.TestCase):
    """What the comparer said outside the labelled cases when it was run on every stored text (30 September 2026): 44
    contradicted rows and 4 product nonconformities for the brewer; 17 of the rows (advertising by-law, consumer law) and all 4
    findings were wrong, and two statements were cited as conflicting that are not (warning sign, warning messages)."""

    @classmethod
    def setUpClass(cls):
        cls.rows = {}
        for regulation_id in ('TR:KANUN:6502', 'TR:YONETMELIK:TICARI_REKLAM', 'TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM',
                              'TR:YONETMELIK:ALKOL_TESIS_TEKNIK_SARTLAR'):
            duties = extract_regulation(regulation_id, REGISTRY, STORE)[1]
            for item in compare_profile(BREWER, duties, BREWER_REGISTER, REGISTRY, STORE).rows:
                cls.rows.setdefault(item.provision_ref, []).append(item)

    def test_a_statement_that_sends_advertisements_to_legal_review_permits_no_prohibited_advertisement(self):
        # "Her türlü iletişim aracında ... örtülü reklam yapılması yasaktır." against "Alkolsüz bira için hazırlanan reklam
        # ve tanıtım çalışmaları, yayına alınmadan önce Hukuk Müşavirliğinin onayına sunulur."
        for ref in ('Kanun 6502 md. 61/f.4/c.2', 'Yönetmelik 20435 md. 22/f.1', 'Yönetmelik 20435 md. 27/f.3', 'Yönetmelik 20435 md. 10/f.1',
                    'Yönetmelik 20435 md. 5/f.1/b.ı'):
            self.assertTrue(self.rows[ref], ref)
            self.assertEqual({r.mapping.status for r in self.rows[ref]} & {'CONTRADICTED'}, set(), ref)

    def test_the_marks_of_a_brand_are_not_a_labelling_duty(self):
        # "... marka, amblem ya da işaretlerini kullanarak destek olamazlar" was matched with "uyarı mesajları konulur"
        # through the word "işaretlerini" read as marking.
        sponsoring = self.rows['Yönetmelik 14646 md. 20/f.7']
        self.assertEqual({r.mapping.status for r in sponsoring}, {'CONTRADICTED'})                  # by the sponsorship statement
        self.assertEqual({r.passage_id for item in sponsoring for r in item.readings if r.relation == 'CONFLICTS'}, {'POL-MKT-01#2'})

    def test_a_unit_of_account_is_not_a_limit_a_product_exceeds(self):
        # "Üründeki %100'lük alkol cinsinden 100 litre alkol için en az 25 kg meyve ..." (a fruit liqueur plant project)
        self.assertEqual([f for rows in self.rows.values() for r in rows for f in r.product_findings if f.result == 'EXCEEDS_LIMIT'], [])

    def test_the_duplicates_of_the_statute_in_its_by_law_are_found_with_it(self):
        self.assertEqual({r.mapping.status for r in self.rows['Yönetmelik 14646 md. 20/f.4']}, {'CONTRADICTED'})   # free distribution
        self.assertEqual({r.coverage_reasons[0] for r in self.rows['Yönetmelik 14646 md. 8/f.1/b.d']}, {'TIME_WINDOW_VIOLATED'})


class SecondReadingTests(unittest.TestCase):
    """The rule comparer and the engine's judge on the same duty and documents (combine_coverage)."""

    def test_an_element_check_of_the_rule_comparer_is_not_overruled(self):
        self.assertEqual(combine_coverage('CONFLICT', ['LIMIT_WEAKER'], 'COVERS_TEXT'), ('CONFLICT', 'RULE_ELEMENT_CHECK', True))
        self.assertEqual(combine_coverage('CONFLICT', ['TIME_WINDOW_VIOLATED'], 'CONFLICT'), ('CONFLICT', 'BOTH_READINGS', True))
        # "Okul kantinlerinde ve spor tesislerinde ..." against a duty that also names hospitals: the judge called it covered.
        self.assertEqual(combine_coverage('PARTIAL', ['PLACE_MISSING:HEALTH_FACILITY'], 'COVERS_TEXT'), ('PARTIAL', 'RULE_ELEMENT_CHECK', False))

    def test_a_conflict_only_the_model_sees_waits_for_a_person(self):
        # The rule comparer's word stays, the row goes to review with the passage the model quoted.
        self.assertEqual(combine_coverage('PARTIAL', ['PLACE_MISSING:HEALTH_FACILITY'], 'CONFLICT'), ('PARTIAL', 'MODEL_CONFLICT_UNCONFIRMED', True))
        self.assertEqual(combine_coverage('NO_EVIDENCE', ['NO_RELATED_STATEMENT'], 'CONFLICT'), ('NO_EVIDENCE', 'MODEL_CONFLICT_UNCONFIRMED', True))
        args = (BOTTLER, BOTTLER_REGISTER, 'TR:TEBLIG:TGK_ENERJI_ICECEKLERI', 'Tebliğ 23706 md. 11/f.2', 'NONALC-BOTTLING', 'SUPPLY')
        quote = '3. Üniversite kampüslerindeki etkinliklerde enerji içeceği numune dağıtımı yapılabilir.'
        joined = row(*args, second={'coverage': 'CONFLICT', 'quotes': {'CONFLICTS': [quote]}})
        self.assertEqual((joined.document_coverage, joined.model_coverage, joined.review_required), ('PARTIAL', 'CONFLICT', True))
        self.assertIn(('POL-MKT-02#3', 'CONFLICTS', ['MODEL_READING']), [(r.passage_id, r.relation, r.reasons) for r in joined.readings])

    def test_paraphrase_is_the_models_to_read_and_a_persons_to_confirm(self):
        self.assertEqual(combine_coverage('NO_EVIDENCE', ['NO_RELATED_STATEMENT'], 'COVERS_TEXT'), ('COVERS_TEXT', 'MODEL_PARAPHRASE', True))
        self.assertEqual(combine_coverage('PARTIAL', ['WORDING_PARTLY_MATCHED'], 'COVERS_TEXT'), ('COVERS_TEXT', 'MODEL_PARAPHRASE', True))
        # A PARTIAL only the model sees changes no coverage (its judge calls passages partial that share only the product).
        self.assertEqual(combine_coverage('COVERS_TEXT', ['SUPPORTING_STATEMENT'], 'PARTIAL'), ('COVERS_TEXT', 'MODEL_SEES_GAP', True))
        self.assertEqual(combine_coverage('NO_EVIDENCE', ['NO_RELATED_STATEMENT'], 'PARTIAL'), ('NO_EVIDENCE', 'MODEL_PARTIAL_UNCONFIRMED', True))
        self.assertEqual(combine_coverage('PARTIAL', ['WORDING_PARTLY_MATCHED'], 'PARTIAL'), ('PARTIAL', 'BOTH_READINGS', False))
        self.assertEqual(combine_coverage('COVERS_TEXT', ['SUPPORTING_STATEMENT'], 'NO_EVIDENCE'), ('COVERS_TEXT', 'RULE_WORDING_MATCH', False))
        self.assertEqual(combine_coverage('NO_EVIDENCE', ['NO_RELATED_STATEMENT'], 'NO_EVIDENCE'), ('NO_EVIDENCE', 'BOTH_READINGS', False))
        self.assertEqual(combine_coverage('NO_EVIDENCE', ['NO_RELATED_STATEMENT'], None), ('NO_EVIDENCE', 'RULE_ONLY', False))

    def test_a_statement_the_model_found_is_tied_to_its_document_and_control(self):
        # "4. Satış belgesi, satış noktasının içinde tüketicilerin görebileceği bir yere asılır." says the duty of
        # Yönetmelik 14646 md. 6/1-e in other words; the rule comparer finds no related statement.
        args = (BREWER, BREWER_REGISTER, 'TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', 'Yönetmelik 14646 md. 6/f.1/b.e', 'ALC-INT-SALES')
        alone = row(*args)
        self.assertEqual((alone.document_coverage, alone.coverage_basis, alone.mapping.status), ('NO_EVIDENCE', 'RULE_ONLY', 'NOT_COVERED'))
        quote = '4. Satış belgesi, satış noktasının içinde tüketicilerin görebileceği bir yere asılır.'
        joined = row(*args, second={'coverage': 'COVERS_TEXT', 'quotes': {'SUPPORTS': [quote]}})
        self.assertEqual((joined.document_coverage, joined.rule_coverage, joined.model_coverage, joined.coverage_basis),
                         ('COVERS_TEXT', 'NO_EVIDENCE', 'COVERS_TEXT', 'MODEL_PARAPHRASE'))
        self.assertTrue(joined.review_required)
        self.assertEqual(joined.mapping.document_ids, ['SOP-SALES-02'])
        self.assertEqual([(r.passage_id, r.reasons) for r in joined.readings if r.relation == 'SUPPORTS'], [('SOP-SALES-02#4', ['MODEL_READING'])])
        self.assertNotEqual(joined.mapping.status, 'NOT_COVERED')

    def test_a_profile_report_joins_each_row_with_the_reading_made_for_its_own_entity(self):
        selected = extract_regulation('TR:YONETMELIK:ALKOLLU_ICKI_SATIS_SUNUM', REGISTRY, STORE, articles=['6'])[1]
        quote = '4. Satış belgesi, satış noktasının içinde tüketicilerin görebileceği bir yere asılır.'
        second = {'ALC-INT-SALES': {'Yönetmelik 14646 md. 6/f.1/b.e': {'coverage': 'COVERS_TEXT', 'quotes': {'SUPPORTS': [quote]}}}}
        report = compare_profile(BREWER, selected, BREWER_REGISTER, REGISTRY, STORE, second)
        licence = [r for r in report.rows if r.provision_ref == 'Yönetmelik 14646 md. 6/f.1/b.e']
        # the wholesaler's row of the same duty has no reading of its own and stays with the rule comparer
        self.assertEqual({(r.entity_id, r.coverage_basis) for r in licence},
                         {('ALC-INT-SALES', 'MODEL_PARAPHRASE'), ('ALC-INT-DISTRIBUTION', 'RULE_ONLY')})
        self.assertTrue(all(r.coverage_basis == 'RULE_ONLY' for r in report.rows if r.provision_ref != 'Yönetmelik 14646 md. 6/f.1/b.e'))


if __name__ == '__main__':
    unittest.main()

"""New production contracts; synthetic unit inputs are not expert gold."""
import unittest

from pydantic import ValidationError

from regchain.evidence import canonical_bytes
from regchain.pilot.financial import (FinancialChangeProposal, FinancialImpact, analyze_financial_change,
                                     normalize_financial_impact, project_change_types_to_v018)
from regchain.pilot.impact import change_kinds, impact_summary


def impact(**updates):
    return dict(financial_impact_type=None, operational_impact_type=None, affected_actor=None,
                specific_monetary_impact=None, impact_confidence=0.0, impact_basis='INFERENCE', **updates)


class StructuredFinancialTests(unittest.TestCase):
    def test_postfix_currency_symbols_keep_values_units_comparators_and_source_spans(self):
        for old, new, expected, unit in [
            ('Limit 12.345,50 ₺ veya üzeri.', 'Limit 18.765,25 ₺ veya üzeri.', (12345.5, 18765.25), 'TL'),
            ('Limit at least 1,250.25 €.', 'Limit at least 2,750.75 €.', (1250.25, 2750.75), 'EUR'),
            ('Limit 2 million £ or more.', 'Limit 3 million £ or more.', (2000000, 3000000), 'GBP'),
            ('Limit 700 US$ or more.', 'Limit 900 US$ or more.', (700, 900), 'USD')]:
            with self.subTest(unit=unit):
                result = analyze_financial_change(old, new)
                self.assertEqual(result.change_types, ['THRESHOLD_CHANGED'])
                self.assertEqual((result.dimensions[0].value_old, result.dimensions[0].value_new), expected)
                self.assertEqual(result.dimensions[0].unit, unit)
                for text, facts in ((old, result.quantities_old), (new, result.quantities_new)):
                    self.assertEqual(len(facts), 1)
                    self.assertEqual(facts[0].comparator, '>=')
                    self.assertEqual(text[facts[0].start:facts[0].end], facts[0].text)

    def test_symbol_adapter_preserves_other_fact_offsets_and_does_not_duplicate_prefix_currency(self):
        text = 'Banks report above $850 and 900 € within 12 days.'
        result = analyze_financial_change(text, text)
        self.assertEqual(len(result.quantities_old), 3)
        self.assertEqual([x.value for x in result.quantities_old], [850, 900, 12])
        for fact in result.quantities_old:
            self.assertEqual(text[fact.start:fact.end], fact.text)

    def test_identical_numbers_have_distinct_temporal_meaning(self):
        for old, new, expected in [
            ('A firm retains records for 30 days.', 'A firm retains records for 60 days.', 'RETENTION_PERIOD_CHANGED'),
            ('A firm submits reports within 30 days.', 'A firm submits reports within 60 days.', 'DEADLINE_CHANGED'),
            ('Kayıtlar otuz gün saklanır.', 'Kayıtlar altmış gün saklanır.', 'RETENTION_PERIOD_CHANGED'),
            ('Bildirim otuz gün içinde yapılır.', 'Bildirim altmış gün içinde yapılır.', 'DEADLINE_CHANGED')]:
            with self.subTest(expected=expected, old=old):
                result = analyze_financial_change(old, new)
                self.assertEqual(result.change_types, [expected])
                self.assertEqual((result.dimensions[0].value_old, result.dimensions[0].value_new), (30, 60))
                self.assertIsNone(result.dimensions[0].numeric_dimension)
                for text, facts in ((old, result.quantities_old), (new, result.quantities_new)):
                    for fact in facts:
                        self.assertEqual(text[fact.start:fact.end], fact.text)

    def test_unrelated_retention_sentence_does_not_change_reporting_deadline_role(self):
        result = analyze_financial_change('Banks submit reports within 15 days. Auditors retain the reports.',
                                         'Banks submit reports within 10 days. Auditors retain the reports.')
        self.assertEqual(result.change_types, ['DEADLINE_CHANGED'])

    def test_deadline_to_store_and_retention_duration_differ(self):
        result = analyze_financial_change('Firms must store records within 5 days.', 'Firms must store records within 7 days.')
        self.assertEqual(result.change_types, ['DEADLINE_CHANGED'])

    def test_money_ratio_and_mixed_changes_are_typed_before_classification(self):
        result = analyze_financial_change('Banks report above 10000 TL within 30 days and keep reserves of 4%.',
                                         'Banks report above 20000 TL within 60 days and keep reserves of 5%.')
        self.assertEqual(result.change_types, ['THRESHOLD_CHANGED', 'DEADLINE_CHANGED'])
        self.assertEqual({x.numeric_dimension for x in result.dimensions if x.numeric_dimension}, {'MONETARY_AMOUNT', 'RATIO'})

    def test_business_days_are_not_converted_into_approximate_calendar_days(self):
        result = analyze_financial_change('Banks respond within 5 days.', 'Banks respond within 5 business days.')
        self.assertEqual(result.change_types, ['DEADLINE_CHANGED'])
        self.assertEqual((result.dimensions[0].unit_old, result.dimensions[0].unit_new), ('gün', 'iş günü'))
        self.assertIsNone(result.dimensions[0].unit)

    def test_number_word_representation_is_not_a_numeric_change(self):
        result = analyze_financial_change('Banks respond within thirty days.', 'Banks respond within 30 days.')
        self.assertEqual(result.change_types, ['TEXT_CHANGED'])
        self.assertEqual(result.dimensions, [])

    def test_unqualified_duration_is_not_guessed_as_retention(self):
        result = analyze_financial_change('The period is 30 days.', 'The period is 60 days.')
        self.assertEqual(result.change_types, ['TEXT_CHANGED'])
        self.assertTrue(any('DURATION_ROLE_UNRESOLVED' in x for x in result.limitations))

    def test_multi_quantity_pairs_disclose_alignment_limit(self):
        result = analyze_financial_change('Banks report above 1000 TL; firms report above 2000 TL.',
                                         'Banks report above 3000 TL; firms report above 4000 TL.')
        self.assertTrue(all(x.pairing == 'ORDINAL_WITHIN_DIMENSION_REVIEW_REQUIRED' for x in result.dimensions))

    def test_legacy_comparison_and_new_taxonomy_are_explicitly_distinct(self):
        old, new = 'Firms retain records for five years.', 'Firms retain records for seven years.'
        result = analyze_financial_change(old, new)
        self.assertEqual(change_kinds(old, new), ['DEADLINE_CHANGED'])
        self.assertEqual(result.change_types, ['RETENTION_PERIOD_CHANGED'])
        self.assertEqual(result.legacy_change_types, ['DEADLINE_CHANGED'])
        self.assertEqual(project_change_types_to_v018(result.change_types), ['DEADLINE_CHANGED'])
        with self.assertRaises(ValueError):
            project_change_types_to_v018(['invented'])

    def test_scope_and_exemption_are_separate_from_quantities(self):
        result = analyze_financial_change('Banks report above 1000 TL.', 'Banks and payment institutions report above 1000 TL, except exempt transactions.')
        self.assertEqual(result.change_types, ['ENTITY_SCOPE_CHANGED', 'EXEMPTION_CHANGED'])

    def test_no_change_after_spacing_preserves_literal_inputs(self):
        result = analyze_financial_change('Banks retain records.', ' Banks  retain records. ')
        self.assertEqual(result.change_types, [])
        self.assertEqual(result.new_requirement, ' Banks  retain records. ')

    def test_packet_integration_keeps_legacy_and_adds_new_fields_without_float_evidence(self):
        case = {'source': {'printed_label': 'A1', 'text': 'Banks retain records for 8 years.'},
                'change': {'status': 'TEXT_CHANGED_REVIEW_REQUIRED', 'old': {'text': 'Banks retain records for 5 years.'}}}
        result = impact_summary(None, None, [case], [], 0, 1, [], ['A1'])
        changed = result['changed'][0]
        self.assertEqual(changed['kinds'], ['DEADLINE_CHANGED'])
        self.assertEqual(changed['structured_change']['change_types'], ['RETENTION_PERIOD_CHANGED'])
        self.assertIsNone(changed['financial_impact']['impact']['specific_monetary_impact'])
        canonical_bytes(result)


class ImpactContractTests(unittest.TestCase):
    def test_strict_contract_rejects_literal_null_free_category_and_object_actor(self):
        for patch in ({'financial_impact_type': 'null'}, {'financial_impact_type': 'some cost'},
                      {'affected_actor': {'name': 'transactions', 'entity_type': 'ORGANIZATION', 'source_quote': 'transactions'}},
                      {'affected_actor': {'name': 'bank transactions', 'entity_type': 'ORGANIZATION', 'source_quote': 'bank transactions'}},
                      {'specific_monetary_impact': '12'}, {'specific_monetary_impact': True},
                      {'impact_confidence': 1.1}, {'impact_confidence': float('nan')}):
            with self.subTest(patch=patch), self.assertRaises(ValidationError):
                FinancialImpact.model_validate({**impact(), **patch})

    def test_numeric_company_impact_requires_explicit_matching_fact_not_just_profile(self):
        value = {**impact(), 'specific_monetary_impact': 1250, 'impact_basis': 'FACT'}
        for company in (None, {}, {'name': 'Bank A'}, {'specific_monetary_impact': '1250'}, {'specific_monetary_impact': 500}):
            with self.subTest(company=company), self.assertRaises(ValidationError):
                FinancialImpact.model_validate(value, context={'company_data': company})
        self.assertEqual(FinancialImpact.model_validate(value, context={'company_data': {'specific_monetary_impact': 1250}}).specific_monetary_impact, 1250)

    def test_entity_requires_exact_supplied_source(self):
        value = {**impact(), 'affected_actor': {'name': 'Banks', 'entity_type': 'ORGANIZATION', 'source_quote': 'Banks'}}
        with self.assertRaises(ValidationError):
            FinancialImpact.model_validate(value, context={'old_text': 'Firms report.', 'new_text': 'Firms report promptly.'})
        self.assertEqual(FinancialImpact.model_validate(value, context={'new_text': 'Banks report.'}).affected_actor.name, 'Banks')

    def test_adapter_normalizes_sentinels_preserves_unknown_free_categories_and_rejects_estimate(self):
        result = normalize_financial_impact({'affected_actor': 'transactions', 'financial_impact_category': 'unexpected category',
            'operational_impact_category': 'record retention', 'specific_monetary_impact': 100, 'financial_impact_type': 'null'},
            old_text='Firms retain records.', new_text='Firms retain records longer.')
        value = result['impact']
        self.assertIsNone(value['financial_impact_type'])
        self.assertEqual(value['financial_impact_category'], 'unexpected category')
        self.assertEqual(value['operational_impact_type'], 'RECORD_RETENTION')
        self.assertIsNone(value['affected_actor'])
        self.assertIsNone(value['specific_monetary_impact'])
        self.assertEqual(len(result['normalizations']), 4)

    def test_model_proposal_must_preserve_source_text_and_have_no_duplicate_type(self):
        value = {**impact(), 'old_requirement': 'Banks report.', 'new_requirement': 'Banks report monthly.', 'change_types': ['TEXT_CHANGED']}
        context = {'old_text': value['old_requirement'], 'new_text': value['new_requirement']}
        FinancialChangeProposal.model_validate(value, context=context)
        for patch in ({'old_requirement': 'banks report.'}, {'change_types': ['TEXT_CHANGED', 'TEXT_CHANGED']}):
            with self.assertRaises(ValidationError):
                FinancialChangeProposal.model_validate({**value, **patch}, context=context)


if __name__ == '__main__':
    unittest.main()

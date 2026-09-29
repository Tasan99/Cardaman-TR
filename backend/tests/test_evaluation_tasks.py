"""Per-task evaluation primitives: one-vs-rest rates, sample sufficiency, unsupported conclusions, contrast pairs."""
import unittest

from regchain.evaluation.tasks import (EXPERT_GOLD, DEVELOPMENT, one_vs_rest, pair_consistency, task_report,
                                       unsupported_reasons)

LABELS = ('APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN')


class OneVsRestTests(unittest.TestCase):
    def test_every_class_gets_precision_recall_f1_and_both_error_rates(self):
        rows = [('APPLIES', 'APPLIES'), ('APPLIES', 'DOES_NOT_APPLY'), ('DOES_NOT_APPLY', 'DOES_NOT_APPLY'),
                ('DOES_NOT_APPLY', 'APPLIES'), ('UNKNOWN', 'UNKNOWN'), ('DOES_NOT_APPLY', 'DOES_NOT_APPLY')]
        applies = one_vs_rest(rows, LABELS)['APPLIES']
        self.assertEqual((applies['tp'], applies['fp'], applies['fn'], applies['tn']), (1, 1, 1, 3))
        self.assertEqual((applies['precision'], applies['recall'], applies['f1']), (0.5, 0.5, 0.5))
        self.assertEqual(applies['false_positive_rate'], 0.25)
        self.assertEqual(applies['false_negative_rate'], 0.5)
        self.assertEqual(applies['support'], 2)

    def test_a_class_nobody_expected_or_predicted_has_undefined_rates_not_zero(self):
        unknown = one_vs_rest([('APPLIES', 'APPLIES')], LABELS)['UNKNOWN']
        self.assertIsNone(unknown['precision'])
        self.assertIsNone(unknown['recall'])
        self.assertIsNone(unknown['false_negative_rate'])
        self.assertEqual(unknown['false_positive_rate'], 0.0)


class SampleTests(unittest.TestCase):
    def rows(self, n):
        return [('APPLIES', 'APPLIES')] * n

    def test_below_the_target_the_task_is_insufficient_and_only_indicative(self):
        report = task_report('APPLICABILITY', self.rows(40), LABELS, target=100, label_status=EXPERT_GOLD)
        self.assertEqual(report['sample_status'], 'INSUFFICIENT_SAMPLE')
        self.assertEqual(report['claim'], 'INDICATIVE')
        self.assertEqual((report['n'], report['target']), (40, 100))
        self.assertEqual(report['per_class']['APPLIES']['precision'], 1.0)

    def test_accuracy_is_claimed_only_for_sufficient_expert_gold(self):
        self.assertEqual(task_report('APPLICABILITY', self.rows(100), LABELS, target=100, label_status=EXPERT_GOLD)['claim'],
                         'ACCURACY')
        development = task_report('APPLICABILITY', self.rows(500), LABELS, target=100, label_status=DEVELOPMENT)
        self.assertEqual((development['sample_status'], development['claim']), ('SUFFICIENT', 'INDICATIVE'))

    def test_a_positive_target_gates_on_the_positive_class_as_well(self):
        rows = [(True, True)] * 50 + [(False, False)] * 400
        report = task_report('CONTRADICTION', rows, (True, False), target=100, positive=True, positive_target=300,
                             label_status=EXPERT_GOLD)
        self.assertEqual(report['sample_status'], 'INSUFFICIENT_SAMPLE')
        self.assertEqual(report['positives'], 50)
        self.assertEqual(report['claim'], 'INDICATIVE')

    def test_the_target_cannot_be_below_the_floor(self):
        with self.assertRaises(ValueError):
            task_report('APPLICABILITY', self.rows(5), LABELS, target=10, label_status=EXPERT_GOLD)


class UnsupportedTests(unittest.TestCase):
    SOURCE = {'TR:KANUN:1@v1': 'Madde 5 – (1) Satış yerlerinde uyarı yazısı bulundurulur.'}

    def decision(self, **fields):
        base = {'status': 'APPLIES', 'stage': 'GROUNDED',
                'source': {'regulation_id': 'TR:KANUN:1', 'version_id': 'v1', 'provision_ref': 'md.5/f.1',
                           'quote': 'uyarı yazısı bulundurulur'},
                'facts': [{'path': 'legal_entities[0].activity_classes', 'value': 'YES'}]}
        return base | fields

    def test_a_grounded_decision_with_a_verbatim_quote_and_known_facts_is_supported(self):
        self.assertEqual(unsupported_reasons(self.decision(), self.SOURCE), [])

    def test_a_conclusion_without_provision_or_verbatim_quote_or_known_fact_is_unsupported(self):
        no_provision = self.decision(source={'regulation_id': 'TR:KANUN:1', 'version_id': 'v1', 'provision_ref': None,
                                             'quote': 'uyarı yazısı bulundurulur'})
        self.assertIn('NO_PROVISION', unsupported_reasons(no_provision, self.SOURCE))
        paraphrase = self.decision(source={'regulation_id': 'TR:KANUN:1', 'version_id': 'v1', 'provision_ref': 'md.5/f.1',
                                           'quote': 'uyarı yazısı asılır'})
        self.assertIn('QUOTE_NOT_IN_SOURCE', unsupported_reasons(paraphrase, self.SOURCE))
        unknown_fact = self.decision(facts=[{'path': 'products[1].attributes.abv_percent', 'value': 'UNKNOWN'}])
        self.assertIn('FACT_UNKNOWN', unsupported_reasons(unknown_fact, self.SOURCE))
        missing_version = self.decision(source={'regulation_id': 'TR:KANUN:1', 'version_id': 'v9', 'provision_ref': 'md.5/f.1',
                                                'quote': 'uyarı yazısı bulundurulur'})
        self.assertIn('SOURCE_NOT_STORED', unsupported_reasons(missing_version, self.SOURCE))

    def test_an_unknown_answer_is_never_unsupported_and_a_routed_conclusion_is_not_grounded(self):
        self.assertEqual(unsupported_reasons(self.decision(status='UNKNOWN', facts=[{'path': 'x', 'value': 'UNKNOWN'}]),
                                             self.SOURCE), [])
        routed = self.decision(stage='ROUTED', source={'regulation_id': 'TR:KANUN:1', 'version_id': None, 'provision_ref': None,
                                                       'quote': None})
        self.assertEqual(unsupported_reasons(routed, self.SOURCE), ['NOT_GROUNDED'])

    def test_the_report_counts_decision_level_and_overconfident_answers(self):
        rows = [('UNKNOWN', 'DOES_NOT_APPLY'), ('UNKNOWN', 'UNKNOWN'), ('APPLIES', 'APPLIES')]
        report = task_report('APPLICABILITY', rows, LABELS, target=100, label_status=DEVELOPMENT,
                             unsupported=[['QUOTE_NOT_IN_SOURCE'], [], []])
        self.assertEqual(report['overconfident'], 1)
        self.assertEqual(report['unsupported_conclusions'], 1)
        self.assertEqual(report['unsupported_by_reason'], {'QUOTE_NOT_IN_SOURCE': 1})
        self.assertEqual(report['abstention_rate'], round(1 / 3, 4))


class PairTests(unittest.TestCase):
    def test_a_pair_is_consistent_only_when_both_sides_match_gold(self):
        pairs = [(('APPLIES', 'APPLIES'), ('DOES_NOT_APPLY', 'DOES_NOT_APPLY')),
                 (('APPLIES', 'APPLIES'), ('DOES_NOT_APPLY', 'APPLIES')),
                 (('APPLIES', 'APPLIES'), ('APPLIES', 'APPLIES'))]
        result = pair_consistency(pairs)
        self.assertEqual((result['n'], result['consistent'], result['consistency']), (3, 2, 0.6667))
        self.assertEqual((result['contrast_pairs'], result['contrast_detected']), (2, 1))


if __name__ == '__main__':
    unittest.main()

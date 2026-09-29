import unittest
from regchain.pilot import conflict as c


def duty(action='disclose audit plans', modality='MUST_NOT', **extra):
    return {'subject': 'A firm', 'modality': modality,
            'required_action': action if modality == 'MUST' else None,
            'prohibited_action': action if modality == 'MUST_NOT' else None,
            'conditions': [], 'exceptions': [], **extra}


class ModalAndContextTests(unittest.TestCase):
    def test_negative_subject_modal_is_prohibition(self):
        for sentence in ('No employee may disclose audit plans.', 'No manager or contractor can disclose audit plans.'):
            self.assertEqual(c.polarities(sentence, {'@disclose'}), {'PROHIBITED'})
            self.assertEqual(c.conflict_gate(duty(), sentence, sentence)[0], c.GATE_EFFECT)

    def test_permission_and_negated_permission_have_different_effects(self):
        self.assertEqual(c.polarities('Agents may disclose audit plans.', {'@disclose'}), {'PERMITTED'})
        self.assertEqual(c.polarities('Agents are not permitted to disclose audit plans.', {'@disclose'}), {'PROHIBITED'})
        self.assertEqual(c.polarities('Agents are not required to disclose audit plans.', {'@disclose'}), {'OPTIONAL'})

    def test_negative_numeric_bound_is_not_a_negative_subject(self):
        self.assertEqual(c.polarities('No later than Friday staff may disclose audit plans.', {'@disclose'}), {'PERMITTED'})

    def test_coordinated_bare_verbs_share_negative_modal(self):
        self.assertEqual(c.polarities('No employee may report or disclose audit plans.', {'@disclose'}), {'PROHIBITED'})
        self.assertEqual(c.polarities('Staff are not permitted to report or disclose audit plans.', {'@disclose'}), {'PROHIBITED'})

    def test_unrelated_previous_sentence_cannot_anchor_quote(self):
        passage = 'Audit plans are retained. The lunch menu may be changed.'
        quote = 'The lunch menu may be changed.'
        self.assertFalse(c.anchored(duty(), quote, passage))
        self.assertEqual(c.evidence_context(quote, passage), quote)

    def test_explicit_reference_retains_dependent_evidence(self):
        passage = 'Audit plans are kept in a secure archive. These documents may be destroyed immediately.'
        quote = 'These documents may be destroyed immediately.'
        self.assertTrue(c.anchored(duty('retain audit plans', 'MUST'), quote, passage))
        self.assertIn('secure archive', c.evidence_context(quote, passage))

    def test_full_structure_object_anchors_separate_action_field(self):
        payload = duty('handle them', 'MUST', elements=[{'kind': 'object', 'text': 'audit plans'}])
        self.assertTrue(c.anchored(payload, 'Audit plans must be destroyed.'))
        self.assertFalse(c.anchored(payload, 'Audit plans must be destroyed.', 'Only training manuals are available.'))

    def test_passive_recipient_is_not_authority_inside_communicated_fact(self):
        text = "Müşteri, denetim raporunun MASAK'a iletildiği konusunda bilgilendirilir."
        self.assertEqual(c.communication_recipient(text), 'THIRD_PARTY')

    def test_qualified_unknown_verb_is_not_opposite_just_from_polarity(self):
        payload = duty('treat customers with respect and patience', 'MUST')
        code, _ = c.support_gate(payload, 'Employees must never treat customers as obstacles.')
        self.assertNotEqual(code, c.PG_POLARITY)

    def test_modal_only_element_uses_full_action_verb(self):
        payload = duty('disclose the audit plans', elements=[{'kind': 'prohibition', 'text': 'must not'}])
        self.assertEqual(c.verb_keys(payload), {'@disclose'})

    def test_short_english_action_uses_governing_source_language(self):
        payload = duty('alter invoices without authorization', elements=[{'kind': 'prohibition', 'text': 'must not'}])
        self.assertEqual(c.verb_words(payload), ['alter'])
        payload['prohibited_action'] = 'alter invoices'
        self.assertEqual(c.verb_words(payload), ['alter'])

    def test_exception_does_not_erase_explicit_object(self):
        payload = duty('circulate audit plans', exceptions=['unless audit plans are approved'],
                       elements=[{'kind': 'object', 'text': 'audit plans'}, {'kind': 'prohibition', 'text': 'must not'}])
        keys, _ = c.gate_acts(payload)
        self.assertIn('plans', c._object_keys(payload, keys))

    def test_entire_condition_duplicated_as_object_does_not_anchor_another_act(self):
        condition = 'when the remote warehouse closes'
        payload = duty('apply the segregation rule ' + condition, conditions=[condition],
                       elements=[{'kind': 'object', 'text': condition}, {'kind': 'prohibition', 'text': 'must not'}])
        keys, _ = c.gate_acts(payload)
        self.assertNotIn('wareh', c._object_keys(payload, keys))

    def test_unresolved_conditional_prohibition_is_reviewed_not_assumed_conflict(self):
        from regchain.pilot.engine import settle_v19
        from regchain.pilot.schema import ConflictVerdict
        payload = duty('cause a supplier to believe that invoices are payable where no contract exists',
                       conditions=['where no contract exists'],
                       elements=[{'kind': 'prohibition', 'id': 'prohibition', 'text': 'must not'}])
        text = 'No email may suggest that invoices are payable unless a signed contract requires payment.'
        claim = ConflictVerdict(conflict=True, contradiction_type='EXEMPTION_ADDED', contradiction_span=text,
                                regulation_requirement='conditional prohibition', policy_statement='conditional prohibition',
                                confidence='HIGH', relation_if_no_conflict='UNRELATED', support_quote='',
                                rationale='The alleged exception needs alignment.')
        notes = []
        result, fields = settle_v19(payload, None, claim, None, 'FAST_FAILED', notes, ['prohibition'], text)
        self.assertEqual(result.relation, 'UNCLEAR')
        self.assertEqual(fields['uncertainty'], 'strong')


if __name__ == '__main__':
    unittest.main()

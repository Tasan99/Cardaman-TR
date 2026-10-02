"""Sales-channel completeness (routing): the general profile flag says nothing about the channels.

A channel the profile names is known to be used, whether or not the list is complete. A channel it does not name is
known to be unused only when the channel list itself is stated complete (LegalEntity.sales_channels_complete,
Activity.channels_complete). Otherwise the channel gate is UNDETERMINED, and a duty that turns on it is UNKNOWN unless
another, independent and decided gate already rules it out. Contradictory channel facts are never a silent exclusion."""
import unittest

from regchain.tr.profile import load_pilot_profiles
from regchain.tr.routing import evaluate_scope
from test_tr_routing import REGISTRY, by_target, profile, scope

ONLINE = dict(sales_channels=['ONLINE'])
MODERN = dict(sales_channels=['MODERN_RETAIL'])


def channel_gate(decision):
    return next(g for g in decision.gates if g['gate'] == 'SALES_CHANNEL')


def entity(update):
    """The sample profile with its producer entity changed: complete profile, no channels unless the update says so."""
    return profile(lambda d: d['legal_entities'][0].update(update))


class EntityChannelTests(unittest.TestCase):
    def decide(self, prof, **constraints):
        return by_target(evaluate_scope(scope('LEGAL_ENTITY', **constraints), prof, REGISTRY))

    def test_a_missing_channel_completeness_field_is_not_a_complete_list(self):
        prof = entity({'sales_channels': [], 'profile_complete': True})            # the field is absent, as in every old profile
        self.assertFalse(prof.entity('PROD').sales_channels_complete)
        decision = self.decide(prof, **ONLINE)['PROD']
        self.assertEqual((decision.status, decision.reason_codes), ('UNKNOWN', ['PROFILE_INCOMPLETE']))
        self.assertEqual(channel_gate(decision)['status'], 'UNDETERMINED')

    def test_an_empty_list_stated_complete_rules_the_channel_out(self):
        prof = entity({'sales_channels': [], 'sales_channels_complete': True})
        decision = self.decide(prof, **ONLINE)['PROD']
        self.assertEqual((decision.status, decision.reason_codes), ('DOES_NOT_APPLY', ['SALES_CHANNEL_MISMATCH']))

    def test_a_partial_list_knows_what_it_names_and_nothing_else(self):
        prof = entity({'sales_channels': ['MODERN_RETAIL'], 'sales_channels_complete': False})
        self.assertEqual(self.decide(prof, **MODERN)['PROD'].status, 'APPLIES')
        self.assertEqual(self.decide(prof, **ONLINE)['PROD'].status, 'UNKNOWN')

    def test_a_complete_list_rules_out_only_the_channels_it_does_not_name(self):
        prof = entity({'sales_channels': ['MODERN_RETAIL'], 'sales_channels_complete': True})
        self.assertEqual(self.decide(prof, **MODERN)['PROD'].status, 'APPLIES')
        decision = self.decide(prof, **ONLINE)['PROD']
        self.assertEqual((decision.status, decision.reason_codes), ('DOES_NOT_APPLY', ['SALES_CHANNEL_MISMATCH']))
        # a duty without a channel condition is not touched by the channel list
        self.assertEqual(self.decide(prof, activity_classes=['PRODUCTION'])['PROD'].status, 'APPLIES')

    def test_a_named_channel_is_known_however_incomplete_the_profile_is(self):
        prof = entity({'sales_channels': ['ONLINE'], 'sales_channels_complete': False, 'profile_complete': False})
        decision = self.decide(prof, **ONLINE)['PROD']
        self.assertEqual((decision.status, channel_gate(decision)['status']), ('APPLIES', 'MATCH'))

    def test_an_independent_decided_gate_still_rules_the_duty_out(self):
        # channels open; the activity list is complete and lacks the activity the duty needs
        prof = entity({'sales_channels': [], 'profile_complete': True})
        decision = self.decide(prof, activity_classes=['ECOMMERCE_SALE'], **ONLINE)['PROD']
        self.assertEqual(decision.status, 'DOES_NOT_APPLY')
        self.assertEqual(decision.reason_codes, ['ACTIVITY_MISMATCH'])
        self.assertEqual(channel_gate(decision)['status'], 'UNDETERMINED')     # the channel itself stays undecided

    def test_a_complete_list_contradicted_by_the_profile_is_not_an_exclusion(self):
        # the entity sells online by its own activity class, yet its "complete" channel list does not name ONLINE
        prof = profile(lambda d: d['legal_entities'][1].update(
            {'activity_classes': ['WHOLESALE', 'RETAIL_SALE', 'ADVERTISING', 'ECOMMERCE_SALE'], 'sales_channels': ['MODERN_RETAIL'],
             'sales_channels_complete': True}))
        decision = self.decide(prof, **ONLINE)['SALES']
        self.assertEqual(decision.status, 'UNKNOWN')
        self.assertIn('SALES_CHANNEL_PROFILE_CONFLICT', decision.reason_codes)
        self.assertEqual(channel_gate(decision)['status'], 'UNDETERMINED')
        # an activity record that names a channel the entity's complete list leaves out: the named channel is known,
        # the list is no longer trusted for any other channel
        prof = profile(lambda d: (d['legal_entities'][1].update({'sales_channels': ['MODERN_RETAIL'], 'sales_channels_complete': True}),
                                  d['activities'][0].update({'channels': ['ONLINE']})))
        self.assertEqual(self.decide(prof, **ONLINE)['SALES'].status, 'APPLIES')
        other = self.decide(prof, sales_channels=['TRADITIONAL_RETAIL'])['SALES']
        self.assertEqual(other.status, 'UNKNOWN')
        self.assertIn('SALES_CHANNEL_PROFILE_CONFLICT', other.reason_codes)


class ActivityChannelTests(unittest.TestCase):
    def decide(self, prof, **constraints):
        return by_target(evaluate_scope(scope('ACTIVITY', activity_classes=['ADVERTISING'], **constraints), prof, REGISTRY))['A-ADV']

    def test_an_activity_channel_list_is_complete_only_when_it_says_so(self):
        named = profile(lambda d: d['activities'][0].update({'channels': ['ONLINE']}))
        self.assertEqual(self.decide(named, **ONLINE).status, 'APPLIES')
        self.assertEqual(self.decide(named, **MODERN).status, 'UNKNOWN')                      # was DOES_NOT_APPLY
        complete = profile(lambda d: d['activities'][0].update({'channels': ['ONLINE'], 'channels_complete': True}))
        self.assertEqual(self.decide(complete, sales_channels=['TRADITIONAL_RETAIL']).status, 'DOES_NOT_APPLY')

    def test_an_activity_cannot_use_a_channel_its_entity_states_it_does_not_use(self):
        prof = profile(lambda d: (d['legal_entities'][1].update({'sales_channels': ['MODERN_RETAIL', 'ONLINE'], 'sales_channels_complete': True}),
                                  d['activities'][0].update({'channels': ['ONLINE']})))
        self.assertEqual(self.decide(prof, sales_channels=['TRADITIONAL_RETAIL']).status, 'DOES_NOT_APPLY')
        self.assertEqual(self.decide(prof, **MODERN).status, 'UNKNOWN')       # the entity uses it; this activity may or may not


class ProfileDataTests(unittest.TestCase):
    def test_no_pilot_profile_claims_a_complete_channel_list_without_a_stated_basis(self):
        for prof in load_pilot_profiles(REGISTRY.vocabulary).values():
            for item in prof.legal_entities:
                if item.sales_channels_complete:
                    self.assertTrue(item.sales_channels_basis.strip(), f'{prof.profile_id} {item.entity_id}')
            for activity in prof.activities:
                if activity.channels_complete:
                    self.assertTrue(activity.channels_basis.strip(), f'{prof.profile_id} {activity.activity_id}')


if __name__ == '__main__':
    unittest.main()

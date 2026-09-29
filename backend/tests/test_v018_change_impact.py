"""v0.18 regression: change impact across regulation snapshots.

A new snapshot of the same regulation re-reads only what changed: a new provision and a changed
one cost model calls, an unchanged provision keeps its rows without a call, as long as the
company, the policies, the prompts, the models and the retrieval are those of the compared
analysis; otherwise everything is read again and the record names why. The kind of a change
(threshold, deadline, entity scope, exemption, wording) is read from Turkish and English text
by rule. Fixtures only: no model, no network.
"""
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes, verify_chain
from regchain.extraction.providers import AI_CONTEXT, AI_TASK
from regchain.pilot import engine
from regchain.pilot.engine import analyze
from regchain.pilot.impact import ACTIONS, change_kinds, impact_summary, remap_source_ids, reuse_blockers, suggested_action
from regchain.pilot.report import render
from test_phase15 import TurkishRules, anadolu, turkish_sections
from test_pilot import FixtureProvider, company, policies, sections
from test_semantic import ConceptEmbedder


def counting(base):
    """``base`` with every model call recorded as (task, provision id). Extraction runs by rule in
    these fixtures, but a model would be asked there, so it is recorded too, with its text."""
    class Counting(base):
        def __init__(self):
            self.calls = []

        def _chat(self, prompt, payload, schema):
            self.calls.append((AI_TASK.get(), (AI_CONTEXT.get() or {}).get('provision_id')))
            return super()._chat(prompt, payload, schema)

        def generate(self, text):
            self.calls.append(('extraction', text))
            return super().generate(text)
    return Counting


Counting = counting(FixtureProvider)
TurkishCounting = counting(TurkishRules)


def asked(provider, snapshot):
    """The provisions (printed labels) the provider was asked anything about."""
    by_id = {s['id']: s['printed_label'] for s in snapshot}
    labels = set()
    for task, subject in provider.calls:
        if task == 'extraction':
            labels |= {s['printed_label'] for s in snapshot if subject in s['text']}
        else:
            labels.add(by_id.get(subject, subject))
    return labels


def notify(text='A firm must notify the customer in writing.'):
    """A second duty in the fixture chapter, so one provision can change while the other does not."""
    return dict(sections()[1], id='s-notify', version_id='v-notify', paragraph_number='5', printed_label='CONC 7.3.5',
                text=text, content_hash='e' * 64, ordinal=1)


def kyc(text='(1) Yükümlüler, 75.000 TL ve üzerindeki işlemlerde müşterilerinin kimliğini işlem yapılmadan önce tespit etmek zorundadır.'):
    """A second Turkish duty article beside md. 4 (şüpheli işlem bildirimi)."""
    base = turkish_sections()[1]
    return dict(base, id='tr-kyc', paragraph_number='5', ordinal=4, printed_label='Kanun 5549 md. 5', content_hash='e' * 64,
                heading_path=[base['heading_path'][0], 'İKİNCİ BÖLÜM Yükümlülükler', 'Kimlik tespiti'], text=text)


def renewed(snapshot, changes=None):
    """The next download of the same regulation: every section and version id is new; ``changes``
    maps a printed label to its new wording."""
    changes = changes or {}
    return [dict(s, id='n-' + s['id'], version_id='n-' + s['version_id'], text=changes.get(s['printed_label'], s['text'])) for s in snapshot]


def compared(first):
    """What the workspace hands analyze() for a compared run: the old sources, head and payload."""
    old = first['events'][0]['payload']
    return [c['source'] for c in old['cases']], first['head'], old


def payload_of(packet):
    return packet['events'][0]['payload']


class ChangeKindTests(unittest.TestCase):
    """Each kind alone, in Turkish and in English wording; the exact list, so a pure threshold edit
    is not also reported as a deadline or as plain wording."""

    def assertKinds(self, pairs, expected):
        for old, new in pairs:
            with self.subTest(new=new):
                self.assertEqual(change_kinds(old, new), expected)

    def test_a_threshold_change_is_read_in_either_language_and_either_currency_position(self):
        self.assertKinds([
            ('İşlem tutarı 75.000 TL veya üzerinde ise kimlik tespiti yapılır.', 'İşlem tutarı 185.000 TL veya üzerinde ise kimlik tespiti yapılır.'),
            ('Tutar 20.000 ₺ üzerinde ise bildirim yapılır.', 'Tutar 50.000 ₺ üzerinde ise bildirim yapılır.'),
            ('Tutar 10.000 Türk Lirası üzerinde ise bildirim yapılır.', 'Tutar 15.000 Türk Lirası üzerinde ise bildirim yapılır.'),
            ('A firm must report transactions above £10,000.', 'A firm must report transactions above £15,000.'),
            ('A firm must report payments above $5,000.', 'A firm must report payments above $7,500.'),
            # The currency code before the amount, as EU texts write it; v0.17 called this plain wording.
            ('A firm must report transactions of EUR 10,000 or more.', 'A firm must report transactions of EUR 15,000 or more.'),
        ], ['THRESHOLD_CHANGED'])
        # The same amount before a full stop and before more words is the same threshold.
        self.assertEqual(change_kinds('A firm must report transactions of £10,000.', 'A firm must report transactions of £10,000 or more.'),
                         ['TEXT_CHANGED'])

    def test_a_deadline_change_is_read_from_digits_number_words_and_turkish_suffixed_units(self):
        self.assertKinds([
            ('Bildirim on gün içinde yapılır.', 'Bildirim yirmi gün içinde yapılır.'),
            ('Bildirim 10 iş günü içinde yapılır.', 'Bildirim 5 iş günü içinde yapılır.'),
            ('Kayıtlar beş yıl süreyle saklanır.', 'Kayıtlar sekiz yıl süreyle saklanır.'),
            # Suffixed units ("aylık", "yıldan", "ayda", "gününde") were not read in v0.17.
            ('Kayıtlar altı aylık süre içinde gönderilir.', 'Kayıtlar üç aylık süre içinde gönderilir.'),
            ('Belgeler beş yıldan az olmamak üzere saklanır.', 'Belgeler sekiz yıldan az olmamak üzere saklanır.'),
            ('Raporlar altı ayda bir gönderilir.', 'Raporlar üç ayda bir gönderilir.'),
            ('Bildirim on iş gününde yapılır.', 'Bildirim beş iş gününde yapılır.'),
            # A number in several words is one number: v0.17 read "on beş" and "yirmi beş" both as "beş".
            ('Belgeler on beş gün içinde verilir.', 'Belgeler yirmi beş gün içinde verilir.'),
            ('İtiraz yüz seksen gün içinde yapılır.', 'İtiraz yüz yetmiş gün içinde yapılır.'),
            ('A firm must respond within fourteen days.', 'A firm must respond within sixteen days.'),
            ('A firm must respond within 30 days.', 'A firm must respond within 60 days.'),
            ('A firm must respond within 8 weeks.', 'A firm must respond within 15 business days.'),
            # The Handbook writes periods in words; v0.17 knew only the Turkish number words.
            ('A firm must retain records for five years.', 'A firm must retain records for seven years.'),
            ('A firm must respond within one month.', 'A firm must respond within three months.'),
        ], ['DEADLINE_CHANGED'])
        # A suffix on the same period, a word that only starts like a unit ("ayrı", "gündem"), and a
        # number word glued to one ("onay" is approval, not ten months) are no deadline.
        self.assertKinds([
            ('Bildirim 30 gün içinde yapılır.', 'Bildirim 30 günlük süre içinde yapılır.'),
            ('Üç ayrı belge alınır.', 'Beş ayrı belge alınır.'),
            ('Kurul iki gündem maddesini görüşür.', 'Kurul üç gündem maddesini görüşür.'),
            ('İşlem yönetim kurulunun onayı ile yapılır.', 'İşlem yönetim kurulunun kararı ile yapılır.'),
            ('Yönetim kurulu onay verir.', 'Yönetim kurulu karar verir.'),
        ], ['TEXT_CHANGED'])

    def test_an_entity_scope_change_is_read_from_the_named_parties(self):
        self.assertKinds([
            ('Müşteri gerçek kişi ise kimlik tespiti yapılır.', 'Müşteri gerçek kişi veya dernek ise kimlik tespiti yapılır.'),
            ('Bankalar müşterinin kimliğini tespit eder.', 'Bankalar ve ödeme kuruluşları müşterinin kimliğini tespit eder.'),
            ('A firm must assess affordability for individuals.', 'A firm must assess affordability for individuals and companies.'),
            ('A firm must keep a register.', 'A bank must keep a register.'),
        ], ['ENTITY_SCOPE_CHANGED'])

    def test_an_exemption_change_is_read_from_exclusion_wording_including_its_inflected_forms(self):
        self.assertKinds([
            ('Kimlik tespiti bütün işlemlerde yapılır.', 'Kimlik tespiti, havale işlemleri hariç, bütün işlemlerde yapılır.'),
            ('Yükümlüler kimlik tespiti yapar.', 'Yükümlüler kimlik tespiti yapar; sigortacılık işlemleri muafiyet kapsamındadır.'),
            ('Yükümlüler kimlik tespiti yapar.', 'Yükümlüler kimlik tespiti yapar; bu hükmün istisnası yönetmelikle belirlenir.'),
            ('A firm must assess affordability.', 'A firm must assess affordability unless the customer is exempt.'),
            ('A firm must assess affordability.', 'Exempted transactions need not be assessed; a firm must assess affordability.'),
            ('A firm must respond to consumers.', 'A firm must respond to consumers, except where the complaint is withdrawn.'),
        ], ['EXEMPTION_CHANGED'])
        # "exceptional" is not exclusion wording.
        self.assertEqual(change_kinds('A firm must act in exceptional circumstances.', 'A firm must act promptly in exceptional circumstances.'),
                         ['TEXT_CHANGED'])
        # Removing an exclusion that names a party changes both the scope and the exemption.
        self.assertEqual(change_kinds('Bu madde bütün işlemlere uygulanır; kamu kurumlarına uygulanmaz.', 'Bu madde bütün işlemlere uygulanır.'),
                         ['ENTITY_SCOPE_CHANGED', 'EXEMPTION_CHANGED'])

    def test_plain_wording_spacing_and_combined_changes(self):
        self.assertKinds([('Yükümlüler kimlik tespiti yapar.', 'Yükümlüler kimlik tespitini eksiksiz yapar.'),
                          ('A firm must retain records.', 'A firm must keep records.')], ['TEXT_CHANGED'])
        self.assertEqual(change_kinds('A firm must retain records.', ' A  firm must retain\nrecords. '), [])
        # Several kinds come in the fixed order, and so do their actions.
        kinds = change_kinds('Tutar 75.000 TL ise bildirim on gün içinde yapılır.', 'Tutar 100.000 TL ise bildirim yirmi gün içinde yapılır.')
        self.assertEqual(kinds, ['THRESHOLD_CHANGED', 'DEADLINE_CHANGED'])
        self.assertEqual(suggested_action(list(reversed(kinds))), ACTIONS['THRESHOLD_CHANGED'] + ' ' + ACTIONS['DEADLINE_CHANGED'])
        self.assertEqual(suggested_action([]), ACTIONS['TEXT_CHANGED'])


class SnapshotChangeTests(unittest.TestCase):
    """Two duties, CONC 7.3.4 and 7.3.5; the next snapshot adds, deletes or changes one of them."""

    def setUp(self):
        self.snapshot = [*sections(), notify()]
        self.labels = ['CONC 7.3.4', 'CONC 7.3.5']
        self.first = analyze(company(), policies(), self.snapshot, Counting(), self.labels)

    def alone(self, snapshot, label):
        """The calls a fresh analysis of one provision costs, for comparison with an incremental one."""
        provider = Counting()
        analyze(company(), policies(), snapshot, provider, [label])
        return provider.calls

    def test_a_new_provision_is_the_only_one_read_and_costs_what_it_costs_on_its_own(self):
        first = analyze(company(), policies(), sections(), Counting(), ['CONC 7.3.4'])
        snapshot = renewed([*sections(), notify()])
        provider = Counting()
        second = analyze(company(), policies(), snapshot, provider, self.labels, *compared(first))
        payload = payload_of(second)
        self.assertEqual(asked(provider, snapshot), {'CONC 7.3.5'})
        self.assertEqual(sorted(task for task, _ in provider.calls), sorted(task for task, _ in self.alone(snapshot, 'CONC 7.3.5')))
        impact = payload['impact']
        self.assertEqual((impact['new'], impact['changed'], impact['deleted_or_unselected']), (['CONC 7.3.5'], [], []))
        self.assertEqual((impact['carried_forward'], impact['reanalysed'], impact['reused_unchanged'], impact['affects_company']), (1, 1, True, True))
        self.assertEqual({c['source']['printed_label']: c['change']['status'] for c in payload['cases']},
                         {'CONC 7.3.4': 'TEXT_UNCHANGED', 'CONC 7.3.5': 'NEW_IN_SNAPSHOT'})
        rows = {r['source_label']: r for r in payload['obligations']}
        self.assertEqual(rows['CONC 7.3.4']['carried_forward']['previous_id'], payload_of(first)['obligations'][0]['id'])
        self.assertNotIn('carried_forward', rows['CONC 7.3.5'])
        self.assertTrue(verify_chain(second['events'], second['head'], second['count']))
        canonical_bytes(payload)
        self.assertIn("Bu snapshot'ta yeni provision:</b> CONC 7.3.5", render(second))

    def test_a_deleted_or_unselected_provision_is_listed_and_nothing_is_read_again(self):
        for name, snapshot in (('deleted from the regulation', renewed(sections())), ('left out of the selection', renewed(self.snapshot))):
            with self.subTest(name):
                provider = Counting()
                second = analyze(company(), policies(), snapshot, provider, ['CONC 7.3.4'], *compared(self.first))
                impact = payload_of(second)['impact']
                self.assertEqual(provider.calls, [])
                self.assertEqual(impact['deleted_or_unselected'], ['CONC 7.3.5'])
                self.assertEqual((impact['carried_forward'], impact['reanalysed'], impact['new'], impact['changed']), (1, 0, [], []))
                self.assertIn('Önceki analizde olup bu seçimde olmayan: CONC 7.3.5.', render(second))

    def test_a_changed_provision_is_read_again_and_the_unchanged_one_is_carried_without_a_call(self):
        snapshot = renewed(self.snapshot, {'CONC 7.3.5': 'A firm must notify the customer in writing within 14 days.'})
        provider = Counting()
        second = analyze(company(), policies(), snapshot, provider, self.labels, *compared(self.first))
        payload = payload_of(second)
        self.assertEqual(asked(provider, snapshot), {'CONC 7.3.5'})
        self.assertEqual(sorted(task for task, _ in provider.calls), sorted(task for task, _ in self.alone(snapshot, 'CONC 7.3.5')))
        old = {r['source_label']: r for r in payload_of(self.first)['obligations']}
        rows = {r['source_label']: r for r in payload['obligations']}
        carried = rows['CONC 7.3.4']
        self.assertEqual((carried['carried_forward']['from_head'], carried['carried_forward']['previous_id']), (self.first['head'], old['CONC 7.3.4']['id']))
        self.assertEqual(carried['source_id'], 'n-s-duty')                                      # the row points into the new snapshot
        self.assertEqual(carried['proposal']['applicability'], old['CONC 7.3.4']['proposal']['applicability'])
        self.assertNotIn('carried_forward', rows['CONC 7.3.5'])
        self.assertEqual((payload['regulation']['carried_forward'], payload['regulation']['reanalysed']), (1, 1))
        self.assertTrue(payload['input_changes']['unchanged_reused'])
        (change,) = payload['impact']['changed']
        self.assertEqual({key: change[key] for key in ('label', 'kinds', 'old_text', 'new_text', 'affected_policies', 'previous_coverage')},
                         {'label': 'CONC 7.3.5', 'kinds': ['DEADLINE_CHANGED'], 'old_text': notify()['text'],
                          'new_text': 'A firm must notify the customer in writing within 14 days.', 'affected_policies': ['fixture.txt'],
                          'previous_coverage': [old['CONC 7.3.5']['proposal']['coverage']]})
        self.assertEqual(change['suggested_action'], ACTIONS['DEADLINE_CHANGED'])
        self.assertTrue(payload['impact']['affects_company'])
        self.assertTrue(any('carried_forward' in line for line in payload['limitations']))
        self.assertTrue(verify_chain(second['events'], second['head'], second['count']))

    def test_a_spacing_only_edit_is_read_again_and_reported_as_a_wording_change(self):
        # change_for compares raw text, so the provision is read again; the impact must still name a kind.
        snapshot = renewed(self.snapshot, {'CONC 7.3.5': 'A firm must  notify the customer in writing. '})
        provider = Counting()
        payload = payload_of(analyze(company(), policies(), snapshot, provider, self.labels, *compared(self.first)))
        self.assertEqual(asked(provider, snapshot), {'CONC 7.3.5'})
        self.assertEqual([(c['label'], c['kinds']) for c in payload['impact']['changed']], [('CONC 7.3.5', ['TEXT_CHANGED'])])

    def test_a_row_the_compared_analysis_never_assessed_is_read_again(self):
        stop = {'now': False}

        def progress(done, total, label, phase):
            # Stop once 7.3.4 has been assessed: 7.3.5 is extracted but never judged.
            stop['now'] = stop['now'] or (phase == 'assessment' and label == 'CONC 7.3.4')
        first = analyze(company(), policies(), self.snapshot, Counting(), self.labels, progress=progress, should_stop=lambda: stop['now'])
        reasons = {r['source_label']: r['proposal']['applicability_reason'] for r in payload_of(first)['obligations']}
        self.assertTrue(reasons['CONC 7.3.5'].startswith('Analysis stopped'), reasons)
        snapshot = renewed(self.snapshot)
        provider = Counting()
        payload = payload_of(analyze(company(), policies(), snapshot, provider, self.labels, *compared(first)))
        self.assertEqual(asked(provider, snapshot), {'CONC 7.3.5'})
        self.assertEqual((payload['regulation']['carried_forward'], payload['regulation']['reanalysed']), (1, 1))
        rows = {r['source_label']: r for r in payload['obligations']}
        self.assertNotIn('carried_forward', rows['CONC 7.3.5'])
        self.assertFalse(rows['CONC 7.3.5']['proposal']['applicability_reason'].startswith('Analysis stopped'))


class ReuseBlockedTests(unittest.TestCase):
    """Unchanged text is not enough: any other input of the judgement re-reads every provision."""

    def setUp(self):
        self.snapshot = [*sections(), notify()]
        self.labels = ['CONC 7.3.4', 'CONC 7.3.5']
        self.second_policy = dict(policies()[0], filename='second.txt', raw_hash='e' * 64,
                                  chunks=[dict(policies()[0]['chunks'][0], source_id='policy-2', policy_hash='e' * 64, filename='second.txt',
                                               text='Staff must notify customers in writing.')])
        self.policies = [*policies(), self.second_policy]
        self.first = analyze(company(), self.policies, self.snapshot, Counting(), self.labels)

    def again(self, profile=None, package=None, provider_class=Counting, **options):
        snapshot = renewed(self.snapshot)
        provider = provider_class()
        packet = analyze(profile or company(), package or self.policies, snapshot, provider, self.labels, *compared(self.first), **options)
        return payload_of(packet), asked(provider, snapshot)

    def test_the_same_inputs_carry_every_unchanged_provision_whatever_the_policy_order(self):
        payload, read = self.again(package=list(reversed(self.policies)))
        self.assertEqual(read, set())
        self.assertEqual((payload['impact']['carried_forward'], payload['impact']['reanalysed'], payload['impact']['reuse_blocked_reason']), (2, 0, None))

    def test_a_different_company_policy_package_prompt_model_or_retrieval_blocks_reuse_and_is_named(self):
        edited = company()
        edited.description = 'Synthetic fixture, now also a debt collector'

        class OtherModel(Counting):
            model_version = 'other-model-v9'
        variants = [
            ('company profile changed', dict(profile=edited), None),
            ('policy package changed', dict(package=[*policies(), dict(self.second_policy, raw_hash='f' * 64)]), None),
            ('prompts or judgement version changed', {}, patch.object(engine, 'PROMPT_HASH', 'f' * 64)),
            ('prompts or judgement version changed', {}, patch.object(engine, 'EXTRACTION_PROMPT_HASH', 'f' * 64)),
            ('model runtime changed', dict(provider_class=OtherModel), None),
            ('model runtime changed', dict(judge=OtherModel()), None),
            ('model runtime changed', dict(votes=2), None),
            ('retrieval configuration changed', dict(embedder=ConceptEmbedder()), None),
            ('company profile changed; policy package changed', dict(profile=edited, package=policies()), None),
        ]
        for reason, options, patcher in variants:
            with self.subTest(reason=reason, options=sorted(options)):
                if patcher is not None:
                    with patcher:
                        payload, read = self.again(**options)
                else:
                    payload, read = self.again(**options)
                impact = payload['impact']
                self.assertEqual(impact['reuse_blocked_reason'], reason)
                self.assertFalse(impact['reused_unchanged'])
                self.assertFalse(payload['input_changes']['unchanged_reused'])
                self.assertEqual((impact['carried_forward'], impact['reanalysed']), (0, 2))
                self.assertFalse(any('carried_forward' in row for row in payload['obligations']))
                if 'judge' not in options:                                   # a separate judge gets the calls itself
                    self.assertEqual(read, {'CONC 7.3.4', 'CONC 7.3.5'})

    def test_reuse_blockers_directly(self):
        old = payload_of(self.first)
        self.assertEqual(reuse_blockers(None, old['company_hash'], self.policies, 'x', 'y'), ['no previous analysis'])
        self.assertEqual(reuse_blockers(old, old['company_hash'], list(reversed(self.policies)), old['extraction_prompt_hash'], old['pilot_prompt_hash']), [])
        self.assertEqual(reuse_blockers(old, 'other', policies(), 'x', old['pilot_prompt_hash']),
                         ['company profile changed', 'policy package changed', 'prompts or judgement version changed'])


class TurkishSnapshotTests(unittest.TestCase):
    """The same guarantees on a Turkish regulation: md. 4 (bildirim) and md. 5 (kimlik tespiti)."""

    def setUp(self):
        self.snapshot = [*turkish_sections(), kyc()]
        self.labels = ['Kanun 5549 md. 4', 'Kanun 5549 md. 5']
        self.first = analyze(anadolu(), policies(), self.snapshot, TurkishCounting(), self.labels)
        self.old = {}
        for row in payload_of(self.first)['obligations']:
            self.old.setdefault(row['source_label'], []).append(row)

    def test_a_threshold_change_rereads_only_that_article(self):
        new_text = kyc()['text'].replace('75.000 TL', '185.000 TL')
        snapshot = renewed(self.snapshot, {'Kanun 5549 md. 5': new_text})
        provider = TurkishCounting()
        payload = payload_of(analyze(anadolu(), policies(), snapshot, provider, self.labels, *compared(self.first)))
        self.assertEqual(asked(provider, snapshot), {'Kanun 5549 md. 5'})
        self.assertEqual(payload['impact']['carried_forward'], len(self.old['Kanun 5549 md. 4']))
        self.assertTrue(all(r['carried_forward']['reason'] == 'TEXT_UNCHANGED_SAME_INPUTS' for r in payload['obligations']
                            if r['source_label'] == 'Kanun 5549 md. 4'))
        (change,) = payload['impact']['changed']
        self.assertEqual((change['label'], change['kinds'], change['new_text']), ('Kanun 5549 md. 5', ['THRESHOLD_CHANGED'], new_text))
        self.assertTrue(change['suggested_action'].startswith('Eşik veya tutar değişti'))

    def test_a_deadline_change_rereads_only_that_article(self):
        new_text = turkish_sections()[1]['text'].replace('Başkanlığa bildirilmesi zorunludur', 'on iş günü içinde Başkanlığa bildirilmesi zorunludur')
        snapshot = renewed(self.snapshot, {'Kanun 5549 md. 4': new_text})
        provider = TurkishCounting()
        payload = payload_of(analyze(anadolu(), policies(), snapshot, provider, self.labels, *compared(self.first)))
        self.assertEqual(asked(provider, snapshot), {'Kanun 5549 md. 4'})
        self.assertEqual(payload['impact']['carried_forward'], len(self.old['Kanun 5549 md. 5']))
        (change,) = payload['impact']['changed']
        self.assertEqual((change['label'], change['kinds']), ('Kanun 5549 md. 4', ['DEADLINE_CHANGED']))
        self.assertEqual(change['previous_coverage'], [r['proposal']['coverage'] for r in self.old['Kanun 5549 md. 4']])


class ImpactHelperTests(unittest.TestCase):
    def test_remap_rewrites_regulation_ids_and_leaves_policy_provenance_alone(self):
        row = {'source_id': 'old-duty', 'proposal': {
            'scope_evidence': [{'source_id': 'old-scope', 'quote': 'q'}], 'basis': [{'source_id': 'old-scope', 'company_fact': 'f'}],
            'provenance': [{'kind': 'regulation', 'source_id': 'old-scope', 'passage_id': 'old-scope'},
                           {'kind': 'policy', 'source_id': 'policy-1', 'passage_id': 'policy-1'}]}}
        copy = remap_source_ids(row, {'old-duty': 'new-duty', 'old-scope': 'new-scope'})
        self.assertEqual(copy['source_id'], 'new-duty')
        self.assertEqual(copy['proposal']['scope_evidence'][0]['source_id'], 'new-scope')
        self.assertEqual(copy['proposal']['basis'][0]['source_id'], 'new-scope')
        self.assertEqual([(p['source_id'], p['passage_id']) for p in copy['proposal']['provenance']], [('new-scope', 'new-scope'), ('policy-1', 'policy-1')])
        self.assertEqual(row['source_id'], 'old-duty')                                         # the previous row is not edited

    def test_the_summary_without_a_previous_analysis_lists_nothing_as_deleted(self):
        summary = impact_summary(None, None, [], [], 0, 0, ['no previous analysis'], ['CONC 7.3.4'])
        self.assertEqual((summary['deleted_or_unselected'], summary['reused_unchanged'], summary['reuse_blocked_reason']),
                         ([], False, 'no previous analysis'))


if __name__ == '__main__':
    unittest.main()

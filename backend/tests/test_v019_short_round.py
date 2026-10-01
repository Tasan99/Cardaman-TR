"""v0.19 WS6b-lite (short validation round): model roles, per-obligation staging, the judge window per
call (providers.fit_call) with passage compression, the definitions article, and the three C12 smoke
regressions (run 20260924-173120-v019-smoke-ollama, packets/C12.json).

Every model is a scripted fake (no Ollama). `Fast` stands for the extraction provider's no-thinking
instance (qwen3:4b), `Strong` for the thinking judge (qwen3:8b); both write to one shared log, so the
order of the calls and the model switches can be counted.
"""
import json
import unittest
from unittest.mock import patch

from regchain.evidence import canonical_bytes
from regchain.extraction import providers
from regchain.extraction.providers import AI_TASK, AI_UNCACHED, failure
from regchain.pilot import conflict as precheck
from regchain.pilot import definitions as defined
from regchain.pilot.engine import (FAST_PROMPT, SCOPE_PROMPT, VERIFY_PROMPT, analyze, coverage_of_v19, join_kept, passage_compressor,
                                   passages_v19, pipeline_settings, sentence_spans)
from test_pilot import company, policies, sections
from test_v019_conflict_pipeline import _SECTIONS, KYC_SKIP, KYC_SKIP_SPAN, MONTH_END, NOISE, V19, fast, md5_2, md28, md46, tedbirler, verdict

# The three C12 policy passages, verbatim from the smoke packet.
REGISTER = ('Müşteri kabulünde müşterinin adı, soyadı ve kimlik numarası sisteme kaydedilir. İş ilişkisinin amacı hakkında not '
            'alınması operasyon ekibinin takdirindedir.')
REGISTER_FIRST = 'Müşteri kabulünde müşterinin adı, soyadı ve kimlik numarası sisteme kaydedilir.'
REPORT = "Şüpheli görülen işlemler uyum birimi aracılığıyla MASAK'a bildirilir."
RECORDS = 'İşlem kayıtları ve müşteri dosyaları belge yönetim sisteminde saklanır.'


def tedbirler_definitions():
    tedbirler()                                                   # loads the snapshot (every article, 'md. 3' Tanımlar among them)
    return defined.definitions_of(list(_SECTIONS))


class Logged(V19):
    """A V19 fake that writes (role, kind) into a shared log."""

    def __init__(self, role, log, **answers):
        super().__init__(**answers)
        self.role, self.log = role, log

    def _chat(self, prompt, payload, schema):
        fields = schema['properties']
        kind = next((k for k in ('label', 'conflict', 'items', 'summary', 'clause', 'applicability') if k in fields), 'other')
        self.log.append((self.role, kind, AI_TASK.get()))
        return super()._chat(prompt, payload, schema)


def rows(*texts):
    return [{'source_id': f'p{i}', 'text': text} for i, text in enumerate(texts, 1)]


def run(judge, payload, quantities, texts, fast_model=None, definitions=None):
    checks, _, results, _, _ = passages_v19(judge, payload, rows(*texts), frozenset(), quantities, 1, 's', 'o', False,
                                            fast=fast_model, definitions=definitions)
    outcome = coverage_of_v19(checks, results, frozenset(), quantities, payload.get('elements'))
    canonical_bytes(results)                                      # packets forbid floats
    return {c.source_id: c.relation for c in checks}, {r['source_id']: r for r in results}, outcome


def switches(log):
    return sum(1 for a, b in zip(log, log[1:]) if a[0] != b[0])


class DefinitionsTests(unittest.TestCase):
    def test_the_tedbirler_definitions_article_is_read_by_rule(self):
        found = tedbirler_definitions()
        self.assertTrue(found['Başkanlık'].startswith('Mali Suçları Araştırma Kurulu Başkanlığı'))
        self.assertEqual(found['Bakanlık'], 'Hazine ve Maliye Bakanlığını')
        # An amendment note is not part of the term or the definition; the closing "ifade eder" is dropped.
        self.assertTrue(found['Denetim elemanı'].startswith('Vergi Müfettişleri'))
        self.assertTrue(found['Finansal olmayan belirli iş ve meslekler'].endswith('bentlerinde sayılan yükümlüleri'))
        self.assertEqual(defined.definitions_of(sections()), {})                           # FCA: no definitions wording

    def test_only_terms_in_the_duty_or_the_passage_are_shown_longest_first(self):
        found = tedbirler_definitions()
        payload, _ = md28()
        shown = defined.relevant(found, payload['required_action'], REPORT)
        # "Başkanlığa" is the defined "Başkanlık" (k→ğ), not "Başkan".
        self.assertEqual(shown, {'Başkanlık': 'Mali Suçları Araştırma Kurulu Başkanlığını'})
        self.assertIn('Başkan', defined.relevant(found, 'Başkan onaylar.'))
        many = defined.relevant(found, ' '.join(found))
        self.assertEqual(len(many), defined.MAX_TERMS)
        self.assertTrue(all(len(v) <= defined.MAX_CHARS for v in many.values()))
        self.assertEqual(defined.relevant({}, 'Başkanlığa'), {})


class TopicOverlapTests(unittest.TestCase):
    def test_synonyms_of_a_core_act_and_the_passage_share_count(self):
        payload, _ = md46()
        # C12 md. 46(1): 0.176 in the smoke run; "saklanır" is "muhafaza" (keeping).
        self.assertGreaterEqual(precheck.topic_overlap(payload, RECORDS), 0.4)
        self.assertTrue(precheck.shares_act(payload, RECORDS))
        self.assertEqual(precheck.core_acts(md28()[0]), {'@report'})
        self.assertTrue(precheck.shares_act(md28()[0], REPORT))
        self.assertIn('@identify', precheck.core_acts(md5_2()[0]))                       # the act is the subject: "Kimlik tespiti"
        for text in NOISE:
            self.assertLess(precheck.topic_overlap(payload, text), 0.4, text)
            self.assertFalse(precheck.shares_act(payload, text), text)
        # A two-word fragment does not reach 1.0 through the passage→duty share.
        self.assertLess(precheck.topic_overlap(payload, '## Kayıtlar'), 0.4)


class C12RegressionTests(unittest.TestCase):
    def test_md28_a_missing_deadline_and_masak_for_baskanlik_is_partial_not_conflict(self):
        payload, quantities = md28()
        definitions = tedbirler_definitions()
        # The fast reading as the v2 prompt asks: the act without its deadline is PARTIAL; no conflict question, one
        # confirmation of the PARTIAL the result rests on (short round 3).
        judge = V19(fast=fast('PARTIAL', REPORT, covered=('action',), missing=('deadline_1',)),
                    verify={**verdict(relation='PARTIAL', support=REPORT), 'covered_elements': ['action'], 'missing_elements': ['deadline_1']})
        relations, results, (coverage, reason, _, flags) = run(judge, payload, quantities, [REPORT], definitions=definitions)
        self.assertEqual((relations['p1'], coverage, flags, [c['task'] for c in judge.kinds('verify')]),
                         ('PARTIAL', 'PARTIAL', [], ['judge.verify.support']))
        self.assertIn('QUANTITY_NOT_STATED', reason)
        self.assertEqual(results['p1']['definitions'], ['Başkanlık'])
        # Even when the fast reading still says POSSIBLE_CONFLICT (as in the smoke run), the verifier is shown the
        # definition and the rule, and a PARTIAL answer is a PARTIAL.
        judge = V19(fast=fast('POSSIBLE_CONFLICT', REPORT, missing=('deadline_1',)),
                    verify={**verdict(relation='PARTIAL', support=REPORT), 'covered_elements': ['action'], 'missing_elements': ['deadline_1']})
        relations, results, (coverage, _, _, flags) = run(judge, payload, quantities, [REPORT], definitions=definitions)
        self.assertEqual((relations['p1'], coverage, flags), ('PARTIAL', 'PARTIAL', []))
        shown = judge.kinds('verify')[0]['payload']['duty']
        self.assertEqual(shown['definitions'], {'Başkanlık': 'Mali Suçları Araştırma Kurulu Başkanlığını'})
        self.assertEqual(results['p1']['verifier']['missing'], ['deadline_1'])
        for prompt in (FAST_PROMPT, VERIFY_PROMPT):
            self.assertIn('acronym', prompt)
        self.assertIn('never POSSIBLE_CONFLICT', FAST_PROMPT)
        self.assertIn('silence or a missing element', VERIFY_PROMPT)

    def test_md46_keeping_records_without_the_period_reaches_partial_through_the_ambiguity_check(self):
        payload, quantities = md46()
        # The smoke run's fast answer: IRRELEVANT, listing the act and the period as missing.
        # (md. 46(1) has two acts since short round 2: keeping is act_1; a PARTIAL needs a critical element stated.)
        judge = V19(fast=fast('IRRELEVANT', covered=(), missing=('act_1', 'deadline_1')),
                    verify={**verdict(relation='PARTIAL', support=RECORDS), 'covered_elements': ['act_1'],
                            'missing_elements': ['act_2', 'deadline_1']})
        relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [RECORDS, NOISE[0], NOISE[1]])
        self.assertEqual((relations['p1'], results['p1']['escalation'], coverage), ('PARTIAL', 'AMBIGUOUS_IRRELEVANT', 'PARTIAL'))
        self.assertEqual([c['task'] for c in judge.kinds('verify')], ['judge.verify.support'])        # the noise is not re-asked
        self.assertIn('sekiz yıl süre ile', reason)
        # An IRRELEVANT that lists a covered element contradicts itself: a best-ranked one is re-asked too.
        judge = V19(fast=fast('IRRELEVANT', covered=('act_1',)), verify=verdict(relation='UNRELATED'))
        relations, results, _ = run(judge, payload, quantities, [NOISE[2]])
        self.assertEqual((relations['p1'], results['p1']['escalation']), ('UNRELATED', 'AMBIGUOUS_IRRELEVANT'))

    def test_md5_2_a_verifier_support_without_the_before_timing_is_partial(self):
        payload, quantities = md5_2()
        self.assertEqual([(q['element'], q['kind']) for q in quantities], [('deadline_1', 'BEFORE')])
        # The smoke run: IRRELEVANT, sent to the verifier by "takdirindedir", answered SUPPORTS.
        smoke = fast('IRRELEVANT', covered=(), missing=('action', 'deadline_1'))
        answers = {
            'missing listed': ({'covered_elements': ['action'], 'missing_elements': ['deadline_1']}, 'PARTIAL', 'PARTIAL'),
            'timing not listed': ({'covered_elements': ['action'], 'missing_elements': []}, 'SUPPORTS', 'PARTIAL'),
            'no lists (v1 answer)': ({}, 'SUPPORTS', 'PARTIAL'),
            # t6: a listed timing counts only when the passage's text states it; this one states none (gold PARTIAL).
            'timing listed': ({'covered_elements': ['action', 'deadline_1'], 'missing_elements': []}, 'SUPPORTS', 'PARTIAL')}
        for case, (lists, relation, expected) in answers.items():
            with self.subTest(case=case):
                judge = V19(fast=smoke, verify={**verdict(relation='SUPPORTS', support=REGISTER_FIRST), **lists})
                relations, results, (coverage, reason, _, _) = run(judge, payload, quantities, [REGISTER])
                self.assertEqual((results['p1']['escalation'], relations['p1'], coverage), ('STRUCTURAL_SIGNAL', relation, expected))
                if expected == 'PARTIAL' and relation == 'SUPPORTS':
                    self.assertIn('TIMING_NOT_STATED', reason)
        # The same timing rule for a fast SUPPORTS that does not list the timing (confirmed once, short round 3).
        judge = V19(fast=fast('SUPPORTS', REGISTER_FIRST), verify={**verdict(relation='SUPPORTS', support=REGISTER_FIRST),
                                                                  'covered_elements': ['action'], 'missing_elements': []})
        relations, _, (coverage, _, _, _) = run(judge, payload, quantities, [REGISTER_FIRST])
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'PARTIAL'))

    def test_the_c13_and_c14_conflicts_still_stand(self):
        payload, quantities = md5_2()
        judge = V19(fast=fast('SUPPORTS', 'müşteri yalnızca telefon numarasını doğrular.'),
                    verify=verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE'))
        self.assertEqual(run(judge, payload, quantities, [KYC_SKIP])[2][0], 'CONFLICT')
        payload, quantities = md28()
        span = "uygun görülenler o ay sonunda MASAK'a topluca bildirilir."
        judge = V19(fast=fast('PARTIAL', MONTH_END, missing=('deadline_1',)), verify=verdict(True, 'DEADLINE_MISMATCH', span, 'NOT_APPLICABLE'))
        relations, results, (coverage, _, _, _) = run(judge, payload, quantities, [MONTH_END], definitions=tedbirler_definitions())
        self.assertEqual((relations['p1'], coverage, results['p1']['escalation']), ('CONFLICTS', 'CONFLICT', 'STRUCTURAL_SIGNAL'))


class ModelRoleTests(unittest.TestCase):
    def analyze(self, fast_answer, verify_answer=None, **settings):
        log = []
        extraction = Logged('FAST', log, fast=fast_answer)
        judge = Logged('STRONG', log, fast=lambda text: self.fail('the judge was asked a fast question'), verify=verify_answer)
        packet = analyze(company(), policies(), sections(), extraction, ['CONC 7.3.4'], judge=judge, categories=['Diğer'],
                         settings=pipeline_settings(coverage_pipeline='v19', **settings))
        return packet['events'][0]['payload'], log

    def test_fast_work_goes_to_the_extraction_model_and_only_verification_to_the_judge(self):
        payload, log = self.analyze(lambda text: fast('POSSIBLE_CONFLICT'), verdict(relation='UNRELATED'))
        self.assertEqual([(role, kind) for role, kind, _ in log],
                         [('FAST', 'summary'), ('FAST', 'label'), ('STRONG', 'applicability'), ('STRONG', 'conflict'), ('FAST', 'clause')])
        self.assertEqual({task for role, _, task in log if role == 'STRONG'}, {'judge.applicability', 'judge.verify'})
        self.assertEqual(switches(log), 2)                                  # one 4b→8b→4b round trip
        self.assertEqual(payload['obligations'][0]['proposal']['remediation']['draft_status'], 'DRAFTED')

    def test_nothing_escalated_and_rules_deciding_leave_the_judge_unloaded(self):
        # A fast PARTIAL no longer stands alone (short round 3): nothing is escalated to the conflict question, and the
        # judge's one passage call confirms the PARTIAL the result rests on (CONFIRM_PARTIAL, task judge.verify.support).
        payload, log = self.analyze(lambda passage: fast('PARTIAL', passage, missing=()),
                                    lambda passage: verdict(relation='PARTIAL', support=passage), applicability_clear_match='rule')
        self.assertEqual(payload['obligations'][0]['proposal']['coverage'], 'PARTIAL')
        if payload['obligations'][0]['proposal']['applicability_rule'] != 'RULE_CLEAR_MATCH':
            self.assertEqual([k for r, k, _ in log if r == 'STRONG'][:1], ['applicability'])   # the rules left it open
        self.assertEqual([t for r, k, t in log if r == 'STRONG' and k == 'conflict'], ['judge.verify.support'])
        # A fast SUPPORTS alone never covers: the judge's one call is the COVERS confirmation (v0.19 gate).
        payload, log = self.analyze(lambda passage: fast('SUPPORTS', passage), lambda passage: verdict(relation='SUPPORTS', support=passage),
                                    applicability_clear_match='rule')
        self.assertEqual(payload['obligations'][0]['proposal']['coverage'], 'COVERS_TEXT')
        self.assertIn(('STRONG', 'conflict', 'judge.verify.support'), log)
        self.assertEqual([t for r, k, t in log if r == 'STRONG' and k == 'conflict'], ['judge.verify.support'])

    def test_every_fast_reading_of_an_obligation_comes_before_any_verifier_call(self):
        payload, quantities = md5_2()
        log = []
        answers = {KYC_SKIP: fast('POSSIBLE_CONFLICT'), MONTH_END: fast('POSSIBLE_CONFLICT')}
        fast_model = Logged('FAST', log, fast=lambda text: answers.get(text) or fast('IRRELEVANT', covered=()))
        judge = Logged('STRONG', log, verify=lambda text: verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE')
                       if text == KYC_SKIP else verdict(relation='UNRELATED'))
        texts = [KYC_SKIP, NOISE[0], MONTH_END, NOISE[1], NOISE[2]]
        relations, _, (coverage, _, _, _) = run(judge, payload, quantities, texts, fast_model=fast_model)
        self.assertEqual(coverage, 'CONFLICT')
        kinds = [(role, kind) for role, kind, _ in log]
        self.assertEqual(kinds, [('FAST', 'label')] * 5 + [('STRONG', 'conflict')] * 2)
        self.assertEqual(switches(log), 1)

    def test_v19_applicability_goes_through_fit_call_on_the_judge_never_its_wide_twin_in_fixed_mode(self):
        log = []
        wide = Logged('WIDE', log)
        judge = Logged('STRONG', log, verify=verdict(relation='UNRELATED'))
        judge.wide, judge.num_ctx, judge.num_predict = wide, 8192, 4096
        extraction = Logged('FAST', log, fast=lambda text: fast('IRRELEVANT', covered=()))
        seen = []
        real = providers.fit_call

        def spy(instance, prompt, payload, schema, compress=None):
            seen.append((instance.role, prompt))
            return real(instance, prompt, payload, schema, compress)
        with patch('regchain.pilot.engine.fit_call', side_effect=spy):
            analyze(company(), policies(), sections(), extraction, ['CONC 7.3.4'], judge=judge, categories=['Diğer'],
                    settings=pipeline_settings(coverage_pipeline='v19'))
        self.assertIn(('STRONG', SCOPE_PROMPT), seen)
        self.assertIn(('FAST', FAST_PROMPT), seen)
        self.assertNotIn('WIDE', [role for role, _, _ in log])
        # The v18 path keeps the full-window twin for applicability, as before.
        log.clear()
        analyze(company(), policies(), sections(), extraction, ['CONC 7.3.4'], judge=judge, categories=['Diğer'],
                settings=pipeline_settings(coverage_pipeline='v18'))
        self.assertIn(('WIDE', 'applicability'), [(r, k) for r, k, _ in log])


class Narrow(V19):
    """A fake with a window, for fit_call: `wide` and ctx_mode make it an adaptive judge."""

    def __init__(self, num_ctx, num_predict, **answers):
        super().__init__(**answers)
        self.num_ctx, self.num_predict, self.sent = num_ctx, num_predict, []

    def _chat(self, prompt, payload, schema):
        self.sent.append(payload.get('passage'))
        return super()._chat(prompt, payload, schema)


class WindowTests(unittest.TestCase):
    FILLER = [f'Ofis düzeni {i} numaralı kat planına göre yürütülür ve toplantı salonları haftalık takvimle paylaştırılır.'
              for i in range(60)]

    def long_passage(self, middle=KYC_SKIP):
        return ' '.join(self.FILLER[:30] + [middle] + self.FILLER[30:])

    def test_a_long_passage_is_compressed_to_its_signal_sentences_verbatim_and_quotes_still_validate(self):
        payload, quantities = md5_2()
        text = self.long_passage()
        small = Narrow(6144, 1024, fast=fast('POSSIBLE_CONFLICT'), verify=verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE'))
        relations, results, (coverage, _, _, _) = run(small, payload, quantities, [text])
        self.assertEqual((relations['p1'], coverage), ('CONFLICTS', 'CONFLICT'))
        sent = small.sent[0]
        self.assertIn(KYC_SKIP, sent)                                       # the signal sentences, verbatim and in order
        self.assertIn('[...]', sent)
        self.assertLess(len(sent), len(text))
        self.assertEqual(results['p1']['verifier']['contradiction_span'], KYC_SKIP_SPAN)
        self.assertEqual({n.get('window_action') for n in results['p1']['notes'] if n['code'] == 'WINDOW'}, {'compressed'})
        # A quote from a sentence the compression dropped still validates against the full passage.
        dropped, text = self.FILLER[59], self.long_passage(REGISTER_FIRST)
        small = Narrow(6144, 1024, fast=fast('PARTIAL', dropped, missing=('deadline_1',)))
        relations, results, _ = run(small, payload, quantities, [text])
        self.assertNotIn(dropped, small.sent[0])
        self.assertIn(REGISTER_FIRST, small.sent[0])                        # the sentence that shares the duty's words is kept
        self.assertEqual(results['p1']['fast']['quote'], dropped)
        # The quote validates, but it names nothing of the duty (office layout): since the anchor rule (C14 md. 26(2)
        # smoke run) such a favourable fast reading goes to the strong model as AMBIGUOUS_SUPPORT instead of standing.
        self.assertEqual(results['p1']['escalation'], 'AMBIGUOUS_SUPPORT')

    def test_the_compressor_keeps_order_and_marks_every_dropped_run(self):
        text = 'Birinci cümle uzundur. İki atlanır. Üç. Dört.'
        spans = sentence_spans(text)
        self.assertEqual([text[a:b] for a, b in spans], ['Birinci cümle uzundur.', 'İki atlanır.', 'Üç.', 'Dört.'])
        self.assertEqual(join_kept(text, spans, [1, 3]), '[...] İki atlanır. [...] Dört.')
        payload, quantities = md5_2()
        signals = precheck.structural_signals(payload, text, quantities)
        compress = passage_compressor(payload, text, signals, quantities)
        out = compress({'duty': {}, 'passage': text}, providers.payload_tokens({'duty': {}, 'passage': '[...] İki atlanır. [...]'}))
        self.assertEqual(out['passage'], '[...] İki atlanır. [...]')
        self.assertIsNone(compress({'duty': {}, 'passage': text}, 1))

    def test_a_short_passage_fits_and_is_sent_whole(self):
        payload, quantities = md5_2()
        small = Narrow(6144, 1024, fast=fast('PARTIAL', REGISTER_FIRST, missing=('deadline_1',)))
        _, results, _ = run(small, payload, quantities, [REGISTER])
        self.assertEqual(set(small.sent), {REGISTER})                     # the fast reading and the verifier: whole
        self.assertNotIn('WINDOW', [n['code'] for n in results['p1'].get('notes') or []])

    def test_a_truncated_verifier_answer_is_retried_once_on_the_wide_twin_in_adaptive_mode(self):
        payload, quantities = md5_2()
        cut = failure('Provider output incomplete or truncated', 'output_truncated')
        wide = Narrow(16384, 6144, verify=verdict(True, 'EXEMPTION_ADDED', KYC_SKIP_SPAN, 'NOT_APPLICABLE'))
        narrow = Narrow(8192, 4096, fast=fast('POSSIBLE_CONFLICT'), verify=cut)
        narrow.wide, narrow.ctx_mode, narrow.thinking = wide, 'adaptive', True
        relations, results, (coverage, _, _, flags) = run(narrow, payload, quantities, [KYC_SKIP])
        self.assertEqual((relations['p1'], coverage, flags), ('CONFLICTS', 'CONFLICT', []))
        # t6 (truncation ladder): the compact request on the 8k judge first; only its cut answer goes to the wide twin.
        self.assertEqual((len(narrow.kinds('verify')), len(wide.kinds('verify'))), (2, 1))
        self.assertTrue(wide.kinds('verify')[0]['uncached'])
        self.assertIn('compact_after_truncation', [n.get('retry', '') for n in results['p1']['notes']][0])
        self.assertIn('fallback_large_after_truncation', [n.get('retry', '') for n in results['p1']['notes']][1])
        # Fixed mode: the one retry stays on the 8k judge.
        narrow = Narrow(8192, 4096, fast=fast('POSSIBLE_CONFLICT'), verify=cut)
        narrow.wide, narrow.ctx_mode, narrow.thinking = Narrow(16384, 6144), 'fixed', True
        relations, results, _ = run(narrow, payload, quantities, [KYC_SKIP])
        self.assertEqual((relations['p1'], len(narrow.kinds('verify')), len(narrow.wide.kinds('verify'))), ('UNCLEAR', 2, 0))


if __name__ == '__main__':
    unittest.main()


class AnchorRuleTests(unittest.TestCase):
    """v0.19 smoke run (2026-09-24, C14 md. 26(2)): a contradiction or a support must name the duty's own act or object."""
    D26 = {'subject': 'Yükümlüler', 'modality': 'MUST_NOT', 'required_action': None, 'prohibited_action': 'basitleştirilmiş tedbirleri uygulayamazlar',
           'conditions': ['işlem nedeniyle aklama veya terörün finansmanı riskinin oluşabileceği durumlarda'], 'exceptions': [],
           'elements': [{'id': 'action', 'kind': 'action', 'text': 'basitleştirilmiş tedbirleri uygulayamazlar'}]}
    SKIP = ("Müşteri deneyimini hızlandırmak için, tek seferlik ve tutarı 15.000 TL'nin altında kalan havale işlemlerinde kimlik tespiti ve "
            "teyit adımı atlanır; müşteri yalnızca telefon numarasını doğrular.")

    def test_a_contradiction_about_another_measure_is_not_a_conflict_of_this_duty(self):
        from regchain.pilot.engine import settle_v19
        from regchain.pilot.schema import ConflictVerdict
        verdict = ConflictVerdict(conflict=True, contradiction_type='EXEMPTION_ADDED', contradiction_span=self.SKIP,
                                  regulation_requirement='no simplified measures in risky cases', policy_statement='skips identification',
                                  confidence='HIGH', relation_if_no_conflict='NOT_APPLICABLE', support_quote='', rationale='a skip is a simplified measure')
        notes = []
        judgement, _ = settle_v19(self.D26, None, verdict, None, 'STRUCTURAL_SIGNAL', notes, ['action'])
        self.assertEqual(judgement.relation, 'UNRELATED')
        self.assertIn('CONFLICT_NOT_ANCHORED', [n['code'] for n in notes])

    def test_a_contradiction_that_names_the_duty_act_still_stands(self):
        from regchain.pilot.engine import settle_v19
        from regchain.pilot.schema import ConflictVerdict
        duty = {'subject': 'Kimlik tespiti', 'modality': 'MUST', 'required_action': 'iş ilişkisi tesisinden veya işlem yapılmadan önce tamamlanır.',
                'prohibited_action': None, 'conditions': [], 'exceptions': [], 'elements': [{'id': 'action', 'kind': 'action', 'text': 'x'}]}
        verdict = ConflictVerdict(conflict=True, contradiction_type='EXEMPTION_ADDED', contradiction_span=self.SKIP,
                                  regulation_requirement='identify before', policy_statement='skips identification', confidence='HIGH',
                                  relation_if_no_conflict='NOT_APPLICABLE', support_quote='', rationale='skip')
        judgement, _ = settle_v19(duty, None, verdict, None, 'STRUCTURAL_SIGNAL', [], ['action'])
        self.assertEqual(judgement.relation, 'CONFLICTS')

    def test_an_off_topic_fast_support_goes_to_the_strong_model(self):
        from regchain.pilot.engine import escalation_of
        from regchain.pilot.schema import FastJudgement
        fast = FastJudgement(label='SUPPORTS', quote='Personel, şüpheli bulduğu işlemi aynı gün içinde uyum birimine iletir.',
                             covered_elements=['action'], missing_elements=[], reason='r')
        self.assertEqual(escalation_of(fast, [], 0.118, 3, None, self.D26), 'AMBIGUOUS_SUPPORT')
        on_topic = FastJudgement(label='SUPPORTS', quote='Riskli durumlarda basitleştirilmiş tedbir uygulanmaz.', covered_elements=['action'],
                                 missing_elements=[], reason='r')
        self.assertIsNone(escalation_of(on_topic, [], 0.9, 0, None, self.D26))


class ContradictionTypeSupportTests(unittest.TestCase):
    """v0.19 short round 2 (C03, 2026-09-25): a contradiction type must show in the wording of the sentence."""
    ASSOC = {'subject': 'Dernekler', 'modality': 'MUST', 'prohibited_action': None,
             'required_action': 'kimlik tespitinde; derneğin adı, amacı, kütük numarası ve vergi kimlik numarası bilgileri alınır',
             'conditions': [], 'exceptions': [], 'elements': [{'id': 'action', 'kind': 'action', 'text': 'x'}]}
    COMPANIES = ('Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde; tüzel kişinin unvanı, ticaret sicil numarası, vergi kimlik '
                 'numarası, faaliyet konusu, açık adresi ve temsile yetkili kişinin kimlik bilgileri alınır ve ticaret sicil belgeleri '
                 'üzerinden teyit edilir.')

    def verdict(self, ctype, span):
        from regchain.pilot.schema import ConflictVerdict
        return ConflictVerdict(conflict=True, contradiction_type=ctype, contradiction_span=span, regulation_requirement='r',
                               policy_statement='p', confidence='HIGH', relation_if_no_conflict='NOT_APPLICABLE', support_quote='',
                               rationale='another group', covered_elements=['action'], missing_elements=[])

    def test_a_rule_for_another_group_is_not_a_narrowed_scope(self):
        from regchain.pilot.engine import settle_v19
        notes = []
        judgement, _ = settle_v19(self.ASSOC, None, self.verdict('SCOPE_NARROWED', self.COMPANIES), None, 'CONFIRM_COVERS', notes,
                                  ['action'], self.COMPANIES)
        self.assertEqual(judgement.relation, 'UNRELATED')
        self.assertIn('CONFLICT_TYPE_UNSUPPORTED', [n['code'] for n in notes])

    def test_explicit_limiting_or_excepting_wording_keeps_the_conflict(self):
        from regchain.pilot.engine import settle_v19
        text = 'Kimlik tespiti yalnızca ticaret siciline kayıtlı tüzel kişiler için yapılır; dernekler bu uygulamanın dışındadır.'
        judgement, _ = settle_v19(self.ASSOC, None, self.verdict('SCOPE_NARROWED', text), None, 'STRUCTURAL_SIGNAL', [], ['action'], text)
        self.assertEqual(judgement.relation, 'CONFLICTS')

    def test_the_pointing_back_exemption_sentence_is_supported_by_its_predecessor(self):
        from regchain.pilot.conflict import type_supported
        passage = ("Tutarı 15.000 TL'nin altında kalan havale işlemlerinde kimlik tespiti ve teyit adımı atlanır. "
                   "Bu istisna kampanya dönemlerinde bütün tutarlar için uygulanabilir.")
        self.assertTrue(type_supported({}, 'EXEMPTION_ADDED', 'Bu istisna kampanya dönemlerinde bütün tutarlar için uygulanabilir.', passage))

    def test_a_deadline_claim_without_any_period_wording_is_a_partial_statement(self):
        from regchain.pilot.engine import settle_v19
        duty = {'subject': 'Şüpheli işlemler', 'modality': 'MUST', 'prohibited_action': None,
                'required_action': 'en geç on iş günü içinde Başkanlığa bildirilir', 'conditions': [], 'exceptions': [],
                'elements': [{'id': 'action', 'kind': 'action', 'text': 'x'}, {'id': 'deadline_1', 'kind': 'deadline', 'text': 'en geç on iş günü'}]}
        text = "Şüpheli görülen işlemler uyum birimi aracılığıyla MASAK'a bildirilir."
        notes = []
        judgement, _ = settle_v19(duty, None, self.verdict('DEADLINE_MISMATCH', text), None, 'FAST_POSSIBLE_CONFLICT', notes, ['action', 'deadline_1'], text)
        self.assertEqual(judgement.relation, 'PARTIAL')
        self.assertIn('CONFLICT_TYPE_UNSUPPORTED', [n['code'] for n in notes])

    def test_real_conflict_wordings_stay_supported(self):
        from regchain.pilot.conflict import type_supported
        self.assertTrue(type_supported({}, 'DEADLINE_MISMATCH', 'uygun görülenler o ay sonunda MASAK\'a topluca bildirilir.'))
        self.assertTrue(type_supported({}, 'DEADLINE_MISMATCH', 'kayıtlar işlem tarihinden itibaren beş yıl saklanır'))
        self.assertTrue(type_supported({}, 'THRESHOLD_MISMATCH', "yalnızca tutarı 25.000 TL'yi aşan işlemler bildirilir"))
        self.assertTrue(type_supported({}, 'THRESHOLD_MISMATCH', 'Charges are set by reference to our revenue target and may exceed the cost.'))
        self.assertTrue(type_supported({}, 'REQUIREMENT_REMOVED', 'Bu bilgiler mahkeme kararı bulunmadıkça hiçbir kamu kurumuna verilmez.'))


class SettlementEdgeTests(unittest.TestCase):
    """Short round 2 (2026-09-25): C11 md. 46(1) and C02 md. 6(1) settled from the verifier's own non-conflict answer."""

    def test_an_unsupported_scope_claim_falls_back_to_the_verifiers_reading(self):
        from regchain.pilot.engine import settle_v19
        from regchain.pilot.schema import ConflictVerdict
        duty = {'subject': 'Yükümlüler', 'modality': 'MUST', 'prohibited_action': None, 'required_action': 'belgeleri sekiz yıl muhafaza etmek',
                'conditions': [], 'exceptions': [], 'elements': [{'id': 'action', 'kind': 'action', 'text': 'x'}]}
        text = 'Müşteri dosyaları, sözleşmeler ve işlem kayıtları 8 yıl boyunca saklanır.'
        verdict = ConflictVerdict(conflict=True, contradiction_type='SCOPE_NARROWED', contradiction_span=text, regulation_requirement='r',
                                  policy_statement='p', confidence='HIGH', relation_if_no_conflict='PARTIAL', support_quote=text,
                                  rationale='lists specific documents', covered_elements=['action'], missing_elements=[])
        # Short round 3: the rejected claim was the PARTIAL's only gap (every critical element covered, none missing):
        # the verifier's reading is the SUPPORTS it describes (C11 md. 46(1), label COVERS_TEXT).
        judgement, _ = settle_v19(duty, None, verdict, None, 'CONFIRM_COVERS', [], ['action'], text)
        self.assertEqual((judgement.relation, judgement.quote), ('SUPPORTS', text))
        # With an element the verifier lists missing, the fallback stays PARTIAL.
        duty = {**duty, 'elements': [*duty['elements'], {'id': 'deadline_1', 'kind': 'deadline', 'text': 'sekiz yıl'}]}
        verdict = verdict.model_copy(update={'missing_elements': ['deadline_1']})
        judgement, _ = settle_v19(duty, None, verdict, None, 'CONFIRM_COVERS', [], ['action', 'deadline_1'], text)
        self.assertEqual((judgement.relation, judgement.quote), ('PARTIAL', text))

    def test_conflict_true_with_type_none_is_read_as_its_non_conflict_answer(self):
        import json as _json
        from regchain.pilot.engine import parse_verdict
        text = 'Gerçek kişilerin kimlik tespitinde; ilgilinin adı, soyadı ve adresi alınır.'
        raw = _json.dumps({'conflict': True, 'contradiction_type': 'NONE', 'contradiction_span': '', 'regulation_requirement': 'r',
                           'policy_statement': 'p', 'confidence': 'MEDIUM', 'relation_if_no_conflict': 'PARTIAL', 'support_quote': text,
                           'rationale': 'omits e-mail', 'covered_elements': ['action'], 'missing_elements': []})
        verdict = parse_verdict(raw, text, ['action'])
        self.assertFalse(verdict.conflict)
        self.assertEqual((verdict.relation_if_no_conflict, verdict.support_quote), ('PARTIAL', text))

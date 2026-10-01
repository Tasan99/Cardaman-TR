"""v0.19 t6 review 1 (25 September 2026): the confirmed findings of the t6 review, each pinned by the general invariant fixed.

Measured by the review's probes against the live t6 tree and the frozen t5 snapshot:
- the t6 conflict gate rejected true contradictions t5 kept when the duty's act was worded outside its word lists: an unlisted
  synonym ("teslim edilmez", "reddedilir", "yanıtsız bırakılır"), a negative mood other than the aorist ("bildirilmeyecektir",
  "saklanmamaktadır"), the passive of a short verb ("yazar" / "yazılmaz", "alır" / "alınmaz"), a softened noun ("teyit" /
  "teyidi aranmaz"), a permission inside the duty's own condition, a dative modifier read as the addressee. OBJECT now needs
  positive evidence; the act is read as a morphological class;
- the timing reader missed ordinary wordings ("gecikmeye mahal vermeden", "kurulmadan tamamlanır") and read an unrelated
  "-madan ... -maz" clause anywhere in the passage;
- the t6 fixes changed deterministic outputs under unchanged version hashes;
- a cut-mark quote of one token was accepted; a repair after the 16k rung went back to the 8k window; a cut after a TIMEOUT
  retry had no VERIFIER_TRUNCATED note; the run metrics did not list the compact retry.
Every duty and passage is synthetic or a Tedbirler fixture duty; every model is a scripted fake. No case id, article number
or fixture sentence is in the engine.
"""
import unittest

from regchain.evaluation import metrics
from regchain.evidence import digest
from regchain.extraction.providers import AI_WINDOW, failure
from regchain.extraction.service import PROMPT_HASH as EXTRACTION_PROMPT_HASH
from regchain.pilot import conflict as precheck
from regchain.pilot import engine
from regchain.pilot.engine import QUOTE_FULL_SOURCE_OK, QUOTE_NOT_IN_SOURCE, VERIFIER_TRUNCATED, check_scope_quote, timing_match
from regchain.pilot.impact import reuse_blockers
from test_v019_conflict_pipeline import md5_1, md26, md28, md46
from test_v019_t5_engine import disclosure, plain
from test_v019_t6_conflict_gate import GIVE, GIVE_NONE, MASAK, RESERVATION, claim
from test_v019_t6_partial import BEFORE_ACT, BEFORE_UNIT, BOTH, V19, fast, lists, made, now
from test_v019_t6_partial import gate as partial_gate
from test_v019_t6_quote import MD24_OPENING, md
from test_v019_t6_truncation import Isolated, STRICTER, cut, pair, relief, supports
from test_v019_t6_truncation import gate as ladder_gate

UNREAD = precheck.GATE_UNREAD_ACT
# t5 values recorded by the review (t6/review/provenance.py): identical in the t5 and the pre-review t6 tree.
T5_EXTRACTION_PROMPT_HASH = 'fdd5dc0b1d25ce3bdfcd30e7c6df27c66c927c983b06b72c674a763dc4b427a7'
T5_PILOT_PROMPT_HASH = 'f252b099b1274fba754bce9c7da71fd102f47757a2b6928160ef95ac3a2674a3'
EARLIER_V19_PREFIXES = ('28e03f9b', '175afc2d')                  # t5, and t6 before this review
REPORT = plain('şüpheli işlemleri Başkanlığa bildirir')
IDENT = plain('müşterinin kimliğini tespit eder')
VERIFY = plain('müşterinin adres bilgisini resmi belge ile teyit eder')
MONITOR = plain('yüksek riskli müşterilerin işlemlerini sürekli olarak izler')
REFUSE = plain('kimlik tespiti yapılamayan durumlarda iş ilişkisi kurma talebini reddeder')


def gate_code(duty, text, quantities=()):
    return precheck.conflict_gate(duty, text, text, quantities)[0]


def stands(test, payload, text, kind='REQUIRED_ACTION_FORBIDDEN', quantities=(), **extra):
    """The verifier claims `kind` on `text`: the claim must stand as a contradiction (t5 kept every one of these)."""
    relation, coverage, found, _ = claim(payload, quantities, text, kind, **extra)
    test.assertEqual((relation, coverage), ('CONFLICTS', 'CONFLICT'), (text, found))
    test.assertIn(precheck.GATE_PASSED, found)
    test.assertFalse(set(precheck.GATE_CODES) & set(found), found)
    return found


class UnreadActTests(unittest.TestCase):
    """OVF-1, F1: OBJECT rejects a claim only on positive evidence, never for want of a listed word."""

    PARAPHRASES = ("MASAK'ın talep ettiği belgeler mahkeme kararı olmadan teslim edilmez.",
                   "MASAK'tan gelen bilgi talepleri, müşteri gizliliği nedeniyle reddedilir.",
                   'Kamu kurumlarının bilgi talepleri yanıtsız bırakılır.',
                   'Denetim elemanlarına müşteri dosyaları gösterilmez.',
                   'Kamu kurumlarının bilgi ve belge talepleri karşılanmaz.')

    def test_refusing_the_authority_in_other_words_stays_a_contradiction(self):
        for text in self.PARAPHRASES:
            with self.subTest(text=text):
                self.assertEqual(gate_code(GIVE, text), '')
                self.assertIn(UNREAD, stands(self, GIVE, text))
        # A family verb in the clause ("sunulmadıkça") is read as the act, as before.
        stands(self, GIVE, "MASAK'ın bilgi ve belge talepleri, mahkeme kararı sunulmadıkça karşılanmaz.")

    def test_the_recorded_false_conflicts_stay_rejected(self):
        self.assertEqual(gate_code(disclosure(), MASAK), precheck.GATE_OBJECT)            # a description: no effect at all
        code, why = precheck.conflict_gate(GIVE, RESERVATION, RESERVATION)
        self.assertEqual(code, precheck.GATE_OBJECT)                                       # asking customers: another party
        self.assertIn('müşterilerimizden', why)
        relation, coverage, found, _ = claim(GIVE, (), RESERVATION, 'EXEMPTION_ADDED')
        self.assertEqual((relation, coverage), ('UNRELATED', 'NO_EVIDENCE'))
        self.assertNotIn(UNREAD, found)

    def test_a_clause_without_effect_another_act_or_a_real_anchor_is_still_rejected(self):
        self.assertEqual(gate_code(GIVE, 'Bilgi talepleri için ayrı bir kayıt defteri tutulur.'), precheck.GATE_OBJECT)
        payload, quantities = md46()
        skip = 'Tek seferlik işlemlerde kimlik tespiti ve teyit adımı atlanır.'           # another act (identifying) as main verb
        self.assertEqual(gate_code(payload, skip, quantities), precheck.GATE_OBJECT)
        # Limiting wording and one shared qualifier ("riskli") are no anchor on monitoring high-risk customers.
        self.assertEqual(gate_code(MONITOR, 'Düşük riskli müşterilerin işlemleri yalnızca yılda bir gözden geçirilir.'), precheck.GATE_OBJECT)
        # A duty whose own act refuses: a negation without that act may say the same thing.
        self.assertEqual(gate_code(REFUSE, 'Kimlik tespiti yapılamayan durumlarda iş ilişkisi kurulmaz.'), precheck.GATE_OBJECT)

    def test_a_rejected_contradiction_never_becomes_a_cover(self):
        payload = plain('havale mesajına gönderenin adını ve adresini yazar')
        told = 'Havale mesajına alıcının adı yazılır.'
        denied = 'Havale mesajına gönderenin adı ve adresi yazılmaz.'
        relation, coverage, found, _ = claim(payload, (), told + ' ' + denied, 'DIRECT_OPPOSITE', span=denied, relation='SUPPORTS',
                                             support=told)
        self.assertEqual((relation, coverage), ('CONFLICTS', 'CONFLICT'), found)


class ShortRootTests(unittest.TestCase):
    """P1-ACTIVE-PASSIVE, F1: a short verb is read by its root in every voice, with the duty's object."""

    PAIRS = (('yazar', 'yazılmaz'), ('ekler', 'eklenmez'), ('sorar', 'sorulmaz'), ('öder', 'ödenmez'), ('açar', 'açılmaz'),
             ('siler', 'silinmez'), ('ister', 'istenmez'), ('tutar', 'tutulmaz'), ('izler', 'izlenmez'), ('korur', 'korunmaz'),
             ('okur', 'okunmaz'), ('alır', 'alınmaz'), ('seçer', 'seçilmez'), ('satar', 'satılmaz'))

    def test_an_active_aorist_duty_against_its_negated_passive(self):
        for verb, negated in self.PAIRS:
            with self.subTest(verb=verb):
                payload, text = plain(f'müşteri bilgilerini {verb}'), f'Müşteri bilgileri {negated}.'
                self.assertEqual(gate_code(payload, text), '')
        for verb, negated in (('yazar', 'yazılmaz'), ('tutar', 'tutulmaz'), ('alır', 'alınmaz')):
            stands(self, plain(f'müşteri bilgilerini {verb}'), f'Müşteri bilgileri {negated}.', 'DIRECT_OPPOSITE')
        for verb, noun in (('yazmak', 'yazılmasına gerek görülmez'), ('eklemek', 'eklenmez'), ('sormak', 'sorulmaz')):
            with self.subTest(verb=verb):
                self.assertEqual(gate_code(plain(f'müşteri bilgilerini {verb}'), f'Müşteri bilgileri {noun}.'), '')

    def test_the_root_is_read_as_a_verb_form_only(self):
        form = precheck.root_form(precheck.verb_roots(plain('müşteri bilgilerini yazar')))
        for word in ('yazılmaz', 'yazılması', 'yazmaz', 'yazılır', 'yazılabilir'):
            self.assertTrue(form.match(word), word)
        for word in ('yazılı', 'yazılım', 'yazı'):
            self.assertIsNone(form.match(word), word)
        form = precheck.root_form(precheck.verb_roots(plain('müşterinin kimlik bilgilerini üçüncü taraftan derhal alır')))
        for word in ('alınmaz', 'alınır', 'alınmasına', 'almaz'):
            self.assertTrue(form.match(word), word)
        for word in ('altında', 'altındaki', 'alan', 'alacak', 'alım'):
            self.assertIsNone(form.match(word), word)

    def test_a_short_root_names_the_act_only_with_the_duty_s_object(self):
        taking = plain('kimliğe ilişkin bilgileri almak')
        self.assertEqual(gate_code(taking, 'Müşterinin kimlik bilgileri alınmaz.'), '')
        self.assertEqual(gate_code(taking, 'İş ilişkisinin amacı hakkında not alınması operasyon ekibinin takdirindedir.'), precheck.GATE_OBJECT)

    def test_a_root_that_is_a_known_act_in_another_form(self):
        self.assertIn('@monitor', precheck.root_acts(MONITOR))
        self.assertEqual(gate_code(MONITOR, 'Yüksek riskli müşterilerin işlemlerinin takibi zorunlu değildir.'), '')


class NegativeMoodTests(unittest.TestCase):
    """P1-NEG-TENSE: the negative -mA before any tense or mood, and the negated verbal noun under a rule word."""

    CASES = ((REPORT, ('Şüpheli işlemler Başkanlığa bildirilmeyecektir.', 'Şüpheli işlemler Başkanlığa bildirilmemektedir.',
                       'Şüpheli işlemler Başkanlığa bildirilmiyor.', 'Şüpheli işlemlerin Başkanlığa bildirilmemesi esastır.',
                       'Şüpheli işlemler için Başkanlığa bildirim yapılmayacaktır.', 'Şüpheli işlemler Başkanlığa bildirilmemelidir.')),
             (IDENT, ('Müşterinin kimliği tespit edilmeyecektir.', 'Müşterinin kimliği tespit edilmemektedir.')),
             (VERIFY, ('Müşterinin adres bilgisi teyit edilmeyecektir.',)),
             (GIVE, ('Başkanlık ve denetim elemanlarınca istenen bilgi ve belgeler verilmeyecektir.',)),
             (REFUSE, ('Kimlik tespiti yapılamayan durumlarda iş ilişkisi kurma talepleri reddedilmeyecektir.',)))

    def test_every_negative_mood_of_the_duty_s_act_contradicts_a_requirement(self):
        for payload, texts in self.CASES:
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(gate_code(payload, text), '')
                    stands(self, payload, text)

    def test_the_keeping_duty_and_its_negative_moods(self):
        payload, quantities = md46()
        for text in ('İşlem belgeleri saklanmayacaktır.', 'İşlem belgeleri saklanmamaktadır.'):
            with self.subTest(text=text):
                self.assertEqual(gate_code(payload, text, quantities), '')

    def test_the_same_direction_stays_no_contradiction(self):
        text = 'Müşterilere, haklarında bildirim yapıldığına dair hiçbir bilgi verilmeyecektir.'
        self.assertEqual(gate_code(GIVE_NONE, text), precheck.GATE_EFFECT)
        self.assertFalse(precheck.same_direction(REPORT, 'Şüpheli işlemler Başkanlığa bildirilmeyecektir.'))
        # A condition is no rule: "bildirilmemesi halinde" (in case it is not reported) reads nothing.
        self.assertEqual(precheck._polarity_at(['bildirilmemesi', 'halinde'], 0), 'REQUIRED')
        self.assertEqual(precheck._polarity_at(['bildirilmemesi', 'esastır'], 0), 'PROHIBITED')


class SofteningTests(unittest.TestCase):
    def test_the_softened_noun_of_the_duty_s_act(self):
        self.assertEqual(precheck._gate_token('teyidi'), '@identify')
        self.assertEqual(precheck._gate_token('teyidine'), '@identify')
        for text in ('Adres teyidi aranmaz.', 'Müşteri adres bilgisinin teyidi yapılmaz.', 'Adres bilgisinin teyidine gerek görülmez.',
                     'Adres bilgisi teyit edilmez.'):
            with self.subTest(text=text):
                self.assertEqual(gate_code(VERIFY, text), '')
        stands(self, VERIFY, 'Adres teyidi aranmaz.', 'REQUIREMENT_REMOVED')


class OwnConditionTests(unittest.TestCase):
    """P1-OWN-CONDITION: a permission or waiver inside the duty's own condition is the contradiction; its exception is not."""

    def test_permitting_or_waiving_the_act_in_the_duty_s_own_condition(self):
        payload, quantities = md26()
        text = 'Aklama veya terörün finansmanı riskinin bulunduğu durumlarda da basitleştirilmiş tedbirler uygulanabilir.'
        self.assertEqual(gate_code(payload, text, quantities), '')
        payload, quantities = md5_1()
        self.assertEqual(gate_code(payload, 'Şüpheli işlem bildirimini gerektiren durumlarda kimlik tespiti aranmaz.', quantities), '')

    def test_the_duty_s_own_exception_restated_stays_no_contradiction(self):
        restated = ('Bildirim bilgisi yalnızca yükümlülük denetimi ile görevlendirilen denetim elemanlarına ve yargılama sırasında '
                    'mahkemelere açıklanabilir.')
        self.assertEqual(gate_code(disclosure(), restated), precheck.GATE_SCOPE)
        # A condition worded as an exception ("... olmadıkça") is the duty's exception.
        action = 'müşteri bilgilerini üçüncü kişilerle paylaşamazlar'
        consent = {'subject': 'Yükümlüler', 'modality': 'MUST_NOT', 'required_action': None, 'prohibited_action': action,
                   'conditions': ['müşterinin yazılı onayı olmadıkça'], 'exceptions': [],
                   'elements': [{'id': 'action', 'kind': 'action', 'text': action}, {'id': 'subject', 'kind': 'subject', 'text': 'Yükümlüler'},
                                {'id': 'prohibition', 'kind': 'prohibition', 'text': 'paylaşamazlar'}]}
        self.assertEqual(gate_code(consent, 'Müşterinin yazılı onayı ile paylaşılabilir.'), precheck.GATE_SCOPE)
        self.assertEqual(gate_code(consent, 'Müşteri bilgileri iş ortaklarıyla paylaşılabilir.'), '')


class AddresseeTests(unittest.TestCase):
    def test_a_dative_that_modifies_a_noun_is_no_addressee(self):
        payload, quantities = md28()
        for text in ('Kurumsal müşterilere ait şüpheli işlemler bildirim yükümlülüğünün istisnasıdır.',
                     'Kurumsal müşterilere yapılan transferlerde şüpheli işlem bildirimi aranmaz.'):
            with self.subTest(text=text):
                self.assertEqual(gate_code(payload, text, quantities), '')
        customer = 'Şüpheli işlem bildirimi yapıldığı, müşteriye en geç otuz gün içinde bildirilir.'
        self.assertEqual(gate_code(payload, customer, quantities), precheck.GATE_ACTOR)


class EnglishNegationTests(unittest.TestCase):
    def test_the_turkish_noun_not_is_no_negation(self):
        taken = plain('iş ilişkisinin amacı ve mahiyeti hakkında bilgi alınır', subject='Sürekli iş ilişkisi tesisinde')
        noted = 'İş ilişkisinin amacı ve mahiyeti hakkında müşteriyle görüşülür ve not alınır.'
        self.assertEqual(gate_code(taken, noted), precheck.GATE_EFFECT)
        english = plain("must obtain the customer's identity documents", subject='The firm')
        self.assertEqual(gate_code(english, 'Identity documents are not obtained from walk-in customers.'), '')


class TimingWordingTests(unittest.TestCase):
    """P2-TIMING, OVF-2: ordinary wordings of a timing are read; an order is read in the clause of the duty's own act."""

    def test_immediate_wordings(self):
        payload, quantities = now()
        stated = ('geciktirilmeksizin', 'gecikmeye mahal vermeden', 'gecikmeye mahal verilmeksizin', 'vakit geçirmeden', 'acilen',
                  'ilk fırsatta', 'aynı anda')
        for words in stated:
            text = f'Aracı kuruluştan alınan müşteri bilgileri {words} müşteri dosyasına eklenir.'
            with self.subTest(text=text):
                self.assertEqual(timing_match(quantities, text, payload), {'deadline_1': 'STATED'})
        for text in ('Aracı kuruluştan bilgiler alındığı anda müşteri dosyasına eklenir.',
                     'Aracı kuruluştan bilgiler alınır alınmaz müşteri dosyasına eklenir.'):
            self.assertEqual(timing_match(quantities, text, payload), {'deadline_1': 'STATED'}, text)
        self.assertEqual(timing_match(quantities, 'Aracı kuruluştan alınan müşteri bilgileri zamanında müşteri dosyasına eklenir.', payload),
                         {'deadline_1': 'NOT_STATED'})

    def test_before_orders_of_the_duty_s_own_act(self):
        payload, quantities = made(BEFORE_UNIT, 'Şirket', BEFORE_ACT)
        for text, want in (('Müşterinin adres bilgisi hesap açılmadan doğrulanır.', 'STATED'),
                           ('Hesap açılmadan müşterinin adres bilgisi doğrulanır.', 'STATED'),
                           ('Müşterinin adres bilgisi doğrulanmadan hesap açılmaz.', 'STATED'),
                           ('Müşterinin adres bilgisi doğrulanmaksızın hesap açılmaz.', 'STATED'),
                           ('Hesap, müşterinin adres bilgisi doğrulandıktan sonra açılır.', 'STATED'),
                           ('Müşterinin adres bilgisi doğrulanır; müşteri onayı alınmadan veriler üçüncü kişilerle paylaşılmaz.', 'NOT_STATED'),
                           ('Müşteri bilgileri, mahkeme kararı bulunmadıkça hiçbir kamu kurumuna verilmez; adres bilgisi doğrulanır.', 'NOT_STATED'),
                           ('Hesap açıldıktan sonra müşterinin adres bilgisi doğrulanır.', 'NOT_STATED'),
                           ('Müşterinin adres bilgisi doğrulanmadan hesap açılır.', 'NOT_STATED'),
                           ('Hesap açılmadan müşteri ile görüşülür.', 'NOT_STATED')):
            with self.subTest(text=text):
                self.assertEqual(timing_match(quantities, text, payload), {'deadline_1': want})
        # Without a duty the reader keeps its t6 reading: the negated order, anywhere.
        self.assertEqual(engine.stated_timings('Adres doğrulanmadan, hesap açılmaz.'), {'BEFORE'})

    def test_a_covering_passage_in_such_words_stays_covers_text(self):
        cases = ((now(), 'Aracı kuruluştan alınan müşteri bilgileri gecikmeye mahal vermeden müşteri dosyasına eklenir.'),
                 (made(BEFORE_UNIT, 'Şirket', BEFORE_ACT), 'Müşterinin adres bilgisi hesap açılmadan doğrulanır.'),
                 (made(BEFORE_UNIT, 'Şirket', BEFORE_ACT), 'Hesap, müşterinin adres bilgisi doğrulandıktan sonra açılır.'))
        for (payload, quantities), text in cases:
            with self.subTest(text=text):
                judge = V19(fast=fast('SUPPORTS', text, covered=BOTH), verify=lists('SUPPORTS', text, BOTH))
                relations, results, (coverage, _, _, flags) = partial_gate(judge, payload, quantities, [text])
                self.assertEqual((relations['p1'], results['p1']['timing_match'], coverage, flags),
                                 ('SUPPORTS', {'deadline_1': 'STATED'}, 'COVERS_TEXT', []))


class ProvenanceTests(unittest.TestCase):
    """PROV-1, PROV-2: the deterministic changes move the hashes a packet, a manifest and an extraction run carry."""

    def test_the_hashes_moved_and_a_t5_packet_is_not_reused(self):
        self.assertNotEqual(EXTRACTION_PROMPT_HASH, T5_EXTRACTION_PROMPT_HASH)
        self.assertNotEqual(engine.PROMPT_HASH, T5_PILOT_PROMPT_HASH)
        self.assertFalse(engine.V19_PROMPT_HASH.startswith(EARLIER_V19_PREFIXES))
        for pipeline in ('v18', 'v19'):
            settings = engine.pipeline_settings(coverage_pipeline=pipeline, relevance_screen='on', applicability_clear_match='rule')
            previous = {'company_hash': 'c', 'policies': [{'raw_hash': 'r'}], 'extraction_prompt_hash': T5_EXTRACTION_PROMPT_HASH,
                        'pilot_prompt_hash': T5_PILOT_PROMPT_HASH, 'pipeline_settings': settings}
            with self.subTest(pipeline=pipeline):
                self.assertEqual(reuse_blockers(previous, 'c', [{'raw_hash': 'r'}], EXTRACTION_PROMPT_HASH, engine.PROMPT_HASH),
                                 ['prompts or judgement version changed'])

    def test_the_coverage_hash_names_the_module_versions(self):
        parts = engine.V19_HASH_PARTS
        self.assertEqual((parts['precheck'], parts['covers_gate']), (precheck.GATE_VERSION, engine.COVERS_GATE_VERSION))
        self.assertNotIn(parts['precheck'], ('conflict-v4-roles-polarity',))
        self.assertNotIn(parts['covers_gate'], ('v3-verified-elements',))
        self.assertEqual(engine.V19_PROMPT_HASH, digest(parts))
        self.assertNotEqual(digest({**parts, 'precheck': parts['precheck'] + '-next'}), engine.V19_PROMPT_HASH)
        self.assertIn('scope-quote', engine.JUDGEMENT_RULES_VERSION)


class CutMarkTests(unittest.TestCase):
    """F3: a quote left after its cut mark names a span only with content, on a word boundary."""

    def test_a_token_closed_with_a_cut_mark_is_no_scope_evidence(self):
        sources = {'p0': md(24)['text']}
        for quote in ('Onbe...', 've...', '(1)...', 'a)...', '(1) Onbeşbin TL veya üze...'):
            with self.subTest(quote=quote):
                self.assertEqual(check_scope_quote('p0', quote, sources)[1], QUOTE_NOT_IN_SOURCE)
        self.assertEqual(check_scope_quote('p0', MD24_OPENING, sources)[1:], (QUOTE_FULL_SOURCE_OK, 'cut_mark'))
        self.assertEqual(check_scope_quote('p0', '(1)', sources)[1:], (QUOTE_FULL_SOURCE_OK, 'exact'))      # as in t5


class LadderEdgeTests(Isolated):
    """F4: after a ladder rung the repair goes to the instance that answered; a cut after a TIMEOUT retry is named."""

    def test_a_repair_after_the_wide_rung_stays_on_the_wide_window(self):
        payload, quantities = relief()
        judge = pair([cut(), cut()], ['not a verdict', supports(payload, STRICTER)],
                     fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=[e['id'] for e in payload['elements']]))
        relations, results, (coverage, _, _, _) = ladder_gate(judge, payload, quantities, [STRICTER])
        self.assertEqual(len(judge.asked), 2)                                   # the full request and the compact one, on 8k
        self.assertEqual([c['num_ctx'] for c in judge.wide.asked], [16384, 16384])
        self.assertIn('validation_feedback', judge.wide.asked[1]['payload'])
        self.assertEqual([c['window_action'] for c in judge.wide.asked], ['fallback_large_after_truncation'] * 2)
        self.assertEqual((relations['p1'], coverage), ('SUPPORTS', 'COVERS_TEXT'))

    def test_a_cut_after_the_timeout_retry_is_named(self):
        payload, quantities = relief()
        judge = pair([failure('slow', 'transport', 'TIMEOUT'), cut()], [supports(payload, STRICTER)],
                     fast_answer=fast('POSSIBLE_SUPPORT', STRICTER, covered=[e['id'] for e in payload['elements']]))
        _, results, (coverage, reason, _, flags) = ladder_gate(judge, payload, quantities, [STRICTER])
        self.assertEqual((len(judge.asked), judge.wide.asked), (2, []))
        codes = [n['code'] for n in results['p1']['notes']]
        self.assertIn(VERIFIER_TRUNCATED, codes)
        self.assertEqual((coverage, flags), ('UNKNOWN', ['COVERS_UNCONFIRMED', 'VERIFIER_FAILED']))


class WindowMetricsTests(unittest.TestCase):
    def test_the_compact_retry_is_a_known_window_action_and_no_fallback(self):
        actions = metrics.WINDOW_ACTIONS
        self.assertLess(actions.index('compact_after_truncation'), actions.index('fallback_large_after_truncation'))
        self.assertNotIn('compact_after_truncation', metrics.FALLBACK_ACTIONS)


if __name__ == '__main__':
    unittest.main()

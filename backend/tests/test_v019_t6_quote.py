"""v0.19 t6 (25 September 2026): the applicability quote is checked against the full text of the source it names.

Measured on the recorded answers of the v019t3-t5 runs: every scope quote refused as "does not occur in the supplied
source_id" that decided a row (Tedbirler md. 24 and md. 31, Kanun 5549 md. 7; 27 quotes in 11 refused answers) was an
exact copy of the opening of a clause of the very source it named, closed with "...". source_span reads "..." only
between two fragments, so the closing mark stayed glued to the last word and every duty of md. 24 ended UNKNOWN. The
quote was validated against the full provision all along: the compressed prompt drops sibling duties and scope
articles, never the provision. The model answers here are scripted; the regulation texts are the retained snapshots.
"""
import json
import unittest
from pathlib import Path

from regchain.evidence import canonical_bytes
from regchain.extraction.providers import CLIP_MARKER, ContextBudgetError, fits_budget
from regchain.pilot.engine import (FULL_SOURCE_UNAVAILABLE, QUOTE_FULL_SOURCE_OK, QUOTE_NOT_IN_SOURCE, QUOTE_OTHER_SOURCE, QUOTE_REASON_CODES,
                                   QUOTE_UNKNOWN_SOURCE, check_scope_quote, judge_scope, quote_span)
from regchain.pilot.schema import QUOTE_LIMIT, Company
from regchain.pilot.sources import load_sources

REPO = Path(__file__).resolve().parents[2]
TEDBIRLER = REPO / 'evaluation' / 'fixtures' / 'regulations' / 'tedbirler-200713012'
KANUN = REPO / 'evaluation' / 'independent' / 'fixtures' / 'regulations' / 'kanun-5549'
_LOADED = {}
# The quotes of the recorded md. 24 answer (v019t5 micro, I05; twice, both refused).
MD24_OPENING = ('(1) Onbeşbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında, gönderenin; a) Adı ve soyadına...')
MD1_OPENING = ('(1) Bu Yönetmeliğin amacı, 11/10/2006 tarihli ve 5549 sayılı Suç Gelirlerinin Aklanmasının Önlenmesi Hakkında Kanunun '
               'uygulanmasına yönelik olarak...')


def article(fixture, label):
    if fixture not in _LOADED:
        _LOADED[fixture] = {s['printed_label']: s for s in load_sources(fixture)[1]}
    return _LOADED[fixture][label]


def md(number):
    return article(TEDBIRLER, f'Yönetmelik 200713012 md. {number}')


def bank():
    return Company(id='t6-bank', name='Deneme Bankası A.Ş. (sentetik)', version='1', synthetic=True, jurisdictions=['TR'],
                   activities=['mevduat bankacılığı'], licences=['BDDK mevduat bankası faaliyet izni'], products=['havale'],
                   customer_types=['gerçek kişiler'], description='Sentetik mevduat bankası.')


DUTY = {'subject': 'Onbeşbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında, gönderenin', 'modality': 'MUST',
        'required_action': 'yer verilmesi zorunlu olup bu bilgilerin doğruluğu ayrıca teyit edilir.', 'prohibited_action': None,
        'conditions': [], 'exceptions': []}
BANKS = [{'company_fact': 'BDDK mevduat bankası faaliyet izni', 'regulatory_condition': 'a) Bankalar.', 'match': 'YES'}]


def answer(*quotes, applicability='APPLIES', basis=BANKS):
    return {'applicability': applicability, 'company_fact_keys': ['licences'],
            'scope_evidence': [{'source_id': sid, 'quote': quote} for sid, quote in quotes], 'basis': list(basis),
            'applicability_reason': 'Banka, md. 4 a) bendinde sayılan yükümlüdür.', 'missing_information': []}


class Judge:
    """Answers the applicability question from a script, one answer per call, and keeps every payload it was sent.
    With `window` it refuses a request that does not fit, as a fixed-window provider does."""
    name = 'scripted'
    # The engine requires an explicit local origin even for scripted model doubles.
    base_url = 'http://localhost:11434'

    def __init__(self, *answers, window=None):
        self.answers, self.sent = list(answers), []
        if window:
            self.num_predict, self.num_ctx = window

    def _chat(self, prompt, payload, schema):
        if getattr(self, 'num_ctx', None) and not fits_budget(prompt, payload, schema, self.num_predict, self.num_ctx):
            raise ContextBudgetError('CONTEXT_BUDGET_EXCEEDED: prompt and reserved output exceed admission budget')
        self.sent.append(payload)
        return json.dumps(self.answers.pop(0), ensure_ascii=False)


def judge(provider, section=None, scopes=None):
    scopes = {'s1': md(1), 's2': md(4)} if scopes is None else scopes
    return judge_scope(provider, bank(), section or md(24), DUTY, scopes)


def codes(notes):
    return [n.get('code') for n in notes]


class FullSourceTests(unittest.TestCase):
    def test_the_recorded_md24_answer_is_accepted_from_the_full_provision(self):
        # The answer the v019t5 run refused twice: both quotes are the opening of a clause of their own source + "...".
        scope, basis, rule, notes, failure = judge(Judge(answer(('p0', MD24_OPENING), ('s1', MD1_OPENING))))
        self.assertIsNone(failure)
        self.assertEqual((scope.applicability, rule), ('APPLIES', 'MODEL'))
        self.assertNotIn('PROPOSAL_INVALID', codes(notes))
        self.assertEqual([q.quote for q in scope.scope_evidence], [MD24_OPENING[:-3], MD1_OPENING[:-3]])
        note = next(n for n in notes if n['code'] == QUOTE_FULL_SOURCE_OK)
        self.assertEqual(note['attempt'], 1)
        self.assertIn('p0 (cut_mark)', note['detail'])
        self.assertEqual([b.match for b in basis], ['YES'])

    def test_1_a_quote_left_out_of_the_trimmed_prompt_is_accepted_from_the_full_text(self):
        # A provision too long for the window reaches the model clipped at a sentence end (CLIP_MARKER); the clause
        # the model quotes lies in the part that was cut away. The check reads the full text, so the quote stands.
        filler = ' '.join(f'({n}) Yükümlüler bu fıkrada sayılan bilgileri işlem tarihinden itibaren kayıt altında tutar ve saklar.'
                          for n in range(2, 260))
        long = dict(md(24), id='t6-long', text=filler + ' ' + md(24)['text'])
        provider = Judge(answer(('p0', MD24_OPENING)), window=(1024, 8192))
        scope, basis, rule, notes, failure = judge(provider, section=long)
        sent = provider.sent[-1]['provision']['text']
        self.assertTrue(sent.endswith(CLIP_MARKER.strip()) or CLIP_MARKER in sent)
        self.assertNotIn('Onbeşbin', sent)
        self.assertIsNone(failure)
        self.assertEqual(scope.applicability, 'APPLIES')
        self.assertIn(scope.scope_evidence[0].quote, long['text'])
        self.assertIn('ContextBudgetError', codes(notes))
        self.assertIn(QUOTE_FULL_SOURCE_OK, codes(notes))

    def test_2_a_quote_in_no_source_is_refused(self):
        invented = 'Bankalar her elektronik transferde müşteriden ayrıca yazılı onay alır.'
        span, reason, detail = check_scope_quote('p0', invented, {'p0': md(24)['text'], 's1': md(1)['text']})
        self.assertEqual((span, reason, detail), (None, QUOTE_NOT_IN_SOURCE, ''))
        scope, basis, rule, notes, failure = judge(Judge(answer(('p0', invented)), answer(('p0', invented + '...'))))
        self.assertIsNone(scope)
        self.assertEqual(failure, 'PROPOSAL_INVALID: manual analysis required')
        dropped = [n for n in notes if n['code'] == 'SCOPE_QUOTE_DROPPED']
        self.assertEqual([n['reason_codes'] for n in dropped], [[QUOTE_NOT_IN_SOURCE]] * 2)
        self.assertIn(invented[:40], dropped[0]['detail'])

    def test_3_a_quote_from_another_article_is_refused_and_named(self):
        # md. 1's opening cited as the provision (p0 = md. 24): in a supplied source, but not the one it names.
        sources = {'p0': md(24)['text'], 's1': md(1)['text'], 's2': md(4)['text']}
        self.assertEqual(check_scope_quote('p0', MD1_OPENING, sources), (None, QUOTE_OTHER_SOURCE, 's1'))
        self.assertEqual(check_scope_quote('s2', MD24_OPENING, sources), (None, QUOTE_OTHER_SOURCE, 'p0'))
        # An article that was not supplied at all (md. 31) is simply not in the source.
        md31 = md(31)['text'][:160]
        self.assertEqual(check_scope_quote('p0', md31, sources)[:2], (None, QUOTE_NOT_IN_SOURCE))
        provider = Judge(answer(('p0', MD1_OPENING)), answer(('s1', MD1_OPENING)))
        scope, basis, rule, notes, failure = judge(provider)
        self.assertEqual(next(n for n in notes if n['code'] == 'SCOPE_QUOTE_DROPPED')['reason_codes'], [QUOTE_OTHER_SOURCE])
        # The repair is told where the quote was found; the model re-cites it under its own id and it stands.
        self.assertIn('QUOTE_OTHER_SOURCE in s1', provider.sent[1]['validation_feedback'])
        self.assertEqual([(q.source_id, q.quote) for q in scope.scope_evidence], [('s1', MD1_OPENING[:-3])])

    def test_one_wrong_quote_beside_a_good_one_is_dropped_and_the_answer_stands(self):
        scope, basis, rule, notes, failure = judge(Judge(answer(('p0', MD24_OPENING), ('p0', MD1_OPENING))))
        self.assertEqual((scope.applicability, [q.source_id for q in scope.scope_evidence]), ('APPLIES', ['p0']))
        dropped = next(n for n in notes if n['code'] == 'SCOPE_QUOTE_DROPPED')
        self.assertEqual(dropped['reason_codes'], [QUOTE_OTHER_SOURCE])
        self.assertIn('[QUOTE_OTHER_SOURCE in s1]', dropped['detail'])

    def test_4_small_normalisation_differences_are_accepted(self):
        text24, text31 = md(24)['text'], md(31)['text']
        cases = [
            # letter case
            ('(1) onbeşbin tl veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında', text24, 'normalized'),
            # a typographic apostrophe typed straight
            ("Onbeşbin TL'nin altındaki yurt içi ve yurt dışı elektronik transfer mesajlarında", text24, 'normalized'),
            # diacritics lost
            ('Onbesbin TL veya uzeri yurt ici ve yurt disi elektronik transfer mesajlarinda', text24, 'normalized'),
            # the source's stray space before a comma closed up
            ('mikrofiş, mikrofilm, manyetik teyp, disket ve benzeri ortamlar', text31, 'normalized'),
            # two source words run together (minor OCR)
            ('c) Adresi veyadoğum yeri ve tarihi veya müşteri numarası', text24, 'compact'),
            # a line break and doubled spaces
            ('Onbeşbin TL veya\nüzeri  yurt içi ve yurt dışı', text24, 'exact'),
        ]
        for quote, text, how in cases:
            with self.subTest(quote=quote):
                span, found = quote_span(quote, text)
                self.assertEqual(found, how)
                self.assertIn(span, text)
                self.assertEqual(check_scope_quote('p0', quote, {'p0': text})[1], QUOTE_FULL_SOURCE_OK)

    def test_no_reading_accepts_a_word_that_is_not_the_sources(self):
        text24 = md(24)['text']
        for quote in ('Onbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında',        # another amount
                      '(1) Onbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında...',  # the same, cut short
                      'Onbeşbin TL veya üzeri yurt içi ve yurt dışı elektronik para transfer mesajlarında',  # a word added
                      'Onbeşbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında... alıcının adı yazılmaz',
                      'a) bankalar.',                                                                        # too short to fold
                      '...', '…', '[...]'):
            with self.subTest(quote=quote):
                self.assertEqual(quote_span(quote, text24 + ' ' + md(4)['text']), (None, ''))

    def test_words_the_model_skipped_inside_its_source_keep_the_exact_tail_as_before(self):
        # source_span's documented readings are kept when every word of the quote is the source's: a quote that skips
        # from a list's opening to its item, and one that leaves words out, name the exact copy at their end.
        listed = ('(1) Bu Kanunda geçen; a) Bakanlık: Maliye Bakanlığını, b) Bakan: Maliye Bakanını, d) Yükümlü: Bankacılık, '
                  'sigortacılık ve ödeme hizmetleri alanında faaliyet gösterenleri,')
        self.assertEqual(quote_span('(1) Bu Kanunda geçen; d) Yükümlü: Bankacılık, sigortacılık ve ödeme hizmetleri alanında faaliyet '
                                    'gösterenleri,', listed),
                         ('d) Yükümlü: Bankacılık, sigortacılık ve ödeme hizmetleri alanında faaliyet gösterenleri,', 'exact'))
        self.assertEqual(quote_span('Onbeşbin TL veya üzeri yurt dışı elektronik transfer mesajlarında, gönderenin', md(24)['text']),
                         ('yurt dışı elektronik transfer mesajlarında, gönderenin', 'exact'))

    def test_5_a_quote_from_another_snapshot_of_the_regulation_is_refused(self):
        # Recorded (v019t5, I04 md. 8): the model cited Kanun 5549 md. 2(d) in a wording the retained snapshot does not
        # have ("ödeme hizmetleri ve elektronik para ihracı ..."): another version of the statute, refused.
        md2 = article(KANUN, 'Kanun 5549 md. 2')['text']
        newer = 'ödeme hizmetleri ve elektronik para ihracı alanlarında faaliyet gösterenleri'
        self.assertEqual(check_scope_quote('s1', newer, {'s1': md2}), (None, QUOTE_NOT_IN_SOURCE, ''))
        # The same article in two snapshots: only the snapshot that was supplied counts.
        older = md(24)['text'].replace('Onbeşbin TL', 'Onbin TL')
        quote = '(1) Onbin TL veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında'
        self.assertEqual(check_scope_quote('p0', quote, {'p0': older})[1], QUOTE_FULL_SOURCE_OK)
        self.assertEqual(check_scope_quote('p0', quote, {'p0': md(24)['text']})[:2], (None, QUOTE_NOT_IN_SOURCE))
        # Another regulation: a Tedbirler article is not the Kanun article it is cited as.
        self.assertEqual(check_scope_quote('s1', MD1_OPENING, {'s1': md2})[:2], (None, QUOTE_NOT_IN_SOURCE))

    def test_6_an_accepted_quote_is_stored_as_the_exact_span_of_the_full_source(self):
        text24 = md(24)['text']
        for quote in (MD24_OPENING, '(1) onbeşbin tl veya üzeri yurt içi ve yurt dışı elektronik transfer mesajlarında',
                      'c) Adresi veyadoğum yeri ve tarihi veya müşteri numarası', "Onbeşbin TL'nin altındaki yurt içi ve yurt dışı"):
            with self.subTest(quote=quote):
                span, reason, _ = check_scope_quote('p0', quote, {'p0': text24})
                self.assertEqual(reason, QUOTE_FULL_SOURCE_OK)
                start = text24.index(span)
                self.assertEqual(text24[start:start + len(span)], span)
                self.assertNotIn('...', span)
                self.assertLessEqual(len(span), QUOTE_LIMIT)
        scope, *_ = judge(Judge(answer(('p0', 'c) Adresi veyadoğum yeri ve tarihi veya müşteri numarası'))))
        self.assertEqual(scope.scope_evidence[0].quote, 'c) Adresi veya doğum yeri ve tarihi veya müşteri numarası')

    def test_7_a_source_without_its_full_text_is_refused_with_its_code(self):
        clipped = md(24)['text'][:400] + CLIP_MARKER
        for text in ('', '   ', None, clipped):
            with self.subTest(text=text):
                self.assertEqual(check_scope_quote('p0', '(1) Onbeşbin TL veya üzeri', {'p0': text}), (None, FULL_SOURCE_UNAVAILABLE, ''))
        self.assertEqual(check_scope_quote('s9', MD1_OPENING, {'p0': md(24)['text'], 's1': md(1)['text']}), (None, QUOTE_UNKNOWN_SOURCE, ''))
        # It fails closed: the answer that rests on such a quote is refused, never accepted unread.
        empty = dict(md(1), text='')
        scope, basis, rule, notes, failure = judge(Judge(answer(('s1', MD1_OPENING)), answer(('s1', MD1_OPENING))),
                                                   scopes={'s1': empty, 's2': md(4)})
        self.assertIsNone(scope)
        self.assertEqual(failure, 'PROPOSAL_INVALID: manual analysis required')
        self.assertEqual({tuple(n['reason_codes']) for n in notes if n['code'] == 'SCOPE_QUOTE_DROPPED'}, {(FULL_SOURCE_UNAVAILABLE,)})


class PacketTests(unittest.TestCase):
    def test_a_packet_with_a_shortened_scope_quote_keeps_its_chain_and_passes_review(self):
        # Through analyze(): the stored quote is the source span, so the chain verifies and a review that copies the
        # proposal (autonomous_review) passes review.validate_evidence, which demands an exact substring of the source.
        from regchain.evidence import verify_chain
        from regchain.pilot.engine import analyze, autonomous_review
        from regchain.pilot.review import apply_review
        from test_pilot import FixtureProvider, company, policies, sections

        class Shortening(FixtureProvider):
            def scope(self, payload):
                value = super().scope(payload)
                value['scope_evidence'][0]['quote'] = ' '.join(payload['scope'][0]['text'].split()[:5]) + '...'
                return value
        packet = analyze(company(), policies(), sections(), Shortening(), ['CONC 7.3.4'])
        self.assertTrue(verify_chain(packet['events'], packet['head'], 1))
        canonical_bytes(packet)
        row = packet['events'][0]['payload']['obligations'][0]
        self.assertEqual(row['proposal']['applicability'], 'APPLIES')
        self.assertEqual(row['proposal']['scope_evidence'], [{'source_id': 's-scope', 'quote': 'This chapter applies to consumer'}])
        self.assertIn(QUOTE_FULL_SOURCE_OK, codes(row['diagnostics']))
        reviewed = apply_review(packet, autonomous_review(packet)[0])
        self.assertEqual(reviewed['count'], 2)


class CompatibilityTests(unittest.TestCase):
    def test_an_exact_quote_adds_no_note_and_is_stored_unchanged(self):
        exact = MD24_OPENING[:-3]
        scope, basis, rule, notes, failure = judge(Judge(answer(('p0', exact), ('s2', 'a) Bankalar.'))))
        self.assertEqual([q.quote for q in scope.scope_evidence], [exact, 'a) Bankalar.'])
        self.assertEqual(codes(notes), [])

    def test_the_basis_check_is_unchanged_a_shortened_condition_stays_refused(self):
        # Deliberately not widened: on the recorded answers every basis condition refused for "..." belonged to a
        # DOES_NOT_APPLY that would otherwise have been accepted (Kanun 5549 md. 7 for a software company, labelled APPLIES).
        condition = 'b) Bankalar dışında banka kartı veya kredi kartı düzenleme yetkisini haiz kuruluşlar'
        self.assertIn(condition, md(4)['text'])
        refusal = [{'company_fact': 'mevduat bankacılığı', 'regulatory_condition': condition + '...', 'match': 'NO'}]
        shortened = answer(('p0', MD24_OPENING), applicability='DOES_NOT_APPLY', basis=refusal)
        scope, basis, rule, notes, failure = judge(Judge(shortened, shortened))
        self.assertIsNone(scope)
        self.assertIn('BASIS_DROPPED', codes(notes))

    def test_the_notes_are_digest_safe_and_name_only_known_codes(self):
        scope, basis, rule, notes, failure = judge(Judge(answer(('p0', MD24_OPENING), ('p0', MD1_OPENING), ('s7', 'x y z'))))
        canonical_bytes(notes)
        self.assertFalse(any(isinstance(v, float) for n in notes for v in n.values()))
        dropped = next(n for n in notes if n['code'] == 'SCOPE_QUOTE_DROPPED')
        self.assertEqual(dropped['reason_codes'], [QUOTE_OTHER_SOURCE, QUOTE_UNKNOWN_SOURCE])
        self.assertTrue(set(dropped['reason_codes']) <= set(QUOTE_REASON_CODES))


if __name__ == '__main__':
    unittest.main()

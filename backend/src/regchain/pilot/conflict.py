"""Deterministic conflict pre-check (v0.19): wording and quantities that may set a passage against a duty.

Measured on the v0.18 final run (24 September 2026, 337 passages judged): the thinking "does it
contradict?" question was asked of every passage and cost 4,496 model seconds, for 18 YES answers.
Every planted contradiction of the dataset carried either waiver or limiter wording ("atlanır",
"istisna", "yalnızca") or a quantity that the duty also states ("ay sonunda" against "en geç on iş
günü", "beş yıl" against "sekiz yıl"); a rule on those two (R4r) plus the fast classifier's
POSSIBLE_CONFLICT kept 10 of 10 planted rows with 59 verifier calls instead of 337.
The engine's older DEVIATION_WORDING screen exemption is not reused: its bare \\d matched every
control-register row through its id ("RET-01") and missed "ay sonunda MASAK" and "beş yıl
saklanır", 8 of the 10 labelled conflicts. Numbers here come from extraction.quantities, which
needs a unit ("gün", "yıl", "TL") beside the number.

Nothing in this module calls a model or decides a relation: it names what it found (a signal
with a type and a strength) and the engine decides which passage the thinking verifier reads.
The -maz/-mez negation class is kept but weak: it matched 55 judged passages and none was a YES.
"""
import re

from regchain.extraction.quantities import compare, parse_quantities, period_phrases
from regchain.extraction.structure import KEEPING, duty_quantities
from .policies import fold

L = 'a-zçğıöşüâîû'
NB = r'(?<![%s])' % L
NA = r'(?![%s])' % L
W = '[%s]' % L
# Strong wording: the passage removes, excepts or limits something.
WAIVER = re.compile(NB + r'(?:atlan\w*|uygulanmaz\w*|uygulanmay\w*|yapılmaz\w*|gerekmez\w*|aranmaz\w*|istenmez\w*|vazgeç\w*|ertelen\w*|'
                    r'sonradan|askıya\s+al\w*|(?:zorunlu|şart|gerekli|mecburi|yükümlü)\s+değil\w*|'
                    r'skip\w*|waiv\w*|not\s+required|need\s+not|(?:does|do)\s+not\s+need|no\s+need|suspend\w*|defer\w*|postpon\w*|'
                    r'dispens\w*|not\s+mandatory)' + NA)
# "-madıkça / -medikçe" is Turkish "unless" (English "unless" was already here). Measured (I03 md. 7(1), 25
# September 2026): "... mahkeme kararı bulunmadıkça hiçbir kamu kurumuna verilmez" was an EXEMPTION_ADDED the
# verifier found and the type rule then rejected for want of excepting wording; the row was PARTIAL, label CONFLICT.
EXEMPTION = re.compile(NB + r'(?:hariç\w*|istisna\w*|muaf\w*|(?<!yurt\s)dışında\w*|kapsam\s+dışı\w*|\w+(?:madıkça|medikçe)|except\w*|unless|'
                       r'exempt\w*|exclud\w*|other\s+than)' + NA)
LIMITER = re.compile(NB + r'(?:yalnız\w*|sadece|sınırlı\s+olarak|only|solely|limited\s+to|exclusively)' + NA)
# Only against a MUST_NOT duty: the passage permits.
PERMISSIVE = re.compile(NB + r'(?:\w+(?:abil|ebil)(?:ir|irler|mektedir)|serbest\w*|izin\s+veril\w*|may(?!\s+not)|can(?!not|\s+not)|'
                        r'permitted|allowed)' + NA)
# Only against a MUST duty: the passage makes it discretionary.
WEAKER = re.compile(NB + r'(?:takdir\w*|isteğe\s+bağlı|tavsiye\w*|mümkün\s+olduğunca|mümkünse|önerilir|may(?!\s+not)|should|'
                    r'optional\w*|discretion\w*|encouraged|where\s+(?:possible|practicable)|if\s+possible|recommended)' + NA)
# Weak wording: a prohibition or a plain negation, usually about another measure.
PROHIBITION = re.compile(NB + r'(?:\w+(?:amaz|emez)(?:lar|ler)?|yasak\w*|prohibit\w*|forbid\w*|must\s+not|may\s+not|shall\s+not|'
                         r'cannot)' + NA)
NEGATION = re.compile(NB + r'(?:\w{2,}(?:maz|mez)(?:lar|ler)?(?:dı|di)?|değil\w*|not|never|no\s+longer)' + NA)
WORDING = (('WAIVER', WAIVER, 'strong'), ('EXEMPTION', EXEMPTION, 'strong'), ('LIMITER', LIMITER, 'strong'),
           ('PROHIBITION', PROHIBITION, 'weak'), ('NEGATION', NEGATION, 'weak'))
WHY = {'WAIVER': 'skips, waives, suspends or defers something', 'EXEMPTION': 'makes an exception',
       'LIMITER': 'limits something to some cases', 'PROHIBITION': 'forbids something', 'NEGATION': 'negates something',
       'PERMISSION_OF_PROHIBITED': 'permits something and the duty forbids', 'WEAKER_MODALITY': 'makes something discretionary and the duty requires it'}
# A deadline, a keeping period and a trigger amount of the duty are compared with the passage's own;
# which passage number can be compared with which duty number depends on the words around it.
DEADLINE_LEADS = ('en geç', 'within', 'no later than', 'not later than')
DEADLINE_TRAILS = ('içinde', 'içerisinde', 'zarfında')
DURATION_TRAILS = ('süre ile', 'süreyle', 'süresince', 'boyunca')
MAX_KINDS = ('DEADLINE', 'IMMEDIATE', 'BEFORE')
BEST = ('SAME', 'STRICTER', 'WEAKER', 'INCOMPARABLE')
# Phase-5 screen safety (v0.19): a passage with any of this wording or a quantity is never set aside by
# the unreasoned relevance screen, however it answers. Measured: of 81 passages the v0.18 screen set
# aside, 3 carried this wording, among them the KYC-skip passage of C13 md. 5(3).
PROTECTED = re.compile(NB + r'(?:atlan|uygulanmaz|yapılmaz|hariç|istisna|muaf|zorunlu\s+değil|gerekmez|aranmaz|ertelen|vazgeç|sadece|'
                       r'yalnız|en\s+geç|en\s+az|en\s+fazla|süre|ay\s+sonu|ayın\s+son|toplu|skip|waiv|exempt|except|unless|not\s+required|'
                       r'within|at\s+least|at\s+most|only%s|(?:gün|yıl)(?:ü|ün|e|a|de|da|den|dan|ı|ın|lük|lık|ler|lar|dür|dır)?%s|'
                       r'%s{2,}(?:maz|mez)(?:lar|ler)?%s)' % (NA, NA, W, NA))
# Words that say nothing about which measure a text is about (folded, four letters or more).
STOPWORDS = frozenset({'ilişkin', 'olarak', 'tarafından', 'üzere', 'yapılan', 'dair', 'olan', 'veya', 'için', 'gibi', 'ettiği', 'edilen',
                       'edilir', 'edilmesi', 'etmek', 'olmak', 'yapmak', 'bulunmak', 'zorundadır', 'zorundadırlar', 'zorunludur', 'gerekir',
                       'gerekmektedir', 'yükümlüler', 'yükümlü', 'yükümlülerce', 'hususlar', 'hususunda', 'kapsamında', 'şekilde', 'durumunda',
                       'halinde', 'hâlinde', 'itibaren', 'içinde', 'aşağıdaki', 'ancak', 'daha', 'kadar', 'bunlar', 'diğer', 'ayrıca',
                       'must', 'shall', 'should', 'with', 'that', 'this', 'which', 'from', 'such', 'their', 'there', 'where', 'have',
                       'been', 'firm', 'firms', 'will', 'into', 'only', 'under', 'within', 'each', 'other', 'also', 'than', 'they'})
STEM = 5


def sentence_of(text: str, pos: int) -> str:
    """The sentence (or ';'-clause) of `text` that holds position `pos`, at most 300 characters."""
    start = max(text.rfind('. ', 0, pos), text.rfind('; ', 0, pos), text.rfind('\n', 0, pos))
    start = 0 if start < 0 else start + 1
    ends = [i for i in (text.find('. ', pos), text.find('; ', pos), text.find('\n', pos)) if i >= 0]
    end = min(ends) + 1 if ends else len(text)
    return text[start:end].strip()[:300]


def _folded(text: str):
    folded = fold(text or '')
    return folded, len(folded) == len(text or '')


def _signal(kind, strength, phrase, start, detail):
    return {'type': kind, 'strength': strength, 'text': phrase[:160], 'start': start, 'detail': detail[:300]}


def labelled_quantities(structure) -> list[dict]:
    """duty_quantities(structure), each with the element id duty_payload gave it ('deadline_1', 'threshold_2')."""
    if not structure:
        return []
    deadlines = [(d.get('start'), d.get('end')) for d in structure.get('deadline') or []]
    thresholds = [(t.get('start'), t.get('end')) for t in structure.get('threshold') or [] if not t.get('parenthetical')]
    labelled = []
    for entry in duty_quantities(structure):
        where = (entry.get('start'), entry.get('end'))
        if entry.get('kind') == 'THRESHOLD':
            element = f'threshold_{thresholds.index(where) + 1}' if where in thresholds else None
        else:
            element = f'deadline_{deadlines.index(where) + 1}' if where in deadlines else None
        labelled.append({**entry, 'element': element})
    return labelled


def _role(quantity: dict, text: str) -> str:
    """'deadline', 'duration' (a keeping period) or 'plain', read from the words around a passage number."""
    lead, trail = quantity.get('lead') or '', quantity.get('trail')
    sign, side = quantity.get('comparator'), quantity.get('comparator_side')
    if lead in DEADLINE_LEADS or trail in DEADLINE_TRAILS or (sign in ('<=', '<') and side == 'before'):
        return 'deadline'
    if trail in DURATION_TRAILS or lead.startswith('for') or (sign == '>=' and side == 'before'):
        return 'duration'
    if quantity.get('unit_class') == 'duration' and KEEPING.search(sentence_of(text, quantity['start'])):
        return 'duration'                                   # "beş yıl saklanır"
    return 'plain'


def _fits(duty_q: dict, passage_q: dict, role: str) -> bool:
    if duty_q.get('unit_class') != passage_q.get('unit_class'):
        return False
    if duty_q.get('kind') == 'THRESHOLD':
        return True
    if duty_q.get('direction') == 'min':
        return role in ('duration', 'plain')
    return role in ('deadline', 'plain')


def passage_quantities(text: str) -> list[dict]:
    """The passage's numbers with a unit, each with its role; and its IMMEDIATE timing as a zero-day deadline."""
    found = []
    for quantity in parse_quantities(text or ''):
        found.append({**quantity, 'role': _role(quantity, text)})
    for phrase in period_phrases(text or ''):
        if phrase['kind'] == 'IMMEDIATE':
            found.append({'text': phrase['text'], 'start': phrase['start'], 'end': phrase['end'], 'unit_class': 'duration', 'days': 0,
                          'amount': None, 'unit': None, 'role': 'deadline', 'timing': 'IMMEDIATE'})
    return found


def quantity_readings(quantities, text: str, duty: dict | None = None) -> dict:
    """{element id: (verdict, passage wording or None)} for every numeric duty quantity: the best the passage
    states (a stricter number meets the duty; a weaker one does not), 'NOT_STATED' when it states none.

    With `duty` (round 5): against a duty that REQUIRES something, a passage applying its rule at every amount
    ("tutarına bakılmaksızın", "regardless of the amount": I06 md. 24/A(2)) states each money threshold it gives no
    number for STRICTER: a requirement for every amount covers the duty's amounts."""
    out = _quantity_readings(quantities, text)
    if duty is not None and duty_polarity(duty) == 'REQUIRED':
        every = ALL_AMOUNTS.search(fold(text or ''))
        if every:
            for q in quantities or []:
                if q.get('kind') == 'THRESHOLD' and q.get('unit_class') == 'money' and out.get(q.get('element'), ('',))[0] == 'NOT_STATED':
                    out[q['element']] = ('STRICTER', (text or '')[every.start():every.end()] if len(fold(text or '')) == len(text or '')
                                         else every.group(0))
    return out


# A rule applied at every amount (no threshold at all).
ALL_AMOUNTS = re.compile(NB + r'(?:tutar(?:ı|ına|larına)?\s+(?:bakılmaksızın|gözetilmeksizin|gözetmeksizin)|tüm\s+tutarlarda|her\s+tutarda|'
                         r'tutar\s+sınırı\s+olmaksızın|regardless\s+of\s+(?:the\s+)?(?:amount|value)|irrespective\s+of\s+(?:the\s+)?(?:amount|value)|'
                         r'of\s+any\s+(?:amount|value))' + NA)


def _quantity_readings(quantities, text: str) -> dict:
    mine = passage_quantities(text)
    numeric = [q for q in quantities or [] if q.get('amount') is not None and q.get('element')]
    # A passage number that is exactly another duty number belongs to that one: "185.000 TL" states md.
    # 5(1)'s general floor, not a weaker version of its 15.000 TL wire-transfer floor.
    owner = {id(q): d['element'] for q in mine for d in numeric if _fits(d, q, q['role']) and compare(d, q, d['direction']) == 'SAME'}
    out = {}
    for duty_q in numeric:
        verdicts = [(compare(duty_q, q, duty_q['direction']), q['text']) for q in mine
                    if _fits(duty_q, q, q['role']) and owner.get(id(q), duty_q['element']) == duty_q['element']]
        out[duty_q['element']] = next(((v, t) for best in BEST for v, t in verdicts if v == best), ('NOT_STATED', None))
    return out


def quantity_match(quantities, text: str, duty: dict | None = None) -> dict:
    """{element id: 'SAME' | 'STRICTER' | 'WEAKER' | 'INCOMPARABLE' | 'NOT_STATED'} for every numeric duty
    quantity: the best the passage states (a stricter number meets the duty; a weaker one does not)."""
    return {element: verdict for element, (verdict, _) in quantity_readings(quantities, text, duty).items()}


def quantity_hint(quantities, text: str, duty: dict | None = None) -> list[dict]:
    """What the verifier is told the deterministic comparison found (strings only, digest-safe): for each
    numeric duty quantity the passage states, the duty's wording, the passage's and SAME / STRICTER / WEAKER /
    INCOMPARABLE. A stricter number meets the duty; the hint may be wrong about which number is meant."""
    wording = {q['element']: q.get('text', '') for q in quantities or [] if q.get('element')}
    return [{'element': element, 'duty_states': wording.get(element, ''), 'policy_states': said, 'comparison': verdict}
            for element, (verdict, said) in quantity_readings(quantities, text, duty).items() if said]


# The contradiction kinds a number decides. Measured in the v0.19 short round (I02 md. 8, 25 September 2026): the
# verifier called "on yıl süreyle saklanır" against "sekiz yıl süreyle muhafaza" a THRESHOLD_MISMATCH although
# ten years of keeping is stricter than eight, and the row became CONFLICT instead of COVERS_TEXT.
AMOUNT_TYPES = ('DEADLINE_MISMATCH', 'THRESHOLD_MISMATCH')


def stricter_than_required(quantities, span: str) -> dict:
    """{element id: 'SAME' | 'STRICTER'} when the contradicting sentence meets every duty number it states the
    same or stricter; {} when it states none of them, states one weaker or incomparable, or carries a timing
    against a numeric deadline (a period end, batches: C13 md. 28(2) "... o ay sonunda MASAK'a topluca
    bildirilir" beside "aynı gün" in the sentence before it). Only the span counts, never the rest of the
    passage: a stricter number elsewhere does not answer the sentence the verifier quoted."""
    stated = {e: v for e, v in quantity_match(quantities, span).items() if v != 'NOT_STATED'}
    if not stated or any(v not in ('SAME', 'STRICTER') for v in stated.values()):
        return {}
    against = [s for s in structural_signals({}, span, quantities)
               if s['type'] in AMOUNT_TYPES or (s['type'] == 'QUANTITY' and s['strength'] == 'strong')]
    return {} if against else stated


def structural_signals(duty: dict, text: str, quantities=()) -> list[dict]:
    """[{type, strength, text, start, detail}]: wording and numbers of `text` that may set it against `duty`.

    `duty` is the duty payload (its modality decides whether permission or discretion wording counts);
    `quantities` are the duty's own deadline and threshold entries (structure.duty_quantities, or
    labelled_quantities). A passage number is compared with each duty number of its kind: matching or
    stricter against any of them is no mismatch (Tedbirler md. 5(1) states 185.000 TL and 15.000 TL,
    and a policy's "185.000 TL" must not be read as weaker than the 15.000 TL wire-transfer floor).
    """
    text = text or ''
    folded, aligned = _folded(text)
    phrase = lambda m: text[m.start():m.end()] if aligned else m.group()
    signals, taken = [], []

    def add(kind, strength, match, detail):
        if any(a < match.end() and match.start() < b for a, b in taken):
            return
        taken.append((match.start(), match.end()))
        signals.append(_signal(kind, strength, phrase(match), match.start(), detail))
    modality = str(duty.get('modality') or '')
    if modality.endswith('_NOT'):
        for match in PERMISSIVE.finditer(folded):
            add('PERMISSION_OF_PROHIBITED', 'strong', match, WHY['PERMISSION_OF_PROHIBITED'])
    elif modality in ('MUST', 'SHOULD'):
        for match in WEAKER.finditer(folded):
            add('WEAKER_MODALITY', 'strong', match, WHY['WEAKER_MODALITY'])
    for kind, pattern, strength in WORDING:
        for match in pattern.finditer(folded):
            add(kind, strength, match, WHY[kind])
    duty_qs = [q for q in quantities or [] if q.get('unit_class')]
    for q in passage_quantities(text):
        if q.get('timing'):
            continue
        fitting = [(d, compare(d, q, d['direction'])) for d in duty_qs if _fits(d, q, q['role'])]
        verdicts = {v for _, v in fitting}
        if not fitting:
            signals.append(_signal('QUANTITY', 'weak', q['text'], q['start'], 'a number the duty does not state for this kind'))
        elif 'SAME' in verdicts:
            continue
        elif 'STRICTER' in verdicts:
            duty_q = next(d for d, v in fitting if v == 'STRICTER')
            signals.append(_signal('STRICTER_QUANTITY', 'weak', q['text'], q['start'], f'stricter than the duty\'s "{duty_q.get("text", "")}"'))
        elif 'WEAKER' in verdicts:
            duty_q = next(d for d, v in fitting if v == 'WEAKER')
            kind = 'THRESHOLD_MISMATCH' if duty_q.get('kind') == 'THRESHOLD' else 'DEADLINE_MISMATCH'
            signals.append(_signal(kind, 'strong', q['text'], q['start'], f'weaker than the duty\'s "{duty_q.get("text", "")}"'))
        else:
            signals.append(_signal('QUANTITY', 'strong', q['text'], q['start'], 'a number of the kind the duty states that cannot be compared with it'))
    # "ay sonunda toplu olarak" against "en geç on iş günü içinde": no number, a later time all the same.
    deadline = next((d for d in duty_qs if d.get('direction') == 'max' and d.get('kind') in MAX_KINDS), None)
    if deadline is not None and not any(d.get('kind') == 'PERIOD_END' for d in duty_qs):
        for p in period_phrases(text):
            if p['kind'] in ('PERIOD_END', 'BATCH'):
                signals.append(_signal('DEADLINE_MISMATCH', 'strong', p['text'], p['start'],
                                       f'{p["kind"]} timing against the duty\'s "{deadline.get("text", "")}"'))
    signals.sort(key=lambda s: (s['strength'] != 'strong', s['start']))
    return signals


# Wording that shows a timing or an amount is compared ("aşan", "üzerinde", "sonra", "exceed", "later").
COMPARATIVE = re.compile(NB + r'(?:aşan\w*|aşmayan\w*|üzer\w*|altında\w*|altındaki\w*|sonra\w*|geç\w*|izleyen\w*|fazla\w*|'
                         r'exceed\w*|later|after|more\s+than|less\s+than|higher|lower|above|below|over|under)' + NA)
SCOPE_TYPES = ('EXEMPTION_ADDED', 'SCOPE_NARROWED')


def type_supported(duty: dict, ctype: str, span: str, passage: str = '') -> bool:
    """True when the wording of the contradicting sentence (with the sentence before it) shows the kind of
    contradiction the verifier named. Measured in the v0.19 short round 2 (C03, 2026-09-25): against the duty to
    identify associations (md. 8(1)) and unions (md. 9(1)) the verifier called the policy's rule for registered
    companies ("Ticaret siciline kayıtlı tüzel kişilerin kimlik tespitinde ...") a SCOPE_NARROWED contradiction;
    a rule for another group says nothing about this one. An exception or a narrower scope needs wording that
    excepts, limits or skips; a later deadline or a higher amount needs a number, a period or a comparison;
    a removed requirement needs wording that removes, negates or makes it optional. Other types rest on the
    anchor rule alone."""
    at = passage.find(span) if passage and span else -1
    context = span + ' ' + sentence_of(passage, at - 2) if at > 1 else (span or '')
    folded, _ = _folded(context)
    if ctype in SCOPE_TYPES:
        return bool(WAIVER.search(folded) or EXEMPTION.search(folded) or LIMITER.search(folded)
                    or any(q.get('comparator') for q in parse_quantities(context)))
    if ctype in AMOUNT_TYPES:
        return bool(parse_quantities(context) or period_phrases(context) or COMPARATIVE.search(folded))
    if ctype == 'REQUIREMENT_REMOVED':
        return bool(WAIVER.search(folded) or EXEMPTION.search(folded) or NEGATION.search(folded) or PROHIBITION.search(folded)
                    or WEAKER.search(folded) or PERMISSIVE.search(folded))
    return True


def compact(signals, limit=6) -> list[dict]:
    """What the verifier is shown: type, strength and the wording, strongest first."""
    return [{'type': s['type'], 'strength': s['strength'], 'wording': s['text']} for s in list(signals)[:limit]]


def screen_protected(text: str) -> str:
    """The wording (or 'quantity') that keeps a passage from being set aside by the relevance screen, else ''."""
    folded, aligned = _folded(text or '')
    match = PROTECTED.search(folded)
    if match:
        return (text[match.start():match.end()] if aligned else match.group()) or 'wording'
    return 'quantity' if parse_quantities(text or '') else ''


# The core compliance acts, each a group of folded word beginnings that name the same act (TR/EN). Measured
# in the C12 smoke run (md. 46(1)): "İşlem kayıtları ve müşteri dosyaları belge yönetim sisteminde saklanır"
# against "... muhafaza etmek ve ... ibraz etmek" shared 0.176 of the duty's stems, because "saklanır" and
# "muhafaza" are different words for keeping, and the fast reading's IRRELEVANT was never re-asked. A word
# that begins with one of these counts as its act; everything else stays a five-letter stem.
ACTS = {'@keep': ('sakla', 'muhafaz', 'arşiv', 'retain', 'retent', 'keep', 'kept', 'store', 'storing', 'storage'),
        '@report': ('bildir', 'ilet', 'rapor', 'report', 'notif'),
        '@identify': ('tespit', 'teyit', 'doğrula', 'kimlik', 'identif', 'verif'),
        '@submit': ('ibraz', 'sunul', 'sunmak', 'sunar', 'submit'),
        '@monitor': ('izlen', 'izlem', 'takip', 'monitor'),
        '@destroy': ('imha', 'silin', 'silme', 'destroy', 'delet'),
        '@provide': ('verme', 'veril', 'verir', 'provid', 'furnish'),
        # v0.19 round 5 (I02 md. 4(2), I05 aa232c): "paylaşılmaz" restates "açıklayamazlar"; one act, disclosing.
        '@disclose': ('açıkla', 'ifşa', 'paylaş', 'bilgilendir', 'disclos', 'reveal', 'tipping', 'inform')}
PASSAGE_MIN_STEMS = 3
ACT_OF = sorted(((root, act) for act, roots in ACTS.items() for root in roots), key=lambda pair: -len(pair[0]))


def _token(word: str) -> str:
    return next((act for root, act in ACT_OF if word.startswith(root)), word[:STEM])


def stems(text: str) -> set:
    """Folded content stems of `text`, a word of a core act (ACTS) as that act."""
    return {_token(word) for word in re.findall('%s{4,}' % W, fold(text or '')) if word not in STOPWORDS}


def core_acts(duty: dict) -> set:
    """The core acts (ACTS) the duty's subject and action name: {'@keep', '@submit', ...} for md. 46(1),
    {'@identify'} for md. 5(2), whose act is its subject ("Kimlik tespiti ... tamamlanır")."""
    duty = duty or {}
    words = ' '.join([str(duty.get('subject') or ''), duty.get('prohibited_action') or duty.get('required_action') or ''])
    return {s for s in stems(words) if s.startswith('@')}


def shares_act(duty: dict, text: str) -> bool:
    """True when the passage names one of the duty's core acts, in any of its words."""
    return bool(core_acts(duty) & stems(text))


# Stems too common in this domain to say which measure a sentence is about (a transaction, a customer, an
# institution): they never anchor a finding on their own.
GENERIC = frozenset({'işlem', 'müşte', 'kurum', 'kurul', 'kuruluş', 'hizme', 'şirke', 'perso', 'custo', 'compa', 'servi', 'busin',
                     # verbs of doing or applying anything: 'uygulanabilir' is not about simplified measures
                     'uygul', 'yapıl', 'yapma', 'gerçe', 'apply', 'appli'})


def anchor_stems(duty: dict) -> set:
    """What a finding about this duty must mention: the stems and core acts of its subject, action and items."""
    duty = duty or {}
    words = ' '.join([str(duty.get('subject') or ''), duty.get('prohibited_action') or duty.get('required_action') or '',
                      str(duty.get('object') or ''), str(duty.get('recipient') or ''),
                      *[str(i) for i in duty.get('items') or []],
                      *[str(e.get('text') or '') for e in duty.get('elements') or []
                        if e.get('kind') in ('action', 'object', 'item', 'prohibition', 'recipient')]])
    return {s for s in stems(words) if s not in GENERIC}


def anchored(duty: dict, text: str, passage: str = '') -> bool:
    """True when `text` names the duty's own act or object (a shared stem or core act of its subject, action or
    items). Measured in the v0.19 smoke run (C14 md. 26(2), 2026-09-24): against "basitleştirilmiş tedbirleri
    uygulayamazlar" the verifier called the KYC-skip sentence ("... kimlik tespiti ve teyit adımı atlanır ...")
    an EXEMPTION_ADDED, reasoning that a skip is a simplified measure; the sentence names neither simplified
    measures nor applying them, so it is a finding about another duty (md. 5), not this one. The 4b fast
    reading likewise called off-topic passages SUPPORTS. A duty with no anchor stems anchors everything."""
    anchors = anchor_stems(duty)
    if not anchors:
        return True
    if passage and text not in passage:
        return False
    context = evidence_context(text, passage)
    return bool(anchors & stems(context))


# An adjacent sentence supplies context only through an explicit backward reference.
# Mere adjacency cannot turn an unrelated quoted rule into evidence for this duty.
BACK_REFERENCE = re.compile(r'^(?:bu\s+(?:istisna|kural|durum|işlem|belge|bilgi|süre|koşul|şart)|'
                            r'söz\s+konusu\b|bunlar\b|bunların\b|böyle\b|'
                            r'(?:this|that|these|those)\s+(?:rule|exception|condition|requirement|record|document|information|period|case)|'
                            r'they\b|it\b|such\s+(?:records?|documents?|information|requests?|cases?))', re.I)


def evidence_context(span: str, passage: str = '') -> str:
    sentence = quoted_sentences(span, passage)
    at = passage.find(sentence) if passage and sentence else -1
    if at <= 1 or not BACK_REFERENCE.search(fold(sentence)):
        return sentence
    head = passage[:at].rstrip()
    boundaries = list(re.finditer(r'(?<=[.!?])\s+|\n+', head))
    before = head[boundaries[-1].end():] if boundaries else head
    return before + ' ' + sentence


def topic_overlap(duty: dict, text: str) -> float:
    """How much the duty and the passage share, from 0 to 1: the larger of duty→passage and passage→duty.

    duty→passage: the share of the duty's content stems found in the passage. The duty's words are
    its action (with the object inside it) and its listed items; with the governing sentence as well,
    the larger of the two shares counts, so a passage that restates the action alone is not diluted
    by the rest of a long sentence. passage→duty: the share of the passage's own stems found among
    the duty's, so a short passage that restates part of a long duty is not diluted by the duty's
    length (C12 md. 46(1): 0.176 duty→passage, 0.5 passage→duty). Folded, five-letter stems (Turkish
    suffixes vary: "tedbirleri" / "tedbir"), stopwords removed, the words of one core act (ACTS:
    sakla/muhafaza/arşiv, bildir/ilet/rapor ...) counted as one.
    """
    duty = duty or {}
    action = duty.get('prohibited_action') or duty.get('required_action') or ''
    core = stems(' '.join([action, *[str(i) for i in duty.get('items') or []]]))
    whole = core | stems(duty.get('source_sentence') or '')
    found = stems(text)
    shares = [len(group & found) / len(group) for group in (core, whole) if group]
    # Not for a fragment of one or two words ("## Kayıtlar"): one shared word would be all of it.
    if len(found) >= PASSAGE_MIN_STEMS and whole:
        shares.append(len(found & whole) / len(found))
    return max(shares) if shares else 0.0


def strong(signals) -> list[dict]:
    return [s for s in signals if s['strength'] == 'strong']



# ---- roles, acts and polarity (v0.19 round 5, 25 September 2026) ------------------------------------------------
# Measured on the v0.19 t4 runs (micro set): every false CONFLICT was a sentence about another party or another act.
#   - the bank's own OUTGOING messages ("Bankamızca gönderilen ...") against a duty of the RECEIVING institution
#     ("... mesajı alan finansal kuruluş ... iade eder", I05 md. 24(4)) or of the INTERMEDIARY institutions ("...
#     transfere aracılık eden tüm finansal kuruluşlarca ...", I05 md. 24(6), I06 md. 24/A(4));
#   - customers exempted from THEIR declaration ("... bu beyandan muaftır") against the institution's duty to fill its
#     transfer messages (I06 md. 24/A(2), a COVERS_TEXT trap);
#   - reporting to MASAK ("... MASAK'a bildirilmez", "... otoriteye raporlanır") against the duty NOT to disclose to
#     anyone that a report was made (I02 md. 4(2), I04 md. 4(2)).
# A duty and a policy sentence are therefore compared as role records (who acts, in which role and direction, to
# whom, which act, with which polarity), all read deterministically from the wording. Nothing here decides that a
# passage supports a duty; it names a mismatch that makes a claim (or a favourable reading) about another duty.
INSTITUTION = (r'(?:kripto\s+varlık\s+hizmet\s+sağlayıcı|hizmet\s+sağlayıcı|finansal\s+kuruluş|ödeme\s+kuruluş|elektronik\s+para\s+kuruluş|'
               r'kuruluş|banka|payment\s+service\s+providers?|service\s+providers?|psps?|institutions?|providers?|firms?|banks?)')
_EVERY = r'(?:(?:tüm|bütün|her|diğer|the|all|any|an?)\s+)?'
_REST = r'(?P<rest>%s*)' % W
ROLE_PHRASES = (
    ('INTERMEDIARY', re.compile(NB + r'(?:aracılık\s+eden|aracı|intermediary|intermediaries)\s+' + _EVERY + INSTITUTION + _REST)),
    ('RECEIVING', re.compile(NB + r'(?:(?:mesaj|transfer|ödeme|havale)%s*\s+alan|alıcı|receiving|beneficiary\W?s?|payee\W?s?)\s+' % W
                             + _EVERY + INSTITUTION + _REST)),
    ('SENDING', re.compile(NB + r'(?:(?:(?:mesaj|transfer|ödeme|havale)%s*\s+)?gönderen|(?:transfer|ödeme)\s+emrinin\s+verildiği|ordering|'
                           r'originating|sending|payer\W?s?)\s+' % W + _EVERY + INSTITUTION + _REST)),
)
# The institution noun in the nominative or agentive ("kuruluş", "kuruluşlar", "kuruluşlarca", "... tarafından") is the
# actor; with another case ending ("... gönderen finansal kuruluştan", "... kuruluşa kadar") it is a counterparty.
ACTOR_REST = re.compile(r'(?:lar|ler)?(?:ı|i|u|ü)?(?:n?ca|n?ce|n?ça|n?çe|dır|dir|dur|dür|tır|tir|tur|tür)?')
DIRECTION_OF = {'SENDING': 'OUTGOING', 'RECEIVING': 'INCOMING', 'INTERMEDIARY': 'RELAY'}
ROLE_WORDS = {'SENDING': 'sending institution', 'RECEIVING': 'receiving institution', 'INTERMEDIARY': 'intermediary institution',
              'OBLIGED': 'obliged party', 'CUSTOMER': 'customer', 'AUTHORITY': 'authority', 'NONE': 'obliged party (not named)'}
OBLIGED = re.compile(NB + r'(?:yükümlü\w*|' + INSTITUTION + r'\w*|şirket\w*|we|our)' + NA)
CUSTOMER = re.compile(NB + r'(?:müşteri\w*|customers?|clients?)' + NA)
# Customer groups: a duty written for one of them and a passage for another are about different customers (C03 md. 8(1)).
GROUPS = {'dernek': r'dernek\w*|associations?', 'vakıf': r'vakı?f\w*|foundations?', 'sendika': r'sendika\w*|trade\s+unions?',
          'tüzel': r'tüzel\s+kişi\w*|legal\s+(?:persons?|entit\w*)|corporate\w*|kurumsal\w*',
          'gerçek': r'gerçek\s+kişi\w*|natural\s+persons?|individuals?|bireysel\w*',
          'sicil': r'ticaret\s+sicil\w*', 'kamu': r'kamu\s+kurum\w*|public\s+bod\w*', 'yabancı': r'yabancı\w*|foreign\w*',
          'pep': r'siyasi\s+nüfuz\w*|politically\s+exposed'}
GROUP = {name: re.compile(NB + '(?:' + pattern + ')' + NA) for name, pattern in GROUPS.items()}
AUTHORITY = re.compile(NB + r"(?:masak\w*|mali\s+suçları\s+araştırma\s+kurulu\s+başkanlı\w*|başkanlı\w*|otorite\w*|yetkililer\w*|"
                       r"yetkili\s+merci\w*|savcılı\w*|authorit\w*|regulators?|fca|fiu)" + NA)
THIRD_PARTY = re.compile(NB + r'(?:kimse(?:ye|yle|ler\w*)|müşteri\w*(?:ye|ya|lere|lara|yle|yla)|taraf\w*(?:lara|lere)|tarafına|'
                         r'üçüncü\s+kişi\w*|anyone|anybody|third\s+part\w*|the\s+customer)' + NA)
# A customer's own act: a declaration or undertaking the customer gives ("müşteriden ... beyan alınır").
CUSTOMER_ACT = re.compile(NB + r'(?:beyan\w*|taahhüt\w*|taahhüd\w*|declar\w*|self-?certif\w*|attest\w*)' + NA)
# Customers exempted from their own act: "... bu beyandan muaftır", "beyan aranmaz / alınmaz", "exempt from the declaration".
CUSTOMER_ACT_EXEMPTED = re.compile(NB + r'(?:(?:beyan|taahhüt|taahhüd)%s*(?:dan|den|tan|ten)\s+muaf%s*|(?:beyan|taahhüt|taahhüd)%s*\s+'
                                   r'(?:aranmaz|alınmaz|istenmez|gerekmez)%s*|exempt%s*\s+from\s+(?:the\s+|their\s+|this\s+|that\s+)?'
                                   r'(?:declaration|self-?certification|attestation))' % (W, W, W, W, W) + NA)
TRANSFER_NOUN = re.compile(r'(?:transfer|mesaj|havale|ödeme|eft|swift|message|payment|wire)')
OUTGOING = re.compile(NB + r'(?:gönderil(?:en|ir|ecek|mesi|mekte|diği)\w*|gönderdiğ\w*|giden|outgoing|sent|send)' + NA)
INCOMING = re.compile(NB + r'(?:alın(?:an|dığında|ır|ması|acak)\w*|aldığı\w*|gelen|incoming|received|receiv(?:e|es|ing))' + NA)
RELAY = re.compile(NB + r'(?:aracılık\s+(?:ettiğ|eden|edil)\w*|aracı\s+(?:sıfat|kurum|kuruluş|olarak)\w*|intermediar\w*|relay\w*)' + NA)
NEAR_WORDS = 8


def _actors(folded: str) -> list:
    """The role names (ROLE_PHRASES) of the institutions `folded` names as actors, in text order."""
    found = []
    for role, pattern in ROLE_PHRASES:
        for match in pattern.finditer(folded):
            if ACTOR_REST.fullmatch(match.group('rest')) or re.match(r'\s+tarafından', folded[match.end():]):
                found.append((match.start(), role))
    return [role for _, role in sorted(found)]


def _near_transfer(folded: str, match) -> bool:
    words = folded[:match.start()].split()[-NEAR_WORDS:] + folded[match.end():].split()[:NEAR_WORDS]
    return any(TRANSFER_NOUN.match(word) for word in words)


def recipient(text: str) -> str:
    """'AUTHORITY', 'THIRD_PARTY' or '': to whom the text communicates, read from the LAST addressee it names (the one
    next to the verb: "Başkanlığa ... bildiriminde bulunulduğunu ... hiç kimseye açıklayamazlar" is to anyone)."""
    folded = fold(text or '')
    found = [(m.start(), 'AUTHORITY') for m in AUTHORITY.finditer(folded)] + [(m.start(), 'THIRD_PARTY') for m in THIRD_PARTY.finditer(folded)]
    return max(found)[1] if found else ''


def _groups(folded: str) -> set:
    return {name for name, pattern in GROUP.items() if pattern.search(folded)}


def duty_role(duty: dict) -> dict:
    """The duty's role record, from its wording: actor (SENDING / RECEIVING / INTERMEDIARY institution, OBLIGED party,
    CUSTOMER, AUTHORITY, or NONE when the subject names an act or an object), direction (OUTGOING / INCOMING / RELAY /
    NONE), the customer groups its subject names, whether its act is a customer's own (a declaration), to whom it
    communicates (recipient), its polarity and its act keys. Strings, booleans and lists only (digest-safe)."""
    duty = duty or {}
    subject = fold(str(duty.get('subject') or ''))
    action_text = duty.get('prohibited_action') or duty.get('required_action') or ''
    action = fold(action_text)
    sentence = fold(duty.get('source_sentence') or '')
    actors = _actors(subject) or _actors(action) or _actors(sentence)
    if actors:
        actor = actors[0]
    elif CUSTOMER.search(subject) or _groups(subject):
        actor = 'CUSTOMER'                        # a customer group or a kind of entity ("Dernekler", "Kamu kurum ve kuruluşları")
    elif OBLIGED.search(subject):
        actor = 'OBLIGED'
    elif AUTHORITY.search(subject):
        actor = 'AUTHORITY'
    else:
        actor = 'NONE'
    return {'actor': actor, 'direction': DIRECTION_OF.get(actor, 'NONE'), 'groups': sorted(_groups(subject)),
            'customer_act': bool(CUSTOMER_ACT.search(subject + ' ' + action)), 'recipient': recipient(action_text),
            'polarity': duty_polarity(duty) or 'NONE', 'acts': sorted(verb_keys(duty))}


def role_view(duty: dict) -> dict:
    """What the model is told of the duty's role record (only what is known), strings only."""
    role = duty_role(duty)
    view = {'actor': ROLE_WORDS[role['actor']]}
    if role['direction'] != 'NONE':
        view['transfers'] = {'OUTGOING': 'those it sends', 'INCOMING': 'those it receives',
                             'RELAY': 'those it relays as an intermediary'}[role['direction']]
    if role['recipient']:
        view['to'] = {'AUTHORITY': 'the authority', 'THIRD_PARTY': 'anyone else'}[role['recipient']]
    if role['polarity'] != 'NONE':
        view['polarity'] = role['polarity']
    return view


def context_of(span: str, passage: str = '') -> str:
    """The full sentence(s) of `passage` that hold `span` (a ';' does not end a sentence here), else the span."""
    at = passage.find(span) if passage and span else -1
    if at < 0:
        return span or ''
    start = max(passage.rfind('. ', 0, at), passage.rfind('\n', 0, at))
    start = 0 if start < 0 else start + 1
    ends = [i for i in (passage.find('. ', at + len(span)), passage.find('\n', at + len(span))) if i >= 0]
    return passage[start:min(ends) + 1 if ends else len(passage)].strip()


def sentence_role(text: str) -> dict:
    """The policy sentence's role record: the transfer directions it is about (OUTGOING / INCOMING / RELAY: "Bankamızca
    gönderilen ... mesajlarında", "... mesajı alındığında"), its customer groups, whether it exempts customers from
    their own act ("Kurumsal müşterilerimiz ... bu beyandan muaftır") and to whom it communicates."""
    folded = fold(text or '')
    directions = {DIRECTION_OF[role] for role in _actors(folded)}
    for name, pattern in (('OUTGOING', OUTGOING), ('INCOMING', INCOMING)):
        if any(_near_transfer(folded, match) for match in pattern.finditer(folded)):
            directions.add(name)
    if RELAY.search(folded):
        directions.add('RELAY')
    exempted = bool(CUSTOMER_ACT_EXEMPTED.search(folded))
    return {'directions': sorted(directions), 'groups': sorted(_groups(folded)),
            'customer_act_exempted': exempted and bool(CUSTOMER.search(folded)), 'recipient': recipient(text)}


def role_mismatch(duty: dict, span: str, passage: str = '', claim: bool = True) -> str:
    """Why the policy sentence (`span`, read with its full sentence in `passage`) is about another party than the duty,
    or '' when nothing shows it:
      - a duty of the sending, receiving or intermediary institution against a sentence that names only other transfer
        directions (the firm's own outgoing messages against a duty of the receiving institution);
      - with `claim`, a sentence exempting customers from their own declaration against a duty whose act is not that
        declaration (the duty binds the institution, the sentence is about what the customer does).
    Role-equivalent wording ("yükümlü", "Şirket", "Bankamız", "finansal kuruluş") is one obliged party: no mismatch."""
    if not span:
        return ''
    role = duty_role(duty)
    said = sentence_role(context_of(span, passage))
    if role['direction'] != 'NONE' and not any(TRANSFER_NOUN.match(word) for word in fold(passage or span).split()):
        # A duty of an institution in its role in a transfer, and a passage that names no transfer, message or payment at
        # all (I05 md. 24(6): sharing a suspicious-transaction report with a branch).
        return f'the duty binds the {ROLE_WORDS[role["actor"]]} of a transfer; the passage names no transfer, message or payment'
    if role['direction'] != 'NONE' and said['directions'] and role['direction'] not in said['directions']:
        return (f'the duty binds the {ROLE_WORDS[role["actor"]]} ({role["direction"].lower()} transfers); the sentence is about '
                f'{" and ".join(d.lower() for d in said["directions"])} transfers')
    if claim and said['customer_act_exempted'] and not role['customer_act'] and role['actor'] != 'CUSTOMER':
        return 'the sentence exempts customers from their own declaration; the duty binds the institution to another act'
    return ''


def same_subject(duty: dict, text: str) -> bool:
    """True when a passage the verifier found not to name the duty's subject is still about it (role and customer-family
    equivalence): the subject is the obliged party, or names an act or object (a company policy speaks for the obliged
    party); a sending, receiving or intermediary institution and a passage naming no other transfer direction; a
    customer group and a passage naming no group the duty does not ("müşteri" against "gerçek kişi müşteri"). False for
    another customer group (registered companies against associations) and for an authority as the duty's actor."""
    role = duty_role(duty)
    said = sentence_role(text)
    if role['actor'] in ('OBLIGED', 'NONE'):
        return True
    if role['actor'] in DIRECTION_OF:
        return not said['directions'] or role['direction'] in said['directions']
    if role['actor'] == 'CUSTOMER':
        mine, theirs = set(role['groups']), set(said['groups'])
        if theirs - mine and mine - theirs:
            return False                          # another group: registered companies against associations
        # the same group, or the customer family without another group ("müşteri" against "gerçek kişi müşteri")
        return bool(mine & theirs) or bool(CUSTOMER.search(fold(text or '')))
    return False


# Polarity of an act: what the duty or a sentence does with it.
POLARITIES = ('REQUIRED', 'PROHIBITED', 'PERMITTED', 'OPTIONAL', 'EXEMPTED')
MODALITY_POLARITY = {'MUST': 'REQUIRED', 'SHOULD': 'REQUIRED', 'MUST_NOT': 'PROHIBITED', 'SHOULD_NOT': 'PROHIBITED', 'MAY': 'PERMITTED'}
LIGHT = re.compile(r'(?:e[dt]%s*|ol(?:mak|ur|unur|ması|malı|up)|yap(?:mak|ılır|ılması|ar|ılmalı|ılmalıdır)|kıl(?:mak|ınır)|bulun(?:mak|ur|ulur)|'
                   r'zorunda%s*|zorunlu%s*|yükümlü%s*|gerek%s*|mecbur%s*|şart%s*|esas%s*|must|shall|should|be|is|are|to|not|also)' % ((W,) * 8))
# The negative -mA before a tense or mood, not only the aorist -mAz (t6 review 1, P1-NEG-TENSE): "bildirilmez", and also the
# future "bildirilmeyecektir", the progressive "bildirilmiyor", "bildirilmemektedir" and the necessitative "bildirilmemelidir".
# Measured (review probes, 25 September 2026): "Şüpheli işlemler MASAK'a bildirilmeyecektir", "İşlem kayıtları saklanmamaktadır"
# and "Müşterinin kimliği tespit edilmeyecektir" read REQUIRED, and the gate rejected the true contradiction as EFFECT.
NEG_WORD = re.compile(r'%s{2,}(?:m[ae]z(?:lar|ler)?(?:d[ıi]r)?|m[ae]y[ae]c[ae]k(?:t[ıi]r|l[ae]r(?:d[ıi]r)?)?|m[ıiuü]yor(?:lar(?:d[ıi]r)?|d[ıiuü]r)?|'
                      r'm[ae]m[ae]kt[ae](?:d[ıi]r(?:l[ae]r)?|l[ae]r)?|m[ae]m[ae]l[ıi](?:d[ıi]r(?:l[ae]r)?|l[ae]r)?)$|^yasak' % W)
# The negated verbal noun under a word that makes it the rule: "bildirilmemesi esastır / gerekir / zorunludur" (not reporting is
# the rule: prohibited); "bildirilmemesi halinde" (a condition) is not read.
NEG_NOUN = re.compile(r'%s{2,}m[ae]m[ae](?:s[ıi]|k)$' % W)
NEG_NOUN_RULE = re.compile(r'^(?:esas|gerek(?!me)|şart|zorunlu|lazım|mecbur|özen)')
PERMIT_WORD = re.compile(r'%s+(?:abil|ebil)(?:ir|irler|mektedir|ecek)|^serbest|^izin' % W)
DOUBLE_NEGATIVE = ('engel', 'vazgeç', 'yasakla', 'kısıtla', 'kaçın')
OPTIONAL_AFTER = re.compile(r'^(?:gerekmez|zorunlu|şart|mecburi|isteğe|takdir|optional)')
EXEMPTED_AFTER = re.compile(r'^(?:aranmaz|atlan|muaf|hariç|istisna|uygulanmaz|waiv|exempt|skip)')
EN_BEFORE = {'not': 'PROHIBITED', 'never': 'PROHIBITED', 'cannot': 'PROHIBITED', 'may': 'PERMITTED', 'can': 'PERMITTED'}
# A Turkish word: a letter only Turkish has, or a dotless i inside a word (the fold turns an English capital I into a
# leading "ı": "Identity" -> "ıdentity" stays English).
TURKISH_LETTER = re.compile('[çğöşü]|%sı' % W)


def duty_polarity(duty: dict) -> str:
    return MODALITY_POLARITY.get(str((duty or {}).get('modality') or ''), '')


def verb_keys(duty: dict) -> set:
    """The duty's own verb(s) as act keys (ACTS, else five-letter stems): of a MUST_NOT duty its negated verb (the
    prohibition element: "açıklayamazlar" -> @disclose); of a MUST duty the last word of each act that is not a light
    verb or a modal ("... Başkanlığa bildirilmesi" -> @report, "... tespit etmek" -> @identify); of an English action
    its first such word."""
    return {_token(word) for word in verb_words(duty)}


def verb_words(duty: dict) -> list:
    """The folded words verb_keys reads as the duty's verb(s), one per act text."""
    duty = duty or {}
    elements = duty.get('elements') or []
    if str(duty.get('modality') or '').endswith('_NOT'):
        texts = [e['text'] for e in elements if e.get('kind') == 'prohibition'] or [duty.get('prohibited_action') or '']
    else:
        texts = [e['text'] for e in elements if e.get('kind') == 'action'] or [duty.get('required_action') or '']
    # A structured prohibition element may contain only the governing modal
    # ("must not"). Its verb belongs to the full action, not to an arbitrary
    # object noun selected later by gate_acts.
    if str(duty.get('modality') or '').endswith('_NOT') and texts and all(
            re.findall('%s{3,}' % W, fold(t or '')) and
            all(LIGHT.fullmatch(w) for w in re.findall('%s{3,}' % W, fold(t or ''))) for t in texts):
        texts = [duty.get('prohibited_action') or duty.get('required_action') or '']
    found = []
    for text in texts:
        folded = fold(text or '')
        words = [w for w in re.findall('%s{3,}' % W, folded) if not LIGHT.fullmatch(w)]
        if not words:
            continue
        english = not re.search('[çğıöşü]', folded) and (
            re.search(r'\b(?:the|and|or|of|to|a|an|all|any|on|in|with|for|where|without)\b', folded)
            or _english(str(duty.get('subject') or '') + ' ' + str(duty.get('source_sentence') or '')))
        if not TURKISH_LETTER.search(folded) and _token(words[0]).startswith('@') \
                and not _token(words[-1]).startswith('@') and not re.search(r'(?:mak|mek|ması|mesi)$', words[-1]):
            english = True
        found.append(words[0] if english else words[-1])
    return found


def _polarity_at(words: list, index: int) -> str:
    """The polarity of the act word at `index`: its own suffix, then the two words after it (a negated, optional or
    exempting verb governing it), then an English negation or modal just before it; REQUIRED otherwise."""
    word, after = words[index], words[index + 1:index + 3]
    if PERMIT_WORD.search(word):
        return 'PERMITTED'
    if NEG_WORD.search(word):
        return 'PROHIBITED'
    if NEG_NOUN.search(word) and after and NEG_NOUN_RULE.match(after[0]) and not (len(after) > 1 and after[1].startswith('değil')):
        return 'PROHIBITED'
    for next_word in after:
        if OPTIONAL_AFTER.match(next_word):
            return 'OPTIONAL'
        if EXEMPTED_AFTER.match(next_word):
            return 'EXEMPTED'
        if PERMIT_WORD.search(next_word):
            return 'PERMITTED'
        if NEG_WORD.search(next_word):
            # "uygulanması engellenmez": not preventing it permits it
            return 'PERMITTED' if next_word.startswith(DOUBLE_NEGATIVE) else 'PROHIBITED'
    # An English negation or modal only in English text (t6 review 1, P1-NOT-NOUN): the Turkish noun "not" (a note: "not
    # alınır") is no negation.
    if not any(TURKISH_LETTER.search(w) for w in words):
        before_text = ' '.join(words[:index])
        # Coordinated bare verbs share their governing modal: "No employee may
        # report or disclose". Do not cross clauses or object noun phrases.
        governing = re.sub(r'(?:\s+\w+\s+(?:or|and)){1,2}$', '', before_text)
        # Negated permissions differ from negated obligations. Read the complete
        # governing modal phrase, not just the two tokens nearest the verb.
        if re.search(r'\bnot\s+(?:required|obliged|obligated)\s+to$', governing):
            return 'OPTIONAL'
        if re.search(r'\bnot\s+(?:permitted|allowed|authori[sz]ed)\s+to$', governing):
            return 'PROHIBITED'
        if re.search(r'\b(?:may|can|must|shall|should)\s+(?:not|never)(?:\s+\w+ly)?$', governing):
            return 'PROHIBITED'
        # "No employee may disclose" is a prohibition. Exclude numeric bounds
        # and modal phrases with an intervening clause/finite verb.
        if re.search(r'\b(?:no(?!\s+(?:later|less|more|fewer|longer)\b)|neither)\s+'
                     r'(?!(?:\w+\s+){0,8}(?:must|shall|says?|said|reports?|reported|believes?|knows?)\b)'
                     r'(?:\w+\s+){0,8}(?:may|can)(?:\s+\w+ly)?$', governing):
            return 'PROHIBITED'
        for before in reversed(words[max(0, index - 2):index]):
            if before in EN_BEFORE:
                return EN_BEFORE[before]
    return 'REQUIRED'


def polarities(text: str, keys) -> set:
    """The polarities with which `text` names any of the act keys (verb_keys)."""
    words = re.findall('%s+' % W, fold(text or ''))
    return {_polarity_at(words, i) for i, word in enumerate(words) if len(word) >= 3 and _token(word) in keys}


# Wording that may narrow a restated rule (from engine.restates_prohibition); "dahil / dâhil" widens ("müşteri dâhil hiç
# kimseye"): a narrowing word right before it narrows nothing.
NARROWING = re.compile(r'\d|üzer|altında|yalnız|sadece|\bonly\b|\bdahi\b|engel|vazgeç|yasakla|reddedil|refus|prohibit|forbid|müşteri|'
                       r'customer|client|risk|\bdeğil|\bnot\b')
INCLUSIVE = re.compile(r'%s*\s+(?:d[aâ]hil|including|included)' % W)
PERMITS = re.compile(r'(?:abilir|ebilir|ılabilir|ilebilir)\b|\bhariç|\bistisna|\bancak\b|\bdışında|\bmuaf|\bkampanya|\bexcept\b|\bunless\b|'
                     r'\bmay\b(?! not)|\bpermitted\b|\bexempt')
LIST_NUMBER = re.compile(r'^\s*(?:\d{1,2}|[a-zçğıöşü])[.)]\s+')
RESTATE_SHARE = 0.6
OWN_EXCEPTION_SHARE = 0.5


def _own(duty: dict) -> str:
    return fold(' '.join([duty.get('prohibited_action') or duty.get('required_action') or '', *(duty.get('conditions') or []),
                          *(duty.get('exceptions') or [])]))


def same_direction(duty: dict, span: str) -> bool:
    """True when the sentence gives the duty's own act the duty's own polarity and nothing more: "X yapılamaz" against
    "X kesinlikle yapılamaz", a confidentiality prohibition restated with the duty's own exceptions, "X yapılır" against
    a duty to do X. Never with another polarity of that act in the sentence, with wording that narrows or permits
    beyond the duty's own conditions and exceptions, or when the sentence shares less than RESTATE_SHARE of the duty
    action's stems. The measured narrow cases (adversarial review, 24 September 2026) stay contradictions: "...
    tedbirlerden vazgeçilmez", "... uygulanması engellenmez", "50.000 TL üzerindeki işlemlerde ... açılmaz", "does not
    prohibit ..."."""
    duty = duty or {}
    polarity, keys = duty_polarity(duty), verb_keys(duty)
    if not polarity or not keys or not span:
        return False
    if polarities(span, keys) != {polarity}:
        return False
    # A list number opening the sentence ("3. Şüpheli işlem ...") is not a number of the rule.
    folded, own = LIST_NUMBER.sub('', fold(span), count=1), _own(duty)
    for match in NARROWING.finditer(folded):
        if match.group(0)[:4] in own or INCLUSIVE.match(folded, match.end()):
            continue
        return False
    duty_stems = {s for s in stems(duty.get('prohibited_action') or duty.get('required_action') or '') if s not in GENERIC}
    if not duty_stems or len(duty_stems & stems(span)) / len(duty_stems) < RESTATE_SHARE:
        return False
    if polarity == 'PROHIBITED':
        # An exception or a permission only inside a clause that restates the duty's own exceptions.
        exceptions = stems(_own(duty))
        for clause in re.split(r'[;:]', folded):
            permission = PERMITS.search(clause)
            # A modal under a negative subject is not an exception or permission.
            if permission and permission.group(0) == 'may' and polarities(clause, keys) == {'PROHIBITED'}:
                permission = None
            if permission:
                mine = {s for s in stems(clause) if s not in GENERIC}
                if not mine or len(mine & exceptions) / len(mine) < OWN_EXCEPTION_SHARE:
                    return False
        return True
    return not any(p.search(folded) for p in (WAIVER, EXEMPTION, LIMITER, WEAKER, PROHIBITION, NEGATION, PERMISSIVE))


# Acts that answer to each other: communicating (to whom decides) and keeping records (destroying them is the opposite).
ACT_FAMILY = {'@report': '@communicate', '@provide': '@communicate', '@disclose': '@communicate', '@submit': '@communicate',
              '@keep': '@records', '@destroy': '@records'}
POLARITY_TYPES = ('DIRECT_OPPOSITE', 'PROHIBITED_ACTION_ALLOWED', 'REQUIRED_ACTION_FORBIDDEN', 'REQUIREMENT_REMOVED')


def different_action(duty: dict, span: str) -> str:
    """Why a sentence claimed to order the opposite of the duty (POLARITY_TYPES) is about another act, or '' when
    nothing shows it: only when both name known acts (ACTS) of different families, or both communicate to different
    addressees (reporting to MASAK against not disclosing to anyone). A verb not in ACTS decides nothing."""
    acts = {k for k in verb_keys(duty) if k.startswith('@')}
    said = {s for s in stems(span) if s.startswith('@')}
    if not acts or not said:
        return ''
    family = lambda found: {ACT_FAMILY.get(a, a) for a in found}
    if family(acts) & family(said):
        if '@communicate' in family(acts) & family(said):
            mine = recipient((duty or {}).get('prohibited_action') or (duty or {}).get('required_action') or '')
            theirs = communication_recipient(span)
            if mine and theirs and mine != theirs:
                return (f'the duty\'s act ({", ".join(sorted(acts))}) is addressed to {mine.lower().replace("_", " ")}; the sentence '
                        f'communicates to {theirs.lower().replace("_", " ")}')
        return ''
    return f'the sentence\'s act ({", ".join(sorted(said))}) is not the duty\'s ({", ".join(sorted(acts))})'


def communication_recipient(text: str) -> str:
    """Recipient of the governing communication, excluding embedded report nouns.

    In passive informing, the patient is the recipient: an authority mentioned
    inside what the customer is told is not whom that telling addresses.
    Unknown predicates yield no positive recipient mismatch.
    """
    words = re.findall('%s+' % W, fold(text or ''))
    main = _main_act(words, {'@communicate'})
    if main is None:
        return ''
    if re.match(r'bilgilendiril', words[main]):
        head = ' '.join(words[:main])
        if re.search(r'\bmüşteri(?:ler)?\b', head):
            return 'THIRD_PARTY'
    return _gate_addressee(words, main)


# ---- the conflict gate (v0.19 t6, 25 September 2026) ------------------------------------------------------------------
# Measured on the v019t5 micro runs: both false CONFLICT rows passed every earlier rule because no rule asked whether the
# claimed contradiction applies to the duty's own act. I04 md. 4(2): a descriptive sentence ("... Mali Suçları Araştırma
# Kurulu Başkanlığı (MASAK) olup merkezi Ankara'dadır"), with no act and no polarity, anchored only on the authority's
# name and a light verb of the sentence before it, against the duty not to disclose a report. I08 md. 31(1):
# "Müşterilerimizden kimlik belgesi istenmez" (asking customers for identity papers: another act, another counterparty)
# against the duty to give the authority what it asks for, anchored on the object noun "belge" and kept as an
# EXEMPTION_ADDED because waiver wording stood somewhere in the sentence. After the earlier rules a claim therefore
# stands only when one clause of the contradicting sentence (read with the sentence before it, as the anchor rule reads
# it) holds all four of:
#   - OBJECT: the clause names the duty's own act (gate_acts), not only its object, its addressee or a word nearby;
#   - ACTOR: that act is addressed to the duty's own counterparty (reporting to the authority is not disclosing to
#     anyone else); the transfer role and the customer's own declaration stay with role_mismatch, which runs before;
#   - EFFECT: the clause sets an incompatible normative effect on that act: the opposite polarity, waiver, exemption,
#     limiting or discretionary wording, or a weaker number or timing (structural_signals);
#   - SCOPE: that effect is not the duty's own exception restated, and not a rule for another customer group only.
# A missing dimension rejects the claim with its code; semantic similarity alone never makes a contradiction.
GATE_OBJECT, GATE_ACTOR, GATE_EFFECT, GATE_SCOPE = ('CONFLICT_GATE_OBJECT', 'CONFLICT_GATE_ACTOR', 'CONFLICT_GATE_EFFECT',
                                                    'CONFLICT_GATE_SCOPE')
GATE_PASSED = 'CONFLICT_GATE_PASSED'
GATE_CODES = (GATE_ACTOR, GATE_OBJECT, GATE_SCOPE, GATE_EFFECT)
# The stage a clause reached, in the order the dimensions are read; the code of the furthest one is reported.
GATE_STAGES = (GATE_OBJECT, GATE_ACTOR, GATE_EFFECT, GATE_SCOPE)
# A clause ends at . ; : ! ? before a space, or at a line break ("Rezervasyon: ...; ... alınır.").
CLAUSE_END = re.compile(r'(?<=[.;:!?])\s+|\n+')
# The polarities of the duty's act a sentence may set against the duty's own.
INCOMPATIBLE = {'REQUIRED': ('PROHIBITED', 'EXEMPTED', 'OPTIONAL', 'PERMITTED'), 'PROHIBITED': ('PERMITTED', 'REQUIRED', 'OPTIONAL')}
# The customer's own declaration (CUSTOMER_ACT wording) as one act: "beyan alınır" and "bu beyandan muaftır".
DECLARE = '@declare'
# The numbers and timings structural_signals finds weaker than the duty's, or not comparable with them.
GATE_NUMBERS = ('DEADLINE_MISMATCH', 'THRESHOLD_MISMATCH', 'QUANTITY')
# Against a requirement: something lesser suffices or replaces it, or it is done only when someone sees fit.
SUBSTITUTE = re.compile(NB + r'(?:yeterli%s*|yerine|uygun\s+gör%s*|gerekli?\s+gör%s*|sufficient|suffices?|instead\s+of|in\s+lieu\s+of|'
                        r'if\s+deemed|deemed\s+(?:necessary|appropriate))' % (W, W, W) + NA)


# "veremezler", "bildiremez": a verb in the ability negative (-(y)AmAz), read as the verb it negates ("verme", "bildirme").
ABLE_NOT = re.compile(r'(%s{2,}?)y?([ae])m[ae]z' % W)
# The tense, negation and verbal-noun endings of a Turkish verb ("alın-ır", "alın-maz", "alın-ması", "alın-malıdır").
VERB_END = re.compile(r'(?:m[ae]z|[ıiuü]r|[ae]r|m[ae]k|m[ae]s[ıi]|m[ae]l[ıi])(?:lar|ler|d[ıi]r)?$')
# A four-letter passive root ("alın", "açıl"): the five-letter stem cut would keep a letter of the ending.
PASSIVE_ROOT = re.compile(r'[ıiuü][ln]$')
# The verb of a compound act ("iade eder", "tespit etmek", "teyit edilir", "ibraz etmek") and a verb of taking, seeking,
# keeping or applying ("onayı alınır", "belge istenir", "kayıt tutulur", "tedbir uygulanır"): the noun before it names the
# act ("onayı aranmaz" against "onayı alınır").
COMPOUND_VERB = re.compile(r'e[dt](?:er|erler|mek|mekle|ilir|ilmesi)|yap(?:ar|arlar|mak|ılır|ılması)|kıl(?:mak|ınır)|'
                           r'al(?:ınır|ınması|mak|ır|ınmalı%s*)|ara(?:nır|nması|nmalı%s*|mak|r)|iste(?:nir|nmesi|nmeli%s*|mek|r)|'
                           r'tut(?:ulur|ulması|ulmalı%s*|mak)|uygula(?:nır|nması|mak|r)' % (W, W, W, W))
# Acts of one family that undo each other: destroying records is the opposite of keeping them.
OPPOSITE_ACT = {'@keep': '@destroy', '@destroy': '@keep'}
OPPOSITE_POLARITY = {'REQUIRED': 'PROHIBITED', 'PROHIBITED': 'REQUIRED', 'PERMITTED': 'PROHIBITED'}
# The gate's version: named in the v0.19 coverage prompt hash (engine.V19_PROMPT_HASH 'precheck'), so a packet made by another
# set of these rules is never carried forward as if current. v5: the t6 gate; review1: the readings below.
GATE_VERSION = 'conflict-v6-structured-anchor-modal-context'
# ---- t6 review 1 (25 September 2026). Measured by the review probes: the gate rejected true contradictions that t5 kept
# because it read the duty's act only in listed words. The readings below widen what names the act; a clause that still names
# none of it is rejected only on positive evidence (_unread_clause).
# Final-consonant softening before a vowel (P1-SOFTENING): "teyit" -> "teyidi", "teyidine"; the noun form is how a policy
# usually waives a verification ("Adres teyidi aranmaz").
SOFTENED = {'t': 'd', 'ç': 'c', 'p': 'b', 'k': 'ğ'}
SOFT_ACT_OF = [(root[:-1] + SOFTENED[root[-1]], act) for root, act in ACT_OF if len(root) >= 4 and root[-1] in SOFTENED]
VOWELS = 'aeıioöuüâîû'
# The root of a short Turkish verb (P1-ACTIVE-PASSIVE, F1): the five-letter stem of an active aorist ("yazar", "ekler", "tutar",
# "alır") never begins its passive ("yazılmaz", "eklenmez", "tutulmaz", "alınmaz"). The root is the verb without its tense,
# negation or verbal-noun ending (VERB_END), also with the aorist's vowel ("ekle-r", "izle-r"); a clause word is that act when it
# is the root, a voice suffix (ROOT_VOICE) and a verb ending (AFTER_VOICE), or the root and a verb ending (AFTER_ROOT): "yazılmaz",
# "yazılması", "yazmaz", "alınır", "eklenmez", never "yazılı" (in writing), "yazılım", "altında", "alan".
ROOT_VOICE = r'[ıiuü]?[ln]'
AFTER_VOICE = (r'(?:m[ae]|[ıiuü]?yor|[ıiuüae]?r(?:l[ae]r)?(?:d[ıi]r)?$|y?[ae]bil|[ae]?c[ae]k|d[ıiuü][kğ]|t[ıiuü][kğ]|[ae]n(?:l[ae]r)?$|'
               r'[ıiuü]p$)')
AFTER_ROOT = r'(?:m[ae]|[ıiuü]?yor|[ıiuüae]?r(?:l[ae]r)?(?:d[ıi]r)?$|y?[ae]bil)'
# A customer in the dative, ablative or comitative ("müşterilerimizden ... istenmez": asking customers) is another counterparty
# than the authority a duty communicates to; the locative ("müşteride") is not a counterparty.
OBLIQUE = re.compile(r'(?:[dt][ae]n|[^dt][ae])$')
# Refusing what the duty requires to give or answer ("talepler reddedilir", "yanıtsız bırakılır"): an effect on the act without
# naming it.
REFUSE = re.compile(NB + r'(?:redd\w*|geri\s+çevril\w*|yanıtsız\w*|cevapsız\w*|karşılıksız\s+bırak\w*|refus\w*|reject\w*|declin\w*)' + NA)
# A dative or ablative party followed by one of these modifies a noun ("müşterilere ait işlemler", "müşterilere yapılan transferler"):
# it is not the addressee of the act (P1-DATIVE-ACTOR).
MODIFIER_AFTER = frozenset({'ait', 'ilişkin', 'yönelik', 'dair', 'ilgili', 'yapılan', 'gönderilen', 'verilen', 'sunulan', 'sağlanan',
                            'açılan', 'gelen', 'giden'})
# The note of a claim the gate keeps although no clause names the duty's act in a word it reads: the clause sets the incompatible
# effect on the duty's own object or counterparty and nothing shows another act or party (_unread_clause).
GATE_UNREAD_ACT = 'CONFLICT_GATE_ACT_UNREAD'


def _gate_token(word: str) -> str:
    """The act key of a word: the customer's declaration as DECLARE, a verb in the ability negative as the verb it
    negates, else _token (ACTS, or a five-letter stem)."""
    if CUSTOMER_ACT.match(word):
        return DECLARE
    token, able = _token(word), ABLE_NOT.match(word)
    if able and not token.startswith('@'):
        negated = _token(able.group(1) + 'm' + able.group(2))
        return negated if negated.startswith('@') else token
    if not token.startswith('@'):
        # "teyidi", "teyidine": the softened final consonant of a known act before a vowel.
        return next((act for root, act in SOFT_ACT_OF if word.startswith(root) and len(word) > len(root) and word[len(root)] in VOWELS),
                    token)
    return token


def _content_keys(text: str) -> set:
    """The content stems of `text` as act keys; stopwords, light verbs, authority names and GENERIC stems left out."""
    keys = set()
    for word in re.findall('%s{4,}' % W, fold(text or '')):
        if word in STOPWORDS or LIGHT.fullmatch(word) or AUTHORITY.fullmatch(word):
            continue
        key = _gate_token(word)
        if key not in GENERIC:
            keys.add(key)
    return keys


def gate_acts(duty: dict) -> tuple:
    """(act keys, polarity) the conflict gate reads the duty's act as: its own verb (verb_words, read with _gate_token:
    "hiç kimseye bilgi veremezler" is giving; a GENERIC one such as "uygulanmaz" left out; a verb whose root has four
    letters also as that root, VERB_END: "alınır" and "alınmaz" share "alın", not a five-letter stem); the acts its subject
    names ("Kimlik tespiti ... tamamlanır"); the subject's content stems when it names no party (actor NONE: "Üçüncü tarafa
    güven ilkesi ... uygulanmaz"); the customer's declaration when that is the duty's act; every alternative act of a
    required action; and for a double negative ("... bilgi ve belge vermekten kaçınamazlar": must give) the governed verb,
    with polarity REQUIRED. When none of these is found, the content stems of its action and object elements. A clause
    is matched on these keys (a key shorter than STEM as a word beginning); another act of the same family counts only as
    a clause's main verb (_gate_clause): "bildirim" (a report, the object of a disclosure) is not disclosing."""
    duty = duty or {}
    role, polarity = duty_role(duty), duty_polarity(duty)
    keys = set()
    for word in verb_words(duty):
        key, able = _gate_token(word), ABLE_NOT.match(word)
        root = able.group(1) if able else VERB_END.sub('', word)
        keys |= {key, root} if len(root) == STEM - 1 and PASSIVE_ROOT.search(root) and not key.startswith('@') else {key}
    keys -= GENERIC
    subject = str(duty.get('subject') or '')
    keys |= {key for key in stems(subject) if key.startswith('@')}
    if role['actor'] == 'NONE':
        keys |= _content_keys(subject)
    if role['customer_act']:
        keys.add(DECLARE)
    if polarity == 'PROHIBITED':
        words = [w for w in re.findall('%s{3,}' % W, fold(duty.get('prohibited_action') or '')) if not LIGHT.fullmatch(w)]
        if len(words) > 1 and words[-1].startswith(DOUBLE_NEGATIVE) and _gate_token(words[-2]) not in GENERIC:
            keys = (keys - {_gate_token(words[-1])}) | {_gate_token(words[-2])}
            polarity = 'REQUIRED'
    elif polarity == 'REQUIRED':
        # Every alternative act of a required action, not only its last verb: "... transferi iade eder veya ... eksik olan
        # bilgilerin tamamlanmasını sağlar" is returning the transfer as well (the noun of a compound verb, COMPOUND_VERB).
        for text in [e.get('text') or '' for e in duty.get('elements') or [] if e.get('kind') == 'action'] or [duty.get('required_action') or '']:
            words = re.findall('%s{3,}' % W, fold(text))
            keys |= {_gate_token(words[i - 1]) for i in range(1, len(words))
                     if COMPOUND_VERB.fullmatch(words[i]) and len(words[i - 1]) >= 4 and words[i - 1] not in STOPWORDS} - GENERIC
    if not keys:
        keys = _content_keys(' '.join(str(e.get('text') or '') for e in duty.get('elements') or [] if e.get('kind') in ('action', 'object')))
    return keys, polarity


def _english(text: str) -> bool:
    folded = fold(text or '')
    return not TURKISH_LETTER.search(folded) and bool(re.search(r'\b(?:the|and|of|to|a|an|all|any|must|shall)\b', folded))


def verb_roots(duty: dict) -> frozenset:
    """The short roots (two to four letters) of the duty's own Turkish verbs whose key is no known act (ACTS): "yazar" -> {"yaz",
    "yaza"}, "ekler" -> {"ekl", "ekle"}, "alır" -> {"al", "alı"}, "yazamazlar" -> {"yaz"}. Read by the gate with root_form."""
    duty = duty or {}
    if _english(duty.get('prohibited_action') or duty.get('required_action') or ''):
        return frozenset()
    roots = set()
    for word in verb_words(duty):
        if _gate_token(word).startswith('@') or word.startswith(DOUBLE_NEGATIVE):
            continue
        able = ABLE_NOT.match(word)
        if able:
            found = [able.group(1)]
        else:
            end = VERB_END.search(word)
            if not end:
                continue
            root = word[:end.start()]
            vowel = re.match(r'([aeıiuü])r', end.group(0))
            found = [root] + ([root + vowel.group(1)] if vowel and root and root[-1] not in VOWELS else [])
        roots |= {root for root in found if 2 <= len(root) < STEM}
    return frozenset(roots)


def root_acts(duty: dict) -> set:
    """The known acts (ACTS) the duty's short verb roots name in another form: "izler" (key "izler") is monitoring ("izlen-",
    "izlem-"), so "işlemlerin takibi zorunlu değildir" names the duty's act (t6 review 1)."""
    return {act for root in verb_roots(duty) for act in (_token(root + ending) for ending in ('n', 'me', 'ma', 'mek', 'mak'))
            if act.startswith('@')}


def root_objects(duty: dict) -> set:
    """The object words of the duty's own verbs, as act keys: up to five content words before each verb verb_words reads ("müşterinin
    kimlik bilgilerini üçüncü taraftan derhal alır" -> bilgi, @identify, üçünc, taraf, derha)."""
    duty = duty or {}
    elements = duty.get('elements') or []
    if str(duty.get('modality') or '').endswith('_NOT'):
        texts = [e['text'] for e in elements if e.get('kind') == 'prohibition'] or [duty.get('prohibited_action') or '']
        texts += [duty.get('prohibited_action') or '']
    else:
        texts = [e['text'] for e in elements if e.get('kind') == 'action'] or [duty.get('required_action') or '']
    keys = set()
    for text in texts:
        words = [w for w in re.findall('%s{3,}' % W, fold(text or '')) if not LIGHT.fullmatch(w)]
        keys |= _content_keys(' '.join([w for w in words[:-1] if w not in STOPWORDS][-5:]))
    return keys - {DECLARE}


def root_form(roots):
    """A matcher of the clause words that are one of `roots` as a verb (ROOT_VOICE, AFTER_VOICE, AFTER_ROOT), or None."""
    if not roots:
        return None
    alternatives = '|'.join(re.escape(root) for root in sorted(roots, key=len, reverse=True))
    return re.compile(r'(?:%s)(?:%s%s|%s)' % (alternatives, ROOT_VOICE, AFTER_VOICE, AFTER_ROOT))


def names_act(duty: dict):
    """A test of one folded word: whether it names the duty's own act as the gate reads it (gate_acts keys, root_acts,
    verb_roots), never only its object. Used by the engine's timing reader: an order of two events ("Kimlik tespiti
    tamamlanmadan iş ilişkisi kurulmaz") states the duty's BEFORE timing only when the event that comes first is its act."""
    keys = gate_acts(duty)[0] | root_acts(duty)
    roots = tuple(key for key in keys if len(key) < STEM and not key.startswith('@'))
    form = root_form(verb_roots(duty))
    return lambda word: len(word) >= 3 and (_gate_token(word) in keys or bool(roots and word.startswith(roots))
                                            or bool(form and form.match(word)))


def gate_clauses(span: str, passage: str = '') -> list:
    """The clauses the gate reads: those of the full sentence(s) holding `span`, after those of the sentence before it (a
    contradicting sentence often points back: "Bu istisna ... uygulanabilir" after the sentence that skips identification)."""
    return [clause.strip() for clause in CLAUSE_END.split(evidence_context(span, passage)) if clause.strip()]


def _addressee_before(words: list, index: int) -> str:
    """To whom the act word at `index` communicates: the last authority or third-party addressee named between the act
    word before it (ACTS) and it, '' when none is named there ("MASAK'a bildirildiği ... hiç kimseyle paylaşılmaz": to anyone)."""
    start = max([i + 1 for i in range(index) if _gate_token(words[i]).startswith('@')] or [0])
    # A party that modifies a noun ("Kurumsal müşterilere ait şüpheli işlemler", "müşterilere yapılan transferler") is not the
    # addressee of the act word (t6 review 1, P1-DATIVE-ACTOR).
    between = [word for j, word in enumerate(words[start:index], start) if not (j + 1 < index and words[j + 1] in MODIFIER_AFTER)]
    return recipient(' '.join(between))


def _own_exception(clause: str, duty: dict) -> bool:
    """True when the clause is the duty's own exception restated (OWN_EXCEPTION_SHARE of its stems): its exceptions, and a condition
    worded as one ("... mahkemeler dışında", "... onayı olmadıkça"). Not an ordinary condition (t6 review 1, P1-OWN-CONDITION): a
    permission or waiver in exactly the situation where the duty applies ("Aklama riskinin bulunduğu durumlarda da basitleştirilmiş
    tedbirler uygulanabilir" against the duty not to apply them in such situations) is the contradiction, not its exception."""
    mine = stems(clause) - GENERIC
    excepting = [c for c in duty.get('conditions') or [] if EXEMPTION.search(fold(str(c)))]
    theirs = stems(' '.join([*(duty.get('exceptions') or []), *excepting]))
    return bool(mine) and len(mine & theirs) / len(mine) >= OWN_EXCEPTION_SHARE


def _gate_clause(clause: str, keys: set, polarity: str, duty: dict, role: dict, quantities) -> tuple:
    """(stage, why) of one clause: len(GATE_STAGES) when it carries the contradiction, else the index of the first
    dimension its main act word fails (0 OBJECT, 1 ACTOR, 2 EFFECT, 3 SCOPE). Each act word of the duty has its own
    polarity and addressee; the clause's wording and numbers belong to its main act word (the last one): in "Şüpheli işlem
    bildirimi yapıldığı müşteriye en geç otuz gün içinde bildirilir" the thirty days are for telling the customer."""
    folded = fold(clause)
    words = re.findall('%s+' % W, folded)
    roots = tuple(key for key in keys if len(key) < STEM and not key.startswith('@'))
    # A short root names the act only with the duty's own object of that verb in the clause ("kayıtları tutar" / "kayıtlar
    # tutulmaz"): "al-", "tut-", "aç-" alone are any taking, holding or opening ("not alınması" is no obtaining of identity
    # information).
    objects = root_objects(duty)
    form = root_form(verb_roots(duty)) if not objects or objects & _content_keys(clause) else None
    at = [i for i, word in enumerate(words) if len(word) >= 3 and (_gate_token(word) in keys or word.startswith(roots)
                                                                   or bool(form and form.match(word)))]
    # Another act of the duty's act family counts only as the clause's main verb (its last act word, followed by nothing
    # but a light verb or a word that sets its polarity): "... Erbil şubesi uyum sorumlusuna iletilir" gives what the duty
    # forbids to give; "bildirim" before another verb is the object of that verb, not the act.
    families = {ACT_FAMILY[key] for key in keys if key in ACT_FAMILY}
    main = [i for i, word in enumerate(words) if len(word) >= 3 and ACT_FAMILY.get(_gate_token(word)) in families]
    if main and main[-1] not in at and all(LIGHT.fullmatch(w) or NEG_WORD.search(w) or PERMIT_WORD.search(w) or OPTIONAL_AFTER.match(w)
                                           or EXEMPTED_AFTER.match(w) for w in words[main[-1] + 1:]):
        at = sorted(at + [main[-1]])
    if not at:
        return 0, f"no clause names the duty's act ({', '.join(sorted(keys))})"
    if polarity == 'REQUIRED':
        wording = [p.search(folded) for p in (WAIVER, EXEMPTION, LIMITER, WEAKER, SUBSTITUTE)]
    else:
        wording = [p.search(folded) for p in (PERMISSIVE, EXEMPTION, LIMITER)]
    wording = [m.group(0) for m in wording if m]
    if polarity == 'PROHIBITED' and polarities(clause, keys) == {'PROHIBITED'}:
        wording = [w for w in wording if w not in ('may', 'can', 'permitted', 'allowed')]
    numbers = [s['text'] for s in structural_signals(duty, clause, quantities) if s['strength'] == 'strong' and s['type'] in GATE_NUMBERS]
    if not numbers and quantities and not parse_quantities(clause) and COMPARATIVE.search(folded):
        # A timing or amount compared without a number ("işlemden sonra" against "işlem yapılmadan önce"); a clause that
        # states a number is left to the comparison above (a stricter number is no contradiction).
        numbers = [COMPARATIVE.search(folded).group(0)]
    groups, clause_groups = set(role['groups']), _groups(folded)
    verdict = (0, '')
    for i in at:
        act, word = _gate_token(words[i]), words[i]
        theirs = (communication_recipient(clause) if i == at[-1] else _gate_addressee(words, i)) if ACT_FAMILY.get(act) == '@communicate' else ''
        effect = _polarity_at(words, i)
        if act not in keys and OPPOSITE_ACT.get(act) in keys:
            effect = OPPOSITE_POLARITY.get(effect, effect)    # "imha edilir" against a duty to keep: keeping is ended
        if role['recipient'] and theirs and theirs != role['recipient']:
            verdict = (1, f"\"{word}\" is addressed to {theirs.lower().replace('_', ' ')}; the duty's act to "
                          f"{role['recipient'].lower().replace('_', ' ')}")
        elif effect in INCOMPATIBLE[polarity]:
            verdict = (3, f"\"{word}\" is {effect} and the duty's act is {polarity}")
            restating = effect in ('PERMITTED', 'EXEMPTED')
        elif i == at[-1] and wording:
            verdict, restating = (3, f'"{word}" with "{wording[0]}"'), True
        elif i == at[-1] and numbers:
            verdict, restating = (3, f"\"{word}\" with \"{numbers[0]}\" against the duty's number or timing"), False
        else:
            verdict = (2, f"\"{word}\" keeps the duty's own polarity ({polarity}) with no waiving, limiting or weaker wording or number")
        if verdict[0] < 3:
            continue
        if restating and _own_exception(clause, duty):
            verdict = (3, f"{verdict[1]}, but the clause restates the duty's own exception or condition")
        elif groups and clause_groups and not groups & clause_groups:
            verdict = (3, f'{verdict[1]}, but only for {", ".join(sorted(clause_groups))}; the duty is for {", ".join(sorted(groups))}')
        else:
            return len(GATE_STAGES), verdict[1]
    return verdict


def conflict_gate(duty: dict, span: str, passage: str = '', quantities=()) -> tuple:
    """('', why) when a claimed contradiction holds on all four dimensions (actor, object, scope, effect: see above),
    else (CONFLICT_GATE_* code, why) naming the dimension the best clause misses. A duty whose act or polarity cannot
    be read (no act keys; a permission) is not gated: ('', why). Strings only (digest-safe)."""
    keys, polarity = gate_acts(duty)
    if polarity not in INCOMPATIBLE or not keys:
        return '', 'not gated: the duty\'s act or its polarity cannot be read'
    keys = keys | root_acts(duty)
    role = duty_role(duty)
    best = (0, f'no clause names the duty\'s act ({", ".join(sorted(keys))})')
    for clause in gate_clauses(span, passage):
        stage, why = _gate_clause(clause, keys, polarity, duty, role, quantities)
        if stage == len(GATE_STAGES):
            return '', f'{why} in "{clause[:160]}"'
        best = max(best, (stage, why))
    if best[0] == 0:
        # t6 review 1 (OVF-1, F1): no clause names the act in a word the gate reads. The claim is rejected on OBJECT only on
        # positive evidence (no effect at all, another counterparty, another act); a clause of the claimed sentence that sets the
        # incompatible effect on the duty's own object or counterparty keeps the verifier's reading, as in t5 (GATE_UNREAD_ACT).
        for clause in [c.strip() for c in CLAUSE_END.split(context_of(span, passage)) if c.strip()]:
            stage, why = _unread_clause(clause, keys, polarity, duty, role, quantities)
            if stage == len(GATE_STAGES):
                return '', f'{GATE_UNREAD_ACT}: {why} in "{clause[:160]}"'
            best = max(best, (stage, why))
    return GATE_STAGES[best[0]], best[1]


def _object_keys(duty: dict, keys: set) -> set:
    """The duty's own object and counterparty words as act keys: the content keys of its action, object and item texts that are
    not its act keys (another act named as an object, "bildirim" of a disclosure ban, stays) and not the words of its conditions
    and exceptions (the situation the duty applies in is not what it acts on: "riskli ülkelerde yerleşik")."""
    duty = duty or {}
    texts = [str(e.get('text') or '') for e in duty.get('elements') or [] if e.get('kind') in ('action', 'object', 'item')]
    texts += [duty.get('prohibited_action') or duty.get('required_action') or '', *[str(i) for i in duty.get('items') or []]]
    situation = _content_keys(' '.join(str(c) for c in [*(duty.get('conditions') or []), *(duty.get('exceptions') or [])]))
    object_texts = [str(duty.get('object') or ''),
        *[str(e.get('text') or '') for e in duty.get('elements') or [] if e.get('kind') in ('object', 'item')],
        *[str(i) for i in duty.get('items') or []]]
    normalized = lambda value: ' '.join(fold(value).split()).strip(' .')
    condition_texts = {normalized(str(c)) for c in duty.get('conditions') or []}
    # Extraction can duplicate the entire condition into an object field. That
    # does not independently identify the object of the act.
    explicit_objects = _content_keys(' '.join(t for t in object_texts if normalized(t) not in condition_texts))
    # An exception commonly names the very object it qualifies. It must not
    # erase an explicit object from the production structure.
    return _content_keys(' '.join(texts)) - set(keys) - {DECLARE} - (situation - explicit_objects)


def _other_party(folded: str, role: dict) -> str:
    """The other counterparty a clause's act is for, '' when nothing shows one: another addressee than the duty's (recipient),
    or, against a duty that communicates to the authority, a customer in an oblique case with no authority named
    ("Müşterilerimizden kimlik belgesi istenmez": asking customers)."""
    theirs = recipient(folded)
    if role['recipient'] and theirs and theirs != role['recipient']:
        return theirs.lower().replace('_', ' ')
    if role['recipient'] == 'AUTHORITY' and not AUTHORITY.search(folded):
        party = next((w for w in re.findall('%s+' % W, folded) if w.startswith('müşteri') and OBLIQUE.search(w)), '')
        if party:
            return f'"{party}"'
    return ''


def _unread_clause(clause: str, keys: set, polarity: str, duty: dict, role: dict, quantities) -> tuple:
    """(stage, why) of a clause of the claimed sentence that names none of the duty's act words (_gate_clause found none):
    len(GATE_STAGES) when it still carries the claimed contradiction, i.e. it sets an effect incompatible with the duty's polarity
    (a negated verb, removing, excepting, limiting, discretionary, substituting or refusing wording against a requirement; a
    permission, exception or limit against a prohibition; a weaker number) on the duty's own object or counterparty, and nothing
    shows another counterparty or another act as its main verb; 3 (SCOPE) when that effect restates the duty's own exception or is
    for another customer group; 0 (OBJECT) otherwise. Measured by the t6 review: "MASAK'ın talep ettiği belgeler ... teslim
    edilmez", "... bilgi talepleri ... reddedilir", "Denetim elemanlarına müşteri dosyaları gösterilmez" against the duty to give
    the authority what it asks for were CONFLICTS in t5 and OBJECT in t6; the descriptive "... MASAK olup merkezi Ankara'dadır" (no
    effect) and "Müşterilerimizden kimlik belgesi istenmez" (asking customers) stay rejected."""
    folded = fold(clause)
    words = re.findall('%s+' % W, folded)
    none = (0, f"no clause names the duty's act ({', '.join(sorted(keys))})")
    if any(REFUSE.search(w) or w.startswith(DOUBLE_NEGATIVE) for w in verb_words(duty)):
        return none                   # the duty's own act refuses or prevents: a negation without it may say the same
    if polarity == 'REQUIRED':
        negated = next((w for i, w in enumerate(words) if NEG_WORD.search(w) or _polarity_at(words, i) == 'PROHIBITED'), '')
        strong = [p.search(folded) for p in (WAIVER, REFUSE)]
        weak = [p.search(folded) for p in (EXEMPTION, LIMITER, WEAKER, SUBSTITUTE)]
    else:
        negated = next((w for w in words if PERMIT_WORD.search(w)), '')
        strong = [PERMISSIVE.search(folded)]
        weak = [p.search(folded) for p in (EXEMPTION, LIMITER)]
    strong, weak = [m.group(0) for m in strong if m], [m.group(0) for m in weak if m]
    numbers = [s['text'] for s in structural_signals(duty, clause, quantities) if s['strength'] == 'strong' and s['type'] in GATE_NUMBERS]
    said = negated or (strong[0] if strong else '') or (weak[0] if weak else '') or (numbers[0] if numbers else '')
    if not said:
        return none
    named = sorted(_object_keys(duty, keys) & _content_keys(clause))
    party = bool(role['recipient']) and recipient(clause) == role['recipient']
    # A negated verb, a waiver or a refusal needs the duty's object or counterparty; limiting, excepting, discretionary or
    # substituting wording and a number alone need its counterparty or two of its object words ("Düşük riskli müşterilerin
    # işlemleri yalnızca yılda bir gözden geçirilir" shares only "riskli" with monitoring high-risk customers).
    if not party and (len(named) < 1 or (not negated and not strong and len(named) < 2)):
        return none
    other = _other_party(folded, role)
    if other:
        return 0, f"no clause names the duty's act ({', '.join(sorted(keys))}); the clause's act is for {other}"
    families = {ACT_FAMILY.get(k, k) for k in keys if k.startswith('@')}
    acts = [i for i, w in enumerate(words) if len(w) >= 3 and _gate_token(w).startswith('@')
            and ACT_FAMILY.get(_gate_token(w), _gate_token(w)) not in families]
    if acts and (acts[-1] >= len(words) - 3 or all(LIGHT.fullmatch(w) or NEG_WORD.search(w) or PERMIT_WORD.search(w)
                                                   for w in words[acts[-1] + 1:])):
        return 0, f"no clause names the duty's act ({', '.join(sorted(keys))}); the clause's main act is {_gate_token(words[acts[-1]])}"
    why = (f'"{said}" on the duty\'s own ' + (f'object ({", ".join(named)})' if named else 'counterparty')
           + f' with no word of its act ({", ".join(sorted(keys))}) read')
    restating = polarity == 'PROHIBITED' or not (negated or numbers)
    if restating and _own_exception(clause, duty):
        return 3, f"{why}, but the clause restates the duty's own exception"
    groups, clause_groups = set(role['groups']), _groups(folded)
    if groups and clause_groups and not groups & clause_groups:
        return 3, f'{why}, but only for {", ".join(sorted(clause_groups))}; the duty is for {", ".join(sorted(groups))}'
    return len(GATE_STAGES), why


# ---- the positive evidence gate (v0.19 t7, 26 September 2026) ---------------------------------------------------------------------
# Measured on the v019t6 live micro run (I04 md. 4(2), the prohibition to disclose that a report was made): a favourable reading was
# accepted on semantic closeness alone. The 8b verifier confirmed "Tüm şüpheli işlemler ilgili otoriteye raporlanır" as PARTIAL
# (subject and action covered, the prohibition missing): a requirement to report to the authority, offered for a prohibition to
# disclose to anyone else. The duty's "action" element carries its object ("... bildiriminde bulunulduğunu ..."), so naming the object
# passed as "action covered"; different_action and the conflict gate read such sentences only on conflict claims. A favourable reading
# (SUPPORTS or PARTIAL, fast or verifier) therefore stands only when nothing in the sentence(s) holding its quote shows that the
# sentence is about another duty. As in the conflict gate after review OVF-1, a reading is rejected only on POSITIVE evidence of a
# mismatch, never because a word is missing from a list; a sentence these readers cannot read keeps the model's reading:
#   - ACTOR: the sentence is about another party (role_mismatch: another transfer role or direction);
#   - RECIPIENT: every word of the duty's act in a clause (or, when the clause names none, its main verb of the same act family) is
#     addressed to another party than the duty's: reporting to the authority against disclosing to anyone else;
#   - ACTION: the sentence names no act of the duty's family, and every predicate it has is a known act (ACTS) of another family:
#     keeping records or monitoring offered for reporting. One unread predicate ("... yer alır ve ... teyit edilir") keeps the reading;
# and, on a reading the strong verifier gave (`verified`), also:
#   - POLARITY: every such word has an effect opposed to the duty's: a requirement to report, send or notify against a prohibition
#     to disclose, a negated act against a requirement. Read only in a clause with no wording that could reverse or limit it
#     (a negation or limiter against a prohibition, an order of two events against a requirement);
#   - OBJECT: of a prohibition (not a double negative), a reading that itself lists the prohibition element missing, on a sentence that
#     forbids none of the duty's acts: the object (a report) is named, the prohibited act and its prohibition are not;
#   - SCOPE: the sentence names, as parties, only customer groups the duty's subject does not (a reading that lists the subject
#     missing is left to the element rule OTHER_SUBJECT, which runs first).
# Another party or another act is never a contradiction of this duty (the conflict gate rejects such claims: ACTOR, OBJECT), so a fast
# reading showing one loses no question. An opposed effect, a missing prohibition or another group may be the very contradiction a
# CONFIRM question finds, so those passages still go to the verifier. Measured (v019t3 heldout8, I10 md. 4(2)): "şüpheli görülen
# işlemler iş ortağı kuruluşa bildirilir" was a fast SUPPORTS of the prohibition to disclose; its CONFIRM_COVERS answer was a
# contradiction the conflict gate kept. The engine records the code and makes the passage UNRELATED (the row is NO_EVIDENCE when
# nothing else supports the duty). The gate never makes a contradiction and never reads a conflict claim (conflict_gate does).
PG_ACTOR, PG_SCOPE, PG_RECIPIENT, PG_POLARITY, PG_ACTION, PG_OBJECT = (
    'POSITIVE_GATE_ACTOR', 'POSITIVE_GATE_SCOPE', 'POSITIVE_GATE_RECIPIENT', 'POSITIVE_GATE_POLARITY', 'POSITIVE_GATE_ACTION',
    'POSITIVE_GATE_OBJECT')
POSITIVE_GATE_CODES = (PG_ACTOR, PG_SCOPE, PG_RECIPIENT, PG_POLARITY, PG_ACTION, PG_OBJECT)
# Named in the v0.19 coverage prompt hash (engine.V19_HASH_PARTS 'positive_gate'), so a packet made without this gate is not carried
# forward as if current.
# positive-v2-review1 (t7 review 1): SCOPE reads the groups whose data the act is about (_scope_groups), OBJECT needs an unguarded
# statement about the object (_object_statement), RECIPIENT reads its own addressees (_gate_recipients: "X tarafına", "şirketin
# yetkilileri") and the duty's own exception limits only the kinds of addressee it names; the F1 escalation (opposed_effect_signal).
POSITIVE_GATE_VERSION = 'positive-v3-qualified-effect'
# The effects on the duty's act that oppose its polarity. Narrower than INCOMPATIBLE: permitting or making optional what the duty
# requires weakens it (a PARTIAL or a contradiction for the verifier to say), it does not forbid it.
OPPOSED = {'REQUIRED': ('PROHIBITED', 'EXEMPTED'), 'PROHIBITED': ('REQUIRED', 'PERMITTED', 'OPTIONAL')}
# SCOPE reads a customer group only where it names a party: a phrase of two words or more ("tüzel kişi", "ticaret siciline kayıtlı"),
# an entity ("dernek", "vakıf"), or a single group word followed by a party noun ("kurumsal müşterilerimiz"). Measured (v019t3 indep5):
# "Tüm çalışanlar kurumsal hesaplarına ... girer" names no customer group.
PARTY_NOUN = re.compile(r'(?:müşteri|kişi|customer|client|person|entit|üye|member)')
# Words that reverse the polarity read on a word nearby, so a requirement read with them is no evidence: an order of two events
# ("kimliği tespit edilmeden işlem yapılmaz" requires identifying first), a negative participle ("kimliği doğrulanmayan müşteriyle
# ... yapılmaz") and a negative condition ("bildirilmezse").
REVERSING = re.compile(r'%s{2,}(?:m[ae]d[ae]n|m[ae]d[ıi]kç[ae]|m[ae]ks[ıi]z[ıi]n|m[ae]y[ae]n%s*|m[ae]zs[ae]%s*|m[ae]s[ae]%s*)$' % (W, W, W, W))
# A predication ends at a comma or a coordinating conjunction ("... yer alır ve bu bilgilerin doğruluğu teyit edilir").
PREDICATION_END = re.compile(r',\s*|\s+(?:ve|veya|ya\s+da|ile|and|or)\s+')
# A finite Turkish verb or copula: the aorist and its negative ("alır", "eklenir", "vermez"), the copula ("muaftır"), the future, the
# progressive and "-mAktA"; with the plural person ending ("bildirirler"). A plural noun ("işlemler", "kayıtlar") is none (FINITE_PLURAL).
FINITE = re.compile(r'%s{2,}(?:[ıiuüae]r|m[ae]z|[dt][ıiuü]r|[ae]c[ae]k(?:t[ıi]r)?|[ıiuü]?yor|m[ae]kt[ae](?:d[ıi]r)?)(?:l[ae]r)?$' % W)
FINITE_PLURAL = re.compile(r'(?:[ıiuüae]r|m[ae]z|yor|[dt][ıiuü]r)l[ae]r$')
# t7 review 1 (F2): SCOPE compares the customer groups whose data the duty's act is about with those the sentence names as parties.
# A group of the duty counts only as a possessor of the act's object: a genitive ("Derneklerin kimlik tespitinde", "Sendika ve
# konfederasyonların", "tüzel kişinin unvanı"; a coordination ending in one: "Yabancı dernek ve vakıfların ... temsilciliklerinin"),
# or an English "of/for" ("identification of associations"). An addressee list in the nominative, the parties that act ("Kamu kurum
# ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar ... vermek zorundadır"), names no customer group:
# measured (t7 review, I08 md. 31(1)), SCOPE rejected "Bireysel müşterilere ait bilgiler ... Başkanlık istediğinde eksiksiz verilir".
GENITIVE = re.compile(r'%s{2,}(?:n[ıiuü]n|[ıiuü]n)$' % W)
NOT_GENITIVE = frozenset({'için'})
GENITIVE_REACH = 6                       # words after a group word within which its coordination may end in a genitive
OF_BEFORE = re.compile(r'(?:^|\s)(?:of|for)\s+(?:the\s+|all\s+|any\s+)?$')
# "gerçek ve tüzel kişiler": the natural persons of a coordination GROUPS reads as legal persons only.
COORDINATED_REAL = re.compile(NB + r'gerçek(?=\s+(?:ve|veya|ile|ya\s+da)\s+tüzel\s+kişi)' + NA)
# t7 review 1 (F4): the positive gate's own reading of an addressee (the t5/t6 AUTHORITY / THIRD_PARTY patterns stay as they are for
# the conflict gate). "tarafına" is the dative of its head noun: a third party only after a party noun or the genitive of a transaction
# ("müşteri tarafına", "işlemin tarafına"), not "talep eden kamu kurumu tarafına ibraz edilir". "yetkililer" governed by a genitive
# or an entity noun ("şirketin yetkililerine", "grup şirketlerinin yetkililerine", "muhabir bankanın yetkililerine") are that
# entity's officials, not the authority. Neither is read as an addressee: no RECIPIENT evidence, the reading stands.
GOVERNING = re.compile(r'(?:%s{2,}(?:n[ıiuü]n|[ıiuü]n)|(?:şirket|banka|kuruluş|firma|grup|işyeri|şube)%s*)$' % (W, W))
PARTY_HEAD = re.compile(r'(?:müşteri|kişi|customer|client|person|üye|member|%s*(?:işlem|sözleşme|transfer|ilişki)%s*n)' % (W, W))
# t7 review 1 (F3): wording that keeps a sentence about a prohibition's object from being evidence of another act on it: a negation,
# a prohibition, a limiter or an exception, and confidentiality ("... gizli tutulur", "... gizlidir", "kept strictly confidential").
CONFIDENTIAL = re.compile(NB + r'(?:gizli%s*|mahrem%s*|confidential%s*|secre(?:t|cy)%s*)' % (W, W, W, W) + NA)
# t7 review 1 (F1): the structural signal of a sentence that gives a prohibited act of the duty an opposed effect.
OPPOSED_EFFECT = 'OPPOSED_EFFECT'


def _gate_recipients(folded: str) -> list:
    """[(start, 'AUTHORITY' | 'THIRD_PARTY')] of the addressees a folded text names, in text order, as the positive gate reads them:
    recipient's patterns without "tarafına" after a head that is no party, and without "yetkililer" governed by an entity."""
    found = []
    for kind, pattern in (('AUTHORITY', AUTHORITY), ('THIRD_PARTY', THIRD_PARTY)):
        for match in pattern.finditer(folded):
            word, head = match.group(0), (folded[:match.start()].split() or [''])[-1]
            if kind == 'AUTHORITY' and word.startswith('yetkililer') and head not in NOT_GENITIVE and GOVERNING.search(head):
                continue
            if kind == 'THIRD_PARTY' and word == 'tarafına' and not PARTY_HEAD.match(head):
                continue
            found.append((match.start(), kind))
    return sorted(found)


def _possessor_groups(folded: str) -> set:
    """The customer groups (GROUPS, and the natural persons of "gerçek ve tüzel kişi") a folded text names as possessors: a group
    phrase in the genitive, or followed within its coordination (GENITIVE_REACH words, no comma) by a genitive; in English after "of" or
    "for"."""
    found = set()
    for name, pattern in [*GROUP.items(), ('gerçek', COORDINATED_REAL)]:
        for match in pattern.finditer(folded):
            own = folded[match.start():match.end()].split()
            reach = re.split(r'[,;:]', folded[match.end():], maxsplit=1)[0].split()[:GENITIVE_REACH]
            if any(GENITIVE.search(w) and w not in NOT_GENITIVE for w in own[-1:] + reach) or OF_BEFORE.search(folded[:match.start()]):
                found.add(name)
    return found


def _scope_groups(duty: dict) -> set:
    """The customer groups the duty's act is about (F2): those its subject or action names as possessors (_possessor_groups), and those
    of the subject where the duty's source sentence gives it a genitive ("Dernekler" read from "Derneklerin kimlik tespitinde"). An
    addressee list naming the parties that act gives none."""
    duty = duty or {}
    subject = fold(str(duty.get('subject') or ''))
    action = fold(str(duty.get('prohibited_action') or duty.get('required_action') or ''))
    found = _possessor_groups(subject) | _possessor_groups(action)
    sentence = fold(str(duty.get('source_sentence') or ''))
    at = sentence.find(subject) if subject.strip() else -1
    if at >= 0:
        found |= _possessor_groups(sentence[at:])
    return found


def _object_statement(sentence: str) -> str:
    """What the sentence predicates of the object (F3): its predicates (_predicate: a known act or another finite verb, 'unread')
    and the known acts (ACTS) it names in any word ("... are reported to the FIU"); '' when it has none, or when it has negating,
    prohibiting, limiting, excepting or confidentiality wording, which may well state the prohibition in other words ("... gizli
    tutulur", "müşteri ... bilgilendirilmez", "... uyarılmaz")."""
    folded = fold(sentence or '')
    words = re.findall('%s+' % W, folded)
    if any(NEG_WORD.search(w) or NEG_NOUN.search(w) for w in words) or NEGATION.search(folded) or PROHIBITION.search(folded) \
            or LIMITER.search(folded) or EXEMPTION.search(folded) or CONFIDENTIAL.search(folded):
        return ''
    predicates = {p for clause in CLAUSE_END.split(sentence or '') for part in PREDICATION_END.split(fold(clause))
                  for p in [_predicate(re.findall('%s+' % W, part))] if p}
    predicates |= {key for key in (_gate_token(w) for w in words if len(w) >= 3) if key.startswith('@') and key != DECLARE}
    return ', '.join(sorted(predicates))


def _finite(word: str) -> bool:
    """Whether `word` is a finite verb or copula (FINITE), not a plural noun."""
    if re.search(r'l[ae]r$', word) and not FINITE_PLURAL.search(word):
        return False
    return bool(FINITE.search(word))


def _closing(word: str) -> bool:
    """A word that closes a predicate after an act word and belongs to it: a light verb or a polarity word that names no act
    ("teyit edilir", "bildirilmesi zorunludur", "tespit edilmez")."""
    return bool(LIGHT.fullmatch(word)) or (not _gate_token(word).startswith('@') and bool(
        NEG_WORD.search(word) or PERMIT_WORD.search(word) or OPTIONAL_AFTER.match(word) or EXEMPTED_AFTER.match(word)))


def _predicate(words: list) -> str:
    """What ends one predication: the act key (ACTS) of its predicate when that is a known act used as a verb (finite, or closed by a
    light verb: "iletilir", "ibraz edilir"); 'unread' when it ends in another finite verb, a copula or a light verb ("yer alır",
    "muaftır", "konu edilir"); '' when it ends in a noun phrase (a heading, a list)."""
    i = len(words) - 1
    while i >= 0 and _closing(words[i]):
        i -= 1
    if i < 0:
        return ''
    closed, key = i < len(words) - 1, _gate_token(words[i])
    if key.startswith('@') and key != DECLARE:
        return key if closed or _finite(words[i]) else ''
    return 'unread' if closed or _finite(words[i]) else ''


def _main_act(words: list, families: set):
    """The index of the clause's main verb of one of the act `families`: its last act word of those families, used as a verb (finite, or
    closed by a light or polarity word) and followed by nothing else; None when there is none. "Tüm şüpheli işlemler ilgili otoriteye
    raporlanır" -> "raporlanır"; the heading "Şüpheli İşlem Bildirimi" has no verb."""
    acts = [i for i, w in enumerate(words) if len(w) >= 3 and ACT_FAMILY.get(_gate_token(w)) in families]
    if not acts:
        return None
    last = acts[-1]
    after = words[last + 1:]
    if all(_closing(w) for w in after) and (after or _finite(words[last])):
        return last
    return None


def _party_groups(folded: str) -> set:
    """The customer groups (GROUPS) a folded sentence names as parties (PARTY_NOUN): unlike _groups, "kurumsal hesap" is none."""
    found = set()
    for name, pattern in GROUP.items():
        for match in pattern.finditer(folded):
            after = folded[match.end():].split()[:1]
            if ' ' in match.group(0) or name in ('dernek', 'vakıf', 'sendika') or (after and PARTY_NOUN.match(after[0])):
                found.add(name)
    if COORDINATED_REAL.search(folded):
        found.add('gerçek')                      # t7 review 1 (F2): "gerçek ve tüzel kişi müşteriler" names both groups
    return found


def _addressees_before(words: list, index: int) -> set:
    """Every authority or third-party addressee (AUTHORITY, THIRD_PARTY) named between the act word before `index` and it, with
    _addressee_before's reading of a modifying party; that one names the last. "MASAK'a ve müşteriye bildirilir" is addressed to
    both, so it is no evidence against a duty to report to the authority."""
    return {kind for _, kind in _gate_between(words, index)}


def _gate_between(words: list, index: int) -> list:
    """_gate_recipients of the words between the act word before `index` and it, with _addressee_before's reading of a modifying
    party (t7 review 1, F4: the gate's own addressee reading)."""
    start = max([i + 1 for i in range(index) if _gate_token(words[i]).startswith('@')] or [0])
    between = ' '.join(word for j, word in enumerate(words[start:index], start) if not (j + 1 < index and words[j + 1] in MODIFIER_AFTER))
    return _gate_recipients(between)


def _gate_addressee(words: list, index: int) -> str:
    """The last addressee _gate_between reads before the act word at `index` ('' when none): _addressee_before for the gate."""
    found = _gate_between(words, index)
    return found[-1][1] if found else ''


def _verb_stems(duty: dict) -> set:
    """The act keys of the duty's own verb words (verb_words): a match on them is a form of its verb, whose polarity and addressee can
    be read; the content stems gate_acts adds for a duty without a readable verb (its subject's, its object's) are not."""
    return {_gate_token(word) for word in verb_words(duty)} - {DECLARE}


def _support_clause(clause: str, keys: set, polarity: str, duty: dict, role: dict, verified: bool):
    """None when the clause names no act of the duty (nor, as its main verb, one of the act's family); ('', why) when one such word
    is compatible with the duty, or is a word whose polarity these readers cannot read; (code, why) when every such word is
    addressed to another party (RECIPIENT) or, with `verified`, has an opposed effect (POLARITY)."""
    folded = fold(clause)
    words = re.findall('%s+' % W, folded)
    roots = tuple(key for key in keys if len(key) < STEM and not key.startswith('@'))
    objects = root_objects(duty)
    form = root_form(verb_roots(duty)) if not objects or objects & _content_keys(clause) else None
    own = [i for i, w in enumerate(words) if len(w) >= 3 and (_gate_token(w) in keys or w.startswith(roots) or bool(form and form.match(w)))]
    if not own:
        main = _main_act(words, {ACT_FAMILY[key] for key in keys if key in ACT_FAMILY})
        own = [main] if main is not None else []
    if not own:
        return None
    verbs = _verb_stems(duty)

    def readable(i):
        # A known act, or a form of the duty's own verb; a content stem of its subject or object is not read for an effect.
        key = _gate_token(words[i])
        return (key != DECLARE and (key.startswith('@') or key in verbs)) or words[i].startswith(roots) or bool(form and form.match(words[i]))

    unread = [words[i] for i in own if not readable(i)]
    if unread:
        return '', f'"{unread[0]}" names the duty\'s act in a word whose effect is not read'
    # Wording that could reverse or limit the reading of a polarity: against a prohibition a negation, a limiter, an exception or
    # the duty's own exception (a limiter also makes another addressee no evidence: "yalnızca denetim elemanlarına açıklanır");
    # against a requirement an order of two events or a negative participle or condition (REVERSING).
    if polarity == 'PROHIBITED':
        worded = bool(LIMITER.search(folded) or EXEMPTION.search(folded))
        restated = _own_exception(clause, duty)
        limited = worded or restated
        guarded = limited or any(NEG_WORD.search(w) or NEG_NOUN.search(w) for w in words) or bool(PROHIBITION.search(folded)
                                                                                                   or NEGATION.search(folded))
    else:
        worded = restated = limited = False
        guarded = any(REVERSING.search(w) for w in words)
    # t7 review 1 (F7): the duty's own exception restated limits the addressee only when the exception itself names that kind of
    # addressee. Measured: with the exception cut as "şüpheli işlem bildiriminde bulunulduğuna ... denetim elemanlarına ve ...
    # mahkemelere verilen bilgiler dışında", "Tüm şüpheli işlemler ilgili otoriteye raporlanır" shared the reported fact with it and
    # was read as the exception; the exception names no authority, so the report to the authority stays another recipient.
    excepted = {kind for _, kind in _gate_recipients(fold(' '.join(
        [*(duty.get('exceptions') or []), *(c for c in duty.get('conditions') or [] if EXEMPTION.search(fold(str(c))))])))} if restated else set()
    mismatch = None
    for i in own:
        act, word = _gate_token(words[i]), words[i]
        theirs = _gate_addressee(words, i) if ACT_FAMILY.get(act) == '@communicate' else ''
        effect = _polarity_at(words, i)
        if act not in keys and OPPOSITE_ACT.get(act) in keys:
            effect = OPPOSITE_POLARITY.get(effect, effect)    # "imha edilir" against a duty to keep: keeping is ended
        addressed = _addressees_before(words, i)
        if role['recipient'] and theirs and role['recipient'] not in addressed and not (worded or (restated and not addressed - excepted)):
            mismatch = (PG_RECIPIENT, f'"{word}" is addressed to {theirs.lower().replace("_", " ")}; the duty\'s act to '
                                      f'{role["recipient"].lower().replace("_", " ")}')
        elif verified and effect in OPPOSED[polarity] and not guarded and (
                act.startswith('@') or _same_qualified_action(duty, clause)):
            mismatch = mismatch or (PG_POLARITY, f'"{word}" is {effect}; the duty\'s act is {polarity}')
        else:
            return '', f'"{word}" ({act}) is {effect}' + (f', to {theirs.lower().replace("_", " ")}' if theirs else '')
    return mismatch


def _same_qualified_action(duty: dict, clause: str) -> bool:
    """A bare unknown verb does not establish opposite normative effects.

    Its object and manner qualifiers must also be represented. Otherwise a
    prohibition on treating someone badly could be mistaken for a prohibition
    on treating them fairly. Unknown semantic entailment stays with the model.
    """
    own = _content_keys(duty.get('prohibited_action') or duty.get('required_action') or '')
    own -= _verb_stems(duty) | GENERIC
    return bool(own) and own <= _content_keys(clause)


def _foreign_action(duty: dict, keys: set, sentence: str) -> str:
    """Why every predicate of the sentence is another known act than the duty's, '' when nothing shows it: the duty names known acts
    (its keys and core_acts), the sentence names no act of their families in any word, it has at least one predicate that is a known
    act and none that is unread (_predicate)."""
    mine = {key for key in keys | core_acts(duty) if key.startswith('@') and key != DECLARE}
    if not mine:
        return ''
    families = {ACT_FAMILY.get(key, key) for key in mine}
    tokens = [_gate_token(w) for w in re.findall('%s{3,}' % W, fold(sentence))]
    if any(ACT_FAMILY.get(t, t) in families for t in tokens if t.startswith('@')):
        return ''
    acts = []
    for clause in CLAUSE_END.split(sentence):
        for part in PREDICATION_END.split(fold(clause)):
            predicate = _predicate(re.findall('%s+' % W, part))
            if predicate == 'unread':
                return ''
            if predicate:
                acts.append(predicate)
    if not acts:
        return ''
    return f'the sentence\'s act ({", ".join(sorted(set(acts)))}) is not the duty\'s ({", ".join(sorted(mine))}) and it names none of the duty\'s'


def quoted_sentences(quote: str, passage: str = '') -> str:
    """The full sentence(s) of `passage` holding `quote` (context_of), without the sentence after it when the quote ends with its
    own full stop ("... işlemi MASAK'a bildirir." is one sentence, not two); the quote itself when the passage does not hold it."""
    span = (quote or '').strip()
    trimmed = span.rstrip('.;:!? ').strip()
    return context_of(trimmed if trimmed and passage and trimmed in passage else span, passage)


def support_gate(duty: dict, quote: str, passage: str = '', missing=(), verified: bool = True) -> tuple:
    """('', why) when nothing in the sentence(s) of `passage` holding `quote` shows that a favourable reading of it is about another
    duty; else (POSITIVE_GATE_* code, why) naming the dimension with positive evidence of a mismatch (see above), read in this order:
    ACTOR, SCOPE, then per clause RECIPIENT and POLARITY on the words of the duty's act, then ACTION and OBJECT when no clause names
    it. `missing` are the element ids the reading lists missing. `verified`: the reading is the strong verifier's; without it (a fast
    reading that stands) POLARITY, OBJECT and SCOPE are not read. A duty whose act or polarity cannot be read (no act keys; a
    permission) is not gated. Strings only (digest-safe)."""
    keys, polarity = gate_acts(duty)
    if polarity not in OPPOSED or not keys or not quote:
        return '', 'not gated: the duty\'s act or its polarity cannot be read'
    keys = keys | root_acts(duty)
    why = role_mismatch(duty, quote, passage, claim=False)
    if why:
        return PG_ACTOR, why
    role, sentence = duty_role(duty), quoted_sentences(quote, passage)
    groups, theirs = _scope_groups(duty), _party_groups(fold(sentence))
    if verified and groups and theirs and not groups & theirs:
        return PG_SCOPE, f'the sentence is written for {", ".join(sorted(theirs))}; the duty for {", ".join(sorted(groups))}'
    found = []
    for clause in [c.strip() for c in CLAUSE_END.split(sentence) if c.strip()]:
        verdict = _support_clause(clause, keys, polarity, duty, role, verified)
        if verdict is not None and not verdict[0]:
            return verdict
        found += [verdict] if verdict else []
    if found:
        # Every clause that names the act mismatches: another recipient is reported before an opposed effect.
        return next((v for v in found if v[0] == PG_RECIPIENT), found[0])
    why = _foreign_action(duty, keys, sentence)
    if why:
        return PG_ACTION, why
    kinds = {e.get('id'): e.get('kind') for e in (duty or {}).get('elements') or []}
    if verified and polarity == 'PROHIBITED' and any(kinds.get(i) == 'prohibition' for i in missing or ()) \
            and 'PROHIBITED' not in polarities(sentence, keys):
        # t7 review 1 (F3): only on a sentence that states something else of the object, with no wording that may state the
        # prohibition in other words ("... gizli tutulur", "müşteri ... bilgilendirilmez"): the verifier's own missing list alone is
        # no positive evidence.
        stated = _object_statement(sentence)
        if stated:
            return PG_OBJECT, ('the reading lists the prohibition missing and the sentence forbids none of the duty\'s acts '
                               f'({", ".join(sorted(keys))}): it names the object, not the prohibited act, and states of it only '
                               f'{stated}')
    return '', 'no evidence of another act, party, polarity or scope'


def opposed_effect_signal(duty: dict, quote: str, passage: str = '', missing=()):
    """t7 review 1 (F1): the strong structural signal OPPOSED_EFFECT when a favourable fast reading of a prohibition quotes a
    sentence that gives one of the duty's prohibited acts an opposed effect (support_gate POLARITY, read as on a verifier reading:
    unguarded, no negation, limiter or exception); else None. Such a sentence may perform the forbidden act ("Bildirim yapıldığı
    bilgisi ... uyum sorumlusuna iletilir"), the contradiction a fast SUPPORTS or PARTIAL left unread (v019t3 heldout8 and t4
    indep13, I05 md. 29(3), gold CONFLICT): the engine sends the passage to the verifier with it (escalation STRUCTURAL_SIGNAL)
    instead of letting the fast reading stand. The signal never makes a contradiction itself; the verifier and the conflict gate
    decide."""
    if not quote or gate_acts(duty)[1] != 'PROHIBITED':
        return None
    code, why = support_gate(duty, quote, passage, missing, verified=True)
    if code != PG_POLARITY:
        return None
    sentence = quoted_sentences(quote, passage)
    start = passage.find(sentence) if passage and sentence else -1
    return _signal(OPPOSED_EFFECT, 'strong', sentence, max(start, 0), f'the sentence gives the prohibited act an opposed effect: {why}')

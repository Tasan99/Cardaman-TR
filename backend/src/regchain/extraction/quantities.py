"""Numeric quantities and timing phrases of a duty or a policy passage, read deterministically.

A deadline, a retention period or a trigger amount is where a policy most often differs from
the regulation without saying "not": "en geç on iş günü içinde" against "ay sonunda toplu
olarak", "sekiz yıl" against "beş yıl", "185.000 TL ve üzeri" against "250.000 TL üzerindeki".
In the v0.18 final run the judge never saw these numbers: the duty it read carried no deadline
(`deadline` was None in all 93 rows). Turkish legislation writes numbers as words, often
fused ("yüzseksenbeşbin TL", "onbeşbin TL" in Tedbirler Yönetmeliği md. 5(1)), so a digit
pattern alone finds none of them.

parse_quantities() returns plain dicts; `amount` and `days` are Python numbers (int when whole).
They are not canonical evidence as they are: regchain.evidence forbids floats, so a record that
goes into a packet must pass through evidence_safe() first (structure.py does this).
"""
import re

L = 'A-Za-zÇĞİÖŞÜçğıöşüâîû'
NLB = r'(?<![%s])' % L          # not preceded by a letter
NLA = r'(?![%s])' % L           # not followed by a letter

TR_NUMBERS = {'sıfır': 0, 'bir': 1, 'iki': 2, 'üç': 3, 'dört': 4, 'beş': 5, 'altı': 6, 'yedi': 7, 'sekiz': 8, 'dokuz': 9,
              'on': 10, 'yirmi': 20, 'otuz': 30, 'kırk': 40, 'elli': 50, 'altmış': 60, 'yetmiş': 70, 'seksen': 80, 'doksan': 90,
              'yüz': 100, 'bin': 1000, 'milyon': 10 ** 6, 'milyar': 10 ** 9}
EN_NUMBERS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
              'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
              'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
              'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
              'million': 10 ** 6, 'billion': 10 ** 9}
SCALES = {'bin': 1000, 'milyon': 10 ** 6, 'milyar': 10 ** 9, 'thousand': 1000, 'million': 10 ** 6, 'billion': 10 ** 9,
          'mn': 10 ** 6, 'bn': 10 ** 9, 'k': 1000}


def _capital(word: str) -> str:
    first = {'i': 'İ', 'ı': 'I'}.get(word[0], word[0].upper())
    return '[%s%s]%s' % (first, word[0], word[1:])


_TR_ORDER = sorted(TR_NUMBERS, key=len, reverse=True)
_TR_TOKEN = '(?:%s)' % '|'.join(_TR_ORDER)
_EN_TOKEN = '(?:%s)' % '|'.join(sorted(EN_NUMBERS, key=len, reverse=True))
# A word made only of number words, fused or not: "on", "onbeş", "yüzseksenbeşbin". The whole
# word must be number words, so "onay", "altında", "bina" and the "yüz" of "yüz yüze" are not.
TR_RUN = (r'%s(?:%s)%s*%s(?:\s+%s+%s)*(?:\s+buçuk%s)?'
          % (NLB, '|'.join(_capital(t) for t in _TR_ORDER), _TR_TOKEN, NLA, _TR_TOKEN, NLA, NLA))
EN_RUN = r'(?i:\b%s(?:(?:\s+and\s+|\s+|-)%s)*\b)' % (_EN_TOKEN, _EN_TOKEN)
DIGITS = r'(?<![\w.,/])(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)(?![\d/]|[.,]\d)'
NUMBER = re.compile(r'(?P<digits>%s)|(?P<tr>%s)|(?P<en>%s)' % (DIGITS, TR_RUN, EN_RUN))
DIGIT_SCALE = re.compile(r'\s*(?P<word>bin|milyon|milyar)%s|\s*(?P<en>(?i:thousand|million|billion|mn|bn))\b|(?P<k>[kK])\b' % NLA)
RESTATED = re.compile(r'\s*\(\s*(?:%s|%s|%s)\s*\)' % (DIGITS, TR_RUN, EN_RUN))

# (pattern, unit, class). Turkish suffixes are listed, not wildcarded: "ay\w*" would read
# "iki ayrı işlem" as two months.
TR_UNITS = [
    (r'iş\s+gün(?:ü|ün|üne|ünü|ünde|ünden|ler|leri|lerinde|lük)?', 'iş günü', 'duration'),
    (r'gün(?:ü|ün|üne|ünü|ünde|ünden|e|de|den|dür|ler|leri|lerde|lerinde|lük|lüğüne)?', 'gün', 'duration'),
    (r'saat(?:i|in|e|te|ten|tir|ler|leri|lik)?', 'saat', 'duration'),
    (r'hafta(?:lık|ya|da|dan|dır|lar|ları|nın|yı)?', 'hafta', 'duration'),
    (r'ay(?:ı|ın|a|da|dan|dır|lar|ları|larda|lık|lığı)?', 'ay', 'duration'),
    (r'yıl(?:ı|ın|a|da|dan|dır|lar|ları|larda|lık|lığı)?', 'yıl', 'duration'),
    (r'sene(?:yi|nin|ye|de|den|dir|ler|leri|lik)?', 'yıl', 'duration'),
    (r'TL|TRY|Türk\s+[Ll]iras[ıi]|[Ll]ira(?:s[ıi])?', 'TL', 'money'),
    (r'[Aa]vro|[Ee]uro|EUR', 'EUR', 'money'),
    (r'ABD\s+[Dd]olar[ıi]?|[Dd]olar[ıi]?|USD', 'USD', 'money'),
    (r'GBP|[Ss]terlin', 'GBP', 'money'),
]
EN_UNITS = [
    (r'(?:business|working)[\s-]+days?', 'iş günü', 'duration'),
    (r'(?:calendar\s+)?days?', 'gün', 'duration'),
    (r'hours?', 'saat', 'duration'),
    (r'weeks?', 'hafta', 'duration'),
    (r'months?', 'ay', 'duration'),
    (r'years?', 'yıl', 'duration'),
    (r'euros?|EUR', 'EUR', 'money'),
    (r'(?:US\s+)?dollars?|USD', 'USD', 'money'),
    (r'pounds?(?:\s+sterling)?|GBP|sterling', 'GBP', 'money'),
    (r'TL|TRY|(?:Turkish\s+)?lira', 'TL', 'money'),
    (r'percent|per\s+cent', '%', 'percent'),
]
# An attached Turkish case suffix ("TL'yi", "TL’nin") belongs to the amount's word.
TR_UNIT = re.compile(r'\s*-?\s*(?:%s)(?:[\'’][a-zçğıöşü]+)?%s' % ('|'.join('(%s)' % p for p, _, _ in TR_UNITS), NLA))
EN_UNIT = re.compile(r'\s*-?\s*(?i:%s)\b' % '|'.join('(%s)' % p for p, _, _ in EN_UNITS))
PERCENT_AFTER = re.compile(r'\s?%')
CURRENCY_BEFORE = re.compile(r'(?:US\$|[£€$₺]|\b(?:USD|EUR|GBP|TRY))\s?$')
CURRENCY = {'£': 'GBP', '€': 'EUR', '$': 'USD', 'US$': 'USD', '₺': 'TL', 'USD': 'USD', 'EUR': 'EUR', 'GBP': 'GBP', 'TRY': 'TL'}
PERCENT_BEFORE = re.compile(r'(?:%\s?|[Yy]üzde\s+)$')
DAYS = {'saat': 1 / 24, 'gün': 1, 'iş günü': 1.4, 'hafta': 7, 'ay': 30, 'yıl': 365}

# Comparators: before the number ("en az 5 yıl", "at least 10 years") or after the amount
# ("185.000 TL veya üzerinde", "15.000 TL'yi aşan", "15.000 TL'nin altında").
COMPARATOR_BEFORE = [
    (re.compile(r'(?:\ben\s+az|\basgari|(?i:\bat\s+least|\ba\s+minimum\s+of|\bminimum\s+of|\bnot\s+less\s+than|\bno\s+less\s+than))\s+$'), '>='),
    (re.compile(r'(?:\ben\s+fazla|\bazami|(?i:\bat\s+most|\ba\s+maximum\s+of|\bmaximum\s+of|\bup\s+to|\bnot\s+more\s+than|\bno\s+more\s+than|\bnot\s+exceeding))\s+$'), '<='),
    (re.compile(r'(?i:\bmore\s+than|\bexceeding|\bin\s+excess\s+of|\bover|\babove|\bgreater\s+than)\s+$'), '>'),
    (re.compile(r'(?i:\bless\s+than|\bbelow|\bunder|\bfewer\s+than)\s+$'), '<'),
]
_AFTER = [
    (r'\s+(?:ve|veya|ya\s+da)\s+(?:üzeri|üzerinde|üzerindeki|üstü|üstünde|daha\s+fazla(?:sı)?)%s' % NLA, '>='),
    (r'\s+(?i:or\s+(?:more|above|over|greater)|and\s+(?:above|over|more))\b', '>='),
    (r'\s+(?:ve|veya|ya\s+da)\s+(?:altı|altında|altındaki|daha\s+az)%s' % NLA, '<='),
    (r'\s+(?i:or\s+(?:less|fewer|below|under)|and\s+(?:below|under))\b', '<='),
    (r'\s+kadar(?:ki|olan)?%s' % NLA, '<='),
    (r'\s+(?:aşan|aşması|aşarsa|aşkın|üzerinde|üzerindeki|üstünde|üstündeki|fazla)%s' % NLA, '>'),
    (r'\s+(?:altında|altındaki|az)%s' % NLA, '<'),
]
# The amount may be followed by a bracketed variant before its comparator: "185.000 TL (kripto
# varlık hizmet sağlayıcılar için onbeşbin TL) veya üzerinde".
COMPARATOR_AFTER = [(re.compile(r'(?:\s*\([^()]{0,160}\))?' + core), sign) for core, sign in _AFTER]
COMPARATOR_AFTER_BARE = [(re.compile(core), sign) for core, sign in _AFTER]
# Deadline words: before the quantity ("en geç", "within", "for") or after it ("içinde", "süre ile").
LEAD_BEFORE = re.compile(r'(?:\ben\s+geç|(?i:\bwithin|\bno\s+later\s+than|\bnot\s+later\s+than|'
                         r'\bfor(?:\s+at\s+least|\s+a\s+(?:minimum\s+)?period\s+of|\s+a\s+minimum\s+of)?))\s+$')
TRAIL_AFTER = re.compile(r'\s+(?P<trail>içinde|içerisinde|zarfında|süre\s*ile|süreyle|süresince|boyunca)%s' % NLA)
LIST_CLOSE = re.compile(r'(?:^|\s)(?:[a-zçğıöşü]|\d{1,2})\)$')


def turkish_lower(text: str) -> str:
    return text.replace('İ', 'i').replace('I', 'ı').lower()


def words_value(words: str, turkish: bool = True) -> float | None:
    """The value of a run of number words: "iki yüz elli bin" 250000, "yüzseksenbeşbin" 185000."""
    if turkish:
        pattern = re.compile('%s|buçuk' % _TR_TOKEN)
        tokens = []
        for word in turkish_lower(words).split():
            parts = pattern.findall(word)
            if ''.join(parts) != word:
                return None
            tokens += parts
        table = TR_NUMBERS
    else:
        tokens = [t for t in re.split(r'[\s-]+', words.lower()) if t and t != 'and']
        table = EN_NUMBERS
        if any(t not in table for t in tokens):
            return None
    if not tokens:
        return None
    total, current = 0, 0
    for token in tokens:
        if token == 'buçuk':
            current += 0.5
        elif token in ('yüz', 'hundred'):
            current = (current or 1) * 100
        elif token in SCALES:
            total += (current or 1) * SCALES[token]
            current = 0
        else:
            current += table[token]
    return total + current


def digits_value(digits: str) -> float:
    """"15.000" 15000, "15.000,50" 15000.5, "2,5" 2.5, "15,000" 15000, "1.5" 1.5."""
    if '.' in digits and ',' in digits:
        if digits.rfind(',') > digits.rfind('.'):
            return float(digits.replace('.', '').replace(',', '.'))
        return float(digits.replace(',', ''))
    if re.fullmatch(r'\d{1,3}(?:\.\d{3})+', digits):
        return float(digits.replace('.', ''))
    if re.fullmatch(r'\d{1,3}(?:,\d{3})+', digits):
        return float(digits.replace(',', ''))
    return float(digits.replace(',', '.'))


def plain(number):
    """An int when whole (rounded against binary noise: 10 iş günü is 14 days, not 14.000000000000002)."""
    if number is None:
        return None
    number = round(float(number), 6)
    return int(number) if number == int(number) else number


def inside_parentheses(text: str, pos: int) -> bool:
    """Whether `pos` sits inside an open bracket; list markers ("b) ...") are not brackets."""
    depth = 0
    for index in range(pos - 1, max(-1, pos - 240), -1):
        char = text[index]
        if char == ')' and not LIST_CLOSE.search(text[max(0, index - 3):index + 1]):
            depth += 1
        elif char == '(':
            if depth == 0:
                return True
            depth -= 1
    return False


def _comparator(text: str, start: int, end: int, bracketed: bool):
    """(sign, wording, side, phrase start, phrase end) of the comparator of one amount."""
    window = max(0, start - 40)
    for pattern, sign in COMPARATOR_BEFORE:
        found = pattern.search(text, window, start)
        if found:
            return sign, found.group().strip(), 'before', found.start(), end
    if bracketed:
        # The comparator after the closing bracket governs the amount inside it too.
        close = re.compile(r'[^()]{0,160}\)').match(text, end)
        if close:
            for pattern, sign in COMPARATOR_AFTER_BARE:
                found = pattern.match(text, close.end())
                if found:
                    return sign, found.group().strip(), 'after', start, end
        return None, None, None, start, end
    for pattern, sign in COMPARATOR_AFTER:
        found = pattern.match(text, end)
        if found:
            return sign, text[end:found.end()].strip(), 'after', start, found.end()
    return None, None, None, start, end


def _unit_after(text: str, pos: int, languages) -> tuple[str, str, int] | None:
    for language in languages:
        pattern, table = (TR_UNIT, TR_UNITS) if language == 'tr' else (EN_UNIT, EN_UNITS)
        found = pattern.match(text, pos)
        if found:
            index = next(i for i, value in enumerate(found.groups()) if value is not None)
            return table[index][1], table[index][2], found.end()
    return None


def _quantity(text: str, match: re.Match) -> dict | None:
    start, end = match.start(), match.end()
    if match.group('digits'):
        amount, languages = digits_value(match.group('digits')), ('tr', 'en')
        scale = DIGIT_SCALE.match(text, end)
        if scale:
            word = (scale.group('word') or scale.group('en') or scale.group('k')).lower()
            amount, end = amount * SCALES[word], scale.end()
    elif match.group('tr'):
        amount, languages = words_value(match.group('tr'), True), ('tr',)
    else:
        amount, languages = words_value(match.group('en'), False), ('en',)
    if amount is None:
        return None
    before = text[max(0, start - 7):start]
    currency = CURRENCY_BEFORE.search(before) if match.group('digits') else None
    percent = PERCENT_BEFORE.search(before)
    if currency:
        unit, unit_class = CURRENCY[currency.group().strip()], 'money'
        start = start - len(before) + currency.start()
    elif percent:
        unit, unit_class = '%', 'percent'
        start = start - len(before) + percent.start()
    else:
        restated = RESTATED.match(text, end)
        after = restated.end() if restated else end
        sign = PERCENT_AFTER.match(text, after)
        found = ('%', 'percent', sign.end()) if sign else _unit_after(text, after, languages)
        if not found:
            return None
        unit, unit_class, end = found
    bracketed = inside_parentheses(text, start)
    comparator, wording, side, phrase_start, phrase_end = _comparator(text, start, end, bracketed)
    window = max(0, start - 40)
    lead_found = LEAD_BEFORE.search(text, window, phrase_start if side == 'before' else start)
    trail_found = TRAIL_AFTER.match(text, end)
    trail = re.sub(r'\s+', ' ', trail_found.group('trail')).replace('süreile', 'süre ile') if trail_found else None
    lead = re.sub(r'\s+', ' ', lead_found.group()).strip().lower() if lead_found else trail
    if lead_found:
        phrase_start = min(phrase_start, lead_found.start())
    if trail_found:
        phrase_end = max(phrase_end, trail_found.end())
    return {'text': text[start:end], 'start': start, 'end': end, 'amount': plain(amount), 'unit': unit, 'unit_class': unit_class,
            'days': plain(amount * DAYS[unit]) if unit_class == 'duration' else None, 'comparator': comparator, 'lead': lead,
            # Beyond the shared contract: what structure.py builds its spans and wording from.
            'trail': trail, 'comparator_text': wording, 'comparator_side': side, 'parenthetical': bracketed,
            'phrase_start': phrase_start, 'phrase_end': phrase_end}


def parse_quantities(text: str) -> list[dict]:
    """Every number with a unit in `text`: durations, money and percentages; spans are exact slices."""
    found, cursor = [], 0
    for match in NUMBER.finditer(text or ''):
        if match.start() < cursor:
            continue
        try:
            quantity = _quantity(text, match)
        except (ValueError, KeyError, IndexError, StopIteration):
            quantity = None
        if quantity and quantity['start'] >= cursor:
            found.append(quantity)
            cursor = quantity['end']
    return found


_W = '[%s]' % L
_SUFFIXED = r'%s+(?:madan|meden|dan|den|tan|ten)' % _W
PERIOD_PATTERNS = [
    ('PERIOD_END', r'(?:(?:her|o|ilgili|takip\s+eden)\s+)?(?:ayın\s+son\s+iş\s+gün(?:ü|ünde|üne)?|ay(?:ın|ı)?\s+son(?:u|unda|una|undan)?|'
                   r'(?:dönem|yıl|çeyrek)\s+son(?:u|unda|una)?|yılsonu(?:nda)?)%s' % NLA),
    ('PERIOD_END', r'(?i:\b(?:at\s+)?(?:the\s+)?end\s+of\s+(?:the|each|every)\s+(?:calendar\s+)?(?:month|quarter|year|period|reporting\s+period)\b|'
                   r'\b(?:month|quarter|year|period)[-\s]end\b)'),
    # v0.19 round 5 (I06 md. 24/A(5)): "... izleyen iş günü toplu dosya hâlinde ... iletilir" is a batch (one noun between).
    ('BATCH', r'(?:toplu\s+(?:%s+\s+)?(?:olarak|halde|hâlde|halinde|hâlinde|şekilde)|topluca)%s|'
              r'(?i:\bin\s+(?:a\s+)?batch(?:es)?\b|\bin\s+bulk\b|\bbatched\b)' % (_W, NLA)),
    # "eş zamanlı" (simultaneously; md. 24/A(5) "transferle eş zamanlı olarak ... gönderilir") is an immediate timing.
    ('IMMEDIATE', r'(?:derhal|gecikmeksizin|gecikme\s+olmaksızın|vakit\s+geçirmeksizin|beklemeksizin|aynı\s+gün(?:de|ü)?(?:\s+içinde)?|'
                  r'eş\s*zamanlı(?:\s+olarak)?|(?<!hemen\s)hemen(?!\s+hemen))%s|'
                  r'(?i:\bimmediately\b|\bwithout\s+(?:undue\s+)?delay\b|\b(?:on\s+the\s+)?same[-\s]day\b|\bforthwith\b|\bsimultaneously\b)'
                  % NLA),
    ('ASAP', r'(?:(?:mümkün\s+olan\s+)?en\s+kısa\s+(?:sürede|zamanda)|ivedilikle|ivedi\s+olarak)%s|'
             r'(?i:\bas\s+soon\s+as\s+(?:reasonably\s+)?(?:possible|practicable)\b|\bpromptly\b)' % NLA),
    # Only the last event word here; the coordinated ones before it ("tesisinden veya işlem
    # yapılmadan önce") are added by _before_chain. One pattern for the whole chain backtracked
    # exponentially on a long "...den veya ...den veya" run.
    ('BEFORE', r'%s\s+önce%s|(?i:\b(?:before|prior\s+to)\b[^,.;:()]{1,60})' % (_SUFFIXED, NLA)),
]
PERIOD = re.compile('|'.join('(?P<%s_%d>%s(?:%s))' % (kind, i, NLB, pattern) for i, (kind, pattern) in enumerate(PERIOD_PATTERNS)))
BEFORE_CHAIN = re.compile(r'%s%s\s+(?:veya|ve|ya\s+da)\s+(?:%s+\s+){0,3}$' % (NLB, _SUFFIXED, _W))
PERIOD_DAYS = {'PERIOD_END': 30, 'BATCH': None, 'IMMEDIATE': 0, 'BEFORE': 0, 'ASAP': None}


def _before_chain(text: str, start: int, floor: int) -> int:
    """Start of "A-dan veya B-den önce" when the match began at B; never before `floor`."""
    for _ in range(4):
        found = BEFORE_CHAIN.search(text, max(floor, start - 120), start)
        if not found:
            break
        start = found.start()
    return start


def period_phrases(text: str) -> list[dict]:
    """Timing without a number: period end, batches, immediately, before an event, as soon as possible."""
    found, floor = [], 0
    for match in PERIOD.finditer(text or ''):
        kind = match.lastgroup.rsplit('_', 1)[0]
        start = _before_chain(text, match.start(), floor) if kind == 'BEFORE' and not match.group().lower().startswith(('before', 'prior')) \
            else match.start()
        span = text[start:match.end()].rstrip()
        found.append({'text': span, 'start': start, 'end': start + len(span), 'kind': kind, 'days': PERIOD_DAYS[kind]})
        floor = match.end()
    return found


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# At the same number a comparator moves the bound by an infinitesimal step: "8 yıldan fazla" keeps longer than
# "8 yıl", "1.000 TL'nin altında" caps lower than "en fazla 1.000 TL", "185.000 TL'yi aşan" starts above
# "185.000 TL ve üzeri". One rule for every direction instead of one special case per direction.
EDGE = {'>': 1, '<': -1}
DIRECTIONS = ('max', 'cap', 'min', 'floor')


def compare(duty_q: dict, passage_q: dict, direction: str) -> str:
    """'SAME' | 'STRICTER' | 'WEAKER' | 'INCOMPARABLE': the passage's quantity against the duty's.

    direction 'max': a deadline, less time is stricter ("en geç 10 gün" against "5 gün": STRICTER);
    'cap': a maximum amount or count, a lower one is stricter (the same rule as 'max'); 'min': a
    keeping duration or a minimum, more is stricter ("en az 8 yıl" against "10 yıl": STRICTER, "8
    yıl" against "5 yıl": WEAKER); 'floor': an amount at or above which the duty applies, a lower
    amount is stricter. Durations compare by approximate days (iş günü 1.4), money only within one
    currency; a record without `unit_class` (a period phrase) is INCOMPARABLE. At the same number the
    comparators decide (EDGE). Records whose numbers are decimal strings (structure.py, evidence-safe)
    compare the same way.
    """
    if not duty_q or not passage_q or direction not in DIRECTIONS:
        return 'INCOMPARABLE'
    kind = duty_q.get('unit_class')
    if kind is None or kind != passage_q.get('unit_class'):
        return 'INCOMPARABLE'
    if kind == 'duration':
        a, b = _number(duty_q.get('days')), _number(passage_q.get('days'))
    else:
        if duty_q.get('unit') != passage_q.get('unit'):
            return 'INCOMPARABLE'
        a, b = _number(duty_q.get('amount')), _number(passage_q.get('amount'))
    if a is None or b is None:
        return 'INCOMPARABLE'
    if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)):
        # "185.000 TL ve üzeri" against "185.000 TL'yi aşan": the policy misses the amount itself.
        a, b = EDGE.get(duty_q.get('comparator'), 0), EDGE.get(passage_q.get('comparator'), 0)
        if a == b:
            return 'SAME'
    less = b < a
    if direction == 'min':
        return 'WEAKER' if less else 'STRICTER'
    return 'STRICTER' if less else 'WEAKER'


def evidence_safe(value):
    """`value` with every float turned into an int (whole) or a decimal string: packets forbid floats."""
    if isinstance(value, float):
        number = round(value, 6)
        return int(number) if number == int(number) else format(number, 'f').rstrip('0').rstrip('.')
    if isinstance(value, dict):
        return {key: evidence_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [evidence_safe(item) for item in value]
    return value

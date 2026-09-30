"""Clause-level obligation frames: who, what, which product, where, on what condition, with what exception.

A frame is what one clause (clauses.py) says, read from its wording by rule and located in the stored text:

  kind / modality   a duty, a prohibition, a permission, an exception, or text that carries no duty for a company
                    (definition, scope, a reference to another regulation, a task of an authority, a penalty)
  actor             the addressee the clause names ("piyasaya sürenler", "üretenler, ithal edenler ve pazarlayanlar"),
                    as the activities and entity classes of the pack vocabulary; empty when the clause is impersonal
  activity          the act it regulates (advertising, sale, labelling, placing on the market ...)
  product           the products it names, or the regulation's own product scope when it says "bu Tebliğ kapsamındaki ürünler"
  facility / place  production sites it binds, and places where the act is forbidden or allowed
  condition         wording that narrows the duty; evaluable against the profile only when it is a product property
                    ("hacmen % 1,2’den fazla alkol içeren")
  exception         wording that lifts it ("ihraç amaçlı üretilenler hariç olmak üzere", a following "Ancak ..." sentence)
  quantity          limits with their unit and direction ("150 mg/L’den fazla olamaz")

Every element carries the exact span it was read from, so the frame can be checked against the source and a model's
reading of the same clause can be compared with it element by element (extraction.py). The lexicon is pack data
(data/lexicon.json); nothing here names a company or a brand.
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from ..extraction.classify import TR_ENFORCEMENT
from ..pilot.schema import Strict
from .clauses import Clause, blank_notes, split_clauses

LEXICON = Path(__file__).resolve().parent / 'data' / 'lexicon.json'
LETTERS = 'a-zçğıöşüâîû'
FRAME_RULES_VERSION = 'tr-frames-v1'

FrameKind = Literal['OBLIGATION', 'PROHIBITION', 'PERMISSION', 'EXCEPTION', 'DEFINITION', 'SCOPE', 'REFERENCE', 'DELEGATION',
                    'ENFORCEMENT', 'OTHER']
Modality = Literal['MUST', 'MUST_NOT', 'MAY']
DUTY_KINDS = ('OBLIGATION', 'PROHIBITION')

_FOLD = {ord('İ'): 'i', ord('I'): 'ı', ord('Â'): 'a', ord('â'): 'a', ord('Î'): 'i', ord('î'): 'i', ord('Û'): 'u', ord('û'): 'u'}


def fold(text: str) -> str:
    """Turkish lower case, one character per character, so a span in the folded text is a span in the source."""
    out = []
    for char in text:
        mapped = _FOLD.get(ord(char))
        if mapped is None:
            mapped = char.lower()
            if len(mapped) != 1:
                mapped = char
        out.append(mapped)
    return ''.join(out)


class Mention(Strict):
    id: str
    text: str
    start: int                      # offsets in the article text
    end: int
    classes: list[str] = []
    entities: list[str] = []
    where: Literal['CLAUSE', 'CHAPEAU', 'CLOSING', 'PREVIOUS'] = 'CLAUSE'
    note: str = ''


class Quantity(Strict):
    attribute: str | None = None
    value: float
    unit: str
    comparator: Literal['le', 'lt', 'ge', 'gt', 'eq']
    text: str
    start: int
    end: int
    role: Literal['LIMIT', 'CONDITION'] = 'LIMIT'


class Condition(Strict):
    kind: Literal['PRODUCT_ATTRIBUTE', 'PLACE', 'COUNTERPARTY', 'TIME', 'CHANNEL', 'OTHER']
    quote: str
    start: int
    end: int
    # A predicate over the profile: {'fact': 'product.abv_percent', 'op': 'gt', 'value': 1.2}. None when the condition
    # describes the circumstances of the act (a place, a time, the other party), which no profile states.
    predicate: dict | None = None


class ExceptionRule(Strict):
    effect: Literal['EXEMPTS', 'PERMITS', 'NARROWS']
    quote: str
    start: int
    end: int
    source_ref: str                 # the clause the wording sits in (its own clause, or a following "Ancak ..." sentence)
    predicate: dict | None = None
    about: Literal['PRODUCT', 'PLACE', 'ACTIVITY', 'ACTOR', 'OTHER'] = 'OTHER'


class Frame(Strict):
    ref: str
    regulation_id: str
    label: str
    heading: str = ''
    kind: FrameKind
    modality: Modality | None = None
    marker: str = ''                # the wording that decided kind and modality
    passive: bool = False
    transitional: bool = False
    text: str
    start: int
    end: int
    chapeau: str = ''
    closing: str = ''
    actors: list[Mention] = []
    authorities: list[Mention] = []
    activities: list[Mention] = []
    products: list[Mention] = []
    facilities: list[Mention] = []
    places: list[Mention] = []
    counterparties: list[Mention] = []
    quantities: list[Quantity] = []
    conditions: list[Condition] = []
    exceptions: list[ExceptionRule] = []
    references: list[str] = []
    # How the products were found: named in the clause, or taken from the regulation's scope article.
    product_basis: Literal['CLAUSE', 'REGULATION_SCOPE', 'NONE'] = 'NONE'
    topic: str = 'GENERAL'
    rules_version: str = FRAME_RULES_VERSION

    @property
    def duty_bearing(self) -> bool:
        return self.kind in DUTY_KINDS

    def activity_classes(self) -> list[str]:
        return list(dict.fromkeys(c for m in self.activities for c in m.classes))

    def product_classes(self) -> list[str]:
        return list(dict.fromkeys(c for m in self.products for c in m.classes))


# -- lexicon -----------------------------------------------------------------------------------------
class Lexicon:
    def __init__(self, data: dict):
        if data.get('format') != 'cardaman-tr-lexicon/1':
            raise ValueError('lexicon format is not cardaman-tr-lexicon/1')
        boundary = lambda pattern: re.compile(r'(?<![%s])(?:%s)' % (LETTERS, pattern))
        self.actors = [(a, re.compile(r'(?<![%s])(?P<stem>%s)(?P<suffix>[%s]*)' % (LETTERS, a['pattern'], LETTERS))) for a in data['actors']]
        self.authorities = [boundary(p) for p in data['authorities']]
        self.activities = [(a, boundary(a['pattern'])) for a in data['activities']]
        self.products = [(p, boundary(p['pattern'])) for p in data['products']]
        self.facilities = [(f, boundary(f['pattern'])) for f in data['facilities']]
        self.places = [(p, boundary(p['pattern'])) for p in data['places']]
        self.counterparties = [(c, boundary(c['pattern'])) for c in data['counterparties']]
        self.sale_activities = list(data['sale_activities'])
        self.regulation_scope = boundary(data['regulation_scope_phrases'])
        self.exception = boundary(data['exception_markers'])
        self.condition = boundary(data['condition_markers'])
        self.data = data


@lru_cache(maxsize=4)
def lexicon(path: str = str(LEXICON)) -> Lexicon:
    return Lexicon(json.loads(Path(path).read_text(encoding='utf-8')))


# -- wording rules -----------------------------------------------------------------------------------
QUOTED = re.compile(r'“[^”]*”|"[^"]*"|‘[^’]*’')
WORD = re.compile(r'[%s]+' % LETTERS)
MUST_WORDS = re.compile(r'(?<![%s])(?:zorunludur|zorundadır(?:lar)?|mecburdur(?:lar)?|mecburidir|şarttır|yükümlüdür(?:ler)?|'
                        r'gerekir|gereklidir|gerekmektedir|esastır|mükelleftir(?:ler)?|sorumludur(?:lar)?)(?![%s])' % (LETTERS, LETTERS))
PROHIBITED_WORDS = re.compile(r'(?<![%s])(?:yasaktır|yasaklanmıştır|izin verilmez|müsaade edilmez)(?![%s])' % (LETTERS, LETTERS))
MAY_WORDS = re.compile(r'(?<![%s])(?:serbesttir|mümkündür|izin verilebilir)(?![%s])' % (LETTERS, LETTERS))
NECESSITATIVE = re.compile(r'[%s]+(?:malı|meli)(?:dır|dir)(?:lar|ler)?$' % LETTERS)
NEGATIVE_NECESSITATIVE = re.compile(r'[%s]+(?:mamalı|memeli)(?:dır|dir)(?:lar|ler)?$' % LETTERS)
ABILITY_NEGATIVE = re.compile(r'[%s]+(?:amaz|emez)(?:lar|ler)?$' % LETTERS)
AORIST_NEGATIVE = re.compile(r'[%s]+(?:maz|mez)(?:lar|ler)?$' % LETTERS)
ABILITY = re.compile(r'[%s]+(?:abilir|ebilir)(?:ler|lar)?$' % LETTERS)
AORIST = re.compile(r'[%s]{2,}(?:[ıiuü]r|[ae]r)(?:lar|ler)?$' % LETTERS)
PASSIVE = re.compile(r'(?:[ıiuü]l|[ıiuü]n|[aeıioöuü]n|len|lan)(?:amaz|emez|maz|mez|[ıiuü]r|malıdır|melidir|mamalıdır|memelidir|abilir|ebilir)(?:lar|ler)?$')
# A negative that characterises rather than forbids: "kapsamaz" is scope, "değerlendirilmez" / "sayılmaz" is a legal reading.
NOT_A_DUTY = re.compile(r'^(?:kapsamaz|değerlendirilmez|sayılmaz|kabul edilmez(?:ler)?|aranmaz|uygulanmaz|gerekmez|oluşturmaz|'
                        r'kaldırmaz|etkilemez|doğurmaz|taşımaz|bulunmaz|düzenlenmez|verilemez ve)$')
# A positive aorist that deems, describes or defines rather than orders: "satış noktası olarak değerlendirilir",
# "gönüllülük esasına dayanır", "... sistemini gösterir".
STATEMENT = re.compile(r'^(?:değerlendirilir|sayılır|kabul edilir|addedilir|addolunur|anlaşılır|dayanır|oluşur|gösterir|belirtir|'
                       r'ifade eder|kapsar|içerir|girer|başlar|biter|geçer|teşkil eder|hükmündedir|kalkar|doğar)$')
EXEMPTING = re.compile(r'(?<![%s])(?:uygulanmaz|aranmaz|dışındadır|muaftır|muaf tutulur|zorunlu değildir|saklıdır|istisnadır|hariçtir|'
                       r'gerekmez|kapsamı dışındadır)(?![%s])' % (LETTERS, LETTERS))
DEFINITION_END = re.compile(r'(?<![%s])(?:ifade eder|anlamına gelir|ifade etmektedir)(?![%s])' % (LETTERS, LETTERS))
DEFINITION_ITEM = re.compile(r'^(?:[%s]{1,2}\)\s+)?[^:]{2,80}:\s' % LETTERS)
SCOPE_VERB = re.compile(r'(?<![%s])(?:kapsar|kapsamaz|kapsamaktadır)(?![%s])' % (LETTERS, LETTERS))
REFERENCE = re.compile(r'(?:hükümler[%s]*|hükümlerine|kurallara|bölümüne|maddesine|esaslara|usul ve esaslara|kriterlere)[^.]{0,60}?'
                       r'(?:uygulanır|uyulur|uygun olmalıdır|uygun olarak|tabidir|yerine getirilir)' % LETTERS)
REFERENCED_TEXT = re.compile(r'((?:[A-ZÇĞİÖŞÜ][%s]+,? (?:(?:ve|ile|veya) )?){1,14}(?:Yönetmeliği|Yönetmeliğin[%s]*|Tebliği|Tebliğin[%s]*|Kanunu|Kanunun[%s]*))'
                             % (LETTERS, LETTERS, LETTERS, LETTERS))
DELEGATION_VERB = re.compile(r'(?<![%s])(?:belirlenir|belirlenebilir|düzenlenir|yönetilir|yayımlanır|ilan edilir|tespit edilir|karar verilir|'
                             r'verilebilir|verilir|yapılır|yapılabilir|denetlenir|yürütülür|alınır|değerlendirilir|bildirilir|iptal edilir|'
                             r'yetkilidir|sorumlu ve yetkilidir|görevlidir)(?![%s])' % (LETTERS, LETTERS))
AGENTIVE = re.compile(r'^(?:ca|ce|ça|çe|nca|nce|larca|lerce|larınca|lerince)$')
GENITIVE = re.compile(r'^(?:ın|in|un|ün|nın|nin|nun|nün|ların|lerin)$')
POSSESSIVE = re.compile(r'^(?:ı|i|u|ü|sı|si|su|sü|ları|leri)$')
TARAFINDAN = re.compile(r'\s+tarafından')
TRANSITIONAL = re.compile(r'md\. Geçici ')
NON_DUTY_HEADINGS = (('tanım', 'DEFINITION'), ('amaç', 'SCOPE'), ('kapsam', 'SCOPE'), ('dayanak', 'OTHER'), ('yürürlük', 'OTHER'),
                     ('yürütme', 'OTHER'), ('ceza', 'ENFORCEMENT'), ('yaptırım', 'ENFORCEMENT'), ('avrupa birliği', 'OTHER'),
                     ('kısaltma', 'DEFINITION'))
# Units a beverage text states limits in; the engine's quantity parser knows durations, money and percent only.
NUMBER = r'(\d+(?:[.,]\d+)?)'
UNITS = (('mg/l', 'mg/L'), ('g/l', 'g/L'), ('mg/kg', 'mg/kg'), ('ml', 'ml'), ('mm', 'mm'), ('metre', 'm'), ('litre', 'L'))
QUANTITY = re.compile(r'(?:%%\s*%s)|(?:%s\s*(mg/l|g/l|mg/kg|ml(?![%s])|mm(?![%s])|metre|litre))' % (NUMBER, NUMBER, LETTERS, LETTERS))
WORD_NUMBERS = {'yüz': 100.0, 'iki yüz': 200.0, 'beş yüz': 500.0, 'bin': 1000.0, 'elli': 50.0}
WORD_QUANTITY = re.compile(r'(?<![%s])(yüz|iki yüz|beş yüz|bin|elli)\s+(metre)' % LETTERS)
ATTRIBUTES = (('kafein', 'caffeine_mg_per_l', 'mg/L'), ('taurin', 'taurine_mg_per_l', 'mg/L'), ('inositol', 'inositol_mg_per_l', 'mg/L'),
              ('glukoronolakton', 'glucuronolactone_mg_per_l', 'mg/L'), ('etil alkol', 'ethanol_g_per_l', 'g/L'),
              ('meyve oranı', 'fruit_percent', '%'), ('karbondioksit', 'co2_percent', '%'), ('malt', 'malt_percent', '%'),
              ('alkol', 'abv_percent', '%'), ('mesafe', 'distance_m', 'm'), ('punto', 'font_x_height_mm', 'mm'),
              ('x-yüksekliğ', 'font_x_height_mm', 'mm'))
PROPERTY_OF_PRODUCT = re.compile(r'\s*[’\']?\s*(?:den|dan|ten|tan)\s+(?:fazla|az|düşük)(?:\s+[%s]+)?\s+(?:olan|içeren)(?:\s+ürün[%s]*)?' % (LETTERS, LETTERS))
ABV_CONDITION = re.compile(r'hacmen\s*%%\s*%s\s*[’\']?\s*(?:(den|dan|ten|tan)\s+fazla|(ve\s+daha\s+fazla)|(ve\s+(?:daha\s+)?(?:az|düşük))|'
                           r'(den|dan|ten|tan)\s+(?:az|düşük))\s+alkol' % NUMBER)
CLOCK = re.compile(r'\d{1,2}[:.]\d{2}\s*(?:ila|ile|-|–)\s*\d{1,2}[:.]\d{2}(?:\s+saatleri\s+arasında)?')
EXPORT_ONLY = re.compile(r'ihraç amaçlı|ihraç edilmek (?:üzere|amacıyla)|ihracat amacıyla')
ANAPHORA = re.compile(r'(?<![%s])bu ürün' % LETTERS)
THIS_FIKRA = re.compile(r'bu fıkra')
THIS_ARTICLE = re.compile(r'bu madde')
TOPICS = (('ADVERTISING', ('ADVERTISING', 'PRODUCT_PROMOTION', 'SPONSORSHIP', 'PROMOTION', 'EVENTS')),
          ('LABELLING', ('LABELLING',)), ('BRANDING', ('BRAND_USE',)), ('ECOMMERCE', ('ECOMMERCE_SALE',)),
          ('SALE', ('SALE', 'RETAIL_SALE', 'WHOLESALE', 'ON_PREMISE_SERVICE')), ('EMPLOYMENT', ('EMPLOYMENT',)),
          ('PACKAGING', ('PACKAGING',)), ('MARKET_PLACEMENT', ('PLACING_ON_MARKET', 'IMPORT', 'EXPORT')),
          ('PRODUCTION', ('PRODUCTION', 'BOTTLING', 'WATER_EXTRACTION')), ('LOGISTICS', ('DISTRIBUTION', 'WAREHOUSING')))
TOPIC_WORDS = (('PACKAGING_WASTE', re.compile(r'ambalaj atı|depozito|geri kazanım|plastik poşet|ambalaj bilgi sistemi')),
               ('AGE_RESTRICTION', re.compile(r'on ?sekiz yaş|18 yaş')),
               ('CLAIMS', re.compile(r'beslenme beyan|sağlık beyan|beyanı ifade')),
               ('COMPOSITION', re.compile(r'mg/l|g/l|bileşim|katkı madde|ürün özellik|hammadde')),
               ('LICENSING', re.compile(r'satış belgesi|izin belgesi|uygunluk belgesi|üretim izni|tesis kurma izni|onay belgesi|kayıt belgesi|ruhsat')),
               ('HYGIENE', re.compile(r'hijyen')))


def _mask(text: str) -> str:
    """Notes and quoted wording blanked (same offsets): a label warning quoted in a clause is not the clause's own verb."""
    masked = blank_notes(text)
    return QUOTED.sub(lambda m: ' ' * (m.end() - m.start()), masked)


def _predicate(masked: str) -> tuple[str, int]:
    """(last word of the clause, its offset): the finite verb of a Turkish sentence closes it."""
    trimmed = masked.rstrip(' .;:,”"’)')
    words = list(WORD.finditer(fold(trimmed)))
    if not words:
        return '', 0
    return words[-1].group(), words[-1].start()


def _modality(folded: str) -> tuple[str | None, str, bool]:
    """(modality, marker, passive) from the closing predicate of one reading, or (None, '', False)."""
    word, position = _predicate(folded)
    if not word:
        return None, '', False
    tail = folded[:position + len(word)]
    previous = WORD.findall(tail)[-2] if len(WORD.findall(tail)) >= 2 else ''
    phrase = f'{previous} {word}'.strip()
    passive = bool(PASSIVE.search(word))
    if PROHIBITED_WORDS.search(tail[-40:]):
        return 'MUST_NOT', PROHIBITED_WORDS.search(tail[-40:]).group(), passive
    if MUST_WORDS.search(word) or MUST_WORDS.search(phrase):
        return 'MUST', (MUST_WORDS.search(phrase) or MUST_WORDS.search(word)).group(), False
    if MAY_WORDS.search(phrase):
        return 'MAY', MAY_WORDS.search(phrase).group(), passive
    if NEGATIVE_NECESSITATIVE.search(word):
        return 'MUST_NOT', word, passive
    if NECESSITATIVE.search(word):
        return 'MUST', word, passive
    if ABILITY.search(word):
        return 'MAY', word, passive
    if ABILITY_NEGATIVE.search(word) or AORIST_NEGATIVE.search(word):
        if NOT_A_DUTY.match(word) or NOT_A_DUTY.match(phrase):
            return None, phrase if NOT_A_DUTY.match(phrase) else word, passive
        return 'MUST_NOT', word, passive
    if word in ('değildir',):
        return None, phrase, False
    if STATEMENT.match(word) or STATEMENT.match(phrase):
        return None, phrase if STATEMENT.match(phrase) else word, passive
    if AORIST.search(word) and not word.endswith(('dır', 'dir', 'dur', 'dür', 'tır', 'tir', 'tur', 'tür')):
        return 'MUST', word, passive
    return None, '', False


def _plain_active_aorist(marker: str, passive: bool) -> bool:
    """A duty that rests on a bare active aorist ("karşılarlar", "talep eder") and on no duty word."""
    word = marker.split(' (')[0].split()[-1] if marker else ''
    return bool(word) and not passive and bool(AORIST.search(word)) and not MUST_WORDS.search(marker) \
        and not NECESSITATIVE.search(word)


def _find(patterns, folded: str, text: str, base: int, where: str = 'CLAUSE') -> list[Mention]:
    """Lexicon matches in one reading, longest first; a match inside a longer match of the same kind is dropped."""
    found = []
    for entry, pattern in patterns:
        for match in pattern.finditer(folded):
            if match.end() > match.start():
                found.append((match.start(), match.end(), entry))
    found.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    kept = []
    for start, end, entry in found:
        if any(start < b and a < end for a, b, _ in kept):
            continue
        kept.append((start, end, entry))
    kept.sort()
    return [Mention(id=entry.get('id') or entry.get('class'), text=text[a:b].strip(), start=base + a,
                    end=base + a + len(text[a:b].rstrip()),
                    classes=list(entry.get('classes') or ([entry['class']] if 'class' in entry else [])) if 'activities' not in entry
                    else list(entry['activities']), entities=list(entry.get('entities') or []), where=where)
            for a, b, entry in kept]


def _actors(lex: Lexicon, folded: str, text: str, base: int, where: str, active: bool) -> list[Mention]:
    """The addressees a reading names: an agent noun in the agentive ("sürenlerce", "... tarafından"), in the genitive
    before a nominalised verb ("isteyenlerin ... almaları zorunludur"), or in the nominative with an active verb."""
    candidates = []
    for entry, pattern in lex.actors:
        for match in pattern.finditer(folded):
            candidates.append((match.start(), match.end('stem'), match.end(), match.group('suffix'), entry))
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    kept = []
    for start, stem_end, end, suffix, entry in candidates:
        if any(start < b and a < end for a, _, b, _, _ in kept):
            continue
        kept.append((start, stem_end, end, suffix, entry))
    kept.sort()
    out = []
    for start, stem_end, end, suffix, entry in kept:
        after = folded[end:end + 14]
        agentive = bool(AGENTIVE.match(suffix)) or bool(TARAFINDAN.match(after))
        genitive = bool(GENITIVE.match(suffix))
        nominative = suffix == '' or (bool(POSSESSIVE.match(suffix)) and folded[end:end + 1] in (',', ';'))
        separated = folded[end:end + 1] in (',', ';', '/') or folded[end:end + 4] in (' ve ', ' vey') or folded[end:end + 6] == ' ile '
        if agentive or (genitive and where != 'CLAUSE') or (active and (nominative or genitive)) or (where == 'CHAPEAU' and (nominative or separated)):
            out.append(Mention(id=entry['id'], text=text[start:end], start=base + start, end=base + end,
                               classes=list(entry['activities']), entities=list(entry['entities']), where=where,
                               note='agentive' if agentive else 'genitive' if genitive else 'nominative'))
    return out


def _quantities(folded: str, text: str, base: int) -> list[Quantity]:
    out = []
    matches = [(m.start(), m.end(), float((m.group(1) or m.group(2)).replace('.', '').replace(',', '.')) if (m.group(1) or m.group(2)) else 0.0,
                '%' if m.group(1) else dict(UNITS)[m.group(3)]) for m in QUANTITY.finditer(folded)]
    matches += [(m.start(), m.end(), WORD_NUMBERS[m.group(1)], 'm') for m in WORD_QUANTITY.finditer(folded)]
    for start, end, value, unit in sorted(matches):
        before, after = folded[max(0, start - 60):start], folded[end:end + 40]
        if re.match(r'\s*[’\']?\s*(?:den|dan|ten|tan)\s+fazla\s+(?:ola(?:maz|mayacak)|kullanıla(?:maz)|içere(?:mez))', after) \
                or re.search(r'en (?:fazla|çok)(?:\s+[%s]+){0,2}\s*$' % LETTERS, before) or re.match(r'\s*(?:[’\']?\s*(?:i|ı|u|ü|yi|yı)\s+)?(?:aşamaz|geçemez)', after):
            comparator = 'le'
        elif re.search(r'en az(?:\s+[%s]+){0,2}\s*$' % LETTERS, before) or re.match(r'\s*[’\']?\s*(?:den|dan|ten|tan)\s+az\s+ola(?:maz)', after):
            comparator = 'ge'
        elif re.match(r'\s*[’\']?\s*(?:den|dan|ten|tan)\s+fazla', after):
            comparator = 'gt'
        elif re.match(r'\s*ve\s+daha\s+fazla', after):
            comparator = 'ge'
        elif re.match(r'\s*[’\']?\s*(?:den|dan|ten|tan)\s+(?:az|düşük)', after):
            comparator = 'lt'
        else:
            comparator = 'eq'
        # A dash list ("-İnositol 100 mg/L -Glukoronolakton 20 mg/L ... den fazla olamaz") shares one closing predicate.
        if comparator == 'eq' and re.search(r'(?:den|dan)\s+fazla\s+olamaz', folded[end:]) and re.search(r'[-–]\s*[%s]+\s*$' % LETTERS, before):
            comparator = 'le'
        attribute = None
        window = folded[max(0, start - 70):start] + ' ' + folded[end:end + 30]
        near = folded[max(0, start - 28):start]
        for word, name, wanted in ATTRIBUTES:
            if wanted == unit and (word in near or (attribute is None and word in window)):
                attribute = name
                if word in near:
                    break
        role = 'CONDITION' if comparator in ('gt', 'lt') or (unit == '%' and re.match(r'\s*ve\s+daha\s+fazla', after)) else 'LIMIT'
        out.append(Quantity(attribute=attribute, value=value, unit=unit, comparator=comparator, text=text[start:end],
                            start=base + start, end=base + end, role=role))
    return out


def _segment(folded: str, position: int) -> tuple[int, int]:
    """The comma-delimited segment of a sentence that holds a position."""
    start = max(folded.rfind(',', 0, position), folded.rfind(';', 0, position), folded.rfind(':', 0, position)) + 1
    ends = [i for i in (folded.find(',', position), folded.find(';', position)) if i != -1]
    end = min(ends) if ends else len(folded)
    while start < end and folded[start].isspace():
        start += 1
    return start, end


def _conditions(lex: Lexicon, folded: str, text: str, base: int, frame_places, frame_counterparties, quantities=()) -> list[Condition]:
    out = []
    for match in ABV_CONDITION.finditer(folded):
        value = float(match.group(1).replace(',', '.'))
        op = 'gt' if match.group(2) else 'ge' if match.group(3) else 'le' if match.group(4) else 'lt'
        out.append(Condition(kind='PRODUCT_ATTRIBUTE', quote=text[match.start():match.end()], start=base + match.start(),
                             end=base + match.end(), predicate={'fact': 'product.abv_percent', 'op': op, 'value': value}))
    for quantity in quantities:
        # "Kafein miktarı 1,0 mg/L’den fazla olan ürünlerde ...": a stated property that selects the products.
        a, b = quantity.start - base, quantity.end - base
        tail = PROPERTY_OF_PRODUCT.match(folded[b:b + 40])
        if quantity.role != 'CONDITION' or quantity.attribute in (None, 'abv_percent') or not tail:
            continue
        start = max(folded.rfind(',', 0, a), folded.rfind(';', 0, a), folded.rfind(')', 0, a), folded.rfind('.', 0, a)) + 1
        while start < a and folded[start].isspace():
            start += 1
        end = b + tail.end()
        out.append(Condition(kind='PRODUCT_ATTRIBUTE', quote=text[start:end], start=base + start, end=base + end,
                             predicate={'fact': f'product.{quantity.attribute}', 'op': quantity.comparator, 'value': quantity.value}))
    for match in CLOCK.finditer(folded):
        out.append(Condition(kind='TIME', quote=text[match.start():match.end()], start=base + match.start(), end=base + match.end()))
    for mention in frame_places:
        if mention.where == 'CLAUSE':
            out.append(Condition(kind='PLACE', quote=mention.text, start=mention.start, end=mention.end))
    for mention in frame_counterparties:
        if mention.where == 'CLAUSE' and mention.id == 'MINOR':
            out.append(Condition(kind='COUNTERPARTY', quote=mention.text, start=mention.start, end=mention.end))
    taken = [(c.start - base, c.end - base) for c in out]
    for match in lex.condition.finditer(folded):
        if match.group() in ('için', 'üzere', 'ise', 'amacıyla', 'itibaren'):
            continue                                          # purpose and time adjuncts are recorded only when evaluable
        a, b = _segment(folded, match.start())
        if lex.exception.search(folded, a, b) or any(a < e and s < b for s, e in taken):
            continue
        taken.append((a, b))
        out.append(Condition(kind='OTHER', quote=text[a:b].strip(), start=base + a, end=base + a + len(text[a:b].rstrip())))
    return sorted(out, key=lambda c: c.start)


def _exceptions(lex: Lexicon, folded: str, text: str, base: int, ref: str, products, places) -> list[ExceptionRule]:
    out = []
    for match in lex.exception.finditer(folded):
        if match.group() in ('uygulanmaz', 'aranmaz', 'dışındadır', 'zorunlu değildir', 'kapsamaz') or match.group().startswith(('muaf', 'istisna', 'kapsamı dışında')):
            continue                                          # the main predicate of an EXCEPTION clause, handled per clause
        if match.group().startswith('saklı'):
            a, b = _segment(folded, match.start())
        else:
            a, _ = _segment(folded, match.start())
            b = match.end()
        if any(a < e.end - base and e.start - base < b for e in out):
            continue
        quote = text[a:b].strip()
        start = base + a + (len(text[a:b]) - len(text[a:b].lstrip()))
        inside = lambda mention: start <= mention.start and mention.end <= start + len(quote)
        named = [m for m in products if inside(m)]
        predicate, about = None, 'OTHER'
        if EXPORT_ONLY.search(folded, a, b):
            predicate, about = {'fact': 'product.export_only', 'op': 'eq', 'value': True}, 'PRODUCT'
        elif named and any(m.classes for m in named):
            predicate = {'fact': 'product.class', 'op': 'in', 'value': sorted({c for m in named for c in m.classes})}
            about = 'PRODUCT'
        elif any(inside(m) for m in places) or re.search(r'mahal|yer(?:ler)?i|tesis', folded[a:b]):
            about = 'PLACE'
        out.append(ExceptionRule(effect='EXEMPTS', quote=quote, start=start, end=start + len(quote), source_ref=ref,
                                 predicate=predicate, about=about))
    return out


def _topic(frame_activities, folded: str) -> str:
    classes = {m.id for m in frame_activities}
    for topic, pattern in TOPIC_WORDS:
        if pattern.search(folded):
            return topic
    for topic, wanted in TOPICS:
        if classes & set(wanted):
            return topic
    return 'GENERAL'


def _heading_kind(heading: str) -> str | None:
    head = fold(heading)
    for word, kind in NON_DUTY_HEADINGS:
        if word in head:
            return kind
    return None


def frame_of(clause: Clause, regulation_id: str, lex: Lexicon | None = None) -> Frame:
    """The frame of one clause, read from its wording."""
    lex = lex or lexicon()
    text, base = clause.text, clause.start
    masked = _mask(text)
    folded = fold(masked)
    chapeau_folded = fold(_mask(clause.chapeau))
    closing_folded = fold(_mask(clause.closing))
    # -- kind and modality -------------------------------------------------------------------------
    modality, marker, passive = _modality(folded)
    source = 'CLAUSE'
    if modality is None and not marker and clause.closing:
        modality, marker, passive = _modality(closing_folded)
        source = 'CLOSING'
    if modality is None and not marker and clause.chapeau:
        modality, marker, passive = _modality(chapeau_folded)
        source = 'CHAPEAU'
    heading_kind = _heading_kind(clause.heading)
    reading = ' '.join(part for part in (chapeau_folded, folded, closing_folded) if part)
    authorities = [Mention(id='AUTHORITY', text=text[m.start():m.end()], start=base + m.start(), end=base + m.end())
                   for pattern in lex.authorities for m in pattern.finditer(folded)]
    references = [m.group(1).strip() for m in REFERENCED_TEXT.finditer(masked) if not m.group(1).startswith(('Bu ', 'Resmî ', 'Resmi '))]
    kind: str
    if heading_kind == 'DEFINITION' or DEFINITION_END.search(reading) and DEFINITION_ITEM.match(masked) or \
            (heading_kind is None and DEFINITION_END.search(folded)):
        kind = 'DEFINITION'
    elif heading_kind in ('SCOPE', 'OTHER', 'ENFORCEMENT') and not MUST_WORDS.search(folded):
        kind = heading_kind
    elif TR_ENFORCEMENT.search(masked) and not MUST_WORDS.search(folded):
        kind = 'ENFORCEMENT'
    elif SCOPE_VERB.search(folded) and re.match(r'\s*(?:\(\d+\)\s*)?bu (?:kanun|yönetmelik|tebliğ)', folded):
        kind = 'SCOPE'
    elif EXEMPTING.search(folded[-60:]) or (modality == 'MAY' and re.match(r'\s*(?:\(\d+\)\s*)?ancak', folded)):
        kind = 'EXCEPTION'
    elif _authority_task(lex, folded, modality, passive):
        kind = 'DELEGATION'
    elif REFERENCE.search(folded) and references and modality != 'MUST_NOT':
        kind = 'REFERENCE'
    elif modality == 'MUST_NOT':
        kind = 'PROHIBITION'
    elif modality == 'MUST':
        kind = 'OBLIGATION'
    elif modality == 'MAY':
        kind = 'PERMISSION'
    else:
        kind = 'OTHER'
    # -- elements ----------------------------------------------------------------------------------
    active = modality is not None and not passive
    actors = []
    if clause.chapeau:
        chapeau_text = clause.chapeau
        actors = _actors(lex, chapeau_folded, chapeau_text, clause.chapeau_start, 'CHAPEAU', True)
    if not actors:
        actors = _actors(lex, folded, text, base, 'CLAUSE', active)
    if kind == 'OBLIGATION' and _plain_active_aorist(marker, passive) and not any(a.note in ('nominative', 'agentive') for a in actors):
        # "Bu işaretleme, piyasaya sürenin kayıt altında olduğunu ... gösterir.": an active aorist whose subject is not an
        # addressee describes; Turkish drafting orders a company with a named subject, a passive or a duty word.
        kind, actors = 'OTHER', []
    activities = _find(lex.activities, folded, text, base)
    products = _find(lex.products, folded, text, base)
    facilities = _find(lex.facilities, folded, text, base)
    places = _find(lex.places, folded, text, base)
    counterparties = _find(lex.counterparties, folded, text, base)
    if clause.chapeau:
        chapeau = clause.chapeau
        activities += [m for m in _find(lex.activities, chapeau_folded, chapeau, clause.chapeau_start, 'CHAPEAU')
                       if m.id not in {a.id for a in activities}]
        products += _find(lex.products, chapeau_folded, chapeau, clause.chapeau_start, 'CHAPEAU')
        facilities += _find(lex.facilities, chapeau_folded, chapeau, clause.chapeau_start, 'CHAPEAU')
    exceptions = _exceptions(lex, folded, text, base, clause.ref, products, places)
    quantities = _quantities(folded, text, base)
    # A mention inside an agent-noun phrase ("alkollü içkileri üreten, ithal eden ve pazarlayan firmaların") says who,
    # and one inside an exception ("ihraç amaçlı üretilenler hariç") says what is left out: neither is the regulated act.
    spans = [(base + m.start(), base + m.end()) for _, pattern in lex.actors for m in pattern.finditer(folded)]
    spans += [(e.start, e.end) for e in exceptions]
    activities = [m for m in activities if m.where != 'CLAUSE' or not any(a <= m.start and m.end <= b for a, b in spans)]
    for mention in activities:
        if mention.id == 'SALE':
            mention.classes = list(lex.sale_activities)
    basis = 'CLAUSE' if products else 'REGULATION_SCOPE' if lex.regulation_scope.search(reading) else 'NONE'
    conditions = _conditions(lex, folded, text, base, places, counterparties, quantities)
    if kind == 'EXCEPTION' and EXEMPTING.search(folded[-60:]):
        marker = EXEMPTING.search(folded[-60:]).group()
    return Frame(ref=clause.ref, regulation_id=regulation_id, label=clause.label, heading=clause.heading, kind=kind,
                 modality=modality if kind in ('OBLIGATION', 'PROHIBITION', 'PERMISSION', 'REFERENCE') else None,
                 marker=f'{marker} ({source.lower()})' if marker and source != 'CLAUSE' else marker, passive=passive,
                 transitional=bool(TRANSITIONAL.search(clause.label)), text=text, start=clause.start, end=clause.end,
                 chapeau=clause.chapeau, closing=clause.closing, actors=actors, authorities=authorities, activities=activities,
                 products=products, facilities=facilities, places=places, counterparties=counterparties, quantities=quantities,
                 conditions=conditions, exceptions=exceptions, references=references, product_basis=basis,
                 topic=_topic(activities, reading))


def _authority_task(lex: Lexicon, folded: str, modality, passive: bool) -> bool:
    """A sentence whose grammatical agent is an authority ("Bakanlıkça belirlenir", "Ajans tarafından yönetilir",
    "... vermeye mahalli mülki amir yetkilidir"): a task of the administration, not a duty of a company."""
    if not DELEGATION_VERB.search(folded[-45:]):
        return False
    for pattern in lex.authorities:
        for match in pattern.finditer(folded):
            word_end = match.end()
            while word_end < len(folded) and folded[word_end].isalpha():
                word_end += 1
            suffix = folded[match.end():word_end]
            whole = folded[match.start():word_end]
            if whole.endswith(('ca', 'ce', 'ça', 'çe')) or TARAFINDAN.match(folded, word_end) or 'yetkili' in folded[-30:] \
                    or AGENTIVE.match(suffix or 'x'):
                return True
    return False


def frames_of(section: dict, regulation_id: str, lex: Lexicon | None = None) -> list[Frame]:
    """The frames of one provision, with each sentence-level exception attached to the duties it lifts.

    "Ancak, ... ihtisas fuarları ... düzenlenebilir." lifts the prohibitions before it in the same fıkra;
    "... bu fıkra hükmü uygulanmaz" every duty of the fıkra; "... bu hükmün dışındadır" the sentence before it.
    """
    lex = lex or lexicon()
    if 'DELETED_PROVISION' in section.get('quality_flags', ()):
        return []
    clauses = split_clauses(section)
    frames = [frame_of(clause, regulation_id, lex) for clause in clauses]
    by_fikra: dict[int, list[int]] = {}
    for index, clause in enumerate(clauses):
        by_fikra.setdefault(clause.fikra, []).append(index)
    for indices in by_fikra.values():
        since = []                                   # duty frames since the last exception sentence in this fıkra
        named = []                                   # the products the fıkra has named so far
        for index in indices:
            frame = frames[index]
            if frame.product_basis == 'CLAUSE':
                named = [m for m in frame.products if m.where != 'PREVIOUS']
            elif frame.product_basis == 'REGULATION_SCOPE' and named and ANAPHORA.search(fold(frame.text)):
                # "Bu ürünler basın ve yayın yoluyla ... satılamaz": the products of the sentence before, not the
                # regulation's whole scope.
                frame.products = [m.model_copy(update={'where': 'PREVIOUS'}) for m in named]
                frame.product_basis = 'CLAUSE'
            if frame.duty_bearing:
                since.append(index)
                continue
            if frame.kind != 'EXCEPTION':
                continue
            folded = fold(frame.text)
            if THIS_ARTICLE.search(folded):
                targets = [i for i, f in enumerate(frames) if f.duty_bearing]
            elif THIS_FIKRA.search(folded):
                targets = [i for i in indices if frames[i].duty_bearing]
            elif 'bu hükmün' in folded or 'bu hüküm' in folded:
                targets = since[-1:]
            else:
                targets = list(since)
            predicate, about = None, 'OTHER'
            if EXPORT_ONLY.search(folded):
                predicate, about = {'fact': 'product.export_only', 'op': 'eq', 'value': True}, 'PRODUCT'
            effect = 'PERMITS' if fold(frame.marker).endswith(('abilir', 'ebilir', 'abilirler', 'ebilirler')) else 'EXEMPTS'
            rule = ExceptionRule(effect=effect, quote=frame.text, start=frame.start, end=frame.end, source_ref=frame.ref,
                                 predicate=predicate, about=about)
            for target in targets:
                frames[target].exceptions.append(rule)
            since = []
    return frames


def regulation_frames(sections: list[dict], regulation_id: str) -> list[Frame]:
    return [frame for section in sections for frame in frames_of(section, regulation_id)]

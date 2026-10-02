"""Clause-level obligations and their applicability scope, extracted from the stored official texts.

frames.py reads what a clause says; this module turns a duty-bearing frame into

  an obligation   the clause, its modality, its elements and the exact quote, under a stable id, and
  a scope         core.ObligationScope with origin EXTRACTED and a resolved provision reference: the level the duty
                  lands on (activity, legal entity, facility or product), the classes it names, the product conditions
                  it states and the exceptions that lift it.

The scope is derived by rule from the frame's elements and says how (`basis`):

  ACTOR_EXPLICIT      the clause names its addressee ("piyasaya sürenler") -> the entities that carry out those activities
  GOVERNING_ACTIVITY  an impersonal clause ("reklamı yapılamaz") -> whoever carries out the act the predicate governs
  PRODUCT_PROPERTY    a composition or labelling rule -> the product itself
  FACILITY            a rule about a production site -> the facility
  UNCLEAR             the wording does not say whom it binds; the scope is routed and always answers UNKNOWN

Products come from the clause, or from the regulation's own scope article ("Bu Tebliğ enerji içeceklerini kapsar, sporcu
içeceklerini kapsamaz") when the clause says "bu Tebliğ kapsamındaki ürünler" or names none. Nothing is invented: a
clause whose addressee the rules cannot place is UNCLEAR, and a model reading of the same clause (ai.py) is compared
with this one element by element before either is trusted.
"""
import re
from typing import Literal

from ..evidence import digest
from ..pilot.schema import Strict
from .core import ObligationScope, RegulationMeta
from .corpus import CorpusStore
from .frames import (article_label, ABILITY_NEGATIVE, AORIST_NEGATIVE, LETTERS, MUST_WORDS, NECESSITATIVE, NEGATIVE_NECESSITATIVE, Frame, Mention,
                     fold, regulation_frames, _mask, _predicate)
from .packs import Registry

EXTRACTION_RULES_VERSION = 'tr-scope-rules-v1'
MARKETING = ('ADVERTISING', 'PRODUCT_PROMOTION', 'PROMOTION', 'SPONSORSHIP', 'EVENTS', 'BRAND_USE')
SALES = ('SALE', 'RETAIL_SALE', 'WHOLESALE', 'ON_PREMISE_SERVICE', 'ECOMMERCE_SALE')
SUPPLY = ('PRODUCTION', 'BOTTLING', 'PACKAGING', 'IMPORT', 'EXPORT', 'PLACING_ON_MARKET', 'DISTRIBUTION', 'WAREHOUSING', 'WATER_EXTRACTION')
GROUPS = (('MARKETING', MARKETING), ('SALES', SALES), ('SUPPLY', SUPPLY), ('LABELLING', ('LABELLING',)), ('EMPLOYMENT', ('EMPLOYMENT',)))
# Digital marketing is advertising by another channel; a profile that records it as its own activity is still reached.
IMPLIED = {'ADVERTISING': ('DIGITAL_MARKETING',), 'PRODUCT_PROMOTION': ('DIGITAL_MARKETING',), 'PRODUCTION': ('BREWING',)}
# What may stand between two coordinated acts: a connector and at most two modifying words ("reklamı ve tüketicilere
# yönelik tanıtımı"); a relative clause in between ("üretilen veya ithal edilen alkollü içkilerin ambalajları") is not
# coordination.
CONNECTOR = re.compile(r'^\s*(?:,|;|ve\b|veya\b|ile\b|ya da\b|/)\s*(?:\S+\s+){0,2}$')
# A mention in the locative says where or in what ("satışında", "etkinlik alanında"), not what is done.
LOCATIVE = re.compile(r'(?:[nl][dt][ae]|[dt][ae])(?:ki)?$')
LOCATIVE_AFTER = re.compile(r'^\s+(?:alan|yer|nokta|ünite)[%s]*[dt][ae]' % LETTERS)
PARTICIPLE_AFTER = re.compile(r'^\s+(?:edilen|eden|olan|yapılan|yapan)(?![%s])' % LETTERS)
# "İmal ve son kullanma tarihi ... yazılır": the date of an act is a thing on the label, not the act.
DATE_AFTER = re.compile(r'^\s+(?:ve son (?:kullanma|tüketim)\s+)?tarih')
VERB = re.compile(r'[%s]+' % LETTERS)
PRODUCTION_SITES = ('BREWERY', 'DISTILLERY', 'WINERY', 'MALTING_PLANT', 'BOTTLING_PLANT', 'WATER_BOTTLING_PLANT', 'WATER_SOURCE',
                    'MANUFACTURING_PLANT')
SCOPE_VERB = re.compile(r'(?<![%s])(kapsar|kapsamaktadır|kapsamaz)(?![%s])' % (LETTERS, LETTERS))
GENERIC = {'ALCOHOLIC_BEVERAGE', 'NON_ALCOHOLIC_BEVERAGE'}
OUT_OF_SCOPE = {'TOBACCO'}


class RegulationScope(Strict):
    """The products a regulation covers, from its own scope article or, failing that, from the catalogue tags."""
    regulation_id: str
    product_classes: list[str] = []
    excluded_product_classes: list[str] = []
    alcohol_scope: Literal['ALCOHOLIC', 'NON_ALCOHOLIC', 'ANY'] = 'ANY'
    basis: Literal['SCOPE_ARTICLE', 'CATALOGUE_TAGS', 'NONE'] = 'NONE'
    refs: list[str] = []
    # The activities the catalogue says the text addresses (pack metadata): the addressees of a clause that names none.
    activity_classes: list[str] = []
    default_level: str | None = None


class ExtractedObligation(Strict):
    obligation_id: str
    regulation_id: str
    version_id: str
    provision_ref: str
    kind: Literal['OBLIGATION', 'PROHIBITION']
    modality: Literal['MUST', 'MUST_NOT']
    topic: str
    group: str
    text: str
    frame: Frame
    scope: ObligationScope
    basis: Literal['ACTOR_EXPLICIT', 'GOVERNING_ACTIVITY', 'PRODUCT_PROPERTY', 'FACILITY', 'UNCLEAR']
    flags: list[str] = []
    rules_version: str = EXTRACTION_RULES_VERSION


def regulation_scope(frames: list[Frame], meta: RegulationMeta, vocabulary) -> RegulationScope:
    included, excluded, alcohol, refs = [], [], 'ANY', []
    activities = [t for t in meta.activity_tags if t in vocabulary.activity_classes]
    for frame in frames:
        if frame.kind != 'SCOPE' or 'kapsam' not in fold(frame.heading):
            continue
        folded = fold(frame.text)
        verbs = [(m.start(), m.group(1)) for m in SCOPE_VERB.finditer(folded)]
        for mention in frame.products:
            position = mention.start - frame.start
            after = [verb for start, verb in verbs if start >= position]
            if not after:
                continue
            refs.append(frame.ref)
            if after[0] == 'kapsamaz':
                excluded += mention.classes
            else:
                included += mention.classes
                if mention.id == 'ALCOHOLIC_BEVERAGE':
                    alcohol = 'ALCOHOLIC'
                elif mention.id == 'NON_ALCOHOLIC_BEVERAGE':
                    alcohol = 'NON_ALCOHOLIC'
    included = [c for c in dict.fromkeys(included) if c not in excluded]
    common = dict(regulation_id=meta.regulation_id, activity_classes=activities, default_level=meta.default_level)
    if included or excluded or alcohol != 'ANY':
        return RegulationScope(product_classes=included, excluded_product_classes=list(dict.fromkeys(excluded)),
                               alcohol_scope=alcohol, basis='SCOPE_ARTICLE', refs=list(dict.fromkeys(refs)), **common)
    tags = [t for t in meta.product_tags if t in vocabulary.product_classes]
    if tags:
        return RegulationScope(product_classes=tags, basis='CATALOGUE_TAGS', **common)
    return RegulationScope(**common)


def _verbs(frame: Frame) -> list[tuple[int, int]]:
    """Spans (in the clause text) of the finite verbs a duty hangs from: the closing predicate and every negative or
    necessitative form before it ("piyasaya arz edilmez ve reklamı yapılmaz")."""
    masked = _mask(frame.text)
    folded = fold(masked)
    word, position = _predicate(folded)
    spans = [(position, position + len(word))] if word else []
    for match in VERB.finditer(folded):
        token = match.group()
        if (ABILITY_NEGATIVE.search(token) or AORIST_NEGATIVE.search(token) or NECESSITATIVE.search(token)
                or NEGATIVE_NECESSITATIVE.search(token) or MUST_WORDS.search(token)) and (match.start(), match.end()) not in spans:
            spans.append((match.start(), match.end()))
    return sorted(spans)


def _adjunct(frame: Frame, folded: str, mention: Mention) -> bool:
    """A mention that is a place, a circumstance or part of a relative clause, not the act the predicate governs."""
    a, b = mention.start - frame.start, mention.end - frame.start
    word = folded[a:b]
    if mention.id == 'ECOMMERCE_SALE':
        return False                                     # "bilgi toplumu hizmetleri ... vasıtasıyla": the channel is the act
    return bool(LOCATIVE.search(word.split()[-1]) or LOCATIVE_AFTER.match(folded[b:b + 24]) or PARTICIPLE_AFTER.match(folded[b:b + 12])
                or DATE_AFTER.match(folded[b:b + 28]) or word.endswith(('ilen', 'ılan', 'ulan', 'ülen')))


def governing(frame: Frame, subject: str = 'first') -> list[Mention]:
    """The activity mentions the predicate governs: those that are the verb or stand right before one, and those
    coordinated with such a mention ("reklamı ve tüketicilere yönelik tanıtımı yapılamaz").

    `subject` says what a passive clause with no such mention falls back on: 'first' the first act it names (a statute's
    long subject), 'head' the first act that is the head of its noun phrase (followed by a comma: "markalarımızın
    logoları, ..." yes, "satış belgesi, ..." no), 'none' nothing."""
    verbs = _verbs(frame)
    folded = fold(frame.text)
    mentions = sorted((m for m in frame.activities if m.where == 'CLAUSE' and not _adjunct(frame, folded, m)), key=lambda m: m.start)
    chosen: list[Mention] = []
    for mention in reversed(mentions):
        a, b = mention.start - frame.start, mention.end - frame.start
        adjacent = any(a <= end and start - b <= 30 and not re.search(r'[,;]|:\s', folded[b:max(b, start)])
                       for start, end in verbs if end >= a)
        if not adjacent and chosen:
            following = min(chosen, key=lambda m: m.start)
            between = folded[b:following.start - frame.start]
            adjacent = len(between) <= 42 and bool(CONNECTOR.match(between))
        if adjacent:
            chosen.append(mention)
    if not chosen and mentions and frame.passive and subject != 'none':
        # A passive clause whose subject is long ("... firmaların isim, marka, logo, amblemleri ile ... görseller iş
        # yerlerinin içinde, ... bulundurulamaz"): the first act it names is its subject.
        if subject == 'first':
            chosen = mentions[:1]
        else:
            chosen = [m for m in mentions if folded[m.end - frame.start:m.end - frame.start + 1] == ','][:1]
    return sorted(chosen, key=lambda m: m.start)


def predicate_segments(frame: Frame) -> list[tuple[int, int]]:
    """(start, end) spans of the clause text, one per finite predicate: "... satılamaz, | ... oyun ve bahse konu edilemez."
    A statement that repeats the first half of such a clause has stated half of it."""
    spans, start = [], 0
    for _, end in _verbs(frame):
        if end > start:
            spans.append((start, end))
            start = end
    if not spans:
        return [(0, len(frame.text))]
    spans[-1] = (spans[-1][0], len(frame.text))
    return spans


def _classes(mentions, registry: Registry, frame: Frame | None = None) -> list[str]:
    known = registry.vocabulary.activity_classes
    # "perakende olarak satılamaz": the qualifier names the kind of sale, so the generic verb adds nothing to it.
    qualified = frame is not None and any(m.id in ('RETAIL_SALE', 'WHOLESALE') for m in frame.activities if m.where == 'CLAUSE')
    out = []
    for mention in mentions:
        if qualified and mention.id == 'SALE':
            continue
        for name in mention.classes:
            for value in (name, *IMPLIED.get(name, ())):
                if value in known and value not in out:
                    out.append(value)
    if qualified:
        for mention in (m for m in frame.activities if m.where == 'CLAUSE' and m.id in ('RETAIL_SALE', 'WHOLESALE')):
            out += [c for c in mention.classes if c in known and c not in out]
    return out


INGREDIENT_AFTER = re.compile(r'^\s+(?:içeren|ihtiva eden|kullanılarak|ilave edil)')
MADE_FROM_AFTER = re.compile(r'^\s+(?:üretilen|elde edilen|hazırlanan)')


def _ingredient(frame: Frame, mention: Mention) -> bool:
    """A product named as what something is made from or contains ("içme suyundan üretilen aromalı sular", "doğal
    mineralli su içeren ürünler"): an ingredient, not the product the clause is about."""
    if mention.where != 'CLAUSE':
        return False
    folded = fold(frame.text)
    a, b = mention.start - frame.start, mention.end - frame.start
    word = folded[a:b]
    return bool(INGREDIENT_AFTER.match(folded[b:b + 22])
                or (word.endswith(('dan', 'den', 'tan', 'ten')) and MADE_FROM_AFTER.match(folded[b:b + 22])))


def _scope_id(frame: Frame, group: str, multiple: bool) -> str:
    # The bents "c" and "ç" are different clauses: a Turkish letter keeps its own spelling in the id ("C" and "C2").
    label, _, place = fold(frame.ref).partition('/')
    plain = str.maketrans('çğıöşü', 'cgiosu')
    place = ''.join(plain[ord(ch)] and chr(plain[ord(ch)]) + '2' if ord(ch) in plain else ch for ch in place)
    ref = re.sub(r'[^A-Z0-9]+', '_', f'{label.translate(plain)}/{place}'.upper()).strip('_')
    return f'X_{ref}' + (f'_{group}' if multiple else '')


def scopes_of(frame: Frame, regulation: RegulationScope, registry: Registry, version_id: str) -> list[tuple[ObligationScope, str, str, list[str]]]:
    """[(scope, group, basis, flags)] for one duty-bearing frame; usually one."""
    vocabulary = registry.vocabulary
    flags = []
    inside_exception = lambda m: any(e.start <= m.start and m.end <= e.end for e in frame.exceptions)
    named = [m for m in frame.products if not inside_exception(m) and not _ingredient(frame, m)]
    own = [m for m in named if m.where == 'CLAUSE']
    if (named and all(m.id in OUT_OF_SCOPE for m in named)) or (own and all(m.id in OUT_OF_SCOPE for m in own)):
        return []                                         # a clause (or a list item) about tobacco only: outside the beverage packs
    named = [m for m in named if m.id not in OUT_OF_SCOPE]
    ids = {m.id for m in named}
    classes = list(dict.fromkeys(c for m in named for c in m.classes if c in vocabulary.product_classes))
    excluded = []
    alcohol = 'ANY'
    mixed = False
    if ids & GENERIC:
        # "alkollü içkiler": every alcoholic class, whatever narrower class the clause also names ("fermente alkollü içki
        # markası, distile alkollü içki markası olarak ..."). A clause that names both sides binds both, so no category
        # is set; whether it means both, or only what is made from one into the other, is not decidable by rule.
        generic = ids & GENERIC
        mixed = len(generic) == 2
        alcohol = 'ANY' if mixed else 'ALCOHOLIC' if generic == {'ALCOHOLIC_BEVERAGE'} else 'NON_ALCOHOLIC'
        other_side = [c for c in classes if alcohol != 'ANY' and vocabulary.alcohol_of(c) != alcohol]
        # Inside a text with its own product scope the generic word means that text's products ("Alkolsüz içecekler
        # tiplerine özgü tat ... olmalı" in the soft-drink communiqué does not reach bottled water).
        within = [c for c in regulation.product_classes if alcohol != 'ANY' and vocabulary.alcohol_of(c) == alcohol] \
            if regulation.basis == 'SCOPE_ARTICLE' else []
        if other_side:
            alcohol = 'ANY'                                # "alkollü içkiler ve alkolsüz bira": a named class of the other side
        elif within:
            classes, alcohol = within, 'ANY'
            flags.append('PRODUCT_FROM_REGULATION_SCOPE')
        else:
            classes = []
    elif not named:
        classes, excluded, alcohol = list(regulation.product_classes), list(regulation.excluded_product_classes), regulation.alcohol_scope
        if regulation.basis != 'NONE':
            flags.append('PRODUCT_FROM_REGULATION_SCOPE')
    conditions = [{**c.predicate, 'quote': c.quote} for c in frame.conditions if c.predicate]
    exceptions = []
    for rule in frame.exceptions:
        if rule.predicate and rule.predicate.get('fact') == 'product.class':
            if set(rule.predicate['value']) & set(c for m in named for c in m.classes):
                # "Enerjisi azaltılmış meyveli şuruplar hariç olmak üzere, meyveli şuruplarda ...": the exception is a
                # part of the class the clause is about, told apart by a property no class carries. It is kept for the
                # reviewer and excludes nothing.
                exceptions.append({'quote': rule.quote, 'effect': 'NARROWS', 'about': 'PRODUCT', 'source_ref': rule.source_ref})
                continue
            excluded += [c for c in rule.predicate['value'] if c not in excluded]
            classes = [c for c in classes if c not in rule.predicate['value']]
            continue
        exceptions.append({**(rule.predicate or {}), 'quote': rule.quote, 'effect': rule.effect, 'about': rule.about,
                           'source_ref': rule.source_ref})
    if any(not e.get('fact') for e in exceptions):
        flags.append('EXCEPTION_UNEVALUATED')
    if frame.transitional:
        flags.append('TRANSITIONAL')
    base = dict(regulation_id=frame.regulation_id, topic=frame.topic, description_tr=frame.text[:400], provision_ref=frame.ref,
                provision_status='RESOLVED', origin='EXTRACTED', alcohol_scope=alcohol, product_classes=classes,
                excluded_product_classes=excluded, conditions=conditions, exceptions=exceptions, quote=frame.text,
                version_id=version_id)
    out = []
    # A bent that is an item of a list ("a) Gıdanın adı.", "c) Spor müsabakası yapılan stadyum ve kapalı spor salonlarında.")
    # has no predicate of its own: the lead-in of the fıkra says who is bound and what is done, and a word inside the item
    # ("muhafaza", "gıda işletmecisinin adı") is content, not addressing.
    list_item = frame.marker.endswith(('(chapeau)', '(closing)'))
    governed = [m for m in frame.activities if m.where == 'CHAPEAU'] if list_item else governing(frame)
    groups = [(name, [m for m in governed if m.id in members]) for name, members in GROUPS]
    groups = [(name, found) for name, found in groups if found]
    actors = [a for a in frame.actors if (a.classes or a.entities) and (a.where == 'CHAPEAU' or not list_item)]
    sites = list(dict.fromkeys(c for m in frame.facilities for c in m.classes if c in PRODUCTION_SITES))
    # The fallback for a clause that names no addressee of its own: the activities the regulation's scope article names,
    # else the level the catalogue gives the text (a product standard speaks about the product).
    def fallback(extra):
        if regulation.default_level == 'PRODUCT' and not mixed:
            return (dict(base, level='PRODUCT'), 'PRODUCT', 'PRODUCT_PROPERTY', flags + extra + ['LEVEL_FROM_CATALOGUE_DEFAULT'])
        # A statute speaks to many parties (Kanun 4250 md. 6 binds producers, sellers and broadcasters in turn): its
        # catalogue tags are no default. A by-law or communiqué is written for the activity its scope article names.
        if regulation.activity_classes and not mixed and not frame.regulation_id.startswith('TR:KANUN:'):
            return (dict(base, level='LEGAL_ENTITY', activity_classes=list(regulation.activity_classes)), 'REGULATION',
                    'GOVERNING_ACTIVITY', flags + extra + ['ADDRESSEE_FROM_CATALOGUE_TAGS'])
        return (dict(base, level='LEGAL_ENTITY', scope_status='UNCLEAR'), 'UNCLEAR', 'UNCLEAR', flags + extra)

    # "işletmeci", "işleticilerce", "... işletmek isteyenler": the operator a by-law is written for. Who that is, the text's
    # own scope says (its catalogue activities); the word itself names no activity.
    operators = [a for a in frame.actors if a.id == 'OPERATOR' and (a.where == 'CHAPEAU' or not list_item)]
    distance = [q for q in frame.quantities if q.attribute == 'distance_m']
    sold_at = [m for m in frame.activities if m.where == 'CLAUSE' and m.id in SALES]
    if not actors and operators and regulation.activity_classes:
        out.append((dict(base, level='LEGAL_ENTITY', activity_classes=list(regulation.activity_classes)), 'ACTOR', 'ACTOR_EXPLICIT',
                    flags + ['ACTOR_IS_REGULATION_PARTY']))
    elif not actors and distance and sold_at:
        # "... perakende veya açık olarak satışının yapıldığı yerler ile ... arasında ... en az yüz metre mesafenin bulunması
        # zorunludur": a rule about where a sale is made binds whoever sells there; a distance is no property of a product.
        out.append((dict(base, level='LEGAL_ENTITY', activity_classes=_classes(sold_at, registry, frame)), 'SALES', 'GOVERNING_ACTIVITY',
                    flags + ['ADDRESSEE_IMPLICIT', 'PLACE_OF_ACTIVITY']))
    elif actors:
        activity = list(dict.fromkeys(c for a in actors for c in a.classes if c in vocabulary.activity_classes))
        # A party that is defined by what it is, not by an activity of the vocabulary (a packaging producer, a contract
        # supplier): the entity class decides.
        entity = [] if activity else list(dict.fromkeys(c for a in actors for c in a.entities if c in vocabulary.entity_classes))
        out.append((dict(base, level='LEGAL_ENTITY', activity_classes=activity, entity_classes=entity), 'ACTOR', 'ACTOR_EXPLICIT', flags))
    elif any(a.id != 'OPERATOR' for a in frame.actors) and not list_item:
        out.append(fallback(['ACTOR_UNMAPPED']))
    else:
        for name, found in groups:
            if name == 'MARKETING':
                out.append((dict(base, level='ACTIVITY', activity_classes=_classes(found, registry)), name, 'GOVERNING_ACTIVITY',
                            flags + ['ADDRESSEE_IMPLICIT']))
            elif name in ('SALES', 'SUPPLY'):
                out.append((dict(base, level='LEGAL_ENTITY', activity_classes=_classes(found, registry, frame)), name,
                            'GOVERNING_ACTIVITY', flags + ['ADDRESSEE_IMPLICIT']))
            elif name == 'LABELLING' and mixed:
                out.append((dict(base, level='PRODUCT', scope_status='UNCLEAR'), name, 'UNCLEAR', flags + ['PRODUCT_SCOPE_MIXED']))
            elif name == 'LABELLING':
                out.append((dict(base, level='PRODUCT'), name, 'PRODUCT_PROPERTY', flags))
            elif name == 'EMPLOYMENT':
                context = [m for m in frame.activities if m.id != 'EMPLOYMENT' and m.where == 'CLAUSE']
                out.append((dict(base, level='LEGAL_ENTITY', activity_classes=_classes(context, registry)), name,
                            'GOVERNING_ACTIVITY', flags + ['ADDRESSEE_IMPLICIT']))
        if not out:
            limited = any(q.role == 'LIMIT' and q.attribute != 'distance_m' for q in frame.quantities)
            if sites and frame.topic in ('LICENSING', 'PRODUCTION', 'HYGIENE', 'GENERAL'):
                out.append((dict(base, level='FACILITY', facility_classes=sites), 'FACILITY', 'FACILITY', flags))
            elif mixed:
                out.append((dict(base, level='PRODUCT', scope_status='UNCLEAR'), 'PRODUCT', 'UNCLEAR', flags + ['PRODUCT_SCOPE_MIXED']))
            elif limited or frame.topic in ('COMPOSITION', 'LABELLING', 'CLAIMS') or (classes and frame.product_basis == 'CLAUSE'
                                                                                         and frame.topic != 'GENERAL'):
                out.append((dict(base, level='PRODUCT'), 'PRODUCT', 'PRODUCT_PROPERTY', flags))
            elif classes and frame.product_basis == 'CLAUSE' and not frame.places and not frame.counterparties:
                out.append((dict(base, level='PRODUCT'), 'PRODUCT', 'PRODUCT_PROPERTY', flags + ['ADDRESSEE_IMPLICIT']))
            else:
                out.append(fallback(['ADDRESSEE_UNCLEAR']))
    multiple = len(out) > 1
    channels, packaging = _channels(frame, registry), _packaging(frame, registry)
    built = []
    for fields, group, basis, scope_flags in out:
        if channels and fields.get('level') in ('LEGAL_ENTITY', 'ACTIVITY', 'FACILITY') and fields.get('scope_status') != 'UNCLEAR':
            fields = dict(fields, sales_channels=channels)
            scope_flags = scope_flags + ['SALES_CHANNEL_NAMED']
        # a packaging named in a product duty narrows the products ("cam ambalajlarda etiket bilgileri ... baskı"); in a duty of
        # the party ("depozito yönetim sistemi ... için Ajansa kayıt") it says what the duty is about, not whom it binds
        if packaging and fields.get('level') == 'PRODUCT' and fields.get('scope_status') != 'UNCLEAR':
            fields = dict(fields, product_attributes=list(dict.fromkeys([*fields.get('product_attributes', []), *packaging])))
            scope_flags = scope_flags + ['PACKAGING_NAMED']
        scope = ObligationScope(scope_id=_scope_id(frame, group, multiple), **fields)
        built.append((scope, group, basis, list(dict.fromkeys(scope_flags))))
    return built


def _lexicon_hits(registry: Registry, key: str, folded: str) -> list[str]:
    from .frames import lexicon
    known = getattr(registry.vocabulary, key, {}) or {}
    out = []
    for entry in lexicon().data.get(key, []):
        if entry['id'] in known and re.search(r'(?<![%s])(?:%s)' % (LETTERS, entry['pattern']), folded):
            out.append(entry['id'])
    return list(dict.fromkeys(out))


def _own_text(frame: Frame) -> str:
    """The clause without its exceptions, folded: what an exception names is not what the duty is about."""
    chars = list(frame.text)
    for rule in frame.exceptions:
        a, b = rule.start - frame.start, rule.end - frame.start
        if 0 <= a < b <= len(chars):
            chars[a:b] = ' ' * (b - a)
    carried = frame.chapeau if frame.marker.endswith(('(chapeau)', '(closing)')) else ''
    return fold(carried + ' ' + ''.join(chars))


def _channels(frame: Frame, registry: Registry) -> list[str]:
    """The sales channels a duty about selling names (lexicon: sales_channels). Only a duty that governs a sale, or is
    about distance selling, is narrowed: a channel word in a licence's name ("toptan satış belgesi") or in a web
    address ("Kurumun internet sitesinde") is no channel of the duty."""
    about_sale = any(m.id in SALES for m in governing(frame)) or frame.topic == 'ECOMMERCE'
    return _lexicon_hits(registry, 'sales_channels', _own_text(frame)) if about_sale else []


def _packaging(frame: Frame, registry: Registry) -> list[str]:
    """The packaging a duty is written for (lexicon: product_tags): deposit-return packaging, glass, PET, a can."""
    return _lexicon_hits(registry, 'product_tags', _own_text(frame))


# The topic of a clause that names none of its own, from what the whole text is about (its printed title).
TITLE_TOPICS = (('etiketleme', 'LABELLING'), ('beyan', 'CLAIMS'), ('ambalaj atık', 'PACKAGING_WASTE'), ('geri kazanım', 'PACKAGING_WASTE'),
                ('reklam', 'ADVERTISING'), ('mesafeli', 'ECOMMERCE'), ('elektronik ticaret', 'ECOMMERCE'), ('hijyen', 'HYGIENE'),
                ('satışına ve sunumuna', 'SALE'), ('kayıt ve onay', 'LICENSING'))


def obligations_of(frames: list[Frame], meta: RegulationMeta, registry: Registry, version_id: str) -> list[ExtractedObligation]:
    regulation = regulation_scope(frames, meta, registry.vocabulary)
    title = fold(meta.title)
    fallback_topic = next((topic for word, topic in TITLE_TOPICS if word in title), 'GENERAL')
    out = []
    for frame in frames:
        if not frame.duty_bearing:
            continue
        if frame.topic == 'GENERAL' and fallback_topic != 'GENERAL':
            frame.topic = fallback_topic
        for scope, group, basis, flags in scopes_of(frame, regulation, registry, version_id):
            out.append(ExtractedObligation(
                obligation_id=digest([meta.regulation_id, version_id, frame.ref, group])[:24], regulation_id=meta.regulation_id,
                version_id=version_id, provision_ref=frame.ref, kind=frame.kind, modality=frame.modality, topic=frame.topic,
                group=group, text=frame.text, frame=frame, scope=scope, basis=basis, flags=flags))
    return out


def extract_regulation(regulation_id: str, registry: Registry, store: CorpusStore | None = None, version_id: str | None = None,
                       articles=None) -> tuple[list[Frame], list[ExtractedObligation]]:
    """(every frame, the obligations) of one stored version; `articles` limits the result to some paragraph numbers.
    The regulation's scope article is always read, whatever articles are asked for."""
    store = store or CorpusStore()
    version = store.version(regulation_id, version_id)
    sections = store.sections(regulation_id, version.version_id)
    frames = regulation_frames(sections, regulation_id)
    obligations = obligations_of(frames, registry.regulations[regulation_id], registry, version.version_id)
    if articles is not None:
        wanted = {str(a) for a in articles}
        labels = {s['printed_label'] for s in sections if s['paragraph_number'] in wanted}
        frames = [f for f in frames if f.label in labels]
        obligations = [o for o in obligations if o.frame.label in labels]
    return frames, obligations


# -- grounding and routing ----------------------------------------------------------------------------
def ground(obligation: ExtractedObligation, store: CorpusStore) -> dict:
    """The validator stage: every quote the obligation rests on must be an exact span of the stored version it names.
    {'grounded': bool, 'checks': [...]}; a failed check names the element."""
    label = obligation.frame.label
    try:
        text = store.section(obligation.regulation_id, label, obligation.version_id)['text']
    except (KeyError, ValueError) as exc:
        return {'grounded': False, 'checks': [{'element': 'provision', 'ok': False, 'detail': str(exc)}]}
    frame = obligation.frame
    checks = [{'element': 'clause', 'ok': text[frame.start:frame.end] == frame.text}]
    for group in ('actors', 'activities', 'products', 'facilities', 'places', 'counterparties'):
        for mention in getattr(frame, group):
            if mention.where != 'PREVIOUS' and text[mention.start:mention.end] != mention.text:
                checks.append({'element': f'{group}:{mention.id}', 'ok': False})
    for condition in frame.conditions:
        if text[condition.start:condition.end] != condition.quote:
            checks.append({'element': 'condition', 'ok': False})
    texts = {label: text}

    def article(ref: str) -> str:
        # An exception may sit in another article ("... 9 uncu maddenin birinci fıkrasının (b) ve (j) bentlerinde ...
        # zorunlu değildir" in md. 19): its quote is checked against the article it is quoted from.
        other = article_label(ref) if ref else label
        if other not in texts:
            try:
                texts[other] = store.section(obligation.regulation_id, other, obligation.version_id)['text']
            except (KeyError, ValueError):
                texts[other] = ''
        return texts[other]

    for rule in frame.exceptions:
        if article(rule.source_ref)[rule.start:rule.end] != rule.quote:
            checks.append({'element': 'exception', 'ok': False})
    for item in (*obligation.scope.conditions, *obligation.scope.exceptions):
        if item.get('quote') and item['quote'] not in article(item.get('source_ref', '')):
            checks.append({'element': 'scope quote', 'ok': False})
    return {'grounded': all(c['ok'] for c in checks), 'checks': checks, 'rules': [frame.rules_version, obligation.rules_version]}


def route(obligation: ExtractedObligation, profile, registry: Registry, store: CorpusStore | None = None, packs=()):
    """(decisions, rollups) of one extracted obligation on one enterprise profile, each with the four audit stages:
    parsed = the frame the rules read, validator = the grounding check, final = the routed status. An obligation whose
    quotes are not in the stored text answers UNKNOWN / GROUNDING_FAILED on every target."""
    from .routing import DecisionAudit, evaluate_scope, rollup
    verdict = ground(obligation, store or CorpusStore())
    frame = obligation.frame
    parsed = {'kind': frame.kind, 'modality': frame.modality, 'marker': frame.marker, 'basis': obligation.basis,
              'flags': obligation.flags, 'actors': [a.text for a in frame.actors], 'activities': [a.id for a in frame.activities],
              'products': [p.text for p in frame.products], 'conditions': [c.quote for c in frame.conditions],
              'exceptions': [e.quote for e in frame.exceptions]}

    def audited(decision):
        audit = DecisionAudit(raw=None, parsed=parsed, validator=verdict, final=decision.audit.final)
        if verdict['grounded']:
            return decision.model_copy(update={'audit': audit, 'review_required': decision.review_required
                                               or 'EXCEPTION_POSSIBLE' in decision.reason_codes})
        final = {**decision.audit.final, 'status': 'UNKNOWN', 'reason_codes': ['GROUNDING_FAILED']}
        return decision.model_copy(update={'status': 'UNKNOWN', 'reason_codes': ['GROUNDING_FAILED'], 'applies_to_products': [],
                                           'review_required': True, 'audit': audit.model_copy(update={'final': final})})

    decisions = [audited(d) for d in evaluate_scope(obligation.scope, profile, registry, packs=packs)]
    rollups = [audited(d) for d in rollup(obligation.scope, decisions, profile, registry, packs=packs)]
    return decisions, rollups

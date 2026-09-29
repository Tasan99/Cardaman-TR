"""Entity gate (v0.16.1): which parties a clause binds or concerns, against the company profile.

A provision that applies to the company as an obliged party still carries sub-paragraphs about
particular counterparties (associations, foundations, unions, political parties, unincorporated
bodies, non-residents ...) or particular obliged parties (banks, foreign-headquartered firms'
Turkish branches). Measured live (Tedbirler Yönetmeliği md. 8 against a payment institution
whose customers are individuals, SMEs and merchants): every association/foundation duty came
back APPLIES because md. 4 lists payment institutions. The gate reads the clause's own words
and the profile's own words, by rule, and never asks a model:

  required entity (subject or counterparty of the clause) x company entity / customer families
    -> MATCH | MISMATCH | UNDETERMINED | NOT_RESTRICTED

Only an all-round MISMATCH narrows the provision-level answer to DOES_NOT_APPLY, and it says
which clause, which entity and which profile field decided it. Anything the profile does not
state (residency, correspondent relationships, a bare "legal entities" customer base) stays
UNDETERMINED and leaves the provision-level answer alone.

v0.19: the company's kind and the obliged parties a clause names are read with one taxonomy
(applicability.CATEGORIES plus the generic categories "finansal kuruluş" and "finansal olmayan belirli iş
ve meslekler"); the obliged parties a subject names are alternatives (obliged_verdict).
"""
import re
from dataclasses import dataclass, field

from regchain.ingestion.mevzuat import article_heading
from .policies import fold

# The party the clause is about (identified, transacted with). Order matters: a specific
# reading ("yabancı dernek") suppresses the general word inside it; "tüzel kişi" is not read
# inside "tüzel kişiliği olmayan".
COUNTERPARTIES = (
    ('FOREIGN_ASSOCIATION_BRANCH', r'yabancı (?:dernek|vakıf)'),
    ('ASSOCIATION', r'\bdernek'),
    ('FOUNDATION', r'\bvak[ıi]f'),
    ('UNION', r'\bsendika|\bkonfederasyon'),
    ('POLITICAL_PARTY', r'siyas[iî] parti'),
    ('UNINCORPORATED', r'tüzel kişiliği olmayan|\bapartman|site veya iş han'),
    ('PUBLIC_BODY', r'\bkamu kurum'),
    ('NON_RESIDENT', r'yurt dışında yerleşik|\bnon-resident'),
    ('CORRESPONDENT', r'\bmuhabir|\bcorrespondent bank'),
    # v0.19: "gerçek ve tüzel kişiler" names both (Tedbirler md. 25(1)); "tüzel kişiliği olmayan" is not a legal entity,
    # but it no longer hides one named beside it ("gerçek ve tüzel kişiler, tüzel kişiliği olmayan teşekküller").
    ('INDIVIDUAL', r'\bgerçek kişi|\bgerçek (?:ve|veya|ya da) tüzel kişi|\bretail client|\bconsumers?\b|\bindividuals?\b'),
    ('LEGAL_ENTITY', r'ticaret siciline kayıtlı|\btüzel kişi(?!liği olmayan)|\bşirketler|\bşirketin|\bcompan(?:y|ies)\b|\bcorporate client'),
)
SUPPRESSES = {'FOREIGN_ASSOCIATION_BRANCH': {'ASSOCIATION', 'FOUNDATION'}, 'NON_RESIDENT': set(), 'CORRESPONDENT': set()}
# Who must act. GENERIC ("yükümlüler", "a firm") restricts nothing.
OBLIGED = (
    ('FOREIGN_HQ_OBLIGED', r'merkezi yurt dışında'),
    ('BANK', r'\bbankalar\b|\bbanka\b|\bbanks?\b'),
    ('PAYMENT_INSTITUTION', r'ödeme kuruluş|payment institution'),
    ('EMONEY_INSTITUTION', r'elektronik para kuruluş|e-?money institution'),
    ('INSURANCE', r'sigorta şirket|sigorta ve emeklilik|\binsurers?\b'),
    ('BROKER', r'aracı kurum|portföy yönetim|\binvestment firms?\b'),
    ('EXCHANGE_OFFICE', r'yetkili müessese|döviz büro'),
    # v0.19 generic categories (applicability.GENERIC_CATEGORIES); membership is the taxonomy's (applicability.membership).
    ('DNFBP_GENERIC', r'finansal olmayan belirli iş ve meslek|designated non-financial business|\bdnfbps?\b'),
    ('FINANCIAL_GENERIC', r'finansal kuruluş|financial institution'),
    ('GENERIC', r'\byükümlü|\bfirms?\b'),
)
# Who must act, not which kind: a restriction on top of the kinds a clause names ("Merkezi yurt dışında bulunan
# bankaların ..."), so it must match as well. The kinds and categories a subject names are alternatives.
QUALIFIERS = {'FOREIGN_HQ_OBLIGED'}
# Customer families as a profile states them ("bireysel müşteriler, KOBİ'ler, üye işyerleri").
CUSTOMER_FAMILIES = (
    ('ANY', r'\btüm müşteri|\bher tür müşteri|\bbütün müşteri|\ball customers|\ball clients'),
    ('INDIVIDUAL', r'\bbireysel|\bgerçek kişi|\btüketici|\bindividual|\bretail|\bconsumer|\bborrowers?\b|\bşahıs'),
    ('BUSINESS', r'\bkobi|\bşirket|\bişletme|\büye işyer|\bkurumsal|\bticari|\btacir|\besnaf|\bcompan(?:y|ies)|\bmerchant|\bsmes?\b|\bbusiness|\bcorporate'),
    ('LEGAL_ENTITY_GENERIC', r'\btüzel kişi|\blegal entit'),
    ('ASSOCIATION', r'\bdernek|\bassociation'),
    ('FOUNDATION', r'\bvak[ıi]f|\bfoundation'),
    ('UNION', r'\bsendika|\bkonfederasyon|\btrade union'),
    ('POLITICAL_PARTY', r'siyas[iî] parti|political part'),
    ('PUBLIC_BODY', r'\bkamu|\bpublic bod|\bpublic sector'),
    ('NON_RESIDENT', r'yurt dışı|\byabancı|\bnon-resident'),
    ('CORRESPONDENT', r'\bmuhabir|\bfinansal kuruluş|\bbanka|\bfinancial institution'),
    ('UNINCORPORATED', r'tüzel kişiliği olmayan|\bapartman|\bsite yönetim'),
)
# Which customer families satisfy a clause's counterparty; a family not listed here leaves it undetermined.
SATISFIES = {
    'INDIVIDUAL': {'INDIVIDUAL'}, 'LEGAL_ENTITY': {'BUSINESS', 'LEGAL_ENTITY_GENERIC', 'ASSOCIATION', 'FOUNDATION', 'PUBLIC_BODY'},
    'ASSOCIATION': {'ASSOCIATION'}, 'FOUNDATION': {'FOUNDATION'}, 'UNION': {'UNION'}, 'POLITICAL_PARTY': {'POLITICAL_PARTY'},
    'PUBLIC_BODY': {'PUBLIC_BODY'}, 'UNINCORPORATED': {'UNINCORPORATED'}, 'FOREIGN_ASSOCIATION_BRANCH': {'ASSOCIATION', 'FOUNDATION'},
    'NON_RESIDENT': {'NON_RESIDENT'}, 'CORRESPONDENT': {'CORRESPONDENT'}}
# A bare "legal entities" customer base may include these; it never proves their absence.
COVERED_BY_GENERIC_LEGAL = {'ASSOCIATION', 'FOUNDATION', 'UNION', 'POLITICAL_PARTY', 'PUBLIC_BODY', 'FOREIGN_ASSOCIATION_BRANCH', 'UNINCORPORATED'}
# Residency and correspondent relationships are rarely stated; their absence is not a mismatch.
NEVER_MISMATCH = {'NON_RESIDENT', 'CORRESPONDENT'}
LEADING_NOTE = re.compile(r'^\s*(?:\(\d+\)\s*)?(?:\((?:Ek|Değişik|Mülga|Yeniden düzenleme)[^()]*\)\s*)*')
CLAUSE_HEAD = re.compile(r'^(.{0,220}?)(?:[;:]|kimlik tespitinde|kimlik tespiti|\bmust\b|\bshall\b|zorundadır|yükümlüdür|$)', re.S)
TURKISH_LABELS = {'FOREIGN_ASSOCIATION_BRANCH': 'yabancı dernek/vakıf şube ve temsilciliği', 'ASSOCIATION': 'dernek', 'FOUNDATION': 'vakıf',
                  'UNION': 'sendika/konfederasyon', 'POLITICAL_PARTY': 'siyasi parti', 'UNINCORPORATED': 'tüzel kişiliği olmayan teşekkül',
                  'PUBLIC_BODY': 'kamu kurumu', 'NON_RESIDENT': 'yurt dışında yerleşik', 'CORRESPONDENT': 'muhabir kuruluş',
                  'INDIVIDUAL': 'gerçek kişi', 'LEGAL_ENTITY': 'ticaret siciline kayıtlı tüzel kişi / şirket',
                  'FOREIGN_HQ_OBLIGED': 'merkezi yurt dışında bulunan yükümlü', 'BANK': 'banka', 'PAYMENT_INSTITUTION': 'ödeme kuruluşu',
                  'EMONEY_INSTITUTION': 'elektronik para kuruluşu', 'INSURANCE': 'sigorta şirketi', 'BROKER': 'aracı kurum',
                  'EXCHANGE_OFFICE': 'yetkili müessese', 'FINANCIAL_GENERIC': 'finansal kuruluş',
                  'DNFBP_GENERIC': 'finansal olmayan belirli iş ve meslekler', 'GENERIC': 'yükümlü (genel)',
                  'BUSINESS': 'işletme / KOBİ / üye işyeri', 'LEGAL_ENTITY_GENERIC': 'tüzel kişi (genel)', 'ANY': 'tüm müşteriler'}

# v0.19: a duty addressed to everyone. Kanun 5549 md. 7(1) and Tedbirler md. 31(1) open with who must give
# information: "Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar".
# Those words are the duty's addressees, not its counterparties, and "gerçek ve tüzel kişiler" binds every
# company. Measured on independent-v1 I03 (a software house): the gate read "tüzel kişiliği olmayan" and
# "kamu kurum" as counterparties the profile lacks and ruled md. 7(1) DOES_NOT_APPLY. The addressee list
# is read only at the clause's opening (the Turkish subject position), bare (no modifier before it, no case
# suffix, no postposition after it) and before a duty verb of the same sentence. "Yükümlüler", "bankalar",
# "finansal kuruluşlar" are not addressee words: a duty addressed to them keeps its reading.
ADDRESSEE_ITEMS = tuple((set(kinds), re.compile(r'(?:' + pattern + r')(?!\w)'), english) for kinds, pattern, english in (
    (('NATURAL_PERSON', 'LEGAL_PERSON'), r'gerçek (?:ve|veya|ya da) tüzel kişiler', False),
    (('LEGAL_PERSON',), r'tüzel kişiler', False),
    (('NATURAL_PERSON',), r'gerçek kişiler', False),
    (('UNINCORPORATED',), r'tüzel kişiliği olmayan (?:kuruluşlar|teşekküller|kurumlar)', False),
    (('PUBLIC_BODY',), r'kamu kurum(?:ları)? ve kuruluşları|kamu kurumları|kamu kuruluşları|kamu idareleri', False),
    (('ANYONE',), r'herkes|hiç ?bir kimse|hiç kimse|her kim', False),
    (('ANYONE',), r'(?:any|every|no) person|all persons|anyone|everyone|no-?one|no one', True),
))
# A set that binds the company whatever it is: everyone, or every legal person.
UNIVERSAL_KINDS = {'ANYONE', 'LEGAL_PERSON'}
ADDRESSEE_SEP = re.compile(r'\s*,\s*(?:(?:ve|ile|veya|ya da|and|or)\s+)?|\s+(?:ve|ile|veya|ya da|and|or)\s+')
# After a bare noun a postposition or a conjunction makes it something other than the subject ("gerçek ve
# tüzel kişiler adına", "... ile yapılan işlemler"); in English a relative or a preposition restricts it.
NOT_SUBJECT_AFTER = re.compile(r'\s+(?:adına|tarafından|hakkında|için|ile|ilgili|arasında|nezdinde|lehine|aleyhine|bakımından|yönünden|'
                               r'açısından|dışında|dahil|dâhil|gibi|kadar|karşı|yerine|namına|hesabına|aracılığıyla|vasıtasıyla|'
                               r'üzerinde|ve|veya|ya|yahut|and|or)(?!\w)')
ENGLISH_SUBJECT_AFTER = re.compile(r'\s*,|\s+(?:must|shall|may|cannot|can not|is required|are required|is obliged|are obliged)\b')
DUTY_VERB = re.compile(r'zorunda|mecbur|zorunludur|yasaktır|yükümlü(?:dür|dürler)?(?!\w)|\b(?:must|shall|may|cannot)\b|\b(?:is|are) (?:required|obliged)\b')
# A prohibition ("açıklayamaz", "veremezler", "kaçınamaz"); a passive one ("yapılamaz", "tutulamaz") binds no one.
PROHIBITION_VERB = re.compile(r'(\w+?)(?:y?amaz|y?emez)(?:lar|ler)?(?!\w)')
PASSIVE_STEM = ('ıl', 'il', 'ul', 'ül')
SENTENCE_END = re.compile(r'\.(?=\s+[A-ZÇĞİÖŞÜ(]|\s*$)')
NATURAL_PERSON_FORM = re.compile(r'şahıs (?:işletme|şirket)|sole (?:trader|proprietor)|serbest meslek|bireysel girişim')
UNINCORPORATED_FORM = re.compile(r'tüzel kişiliği olmayan|adi ortaklık|unincorporated')
ADDRESSEE_LABELS = {'ANYONE': 'herkes', 'LEGAL_PERSON': 'tüzel kişiler', 'NATURAL_PERSON': 'gerçek kişiler',
                    'UNINCORPORATED': 'tüzel kişiliği olmayan kuruluşlar', 'PUBLIC_BODY': 'kamu kurum ve kuruluşları'}


def addressee_item(folded, pos):
    """(kinds, match, english) of the addressee word at pos, or None."""
    for kinds, pattern, english in ADDRESSEE_ITEMS:
        match = pattern.match(folded, pos)
        if match:
            return kinds, match, english
    return None


def duty_verb(folded):
    if DUTY_VERB.search(folded):
        return True
    return any(not m.group(1).endswith(PASSIVE_STEM) for m in PROHIBITION_VERB.finditer(folded))


def universal_addressee(text, quote=None):
    """The clause's opening addressee list when it binds everyone or every legal person, else None.

    {'text': the list, an exact substring of the clause, 'scope': the kinds it names, 'end': its end in the
    clause body}. The list is read greedily from the clause's first word (no backtracking), must end at a
    comma, a semicolon or the next word of the sentence, and that sentence must carry a duty or prohibition
    verb. When the duty's own quote sits in a later sentence of the text, the list is not its addressee.
    """
    body = LEADING_NOTE.sub('', text or '', count=1)
    folded = fold(body)
    if len(folded) != len(body):
        return None
    scope, end, english = set(), 0, False
    item = addressee_item(folded, 0)
    while item:
        kinds, match, english = item
        scope |= kinds
        end = match.end()
        separator = ADDRESSEE_SEP.match(folded, end)
        item = addressee_item(folded, separator.end()) if separator else None
    if not end or not scope & UNIVERSAL_KINDS:
        return None
    if (ENGLISH_SUBJECT_AFTER.match(folded, end) is None) if english else bool(NOT_SUBJECT_AFTER.match(folded, end)):
        return None
    stop = SENTENCE_END.search(body, end)
    stop = stop.start() if stop else len(body)
    if not duty_verb(folded[end:stop]):
        return None
    if quote:
        wanted = ' '.join(fold(LEADING_NOTE.sub('', quote, count=1)).split())[:60]
        at = ' '.join(folded.split()).find(wanted) if wanted else -1
        if at >= len(' '.join(folded[:stop].split())):
            return None
    return {'text': body[:end], 'scope': sorted(scope), 'end': end}


def strip_addressees(text):
    """The text without its addressee words (folded)."""
    folded = fold(text or '')
    for _, pattern, _ in ADDRESSEE_ITEMS:
        folded = pattern.sub(' ', folded)
    return folded


def legal_form(company):
    """What the profile says the company is in law: a legal person unless it says otherwise."""
    facts = company.model_dump() if hasattr(company, 'model_dump') else dict(company)
    identity = fold(' '.join(' '.join(v) if isinstance(v, list) else str(v)
                             for v in (facts.get(k) for k in ('licences', 'activities', 'description', 'name')) if v))
    if UNINCORPORATED_FORM.search(identity):
        return 'UNINCORPORATED'
    return 'NATURAL_PERSON' if NATURAL_PERSON_FORM.search(identity) else 'LEGAL_PERSON'


def universal_covering(candidate, clause_text, company):
    """The clause's universal addressee list with the company's legal form when the company is in it, else None."""
    universal = universal_addressee(clause_text or candidate.get('source_quote') or '', candidate.get('source_quote'))
    if not universal:
        return None
    form = legal_form(company)
    return {**universal, 'form': form} if 'ANYONE' in universal['scope'] or form in universal['scope'] else None


def addressee_words(universal):
    return ', '.join(ADDRESSEE_LABELS[k] for k in universal['scope'])


@dataclass
class Required:
    role: str          # 'obliged_party' | 'counterparty'
    type: str
    text: str
    source: str        # 'subject' | 'clause' | 'heading'
    match: str = 'UNDETERMINED'


@dataclass
class Profile:
    entity_types: list = field(default_factory=list)
    hits: list = field(default_factory=list)          # v0.19: applicability.Hit per kind the profile affirms
    customer_families: list = field(default_factory=list)
    foreign_hq: bool = False
    jurisdictions_stated: bool = False
    customers_stated: bool = False


def found(table, text):
    """[(type, matched text)] for every pattern of the table that occurs in the folded text."""
    lowered = fold(text or '')
    hits = []
    for kind, pattern in table:
        match = re.search(pattern, lowered)
        if match:
            hits.append((kind, match.group(0)))
    kinds = {kind for kind, _ in hits}
    for kind, suppressed in SUPPRESSES.items():
        if kind in kinds:
            hits = [(k, t) for k, t in hits if k not in suppressed]
    return hits


def company_profile(company) -> Profile:
    """What the profile says the company is and whom it serves.

    v0.19: the company's kind is read with the taxonomy of applicability.CATEGORIES (every obliged kind of
    Tedbirler md. 4(1); a denied kind, or one named as another business's object, is not a clear one) from what
    the company is (licences, activities, name, description), not from what it sells. The v0.16.1 table knew
    six financial kinds: a jeweller, a factoring or a real-estate company read as "unstated".
    """
    from .applicability import IDENTITY_FIELDS, company_hits     # applicability imports this module
    facts = company.model_dump() if hasattr(company, 'model_dump') else dict(company)
    text = lambda *keys: ' '.join(' '.join(v) if isinstance(v, list) else str(v) for v in (facts.get(k) for k in keys) if v)
    identity = text('licences', 'activities', 'description', 'name')
    hits = company_hits(company, IDENTITY_FIELDS)
    customers = text('customer_types')
    families = [kind for kind, _ in found(CUSTOMER_FAMILIES, customers)]
    return Profile(entity_types=[hit.kind for hit in hits], hits=hits, customer_families=families,
                   foreign_hq=bool(re.search(r'merkezi yurt dışında|headquartered outside|yabancı merkezli', fold(identity))),
                   jurisdictions_stated=bool(facts.get('jurisdictions')), customers_stated=bool(customers.strip()))


def clause_head(text: str) -> str:
    """The opening of a sub-paragraph without its number and editorial notes: who it is about."""
    body = LEADING_NOTE.sub('', text or '', count=1)
    match = CLAUSE_HEAD.match(body)
    return (match.group(1) if match else body[:220]).strip()


def required_entities(candidate, clause_text, heading, universal=None):
    """What the clause demands of the actor and of the counterparty, from its own words.

    With a universal addressee list (universal_addressee) the listed addressees are who must act, not
    counterparties: they are read out of the subject and the clause head, and the heading is not read
    for a counterparty the clause itself does not name."""
    subject = candidate.get('subject') or ''
    head = clause_head(clause_text or candidate.get('source_quote') or '')
    if universal:
        listed = universal['text']
        head = head[len(listed):] if head.startswith(listed) else '' if listed.startswith(head) else head
        subject = strip_addressees(subject)
    required = []
    obliged = found(OBLIGED, subject)
    if not obliged:
        obliged = [(k, t) for k, t in found(OBLIGED, head) if k == 'FOREIGN_HQ_OBLIGED']
    # v0.19: every kind and category the subject names, not the first one ("Finansal kuruluşlar ile finansal
    # olmayan belirli iş ve meslekler", "Ödeme kuruluşları ile elektronik para kuruluşları").
    for kind, text in obliged:
        if kind != 'GENERIC':
            required.append(Required('obliged_party', kind, text, 'subject'))
    counterparties = found(COUNTERPARTIES, subject + ' ' + head)
    source = 'clause'
    if not counterparties and heading and not universal:
        counterparties, source = found(COUNTERPARTIES, heading), 'heading'
    for kind, text in counterparties:
        required.append(Required('counterparty', kind, text, source))
    return required


def verdict_obliged(kind, profile: Profile) -> str:
    """A required obliged kind or category against the company's kinds (applicability.taxonomy_verdict with the
    static membership: the entity gate does not hold the regulation's definitions; the addressee gate reads them)."""
    if kind == 'FOREIGN_HQ_OBLIGED':
        return 'MATCH' if profile.foreign_hq else 'MISMATCH' if profile.jurisdictions_stated else 'UNDETERMINED'
    from .applicability import taxonomy_verdict                  # applicability imports this module
    return taxonomy_verdict(kind, profile.hits)[0]


def obliged_verdict(rows):
    """The obliged side of a clause from its obliged-party rows (Required or dicts); None without rows.

    A qualifier (FOREIGN_HQ_OBLIGED) must match; the kinds and categories are alternatives, so the company is
    addressed when it is any of them and ruled out only when it is none of them.
    """
    get = lambda row, key: row[key] if isinstance(row, dict) else getattr(row, key)
    if not rows:
        return None
    qualifiers = [get(r, 'match') for r in rows if get(r, 'type') in QUALIFIERS]
    kinds = [get(r, 'match') for r in rows if get(r, 'type') not in QUALIFIERS]
    if 'MISMATCH' in qualifiers or (kinds and all(m == 'MISMATCH' for m in kinds)):
        return 'MISMATCH'
    if all(m == 'MATCH' for m in qualifiers) and (not kinds or 'MATCH' in kinds):
        return 'MATCH'
    return 'UNDETERMINED'


def verdict_counterparty(kind, profile: Profile) -> str:
    families = set(profile.customer_families)
    if 'ANY' in families:
        return 'MATCH'
    if families & SATISFIES.get(kind, set()):
        return 'MATCH'
    if kind in NEVER_MISMATCH or not profile.customers_stated:
        return 'UNDETERMINED'
    if kind in COVERED_BY_GENERIC_LEGAL and 'LEGAL_ENTITY_GENERIC' in families:
        return 'UNDETERMINED'
    specific = families - {'CORRESPONDENT'}
    return 'MISMATCH' if specific else 'UNDETERMINED'


def entity_gate(candidate, clause_text, heading, company, parent_label, clause_offset=0):
    """The gate's reading of one clause: (match, required entities, profile, reason)."""
    profile = company_profile(company)
    universal = universal_addressee(clause_text or candidate.get('source_quote') or '', candidate.get('source_quote'))
    required = required_entities(candidate, clause_text, heading, universal)
    for item in required:
        item.match = verdict_obliged(item.type, profile) if item.role == 'obliged_party' else verdict_counterparty(item.type, profile)
    obliged = [r for r in required if r.role == 'obliged_party']
    counterparties = [r for r in required if r.role == 'counterparty']
    addressed = obliged_verdict(obliged)
    if not required:
        match = 'NOT_RESTRICTED'
    elif addressed == 'MISMATCH':
        match = 'MISMATCH'
    elif counterparties and all(r.match == 'MISMATCH' for r in counterparties):
        match = 'MISMATCH'
    elif any(r.match == 'MATCH' for r in required):
        match = 'MATCH' if addressed in (None, 'MATCH') and (not counterparties or any(r.match == 'MATCH' for r in counterparties)) else 'UNDETERMINED'
    else:
        match = 'UNDETERMINED'
    number = re.match(r'\s*\((\d+)\)', clause_text or '')
    child = f'({number.group(1)})' if number else 'whole provision'
    excerpt = ' '.join((clause_text or '').split())[:200]
    described = '; '.join(f'{r.role} {r.type} ("{r.text}", {r.source}): {r.match}' for r in required) or 'no entity restriction in the clause'
    if len([r for r in obliged if r.type not in QUALIFIERS]) > 1:
        described += f'; the obliged parties named are alternatives (the company needs to be one of them): {addressed}'
    company_words = (f'entity {", ".join(profile.entity_types) or "unstated"}; customers {", ".join(profile.customer_families) or "unstated"}'
                     + ('; headquartered abroad' if profile.foreign_hq else ''))
    if match == 'MISMATCH':
        # What ruled the clause out: the obliged side when it did, the counterparties when all of them did.
        mismatched = ([r for r in obliged if r.match == 'MISMATCH'] if addressed == 'MISMATCH' else []) + \
                     ([r for r in counterparties if r.match == 'MISMATCH'] if counterparties and all(r.match == 'MISMATCH' for r in counterparties) else [])
        reason = (f'Rule (entity gate): {parent_label} {child} concerns {", ".join(TURKISH_LABELS.get(r.type, r.type) for r in mismatched)} '
                  f'({", ".join(r.role.replace("_", " ") for r in mismatched)}); the stated profile ({company_words}) does not include it, '
                  'so the clause is recorded as DOES_NOT_APPLY for this profile. It becomes applicable if the profile changes '
                  '(for example such customers are onboarded). Checks: ' + described)
    elif universal and match == 'NOT_RESTRICTED':
        reason = (f'Entity gate: UNIVERSAL_ADDRESSEE. {parent_label} {child} is addressed to "{" ".join(universal["text"].split())[:200]}" '
                  f'({addressee_words(universal)}); the enumerated addressees are who must act, not counterparties, so the clause '
                  f'restricts no entity. Profile: {company_words}.')
    else:
        reason = f'Entity gate: {described}. Profile: {company_words}.'
    if universal and match != 'NOT_RESTRICTED':
        reason += f' UNIVERSAL_ADDRESSEE: the addressees "{" ".join(universal["text"].split())[:200]}" are not read as counterparties.'
    return {'parent_provision': parent_label, 'child_clause': child, 'child_offset': int(clause_offset or 0), 'child_excerpt': excerpt,
            'required_entities': [{'role': r.role, 'type': r.type, 'text': r.text, 'source': r.source, 'match': r.match} for r in required],
            'company_entity_types': list(profile.entity_types), 'company_customer_families': list(profile.customer_families),
            'company_foreign_hq': bool(profile.foreign_hq), 'match': match, 'reason': reason}


def heading_of(section, turkish: bool) -> str:
    return article_heading(section) if turkish else ''

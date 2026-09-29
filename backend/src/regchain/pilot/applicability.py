"""Rule-first applicability (v0.18): who the regulation binds, read by rule before any model.

Measured on the tr-aml-v1 golden set (v0.17 live run, 30 cases, 92 scored duties, labels of
23 September 2026): 22 applicability errors, and 19 of them were decidable from text the
pipeline already held. Tedbirler Yönetmeliği md. 4(1) lists the obliged parties (a–z); a
foreign foundation's Türkiye office, an association, a foundation and a software company are
not on it, yet the judge answered APPLIES (5) or UNKNOWN (11) for their general duties, and it
hedged UNKNOWN for a bank whose customers include associations (3). The v0.16.1 entity gate
read the clause's own parties; nobody read the list.

The gates run in this order and never ask a model:

  1 REGULATION_SUBJECT_SCOPE  the company against the regulation's obliged-party list
  2 COMPANY_ENTITY            the duty's addressee (an authority, customs, an obliged type) and
                              the clause's obliged party (the entity gate's obliged side)
  3 JURISDICTION              the company's stated jurisdictions against the regulator's
  4 CUSTOMER_ENTITY           the clause's counterparty against the stated customer base
  5 CHILD_CLAUSE              the entity gate's combined reading of the sub-paragraph
  6 EXEMPTION                 exemption wording that names the company's kind as the exempted party

A clear MISMATCH (or an EXEMPT) settles DOES_NOT_APPLY by rule. A gate that cannot tell says
UNDETERMINED and leaves the question to the model. "Clear" is deliberately narrow, because a
false DOES_NOT_APPLY hides a duty from the reviewer and a false APPLIES skips the review:

  - a clear MATCH needs a named, unrestricted list item and the kind stated in the profile's
    licences, activities or name, not denied ("ödeme hizmeti sunmamaktadır") and not the object
    of another business ("bankacılık yazılımları", "sigorta eksperliği", "muhasebe departmanı");
  - a clear MISMATCH needs a closed list, a profile that says what the company is in words of a
    kind the list does not oblige (association, foundation, software house ...) and nothing near
    a listed kind anywhere in it ("menkul değerler", "para transferi", "sarraf" leave it to the model);
  - an EXEMPT needs the company's kind to be the exempted party in the clause itself, not a
    customer group or a kind of transaction ("müşterinin ödeme kuruluşu olması halinde").

Words are compared folded to lower-case ASCII (İ/I/ı, ç, ğ, ö, ş, ü), so "odeme kurulusu" and
"ABC LIFE INSURANCE" read like "ödeme kuruluşu" and "insurance". An adversarial review of the first
version (24 September 2026) produced every one of the cases named above; each is a test.
"""
import os
import re
from dataclasses import dataclass, field
from .scope_integrity import completeness

from .definitions import parse as parse_definitions
from .entities import COVERED_BY_GENERIC_LEGAL, OBLIGED as CLAUSE_OBLIGED, addressee_words, obliged_verdict, universal_covering
from .policies import fold

ASCII = str.maketrans('çğıöşüâîû', 'cgiosuaiu')


def norm(text) -> str:
    """Lower-case ASCII: Turkish capitals folded first (İ -> i, I -> ı), then the Turkish letters."""
    return fold(text or '').translate(ASCII)


def rx(pattern: str):
    """A pattern written in Turkish, compiled against norm() text."""
    return re.compile(pattern.translate(ASCII))


# Obliged-party kinds: (key, Turkish label, how the regulation's list names it, how a company
# profile says it is one). The list side decides which kinds a regulation binds; the company
# side is matched against the profile. Measured on Tedbirler md. 4(1): every one of the 29
# items a–z maps to at least one kind. Order matters where one kind's words contain another's
# ("tasarruf finansman şirketi" is savings finance, not finance/factoring).
CATEGORIES = tuple((key, label, rx(listed), rx(company)) for key, label, listed, company in (
    ('BANK', 'banka', r'\bbanka(?:lar|ların|lara|ları|lardan|larca)?\b(?!\s+(?:dışında|kart))|\bbankacılık',
     r'\bbanka\b|\bbankası\b|\bbankacılık|\bmevduat|katılım bank|\bbank\b|\bbanking\b'),
    ('CARD_ISSUER', 'kart çıkaran kuruluş', r'kartı düzenleme|kart(?:ı)? çıkar', r'kredi kartı düzenle|kart çıkar|kart ihraç|card issu'),
    ('EXCHANGE_OFFICE', 'yetkili müessese', r'yetkili müessese|döviz büro',
     r'yetkili müessese|döviz büro|döviz alım satım|kambiyo|\bsarraf|bureau de change|currency exchange'),
    ('SAVINGS_FINANCE', 'tasarruf finansman şirketi', r'tasarruf finansman', r'tasarruf finansman'),
    ('FINANCE_FACTORING', 'finansman/faktoring şirketi', r'faktoring|finansman şirket',
     r'faktoring|(?<!tasarruf )finansman şirket|tüketici finansman'),
    ('BROKER', 'aracı kurum / portföy yönetim şirketi', r'aracı kurum|portföy yönetim',
     r'aracı kurum|portföy yönetim|menkul değer|sermaye piyasası|yatırım kuruluş|investment firm|\bbroker'),
    ('PAYMENT_INSTITUTION', 'ödeme kuruluşu', r'ödeme kuruluş|payment institution',
     r'ödeme kuruluş|ödeme hizmet|ödeme aracı|sanal pos|para transfer|payment (?:institution|service)|\bpsp\b|money remittance'),
    ('EMONEY_INSTITUTION', 'elektronik para kuruluşu', r'elektronik para kuruluş|e-?money institution',
     r'elektronik para|\be-para|\be-?money'),
    ('INVESTMENT_TRUST', 'yatırım ortaklığı', r'yatırım ortaklık', r'yatırım ortaklığ'),
    ('INSURANCE', 'sigorta/reasürans/emeklilik şirketi', r'sigorta|reasürans|emeklilik şirket|\binsur',
     r'\bsigorta|reasürans|emeklilik şirket|\binsur'),
    ('LEASING', 'finansal kiralama şirketi', r'finansal kiralama', r'finansal kiralama|financial leasing|\bleasing\b'),
    ('CLEARING_CUSTODY', 'takas/saklama kuruluşu', r'takas ve saklama', r'takas ve saklama|takas kuruluş|merkezi kayıt|\bcustody'),
    ('BORSA_ISTANBUL', 'Borsa İstanbul', r'borsa istanbul', r'borsa istanbul'),
    ('POST_CARGO', 'PTT / kargo şirketi', r'posta ve telgraf|kargo', r'\bkargo|\bptt\b|posta ve telgraf|\bkurye|courier'),
    ('ASSET_MANAGEMENT', 'varlık yönetim şirketi', r'varlık yönetim', r'varlık yönetim'),
    # v0.19: md. 4(1)(m) "Kıymetli madenler aracı kuruluşları" is a financial institution (md. 3(1)(f)); the
    # jeweller of md. 4(1)(k) is not. One kind for both put a jeweller among the financial institutions.
    ('PRECIOUS_METALS_INTERMEDIARY', 'kıymetli madenler aracı kuruluşu', r'kıymetli madenler? aracı kuruluş',
     r'kıymetli madenler? aracı kuruluş|precious metals? intermediary institution'),
    ('PRECIOUS_METALS', 'kıymetli maden/taş/mücevher', r'kıymetli maden(?!ler aracı kuruluş)|kıymetli taş|mücevher',
     r'\bkuyum|mücevher|kıymetli maden|külçe|ziynet|\bsarraf|gümüş|pırlanta|\belmas|\btakı\b|\btakılar|precious metal|jewel|altın alım|'
     r'altın ticaret'),
    ('MINT', 'Darphane', r'darphane', r'darphane'),
    ('REAL_ESTATE', 'taşınmaz alım satımı', r'taşınmaz alım satım',
     r'\bemlak|gayrimenkul|taşınmaz alım|konut satış|konut proje|daire satış|real estate'),
    ('VEHICLE_DEALER', 'nakil vasıtası / iş makinesi alım satımı', r'nakil vasıta|iş makine',
     r'otomotiv|oto galeri|araç alım satım|otomobil ticaret|ikinci el (?:araç|otomobil)|nakil vasıta|iş makine|motosiklet|scooter|'
     r'traktör|tarım makine|karavan|\bgemi|\byat\b|\btekne|car dealer'),
    ('ANTIQUES_ART', 'tarihi eser / antika / sanat eseri', r'antika|sanat eseri|tarihi eser',
     r'antika|sanat eseri|tarihi eser|koleksiyon|sanat galeri|tablo satış|müzayede|art dealer|auction'),
    ('GAMBLING', 'talih ve bahis oyunları', r'talih ve bahis|milli piyango|spor toto|jokey',
     r'\bbahis|şans oyun|piyango|spor toto|jokey|at yarış|\biddaa|\bkumar|casino|\bgambling|\bbetting'),
    ('SPORTS_CLUB', 'spor kulübü', r'spor kulüp', r'kulüb|kulüp|football club|sports club'),
    ('NOTARY', 'noter', r'\bnoter', r'\bnoter'),
    ('LAWYER', 'serbest avukat', r'\bavukat', r'\bavukat|hukuk bürosu|law firm'),
    ('ACCOUNTANT', 'serbest muhasebeci / mali müşavir', r'muhasebeci|mali müşavir',
     r'\bsmmm\b|\bymm\b|serbest muhasebeci|mali müşavir|muhasebe bürosu|muhasebecilik|accounting firm|\baccountan'),
    ('AUDITOR', 'bağımsız denetim kuruluşu', r'bağımsız denetim', r'bağımsız denetim|audit firm'),
    ('CRYPTO', 'kripto varlık hizmet sağlayıcı', r'kripto varlık',
     r'\bkripto|\bcrypto|dijital varlık|sanal varlık|virtual asset|bitcoin|ethereum'),
    ('ECOMMERCE', 'elektronik ticaret aracı hizmet sağlayıcı', r'elektronik ticaret', r'elektronik ticaret aracı|pazar ?yeri|marketplace'),
    ('CASH_IN_TRANSIT', 'para / değerli eşya nakli', r'değerli eşya nakli|para veya değerli',
     r'para nakli|değerli eşya nakli|nakit taşıma|cash-in-transit'),
    ('LENDING', 'ödünç para verme', r'ödünç para', r'ödünç para|\blending\b|consumer credit|kredi ver'),
))
LABELS = {key: label for key, label, _, _ in CATEGORIES}
# A financial business in the wide sense (the catch-all "diğer finansal hizmetler" of Kanun 5549 md. 2(d)).
# The defined term "finansal kuruluş" is narrower: FINANCIAL_INSTITUTIONS below.
FINANCIAL = {'BANK', 'CARD_ISSUER', 'EXCHANGE_OFFICE', 'FINANCE_FACTORING', 'BROKER', 'PAYMENT_INSTITUTION', 'EMONEY_INSTITUTION',
             'INVESTMENT_TRUST', 'INSURANCE', 'LEASING', 'CLEARING_CUSTODY', 'ASSET_MANAGEMENT', 'CRYPTO', 'SAVINGS_FINANCE', 'LENDING',
             'PRECIOUS_METALS_INTERMEDIARY'}

# v0.19 entity taxonomy: the specific kinds are CATEGORIES; a clause may instead name a generic category of
# obliged parties. Measured on independent-v1 I07 (a jeweller): Tedbirler md. 25(1) "Finansal kuruluşlar ile
# finansal olmayan belirli iş ve meslekler ..." was read as "finansal kuruluş" only, and the company's kind
# was not read at all. The regulation defines both categories by the letters of its obliged-party list:
# md. 3(1)(f) "Finansal kuruluş" = md. 4(1)(a)–(h), (m), (ü) (and PTT for its banking business), md. 3(1)(m)
# "Finansal olmayan belirli iş ve meslekler" = md. 4(1)(k), (n), (s), (ş), (t), (u). Membership is read from
# those definitions when the scope articles carry them (ObligedList.categories), otherwise from this static
# reading of the same two definitions. The static reading rules a kind out of a category only where no
# definition could rule it in (a jeweller or a notary is not a financial institution, a bank is not a
# non-financial business); a kind whose place depends on the definition (a car dealer or a sports club and
# the DNFBPs, an asset manager or a savings finance company and the financial institutions) stays open.
GENERIC_CATEGORIES = {'FINANCIAL_GENERIC': 'finansal kuruluş', 'DNFBP_GENERIC': 'finansal olmayan belirli iş ve meslekler'}
FINANCIAL_INSTITUTIONS = frozenset({'BANK', 'CARD_ISSUER', 'EXCHANGE_OFFICE', 'FINANCE_FACTORING', 'BROKER', 'PAYMENT_INSTITUTION',
                                    'EMONEY_INSTITUTION', 'INVESTMENT_TRUST', 'INSURANCE', 'LEASING', 'CLEARING_CUSTODY',
                                    'PRECIOUS_METALS_INTERMEDIARY', 'CRYPTO'})
DNFBP = frozenset({'PRECIOUS_METALS', 'REAL_ESTATE', 'NOTARY', 'LAWYER', 'ACCOUNTANT', 'AUDITOR'})
PLAINLY_NON_FINANCIAL = DNFBP | {'VEHICLE_DEALER', 'ANTIQUES_ART', 'GAMBLING', 'SPORTS_CLUB', 'ECOMMERCE', 'CASH_IN_TRANSIT', 'MINT'}
STATIC_MEMBERSHIP = {'FINANCIAL_GENERIC': (FINANCIAL_INSTITUTIONS, PLAINLY_NON_FINANCIAL), 'DNFBP_GENERIC': (DNFBP, FINANCIAL_INSTITUTIONS)}
# The defined terms as a definitions article writes them ("f) Finansal kuruluş: ...").
DEFINED_TERMS = {'FINANCIAL_GENERIC': rx(r'^finansal kuruluş(?:lar)?$'), 'DNFBP_GENERIC': rx(r'^finansal olmayan belirli iş ve meslek(?:ler)?$')}
# "(a) ila (h), (m) ve (ü)": a letter or a range of letters of the list the definition refers to.
LETTER_REF = re.compile(r'\(([a-zçğıöşü])\)(?:\s*(?:ila|-|–)\s*\(([a-zçğıöşü])\))?')
# "Bu Yönetmeliğin 4 üncü maddesinin birinci fıkrasının ..." (the article and paragraph the letters belong to).
ARTICLE_REF = re.compile(r'(\d+)\s*(?:[’\']?\s*[a-zçğıöşü]{1,4})?\s+maddesinin(?:\s+(\w+)\s+fıkrasının)?', re.I)
# What a company is (licences, activities, name, description), not what it sells (products).
IDENTITY_FIELDS = ('licences', 'activities', 'name', 'description')
# A clear match is read only from what the company says it is licensed for, does, or is called.
CLEAR_FIELDS = ('licences', 'activities', 'name')
PROFILE_FIELDS = ('licences', 'activities', 'name', 'description', 'products')
# The kind is the object of another business, not the business itself ("... için yazılım", "software
# for banks", "sigorta şirketlerine danışmanlık"), or not yet the company's ("lisans başvurusu süreçte").
AS_OBJECT = rx(r'yazılım|software|danışman|consult|eksper|departman|bölüm|eğitim|training|reklam|pazarlama|marketing|operasyonel|'
               r'araç kiralama|operational|çalışan|personel|maaş|entegrasyon|teknoloji sağlayıcı|sistem geliştir|için|\bfor\b|yönelik|'
               r'başvuru|süreç|beklen|pending|application|applied')
# Turkish negation sits in a verb after the object ("sunmamaktadır", "bulunmamaktadır", "vermeyen"),
# anywhere in the clause ("ödeme kuruluşu veya elektronik para kuruluşu lisansı bulunmamaktadır");
# English in the words before it ("not a bank", "neither a bank nor ..."). A kind followed by "-sız" or
# "dışı" is denied as well ("sigortasız kargo", "banka dışı finansman").
NEGATED_AFTER = rx(r'(?:maz|mez|mazlar|mezler|mayan|meyen|mamakta|memekte|mamaktadır|memektedir|madığı|mediği|mamış|memiş)\b|'
                   r'\bdeğil|\byok(?:tur)?\b')
NEGATED_BEFORE = rx(r'\b(?:not|non|no|without|neither|nor)\b')
DENIED_SUFFIX = rx(r'^\w*(?:sız|siz|suz|süz)\b|^\w*\s+dışı\b')
# A clause ends at punctuation only: "ve", "ile", "and" must not separate a kind from its negation or its object.
SEGMENT = re.compile(r'[;,.()\n/]')
# A regulator's name is not the company's kind ("BDDK (Bankacılık Düzenleme ve Denetleme Kurumu) izni").
REGULATOR = rx(r'bankacılık düzenleme ve denetleme kurumu|sermaye piyasası kurulu|merkez bankası|mali suçları araştırma kurulu|'
               r'sigortacılık ve özel emeklilik düzenleme|\bbddk\b|\bspk\b|\btcmb\b|\bmasak\b|\bsedddk\b')
# A list item that reaches past its named kinds ("... ve diğer finansal hizmetler alanında faaliyet
# gösterenler", Kanun 5549 md. 2(d)) cannot rule a company out. Only an item whose own subject is
# "other ..." counts; "diğer kanun hükümlerine aykırı olmamak" (md. 4(1)(ş)) is a reference to other law.
CATCH_ALL = rx(r'\b(?:diğer|benzeri)\b[^.;,]{0,60}\b(?:kuruluş|şirket|faaliyet|alanında|hizmet|işletme|meslek)|'
               r'\b(?:other|similar)\b[^.;,]{0,40}\b(?:firms?|institutions?|persons?|services?|business)')
FINANCIAL_CATCH_ALL = rx(r'diğer finansal|finansal hizmetler alanında|other financial')
# An item limited to part of a business ("... saklama hizmeti ile sınırlı olmak üzere Borsa İstanbul",
# md. 4(1)(y) "... işlemlerle sınırlı olarak orta, büyük veya çok büyük ölçekli ... aracı hizmet
# sağlayıcılar") is never a clear match: whether the company falls inside the limit is for the model.
LIMITED = rx(r'sınırlı olmak üzere|sınırlı olarak|ile sınırlı|limited to')
GENERIC_FINANCE = rx(r'\bfinans|\bfintek|\bfintech|\bfinancial')
# Words that come near one of the listed kinds without naming it. A profile that uses one is not
# ruled out by rule: the model decides. Deliberately wide.
FINANCE_WORDS = rx(r'ödeme|\bpara\b|para transfer|\bkart(?:ı|lar|lı)?\b|\bkredi|finans|fintek|fintech|yatırım|sigorta|döviz|altın|'
                   r'kripto|cüzdan|\bfon\b|\bfonu|borsa|banka|mevduat|ödünç|kiralama|leasing|faktoring|emlak|gayrimenkul|'
                   r'taşınmaz|kuyum|mücevher|sarraf|külçe|ziynet|bahis|iddaa|piyango|casino|avukat|hukuk|noter|muhasebe|müşavir|'
                   r'smmm|ymm|kargo|kurye|antika|müzayede|galeri|otomobil|otomotiv|tekne|\byat\b|menkul|sermaye piyasa|dijital varlık|'
                   r'satış|ticaret|\balım|bayi|mağaza|kulüp|kulüb|kambiyo|gümüş|pırlanta|elmas|\btakı|koleksiyon|tarihi eser|motosiklet|'
                   r'traktör|karavan|\bgemi|daire|konut|pazar ?yeri|toto|jokey|yarış|'
                   r'payment|money|credit|loan|invest|insur|\bbank|exchange|crypto|wallet|remittance|lending|asset|betting|gambling|'
                   r'\bsale|trade|dealer|club|jewel')
# What a profile says the company is when it is plainly not a kind the list obliges. Only the kinds that
# were measured (associations, foundations, unions, political parties, software houses): a retailer, a
# builder or a tourism agency may deal in cars, flats, jewellery or foreign currency, which the list obliges.
NON_OBLIGED = rx(r'\bderne(?:k|ğ)|\bvakıf|\bvakf|\bsendika|konfederasyon|siyasi parti|yazılım|software|bilişim|charity|nonprofit|'
                 r'non-profit|association|foundation')
LIST_HEADING = rx(r'^yükümlü(?:ler)?$')
LIST_LEAD = rx(r'yükümlü(?:ler)?[,;:]?\s*(?:aşağıda sayılanlar|şunlardır|şunlar)|uygulanmasında yükümlü[,;:]')
DEFINITION = re.compile(r'(?:^|\s)[a-zçğıöşü]\)\s*yükümlü\s*:\s*', re.I)
ITEM = re.compile(r'(?:(?<=\s)|^)\(?([a-zçğıöşü])\)\s')
UNIT_TWO = re.compile(r'(?:^|(?<=[.;:\])]\s))\(2\)\s')
# The regulator's country, as a profile writes it (KKTC is not Türkiye).
JURISDICTIONS = {'TR': rx(r'\btürkiye\b|\bturkey\b|\btr\b|türkiye cumhuriyeti'), 'UK': rx(
    r'\buk\b|united kingdom|england|great britain|\bgb\b|scotland|wales|northern ireland|ingiltere|birleşik krallık')}
NOT_THE_REGULATOR = rx(r'\bkktc\b|kıbrıs|cyprus')
WORLDWIDE = rx(r'global|worldwide|international|uluslararası|tüm dünya|küresel')
# A part of the regulation addressed to an authority, not to the obliged parties: Tedbirler
# YEDİNCİ BÖLÜM "Gümrük İdaresine Yapılacak Açıklama" (md. 41–45: travellers declare, customs
# officers write the report), ALTINCI BÖLÜM "Yükümlülük Denetimi" (inspectors). Measured: md. 42
# (1)–(3) came back POSSIBLY_APPLIES for a payment institution three times.
AUTHORITY_PART = rx(r'gümrük idaresi|yükümlülük denetimi|denetim elemanı|kamu kurumları tarafından')
AUTHORITY_ACTOR = rx(r'^\s*(?:gümrük (?:idare|görevli|müdürlü|personel)|başkanlık(?:ça)?\b|bakanlık(?:ça)?\b|denetim eleman|'
                     r'denetim yetkisini haiz|denetime yetkili|kamu kurum(?:ları|u)\s+(?:tarafından|,)|gümrük\b)')
# The duty's own subject is the obliged parties in general ("Yükümlüler", "A firm").
GENERIC_SUBJECT = rx(r'^\s*(?:\(\d+\)\s*)?(?:(?:a|an|the|each|every|all)\s+)?(?:yükümlü(?!lük)\w*|firms?\b|obliged)')
# The addressee gate's reading of such a duty. It binds the company only as one of the obliged parties, which gate 1 reads;
# rule_chain marks the reading (Chain.obliged_addressee) for withheld_answer (v0.19 t7).
OBLIGED_GENERAL = 'The duty is addressed to the obliged parties in general.'
# Words that name who acts; a subject with one of them that no table recognises is an addressee the
# rules cannot place ("Merkezi Kayıt Kuruluşu"), so the model decides.
ACTOR_WORDS = rx(r'kuruluş|şirket|\bkurum|idare|başkanlık|bakanlık|müessese|\bborsa|sağlayıcı|görevli|personel|eleman|'
                 r'müdürlü|\bkurul\b|\bnoter|avukat|müşavir|yolcu|\bfirm|\binstitution|\bauthorit')
EXEMPTION = rx(r'(?<![a-z])(?:uygulanmaz|hariç(?:tir)?|(?<!yurt )dışında(?:dır)?|istisna(?:dır|sıdır)?|kapsam dışı|kapsamaz|'
               r'tabi değil(?:dir)?|yükümlü değil(?:dir)?|muaf(?:tır)?|does not apply|do not apply|except|exempt(?:ed)?)(?![a-z])')
# Exemption wording about a customer group or a kind of transaction narrows whom or what the duty
# concerns, not whether it binds the company ("müşterinin ödeme kuruluşu olması halinde",
# "bankalar arasında ... yapılan transferler").
CUSTOMER_CUE = rx(r'müşteri|arasında|ile yapılan|olması h[aâ]linde|işlemlerde|işlemler\b|transfer|karşı taraf|counterpart|customer|between')
CLEAR_SETTINGS = ('model', 'quick', 'rule')
# The kind is the exempted party: "<kind> hakkında (bu fıkra) uygulanmaz", "<kind> ... muaftır / tabi değildir /
# kapsam dışındadır"; English "does not apply to <kind>", "<kind> are exempt".
EXEMPTED_AFTER = rx(r'^\w*\s+(?:hakkında|için|bakımından|yönünden)\s+(?:(?:bu|işbu)\s+(?:fıkra|madde|bent|hüküm)\w*\s+)?'
                    r'(?:uygulanmaz|muaf|kapsam dışı|tabi değil)|^\w*\s+(?:\w+\s+){0,3}(?:muaf|tabi değil|kapsam dışı)|^\w*\s+(?:are|is)\s+exempt')
EXEMPTED_BEFORE = rx(r'(?:does not apply|do not apply|shall not apply|is not applied)\s+to\s+(?:\w+\s+){0,3}$')


def gate(name, status, reason, clear=True, **evidence):
    """One gate's reading, in the shape the trace, the report and the evaluation read.

    Evidence 'quote' and 'list_item' are exact substrings of a cited source; 'term' is a folded
    word the rule matched (not a quote).
    """
    return {'gate': name, 'status': status, 'clear': bool(clear), 'reason': reason,
            'evidence': {key: value for key, value in evidence.items() if value not in (None, '', [], {})}}


def kinds_in(text: str, side: int):
    """Every kind whose pattern (2 = list side, 3 = company side) occurs in the text."""
    folded = norm(text)
    return [row[0] for row in CATEGORIES if row[side].search(folded)]


def denied(segment, match):
    """A Turkish negation anywhere after the match in its clause, an English one before it, or a
    privative suffix / "dışı" right after it; a match inside a regulator's name counts as denied too."""
    if NEGATED_AFTER.search(segment[match.end():]) or NEGATED_BEFORE.search(segment[:match.start()]):
        return True
    if DENIED_SUFFIX.search(segment[match.end():]):
        return True
    return any(r.start() <= match.start() < r.end() for r in REGULATOR.finditer(segment))


def affirmed(pattern, text):
    """The first match of pattern in the (norm) text that its clause does not deny."""
    for segment in SEGMENT.split(text):
        for match in pattern.finditer(segment):
            if not denied(segment, match):
                return match
    return None


def profile_values(company, fields=PROFILE_FIELDS):
    facts = company.model_dump() if hasattr(company, 'model_dump') else dict(company)
    rows = []
    for key in fields:
        value = facts.get(key)
        for item in (value if isinstance(value, list) else [value] if isinstance(value, str) else []):
            if isinstance(item, str) and item.strip():
                rows.append((key, item))
    return facts, rows


@dataclass
class Hit:
    kind: str
    field: str
    value: str
    clear: bool     # stated in licences/activities/name, not denied, not the object of another business, not pending


def company_hits(company, fields=PROFILE_FIELDS):
    """[Hit] per kind the profile affirms; a clear hit is preferred over an unclear one of the same kind."""
    _, rows = profile_values(company, fields)
    hits = []
    for kind, _, _, pattern in CATEGORIES:
        best = None
        for key, value in rows:
            for segment in SEGMENT.split(norm(value)):
                if not affirmed(pattern, segment):
                    continue
                clear = key in CLEAR_FIELDS and not AS_OBJECT.search(segment)
                if best is None or (clear and not best.clear):
                    best = Hit(kind, key, value, clear)
        if best is not None:
            hits.append(best)
    return hits


def company_kinds(company):
    """[(kind, field, value)] the profile affirms (clear or not)."""
    return [(h.kind, h.field, h.value) for h in company_hits(company)]


@dataclass(frozen=True)
class Membership:
    """Which kinds a required kind or category takes in, which it rules out, and where that was read."""
    category: str
    members: frozenset
    non_members: frozenset
    source: str                  # 'DEFINITION' (the regulation's own words) | 'STATIC' (the taxonomy) | 'KIND' (a specific kind)
    source_id: str = ''
    source_label: str = ''
    quote: str = ''              # the definition, an exact substring of source_id's text
    markers: tuple = ()          # the list items the definition names


def membership(required, obliged=None):
    """The Membership of a required obliged kind (BANK ...) or generic category (FINANCIAL_GENERIC ...), or None.

    A category the regulation defines in the scope articles read (ObligedList.categories) is taken from that
    definition; otherwise from the static taxonomy. A specific kind takes in itself and rules out every other kind.
    """
    defined = getattr(obliged, 'categories', None) or {}
    if required in defined:
        return defined[required]
    if required in STATIC_MEMBERSHIP:
        members, non_members = STATIC_MEMBERSHIP[required]
        return Membership(required, members, non_members, 'STATIC')
    if required in LABELS:
        return Membership(required, frozenset({required}), frozenset(LABELS) - {required}, 'KIND')
    return None


def taxonomy_verdict(required, hits, obliged=None):
    """(MATCH | MISMATCH | UNDETERMINED, Membership) of a required obliged kind or category against the company's [Hit].

    MATCH: the company clearly states a member kind (licences, activities, name). MISMATCH: it clearly states a
    kind and every kind it affirms anywhere is ruled out. Anything else (no kind the taxonomy can place, a kind
    whose place the definition leaves open, a member kind named only in passing: "banka kanalıyla satar") is
    UNDETERMINED, never a DOES_NOT_APPLY.
    """
    member = membership(required, obliged)
    if member is None:
        return 'UNDETERMINED', None
    if any(hit.clear and hit.kind in member.members for hit in hits):
        return 'MATCH', member
    if any(hit.clear for hit in hits) and {hit.kind for hit in hits} <= member.non_members:
        return 'MISMATCH', member
    return 'UNDETERMINED', member


def category_definitions(scope_rows, listed):
    """{category: Membership} for each generic category a definitions article among the scope rows defines by the
    letters of the obliged-party list (Tedbirler md. 3(1)(f), (m)). A definition that names another article, a
    letter the list does not have, or no letter at all is not read (the static taxonomy stays)."""
    article = re.search(r'md\. (\S+)$', listed.source_label or '')
    order = [item['marker'][:-1] for item in listed.items]
    found = {}
    for row in scope_rows:
        text = row.get('text') or ''
        for term, definition in parse_definitions(text).items():
            category = next((c for c, pattern in DEFINED_TERMS.items() if pattern.match(norm(term).strip())), None)
            if category is None or category in found or definition not in text:
                continue
            reference = ARTICLE_REF.search(definition)
            if not reference or not article or reference.group(1) != article.group(1) or \
                    (reference.group(2) and norm(reference.group(2)) != 'birinci'):
                continue
            letters, last = [], 0
            for match in LETTER_REF.finditer(definition, reference.end()):
                first, end = match.group(1), match.group(2) or match.group(1)
                if first not in order or end not in order or order.index(end) < order.index(first):
                    letters = None
                    break
                letters += order[order.index(first):order.index(end) + 1]
                last = match.end()
            if not letters:
                continue
            markers = tuple(f'{letter})' for letter in dict.fromkeys(letters))
            members = frozenset(kind for kind in listed.kinds() if listed.item_for(kind)['marker'] in markers)
            # Parties the definition adds in its own words ("... ile bankacılık faaliyetleriyle sınırlı olarak Posta
            # ve Telgraf Teşkilatı Anonim Şirketini"): neither member nor ruled out.
            limited = frozenset(kinds_in(definition[last:], 2)) - members
            found[category] = Membership(category, members, frozenset(listed.kinds()) - members - limited, 'DEFINITION',
                                         row['id'], row.get('printed_label') or row['id'], definition[:400], markers)
    return found


@dataclass
class ObligedList:
    source_id: str
    source_label: str
    items: list = field(default_factory=list)          # [{'marker', 'text', 'kinds', 'limited'}]
    open_ended: bool = False                             # an item reaches past its named kinds
    financial_catch_all: bool = False
    unrecognised: list = field(default_factory=list)     # items no kind describes
    lead: str = ''                                       # the list's opening words, an exact substring of the source
    # v0.19: {category: Membership} of the generic categories the scope articles define by the list's letters.
    categories: dict = field(default_factory=dict)

    def kinds(self):
        return {kind for item in self.items for kind in item['kinds']}

    def item_for(self, kind):
        """The most specific item naming the kind: an unrestricted one first, then the one naming fewest kinds."""
        naming = [item for item in self.items if kind in item['kinds']]
        return min(naming, key=lambda item: (item['limited'], len(item['kinds']))) if naming else None


def split_items(body: str):
    """[(marker, text)] of a lettered enumeration "a) ... b) ... (ü) ..." (text keeps its source characters)."""
    marks = list(ITEM.finditer(body))
    items = []
    for mark, following in zip(marks, [*marks[1:], None]):
        text = body[mark.start():following.start() if following else len(body)].strip()
        if len(text) > 3:
            items.append((mark.group(1) + ')', text))
    return items


def obliged_list(scope_rows):
    """The regulation's own obliged-party list, or None when its scope articles state none.

    First choice is an article whose heading is "Yükümlü(ler)" or whose first paragraph opens
    "yükümlü, aşağıda sayılanlar ..." with lettered items (Tedbirler md. 4(1)); otherwise a
    definition "x) Yükümlü: ..." in a definitions article (Kanun 5549 md. 2(d)). FCA chapters
    have no such list: every FCA duty stays NOT_RESTRICTED here, as before.
    """
    for row in scope_rows:
        heading = norm((row.get('heading_path') or [''])[-1]).strip()
        text = row.get('text') or ''
        if not (LIST_HEADING.match(heading) or LIST_LEAD.search(norm(text))):
            continue
        second = UNIT_TWO.search(text)
        body = text[:second.start()] if second else text
        items = split_items(body)
        if items:
            listed = build_list(row, items)
            listed.categories = category_definitions(scope_rows, listed)
            return listed
    for row in scope_rows:
        text = row.get('text') or ''
        match = DEFINITION.search(text)
        if match:
            rest = text[match.end():]
            end = re.search(r'\s[a-zçğıöşü]\)\s*[A-ZÇĞİÖŞÜ][^:]{1,60}:', rest)
            return build_list(row, [('tanım', rest[:end.start()] if end else rest)])
    return None


def build_list(row, items):
    rows, unrecognised = [], []
    for marker, text in items:
        found = kinds_in(text, 2)
        # The item's own characters (a prefix is still an exact substring): cited as scope evidence.
        rows.append({'marker': marker, 'text': text[:400], 'kinds': found, 'limited': bool(LIMITED.search(norm(text)))})
        if not found:
            unrecognised.append(marker)
    folded = ' '.join(norm(text) for _, text in items)
    source = row.get('text') or ''
    first = source.find(items[0][1]) if items else -1
    # The lead-in ("Kanunun uygulanmasında yükümlü, aşağıda sayılanlar ...:") is what a DOES_NOT_APPLY cites.
    lead = source[:first].strip() if first > 0 else source[:300]
    return ObligedList(source_id=row['id'], source_label=row.get('printed_label') or row['id'], items=rows,
                       open_ended=bool(CATCH_ALL.search(folded)), financial_catch_all=bool(FINANCIAL_CATCH_ALL.search(folded)),
                       unrecognised=unrecognised, lead=lead[:400])


def profile_incomplete(company):
    """The v0.15 rule's fields: a None (not an empty list) means the profile does not say."""
    return [key for key in ('jurisdictions', 'activities', 'licences', 'products', 'customer_types') if getattr(company, key, None) is None]


def stated_field(facts, *keys):
    """The first of keys the profile states (a non-empty value), for the fact a rule decision cites."""
    for key in keys:
        value = facts.get(key)
        if (isinstance(value, list) and any(isinstance(v, str) and v.strip() for v in value)) or (isinstance(value, str) and value.strip()):
            return key
    return None


FORM_LABELS = {'LEGAL_PERSON': 'a legal person', 'NATURAL_PERSON': 'a natural person', 'UNINCORPORATED': 'an unincorporated body'}


def universal_record(name, status, universal, what, **evidence):
    """A gate that reads the duty's universal addressee list (v0.19): UNIVERSAL_ADDRESSEE in the reason and the evidence."""
    listed = ' '.join(universal['text'].split())
    return gate(name, status, f'UNIVERSAL_ADDRESSEE: the duty is addressed to "{listed[:200]}" ({addressee_words(universal)}), which includes '
                              f'the company ({FORM_LABELS[universal["form"]]}); {what}', clear=True, reason_code='UNIVERSAL_ADDRESSEE',
                universal_addressee=universal['text'][:400], **evidence)


def subject_gate(company, obliged, universal=None):
    """Gate 1: is the company one of the kinds the regulation obliges?

    A duty addressed to everyone or to every legal person the company is (universal_covering) is not
    narrowed by the obliged-party list: anything but a list MATCH reads NOT_RESTRICTED (UNIVERSAL_ADDRESSEE).
    """
    record = obliged_list_gate(company, obliged)
    if universal and obliged is not None and record['status'] != 'MATCH':
        return universal_record(record['gate'], 'NOT_RESTRICTED', universal, f'{obliged.source_label}\'s obliged-party list does not '
                                f'narrow it (the list alone reads {record["status"]}).', source_label=obliged.source_label,
                                source_id=obliged.source_id)
    return record


def obliged_list_gate(company, obliged):
    """Gate 1 on the regulation's obliged-party list alone."""
    name = 'REGULATION_SUBJECT_SCOPE'
    if obliged is None:
        return gate(name, 'NOT_RESTRICTED', 'The scope articles read state no obliged-party list; the regulation\'s own scope text is '
                                            'left to the provision-level judgement.', clear=True)
    where = {'source_label': obliged.source_label, 'source_id': obliged.source_id}
    listed = obliged.kinds()
    hits = company_hits(company)
    on_list = [h for h in hits if h.kind in listed]
    for hit in sorted(on_list, key=lambda h: not h.clear):
        item = obliged.item_for(hit.kind)
        if hit.clear and not item['limited']:
            return gate(name, 'MATCH', f'{obliged.source_label} {item["marker"]} names {LABELS[hit.kind]}; the profile states it '
                                       f'({hit.field}: "{hit.value[:120]}").', clear=True, list_item=item['text'], quote=item['text'],
                        company_field=hit.field, company_value=hit.value, matched_items=[item['marker']], **where)
    if on_list:
        hit = on_list[0]
        item = obliged.item_for(hit.kind)
        why = ('the item is limited to part of that business' if hit.clear else
               f'the profile mentions it only in {hit.field} or as the object of another business')
        return gate(name, 'MATCH', f'{obliged.source_label} {item["marker"]} names {LABELS[hit.kind]}, but {why} ("{hit.value[:120]}"); '
                                   'the provision-level judgement decides.', clear=False, list_item=item['text'], company_field=hit.field,
                    company_value=hit.value, **where)
    facts, rows = profile_values(company)
    identity = ' ; '.join(norm(value) for _, value in rows)
    financial = bool({h.kind for h in hits} & FINANCIAL) or bool(affirmed(GENERIC_FINANCE, identity))
    if obliged.financial_catch_all and financial:
        return gate(name, 'MATCH', f'{obliged.source_label} reaches "other financial services" and the profile describes a financial '
                                   'business; a catch-all item is not a named one, so the provision-level judgement decides.',
                    clear=False, **where)
    absent = [key for key in ('activities', 'licences') if facts.get(key) is None]
    if absent:
        return gate(name, 'UNDETERMINED', f'The profile does not state {", ".join(absent)}; whether the company is one of the obliged '
                                          f'parties of {obliged.source_label} cannot be told.', clear=False, **where)
    said = stated_field(facts, 'activities', 'licences', 'description')
    if said is None:
        return gate(name, 'UNDETERMINED', 'The profile states no activity, licence or description to place on the list.', clear=False, **where)
    near = affirmed(FINANCE_WORDS, identity)
    if hits or near:
        term = near.group(0) if near else LABELS[hits[0].kind]
        return gate(name, 'UNDETERMINED', f'The profile describes activity near a listed kind ("{term}") without naming a kind that '
                                          f'{obliged.source_label} lists; the provision-level judgement decides.', clear=False, term=term,
                    **where)
    if obliged.open_ended or obliged.unrecognised:
        why = ('an open-ended item' if obliged.open_ended else 'items the rules cannot read: ' + ', '.join(obliged.unrecognised))
        return gate(name, 'UNDETERMINED', f'{obliged.source_label} has {why}; not being on the named items does not rule the '
                                          'company out.', clear=False, **where)
    plain = affirmed(NON_OBLIGED, identity)
    if not plain:
        return gate(name, 'UNDETERMINED', 'The profile does not say what kind of entity the company is in words the rules can place; '
                                          'the provision-level judgement decides.', clear=False, **where)
    stated = '; '.join(f'{key}: {value[:80]}' for key, value in rows if key in ('activities', 'licences'))[:300]
    field_used = next((key for key, value in rows if affirmed(NON_OBLIGED, norm(value))), said)
    return gate(name, 'MISMATCH', f'{obliged.source_label} lists the obliged parties ({len(obliged.items)} items); the profile says the '
                                  f'company is a "{plain.group(0)}" ({stated}), none of them and nothing near one, so the regulation\'s '
                                  'general duties do not bind it.', clear=True, company_field=field_used, term=plain.group(0), quote=obliged.lead,
                **where)


def jurisdiction_gate(company, turkish: bool):
    """Gate 3: the company's stated jurisdictions against the regulator's country."""
    name = 'JURISDICTION'
    country = 'TR' if turkish else 'UK'
    values = getattr(company, 'jurisdictions', None)
    if not values:
        return gate(name, 'UNDETERMINED', 'The profile states no jurisdiction.', clear=False)
    for value in values:
        # Read part by part: "Türkiye ve KKTC" names Türkiye; "KKTC (Kuzey Kıbrıs Türk Cumhuriyeti)" does not.
        for part in re.split(r'[,;/()]|\s(?:ve|and|ile|&)\s', norm(value)):
            if JURISDICTIONS[country].search(part) and not NOT_THE_REGULATOR.search(part):
                return gate(name, 'MATCH', f'The profile names {value!r}, the regulator\'s jurisdiction ({country}).', clear=True,
                            company_field='jurisdictions', company_value=value)
    if any(WORLDWIDE.search(norm(value)) for value in values):
        return gate(name, 'UNDETERMINED', f'The profile names {values!r}; whether that includes {country} is not stated.', clear=False)
    return gate(name, 'MISMATCH', f'The profile names {", ".join(values)} only; the regulation binds activity in {country}.', clear=True,
                company_field='jurisdictions', company_value=', '.join(values))


# The kinds the v0.16.1 entity gate already reads from a clause's subject; the addressee gate leaves them to it.
CLAUSE_KINDS = {kind for kind, _ in CLAUSE_OBLIGED} & set(LABELS)


READ = object()


def defined_readings(obliged_rows, company, obliged):
    """The clause gate's obliged-party rows, each generic category the regulation defines in the scope articles
    re-read against that definition: ([row], {category: Membership} of the definitions that were read)."""
    categories = getattr(obliged, 'categories', None) or {}
    rows, used = [dict(r) for r in obliged_rows], {}
    if not any(r['type'] in categories for r in rows):
        return rows, used
    hits = company_hits(company, IDENTITY_FIELDS)
    for row in rows:
        if row['type'] in categories:
            row['match'], used[row['type']] = taxonomy_verdict(row['type'], hits, obliged)
    return rows, used


def definition_evidence(used):
    """Gate evidence for the definitions a verdict rests on: each an exact quote of a scope source (strings only)."""
    if not used:
        return {}
    return {'membership': 'DEFINITION', 'definitions': [{'category': m.category, 'source_id': m.source_id, 'source_label': m.source_label,
                                                         'quote': m.quote, 'items': list(m.markers)} for m in used.values()]}


def addressee_gate(section, candidate, clause_text, company, clause_gate, universal=READ, obliged=None):
    """Gate 2: who the duty is addressed to — the obliged parties, a kind of them, or an authority.

    The entity gate's obliged side (bank, payment institution, foreign-headquartered firm ...) is
    folded in. The duty's own subject decides before any mention elsewhere in the clause: "Başkanlık",
    "Denetim yetkisini haiz olanlar" and "Kripto varlık hizmet sağlayıcısı olan yükümlüler" are not
    addressed to a bank, whatever else the clause says about "yükümlüler".

    v0.19: the clause's own opening addressee list comes first. When it binds everyone or every legal
    person and the company is one (entities.universal_covering), the gate reads MATCH (UNIVERSAL_ADDRESSEE),
    whatever the extracted subject says ("Başkanlık ve denetim elemanları tarafından istenilecek ..." is
    the agent of a participle in Tedbirler md. 31(1), not the addressee).

    v0.19 taxonomy: the obliged parties a clause names ("Finansal kuruluşlar ile finansal olmayan belirli iş ve
    meslekler") are alternatives: the company is addressed when it is any of them (entities.obliged_verdict).
    A generic category the regulation defines in the scope articles read (`obliged`, ObligedList.categories) is
    read against that definition, which the evidence quotes.
    """
    name = 'COMPANY_ENTITY'
    if universal is READ:
        universal = universal_covering(candidate, clause_text, company)
    if universal:
        return universal_record(name, 'MATCH', universal, 'the company is one of the parties the duty binds.')
    subject = candidate.get('subject') or ''
    folded_subject = norm(subject)
    heading_path = section.get('heading_path') or []
    part = norm(heading_path[1]) if len(heading_path) > 1 else ''
    article = norm(heading_path[-1]) if heading_path else ''
    obliged_rows, used = defined_readings([r for r in clause_gate['required_entities'] if r['role'] == 'obliged_party'], company, obliged)
    verdict = obliged_verdict(obliged_rows)
    defined = definition_evidence(used)
    if verdict == 'MISMATCH':
        wrong = [r for r in obliged_rows if r['match'] == 'MISMATCH']
        where = ''.join(f' {d["source_label"]} defines {GENERIC_CATEGORIES.get(d["category"], d["category"])}: "{d["quote"][:160]}".'
                        for d in defined.get('definitions', []))
        return gate(name, 'MISMATCH', 'The clause binds ' + ' or '.join(f'{r["type"]} ("{r["text"]}")' for r in wrong) +
                    ('; the stated profile is none of them.' if len(wrong) > 1 else '; the stated profile is not that kind.') + where,
                    clear=True, term=wrong[0]['text'], **defined)
    actor = AUTHORITY_ACTOR.search(folded_subject)
    if actor:
        return gate(name, 'MISMATCH', f'The duty\'s subject "{subject[:80]}" is an authority, not the company.', clear=True, subject=subject[:120])
    named = [kind for kind in kinds_in(subject, 2) if kind not in CLAUSE_KINDS]
    mine = {kind for kind, _, _ in company_kinds(company)}
    if named and not set(named) & mine:
        return gate(name, 'MISMATCH', f'The duty\'s subject "{subject[:80]}" names {", ".join(LABELS[k] for k in named)}; the stated '
                                      'profile is not that kind.', clear=True, subject=subject[:120])
    generic = bool(GENERIC_SUBJECT.search(folded_subject)) or (not folded_subject.strip() and bool(GENERIC_SUBJECT.search(norm(clause_text or ''))))
    authority = AUTHORITY_PART.search(part) or AUTHORITY_PART.search(article)
    if authority and not generic:
        return gate(name, 'MISMATCH', f'The provision sits in "{heading_path[1] if len(heading_path) > 1 else heading_path[-1]}", '
                                      'a procedure addressed to an authority, and the duty\'s subject is not the obliged parties.',
                    clear=True, term=authority.group(0))
    if verdict == 'MATCH' or (named and set(named) & mine):
        return gate(name, 'MATCH', 'The clause binds a kind of obliged party the profile states.', clear=True, **defined)
    if verdict == 'UNDETERMINED':
        return gate(name, 'UNDETERMINED', 'The clause binds a kind of obliged party the profile neither states nor excludes.', clear=False,
                    **defined)
    if generic:
        return gate(name, 'MATCH', OBLIGED_GENERAL, clear=True, subject=subject[:120])
    if ACTOR_WORDS.search(folded_subject):
        return gate(name, 'UNDETERMINED', f'The duty\'s subject "{subject[:80]}" names an actor the rules cannot place.', clear=False)
    return gate(name, 'NOT_RESTRICTED', 'The clause names no actor of its own (a passive duty of the obliged party).', clear=True)


def customer_gate(clause_gate):
    """Gate 4: the clause's counterparties (associations, individuals ...) against the customer base."""
    name = 'CUSTOMER_ENTITY'
    rows = [r for r in clause_gate['required_entities'] if r['role'] == 'counterparty']
    if not rows:
        return gate(name, 'NOT_RESTRICTED', 'The clause names no counterparty.', clear=True)
    described = ', '.join(f'{r["type"]} ("{r["text"]}"): {r["match"]}' for r in rows)
    customers = ', '.join(clause_gate['company_customer_families']) or 'unstated'
    if any(r['match'] == 'MATCH' for r in rows):
        return gate(name, 'MATCH', f'{described}; customers {customers}.', clear=True, term=rows[0]['text'])
    if all(r['match'] == 'MISMATCH' for r in rows):
        return gate(name, 'MISMATCH', f'{described}; customers {customers}.', clear=True, term=rows[0]['text'])
    open_rows = [r for r in rows if r['match'] == 'UNDETERMINED']
    # Only a counterparty a bare "tüzel kişi" base may or may not hold (associations, foundations,
    # unions ...) is the profile's silence; a non-resident or a correspondent is undetermined for
    # another reason and is left alone.
    ambiguous = ('LEGAL_ENTITY_GENERIC' in clause_gate['company_customer_families'] and bool(open_rows)
                 and all(r['type'] in COVERED_BY_GENERIC_LEGAL for r in open_rows))
    return gate(name, 'UNDETERMINED', f'{described}; customers {customers}' + ('; a bare "tüzel kişi" customer base may or may not '
                                                                                 'include them' if ambiguous else '') + '.', clear=False,
                term=rows[0]['text'], profile_ambiguous=ambiguous)


def clause_gate_record(clause_gate, universal=None):
    """Gate 5: the entity gate's combined reading of the sub-paragraph (unchanged since v0.16.1); v0.19 files the
    universal addressee list it set aside (UNIVERSAL_ADDRESSEE) under the evidence."""
    status = clause_gate['match']
    extra = {'reason_code': 'UNIVERSAL_ADDRESSEE', 'universal_addressee': universal['text'][:400]} if universal and status == 'NOT_RESTRICTED' else {}
    return gate('CHILD_CLAUSE', status, clause_gate['reason'][:600], clear=status in ('MATCH', 'MISMATCH', 'NOT_RESTRICTED'),
                child_clause=clause_gate['child_clause'], quote=clause_gate['child_excerpt'] if status == 'MISMATCH' else None, **extra)


def sentences(text):
    """The text's sentences as exact substrings (split after '.' or ';')."""
    return [s.strip() for s in re.split(r'(?<=[.;])\s+', text or '') if s.strip()]


def exempted_kinds(sentence, mine):
    """The company's kinds this sentence exempts: the kind is the party the exemption is about
    ("bu fıkra ödeme kuruluşları hakkında uygulanmaz", "ödeme kuruluşları ... muaftır", "does not apply
    to payment institutions"), not the actor of an exception to something else ("Bankalar, resmî tatil
    günleri hariç, ... bildirir", "A payment institution must, except in an emergency, ...")."""
    folded = norm(sentence)
    if CUSTOMER_CUE.search(folded):
        return []
    found = []
    for kind, _, listed, _ in CATEGORIES:
        if kind not in mine:
            continue
        for match in listed.finditer(folded):
            after = folded[match.end():match.end() + 90]
            before = folded[max(0, match.start() - 60):match.start()]
            if EXEMPTED_AFTER.match(after) or EXEMPTED_BEFORE.search(before):
                found.append(kind)
                break
    return found


def exemption_check(section_text, clause_text, company):
    """Gate 6: exemption wording that names the company's own kind as the exempted party.

    EXEMPT (DOES_NOT_APPLY by rule) only when the clause itself exempts a kind the profile clearly
    is. Exception wording near the company's kind otherwise ("hariç", "dışında", "except"), in
    another sub-paragraph, or about customers or transactions may or may not reach this company:
    POSSIBLE, the model and a person decide.
    """
    name = 'EXEMPTION'
    mine = {h.kind for h in company_hits(company) if h.clear}
    if not mine:
        return gate(name, 'NONE', 'No exemption wording names the company\'s kind.', clear=True)

    def naming(text):
        for sentence in sentences(text):
            folded = norm(sentence)
            marker = EXEMPTION.search(folded)
            if not marker:
                continue
            named = [kind for kind in kinds_in(sentence, 2) if kind in mine]
            if named:
                return sentence, marker.group(0), named, exempted_kinds(sentence, mine)
        return None
    inside = naming(clause_text)
    if inside and inside[3]:
        sentence, marker, _, exempted = inside
        return gate(name, 'EXEMPT', f'The clause says "{marker}" of {", ".join(LABELS[k] for k in exempted)}, which the profile states.',
                    clear=True, quote=sentence[:400])
    rest = (section_text or '').replace(clause_text or '', ' ') if clause_text else ''
    elsewhere = inside or naming(rest)
    if elsewhere:
        sentence, marker, named, _ = elsewhere
        where = 'The clause' if elsewhere is inside else 'Another sub-paragraph'
        return gate(name, 'POSSIBLE', f'{where} says "{marker}" near {", ".join(LABELS[k] for k in named)}; whether it exempts this '
                                      'company is for the reviewer.', clear=False, quote=sentence[:400])
    return gate(name, 'NONE', 'No exemption wording names the company\'s kind.', clear=True)


@dataclass
class Chain:
    gates: list
    decision: str            # 'DOES_NOT_APPLY' | 'UNKNOWN' | 'OPEN'
    decided_by: str | None   # the applicability_rule when the rules settle it
    reason: str
    clear_match: bool        # every gate a clear MATCH / NOT_RESTRICTED / NONE, subject a named list item
    profile_ambiguous: bool  # a counterparty is undetermined only because of a bare "tüzel kişi" customer base
    company: object = None   # the profile the gates read (for the fact a rule decision cites)
    obliged_addressee: bool = False  # v0.19 t7: the addressee gate read the duty as addressed to the obliged parties in general
    completeness_check: dict = field(default_factory=dict)

    def by_name(self, name):
        return next((g for g in self.gates if g['gate'] == name), None)


def rule_chain(company, section, candidate, clause_text, clause_gate, obliged, turkish):
    """Gates 1–6 for one clause, and what they settle before any model is asked."""
    # v0.19: a duty addressed to everyone (or every legal person) the company is: no list or subject rules it out.
    universal = universal_covering(candidate, clause_text, company)
    gates = [subject_gate(company, obliged, universal), addressee_gate(section, candidate, clause_text, company, clause_gate, universal, obliged),
             jurisdiction_gate(company, turkish), customer_gate(clause_gate), clause_gate_record(clause_gate, universal),
             exemption_check(section.get('text') or '', clause_text, company)]
    subject, addressee, jurisdiction, customer, child, exemption = gates
    # Profile silence is not evidence of an excluded company, even when an actor
    # gate would otherwise decide negatively. Empty lists remain explicit facts.
    incomplete = completeness(company, section.get('text'))
    if incomplete:
        return Chain(gates, 'UNKNOWN', incomplete['code'], incomplete['reason'], False,
                     bool(customer['evidence'].get('profile_ambiguous')), company, completeness_check=incomplete)
    rules = {'REGULATION_SUBJECT_SCOPE': 'SUBJECT_SCOPE_GATE', 'COMPANY_ENTITY': 'ADDRESSEE_GATE'}
    # The v0.16.1 clause gate keeps its own verdict and name: a sub-paragraph it rules out stays ENTITY_GATE.
    if child['status'] == 'MISMATCH':
        return Chain(gates, 'DOES_NOT_APPLY', 'ENTITY_GATE', clause_gate['reason'], False, False, company)
    # A stated jurisdiction outside the regulator's is read and shown, but it does not settle the
    # duty by itself: a profile's jurisdiction list is often incomplete (a UK firm with a Turkish
    # branch), and the v0.17 contract analyses such profiles in full. It blocks a clear match and
    # routes an APPLIES to a person (JURISDICTION_MISMATCH) instead.
    for record in (subject, addressee):
        if record['status'] == 'MISMATCH' and record['clear']:
            return Chain(gates, 'DOES_NOT_APPLY', rules[record['gate']], 'Rule (' + record['gate'].lower().replace('_', ' ') + '): '
                         + record['reason'], False, False, company)
    if exemption['status'] == 'EXEMPT':
        return Chain(gates, 'DOES_NOT_APPLY', 'EXEMPTION_GATE', 'Rule (exemption): ' + exemption['reason'], False, False, company)
    ambiguous = bool(customer['evidence'].get('profile_ambiguous'))
    clear = (subject['status'] == 'MATCH' and subject['clear'] and addressee['status'] in ('MATCH', 'NOT_RESTRICTED')
             and jurisdiction['status'] == 'MATCH' and customer['status'] in ('MATCH', 'NOT_RESTRICTED')
             and child['status'] in ('MATCH', 'NOT_RESTRICTED') and exemption['status'] == 'NONE')
    return Chain(gates, 'OPEN', None, '', clear, ambiguous, company,
                 obliged_addressee=addressee['status'] == 'MATCH' and addressee['reason'] == OBLIGED_GENERAL)


# v0.19 t7: the provision-level answer is judged once and shared by every duty of the provision (engine.propose,
# APPLICABILITY_SHARED). Measured on the v019t6 live micro run (independent I08, a restaurant chain, 25 September 2026):
# Tedbirler md. 31(1) binds "Kamu kurum ve kuruluşları, gerçek ve tüzel kişiler ile tüzel kişiliği olmayan kuruluşlar",
# md. 31(3) binds "Yükümlüler" only. The obliged-party list could not place the company (its description says it takes card
# payments, "ödeme": near a listed kind), so the model was asked about md. 31 once. It answered APPLIES on (1)'s addressee list
# ("TR" against "gerçek ve tüzel kişiler"), and (3) inherited that APPLIES although nothing had read the company as an obliged party.
CHILD_ADDRESSEE = 'CHILD_ADDRESSEE'
OBLIGED_ADDRESSEE_UNDETERMINED = 'OBLIGED_ADDRESSEE_UNDETERMINED'


# t7 review 1 (F5): the profile fields that say what the company does or is licensed for.
OWN_BUSINESS = ('activities', 'licences')


def financial_identity(company) -> str:
    """'<field>: <value>' of the first profile value that may make the company an obliged party although the list could not place
    it: a word near a listed kind (FINANCE_WORDS, GENERIC_FINANCE) in what it does or is licensed for (OWN_BUSINESS), or a financial
    supervisor (REGULATOR: "BDDK izni", "SPK lisansı") anywhere in the profile; '' when there is none. A word near a listed kind only
    in the description or the products ("müşterilerinden ... kartla ödeme alır": the company is paid) is not its business."""
    _, rows = profile_values(company)
    for field, value in rows:
        text = norm(value)
        # A regulator's name is read as such here (affirmed denies a match inside one: "Merkez Bankası" is no bank); only a
        # negation denies it ("BDDK lisansı yoktur").
        supervised = any(not (NEGATED_AFTER.search(segment[m.end():]) or NEGATED_BEFORE.search(segment[:m.start()]))
                         for segment in SEGMENT.split(text) for m in REGULATOR.finditer(segment))
        if supervised or (field in OWN_BUSINESS and (affirmed(FINANCE_WORDS, text) or affirmed(GENERIC_FINANCE, text))):
            return f'{field}: {value[:120]}'
    return ''


def withheld_answer(chain, units, company, state, basis, obliged):
    """The CHILD_ADDRESSEE gate when a favourable provision-level answer does not carry to this clause, else None (v0.19 t7).

    A duty addressed to the obliged parties in general binds the company only as one of them. The provision-level `state`
    (APPLIES or POSSIBLY_APPLIES) stands for the clause unless all of these hold:
      - the addressee gate read the duty as addressed to the obliged parties in general (chain.obliged_addressee);
      - the regulation's obliged-party list neither places the company nor rules it out (REGULATION_SUBJECT_SCOPE
        UNDETERMINED). A clear MISMATCH has already settled DOES_NOT_APPLY by rule, before any model. A MATCH carries the
        answer. NOT_RESTRICTED means that no list was read;
      - a sub-paragraph of the provision (`units`, [(offset, text)]) opens with an addressee list that binds everyone or
        every legal person the company is (entities.universal_covering), so the answer may rest on that sub-paragraph;
      - no verified basis pair of the answer (`basis`) reads the company against the list itself (a YES whose source is the
        list), so the answer does not say that the company is an obliged party;
      - t7 review 1 (F5): the profile's own business says nothing that may make it an obliged party (financial_identity: no word
        near a listed kind in what it does or is licensed for, no financial supervisor named anywhere). Measured (t7 review): an
        asset-management company ("sorunlu alacak yönetimi", "BDDK izni"), an obliged party the list could not place, lost the
        APPLIES of md. 31(3). The restaurant of the live run ("restoran işletmeciliği"; "ödeme" only in "müşterilerinden ... kartla
        ödeme alır") shows nothing of the kind, and its clause is still withheld.
    The clause is then UNKNOWN (the engine records it under PROFILE_AMBIGUOUS). The rules cannot tell whether the company is one
    of the parties the clause binds, and a DOES_NOT_APPLY would rest on that silence. The gate never gives APPLIES or
    DOES_NOT_APPLY, and it never reads a sub-paragraph addressed to everyone or to a party other than the obliged parties.
    """
    if state not in ('APPLIES', 'POSSIBLY_APPLIES') or not chain.obliged_addressee or obliged is None:
        return None
    if chain.gates[0]['status'] != 'UNDETERMINED':
        return None
    universal = next((found for found in (universal_covering({}, text, company) for _, text in units) if found), None)
    if universal is None:
        return None
    if any(getattr(b, 'match', None) == 'YES' and getattr(b, 'source_id', None) == obliged.source_id for b in basis):
        return None
    if financial_identity(company):
        return None
    subject = chain.gates[1]['evidence'].get('subject') or ''
    listed = ' '.join(universal['text'].split())
    named = f' ("{subject[:80]}")' if subject else ''
    why = (f'The clause is addressed to the obliged parties only{named}, and {obliged.source_label} neither places the company on its '
           f'list nor rules it out. The provision-level {state} may rest on the addressee list "{listed[:160]}" '
           f'({addressee_words(universal)}), which binds everyone or every legal person. It is not inherited: whether this clause binds '
           'the company stays UNKNOWN until the profile says whether it is an obliged party.')
    return gate(CHILD_ADDRESSEE, 'UNDETERMINED', why, clear=False,
                reason_code=OBLIGED_ADDRESSEE_UNDETERMINED, subject=subject[:120], universal_addressee=universal['text'][:400],
                provision_state=state, source_label=obliged.source_label, source_id=obliged.source_id)


def clear_match_setting(value=None):
    """APPLICABILITY_CLEAR_MATCH: model (default, the v0.17 behaviour), quick or rule."""
    setting = (value if value is not None else os.getenv('APPLICABILITY_CLEAR_MATCH') or 'model').strip().lower()
    if setting not in CLEAR_SETTINGS:
        raise ValueError('APPLICABILITY_CLEAR_MATCH must be model, quick or rule')
    return setting


def decision_evidence(chain, company, section, clause_text, obliged):
    """(company_fact_keys, [(source_id, quote)]) a rule DOES_NOT_APPLY cites, so that a reviewer's
    APPROVE (or the autonomous review) of it carries a profile field and an exact quote.

    The subject gate cites the list's opening words (a scope source); every other gate cites the
    clause itself (the provision). The profile field is the one the deciding gate read, or the first
    one the profile states.
    """
    facts = company.model_dump() if hasattr(company, 'model_dump') else dict(company)
    decider = {'SUBJECT_SCOPE_GATE': 'REGULATION_SUBJECT_SCOPE', 'ADDRESSEE_GATE': 'COMPANY_ENTITY', 'EXEMPTION_GATE': 'EXEMPTION',
               'ENTITY_GATE': 'CHILD_CLAUSE'}.get(chain.decided_by)
    record = chain.by_name(decider) if decider else None
    preferred = [record['evidence'].get('company_field')] if record and record['evidence'].get('company_field') else []
    if chain.decided_by == 'ENTITY_GATE':
        preferred.append('customer_types' if chain.by_name('CUSTOMER_ENTITY')['status'] == 'MISMATCH' else 'licences')
    # 'name' is not a citable fact field (validate_evidence): a kind read from the name cites a stated field.
    preferred = [key for key in preferred if key != 'name']
    key = stated_field(facts, *preferred, 'activities', 'licences', 'customer_types', 'description', 'products', 'jurisdictions')
    quotes = []
    if chain.decided_by == 'SUBJECT_SCOPE_GATE' and obliged is not None and obliged.lead:
        quotes.append((obliged.source_id, obliged.lead[:400]))
    elif chain.decided_by == 'EXEMPTION_GATE' and record and record['evidence'].get('quote'):
        quotes.append((section['id'], record['evidence']['quote'][:400]))
    elif clause_text and clause_text.strip():
        quotes.append((section['id'], clause_text.strip()[:400]))
    # v0.19: a category read from the regulation's own definition cites that definition (a scope source) too.
    for defined in (record['evidence'].get('definitions') or []) if record else []:
        if defined.get('source_id') and defined.get('quote'):
            quotes.append((defined['source_id'], defined['quote'][:400]))
    return ([key] if key else []), quotes


def applies_trace(chain, final, model_basis, reason):
    """What an APPLIES stands on (v0.18): every field a reviewer needs, and which are missing."""
    subject, addressee, jurisdiction, customer, child, exemption = chain.gates[:6]
    yes = [b for b in model_basis if getattr(b, 'match', None) == 'YES']
    subject_basis = (subject['evidence'].get('list_item') if subject['status'] == 'MATCH' and subject['clear'] else None) or \
                    (f'{yes[0].regulatory_condition} [{yes[0].source_id}]' if yes else None)
    company_field = subject['evidence'].get('company_field') if subject['status'] == 'MATCH' and subject['clear'] else None
    company_field = company_field or next((b.company_fact_key for b in yes if b.company_fact_key), None)
    record = {'regulation_subject_basis': subject_basis, 'company_matching_field': company_field,
              'jurisdiction_basis': jurisdiction['reason'] if jurisdiction['status'] == 'MATCH' else None,
              'child_clause_match': f'{child["status"]}: {customer["status"]} customer, {addressee["status"]} addressee'
              if child['status'] in ('MATCH', 'NOT_RESTRICTED') and customer['status'] in ('MATCH', 'NOT_RESTRICTED')
              and addressee['status'] in ('MATCH', 'NOT_RESTRICTED') else None,
              'exclusion_check': exemption['reason'] if exemption['status'] == 'NONE' else None,
              'final_reasoning': (reason or '')[:600] or None}
    missing = [key for key, value in record.items() if not value]
    record.update({'complete': not missing, 'missing': missing if final == 'APPLIES' else []})
    return record

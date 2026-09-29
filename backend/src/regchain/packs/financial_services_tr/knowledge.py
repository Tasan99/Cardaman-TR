"""The AML / financial-services tables the engine's gates were measured with (Kanun 5549, Tedbirler Yönetmeliği).

Moved here verbatim from pilot/applicability.py, pilot/entities.py, pilot/conflict.py and pilot/engine.py.
The comments that explain each table stayed with the code that compiles it; this module is data only.
"""
from ...sector import SectorKnowledge

PACK_ID = 'FINANCIAL_SERVICES_TR'

# Obliged-party kinds: (key, Turkish label, how the regulation's list names it, how a company profile says it
# is one). Measured on Tedbirler md. 4(1): every one of the 29 items a–z maps to at least one kind. Order
# matters where one kind's words contain another's ("tasarruf finansman şirketi" is savings finance).
CATEGORIES = (
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
)
# A financial business in the wide sense (the catch-all "diğer finansal hizmetler" of Kanun 5549 md. 2(d)).
FINANCIAL = frozenset({'BANK', 'CARD_ISSUER', 'EXCHANGE_OFFICE', 'FINANCE_FACTORING', 'BROKER', 'PAYMENT_INSTITUTION', 'EMONEY_INSTITUTION',
                       'INVESTMENT_TRUST', 'INSURANCE', 'LEASING', 'CLEARING_CUSTODY', 'ASSET_MANAGEMENT', 'CRYPTO', 'SAVINGS_FINANCE', 'LENDING',
                       'PRECIOUS_METALS_INTERMEDIARY'})
GENERIC_CATEGORIES = {'FINANCIAL_GENERIC': 'finansal kuruluş', 'DNFBP_GENERIC': 'finansal olmayan belirli iş ve meslekler'}
FINANCIAL_INSTITUTIONS = frozenset({'BANK', 'CARD_ISSUER', 'EXCHANGE_OFFICE', 'FINANCE_FACTORING', 'BROKER', 'PAYMENT_INSTITUTION',
                                    'EMONEY_INSTITUTION', 'INVESTMENT_TRUST', 'INSURANCE', 'LEASING', 'CLEARING_CUSTODY',
                                    'PRECIOUS_METALS_INTERMEDIARY', 'CRYPTO'})
DNFBP = frozenset({'PRECIOUS_METALS', 'REAL_ESTATE', 'NOTARY', 'LAWYER', 'ACCOUNTANT', 'AUDITOR'})
PLAINLY_NON_FINANCIAL = DNFBP | {'VEHICLE_DEALER', 'ANTIQUES_ART', 'GAMBLING', 'SPORTS_CLUB', 'ECOMMERCE', 'CASH_IN_TRANSIT', 'MINT'}
# The defined terms as a definitions article writes them ("f) Finansal kuruluş: ...").
DEFINED_TERMS = {'FINANCIAL_GENERIC': r'^finansal kuruluş(?:lar)?$', 'DNFBP_GENERIC': r'^finansal olmayan belirli iş ve meslek(?:ler)?$'}
# A regulator's name is not the company's kind ("BDDK (Bankacılık Düzenleme ve Denetleme Kurumu) izni").
REGULATOR = (r'bankacılık düzenleme ve denetleme kurumu|sermaye piyasası kurulu|merkez bankası|mali suçları araştırma kurulu|'
             r'sigortacılık ve özel emeklilik düzenleme|\bbddk\b|\bspk\b|\btcmb\b|\bmasak\b|\bsedddk\b')
FINANCIAL_CATCH_ALL = r'diğer finansal|finansal hizmetler alanında|other financial'
GENERIC_FINANCE = r'\bfinans|\bfintek|\bfintech|\bfinancial'
# Words that come near one of the listed kinds without naming it. Deliberately wide.
FINANCE_WORDS = (r'ödeme|\bpara\b|para transfer|\bkart(?:ı|lar|lı)?\b|\bkredi|finans|fintek|fintech|yatırım|sigorta|döviz|altın|'
                 r'kripto|cüzdan|\bfon\b|\bfonu|borsa|banka|mevduat|ödünç|kiralama|leasing|faktoring|emlak|gayrimenkul|'
                 r'taşınmaz|kuyum|mücevher|sarraf|külçe|ziynet|bahis|iddaa|piyango|casino|avukat|hukuk|noter|muhasebe|müşavir|'
                 r'smmm|ymm|kargo|kurye|antika|müzayede|galeri|otomobil|otomotiv|tekne|\byat\b|menkul|sermaye piyasa|dijital varlık|'
                 r'satış|ticaret|\balım|bayi|mağaza|kulüp|kulüb|kambiyo|gümüş|pırlanta|elmas|\btakı|koleksiyon|tarihi eser|motosiklet|'
                 r'traktör|karavan|\bgemi|daire|konut|pazar ?yeri|toto|jokey|yarış|'
                 r'payment|money|credit|loan|invest|insur|\bbank|exchange|crypto|wallet|remittance|lending|asset|betting|gambling|'
                 r'\bsale|trade|dealer|club|jewel')
# What a profile says the company is when it is plainly not a kind the list obliges (measured kinds only).
NON_OBLIGED = (r'\bderne(?:k|ğ)|\bvakıf|\bvakf|\bsendika|konfederasyon|siyasi parti|yazılım|software|bilişim|charity|nonprofit|'
               r'non-profit|association|foundation')

# entities.py: the party the clause is about (identified, transacted with). Order matters.
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
    ('DNFBP_GENERIC', r'finansal olmayan belirli iş ve meslek|designated non-financial business|\bdnfbps?\b'),
    ('FINANCIAL_GENERIC', r'finansal kuruluş|financial institution'),
    ('GENERIC', r'\byükümlü|\bfirms?\b'),
)
QUALIFIERS = frozenset({'FOREIGN_HQ_OBLIGED'})
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
SATISFIES = {
    'INDIVIDUAL': {'INDIVIDUAL'}, 'LEGAL_ENTITY': {'BUSINESS', 'LEGAL_ENTITY_GENERIC', 'ASSOCIATION', 'FOUNDATION', 'PUBLIC_BODY'},
    'ASSOCIATION': {'ASSOCIATION'}, 'FOUNDATION': {'FOUNDATION'}, 'UNION': {'UNION'}, 'POLITICAL_PARTY': {'POLITICAL_PARTY'},
    'PUBLIC_BODY': {'PUBLIC_BODY'}, 'UNINCORPORATED': {'UNINCORPORATED'}, 'FOREIGN_ASSOCIATION_BRANCH': {'ASSOCIATION', 'FOUNDATION'},
    'NON_RESIDENT': {'NON_RESIDENT'}, 'CORRESPONDENT': {'CORRESPONDENT'}}
COVERED_BY_GENERIC_LEGAL = frozenset({'ASSOCIATION', 'FOUNDATION', 'UNION', 'POLITICAL_PARTY', 'PUBLIC_BODY', 'FOREIGN_ASSOCIATION_BRANCH',
                                      'UNINCORPORATED'})
NEVER_MISMATCH = frozenset({'NON_RESIDENT', 'CORRESPONDENT'})
TURKISH_LABELS = {'FOREIGN_ASSOCIATION_BRANCH': 'yabancı dernek/vakıf şube ve temsilciliği', 'ASSOCIATION': 'dernek', 'FOUNDATION': 'vakıf',
                  'UNION': 'sendika/konfederasyon', 'POLITICAL_PARTY': 'siyasi parti', 'UNINCORPORATED': 'tüzel kişiliği olmayan teşekkül',
                  'PUBLIC_BODY': 'kamu kurumu', 'NON_RESIDENT': 'yurt dışında yerleşik', 'CORRESPONDENT': 'muhabir kuruluş',
                  'INDIVIDUAL': 'gerçek kişi', 'LEGAL_ENTITY': 'ticaret siciline kayıtlı tüzel kişi / şirket',
                  'FOREIGN_HQ_OBLIGED': 'merkezi yurt dışında bulunan yükümlü', 'BANK': 'banka', 'PAYMENT_INSTITUTION': 'ödeme kuruluşu',
                  'EMONEY_INSTITUTION': 'elektronik para kuruluşu', 'INSURANCE': 'sigorta şirketi', 'BROKER': 'aracı kurum',
                  'EXCHANGE_OFFICE': 'yetkili müessese', 'FINANCIAL_GENERIC': 'finansal kuruluş',
                  'DNFBP_GENERIC': 'finansal olmayan belirli iş ve meslekler', 'GENERIC': 'yükümlü (genel)',
                  'BUSINESS': 'işletme / KOBİ / üye işyeri', 'LEGAL_ENTITY_GENERIC': 'tüzel kişi (genel)', 'ANY': 'tüm müşteriler'}

# conflict.py: the core compliance acts (folded word beginnings, TR/EN), institution nouns, customer groups, authorities.
ACTS = {'@keep': ('sakla', 'muhafaz', 'arşiv', 'retain', 'retent', 'keep', 'kept', 'store', 'storing', 'storage'),
        '@report': ('bildir', 'ilet', 'rapor', 'report', 'notif'),
        '@identify': ('tespit', 'teyit', 'doğrula', 'kimlik', 'identif', 'verif'),
        '@submit': ('ibraz', 'sunul', 'sunmak', 'sunar', 'submit'),
        '@monitor': ('izlen', 'izlem', 'takip', 'monitor'),
        '@destroy': ('imha', 'silin', 'silme', 'destroy', 'delet'),
        '@provide': ('verme', 'veril', 'verir', 'provid', 'furnish'),
        '@disclose': ('açıkla', 'ifşa', 'paylaş', 'bilgilendir', 'disclos', 'reveal', 'tipping', 'inform')}
INSTITUTION = (r'(?:kripto\s+varlık\s+hizmet\s+sağlayıcı|hizmet\s+sağlayıcı|finansal\s+kuruluş|ödeme\s+kuruluş|elektronik\s+para\s+kuruluş|'
               r'kuruluş|banka|payment\s+service\s+providers?|service\s+providers?|psps?|institutions?|providers?|firms?|banks?)')
GROUPS = {'dernek': r'dernek\w*|associations?', 'vakıf': r'vakı?f\w*|foundations?', 'sendika': r'sendika\w*|trade\s+unions?',
          'tüzel': r'tüzel\s+kişi\w*|legal\s+(?:persons?|entit\w*)|corporate\w*|kurumsal\w*',
          'gerçek': r'gerçek\s+kişi\w*|natural\s+persons?|individuals?|bireysel\w*',
          'sicil': r'ticaret\s+sicil\w*', 'kamu': r'kamu\s+kurum\w*|public\s+bod\w*', 'yabancı': r'yabancı\w*|foreign\w*',
          'pep': r'siyasi\s+nüfuz\w*|politically\s+exposed'}
AUTHORITY = (r"(?:masak\w*|mali\s+suçları\s+araştırma\s+kurulu\s+başkanlı\w*|başkanlı\w*|otorite\w*|yetkililer\w*|"
             r"yetkili\s+merci\w*|savcılı\w*|authorit\w*|regulators?|fca|fiu)")

# engine.py: the risk categories the enrichment step classifies a duty into.
RISK_CATEGORIES = ('Müşterinin tanınması ve kabulü', 'Şüpheli işlem ve raporlama', 'Kayıt saklama ve ibraz',
                   'Müşteri iletişimi ve tanıtım', 'Müşteri muamelesi ve adil davranış',
                   'Yönetişim, eğitim ve iç denetim', 'Diğer')

KNOWLEDGE = SectorKnowledge(
    pack_id=PACK_ID, categories=CATEGORIES, financial=FINANCIAL, generic_categories=GENERIC_CATEGORIES,
    financial_institutions=FINANCIAL_INSTITUTIONS, dnfbp=DNFBP, plainly_non_financial=PLAINLY_NON_FINANCIAL,
    defined_terms=DEFINED_TERMS, regulator_pattern=REGULATOR, financial_catch_all_pattern=FINANCIAL_CATCH_ALL,
    generic_finance_pattern=GENERIC_FINANCE, finance_words_pattern=FINANCE_WORDS, non_obliged_pattern=NON_OBLIGED,
    counterparties=COUNTERPARTIES, suppresses=SUPPRESSES, obliged=OBLIGED, qualifiers=QUALIFIERS,
    customer_families=CUSTOMER_FAMILIES, satisfies=SATISFIES, covered_by_generic_legal=COVERED_BY_GENERIC_LEGAL,
    never_mismatch=NEVER_MISMATCH, entity_labels=TURKISH_LABELS,
    acts=ACTS, institution_pattern=INSTITUTION, groups=GROUPS, authority_pattern=AUTHORITY,
    risk_categories=RISK_CATEGORIES)

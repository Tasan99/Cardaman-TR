"""Turkish legislation from mevzuat.gov.tr (Mevzuat Bilgi Sistemi), article by article.

The public page of a regulation loads its text with JavaScript, but the text itself is
served as a Word-exported HTML document (one <p> per paragraph) by the endpoint the page
embeds. That document is retained byte for byte, hashed, and parsed here into articles
("MADDE 4"), each with its numbered sub-paragraphs (fıkra) and lettered items (bent).

Only the consolidated text is read: footnotes, the amendment-date table and the layout
are dropped; inline editorial notes such as "(Değişik: 7/7/2011 - KHK - 646/10 md.)" are
part of the official consolidated wording and stay. A repealed article ("(Mülga: ...)")
is retained as a placeholder flagged DELETED_PROVISION, never as a duty.
"""
import re
from datetime import date
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup

from .models import Document, Download, IngestionError, Paragraph

MEVZUAT_HOSTS = frozenset({'www.mevzuat.gov.tr', 'mevzuat.gov.tr'})
# v2: page-bottom amendment footnotes ("–––––– (1) 29/2/2016 tarihli ve ... değiştirilmiştir.")
# are no longer read into the article they happen to follow (seen in Tedbirler Yönetmeliği
# md. 4, 5, 6/A, 17/A, 29, 32, 46, 50 on 23 September 2026).
# v3: the rule that opens the footnote block is also printed as underscores ("_________________
# (1) 24/2/2021 tarihli ..."), found in md. 8 of the same regulation by the retained-snapshot
# test; v2 read that block into the article. A retained snapshot is parsed with the rules of
# the version that produced it (recorded in snapshot.json), so its hash keeps reproducing.
MEVZUAT_PARSER_VERSION = 'mevzuat-parser-v3'
# A run of dashes (v3: or underscores) opens the footnote block Word prints at the foot of a
# page; each note starts with its number and the amending instrument's date.
FOOTNOTE_RULES = {'mevzuat-parser-v1': None, 'mevzuat-parser-v2': re.compile(r'[–—-]{6,}'),
                  'mevzuat-parser-v3': re.compile(r'[–—_-]{6,}')}
FOOTNOTE_RULE = FOOTNOTE_RULES[MEVZUAT_PARSER_VERSION]
FOOTNOTE_NOTE = re.compile(r'^\(\d{1,2}\)\s+\d{1,2}/\d{1,2}/\d{4}\s+tarihli')
# v3: a note may open with what was amended before the amending instrument's date ("(1) Bu madde
# başlığı “Merkezi Kayıt Kuruluşu” iken, 29/2/2016 tarihli ..."); v2 closed the block there and
# read the rest of the notes into md. 50.
FOOTNOTE_NOTES = {'mevzuat-parser-v1': FOOTNOTE_NOTE, 'mevzuat-parser-v2': FOOTNOTE_NOTE,
                  'mevzuat-parser-v3': re.compile(r'^\(\d{1,2}\)\s+(?:\d{1,2}/\d{1,2}/\d{4}\s+tarihli|'
                                                  r'Bu (?:madde|fıkra|bent|bend|Yönetmeli|Kanun)[^.]{0,200}?\d{1,2}/\d{1,2}/\d{4}\s+tarihli)')}
TEXT_ENDPOINT = 'https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe'
# MevzuatTur codes of the Mevzuat Bilgi Sistemi, read from its own search form: the
# ASCII group code used inside the pilot, and the word printed in every provision label.
KINDS = {'1': ('KANUN', 'Kanun'), '4': ('KHK', 'KHK'), '19': ('CBK', 'CBK'),
         '21': ('YONETMELIK', 'Yönetmelik'), '10': ('YONETMELIK', 'Yönetmelik'), '7': ('YONETMELIK', 'Yönetmelik'),
         '8': ('YONETMELIK', 'Yönetmelik'), '9': ('TEBLIG', 'Tebliğ'), '2': ('TUZUK', 'Tüzük'), '17': ('TUZUK', 'Tüzük')}
GROUPS = {word: code for code, word in KINDS.values()}
ARTICLE = re.compile(r'^\s*((?:EK|GEÇİCİ|Ek|Geçici)\s+)?(?:MADDE|Madde)\s+(\d+(?:/[A-ZÇĞİÖŞÜ])?)\s*[–\-—]?\s*(.*)$', re.S)
CHAPTER = re.compile(r'^[A-ZÇĞİÖŞÜ]+(?:\s[A-ZÇĞİÖŞÜ]+)?\s+(?:BÖLÜM|KISIM)$')
CONTENT_START = re.compile(r'^\(?\d+\)|^[a-zçğıöşü]{1,2}\)')
REPEALED = re.compile(r'^\s*\(?\s*Mülga\b')
METADATA = re.compile(r'^(Kanun Numarası|Kabul Tarihi|Yayımlandığı Resmî Gazete|Yayımlandığı Düstur|Bakanlar Kurulu Kararının Tarihi|'
                      r'Cumhurbaşkanı Kararının Tarihi|Dayandığı Kanunun Tarihi|Yayımlandığı Resmî Gazetenin Tarihi|Yayımlandığı Düsturun Tertibi|'
                      r'Resmî Gazete Tarihi|Resmi Gazete Tarihi|Dayandığı Mevzuatın Tarihi|Karar Sayısı)\s*:?')
FOOTNOTE_MARK = re.compile(r'\s*\[\d{1,3}\]')
# Düstur page numbers ("5346-2") are printed as centred lines between paragraphs.
PAGE_NUMBER = re.compile(r'^\d{3,5}(?:-\d{1,2})?$')
DATE = re.compile(r'(\d{1,2})[./](\d{1,2})[./](\d{4})')


def text_url(kind: str, number: str, tertip: str = '5') -> str:
    """The text endpoint for one regulation, from the identifiers the search API returns."""
    kind, number, tertip = str(kind).strip(), str(number).strip(), str(tertip).strip() or '5'
    if kind not in KINDS or not re.fullmatch(r'\d{1,12}', number) or not re.fullmatch(r'\d', tertip):
        raise ValueError('Mevzuat; tür kodu, numarası ve tertibi ile seçilir (ör. Kanun 5549: tür 1, tertip 5).')
    return f'{TEXT_ENDPOINT}?MevzuatTur={kind}&MevzuatNo={number}&MevzuatTertip={tertip}'


def identity(url: str) -> tuple[str, str, str]:
    """(kind code, number, tertip) of a retained text URL."""
    parts = urlsplit(url)
    if parts.hostname not in MEVZUAT_HOSTS or not parts.path.lower().endswith('/mevzuatfihristdetayiframe'):
        raise ValueError('Not a mevzuat.gov.tr text URL')
    query = parse_qs(parts.query)
    try:
        kind, number = query['MevzuatTur'][0], query['MevzuatNo'][0]
    except KeyError as exc:
        raise ValueError('Mevzuat URL lacks its identifiers') from exc
    if kind not in KINDS:
        raise ValueError('Unknown mevzuat kind code '+kind)
    return kind, number, query.get('MevzuatTertip', ['5'])[0]


def is_mevzuat(source: Download) -> bool:
    return urlsplit(source.final_url).hostname in MEVZUAT_HOSTS


def document_label(kind: str, number: str) -> str:
    """'Kanun 5549', 'Yönetmelik 200713012': the document part of every printed label."""
    return f'{KINDS[str(kind)][1]} {number}'


def decode(source: Download) -> str:
    for encoding in ('utf-8', 'cp1254'):
        try:
            return source.body.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestionError('Mevzuat text is not UTF-8 or Windows-1254')


def paragraph_lines(node, rule_in_superscript=None) -> list[str]:
    """The text lines of one <p>: a heading and its MADDE line may share a paragraph."""
    for anchor in node.find_all('a'):
        # Footnote references are layout, not law; the footnotes themselves are skipped.
        if (anchor.get('href') or '').startswith('#_ftn') or FOOTNOTE_MARK.fullmatch(anchor.get_text()):
            anchor.decompose()
    for sup in node.find_all('sup'):
        # Older documents mark footnotes as superscript "(4)" instead of a link. v3: the rule
        # that opens a footnote block is sometimes printed as superscript too (Tedbirler
        # Yönetmeliği md. 16 and 52); dropping it let the notes below run into the article.
        if rule_in_superscript is not None and rule_in_superscript.search(sup.get_text()):
            sup.replace_with(sup.get_text())
        else:
            sup.decompose()
    for br in node.find_all('br'):
        # Only an explicit line break separates lines; Word wraps source lines freely.
        br.replace_with(' ')
    lines = []
    for line in node.get_text(' ').replace('\xa0', ' ').split(' '):
        line = re.sub(r'\s+', ' ', FOOTNOTE_MARK.sub('', line)).strip()
        if line and not PAGE_NUMBER.match(line):
            lines.append(line)
    return lines


def centered(node) -> bool:
    return 'center' in (node.get('align') or '') or 'text-align:center' in (node.get('style') or '').replace(' ', '')


def parse_mevzuat(source: Download, version: str = MEVZUAT_PARSER_VERSION) -> Document:
    if not is_mevzuat(source):
        raise IngestionError('Only mevzuat.gov.tr text pages are parsed here')
    if version not in FOOTNOTE_RULES:
        raise IngestionError('Unknown mevzuat parser version '+str(version))
    footnote_rule, footnote_note = FOOTNOTE_RULES[version], FOOTNOTE_NOTES[version]
    kind, number, _ = identity(source.final_url)
    soup = BeautifulSoup(decode(source), 'html.parser')
    label_prefix = document_label(kind, number)
    title_lines: list[str] = []
    metadata: dict[str, str] = {}
    chapter, chapter_title, pending_heading, expecting_title = '', '', '', False
    articles: list[dict] = []
    warnings: list[str] = []
    superscript_rule = footnote_rule if version == 'mevzuat-parser-v3' else None
    lines = [(line, node) for node in soup.find_all('p')
             if not (node.find_parent('table') or node.find_parent(id=re.compile(r'^ftn')) or 'MsoFootnoteText' in (node.get('class') or []))
             for line in paragraph_lines(node, superscript_rule)]
    footnotes = False
    for text, node in lines:
        # A dash rule opens page-bottom amendment notes; what follows it on the same line and
        # every following "(n) dd/mm/yyyy tarihli ..." line is editorial, not the article.
        rule = footnote_rule.search(text) if footnote_rule else None
        if rule:
            before, after = text[:rule.start()].strip(), text[rule.end():].strip()
            footnotes = True
            if after and not footnote_note.match(after):
                footnotes = False
                before = (before + ' ' + after).strip()
            text = before
            if not text:
                continue
        elif footnotes:
            if footnote_note.match(text):
                continue
            footnotes = False
        meta = METADATA.match(text)
        if meta and not articles:
            metadata[meta.group(1)] = text[meta.end():].strip(' :')
            continue
        if CHAPTER.match(text) and centered(node):
            chapter, chapter_title, pending_heading, expecting_title = text, '', '', True
            continue
        if expecting_title and centered(node):
            chapter_title, expecting_title = text, False
            continue
        expecting_title = False
        article = ARTICLE.match(text)
        if article:
            prefix = (article.group(1) or '').strip().upper()
            number_ = ('Ek ' if prefix.startswith('EK') else 'Geçici ' if prefix else '') + article.group(2)
            body = article.group(3).strip()
            articles.append({'number': number_, 'heading': pending_heading, 'chapter': chapter, 'chapter_title': chapter_title,
                             'lines': [body] if body else []})
            pending_heading = ''
            continue
        heading_like = (len(text) <= 120 and not CONTENT_START.match(text) and (node.find('b') or centered(node))
                        and not text.endswith(('.', ',', ';', ':')))
        if not articles:
            if heading_like and not centered(node):
                pending_heading = text
            elif centered(node):
                title_lines.append(text)
            continue
        if heading_like:
            # A short bold line between two articles is the next article's heading.
            pending_heading = text
            continue
        articles[-1]['lines'].append(text)
    if not articles:
        raise IngestionError('No MADDE found in mevzuat.gov.tr text; the identifiers may not name a regulation')
    title = re.sub(r'\s+', ' ', ' '.join(title_lines)).strip() or label_prefix
    published = publication_date(metadata)
    paragraphs = []
    seen = set()
    for item in articles:
        text = ' '.join(item['lines']).strip()
        flags = []
        if item['number'] in seen:
            warnings.append(f'Duplicate article number {item["number"]}')
            flags.append('POSSIBLE_WRAPPED_LABEL')
        seen.add(item['number'])
        if not text or REPEALED.match(text):
            flags.append('DELETED_PROVISION')
            text = text or '(Mülga)'
        chapter_path = ' '.join(filter(None, (item['chapter'], item['chapter_title'])))
        paragraphs.append(Paragraph(section=item['chapter'] or 'BÖLÜMSÜZ', number=item['number'], text=text,
                                    heading_path=(title, chapter_path, item['heading']), locator_kind='mevzuat_madde',
                                    printed_label=f'{label_prefix} md. {item["number"]}', source_kind='CONSOLIDATED',
                                    quality_flags=tuple(flags), legal_type='RULE', effective_from=published))
    if all('DELETED_PROVISION' in p.quality_flags for p in paragraphs):
        raise IngestionError('Every article of this regulation is repealed')
    return Document(title=title, paragraphs=tuple(paragraphs), publication_date=published, warnings=tuple(warnings))


def publication_date(metadata: dict) -> date | None:
    for key in ('Yayımlandığı Resmî Gazete', 'Yayımlandığı Resmî Gazetenin Tarihi', 'Resmî Gazete Tarihi', 'Resmi Gazete Tarihi'):
        match = DATE.search(metadata.get(key, ''))
        if match:
            day, month, year = (int(x) for x in match.groups())
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def article_heading(section: dict) -> str:
    path = section.get('heading_path') or []
    return path[2] if len(path) >= 3 else ''


def scope_rank(section: dict) -> int:
    """How strongly an article speaks to who the regulation applies to: 0 = not at all.

    Turkish regulations open with 'Amaç', 'Kapsam', 'Dayanak' and 'Tanımlar' articles; the
    'Kapsam' article and a 'Yükümlüler' list say who is bound, the definitions say what the
    defined actors are. Those, in that order, are the scope evidence for applicability.
    """
    heading = article_heading(section).lower().replace('i̇', 'i').strip()
    # The heading must name the scope or the obliged parties itself ("Kapsam", "Amaç ve kapsam",
    # "Yükümlü", "Yükümlüler"); "Yükümlülük ihlâlinde ceza" is about breaches, and (v0.16, seen on
    # Tedbirler Yönetmeliği) "Yükümlüler tarafından devamlı bilgi verme" is a duty and "Denetimin
    # kapsamı" a procedure — both had crowded the definitions article out of the scope budget.
    if len(heading.split()) <= 4 and (re.match(r'^(?:amaç ve )?kapsam', heading)
                                       or re.match(r'^yükümlü(?:ler)?(?![a-zçğıöşü])', heading)):
        return 3
    if 'tanım' in heading:
        return 2
    return 0


def preliminary(section: dict) -> bool:
    """'Amaç', 'Dayanak', 'Yürürlük', 'Yürütme' articles carry no duty for a firm."""
    heading = article_heading(section).lower().replace('i̇', 'i')
    return any(word in heading for word in ('amaç', 'dayanak', 'yürürlük', 'yürütme', 'tanım', 'kapsam'))

"""Local policy extraction with retained bytes hash and honest page/block locators."""
import io
import re
import zipfile
import zlib
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree as ET

from pypdf import PdfReader
from regchain.evidence import digest

MAX_BYTES = 12*1024*1024
# v2: a PDF's running header and footer (the same line at the top or bottom of three or more
# pages, page numbers aside) is layout, not policy. Measured on a real bank policy (VakıfBank,
# 23 September 2026): "Kurum İçi Sınırsız Kullanım / Kişisel Veri" opened 16 of 34 passages and
# two of them were judged UNCLEAR for that fragment alone, which blocked a favourable verdict.
# v3: a block longer than one passage is cut at a sentence or line end, never mid-word, and a
# short tail is not left as a passage of its own. Measured on a real 2-page payment-institution
# policy (Garanti Ödeme, 23 September 2026): a 2,087-character block became a 2,000-character
# passage ending "hedefle m" and an 87-character crumb "edürleri, • Programın ..." that the judge
# read as a passage and called UNCLEAR. Packets that retained a v1/v2 extraction verify with
# their own rules.
PARSER = 'pilot-policy-v3'
RUNNING_LINES_MIN_PAGES = 3
CHUNK = 2000
# A cut is looked for this far back from the limit; a remainder shorter than TAIL_MIN joins the
# passage before it when that stays within CHUNK + TAIL_MIN.
CUT_LOOKBACK = 600
TAIL_MIN = 300
CUT_POINT = re.compile(r'(?<=[.!?;:])\s+|\n')


PAGE_LIKE = re.compile(r'^(?:sayfa|page)\b.*\d|^[\d\s/.–-]+$', re.I)


def running_key(line: str) -> str:
    """Identical text is the same running line; only page-number lines are compared digits aside,
    so "Madde 1." and "Madde 2." at the foot of successive pages stay content."""
    return re.sub(r'\d+', '#', line) if PAGE_LIKE.match(line) else line


def running_lines(parts) -> set:
    """Header and footer lines: the first or last line of at least three pages."""
    seen, counts = {}, {}
    for _, text, _ in parts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in ({lines[0], lines[-1]} if lines else ()):
            key = running_key(line)
            seen.setdefault(key, set()).add(line)
            counts[key] = counts.get(key, 0) + 1
    return {line for key, variants in seen.items() if counts[key] >= RUNNING_LINES_MIN_PAGES for line in variants}


def strip_running_lines(text: str, parts) -> str:
    """The page without its running header/footer lines and bare page numbers."""
    running = running_lines(parts)
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in running or re.fullmatch(r'\d{1,4}', stripped):
            continue
        kept.append(line)
    return '\n'.join(kept)


def cut_points(text: str, start: int, end: int, parser: str) -> list[tuple[int, int]]:
    """[(offset, end)] pieces of text[start:end], each at most CHUNK characters.

    v1/v2 cut every 2000 characters. v3 cuts at the last sentence or line end within the
    look-back window (else the last space), and folds a short final remainder into the piece
    before it. Every piece is an exact substring at its stated offsets.
    """
    if parser != PARSER:
        return [(offset, min(offset + CHUNK, end)) for offset in range(start, end, CHUNK)]
    pieces, offset = [], start
    while offset < end:
        if end - offset <= CHUNK:
            pieces.append((offset, end))
            break
        window = text[offset:offset + CHUNK]
        cut = None
        for match in CUT_POINT.finditer(window, max(0, CHUNK - CUT_LOOKBACK)):
            cut = match.end()
        if cut is None:
            space = window.rfind(' ', CHUNK - CUT_LOOKBACK)
            cut = space + 1 if space > 0 else CHUNK
        piece_end = offset + cut
        remainder = end - piece_end
        if 0 < remainder < TAIL_MIN and (end - offset) <= CHUNK + TAIL_MIN:
            piece_end = end
        pieces.append((offset, piece_end))
        offset = piece_end
    return pieces


def read_policy(path: Path, parser: str = PARSER, refuse_unreadable: bool = True):
    # refuse_unreadable=False is for re-reading a retained original (verification.verify_artifacts):
    # the v0.18 refusals of garbled or empty text must not make a v0.17 packet look modified.
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Policy exceeds 12 MiB limit')
    raw = path.read_bytes()
    raw_hash = sha256(raw).hexdigest()
    kind = path.suffix.lower()
    empty = []
    if kind in ('.txt', '.md'):
        parts = [(1, raw.decode('utf-8-sig'), 'text_block')]
    elif kind == '.pdf':
        # v0.18: a file pypdf cannot open (random bytes, a half-finished download) used to fail
        # with pypdf's own exception, which the workspace reports only by its type name.
        try:
            reader = PdfReader(io.BytesIO(raw))
        except Exception as exc:
            raise ValueError(f'corrupted: "{path.name}" PDF olarak açılamadı; dosya bozuk, yarım inmiş ya da PDF değil. '
                             'Belgeyi kaynağından yeniden indirip ya da dışa aktarıp yükleyin.') from exc
        # Regulators publish PDFs locked with an owner password only (FCA final notices and
        # Dear CEO letters, measured 23 September 2026: 15 of 15 FCA files); those open with
        # an empty user password and are readable. A real password still refuses.
        if reader.is_encrypted and not reader.decrypt(''):
            raise ValueError('Encrypted policy PDF must be decrypted by its owner first')
        try:
            count = len(reader.pages)
        except Exception as exc:
            raise ValueError(f'corrupted: "{path.name}" PDF sayfa ağacı okunamadı; dosya bozuk. '
                             'Belgeyi kaynağından yeniden indirip ya da dışa aktarıp yükleyin.') from exc
        if count > 200:
            raise ValueError('Pilot supports at most 200 PDF pages')
        # A damaged page object or content stream fails on its own page. Such a page is empty and
        # counted as failed; a document whose pages mostly fail is corrupted, not scanned.
        parts, failed = [], []
        for index in range(count):
            try:
                text = reader.pages[index].extract_text() or ''
            except Exception:
                text = ''
                failed.append(index+1)
            parts.append((index+1, text, 'pdf_page'))
        # A cover page that is one image, or a blank page, is common in real documents (MASAK
        # guides); a document that is mostly image is scanned and needs OCR. The empty pages
        # are skipped and counted, never silently passed as text.
        empty = [number for number, text, _ in parts if not text.strip()]
        quality = document_quality(parts, empty, failed)
        if quality['state'] == 'corrupted' and refuse_unreadable:
            raise ValueError(corrupted_message(path.name, quality))
        if quality['requires_ocr']:
            raise ValueError('requires_ocr: most PDF pages have no extractable text (scanned or image-only document); '
                             'OCR or a reviewed text export is required before analysis')
        parts = [part for part in parts if part[1].strip()]
        if parser in (PARSER, 'pilot-policy-v2'):
            parts = [(number, strip_running_lines(text, parts), locator) for number, text, locator in parts]
    elif kind == '.csv':
        # A control register: one control per row, the header names the columns. Each row
        # becomes one passage the judge reads like any policy paragraph.
        import csv
        rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
        if len(rows) < 2 or not any(cell.strip() for cell in rows[0]):
            raise ValueError('Kontrol kaydı CSV: bir başlık satırı ve en az bir kontrol satırı gerekir')
        headers = [h.strip() for h in rows[0]]
        parts = [(index, ' — '.join(f'{h}: {" ".join(v.split())}' for h, v in zip(headers, row) if v.strip()), 'control_row')
                 for index, row in enumerate(rows[1:], 1)]
    elif kind == '.docx':
        # v0.18: a .docx that is not a Word package (a renamed file, a half-finished download, a
        # broken document.xml) is corrupted like a PDF pypdf cannot open; it used to reach the
        # workspace as 'İşlem tamamlanamadı: BadZipFile'.
        broken = (f'corrupted: "{path.name}" Word belgesi olarak açılamadı; dosya bozuk, yarım inmiş ya da DOCX değil. '
                  'Belgeyi kaynağından yeniden indirip ya da dışa aktarıp yükleyin.')
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                member = archive.getinfo('word/document.xml')
                if member.file_size > MAX_BYTES:
                    raise ValueError('Expanded Word document exceeds size limit')
                xml = archive.read(member)
        except (zipfile.BadZipFile, KeyError, EOFError, zlib.error) as exc:
            raise ValueError(broken) from exc
        if b'<!DOCTYPE' in xml.upper() or b'<!ENTITY' in xml.upper():
            raise ValueError('XML entities are not allowed in policy documents')
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            raise ValueError(broken) from exc
        parts = [(index+1, ''.join(t.text or '' for t in p.iter(ns+'t')), 'docx_paragraph')
                 for index, p in enumerate(root.iter(ns+'p'))]
    else:
        raise ValueError('Use UTF-8 TXT/MD, text PDF, DOCX policy files or a CSV control register')
    if kind != '.pdf':
        quality = text_quality(parts)
        if quality['state'] == 'corrupted' and refuse_unreadable:
            raise ValueError(corrupted_message(path.name, quality))
    chunks = []
    for number, text, locator in parts:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Preserve exact extracted substrings and offsets, including long paragraphs.
        for match in re.finditer(r'[^\n]+(?:\n(?!\n)[^\n]+)*', text):
            for offset, end in cut_points(text, match.start(), match.end(), parser):
                quote = text[offset:end]
                if quote.strip():
                    cid = digest([raw_hash, locator, number, offset, end])
                    chunks.append({'source_id': cid, 'policy_hash': raw_hash, 'filename': path.name,
                        'locator': locator, 'number': number, 'start': offset, 'end': end, 'text': quote})
    # An empty file, a Word document without paragraph text, or a PDF left with nothing once its
    # running header and page numbers are stripped: an analysis of it would report every duty
    # uncovered as if the policy had been read.
    if not chunks and refuse_unreadable:
        raise ValueError(f'empty: "{path.name}" içinde analiz edilecek metin yok; boş bir policy analiz edilmez. '
                         'Metni olan belgeyi yükleyin; taranmış bir PDF ise OCR gerekir.')
    return {'filename': path.name, 'raw_hash': raw_hash, 'bytes': len(raw),
            'parser': parser, 'chunks': chunks, 'empty_pages': empty, 'document_quality': quality}


# v0.17 document quality: a PDF is classified before it is analysed, so an image-only or thin
# document is named as such instead of being analysed as if it were empty text.
LOW_DENSITY_CHARS = 200
# v0.18 states, in the pilot brief's names. 'kind' keeps its v0.17 values for the records and the
# screen that read it; 'state' names the format and adds corrupted.
PDF_STATES = {'text': 'text_pdf', 'mixed': 'mixed_pdf', 'low_density': 'low_text_density', 'scanned': 'scanned_pdf'}
# requires_ocr stays the hard stop it was in v0.17: only a scanned PDF is refused. The brief asks
# OCR for low quality as well, but a mixed or thin PDF still carries policy text that v0.17
# analysed, and test_phase17 pins requires_ocr False for both; refusing them would drop readable
# text. ocr_recommended carries the recommendation to the packet (policies[].document_quality)
# and the job record (policy_quality) instead.
OCR_RECOMMENDED = {'scanned_pdf', 'mixed_pdf', 'low_text_density'}
# Text that is mostly not letters is a broken extraction, not a policy: control characters, U+FFFD
# or private-use glyphs from a font without a Unicode map, or page numbers alone. Measured on the
# 113 distinct files under data/, samples/ and evaluation/fixtures/policies (88 PDF, 25 TXT/MD/CSV;
# 24 September 2026): letters were at least 78% of the visible characters in every one.
MIN_LETTER_PERCENT = 30


def letter_percent(parts):
    """Letters as a whole percentage of the visible characters; None when nothing is visible.
    An integer, because the record goes into the packet and evidence forbids floats."""
    visible = letters = 0
    for _, text, _ in parts:
        visible += len(''.join(text.split()))
        letters += sum(map(str.isalpha, text))
    return letters * 100 // visible if visible else None


def document_quality(parts, empty, failed=()) -> dict:
    """kind in text | mixed | low_density | scanned, from the share of empty pages and the text density.

    state is kind in the brief's names, or corrupted when most pages failed to extract or the
    text is mostly not letters; corrupted is decided before scanned only for failed pages, since
    a scanned document is refused for OCR either way."""
    pages = len(parts)
    lengths = [len(text.strip()) for _, text, _ in parts]
    mean = int(sum(lengths) / pages) if pages else 0
    if pages and len(empty) * 2 > pages:
        kind = 'scanned'
    elif pages and len(empty) * 10 > pages:
        kind = 'mixed'
    elif mean < LOW_DENSITY_CHARS:
        kind = 'low_density'
    else:
        kind = 'text'
    percent = letter_percent(parts)
    if pages and len(failed) * 2 > pages:
        state, note = 'corrupted', f'Bozuk PDF: {len(failed)}/{pages} sayfanın metni çıkarılamadı.'
    elif kind != 'scanned' and percent is not None and percent < MIN_LETTER_PERCENT:
        state, note = 'corrupted', corrupted_text_note(percent)
    else:
        state = PDF_STATES[kind]
        note = {'text_pdf': 'Metin katmanlı PDF; metin doğrudan okundu.',
                'mixed_pdf': f'Karma PDF: {len(empty)}/{pages} sayfada okunabilir metin yok (görsel ya da boş sayfa); '
                             'bu sayfalar analize girmedi. OCR önerilir.',
                'low_text_density': f'Düşük metin yoğunluğu: sayfa başına ortalama {mean} karakter; '
                                    'metnin bir kısmı görsel olabilir. OCR önerilir.',
                'scanned_pdf': f'Taranmış ya da görsel PDF: {len(empty)}/{pages} sayfada okunabilir metin yok. '
                               'Analiz için OCR ya da incelenmiş bir metin kopyası gerekir.'}[state]
        if failed:
            note += f' {len(failed)} sayfanın içeriği bozuk, metni çıkarılamadı.'
    return {'kind': kind, 'pages': pages, 'empty_pages': len(empty), 'mean_chars_per_page': mean, 'requires_ocr': state == 'scanned_pdf',
            'state': state, 'ocr_recommended': state in OCR_RECOMMENDED, 'failed_pages': len(failed), 'letter_percent': percent,
            'quality_note': note}


def text_quality(parts) -> dict:
    """TXT, MD, DOCX and CSV have no pages to lose: state text, unless the text is mostly not letters."""
    percent = letter_percent(parts)
    corrupted = percent is not None and percent < MIN_LETTER_PERCENT
    return {'kind': 'text', 'pages': len(parts), 'empty_pages': 0, 'mean_chars_per_page': int(sum(len(t) for _, t, _ in parts) / max(1, len(parts))),
            'requires_ocr': False, 'state': 'corrupted' if corrupted else 'text', 'ocr_recommended': False, 'failed_pages': 0,
            'letter_percent': percent, 'quality_note': corrupted_text_note(percent) if corrupted else 'Metin belgesi; metin doğrudan okundu.'}


def corrupted_text_note(percent) -> str:
    return (f'Bozuk metin: görünen karakterlerde harf oranı %{percent} (denetim karakteri, U+FFFD, eşlemesiz yazı tipi '
            'glifi ya da yalnız sayfa numarası); metin katmanı bozuk ya da belge taranmış olabilir.')


def corrupted_message(name, quality) -> str:
    # Counts and a percentage only: the refusal reaches the job record and the logs, the document's text never does.
    return (f'corrupted: "{name}" okunabilir policy metni vermiyor. {quality["quality_note"]} Bozuk metin analiz edilmez; '
            'belgeyi kaynağından yeniden dışa aktarın ya da OCR uygulanmış bir kopya yükleyin.')


PROVISION_LABEL = re.compile(r'\b([A-Z]{2,6}) (\d{1,2}[A-Z]?)\.\d{1,2}[A-Z]?\.-?\d{1,3}[A-Z]?\b')


def handbook_extract(policy):
    """'COBS 4' when an uploaded "policy" is really an FCA Handbook export, else None.

    Users repeatedly uploaded the regulation PDF as the company policy; the analysis then
    compared one regulation with another and looked finished. A real policy cites a few
    provisions. A Handbook export carries the site address and provision labels by the dozen.
    Not part of read_policy(): retained runs that contain such a file must stay verifiable.
    """
    text = '\n'.join(chunk['text'] for chunk in policy['chunks'])
    labels = PROVISION_LABEL.findall(text)
    if 'handbook.fca.org.uk' not in text.lower() or len(labels) < 25:
        return None
    module, chapter = max(set(labels), key=labels.count)
    return f'{module} {chapter}'


MEVZUAT_ARTICLE = re.compile(r'(?:^|\s)(?:MADDE|Madde)\s+\d+(?:/[A-Z])?\s*[–\-—]')
STOPWORDS = {'the', 'and', 'must', 'firm', 'with', 'should',
             'ile', 'için', 'veya', 'olan', 'ilişkin', 'gibi', 'dair', 'olarak', 'tarafından', 'üzere', 'ise', 'yapılan', 'bir', 'her'}


def fold(text: str) -> str:
    # Python folds the Turkish capitals wrongly (İ -> i̇, I -> i); words must still match.
    return text.replace('İ', 'i').replace('I', 'ı').lower()


def lexical_terms(action: str):
    return set(re.findall(r'[a-zçğıöşü]{3,}', fold(action))) - STOPWORDS


def lexical_score(terms, text: str) -> int:
    return len(terms & set(re.findall(r'[a-zçğıöşü]{3,}', fold(text))))


def mevzuat_extract(policy):
    """The number of articles when an uploaded "policy" is really Turkish legislation, else 0.

    The same misuse as handbook_extract: a law or regulation text (mevzuat.gov.tr export,
    Official Gazette copy) carries 'MADDE n –' lines by the dozen; a company policy that
    cites the law does not.
    """
    text = '\n'.join(chunk['text'] for chunk in policy['chunks'])
    articles = len(MEVZUAT_ARTICLE.findall(text))
    cues = ('mevzuat.gov.tr' in text.lower() or 'resmî gazete' in text.lower() or 'resmi gazete' in text.lower()
            or bool(re.search(r'Kanun Numarası|Yayımlandığı Resmî Gazete|BİRİNCİ BÖLÜM', text)))
    return articles if cues and articles >= 15 else 0


def select_chunks(action: str, policies, limit=6, max_chars=6500):
    terms = lexical_terms(action)
    ranked = []
    for policy in policies:
        for chunk in policy['chunks']:
            ranked.append((lexical_score(terms, chunk['text']), chunk['source_id'], chunk))
    chosen = []
    used = 0
    for score, _, chunk in sorted(ranked, key=lambda row: (-row[0], row[1])):
        if score == 0 or len(chosen) >= limit:
            break
        if used+len(chunk['text']) > max_chars:
            continue
        chosen.append(chunk)
        used += len(chunk['text'])
    return chosen

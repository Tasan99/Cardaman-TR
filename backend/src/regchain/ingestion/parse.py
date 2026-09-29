import re
from collections import Counter
from datetime import date
from io import BytesIO

from bs4 import BeautifulSoup
from pypdf import PdfReader
import pdfplumber
from .pdf_structure import segment_pages

from .handbook import is_handbook, parse_handbook
from .models import Document, Download, IngestionError, Paragraph


def normalize(text: str) -> str:
    # Do not lowercase or strip punctuation: legal operators and exceptions survive.
    return re.sub(r"\s+", " ", text).strip()


def _date(value: str | None) -> date | None:
    if not value:
        return None
    match = re.search(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", value)
    if match:
        try:
            return date.fromisoformat(match[1])
        except ValueError:
            pass
    return None


def parse_html(source: Download) -> Document:
    soup = BeautifulSoup(source.body, "html.parser")
    title_node = soup.find("h1") or soup.find("title")
    if not title_node:
        raise IngestionError("HTML has no document title; source needs review")
    title = normalize(title_node.get_text(" "))
    if any(marker in title.lower() for marker in ('access denied', 'just a moment', 'captcha')):
        raise IngestionError("Source returned an access challenge; manual review required")
    published = soup.find("meta", attrs={"property": "article:published_time"}) or soup.find("meta", attrs={"name": "date"})
    publication_date = _date(published.get("content")) if published else None
    # FCA publication pages can retain an earlier CMS article:published_time.
    # Prefer their explicitly labelled publication date, not the CMS creation date.
    published_field = soup.find('meta', attrs={'property':'funnelback:published-date'})
    if published_field:
        publication_date = _date(published_field.get('content','').replace('/', '-')) or publication_date
    # Only machine-labelled effective dates are lifted into metadata. Prose is retained.
    effective = soup.find("meta", attrs={"name": "effective-date"})
    effective_date = _date(effective.get("content")) if effective else None
    main = soup.select_one('article[data-history-node-id]') or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article")
    if main is None:
        raise IngestionError("No main/article content found; JavaScript-only sources need another adapter")
    for node in main.select("script, style, nav, header, footer, form, aside, noscript, button, iframe, [aria-hidden='true']"):
        node.decompose()
    headings: list[tuple[int, str]] = []
    counters: Counter = Counter()
    seen: Counter = Counter()
    section = "preamble"
    paragraphs = []
    for node in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "blockquote"]):
        if node.name.startswith("h"):
            level = int(node.name[1])
            heading = normalize(node.get_text(" "))
            if not heading:
                continue
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading))
            section = str(node.get("id") or " / ".join(h[1] for h in headings))
            continue
        # Keep each list/table/blockquote as a whole to preserve exception context;
        # do not emit its nested paragraphs a second time.
        if node.find_parent(["p", "li", "tr", "blockquote"]):
            continue
        text = normalize(node.get_text(" "))
        if not text:
            continue
        counters[section] += 1
        anchor = node.get("id")
        number = str(anchor or counters[section])
        seen[(section, number)] += 1
        if seen[(section, number)] > 1:
            number = f"{number}~{seen[(section, number)]}"
        paragraphs.append(Paragraph(section, number, text, tuple(h[1] for h in headings),
                                    locator_kind="html_id" if anchor else "generated"))
    if not paragraphs:
        raise IngestionError("No source paragraphs extracted; source needs review")
    if sum(len(p.text) for p in paragraphs) > 2_000_000:
        raise IngestionError("Extracted HTML text exceeds limit")
    warnings = ["HTML paragraph locators are generated unless locator_kind=html_id; they are not official rule numbers."]
    if publication_date is None:
        warnings.append("Publication date unavailable in supported machine-readable metadata.")
    return Document(title, tuple(paragraphs), publication_date, effective_date, tuple(warnings))


def parse_pdf(source: Download) -> Document:
    if not source.body.startswith(b"%PDF-"):
        raise IngestionError("PDF signature does not match content type")
    try:
        reader = PdfReader(BytesIO(source.body), strict=True)
        # Public FCA PDFs may use permissions encryption but have an empty open
        # password. Accept only that case; never guess or request passwords.
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestionError("Encrypted PDF requires manual review")
        if len(reader.pages) > 500:
            raise IngestionError("PDF exceeds 500 page limit")
        title = normalize(str(reader.metadata.title or "")) if reader.metadata else ""
        pages = []
        total = 0
        with pdfplumber.open(BytesIO(source.body),password='') as pdf:
            for page_no,page in enumerate(pdf.pages,1):
                text = page.extract_text(x_tolerance=1,y_tolerance=3) or ''
                total += len(text)
                if total > 2_000_000:
                    raise IngestionError('Extracted PDF text exceeds limit')
                if not normalize(text):
                    raise IngestionError(f'PDF page {page_no} has no extractable text; OCR/manual review required')
                pages.append(text.splitlines())
                page.close()
        paragraphs = segment_pages(pages,commentary='policy statement' in title.lower() or 'consultation' in title.lower())
        if not paragraphs:
            raise IngestionError("PDF contains no text")
        title = title or paragraphs[0].text[:240]
        return Document(title, tuple(paragraphs), warnings=(
            "pdf_printed labels are transcribed locators, not confirmation of operative legal status.",
            "Amendment/redline passages require a consolidated source; deleted and inserted text are not merged into law.",
            "Dates in PDF prose require review; PDF creation timestamp is not publication date.",
        ))
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError("PDF parsing failed; source requires review") from exc


def parse(source: Download) -> Document:
    if source.media_type in {"text/html", "application/xhtml+xml"}:
        # Consolidated Handbook pages carry printed labels, provision types and
        # effective dates. Publication pages do not; they must not be treated alike.
        return parse_handbook(source) if is_handbook(source) else parse_html(source)
    if source.media_type == "application/pdf":
        return parse_pdf(source)
    raise IngestionError("Unsupported source media type")

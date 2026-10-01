"""Consolidated FCA Handbook provisions and glossary definitions.

A policy statement appendix is an amendment instrument: it states how the law is
changed, not what the law now says. Operative text lives in the consolidated Handbook.
This adapter reads the server-rendered consolidated view only. Point-in-time (timeline)
views are rendered client side and are rejected rather than guessed.
"""
import re
from datetime import date
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, NavigableString
from bs4.element import PreformattedString

from .models import Document, Download, IngestionError, Paragraph

HANDBOOK_HOSTS = frozenset({"handbook.fca.org.uk", "www.handbook.fca.org.uk"})
# A provision label is a locator transcribed from the source, not a legal conclusion.
# The FCA numbers provisions inserted ahead of the first one negatively ("COBS 4.7.-2").
# Accepting that form changes nothing for a page that parsed before; it only stops a
# whole chapter being rejected for one such label.
PROVISION = re.compile(r"^([A-Z]{2,6})(?:\s+(App))?\s+([0-9][0-9A-Z]*(?:\.-?[0-9A-Z]+(?:-[A-Z])?)*)$")
PROVISION_TYPES = {"R": "RULE", "G": "GUIDANCE", "D": "DIRECTION", "E": "EVIDENTIAL"}
UK_DATE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
# Glossary identifiers are case sensitive on the FCA site; never normalise them.
GLOSSARY_HREF = re.compile(r"^/glossary/(G[0-9A-Za-z]{1,64})$")
# A chapter page ends with its annexes ("COBS 4 Annex 1R"): prescribed wording and tables
# that can be binding. The trailing letter repeats the provision type shown on the card.
ANNEX = re.compile(r"^([A-Z]{2,6})\s+([0-9][0-9A-Z]*)\s+Annex\s+([0-9][0-9A-Z]*?)([RGDE]?)$")
DELETED = re.compile(r"^\[\s*deleted\s*\]", re.I)
DELETED_SECTION = re.compile(r"^[A-Z]{2,6}(?:\s+App)?\s+[0-9][0-9A-Z.\-]*\s+\[\s*deleted\s*\]$", re.I)
BLOCK = frozenset({"p", "li", "ol", "ul", "div", "table", "thead", "tbody", "tr", "td", "th",
                   "br", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "section"})
# Never part of a provision's text, and never allowed to leak into a quote.
SKIPPED = frozenset({"script", "style", "template", "noscript", "head"})
CHALLENGE = ("access denied", "just a moment", "captcha")


def is_handbook(source: Download) -> bool:
    return urlsplit(source.final_url).hostname in HANDBOOK_HOSTS


def block_text(node) -> str:
    """Join inline markup without inventing spaces; break only on block boundaries.

    FCA provisions carry meaning inside inline glossary links (a possessive such as
    customer's is split across anchors), so a naive separator corrupts the quote that
    later has to match the source. Sub-paragraph markers (1) or (a) stay in place.
    """
    parts: list[str] = []

    def walk(element) -> None:
        for child in element.children:
            if isinstance(child, PreformattedString):
                # Comments, CDATA, doctypes and processing instructions are
                # NavigableString subclasses. They are markup, not the provision.
                continue
            if isinstance(child, NavigableString):
                parts.append(str(child))
            elif child.name in SKIPPED:
                continue
            elif child.name in BLOCK:
                parts.append("\n")
                walk(child)
                parts.append("\n")
            else:
                walk(child)

    walk(node)
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parts).split("\n")]
    return " ".join(line for line in lines if line)


def provision_date(header) -> date | None:
    for span in header.select("span"):
        match = UK_DATE.match(span.get_text(strip=True))
        if match:
            try:
                return date(int(match[3]), int(match[2]), int(match[1]))
            except ValueError:
                return None
    return None


def defined_terms(node) -> tuple[dict, ...]:
    """Glossary anchors are the regulator's own definition links, not a lexical guess."""
    seen: dict[str, dict] = {}
    for anchor in node.select("a[href]"):
        match = GLOSSARY_HREF.match(anchor.get("href", ""))
        term = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
        if match and term:
            seen.setdefault(match[1], {"glossary_id": match[1], "term": term})
    return tuple(seen[key] for key in sorted(seen))


def cards_root(soup: BeautifulSoup):
    cards = soup.select("div.p-card-content")
    if not cards:
        return None, []
    root = cards[0].parent
    while root is not None and not all(card in root.descendants for card in cards):
        root = root.parent
    return root or soup, cards


CHAPTER_PATH = re.compile(r"^/handbook/([A-Z]{2,6})/([0-9][0-9A-Z]*)/([0-9][0-9A-Z]*)\.html$", re.I)


def requested_chapter(source: Download) -> str | None:
    """The chapter the operator asked for, read from the curated request URL."""
    match = CHAPTER_PATH.match(urlsplit(source.requested_url).path)
    return f"{match[1].upper()} {match[2].upper()}.{match[3].upper()}" if match else None


def glossary_id(source: Download) -> str | None:
    match = GLOSSARY_HREF.match(urlsplit(source.final_url).path.rstrip("/"))
    return match[1] if match else None


def parse_handbook(source: Download) -> Document:
    soup = BeautifulSoup(source.body, "html.parser")
    title_node = soup.find("h1") or soup.find("title")
    title = re.sub(r"\s+", " ", title_node.get_text(" ")).strip() if title_node else ""
    if not title:
        raise IngestionError("Handbook page has no title; source needs review")
    if any(marker in title.lower() for marker in CHALLENGE):
        raise IngestionError("Source returned an access challenge; manual review required")
    root, cards = cards_root(soup)
    if not cards:
        raise IngestionError("No server-rendered Handbook provisions found; timeline, "
                             "point-in-time and index views need a different adapter")
    term_id = glossary_id(source)
    headings: list[tuple[int, str]] = []
    paragraphs: list[Paragraph] = []
    skipped: list[str] = []
    total = 0
    for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "div"]):
        if node.name != "div":
            level = int(node.name[1])
            heading = re.sub(r"\s+", " ", node.get_text(" ")).strip()
            if heading:
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, heading))
            continue
        if "p-card-content" not in (node.get("class") or []):
            continue
        header = node.select_one("div.header")
        # Provision cards and glossary entries are rendered with different wrappers.
        content = node.select_one("div.section-content") or node.select_one("div.details_wrap")
        label_node = header.select_one("label span") or header.select_one("h1") if header else None
        if header is None or content is None or label_node is None:
            raise IngestionError("Handbook provision card is missing its label or content")
        label = re.sub(r"\s+", " ", label_node.get_text(" ")).strip()
        if DELETED_SECTION.match(label):
            # A whole-chapter page keeps a card for a section that no longer exists
            # ("COBS 4.12 [deleted]"). It carries no provision; refusing the chapter for
            # it would hide every rule that is in force. It is recorded, not dropped.
            skipped.append(label)
            continue
        text = block_text(content)
        if not text:
            raise IngestionError(f"Handbook provision {label or 'unlabelled'} has no extractable text")
        total += len(text)
        if total > 2_000_000:
            raise IngestionError("Extracted Handbook text exceeds limit")
        marker = header.select_one("span.provison-type")
        code = marker.get_text(strip=True).upper() if marker else ""
        flags: list[str] = []
        match = PROVISION.match(label)
        annex = None if match else ANNEX.match(label)
        if annex:
            section = f"{annex[1]} {annex[2]} Annex"
            number = f"Annex {annex[3]}{annex[4]}"
            legal_type = PROVISION_TYPES.get(code or annex[4], "UNKNOWN")
            if legal_type == "UNKNOWN":
                flags.append("PROVISION_TYPE_UNRECOGNISED")
        elif match:
            book, app, numeric = match[1], match[2], match[3]
            section = f"{book} {app} " if app else f"{book} "
            section += ".".join(numeric.split(".")[:2])
            number = f"{app} {numeric}" if app else numeric
            legal_type = PROVISION_TYPES.get(code, "UNKNOWN")
            if legal_type == "UNKNOWN":
                flags.append("PROVISION_TYPE_UNRECOGNISED")
        elif term_id:
            section = "glossary"
            number = term_id
            legal_type = "DEFINITION"
        else:
            raise IngestionError(f"Unrecognised Handbook provision label: {label[:80]!r}")
        effective_from = provision_date(header)
        if effective_from is None and not term_id:
            flags.append("NO_PROVISION_EFFECTIVE_DATE")
        if DELETED.match(text):
            flags.append("DELETED_PROVISION")
        paragraphs.append(Paragraph(
            section=section, number=number, text=text,
            heading_path=tuple(item[1] for item in headings),
            locator_kind="handbook_glossary" if term_id else "handbook_provision",
            printed_label=label, source_kind="CONSOLIDATED",
            quality_flags=tuple(flags), legal_type=legal_type,
            effective_from=effective_from, defined_terms=defined_terms(content)))
    locators = [(p.section, p.number) for p in paragraphs]
    if len(set(locators)) != len(locators):
        raise IngestionError("Duplicate Handbook provision locators; source needs review")
    # The FCA serves a neighbouring chapter for some withdrawn locators. Storing that
    # silently would attach one chapter's rule text to another chapter's identity.
    expected = requested_chapter(source)
    delivered = {p.section for p in paragraphs}
    if expected and delivered and expected not in delivered:
        raise IngestionError(
            f"Requested {expected} but the source delivered {', '.join(sorted(delivered))}; "
            "the chapter may have been withdrawn or renumbered and needs review")
    warnings = [
        "Consolidated Handbook view as served at retrieval time; it is not a point-in-time "
        "legal snapshot and supersession after retrieval is not detected here.",
        "effective_from is the date printed against each provision, not a confirmation that "
        "the provision applies to any particular firm.",
    ]
    if term_id:
        warnings.append("Glossary definitions are Handbook-wide and remain qualified by their own scope wording.")
    if skipped:
        warnings.append("Deleted section placeholders carry no provision and were not stored: " + ", ".join(skipped))
    if not paragraphs:
        raise IngestionError("Handbook page holds only deleted section placeholders")
    return Document(title, tuple(paragraphs), warnings=tuple(warnings))

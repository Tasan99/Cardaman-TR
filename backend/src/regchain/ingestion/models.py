from dataclasses import dataclass, field
from datetime import date, datetime
from hashlib import sha256

from regchain.evidence import digest

PARSER_VERSION = "fca-parser-v4"
NORMALIZER_VERSION = "whitespace-v1"


class IngestionError(Exception):
    """Expected, actionable ingestion failure; no partial version is committed."""


@dataclass(frozen=True)
class Download:
    requested_url: str
    final_url: str
    body: bytes
    media_type: str
    fetched_at: datetime
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def raw_hash(self) -> str:
        return sha256(self.body).hexdigest()


@dataclass(frozen=True)
class Paragraph:
    section: str
    number: str
    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    locator_kind: str = "generated"
    page_end: int | None = None
    printed_label: str | None = None
    source_kind: str = 'UNKNOWN'
    quality_flags: tuple[str, ...] = ()
    source_spans: tuple[dict, ...] = ()
    legal_type: str = 'UNKNOWN'
    # The date printed against a consolidated provision, never an applicability decision.
    effective_from: date | None = None
    defined_terms: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Document:
    title: str
    paragraphs: tuple[Paragraph, ...]
    publication_date: date | None = None
    effective_date: date | None = None
    warnings: tuple[str, ...] = ()

    @property
    def normalized_text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def content_hash(self) -> str:
        # Metadata and structural changes must not disappear under body-only dedup.
        return digest({
            "title": self.title,
            "publication_date": str(self.publication_date) if self.publication_date else None,
            "effective_date": str(self.effective_date) if self.effective_date else None,
            "paragraphs": [
                {"section": p.section, "number": p.number, "text": p.text,
                 "heading_path": list(p.heading_path), "page": p.page,
                 "locator_kind": p.locator_kind, "page_end":p.page_end,
                 "printed_label":p.printed_label,"source_kind":p.source_kind,
                 "quality_flags":list(p.quality_flags),"source_spans":list(p.source_spans),
                 "legal_type":p.legal_type,
                 "effective_from":str(p.effective_from) if p.effective_from else None,
                 "defined_terms":list(p.defined_terms)}
                for p in self.paragraphs
            ],
        })

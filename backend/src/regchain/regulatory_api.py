"""Read-only official-source endpoints; no tenant or analysis data is exposed."""
import os
from contextlib import contextmanager
from datetime import date, datetime
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

router = APIRouter(tags=["Regulatory sources"])


class RegulationView(BaseModel):
    id: UUID
    external_key: str
    title: str
    canonical_url: str
    latest_version: int


class VersionView(BaseModel):
    id: UUID
    regulation_id: UUID
    version_no: int
    retrieved_at: datetime
    publication_date: date | None
    effective_date: date | None
    source_url: str
    raw_hash: str
    normalized_hash: str
    parser_version: str
    metadata: dict


class ParagraphView(BaseModel):
    id: UUID
    version_id: UUID
    section_number: str
    paragraph_number: str
    page: int | None
    heading_path: list[str]
    text: str
    legal_type: str
    content_hash: str
    ordinal: int | None
    locator_kind: str
    page_end: int | None = None
    printed_label: str | None = None
    source_kind: str = 'UNKNOWN'
    quality_flags: list[str] = []
    source_spans: list[dict] = []


class ChangeView(BaseModel):
    id: UUID
    regulation_id: UUID
    old_version_id: UUID
    new_version_id: UUID
    old_section_id: UUID | None
    new_section_id: UUID | None
    kind: str
    diff_text: str
    created_at: datetime


@contextmanager
def database():
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=3, row_factory=dict_row) as conn:
            conn.execute("SET LOCAL statement_timeout='5s'")
            yield conn
    except (psycopg.Error, KeyError):
        raise HTTPException(status_code=503, detail="Regulatory database unavailable") from None


def require(conn, query: str, identifier: UUID):
    row = conn.execute(query, (identifier,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Regulatory resource not found")
    return row


@router.get("/regulations", response_model=list[RegulationView])
def regulations(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    with database() as conn:
        return conn.execute("""SELECT r.id,r.external_key,COALESCE(v.metadata->>'title',r.title) title,
            r.canonical_url,v.version_no latest_version FROM regulations r
            JOIN LATERAL (SELECT version_no,metadata FROM regulation_versions WHERE regulation_id=r.id
                ORDER BY version_no DESC LIMIT 1) v ON true
            ORDER BY r.created_at DESC,r.id LIMIT %s OFFSET %s""", (limit, offset)).fetchall()


@router.get("/regulations/{regulation_id}/versions", response_model=list[VersionView])
def versions(regulation_id: UUID, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    with database() as conn:
        require(conn, "SELECT id FROM regulations WHERE id=%s", regulation_id)
        return conn.execute("SELECT * FROM regulation_versions WHERE regulation_id=%s ORDER BY version_no DESC LIMIT %s OFFSET %s",
                            (regulation_id, limit, offset)).fetchall()


@router.get("/versions/{version_id}", response_model=VersionView)
def version(version_id: UUID):
    with database() as conn:
        return require(conn, "SELECT * FROM regulation_versions WHERE id=%s", version_id)


@router.get("/versions/{version_id}/paragraphs", response_model=list[ParagraphView])
def paragraphs(version_id: UUID, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)):
    with database() as conn:
        require(conn, "SELECT id FROM regulation_versions WHERE id=%s", version_id)
        return conn.execute("""SELECT * FROM regulation_sections WHERE version_id=%s
            ORDER BY ordinal NULLS LAST,section_number,paragraph_number LIMIT %s OFFSET %s""", (version_id, limit, offset)).fetchall()


@router.get("/versions/{version_id}/changes", response_model=list[ChangeView])
def changes(version_id: UUID, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)):
    with database() as conn:
        require(conn, "SELECT id FROM regulation_versions WHERE id=%s", version_id)
        return conn.execute("SELECT * FROM regulatory_changes WHERE new_version_id=%s ORDER BY created_at,id LIMIT %s OFFSET %s",
                            (version_id, limit, offset)).fetchall()


@router.get('/versions/{version_id}/paragraphs/{section_id}/context')
def paragraph_context(version_id: UUID,section_id: UUID):
    from .extraction.retrieval import retrieve
    with database() as conn:
        target = require(conn,'SELECT * FROM regulation_sections WHERE id=%s',section_id)
        if target['version_id'] != version_id:
            raise HTTPException(status_code=404,detail='Regulatory resource not found')
        sections = conn.execute('SELECT * FROM regulation_sections WHERE version_id=%s ORDER BY ordinal NULLS LAST,section_number,paragraph_number',
                                (version_id,)).fetchall()
        return retrieve(target,sections).manifest()


@router.get("/regulatory/feed", response_model=list[ChangeView])
def feed(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    with database() as conn:
        return conn.execute("SELECT * FROM regulatory_changes ORDER BY created_at DESC,id LIMIT %s OFFSET %s", (limit, offset)).fetchall()


@router.get("/versions/{version_id}/source")
def source(version_id: UUID):
    with database() as conn:
        row = require(conn, """SELECT o.body,o.raw_hash FROM regulation_versions v
            JOIN regulatory_source_objects o ON o.raw_hash=v.raw_hash WHERE v.id=%s""", version_id)
        return Response(bytes(row["body"]), media_type="application/octet-stream", headers={
            "Content-Disposition": f'attachment; filename="{row["raw_hash"]}.bin"',
            "X-Content-SHA256": row["raw_hash"], "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "sandbox; default-src 'none'",
        })


class ComparisonView(BaseModel):
    id: UUID
    regulation_id: UUID
    old_version_id: UUID
    new_version_id: UUID
    parser_version: str
    normalizer_version: str
    old_raw_hash: str
    new_raw_hash: str
    old_reparsed_hash: str
    new_normalized_hash: str
    changed_paragraphs: int
    verdict: str
    human_review_required: bool
    created_at: datetime


class ComparisonChangeView(BaseModel):
    id: UUID
    comparison_id: UUID
    kind: str
    old_locator: str | None
    old_text: str | None
    new_section_id: UUID | None
    diff_text: str


@router.get("/versions/{version_id}/source-comparisons", response_model=list[ComparisonView])
def source_comparisons(version_id: UUID, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    """Comparisons that held the parser constant across a simultaneous source change.

    SOURCE_UNCHANGED means the version bump came from our parser alone; SOURCE_CHANGED
    means the regulator's bytes really differ. Neither verdict is a legal finding.
    """
    with database() as conn:
        require(conn, "SELECT id FROM regulation_versions WHERE id=%s", version_id)
        return conn.execute("""SELECT * FROM source_comparisons WHERE new_version_id=%s
            ORDER BY created_at DESC,id LIMIT %s OFFSET %s""", (version_id, limit, offset)).fetchall()


@router.get("/source-comparisons/{comparison_id}/changes", response_model=list[ComparisonChangeView])
def source_comparison_changes(comparison_id: UUID, limit: int = Query(100, ge=1, le=200),
                              offset: int = Query(0, ge=0)):
    """Differences attributable to the source once the parser is held constant.

    The earlier side is a locator from the reparsed archived bytes, not a stored
    section: those paragraphs were never persisted under the current parser.
    """
    with database() as conn:
        require(conn, "SELECT id FROM source_comparisons WHERE id=%s", comparison_id)
        return conn.execute("""SELECT * FROM source_comparison_changes WHERE comparison_id=%s
            ORDER BY kind,old_locator NULLS LAST,id LIMIT %s OFFSET %s""",
            (comparison_id, limit, offset)).fetchall()

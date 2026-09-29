from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel

from .regulatory_api import database, require

router = APIRouter(tags=['Obligation extraction (review required)'])


class Citation(BaseModel):
    regulator: str
    regulation: str
    version_id: UUID
    section_id: UUID
    section: str
    paragraph: str
    page: int | None
    locator_kind: str
    source_url: str
    source_quote: str
    page_end: int | None = None
    printed_label: str | None = None
    source_kind: str = 'UNKNOWN'
    quality_flags: list[str] = []


class ObligationView(BaseModel):
    id: UUID
    extraction_run_id: UUID
    subject: str
    modality: str
    required_action: str | None
    prohibited_action: str | None
    conditions: list[str]
    exceptions: list[str]
    deadline: str | None
    jurisdiction: str
    affected_products: list[str]
    affected_entities: list[str]
    customer_types: list[str]
    risk_category: str
    confidence: Decimal
    human_review_required: bool
    provider: str
    model_version: str
    citation: Citation
    supporting_evidence: list[dict] = []


class RunView(BaseModel):
    id: UUID
    version_id: UUID
    provider: str
    model_version: str
    prompt_hash: str
    schema_version: str
    runtime_manifest: dict = {}
    input_hash: str
    output_hash: str | None
    status: str
    error_code: str | None
    created_at: datetime


class ReviewView(BaseModel):
    id: UUID
    section_id: UUID
    status: str
    reason: str
    attempts: int
    human_review_required: bool
    result: dict


@router.get('/versions/{version_id}/extractions',response_model=list[RunView])
def runs(version_id: UUID,limit: int=Query(50,ge=1,le=100),offset: int=Query(0,ge=0)):
    with database() as conn:
        require(conn,'SELECT id FROM regulation_versions WHERE id=%s',version_id)
        return conn.execute('SELECT * FROM extraction_runs WHERE version_id=%s ORDER BY created_at DESC,id LIMIT %s OFFSET %s',
                            (version_id,limit,offset)).fetchall()


@router.get('/extractions/{run_id}/reviews',response_model=list[ReviewView])
def reviews(run_id: UUID,limit: int=Query(100,ge=1,le=200),offset: int=Query(0,ge=0)):
    with database() as conn:
        require(conn,'SELECT id FROM extraction_runs WHERE id=%s',run_id)
        return conn.execute('SELECT * FROM extraction_reviews WHERE run_id=%s ORDER BY section_id LIMIT %s OFFSET %s',
                            (run_id,limit,offset)).fetchall()


@router.get('/versions/{version_id}/obligations',response_model=list[ObligationView])
def obligations(version_id: UUID,run_id: UUID | None=None,limit: int=Query(100,ge=1,le=200),offset: int=Query(0,ge=0)):
    with database() as conn:
        require(conn,'SELECT id FROM regulation_versions WHERE id=%s',version_id)
        # Default: latest run. Historical runs are never silently pooled or overwritten.
        return conn.execute('''SELECT o.*,e.provider,e.model_version,
            jsonb_build_object('regulator',g.code,'regulation',COALESCE(v.metadata->>'title',r.title),
             'version_id',v.id,'section_id',s.id,'section',s.section_number,
             'paragraph',s.paragraph_number,'page',s.page,'locator_kind',s.locator_kind,
             'source_url',v.source_url,'source_quote',o.source_quote,
             'page_end',s.page_end,'printed_label',s.printed_label,'source_kind',s.source_kind,
             'quality_flags',s.quality_flags) citation,
            COALESCE((SELECT jsonb_agg(jsonb_build_object('section_id',c.section_id,'role',c.role,
                'quote',c.quote,'source_hash',c.source_hash,'version_id',c.source_version_id,
                'page',cs.page,'page_end',cs.page_end,'printed_label',cs.printed_label,'source_url',cv.source_url)
                ORDER BY c.section_id,c.role) FROM obligation_context_evidence c
                JOIN regulation_sections cs ON cs.id=c.section_id
                JOIN regulation_versions cv ON cv.id=c.source_version_id WHERE c.obligation_id=o.id),'[]'::jsonb) supporting_evidence
            FROM obligations o JOIN extraction_runs e ON e.id=o.extraction_run_id
            JOIN regulation_sections s ON s.id=o.section_id
            JOIN regulation_versions v ON v.id=o.version_id
            JOIN regulations r ON r.id=v.regulation_id JOIN regulators g ON g.id=r.regulator_id
            WHERE o.version_id=%s AND e.status='REVIEW_REQUIRED'
              AND o.extraction_run_id=COALESCE(%s::uuid,
                (SELECT id FROM extraction_runs WHERE version_id=%s ORDER BY created_at DESC,id DESC LIMIT 1))
            ORDER BY s.ordinal NULLS LAST,o.id LIMIT %s OFFSET %s''',
            (version_id,run_id,version_id,limit,offset)).fetchall()


class AmendmentRunView(BaseModel):
    id: UUID
    instrument_version_id: UUID
    matcher_version: str
    status: str
    commencement_date: date | None
    corpus: list[dict]
    created_at: datetime


class AmendmentLinkView(BaseModel):
    id: UUID
    run_id: UUID
    locator: str
    match_basis: str
    commencement_date: date | None
    consolidated_effective_from: date | None
    date_agreement: str
    human_review_required: bool
    instrument: Citation
    consolidated: Citation


class AmendmentGapView(BaseModel):
    id: UUID
    run_id: UUID
    locator: str | None
    reason: str
    instrument: Citation


CITATION = '''jsonb_build_object('regulator',g.code,'regulation',COALESCE(v.metadata->>'title',r.title),
    'version_id',v.id,'section_id',s.id,'section',s.section_number,'paragraph',s.paragraph_number,
    'page',s.page,'locator_kind',s.locator_kind,'source_url',v.source_url,'source_quote',s.text,
    'page_end',s.page_end,'printed_label',s.printed_label,'source_kind',s.source_kind,
    'quality_flags',s.quality_flags)'''


@router.get('/versions/{version_id}/amendment-runs', response_model=list[AmendmentRunView])
def amendment_runs(version_id: UUID, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    with database() as conn:
        require(conn, 'SELECT id FROM regulation_versions WHERE id=%s', version_id)
        return conn.execute('''SELECT * FROM amendment_runs WHERE instrument_version_id=%s
            ORDER BY created_at DESC,id LIMIT %s OFFSET %s''', (version_id, limit, offset)).fetchall()


@router.get('/amendment-runs/{run_id}/links', response_model=list[AmendmentLinkView])
def amendment_links(run_id: UUID, date_agreement: str | None = None,
                    limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)):
    """Amendment paragraphs matched to consolidated provisions. Evidence, not a finding.

    date_agreement separates a provision carrying this instrument's commencement date
    from one amended again since (SUPERSEDED) or one whose text predates it (PREDATES).
    """
    with database() as conn:
        require(conn, 'SELECT id FROM amendment_runs WHERE id=%s', run_id)
        return conn.execute(f'''SELECT l.id,l.run_id,l.locator,l.match_basis,l.commencement_date,
            l.consolidated_effective_from,l.date_agreement,l.human_review_required,
            (SELECT {CITATION} FROM regulation_sections s JOIN regulation_versions v ON v.id=s.version_id
             JOIN regulations r ON r.id=v.regulation_id JOIN regulators g ON g.id=r.regulator_id
             WHERE s.id=l.instrument_section_id) instrument,
            (SELECT {CITATION} FROM regulation_sections s JOIN regulation_versions v ON v.id=s.version_id
             JOIN regulations r ON r.id=v.regulation_id JOIN regulators g ON g.id=r.regulator_id
             WHERE s.id=l.consolidated_section_id) consolidated
            FROM amendment_links l WHERE l.run_id=%s
              AND (%s::text IS NULL OR l.date_agreement=%s)
            ORDER BY l.locator,l.id LIMIT %s OFFSET %s''',
            (run_id, date_agreement, date_agreement, limit, offset)).fetchall()


@router.get('/amendment-runs/{run_id}/gaps', response_model=list[AmendmentGapView])
def amendment_gaps(run_id: UUID, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)):
    """Amendment paragraphs that were not linked, with the reason. Never dropped."""
    with database() as conn:
        require(conn, 'SELECT id FROM amendment_runs WHERE id=%s', run_id)
        return conn.execute(f'''SELECT p.id,p.run_id,p.locator,p.reason,
            (SELECT {CITATION} FROM regulation_sections s JOIN regulation_versions v ON v.id=s.version_id
             JOIN regulations r ON r.id=v.regulation_id JOIN regulators g ON g.id=r.regulator_id
             WHERE s.id=p.instrument_section_id) instrument
            FROM amendment_link_gaps p WHERE p.run_id=%s
            ORDER BY p.reason,p.locator NULLS LAST,p.id LIMIT %s OFFSET %s''',
            (run_id, limit, offset)).fetchall()

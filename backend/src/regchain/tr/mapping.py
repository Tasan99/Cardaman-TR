"""Obligation → policy → SOP → control → evidence → gap.

Coverage is counted from what the existing engine already decides about a policy passage
(COVERS_TEXT / PARTIAL / CONFLICT / NO_EVIDENCE / UNKNOWN) plus whether a control and in-period
evidence are linked. Similar words in a policy are never enough for COVERED: that status needs
COVERS_TEXT from the engine (which already checks actor, action, object, timing, condition, scope).
"""
from typing import Literal

from pydantic import Field

from ..pilot.schema import Strict

DocumentCoverage = Literal['COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN']
MappingStatus = Literal['COVERED', 'PARTIALLY_COVERED', 'NOT_COVERED', 'CONTRADICTED', 'UNKNOWN']
DocumentType = Literal['POLICY', 'SOP', 'STANDARD', 'WORK_INSTRUCTION', 'SPECIFICATION']


class InternalDocument(Strict):
    document_id: str = Field(min_length=1)
    type: DocumentType
    owner_department: str
    version: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    entity_ids: list[str] = []
    facility_ids: list[str] = []
    product_ids: list[str] = []


class Control(Strict):
    control_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    owner_department: str
    documents: list[str] = []
    evidence_types: list[str] = []


class EvidenceItem(Strict):
    evidence_id: str = Field(min_length=1)
    control_id: str
    evidence_type: str
    period: str
    content_hash: str = Field(min_length=1)
    in_period: bool = True


class ObligationMapping(Strict):
    obligation_id: str
    decision_id: str | None = None
    departments: list[str] = []
    document_ids: list[str] = []
    document_coverage: DocumentCoverage
    control_ids: list[str] = []
    evidence_ids: list[str] = []
    status: MappingStatus
    reasons: list[str] = []


def mapping_status(document_coverage: DocumentCoverage, control_mapped: bool, evidence_present: bool) -> tuple[str, list[str]]:
    """(status, reasons). CONFLICT always wins; UNKNOWN stays UNKNOWN; similar wording is not COVERED."""
    if document_coverage == 'CONFLICT':
        return 'CONTRADICTED', ['POLICY_CONFLICT']
    if document_coverage == 'UNKNOWN':
        return 'UNKNOWN', ['COVERAGE_UNKNOWN']
    if document_coverage == 'NO_EVIDENCE':
        return 'NOT_COVERED', ['NO_POLICY_EVIDENCE']
    if document_coverage == 'PARTIAL':
        return 'PARTIALLY_COVERED', ['PARTIAL_POLICY']
    if document_coverage == 'COVERS_TEXT' and control_mapped and evidence_present:
        return 'COVERED', ['POLICY_COVERS', 'CONTROL_MAPPED', 'EVIDENCE_PRESENT']
    if document_coverage == 'COVERS_TEXT' and control_mapped:
        return 'PARTIALLY_COVERED', ['POLICY_COVERS', 'EVIDENCE_MISSING']
    if document_coverage == 'COVERS_TEXT':
        return 'PARTIALLY_COVERED', ['POLICY_COVERS', 'CONTROL_MISSING']
    return 'UNKNOWN', ['COVERAGE_UNKNOWN']


def map_obligation(obligation_id: str, document_coverage: DocumentCoverage, controls: list[Control],
                   evidence: list[EvidenceItem], departments=(), documents=(), decision_id=None) -> ObligationMapping:
    present = any(e.in_period for e in evidence)
    status, reasons = mapping_status(document_coverage, bool(controls), present)
    return ObligationMapping(obligation_id=obligation_id, decision_id=decision_id, departments=list(departments),
                             document_ids=list(documents), document_coverage=document_coverage,
                             control_ids=[c.control_id for c in controls],
                             evidence_ids=[e.evidence_id for e in evidence], status=status, reasons=reasons)

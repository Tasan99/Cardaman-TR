from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 'obligations-v3'


class SupportingEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    section_id: str
    role: Literal['condition','exception','definition','scope']
    quote: str = Field(min_length=1)


class SecondPass(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decisions: list[Literal['SUPPORTED','UNSUPPORTED','UNCERTAIN']] = Field(max_length=20)


class Candidate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_quote: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    modality: Literal['MUST','MUST_NOT','SHOULD','SHOULD_NOT','MAY','MAY_NOT']
    required_action: str | None = None
    prohibited_action: str | None = None
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    deadline: str | None = None
    affected_products: list[str] = Field(default_factory=list)
    affected_entities: list[str] = Field(default_factory=list)
    customer_types: list[str] = Field(default_factory=list)
    confidence_score: Decimal = Field(ge=0, le=1, max_digits=5, decimal_places=4)
    supporting_evidence: list[SupportingEvidence] = Field(default_factory=list,max_length=16)

    @model_validator(mode='after')
    def action_polarity(self):
        negative = self.modality.endswith('_NOT')
        if negative and (not self.prohibited_action or self.required_action is not None):
            raise ValueError('Negative modalities require prohibited_action only')
        if not negative and (not self.required_action or self.prohibited_action is not None):
            raise ValueError('Positive modalities require required_action only')
        return self


class ExtractionOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['EXTRACTED','INSUFFICIENT_EVIDENCE','NO_EXPLICIT_OBLIGATION']
    obligations: list[Candidate] = Field(default_factory=list, max_length=20)

    @model_validator(mode='after')
    def status_matches_content(self):
        if (self.status == 'EXTRACTED') != bool(self.obligations):
            raise ValueError('Status and candidate list disagree')
        return self


class ModelEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_id: str
    role: Literal['condition','exception','definition','scope']
    quote: str = Field(min_length=1)


class ModelCandidate(BaseModel):
    """Small model-facing contract. Backend supplies provenance and action polarity."""
    model_config = ConfigDict(extra='forbid')
    subject: str = Field(min_length=1)
    modality: Literal['MUST','MUST_NOT','SHOULD','SHOULD_NOT','MAY','MAY_NOT']
    action: str = Field(min_length=1)
    conditions: list[str]
    exceptions: list[str]
    evidence: list[ModelEvidence] = Field(max_length=16)


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['EXTRACTED','INSUFFICIENT_EVIDENCE','NO_EXPLICIT_OBLIGATION']
    obligations: list[ModelCandidate] = Field(max_length=20)

    @model_validator(mode='after')
    def consistent(self):
        if (self.status=='EXTRACTED') != bool(self.obligations):
            raise ValueError('EXTRACTED requires candidates; other statuses require an empty list')
        return self

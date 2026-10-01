"""What an evaluation case asserts, in a form that can be scored without a person."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

FORMAT = 'cardaman-evaluation-dataset-v1'
Applicability3 = Literal['APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN']
CoverageExpectation = Literal['COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN', 'NOT_ASSESSED', 'ANY']
GateExpectation = Literal['MATCH', 'MISMATCH', 'UNDETERMINED', 'NOT_RESTRICTED', 'ANY']
# v0.19 label provenance: where a label came from. HUMAN_REVIEWED = an expert reviewed it;
# MANUAL_LEGAL_READING = written by reading the regulation, the profile and the policy (no rule, no model);
# SYNTHETIC_CONTROLLED = the relation was planted on purpose in a synthetic policy passage;
# RULE_DERIVED = computed by a rule function (circular for measuring that rule).
LabelSource = Literal['HUMAN_REVIEWED', 'MANUAL_LEGAL_READING', 'SYNTHETIC_CONTROLLED', 'RULE_DERIVED']
LabelField = Literal['applicability', 'coverage', 'conflict', 'entity_gate']
CATEGORIES = ('parent-applies-child-not', 'entity-gate', 'false-conflict', 'irrelevant-evidence', 'unknown-over-trigger',
              'false-positive-applicability', 'false-negative-applicability', 'semantic-equivalent-policy', 'exact-policy',
              'partial-coverage', 'no-evidence', 'contradictory-policy', 'missing-policy', 'ambiguous-profile', 'control-register',
              'extraction-precision', 'extraction-recall', 'retrieval')


def without_unset_provenance(data: dict, keys) -> dict:
    """A dump without provenance fields left at their defaults, so a dataset written before v0.19
    (tr-aml-v1) dumps, and its manifest digest stays, exactly as before."""
    for key in keys:
        if key in data and data[key] in (None, {}):
            del data[key]
    return data


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ExpectedObligation(Strict):
    """One duty the expert expects the pilot to find, and what it should say about it.

    The duty is named by article and sub-paragraph, optionally narrowed by words of its action,
    so the label survives small differences in the extractor's wording. 'ANY' leaves a field
    unasserted; None leaves the conflict flag unasserted.
    """
    article: str = Field(min_length=1)
    clause: str = ''
    action_keywords: list[str] = Field(default_factory=list)
    applicability: Applicability3
    entity_gate: GateExpectation = 'ANY'
    coverage: CoverageExpectation = 'ANY'
    conflict: bool | None = None
    evidence: list[str] = Field(default_factory=list)
    required: bool = True
    notes: str = ''
    # Overall provenance of this expectation, and optionally per asserted field.
    label_source: LabelSource | None = None
    label_sources: dict[LabelField, LabelSource] = Field(default_factory=dict)

    @model_serializer(mode='wrap')
    def dump(self, handler):
        return without_unset_provenance(handler(self), ('label_source', 'label_sources'))


class EvaluationCase(Strict):
    case_id: str = Field(pattern=r'^[A-Za-z0-9._-]{2,64}$')
    title: str = Field(min_length=1)
    categories: list[str] = Field(default_factory=list)
    jurisdiction: Literal['TR', 'UK']
    # 'YONETMELIK:200713012' or 'CONC:7': the module and chapter the pilot analyses.
    regulation_id: str = Field(pattern=r'^[A-Z]{2,12}:[0-9A-Za-z]{1,12}$')
    # Retained snapshot directory, relative to the dataset file.
    regulation_fixture: str = Field(min_length=1)
    target_sections: list[str] = Field(min_length=1)
    company_profile: dict
    policy_documents: list[str] = Field(default_factory=list)
    optional_control_records: list[str] = Field(default_factory=list)
    expected_obligations: list[ExpectedObligation]
    # Keyed views of the same expectations (obligation key -> value), for readers and diffs.
    expected_applicability: dict[str, str] = Field(default_factory=dict)
    expected_policy_coverage: dict[str, str] = Field(default_factory=dict)
    expected_conflicts: dict[str, bool] = Field(default_factory=dict)
    expected_entity_gate: dict[str, str] = Field(default_factory=dict)
    expected_evidence: dict[str, list[str]] = Field(default_factory=dict)
    notes: str = ''

    @model_validator(mode='after')
    def consistent(self):
        unknown = [c for c in self.categories if c not in CATEGORIES]
        if unknown:
            raise ValueError(f'Unknown evaluation categories: {unknown}')
        if not self.policy_documents and not self.optional_control_records:
            raise ValueError('A case needs at least one policy document or control register')
        return self


class Dataset(Strict):
    format: Literal['cardaman-evaluation-dataset-v1']
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = ''
    cases: list[EvaluationCase] = Field(min_length=1)
    # Provenance of expectations that carry no label_source of their own. A dataset that states no
    # provenance at all (tr-aml-v1, whose labels build_dataset.py derived from rule functions) is
    # read as RULE_DERIVED: readers must assume the conservative case, never HUMAN_REVIEWED.
    default_label_source: LabelSource | None = None

    @model_serializer(mode='wrap')
    def dump(self, handler):
        return without_unset_provenance(handler(self), ('default_label_source',))

    def label_source_of(self, expectation: 'ExpectedObligation') -> str:
        """The expectation's own provenance, else the dataset default, else RULE_DERIVED."""
        return expectation.label_source or self.default_label_source or 'RULE_DERIVED'

    @model_validator(mode='after')
    def unique_ids(self):
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate case_id in dataset')
        return self

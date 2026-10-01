"""Expert annotation framework, deliberately separate from development gold.

No annotations are generated here. A DRAFT_FRAMEWORK cannot be scored or called
locked gold; a FROZEN release requires independent expert reviews and task quotas.
Adjudication is an explicit, hashed overlay and never mutates the strict baseline.
"""
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Task = Literal['EXTRACTION', 'APPLICABILITY', 'COVERAGE', 'CONFLICT', 'RETRIEVAL', 'GROUNDING', 'FINANCIAL_CHANGE']
TASKS = ['EXTRACTION', 'APPLICABILITY', 'COVERAGE', 'CONFLICT', 'RETRIEVAL', 'GROUNDING', 'FINANCIAL_CHANGE']
SHA = r'^[a-f0-9]{64}$'


def annotation_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class SourceSpan(Strict):
    source_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode='after')
    def span_length(self):
        if self.end - self.start != len(self.quote):
            raise ValueError('Source spans use exact Unicode codepoint offsets, end-exclusive')
        return self


class NumericAnnotation(Strict):
    value: int | float | None
    unit: str | None
    comparator: Literal['=', '>', '>=', '<', '<='] | None
    temporal_dimension: Literal['DEADLINE', 'RETENTION_PERIOD', 'FREQUENCY', 'EVENT_TIMING'] | None
    span: SourceSpan


class PolicyPassage(Strict):
    passage_id: str = Field(min_length=1)
    relevance: Literal['SUPPORT', 'CONFLICT', 'PARTIAL', 'IRRELEVANT']
    span: SourceSpan


class GoldLabels(Strict):
    # Required keys: null means unassessed, [] means assessed and absent.
    actor: list[SourceSpan] | None
    action: list[SourceSpan] | None
    object: list[SourceSpan] | None
    condition: list[SourceSpan] | None
    exception: list[SourceSpan] | None
    deadline: list[NumericAnnotation] | None
    threshold: list[NumericAnnotation] | None
    applicability: Literal['APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN'] | None
    coverage: Literal['COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN'] | None
    conflict: bool | None
    relevant_policy_passage: list[PolicyPassage] | None
    grounding: Literal['SUPPORTED', 'UNSUPPORTED', 'UNKNOWN'] | None
    change_type: list[Literal['THRESHOLD_CHANGED', 'DEADLINE_CHANGED', 'RETENTION_PERIOD_CHANGED',
                             'ENTITY_SCOPE_CHANGED', 'EXEMPTION_CHANGED', 'TEXT_CHANGED']] | None


class ExpertReview(Strict):
    reviewer_id: str = Field(min_length=1)
    expertise: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)
    independent_of_development: Literal[True]
    blind_to_model_predictions: Literal[True]
    labels_sha256: str = Field(pattern=SHA)


class Adjudication(Strict):
    status: Literal['ADJUDICATION_REQUIRED', 'ADJUDICATED']
    reason: str = Field(min_length=1)
    adjudicator_id: str | None
    adjudicated_at: str | None
    accepted_labels_sha256: str | None = Field(default=None, pattern=SHA)

    @model_validator(mode='after')
    def resolved(self):
        details = (self.adjudicator_id, self.adjudicated_at, self.accepted_labels_sha256)
        if self.status == 'ADJUDICATED' and not all(details):
            raise ValueError('Resolved adjudication requires identity, date and accepted label hash')
        if self.status == 'ADJUDICATION_REQUIRED' and any(details):
            raise ValueError('Pending adjudication cannot contain an accepted replacement label')
        return self


class ExpertExample(Strict):
    example_id: str = Field(min_length=1)
    independence_group: str = Field(min_length=1)
    tasks: list[Task] = Field(min_length=1)
    regulation: str = Field(min_length=1)
    article: str = Field(min_length=1)
    subclause: str | None
    source_span: SourceSpan
    comparison_source_span: SourceSpan | None
    company_profile_sha256: str | None = Field(pattern=SHA)
    policy_package_sha256: str | None = Field(pattern=SHA)
    development_exposure: Literal['NOT_EXPOSED', 'EXPOSED']
    labels: GoldLabels
    reviews: list[ExpertReview]
    adjudication: Adjudication | None

    @model_validator(mode='after')
    def task_fields(self):
        required = {'EXTRACTION': ('actor', 'action', 'object', 'condition', 'exception', 'deadline', 'threshold'),
                    'APPLICABILITY': ('applicability',), 'COVERAGE': ('coverage',), 'CONFLICT': ('conflict',),
                    'RETRIEVAL': ('relevant_policy_passage',), 'GROUNDING': ('grounding',), 'FINANCIAL_CHANGE': ('change_type',)}
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError('Duplicate task tags')
        for task in self.tasks:
            if any(getattr(self.labels, f) is None for f in required[task]):
                raise ValueError('Assessed task requires all its label fields: ' + task)
        if 'APPLICABILITY' in self.tasks and not self.company_profile_sha256:
            raise ValueError('Applicability labels require a frozen company profile')
        if set(self.tasks) & {'COVERAGE', 'CONFLICT', 'RETRIEVAL'} and not self.policy_package_sha256:
            raise ValueError('Policy relations require a frozen policy package')
        if 'FINANCIAL_CHANGE' in self.tasks and self.comparison_source_span is None:
            raise ValueError('Financial change labels require both frozen old and new source spans')
        return self

    def reviewed_and_unexposed(self):
        if self.development_exposure != 'NOT_EXPOSED' or len({r.reviewer_id for r in self.reviews}) < 2:
            return False
        final_hash = annotation_hash(self.labels.model_dump())
        if self.adjudication:
            return (self.adjudication.status == 'ADJUDICATED' and self.adjudication.accepted_labels_sha256 == final_hash
                    and self.adjudication.adjudicator_id not in {r.reviewer_id for r in self.reviews})
        return all(r.labels_sha256 == final_hash for r in self.reviews)


class ExpertGoldRelease(Strict):
    format: Literal['CARDAMAN_LOCKED_EXPERT_GOLD_V1']
    status: Literal['DRAFT_FRAMEWORK', 'FROZEN']
    version: str = Field(min_length=1)
    targets_per_task: dict[Task, int]
    source_sha256: dict[str, str]
    examples: list[ExpertExample]
    frozen_at: str | None

    @model_validator(mode='after')
    def release_gate(self):
        if set(self.targets_per_task) != set(TASKS) or any(v < 100 for v in self.targets_per_task.values()):
            raise ValueError('Every critical task must target at least 100 independent reviewed examples')
        if len({e.example_id for e in self.examples}) != len(self.examples):
            raise ValueError('Duplicate example id')
        if any(not re_sha(value) for value in self.source_sha256.values()):
            raise ValueError('Invalid source digest')
        if self.status == 'DRAFT_FRAMEWORK':
            if self.examples or self.frozen_at:
                raise ValueError('This framework release has zero labels and is not frozen')
            return self
        if not self.frozen_at or any(not e.reviewed_and_unexposed() for e in self.examples):
            raise ValueError('Frozen gold requires unexposed examples, two blind expert reviews and resolved disagreement')
        for task, target in self.targets_per_task.items():
            groups = {e.independence_group for e in self.examples if task in e.tasks}
            if len(groups) < target:
                raise ValueError('Insufficient independent reviewed groups for ' + task)
        for example in self.examples:
            for digest in (example.company_profile_sha256, example.policy_package_sha256):
                if digest and digest not in self.source_sha256.values():
                    raise ValueError('Company/profile and policy package hashes must be in the frozen source manifest')
            for span in all_spans(example.model_dump()):
                if self.source_sha256.get(span['source_id']) != span['source_sha256']:
                    raise ValueError('Every annotation span must reference a frozen source hash')
        return self


def re_sha(value):
    return len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def all_spans(value):
    if isinstance(value, dict):
        if {'source_id', 'source_sha256', 'start', 'end', 'quote'} <= set(value):
            yield value
        else:
            for item in value.values():
                yield from all_spans(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_spans(item)


def verify_source_texts(release: ExpertGoldRelease, sources: dict[str, str]):
    """Verify actual UTF-8 source content at release/scoring boundaries; no network."""
    if release.status != 'FROZEN':
        raise ValueError('DRAFT_FRAMEWORK is not scoreable gold')
    for source_id, wanted in release.source_sha256.items():
        text = sources[source_id]
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != wanted:
            raise ValueError('Source content changed: ' + source_id)
    for example in release.examples:
        for span in all_spans(example.model_dump()):
            if sources[span['source_id']][span['start']:span['end']] != span['quote']:
                raise ValueError('Source span mismatch: ' + example.example_id)


class AmbiguityItem(Strict):
    dataset_path: str
    dataset_sha256: str = Field(pattern=SHA)
    case_id: str
    obligation_key: str
    status: Literal['ADJUDICATION_REQUIRED']
    reason: str = Field(min_length=1)
    # No replacement label field is permitted in this pending registry.


def metric_gold_view(strict_labels: dict, *, mode: str, adjudications: list[dict] | None = None) -> dict:
    """Return a copy and provenance; strict gold is never edited or silently filtered.

    Adjudicated view requires an explicit accepted overlay bound to the old label
    hash. Pending records are not overlays and cannot change any denominator.
    """
    labels = json.loads(json.dumps(strict_labels))
    if mode == 'STRICT_GOLD_METRICS':
        return {'mode': mode, 'labels': labels, 'applied_adjudications': []}
    if mode != 'ADJUDICATED_METRICS' or not adjudications:
        raise ValueError('Adjudicated metrics require a completed expert overlay')
    applied = []
    for item in adjudications:
        required = {'key', 'original_label_sha256', 'accepted_label', 'accepted_label_sha256', 'status', 'adjudicator_id', 'adjudicated_at'}
        if set(item) != required or item['status'] != 'ADJUDICATED' or not item['adjudicator_id'] or not item['adjudicated_at']:
            raise ValueError('Unresolved or malformed adjudication overlay')
        key = item['key']
        if key not in labels or key in applied or annotation_hash(strict_labels[key]) != item['original_label_sha256']:
            raise ValueError('Adjudication must bind to one unchanged strict gold label')
        if annotation_hash(item['accepted_label']) != item['accepted_label_sha256']:
            raise ValueError('Accepted label hash mismatch')
        labels[key] = item['accepted_label']
        applied.append(key)
    return {'mode': mode, 'labels': labels, 'applied_adjudications': applied}

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Company(Strict):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    jurisdictions: list[str] | None
    activities: list[str] | None
    licences: list[str] | None
    products: list[str] | None
    customer_types: list[str] | None
    description: str | None


class Quote(Strict):
    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


# The AI's proposal may say "possibly": a stated company fact matches the scope in substance
# (payment services under "diğer finansal hizmetler") while a fact needed to be certain is
# missing. Measured on Kanun 5549 (22 September 2026): with only APPLIES / DOES_NOT_APPLY /
# UNKNOWN the judge answered UNKNOWN for every duty and its reasoning was lost. A reviewer's
# decision stays three-valued: "possibly" is not a decision.
Applicability = Literal['APPLIES', 'POSSIBLY_APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN']
ReviewApplicability = Literal['APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN']
Coverage = Literal['COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN']
QUOTE_LIMIT = 400


class ModelQuote(Quote):
    # Model output only; a reviewer may still cite a whole passage. On a real policy PDF
    # the model echoed entire 2000-character blocks as "quotes", reached the output cap
    # and every proposal was discarded as truncated. The schema bound also constrains
    # decoding; a quote cut at the limit is still an exact substring.
    quote: str = Field(min_length=1, max_length=QUOTE_LIMIT)


class PolicyCheck(ModelQuote):
    relation: Literal['SUPPORTS', 'PARTIAL', 'CONFLICTS', 'UNRELATED', 'UNCLEAR']


class Basis(Strict):
    """One applicability check: a company fact against a regulatory condition.

    The fact is copied from the profile and the condition from the scope text or the
    provision; the backend verifies both copies, so the pairing is auditable evidence of
    what the judge compared, not a free-text explanation.
    """
    company_fact: str = Field(min_length=1, max_length=300)
    regulatory_condition: str = Field(min_length=1, max_length=QUOTE_LIMIT)
    match: Literal['YES', 'NO', 'UNCLEAR']


class ResolvedBasis(Basis):
    source_id: str = Field(min_length=1)
    # Which profile field the verified company fact came from (activities, licences, ...), so an
    # APPLIES can be read as "provision X ↔ company field Y" without guessing.
    company_fact_key: str = ''
    # For a NO pair (v0.16.1): whether the regulatory condition is an explicit exclusion
    # ("uygulanmaz", "hariç", "except", "unless") — only such a NO vetoes an APPLIES; a NO on a
    # non-matching list item ("a) Bankalar" against a payment institution) is recorded, not a veto.
    exclusionary: bool | None = None
    note: str = ''


class RequiredEntity(Strict):
    role: Literal['obliged_party', 'counterparty']
    type: str
    text: str
    source: Literal['subject', 'clause', 'heading']
    match: Literal['MATCH', 'MISMATCH', 'UNDETERMINED']


class EntityScope(Strict):
    """The entity gate's reading of one clause (v0.16.1): parent provision, child clause, the
    entity the clause requires, the company's stated entity, the match and the final state."""
    parent_provision: str
    child_clause: str
    child_offset: int
    child_excerpt: str
    required_entities: list[RequiredEntity]
    company_entity_types: list[str]
    company_customer_families: list[str]
    company_foreign_hq: bool
    match: Literal['MATCH', 'MISMATCH', 'UNDETERMINED', 'NOT_RESTRICTED']
    provision_state: Applicability
    # False when the gate settled the clause and the provision-level model judgement was not requested.
    provision_assessed: bool = True
    final: Applicability
    rule: Literal['ENTITY_GATE', 'PROVISION_LEVEL']
    reason: str


# Why a passage was kept from the judge by rule (v0.16): retrieval still ranks a page number, a
# torn line or a heading that shares a word with the duty, and one such crumb judged UNCLEAR
# used to block a favourable verdict on its own.
class FilteredPassage(Strict):
    source_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


# What happened to the drafted wording of a proposal; the rule-based part never depends on it.
DraftStatus = Literal['DRAFTED', 'DRAFT_UNAVAILABLE']
RemediationStatus = Literal['PROPOSED', 'NOT_NEEDED', 'NOT_APPLICABLE', 'UNDETERMINED', 'NOT_ASSESSED', 'DISABLED']
ControlCoverage = Literal['SUPPORTS', 'PARTIAL', 'CONFLICT', 'NONE', 'NOT_READ']


class Remediation(Strict):
    """An AI-generated proposal for a gap; never a decision, always awaiting a person."""
    label: Literal['AI-GENERATED PROPOSAL'] = 'AI-GENERATED PROPOSAL'
    type: Literal['NEW_POLICY_CLAUSE', 'POLICY_UPDATE', 'CONFLICT_RESOLUTION']
    trigger: Coverage
    priority: Literal['HIGH', 'MEDIUM', 'LOW']
    priority_inputs: list[str]
    recommended_action: str = Field(min_length=1)
    missing_requirements: list[str]
    suggested_policy_language: str
    implementation_notes: str
    status: Literal['PROPOSED'] = 'PROPOSED'
    requires_approval: Literal[True] = True
    draft_status: DraftStatus = 'DRAFTED'
    # TIMEOUT, MALFORMED_JSON, EMPTY_RESPONSE, CONTEXT_LENGTH, MODEL_ERROR or PARSING_ERROR when the draft failed.
    draft_failure: str = ''
    draft_attempts: int = 0


class Proposal(Strict):
    applicability: Applicability
    company_fact_keys: list[str]
    scope_evidence: list[ModelQuote]
    applicability_reason: str = Field(min_length=1)
    coverage: Coverage
    policy_evidence: list[ModelQuote]
    policy_checks: list[PolicyCheck]
    coverage_reason: str = Field(min_length=1)
    missing_information: list[str]
    basis: list[ResolvedBasis] = Field(default_factory=list)
    # How the stored applicability was arrived at: the model's own answer, a rule that turned an
    # UNKNOWN with a matching basis into POSSIBLY_APPLIES, or a rule that recorded UNKNOWN because
    # the model's state disagreed with the basis it gave (v0.16, instead of "manual analysis required").
    # v0.18 rule-first gates (pilot/applicability.py): the company is not on the regulation's obliged-party
    # list, the duty is addressed to someone else, the jurisdiction or an exemption rules it out, the
    # profile does not say, a clear structural match decided APPLIES, or a model DOES_NOT_APPLY that
    # rested only on an ambiguous profile was recorded UNKNOWN.
    applicability_rule: Literal['MODEL', 'UPGRADED_FROM_UNKNOWN_BY_BASIS', 'DOWNGRADED_BASIS_INCONSISTENT', 'ENTITY_GATE', 'NOT_ASSESSED',
                                'SUBJECT_SCOPE_GATE', 'ADDRESSEE_GATE', 'JURISDICTION_GATE', 'EXEMPTION_GATE', 'PROFILE_INCOMPLETE',
                                'RULE_CLEAR_MATCH', 'PROFILE_AMBIGUOUS', 'SCOPE_INCOMPLETE', 'SOURCE_INCOMPLETE'] = 'NOT_ASSESSED'
    remediation: Remediation | None = None
    # v0.16: operational evidence from a control register, counted apart from written policy.
    control_coverage: ControlCoverage = 'NOT_READ'
    remediation_status: RemediationStatus = 'NOT_ASSESSED'
    filtered_passages: list[FilteredPassage] = Field(default_factory=list)
    # v0.16.1: the clause-level entity reading behind the applicability, and whether policy
    # coverage was assessed at all (it is not for a clause the entity gate rules out).
    applicability_scope: EntityScope | None = None
    coverage_assessed: bool = True
    # v0.17: the applicability decision laid out (rule, model, aggregate), the provenance of
    # every cited quote, and flags that route the row to a person whatever the mode.
    trace: dict | None = None
    provenance: list[dict] = Field(default_factory=list)
    review_flags: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def conflict_needs_its_quote(self):
        # A CONFLICT without the contradicting sentence is not a finding (v0.17 legal evidence rule).
        if self.coverage == 'CONFLICT' and not any(c.relation == 'CONFLICTS' and c.quote.strip() for c in self.policy_checks):
            raise ValueError('CONFLICT requires at least one policy passage judged CONFLICTS with its explicit contradiction quote')
        return self


class ScopeJudgement(Strict):
    """Applicability on its own. One small answer is validated far more reliably than one
    document that also had to quote and classify every policy passage."""
    applicability: Applicability
    company_fact_keys: list[str]
    scope_evidence: list[ModelQuote]
    basis: list[Basis] = Field(max_length=12)
    applicability_reason: str = Field(min_length=1)
    missing_information: list[str]


class PassageJudgement(Strict):
    """One passage against one duty, composed in code from the two narrow answers below."""
    relation: Literal['SUPPORTS', 'PARTIAL', 'CONFLICTS', 'UNRELATED', 'UNCLEAR']
    quote: str = Field(max_length=QUOTE_LIMIT)
    reason: str = Field(min_length=1)


class Screen(Strict):
    """Does this one passage contradict this one duty? The question that was measured."""
    contradicts: Literal['YES', 'NO', 'UNCLEAR']
    quote: str
    reason: str


class Support(Strict):
    """Does this one passage require what the duty requires? Asked only after Screen said NO."""
    supports: Literal['SUPPORTS', 'PARTIAL', 'UNRELATED', 'UNCLEAR']
    quote: str
    reason: str


class ConflictConfirmation(Strict):
    """A second look at one alleged contradiction (v0.16). Measured live: a sentence that supported
    a reporting duty was called a contradiction; a CONFLICT now needs the quoted sentence AND this
    confirmation that it instructs the opposite, removes the duty or sets a weaker limit."""
    verdict: Literal['CONTRADICTS', 'NOT_A_CONTRADICTION', 'UNCLEAR']
    duty_requires: str
    passage_instructs: str
    reason: str


# v0.19 PARTIAL confirmation (25 September 2026): the fast reading only proposes; its labels say so. The short-
# round names SUPPORTS and PARTIAL are still read (stored packets, scripted test answers) as POSSIBLE_SUPPORT and
# POSSIBLE_PARTIAL; the schema the engine sends to the model offers only FAST_LABELS.
FAST_LABELS = ('IRRELEVANT', 'POSSIBLE_SUPPORT', 'POSSIBLE_PARTIAL', 'POSSIBLE_CONFLICT')
LEGACY_FAST_LABELS = {'SUPPORTS': 'POSSIBLE_SUPPORT', 'PARTIAL': 'POSSIBLE_PARTIAL'}


class FastJudgement(Strict):
    """v0.19 fast reading of one passage (no thinking): sorts it before the thinking verifier is asked.

    covered_elements / missing_elements name the duty's element ids (duty_payload 'elements': 'action',
    'subject', 'object', 'deadline_1', 'threshold_1', 'item_1', 'condition_1' ...); the engine constrains
    them per duty. A legacy label (SUPPORTS, PARTIAL) is read as its FAST_LABELS name.
    """
    label: Literal['IRRELEVANT', 'POSSIBLE_SUPPORT', 'POSSIBLE_PARTIAL', 'POSSIBLE_CONFLICT', 'SUPPORTS', 'PARTIAL']
    quote: str
    covered_elements: list[str]
    missing_elements: list[str]
    reason: str

    @model_validator(mode='after')
    def current_label(self):
        self.label = LEGACY_FAST_LABELS.get(self.label, self.label)
        return self


# STRICTER_THAN_REQUIRED (v0.19 short round 3): the passage states the duty's deadline, period or amount the
# same or stricter. Never a conflict: the engine records it when it rejects a DEADLINE_MISMATCH or
# THRESHOLD_MISMATCH claim resting on such a number, and a verifier answer naming it is read as no conflict.
ContradictionType = Literal['DIRECT_OPPOSITE', 'REQUIREMENT_REMOVED', 'DEADLINE_MISMATCH', 'THRESHOLD_MISMATCH', 'SCOPE_NARROWED',
                            'EXEMPTION_ADDED', 'PROHIBITED_ACTION_ALLOWED', 'REQUIRED_ACTION_FORBIDDEN', 'STRICTER_THAN_REQUIRED', 'NONE']


class ConflictVerdict(Strict):
    """v0.19 thinking verifier: one reading of every sentence of one passage against one duty.

    A conflict stands only with a contradiction type and an exact contradicting span (checked in
    code); otherwise the relation the verifier gives for the no-conflict case is used.
    """
    conflict: bool
    contradiction_type: ContradictionType
    contradiction_span: str
    regulation_requirement: str
    policy_statement: str
    confidence: Literal['HIGH', 'MEDIUM', 'LOW']
    relation_if_no_conflict: Literal['SUPPORTS', 'PARTIAL', 'UNRELATED', 'NOT_APPLICABLE']
    support_quote: str
    rationale: str
    # v0.19 (C12 smoke run, md. 5(2)): a verifier SUPPORTS carried no element list, so the unstated
    # "before" timing could not make it PARTIAL. The engine asks for both lists (required in the schema
    # it sends, constrained to the duty's ids); an answer without them still parses.
    covered_elements: list[str] = []
    missing_elements: list[str] = []


class Decision(Strict):
    obligation_id: str
    extraction: Literal['ACCEPT', 'REJECT', 'UNCERTAIN']
    applicability: ReviewApplicability
    coverage: Coverage
    company_fact_keys: list[str]
    scope_evidence: list[Quote]
    policy_evidence: list[Quote]
    rationale: str = Field(min_length=1)
    # The AI's remediation proposal is approved, rejected or left open by the reviewer.
    remediation: Literal['ACCEPT', 'REJECT', 'UNCERTAIN'] = 'UNCERTAIN'
    # v0.17 review actions: what the reviewer did with the AI proposal. APPROVE keeps it as
    # it was, REJECT sets it aside, OVERRIDE replaces a verdict and must say why, and
    # NEEDS_EVIDENCE parks the duty until more is known. Reviews saved before v0.17 carry none.
    action: Literal['APPROVE', 'REJECT', 'OVERRIDE', 'NEEDS_EVIDENCE'] | None = None
    override_reason: str = ''

    @model_validator(mode='after')
    def override_is_explained(self):
        if self.action == 'OVERRIDE' and not self.override_reason.strip():
            raise ValueError('OVERRIDE requires a nonblank override_reason')
        if self.action != 'OVERRIDE' and self.override_reason.strip():
            raise ValueError('override_reason is only recorded with an OVERRIDE action')
        return self


class Review(Strict):
    format: Literal['regchain-pilot-review-v1']
    analysis_head: str = Field(pattern=r'^[0-9a-f]{64}$')
    reviewer: str = Field(min_length=1)
    reviewer_role: str = Field(min_length=1)
    decisions: list[Decision] = Field(min_length=1)

    @model_validator(mode='after')
    def unique(self):
        if not self.reviewer.strip() or not self.reviewer_role.strip():
            raise ValueError('Reviewer name and role must be nonblank')
        ids = [d.obligation_id for d in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate obligation review')
        if any(not d.rationale.strip() for d in self.decisions):
            raise ValueError('Review rationale must be nonblank')
        return self

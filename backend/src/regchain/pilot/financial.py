"""Versioned, grounded financial-change structure and impact contracts (v0.20).

This module makes no model call. Quantity extraction precedes classification; a
retention period is never a monetary threshold. The old five-class rule remains
available explicitly for frozen v0.18 fixture comparison, not as six-class gold.
"""
from collections import Counter
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from regchain.extraction.quantities import CURRENCY, DIGIT_SCALE, NUMBER, parse_quantities, period_phrases
from .entities import COUNTERPARTIES, OBLIGED, found
from .impact import EXCLUSION_WORDS, change_kinds, normalized

ChangeType = Literal['THRESHOLD_CHANGED', 'DEADLINE_CHANGED', 'RETENTION_PERIOD_CHANGED',
                     'ENTITY_SCOPE_CHANGED', 'EXEMPTION_CHANGED', 'TEXT_CHANGED']
CHANGE_TYPES = ['THRESHOLD_CHANGED', 'DEADLINE_CHANGED', 'RETENTION_PERIOD_CHANGED',
                'ENTITY_SCOPE_CHANGED', 'EXEMPTION_CHANGED', 'TEXT_CHANGED']
FinancialImpactType = Literal['CAPITAL_REQUIREMENT', 'LIQUIDITY_REQUIREMENT', 'RESERVE_REQUIREMENT',
                              'TRANSACTION_LIMIT', 'COMPLIANCE_COST', 'FEE_OR_REVENUE', 'PENALTY_EXPOSURE', 'OTHER']
OperationalImpactType = Literal['REPORTING', 'RECORD_RETENTION', 'CUSTOMER_DUE_DILIGENCE',
                                'DISCLOSURE', 'LICENSING', 'CONTROL_UPDATE', 'PROCESS_UPDATE', 'OTHER']
TemporalDimension = Literal['DEADLINE', 'RETENTION_PERIOD', 'FREQUENCY', 'DURATION_UNSPECIFIED', 'EVENT_TIMING']
NumericDimension = Literal['MONETARY_AMOUNT', 'RATIO']


def _literal_null(value):
    if isinstance(value, str):
        return value.strip().lower() == 'null'
    if isinstance(value, dict):
        return any(_literal_null(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_literal_null(v) for v in value)
    return False


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

    @model_validator(mode='before')
    @classmethod
    def json_null_only(cls, value):
        if _literal_null(value):
            raise ValueError('Literal "null" is forbidden; use JSON null')
        return value


# Entity heads, not the transactions, records or amounts acted upon. Unknown names
# remain null in the compatibility adapter; this is deliberately not an NER model.
ENTITY = re.compile(r'\b(?:firms?|banks?|companies|company|institutions?|insurers?|lenders?|auditors?|'
                    r'customers?|consumers?|individuals?|persons?|authorities|authority|providers?|'
                    r'corporations?|associations?|foundations?)\b|'
                    r'(?<!\w)(?:yükümlü\w*|banka\w*|kuruluş\w*|kurum\w*|şirket\w*|müşteri\w*|'
                    r'tüketici\w*|gerçek kişi\w*|tüzel kişi\w*|dernek\w*|vakıf\w*)(?!\w)', re.I)


def entity_name(text):
    """An entity head at the end of the noun phrase; 'bank transactions' is an object."""
    name = text.rstrip(' .,:;')
    return any(match.end() == len(name) for match in ENTITY.finditer(name))


class AffectedEntity(Strict):
    name: str = Field(min_length=1)
    entity_type: Literal['ORGANIZATION', 'NATURAL_PERSON', 'PUBLIC_BODY', 'OBLIGED_PARTY', 'CUSTOMER']
    source_quote: str = Field(min_length=1)

    @model_validator(mode='after')
    def entity_head(self):
        if self.name not in self.source_quote or not entity_name(self.name):
            raise ValueError('affected_actor must name an evidenced entity, not an action or transaction object')
        return self


class FinancialImpact(Strict):
    financial_impact_type: FinancialImpactType | None
    operational_impact_type: OperationalImpactType | None
    affected_actor: AffectedEntity | None
    specific_monetary_impact: float | int | None
    impact_confidence: float = Field(ge=0, le=1)
    impact_basis: Literal['FACT', 'INFERENCE']
    # Preserve qualitative wording without silently pretending it was an enum.
    financial_impact_category: str | None = None
    operational_impact_category: str | None = None

    @model_validator(mode='after')
    def grounded(self, info: ValidationInfo):
        context = info.context or {}
        if self.affected_actor:
            sources = [context.get('old_text', ''), context.get('new_text', '')]
            if not any(self.affected_actor.source_quote in text for text in sources):
                raise ValueError('affected_actor source_quote is not present in the supplied regulation texts')
        if self.specific_monetary_impact is not None:
            company = context.get('company_data') or {}
            supplied = company.get('specific_monetary_impact') if isinstance(company, dict) else None
            # Merely supplying a company name does not license a made-up estimate.
            if type(supplied) not in (int, float) or supplied != self.specific_monetary_impact:
                raise ValueError('Numeric impact requires an explicit matching company monetary-impact fact')
            if self.impact_basis != 'FACT':
                raise ValueError('A directly supplied monetary-impact fact must be marked FACT')
        return self


class QuantityFact(Strict):
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    value: int | float | None
    unit: str | None
    comparator: str | None = None
    temporal_dimension: TemporalDimension | None
    numeric_dimension: NumericDimension | None
    source_clause: str


class DimensionChange(Strict):
    value_old: int | float | None
    value_new: int | float | None
    unit: str | None
    unit_old: str | None
    unit_new: str | None
    temporal_dimension: TemporalDimension | None
    numeric_dimension: NumericDimension | None
    scope_dimension: Literal['ENTITY_SCOPE', 'EXEMPTION'] | None = None
    old_fact: QuantityFact | None
    new_fact: QuantityFact | None
    pairing: Literal['SINGLE_DIMENSION_PAIR', 'ORDINAL_WITHIN_DIMENSION_REVIEW_REQUIRED']


class StructuredChange(Strict):
    schema_version: Literal['cardaman-financial-change-v020'] = 'cardaman-financial-change-v020'
    old_requirement: str
    new_requirement: str
    change_types: list[ChangeType]
    quantities_old: list[QuantityFact]
    quantities_new: list[QuantityFact]
    dimensions: list[DimensionChange]
    scope_dimension: dict[str, list[str]]
    legacy_comparison_version: Literal['v018-five-class-rule'] = 'v018-five-class-rule'
    legacy_change_types: list[str]
    limitations: list[str]


class FinancialChangeProposal(FinancialImpact):
    """Strict model output contract; context supplies source texts and company facts."""
    old_requirement: str
    new_requirement: str
    change_types: list[ChangeType]

    @model_validator(mode='after')
    def literal_requirements(self, info: ValidationInfo):
        context = info.context or {}
        for field, source in (('old_requirement', 'old_text'), ('new_requirement', 'new_text')):
            if source not in context or getattr(self, field) != context[source]:
                raise ValueError(field + ' must preserve the supplied text byte-for-byte')
        if len(set(self.change_types)) != len(self.change_types):
            raise ValueError('Duplicate change types are forbidden')
        return self


FINANCIAL_IMPACT_SCHEMA = FinancialImpact.model_json_schema()
FINANCIAL_CHANGE_SCHEMA = FinancialChangeProposal.model_json_schema()

RETENTION = re.compile(r'\b(?:retain\w*|keep|kept|store\w*|preserv\w*|retention)\b|'
                       r'(?<!\w)(?:sakla\w*|muhafaza\w*|arşiv\w*)(?!\w)', re.I)
FREQUENCY = re.compile(r'\b(?:every|each)\b|\b(?:ayda|yılda|haftada|günde)\s+bir\b', re.I)
DEADLINE = re.compile(r'\b(?:within|no later than|not later than|respond|report|notify|submit|send|deliver)\b|'
                      r'(?<!\w)(?:içinde|içerisinde|zarfında|en geç|bildir\w*|gönder\w*|verilir|yapılır)(?!\w)', re.I)


def _clause(text, start, end):
    # Split at sentence/semicolon boundaries, not decimal or thousands separators.
    boundaries = list(re.finditer(r'[;\n]|[.!?](?=\s|$)', text))
    left = max((m.end() for m in boundaries if m.end() <= start), default=0)
    right = min((m.start() for m in boundaries if m.start() >= end), default=len(text))
    return text[left:right].strip()


def _financial_quantities(text):
    """Accept postfix currency symbols while retaining exact original source spans.

    Reuse the shared numeric/comparator grammar through a local textual-unit
    adapter. The shared extractor and its frozen callers remain unchanged.
    """
    replacements = {}
    for number in NUMBER.finditer(text):
        end = number.end()
        scale = DIGIT_SCALE.match(text, end) if number.group('digits') else None
        if scale:
            end = scale.end()
        symbol = re.compile(r'\s*(US\$|[£€$₺])').match(text, end)
        if symbol:
            replacements[symbol.start(1)] = (symbol.end(1), ' ' + CURRENCY[symbol.group(1)])
    if not replacements:
        return parse_quantities(text)
    adapted, offsets, index = [], [], 0
    while index < len(text):
        if index in replacements:
            end, unit = replacements[index]
            adapted.extend(unit)
            offsets.extend([index] * (len(unit) - 1) + [end - 1])
            index = end
        else:
            adapted.append(text[index])
            offsets.append(index)
            index += 1
    result = parse_quantities(''.join(adapted))
    for quantity in result:
        quantity['start'] = offsets[quantity['start']]
        quantity['end'] = offsets[quantity['end'] - 1] + 1
        quantity['text'] = text[quantity['start']:quantity['end']]
    return result


def quantity_facts(text: str) -> list[QuantityFact]:
    facts = []
    for q in _financial_quantities(text):
        clause = _clause(text, q['start'], q['end'])
        temporal = numeric = None
        if q['unit_class'] == 'duration':
            # "retain within 5 days" is a deadline to act, not a keeping period.
            if q.get('lead') in ('within', 'no later than', 'not later than', 'en geç') or q.get('trail') in ('içinde', 'içerisinde', 'zarfında'):
                temporal = 'DEADLINE'
            elif FREQUENCY.search(clause):
                temporal = 'FREQUENCY'
            elif RETENTION.search(clause):
                temporal = 'RETENTION_PERIOD'
            elif DEADLINE.search(clause):
                temporal = 'DEADLINE'
            else:
                temporal = 'DURATION_UNSPECIFIED'
        else:
            numeric = 'RATIO' if q['unit_class'] == 'percent' else 'MONETARY_AMOUNT'
        facts.append(QuantityFact(text=q['text'], start=q['start'], end=q['end'], value=q['amount'], unit=q['unit'],
                                  comparator=q.get('comparator'), temporal_dimension=temporal,
                                  numeric_dimension=numeric, source_clause=clause))
    # Non-numeric event timings are retained as exact source phrases, not invented dates.
    for q in period_phrases(text):
        if not any(f.start < q['end'] and q['start'] < f.end for f in facts):
            facts.append(QuantityFact(text=q['text'], start=q['start'], end=q['end'], value=None, unit=None,
                                      temporal_dimension='EVENT_TIMING', numeric_dimension=None,
                                      source_clause=_clause(text, q['start'], q['end'])))
    return sorted(facts, key=lambda f: f.start)


def _signature(fact):
    return (fact.value, fact.unit, fact.comparator,
            normalized(fact.text).lower() if fact.temporal_dimension == 'EVENT_TIMING' else None)


def analyze_financial_change(old_text: str, new_text: str) -> StructuredChange:
    """Extract values and dimensions first, then classify. No company-impact estimate."""
    old, new = quantity_facts(old_text), quantity_facts(new_text)
    entities = lambda text: sorted({kind for kind, _ in found(COUNTERPARTIES, text)} |
                                   {kind for kind, _ in found(OBLIGED, text) if kind != 'GENERIC'})
    exemptions = lambda text: sorted(m.group().lower() for m in EXCLUSION_WORDS.finditer(text))
    scope = {'entities_old': entities(old_text), 'entities_new': entities(new_text),
             'exemptions_old': exemptions(old_text), 'exemptions_new': exemptions(new_text)}
    kinds, dimensions, limits = set(), [], []
    if normalized(old_text) != normalized(new_text):
        groups = sorted({(f.temporal_dimension or '', f.numeric_dimension or '') for f in old + new})
        for temporal, numeric in groups:
            before = [f for f in old if (f.temporal_dimension or '', f.numeric_dimension or '') == (temporal, numeric)]
            after = [f for f in new if (f.temporal_dimension or '', f.numeric_dimension or '') == (temporal, numeric)]
            # Equivalent number words/digits and whitespace do not invent quantity changes.
            if Counter(map(_signature, before)) == Counter(map(_signature, after)):
                continue
            if numeric:
                kinds.add('THRESHOLD_CHANGED')
            elif temporal == 'RETENTION_PERIOD':
                kinds.add('RETENTION_PERIOD_CHANGED')
            elif temporal in ('DEADLINE', 'EVENT_TIMING', 'FREQUENCY'):
                kinds.add('DEADLINE_CHANGED')
            else:
                limits.append('DURATION_ROLE_UNRESOLVED: no deadline/retention label inferred')
            ambiguous = max(len(before), len(after)) > 1
            if ambiguous:
                limits.append('MULTIPLE_QUANTITIES: ordinal pairing does not establish semantic alignment')
            for index in range(max(len(before), len(after))):
                a, b = (before[index] if index < len(before) else None), (after[index] if index < len(after) else None)
                if a and b and _signature(a) == _signature(b):
                    continue
                dimensions.append(DimensionChange(value_old=a.value if a else None, value_new=b.value if b else None,
                    unit=a.unit if a and b and a.unit == b.unit else None, unit_old=a.unit if a else None,
                    unit_new=b.unit if b else None, temporal_dimension=temporal or None, numeric_dimension=numeric or None,
                    old_fact=a, new_fact=b, pairing='ORDINAL_WITHIN_DIMENSION_REVIEW_REQUIRED' if ambiguous else 'SINGLE_DIMENSION_PAIR'))
        if scope['entities_old'] != scope['entities_new']:
            kinds.add('ENTITY_SCOPE_CHANGED')
        if scope['exemptions_old'] != scope['exemptions_new']:
            kinds.add('EXEMPTION_CHANGED')
        if not kinds:
            kinds.add('TEXT_CHANGED')
    return StructuredChange(old_requirement=old_text, new_requirement=new_text,
        change_types=[k for k in CHANGE_TYPES if k in kinds], quantities_old=old, quantities_new=new, dimensions=dimensions,
        scope_dimension=scope, legacy_change_types=change_kinds(old_text, new_text), limitations=sorted(set(limits)))


def project_change_types_to_v018(change_types: list[str]) -> list[str]:
    """Taxonomy-only projection. It does NOT relabel or guarantee agreement with old rules."""
    mapped = {'DEADLINE_CHANGED' if k == 'RETENTION_PERIOD_CHANGED' else k for k in change_types}
    if mapped - set(CHANGE_TYPES):
        raise ValueError('Unknown change type')
    return [k for k in CHANGE_TYPES if k != 'RETENTION_PERIOD_CHANGED' and k in mapped]


FINANCIAL_ALIASES = {'capital requirement': 'CAPITAL_REQUIREMENT', 'capital adequacy': 'CAPITAL_REQUIREMENT',
    'liquidity requirement': 'LIQUIDITY_REQUIREMENT', 'reserve requirement': 'RESERVE_REQUIREMENT',
    'transaction limit': 'TRANSACTION_LIMIT', 'compliance cost': 'COMPLIANCE_COST',
    'fee or revenue': 'FEE_OR_REVENUE', 'penalty exposure': 'PENALTY_EXPOSURE', 'other': 'OTHER'}
OPERATIONAL_ALIASES = {'reporting': 'REPORTING', 'record retention': 'RECORD_RETENTION', 'retention': 'RECORD_RETENTION',
    'customer due diligence': 'CUSTOMER_DUE_DILIGENCE', 'disclosure': 'DISCLOSURE', 'licensing': 'LICENSING',
    'control update': 'CONTROL_UPDATE', 'process update': 'PROCESS_UPDATE', 'other': 'OTHER'}


def normalize_financial_impact(data: dict, *, old_text: str, new_text: str, company_data: dict | None = None) -> dict:
    """Explicit legacy adapter. Returns impact plus audit codes; unknown categories stay null.

    It never parses a numeric string into company impact or turns a transaction
    object into an entity. Strict validation is available without this adapter.
    """
    notes = []
    def nullable(value, field):
        if isinstance(value, str) and value.strip().lower() in ('null', 'none', 'n/a', 'not specified', 'unknown', ''):
            notes.append(field + ': NULL_SENTINEL_NORMALIZED')
            return None
        return value
    out = {key: nullable(data.get(key), key) for key in ('financial_impact_type', 'operational_impact_type',
        'financial_impact_category', 'operational_impact_category', 'affected_actor', 'specific_monetary_impact')}
    for prefix, aliases in (('financial', FINANCIAL_ALIASES), ('operational', OPERATIONAL_ALIASES)):
        field, category = prefix + '_impact_type', prefix + '_impact_category'
        value = out[field] or out[category]
        if value is not None:
            key = str(value).strip().lower().replace('_', ' ')
            out[field] = aliases.get(key)
            if out[field] is None:
                notes.append(field + ': UNMAPPED_CATEGORY_PRESERVED')
    actor = out['affected_actor']
    if isinstance(actor, str):
        if entity_name(actor) and any(actor in text for text in (old_text, new_text)):
            actor = {'name': actor, 'entity_type': 'OBLIGED_PARTY', 'source_quote': actor}
            notes.append('affected_actor: LEGACY_ENTITY_TEXT_STRUCTURED')
        else:
            actor = None
            notes.append('affected_actor: UNGROUNDED_OR_NON_ENTITY_REJECTED')
    if actor is not None:
        try:
            entity = AffectedEntity.model_validate(actor)
            if not any(entity.source_quote in text for text in (old_text, new_text)):
                raise ValueError('ungrounded entity')
            actor = entity.model_dump()
        except (ValueError, TypeError):
            actor = None
            notes.append('affected_actor: UNGROUNDED_OR_NON_ENTITY_REJECTED')
    out['affected_actor'] = actor
    monetary = out['specific_monetary_impact']
    supplied = (company_data or {}).get('specific_monetary_impact')
    if monetary is not None and (type(monetary) not in (int, float) or type(supplied) not in (int, float) or monetary != supplied):
        out['specific_monetary_impact'] = None
        notes.append('specific_monetary_impact: UNSUPPORTED_COMPANY_IMPACT_REJECTED')
    out['impact_confidence'] = data.get('impact_confidence', 0.0)
    out['impact_basis'] = data.get('impact_basis', 'INFERENCE')
    impact = FinancialImpact.model_validate(out, context={'old_text': old_text, 'new_text': new_text, 'company_data': company_data})
    return {'schema_version': 'cardaman-financial-impact-v020', 'impact': impact.model_dump(), 'normalizations': notes}

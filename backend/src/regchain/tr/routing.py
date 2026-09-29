"""Pack selection and applicability routing for an enterprise profile.

select_packs() decides which sector packs a group falls under from what it handles: a product of
one of the pack's product classes, or a licence only that pack uses. evaluate_scope() reads one
obligation scope against every target of the scope's level (legal entity, facility, product or
activity) and answers APPLIES, PARTIAL, DOES_NOT_APPLY or UNKNOWN with reason codes. rollup()
carries a lower-level answer up to the legal entities and the group without spreading it: a
brewery duty applies to the entity that runs the brewery, and the group only PARTIAL.

Each dimension is a fact: YES when the target states a required class, NO only when the target's
list is declared complete, otherwise UNKNOWN (PROFILE_INCOMPLETE). A licence counts only when HELD.
The answer rests on pack metadata (basis PACK_METADATA, provision UNRESOLVED) until a provision is
extracted and grounded; guidance is routed like binding text but never creates an obligation.
"""
from typing import Literal

from pydantic import Field

from ..pilot.applicability import gate
from ..pilot.schema import Strict
from .core import ObligationScope
from .packs import Registry
from .profile import EnterpriseProfile, Fact

Status = Literal['APPLIES', 'PARTIAL', 'DOES_NOT_APPLY', 'UNKNOWN']
Selection = Literal['SELECTED', 'NOT_SELECTED', 'UNKNOWN']

GATE_OF = {'entity_classes': 'ENTITY_CLASS', 'activity_classes': 'ACTIVITY_CLASS', 'facility_classes': 'FACILITY_CLASS',
           'license_classes': 'LICENSE_CLASS', 'product_classes': 'PRODUCT_CLASS',
           'product_attributes': 'PRODUCT_ATTRIBUTE', 'alcohol_scope': 'ALCOHOL_SCOPE'}
CODE_OF = {'entity_classes': 'ENTITY', 'activity_classes': 'ACTIVITY', 'facility_classes': 'FACILITY',
           'license_classes': 'LICENSE', 'product_classes': 'PRODUCT', 'product_attributes': 'PRODUCT_ATTRIBUTE'}
GATE_STATUS = {'YES': 'MATCH', 'NO': 'MISMATCH', 'UNKNOWN': 'UNDETERMINED'}
PRODUCT_DIMENSIONS = ('product_classes', 'product_attributes')


class PackSelection(Strict):
    pack_id: str
    status: Selection
    reason_codes: list[str]
    product_ids: list[str] = []
    license_classes: list[str] = []


class Decision(Strict):
    scope_id: str
    regulation_id: str
    packs: list[str] = []
    level: Literal['GROUP', 'LEGAL_ENTITY', 'FACILITY', 'PRODUCT', 'ACTIVITY']
    target_id: str
    entity_id: str | None = None
    status: Status
    reason_codes: list[str] = Field(min_length=1)
    applies_to_products: list[str] = []
    gates: list[dict] = []
    children: dict[str, Status] = {}
    binding_status: str
    creates_obligation: bool
    basis: Literal['PACK_METADATA', 'EXTRACTED']
    provision_ref: str | None = None
    provision_status: Literal['RESOLVED', 'UNRESOLVED']


class Resolution(Strict):
    profile_id: str
    selections: list[PackSelection]
    decisions: list[Decision]
    rollups: list[Decision]
    out_of_scope: list[Decision] = []


# -- pack selection --------------------------------------------------------------------------------
def _exclusive_licences(registry: Registry, pack_id: str) -> set[str]:
    others = set()
    for pack in registry.selectable_packs():
        if pack.pack_id != pack_id:
            others.update(pack.license_classes)
    return set(registry.pack(pack_id).license_classes) - others


def select_packs(profile: EnterpriseProfile, registry: Registry) -> list[PackSelection]:
    """One selection per active sector pack. A group may fall under several (a brewer with an
    alcohol-free line); modules are never selected on their own, they come with a pack."""
    held = {l.license_class for e in profile.legal_entities for l in e.licenses if l.status == 'HELD'}
    out = []
    for pack in registry.selectable_packs():
        products = [p.product_id for p in profile.products if p.product_class in pack.product_classes]
        licences = sorted(held & _exclusive_licences(registry, pack.pack_id))
        reasons = []
        if products:
            reasons.append('PRODUCT_MATCH')
            if pack.alcohol_scope == 'ALCOHOLIC':
                reasons.append('ALCOHOL_SCOPE_MATCH')
            elif pack.alcohol_scope == 'NON_ALCOHOLIC':
                reasons.append('NON_ALCOHOL_SCOPE_MATCH')
        if licences:
            reasons.append('LICENSE_MATCH')
        if reasons:
            status = 'SELECTED'
        elif profile.products_complete:
            status, reasons = 'NOT_SELECTED', ['PRODUCT_MISMATCH']
        else:
            status, reasons = 'UNKNOWN', ['PROFILE_INCOMPLETE']
        out.append(PackSelection(pack_id=pack.pack_id, status=status, reason_codes=reasons, product_ids=products,
                                 license_classes=licences))
    return out


# -- targets ---------------------------------------------------------------------------------------
class _Target:
    """What one target states for each dimension: (values, complete). A licence stated as UNKNOWN
    leaves the licence list open even on a complete profile; APPLIED or NOT_HELD is a known absence."""

    def __init__(self, level, target_id, entity_id, facts, products, products_complete):
        self.level, self.target_id, self.entity_id = level, target_id, entity_id
        self.facts, self.products, self.products_complete = facts, products, products_complete


def _entity_facts(entity):
    licences_open = any(l.status == 'UNKNOWN' for l in entity.licenses)
    return {'entity_classes': (set(entity.entity_classes), entity.profile_complete),
            'activity_classes': (set(entity.activity_classes), entity.profile_complete),
            'license_classes': ({l.license_class for l in entity.licenses if l.status == 'HELD'},
                                entity.profile_complete and not licences_open)}


def _targets(profile: EnterpriseProfile, level: str) -> list[_Target]:
    if level == 'LEGAL_ENTITY':
        return [_Target(level, e.entity_id, e.entity_id, _entity_facts(e), e.product_ids, profile.products_complete)
                for e in profile.legal_entities]
    if level == 'FACILITY':
        out = []
        for f in profile.facilities:
            facts = _entity_facts(profile.entity(f.entity_id))
            facts |= {'facility_classes': (set(f.facility_classes), True),
                      'activity_classes': (set(f.activity_classes), f.activities_complete)}
            out.append(_Target(level, f.facility_id, f.entity_id, facts, f.product_ids,
                               profile.products_complete and f.activities_complete))
        return out
    if level == 'ACTIVITY':
        out = []
        for a in profile.activities:
            facts = _entity_facts(profile.entity(a.entity_id))
            facts['activity_classes'] = ({a.activity_class}, True)
            out.append(_Target(level, a.activity_id, a.entity_id, facts, a.product_ids, bool(a.product_ids)))
        return out
    return [_Target(level, p.product_id, None, {}, [p.product_id], True) for p in profile.products]


# -- facts -----------------------------------------------------------------------------------------
def _fact(values: set, complete: bool, required: list[str]) -> Fact:
    if values & set(required):
        return 'YES'
    return 'NO' if complete else 'UNKNOWN'


def _product_fact(scope: ObligationScope, product, registry: Registry) -> tuple[Fact, list[str]]:
    """(fact, codes) for one product against the scope's product dimensions and alcohol category."""
    facts, yes, no = [], [], []
    if scope.product_classes:
        fact = 'YES' if product.product_class in scope.product_classes else 'NO'
        facts.append(fact)
        (yes if fact == 'YES' else no).append('PRODUCT_MATCH' if fact == 'YES' else 'PRODUCT_MISMATCH')
    if scope.product_attributes:
        fact = _fact(set(product.tags), product.tags_complete, scope.product_attributes)
        facts.append(fact)
        if fact == 'YES':
            yes.append('PRODUCT_ATTRIBUTE_MATCH')
        elif fact == 'NO':
            no.append('PRODUCT_ATTRIBUTE_MISMATCH')
    if scope.alcohol_scope != 'ANY':
        alcoholic = scope.alcohol_scope == 'ALCOHOLIC'
        category = registry.vocabulary.alcohol_of(product.product_class)
        stem = 'ALCOHOL_SCOPE' if alcoholic else 'NON_ALCOHOL_SCOPE'
        fact = 'YES' if category == scope.alcohol_scope else 'NO'
        facts.append(fact)
        (yes if fact == 'YES' else no).append(f'{stem}_MATCH' if fact == 'YES' else f'{stem}_MISMATCH')
    if 'NO' in facts:
        return 'NO', no
    if 'UNKNOWN' in facts:
        return 'UNKNOWN', ['PROFILE_INCOMPLETE']
    return 'YES', yes


def _constrains_products(scope: ObligationScope) -> bool:
    return bool(scope.product_classes or scope.product_attributes or scope.alcohol_scope != 'ANY')


def _decide(scope: ObligationScope, target: _Target, profile: EnterpriseProfile, registry: Registry) -> Decision:
    gates, yes, no, unknown = [], [], [], []
    for dimension, required in scope.constraints().items():
        if dimension in PRODUCT_DIMENSIONS:
            continue
        values, complete = target.facts[dimension]
        fact = _fact(values, complete, required)
        code = CODE_OF[dimension]
        gates.append(gate(GATE_OF[dimension], GATE_STATUS[fact], f'{dimension} {fact}', clear=fact != 'UNKNOWN',
                          required=sorted(required), stated=sorted(values), complete=complete))
        {'YES': yes, 'NO': no, 'UNKNOWN': unknown}[fact].append(
            f'{code}_MATCH' if fact == 'YES' else f'{code}_MISMATCH' if fact == 'NO' else 'PROFILE_INCOMPLETE')

    matched, partial = [], False
    if _constrains_products(scope):
        readings = {pid: _product_fact(scope, profile.product(pid), registry) for pid in target.products}
        matched = [pid for pid, (fact, _) in readings.items() if fact == 'YES']
        facts = {fact for fact, _ in readings.values()}
        if matched:
            fact = 'YES'
            codes = next(codes for f, codes in readings.values() if f == 'YES')
            partial = len(matched) < len(readings)
        elif 'UNKNOWN' in facts or not target.products_complete:
            fact, codes = 'UNKNOWN', ['PROFILE_INCOMPLETE']
        elif readings:
            fact = 'NO'
            codes = list(dict.fromkeys(c for _, cs in readings.values() for c in cs))
        else:
            fact = 'NO'
            codes = ['ALCOHOL_SCOPE_MISMATCH' if scope.alcohol_scope == 'ALCOHOLIC' else
                     'NON_ALCOHOL_SCOPE_MISMATCH' if scope.alcohol_scope == 'NON_ALCOHOLIC' else 'PRODUCT_MISMATCH']
        gates.append(gate('PRODUCT_SCOPE', GATE_STATUS[fact], f'products {fact}', clear=fact != 'UNKNOWN',
                          matched=matched, products=list(target.products), complete=target.products_complete))
        {'YES': yes, 'NO': no, 'UNKNOWN': unknown}[fact].extend(codes)

    if scope.scope_status == 'UNCLEAR':
        status, reasons = 'UNKNOWN', ['REGULATORY_SCOPE_UNCLEAR']
    elif no:
        status, reasons = 'DOES_NOT_APPLY', no
    elif unknown:
        status, reasons = 'UNKNOWN', unknown
    elif partial:
        status, reasons = 'PARTIAL', yes + ['PARTIAL_PRODUCT_SCOPE']
    else:
        status, reasons = 'APPLIES', yes or ['NO_CONSTRAINT']
    return _decision(scope, registry, level=target.level, target_id=target.target_id, entity_id=target.entity_id,
                     status=status, reasons=reasons, applies_to_products=matched if status in ('APPLIES', 'PARTIAL') else [],
                     gates=gates)


def _decision(scope: ObligationScope, registry: Registry, *, reasons, packs=(), **fields) -> Decision:
    binding = registry.regulations[scope.regulation_id].binding_status
    reasons = list(dict.fromkeys(reasons))
    if binding != 'BINDING':
        reasons.append('GUIDANCE_ONLY')
    return Decision(scope_id=scope.scope_id, regulation_id=scope.regulation_id, packs=list(packs), reason_codes=reasons,
                    binding_status=binding, creates_obligation=binding == 'BINDING', basis=scope.origin,
                    provision_ref=scope.provision_ref, provision_status=scope.provision_status, **fields)


def evaluate_scope(scope: ObligationScope, profile: EnterpriseProfile, registry: Registry, packs=()) -> list[Decision]:
    """One decision per target of the scope's level."""
    out = []
    for target in _targets(profile, scope.level):
        decision = _decide(scope, target, profile, registry)
        out.append(decision.model_copy(update={'packs': list(packs)}) if packs else decision)
    return out


# -- rollups ---------------------------------------------------------------------------------------
def _combine(children: dict[str, Decision], empty_known: bool) -> tuple[str, list[str]]:
    statuses = [d.status for d in children.values()]
    if not statuses:
        return ('DOES_NOT_APPLY', ['NO_TARGET_AT_LEVEL']) if empty_known else ('UNKNOWN', ['PROFILE_INCOMPLETE'])
    if all(s == 'APPLIES' for s in statuses):
        return 'APPLIES', ['ALL_CHILDREN_APPLY']
    if any(s in ('APPLIES', 'PARTIAL') for s in statuses):
        return 'PARTIAL', ['PARTIAL_CHILD_SCOPE'] + (['CHILD_UNKNOWN'] if 'UNKNOWN' in statuses else [])
    if 'UNKNOWN' in statuses:
        unknown = [c for d in children.values() if d.status == 'UNKNOWN' for c in d.reason_codes if c != 'GUIDANCE_ONLY']
        return 'UNKNOWN', unknown
    return 'DOES_NOT_APPLY', ['NO_CHILD_APPLIES']


def _rolled(scope, registry, level, target_id, entity_id, children, empty_known, packs):
    status, reasons = _combine(children, empty_known)
    products = sorted({p for d in children.values() if d.status in ('APPLIES', 'PARTIAL') for p in d.applies_to_products})
    return _decision(scope, registry, packs=packs, level=level, target_id=target_id, entity_id=entity_id, status=status,
                     reasons=reasons, applies_to_products=products,
                     children={k: d.status for k, d in children.items()})


def rollup(scope: ObligationScope, decisions: list[Decision], profile: EnterpriseProfile, registry: Registry,
           packs=()) -> list[Decision]:
    """Entity rollups (for facility, activity and product scopes) and the group rollup."""
    by_target = {d.target_id: d for d in decisions}
    if scope.level == 'LEGAL_ENTITY':
        entities = by_target
    else:
        entities = {}
        for entity in profile.legal_entities:
            if scope.level == 'PRODUCT':
                children = {p: by_target[p] for p in entity.product_ids if p in by_target}
                known = profile.products_complete
            elif scope.level == 'FACILITY':
                children = {f.facility_id: by_target[f.facility_id] for f in profile.facilities_of(entity.entity_id)
                            if f.facility_id in by_target}
                known = entity.profile_complete
            else:
                children = {a.activity_id: by_target[a.activity_id] for a in profile.activities_of(entity.entity_id)
                            if a.activity_id in by_target}
                known = entity.profile_complete
            entities[entity.entity_id] = _rolled(scope, registry, 'LEGAL_ENTITY', entity.entity_id, entity.entity_id,
                                                 children, known, packs)
    group = _rolled(scope, registry, 'GROUP', profile.group.group_id, None, entities, True, packs)
    out = [] if scope.level == 'LEGAL_ENTITY' else list(entities.values())
    return out + [group]


# -- resolution ------------------------------------------------------------------------------------
def resolve(profile: EnterpriseProfile, registry: Registry, include_unselected: bool = False) -> Resolution:
    """Select packs, route every scope of a SELECTED or UNKNOWN pack (UNKNOWN packs are routed so a
    gap in the profile is not read as out of scope), and optionally list the scopes of unselected
    packs as DOES_NOT_APPLY / PACK_NOT_SELECTED."""
    selections = select_packs(profile, registry)
    routable = [s.pack_id for s in selections if s.status != 'NOT_SELECTED']
    packs_of: dict[str, list[str]] = {}
    for pack_id in routable:
        for scope in registry.scopes_of(pack_id):
            packs_of.setdefault(scope.scope_id, []).append(pack_id)
    decisions, rollups = [], []
    for scope_id, packs in packs_of.items():
        scope = registry.scopes[scope_id]
        routed = evaluate_scope(scope, profile, registry, packs=sorted(packs))
        decisions.extend(routed)
        rollups.extend(rollup(scope, routed, profile, registry, packs=sorted(packs)))
    out_of_scope = []
    if include_unselected:
        for selection in selections:
            if selection.status != 'NOT_SELECTED':
                continue
            for scope in registry.scopes_of(selection.pack_id):
                if scope.scope_id in packs_of or any(d.scope_id == scope.scope_id for d in out_of_scope):
                    continue
                out_of_scope.append(_decision(scope, registry, packs=[selection.pack_id], level='GROUP',
                                              target_id=profile.group.group_id, status='DOES_NOT_APPLY',
                                              reasons=['PACK_NOT_SELECTED']))
    return Resolution(profile_id=profile.profile_id, selections=selections, decisions=decisions, rollups=rollups,
                      out_of_scope=out_of_scope)

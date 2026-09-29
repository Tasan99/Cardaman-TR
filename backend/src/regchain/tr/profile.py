"""Enterprise profile: one group, its legal entities, facilities, products and activities.

A duty lands on a legal entity, a facility, a product or one activity of an entity, never on
"the group" as such, so the profile keeps each level as its own record and every reference is
checked inside the group. Lists are open-world unless the record says they are complete: an
entity whose activities are not declared complete can still carry out an activity nobody wrote
down, so a missing class is UNKNOWN there, not a mismatch. The routing (routing.py) reads that
flag; nothing here decides applicability.

to_company() projects one legal entity to the flat Company the existing engine consumes, so the
extraction, grounding and policy comparison pipeline runs per legal entity unchanged.
"""
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from ..pilot.schema import Company, Strict
from .core import Vocabulary

Fact = Literal['YES', 'NO', 'UNKNOWN']
LicenseStatus = Literal['HELD', 'APPLIED', 'NOT_HELD', 'UNKNOWN']
Level = Literal['GROUP', 'LEGAL_ENTITY', 'FACILITY', 'PRODUCT', 'ACTIVITY']


class ProfileError(ValueError):
    pass


class License(Strict):
    license_class: str = Field(min_length=1)
    status: LicenseStatus
    facility_id: str | None = None


class Product(Strict):
    product_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    product_class: str = Field(min_length=1)
    tags: list[str] = []
    # True when tags lists every attribute a scope could ask about (claims, caffeine, packaging).
    tags_complete: bool = False


class Facility(Strict):
    facility_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    facility_classes: list[str] = Field(min_length=1)
    activity_classes: list[str] = []
    product_ids: list[str] = []
    activities_complete: bool = False
    city: str | None = None


class Activity(Strict):
    """One activity of one entity, optionally at one of its facilities and for some of its products:
    the level a duty like an advertising restriction lands on."""
    activity_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    activity_class: str = Field(min_length=1)
    facility_id: str | None = None
    product_ids: list[str] = []
    channels: list[str] = []


class LegalEntity(Strict):
    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    entity_classes: list[str] = []
    activity_classes: list[str] = []
    licenses: list[License] = []
    product_ids: list[str] = []
    sales_channels: list[str] = []
    # True when entity_classes, activity_classes and licenses are all stated in full.
    profile_complete: bool = False


class Group(Strict):
    group_id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class EnterpriseProfile(Strict):
    format: Literal['cardaman-tr-enterprise-profile/1']
    profile_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    synthetic: bool
    representative_type: str | None = None
    disclaimer: str = ''
    jurisdiction: Literal['TR']
    group: Group
    legal_entities: list[LegalEntity] = Field(min_length=1)
    facilities: list[Facility] = []
    products: list[Product] = []
    activities: list[Activity] = []
    # True when products lists every product the group handles.
    products_complete: bool = False

    @model_validator(mode='after')
    def _graph(self):
        if self.synthetic and not self.disclaimer.strip():
            raise ValueError('a synthetic profile needs a disclaimer saying it is not real company data')
        seen = {}
        for kind, key, items in (('group', 'group_id', [self.group]), ('entity', 'entity_id', self.legal_entities),
                                 ('facility', 'facility_id', self.facilities), ('product', 'product_id', self.products),
                                 ('activity', 'activity_id', self.activities)):
            for item in items:
                value = getattr(item, key)
                if value in seen:
                    raise ValueError(f'id {value} is used by a {seen[value]} and a {kind}')
                seen[value] = kind
        entities = {e.entity_id: e for e in self.legal_entities}
        facilities = {f.facility_id: f for f in self.facilities}
        products = {p.product_id for p in self.products}
        for facility in self.facilities:
            if facility.entity_id not in entities:
                raise ValueError(f'facility {facility.facility_id} belongs to unknown entity {facility.entity_id}')
        for entity in self.legal_entities:
            missing = [p for p in entity.product_ids if p not in products]
            if missing:
                raise ValueError(f'entity {entity.entity_id} handles unknown products {missing}')
            for licence in entity.licenses:
                if licence.facility_id is not None and (licence.facility_id not in facilities
                                                        or facilities[licence.facility_id].entity_id != entity.entity_id):
                    raise ValueError(f'licence {licence.license_class} of {entity.entity_id} names facility '
                                     f'{licence.facility_id}, which is not one of its facilities')
        for facility in self.facilities:
            foreign = [p for p in facility.product_ids if p not in entities[facility.entity_id].product_ids]
            if foreign:
                raise ValueError(f'facility {facility.facility_id} holds products {foreign} its entity '
                                 f'{facility.entity_id} does not handle')
        for activity in self.activities:
            if activity.entity_id not in entities:
                raise ValueError(f'activity {activity.activity_id} belongs to unknown entity {activity.entity_id}')
            entity = entities[activity.entity_id]
            if activity.activity_class not in entity.activity_classes:
                raise ValueError(f'activity {activity.activity_id}: {activity.activity_class} is not an activity of '
                                 f'{entity.entity_id}')
            if activity.facility_id is not None and (activity.facility_id not in facilities
                                                     or facilities[activity.facility_id].entity_id != entity.entity_id):
                raise ValueError(f'activity {activity.activity_id} names facility {activity.facility_id}, '
                                 f'which is not one of {entity.entity_id}')
            foreign = [p for p in activity.product_ids if p not in entity.product_ids]
            if foreign:
                raise ValueError(f'activity {activity.activity_id} covers products {foreign} {entity.entity_id} '
                                 f'does not handle')
        return self

    def entity(self, entity_id: str) -> LegalEntity:
        return next(e for e in self.legal_entities if e.entity_id == entity_id)

    def facility(self, facility_id: str) -> Facility:
        return next(f for f in self.facilities if f.facility_id == facility_id)

    def product(self, product_id: str) -> Product:
        return next(p for p in self.products if p.product_id == product_id)

    def activity(self, activity_id: str) -> Activity:
        return next(a for a in self.activities if a.activity_id == activity_id)

    def facilities_of(self, entity_id: str) -> list[Facility]:
        return [f for f in self.facilities if f.entity_id == entity_id]

    def activities_of(self, entity_id: str) -> list[Activity]:
        return [a for a in self.activities if a.entity_id == entity_id]


def vocabulary_errors(profile: EnterpriseProfile, vocab: Vocabulary) -> list[str]:
    errors = []

    def check(where, kind, values):
        unknown = vocab.unknown(kind, values)
        if unknown:
            errors.append(f'{where}: unknown {kind} {unknown}')

    for entity in profile.legal_entities:
        check(entity.entity_id, 'entity_classes', entity.entity_classes)
        check(entity.entity_id, 'activity_classes', entity.activity_classes)
        check(entity.entity_id, 'license_classes', [l.license_class for l in entity.licenses])
        check(entity.entity_id, 'sales_channels', entity.sales_channels)
    for facility in profile.facilities:
        check(facility.facility_id, 'facility_classes', facility.facility_classes)
        check(facility.facility_id, 'activity_classes', facility.activity_classes)
    for product in profile.products:
        check(product.product_id, 'product_classes', [product.product_class])
        check(product.product_id, 'product_tags', product.tags)
    for activity in profile.activities:
        check(activity.activity_id, 'sales_channels', activity.channels)
    return errors


def load_profile(data: dict, vocab: Vocabulary) -> EnterpriseProfile:
    try:
        profile = EnterpriseProfile.model_validate(data)
    except ValidationError as exc:
        raise ProfileError(str(exc)) from exc
    errors = vocabulary_errors(profile, vocab)
    if errors:
        raise ProfileError('; '.join(errors))
    return profile


def product_alcohol(product: Product, vocab: Vocabulary) -> str:
    return vocab.alcohol_of(product.product_class)


def _stated(values: list[str], complete: bool):
    """A list as the legacy Company reads it: None when nothing is stated and the list may be partial."""
    return values if values or complete else None


def to_company(profile: EnterpriseProfile, entity_id: str, vocab: Vocabulary) -> Company:
    entity = profile.entity(entity_id)
    products = [f'{p.name} ({vocab.label("product_classes", p.product_class)})'
                for p in profile.products if p.product_id in entity.product_ids]
    classes = ', '.join(vocab.label('entity_classes', c) for c in entity.entity_classes)
    return Company(
        id=f'{profile.profile_id}/{entity.entity_id}', name=entity.name, version=profile.version, synthetic=profile.synthetic,
        jurisdictions=['Türkiye'],
        activities=_stated([vocab.label('activity_classes', a) for a in entity.activity_classes], entity.profile_complete),
        licences=_stated([vocab.label('license_classes', l.license_class) for l in entity.licenses if l.status == 'HELD'],
                         entity.profile_complete),
        products=_stated(products, profile.products_complete),
        customer_types=None,
        description=f'{entity.name}: {classes or "sınıfı belirtilmemiş"} ({profile.group.name} grubu).')


class PolicyScope(Strict):
    """Where an internal policy or procedure is in force: the group, one entity or one facility,
    optionally limited to activity or product classes. inherits=False keeps a policy at its owner."""
    policy_id: str = Field(min_length=1)
    owner_level: Literal['GROUP', 'LEGAL_ENTITY', 'FACILITY']
    entity_id: str | None = None
    facility_id: str | None = None
    inherits: bool = True
    activity_classes: list[str] = []
    product_classes: list[str] = []

    @model_validator(mode='after')
    def _owner(self):
        wants = {'GROUP': (False, False), 'LEGAL_ENTITY': (True, False), 'FACILITY': (False, True)}[self.owner_level]
        if (self.entity_id is not None, self.facility_id is not None) != wants:
            raise ValueError(f'{self.policy_id}: a {self.owner_level} policy names '
                             f'{"an entity" if wants[0] else "a facility" if wants[1] else "neither entity nor facility"}')
        return self


def _reach(policy: PolicyScope, profile: EnterpriseProfile, level: str, target_id: str):
    """(covers, reason) from the policy owner alone."""
    if policy.owner_level == 'GROUP':
        if level == 'GROUP' or policy.inherits:
            return 'YES', 'GROUP_POLICY' if level == 'GROUP' else 'GROUP_POLICY_INHERITED'
        return 'NO', 'GROUP_POLICY_NOT_INHERITED'
    if policy.owner_level == 'LEGAL_ENTITY':
        if level == 'GROUP':
            return 'NO', 'NARROWER_THAN_TARGET'
        owner = target_id if level == 'LEGAL_ENTITY' else profile.facility(target_id).entity_id
        if owner != policy.entity_id:
            return 'NO', 'OTHER_ENTITY_POLICY'
        if level == 'LEGAL_ENTITY':
            return 'YES', 'ENTITY_POLICY'
        return ('YES', 'ENTITY_POLICY_INHERITED') if policy.inherits else ('NO', 'ENTITY_POLICY_NOT_INHERITED')
    if level != 'FACILITY':
        return 'NO', 'NARROWER_THAN_TARGET'
    return ('YES', 'FACILITY_POLICY') if target_id == policy.facility_id else ('NO', 'OTHER_FACILITY_POLICY')


def policy_covers(policy: PolicyScope, profile: EnterpriseProfile, level: str, target_id: str) -> dict:
    """Whether a policy is in force for a target (GROUP, LEGAL_ENTITY or FACILITY): {'covers': Fact, 'reason': code}.

    Scope only: a YES says the policy is the target's to compare, not that it covers any duty."""
    covers, reason = _reach(policy, profile, level, target_id)
    if covers != 'YES' or not policy.activity_classes or level == 'GROUP':
        return {'covers': covers, 'reason': reason}
    if level == 'LEGAL_ENTITY':
        entity = profile.entity(target_id)
        have, complete = set(entity.activity_classes), entity.profile_complete
    else:
        facility = profile.facility(target_id)
        have, complete = set(facility.activity_classes), facility.activities_complete
    if have & set(policy.activity_classes):
        return {'covers': 'YES', 'reason': reason}
    if complete:
        return {'covers': 'NO', 'reason': 'POLICY_ACTIVITY_MISMATCH'}
    return {'covers': 'UNKNOWN', 'reason': 'POLICY_ACTIVITY_UNSTATED'}

"""CARDAMAN_TR_CORE: sector-independent metadata for Turkish regulatory sources.

A regulation is catalogued once, with its regulator, its source class and the tags that say
whom it can concern; packs (packs.py) group catalogued regulations for a sector and never copy
them. An ObligationScope says, for one regulation, which kind of target a duty lands on
(legal entity, facility, product or activity) and which classes of that target it names. The
scope comes from pack metadata until the provision is ingested and its scope extracted;
provision_status says which, so a scope without an article is never presented as grounded.

Source classes:
  BINDING             kanun, Cumhurbaşkanlığı kararnamesi, yönetmelik, tebliğ
  OFFICIAL_GUIDANCE   resmi rehber, resmi kılavuz, resmi kurum açıklaması: reads binding text,
                      never creates a duty of its own
  DECISION_PRECEDENT  kurul kararları: second phase, not enabled in an active MVP pack
"""
import re
from datetime import date
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..pilot.schema import Strict

JURISDICTION = 'TR'
BindingStatus = Literal['BINDING', 'OFFICIAL_GUIDANCE', 'DECISION_PRECEDENT']
RegulationType = Literal['KANUN', 'CUMHURBASKANLIGI_KARARNAMESI', 'YONETMELIK', 'TEBLIG', 'RESMI_REHBER', 'RESMI_KILAVUZ',
                         'KURUM_ACIKLAMASI', 'KURUL_KARARI']
BINDING_OF = {'KANUN': 'BINDING', 'CUMHURBASKANLIGI_KARARNAMESI': 'BINDING', 'YONETMELIK': 'BINDING', 'TEBLIG': 'BINDING',
              'RESMI_REHBER': 'OFFICIAL_GUIDANCE', 'RESMI_KILAVUZ': 'OFFICIAL_GUIDANCE', 'KURUM_ACIKLAMASI': 'OFFICIAL_GUIDANCE',
              'KURUL_KARARI': 'DECISION_PRECEDENT'}
MVP_LAYERS = ('BINDING', 'OFFICIAL_GUIDANCE')
SECOND_PHASE_LAYERS = ('DECISION_PRECEDENT',)
EffectiveStatus = Literal['IN_FORCE', 'NOT_YET_IN_FORCE', 'REPEALED', 'UNKNOWN']
MetadataStatus = Literal['VERIFIED', 'UNVERIFIED']
AlcoholCategory = Literal['ALCOHOLIC', 'NON_ALCOHOLIC']
AlcoholScope = Literal['ALCOHOLIC', 'NON_ALCOHOLIC', 'ANY']
ScopeLevel = Literal['LEGAL_ENTITY', 'FACILITY', 'PRODUCT', 'ACTIVITY']
SHA256 = re.compile(r'^[0-9a-f]{64}$')
REGULATION_ID = r'^TR:[A-Z_]+:[A-Z0-9][A-Z0-9_.-]*$'
TAG_FIELDS = ('sector_tags', 'entity_tags', 'activity_tags', 'product_tags', 'facility_tags')
# Which vocabulary list each tag or class field is checked against.
VOCABULARY_OF = {'sector_tags': 'sector_tags', 'entity_tags': 'entity_classes', 'activity_tags': 'activity_classes',
                 'product_tags': 'product_classes', 'facility_tags': 'facility_classes', 'entity_classes': 'entity_classes',
                 'activity_classes': 'activity_classes', 'facility_classes': 'facility_classes',
                 'license_classes': 'license_classes', 'product_classes': 'product_classes', 'product_attributes': 'product_tags'}


class ProductClass(Strict):
    label_tr: str = Field(min_length=1)
    alcohol: AlcoholCategory


class Vocabulary(Strict):
    """The closed lists every pack, scope and profile is written in; an unknown term is an error,
    because a misspelt class would silently never match."""
    format: Literal['cardaman-tr-vocabulary/1']
    regulators: dict[str, str]
    sector_tags: dict[str, str]
    entity_classes: dict[str, str]
    activity_classes: dict[str, str]
    facility_classes: dict[str, str]
    license_classes: dict[str, str]
    product_classes: dict[str, ProductClass]
    product_tags: dict[str, str]
    sales_channels: dict[str, str]

    def unknown(self, kind: str, values) -> list[str]:
        known = getattr(self, kind)
        return sorted({v for v in values or () if v not in known})

    def alcohol_of(self, product_class: str) -> str:
        return self.product_classes[product_class].alcohol

    def label(self, kind: str, key: str) -> str:
        value = getattr(self, kind)[key]
        return value.label_tr if isinstance(value, ProductClass) else value


class RegulationVersion(Strict):
    """One published text of a regulation; previous_version_id links the versions into one chain
    so a later change-impact step can compare a text with the one it replaced."""
    regulation_id: str = Field(pattern=REGULATION_ID)
    version_id: str = Field(min_length=1)
    publication_date: date | None
    effective_date: date | None
    source_hash: str | None
    previous_version_id: str | None

    @field_validator('source_hash')
    @classmethod
    def _hash(cls, value):
        if value is not None and not SHA256.match(value):
            raise ValueError('source_hash must be a lower-case SHA-256 hex digest')
        return value

    @model_validator(mode='after')
    def _not_self(self):
        if self.previous_version_id == self.version_id:
            raise ValueError('a version cannot replace itself')
        return self


class SourceRef(Strict):
    """Where the official text is fetched from: the identifiers of the Mevzuat Bilgi Sistemi (mevzuat.gov.tr) record.
    The corpus adapter (corpus.py) turns them into the text URL; nothing else may be fetched for the regulation."""
    adapter: Literal['mevzuat']
    kind: str = Field(pattern=r'^\d{1,2}$')
    number: str = Field(pattern=r'^\d{1,12}$')
    tertip: str = Field(default='5', pattern=r'^\d$')


class RegulationMeta(Strict):
    regulation_id: str = Field(pattern=REGULATION_ID)
    title: str = Field(min_length=1)
    jurisdiction: Literal['TR']
    regulator: str = Field(min_length=1)
    regulation_type: RegulationType
    binding_status: BindingStatus
    number: str | None
    effective_date: date | None
    effective_status: EffectiveStatus
    # UNVERIFIED: title, number or dates were not yet checked against mevzuat.gov.tr / Resmî Gazete.
    # VERIFIED: the text was fetched from the official source named by source_ref, its printed title matches `title`,
    # and the first version carries the hash of the stored text (corpus.verify_catalogue checks all three).
    metadata_status: MetadataStatus
    verification_note: str = ''
    source_url: str | None = None
    source_ref: SourceRef | None = None
    # The Resmî Gazete date and issue the official catalogue records for the text; never typed from memory.
    gazette_date: date | None = None
    gazette_number: str | None = None
    # The level an impersonal clause of this text lands on when its wording names no addressee: a product standard
    # (a Türk Gıda Kodeksi communiqué, the labelling regulation) speaks about the product. Pack metadata, flagged on
    # every scope that relies on it; never used when the clause itself says whom it binds.
    default_level: Literal['LEGAL_ENTITY', 'FACILITY', 'PRODUCT', 'ACTIVITY'] | None = None
    interprets: list[str] = []
    sector_tags: list[str] = Field(min_length=1)
    entity_tags: list[str] = []
    activity_tags: list[str] = []
    product_tags: list[str] = []
    facility_tags: list[str] = []
    versions: list[RegulationVersion] = []

    @model_validator(mode='after')
    def _consistent(self):
        if self.regulation_id.split(':')[1] != self.regulation_type:
            raise ValueError(f'{self.regulation_id}: the id must carry the regulation type {self.regulation_type}')
        if BINDING_OF[self.regulation_type] != self.binding_status:
            raise ValueError(f'{self.regulation_id}: a {self.regulation_type} is {BINDING_OF[self.regulation_type]}, '
                             f'not {self.binding_status}')
        if self.binding_status == 'BINDING' and self.interprets:
            raise ValueError(f'{self.regulation_id}: binding text does not interpret other text')
        if self.binding_status != 'BINDING' and not self.interprets:
            raise ValueError(f'{self.regulation_id}: {self.binding_status} must name the binding text it interprets')
        if self.metadata_status == 'UNVERIFIED' and not self.verification_note.strip():
            raise ValueError(f'{self.regulation_id}: an UNVERIFIED entry must say what is still to be verified')
        if self.effective_date is None and self.effective_status != 'UNKNOWN':
            raise ValueError(f'{self.regulation_id}: effective_status needs an effective_date')
        if self.metadata_status == 'VERIFIED' and (self.source_ref is None or not self.source_url
                                                   or not any(v.source_hash for v in self.versions)):
            raise ValueError(f'{self.regulation_id}: a VERIFIED entry names its official source and the hash of the fetched text')
        version_chain(self)
        return self


def version_chain(meta: RegulationMeta) -> list[RegulationVersion]:
    """The versions oldest first; raises ValueError unless they form one unbroken chain."""
    if not meta.versions:
        return []
    by_id = {}
    for version in meta.versions:
        if version.regulation_id != meta.regulation_id:
            raise ValueError(f'{version.version_id} belongs to {version.regulation_id}, not {meta.regulation_id}')
        if version.version_id in by_id:
            raise ValueError(f'duplicate version {version.version_id}')
        by_id[version.version_id] = version
    roots = [v for v in meta.versions if v.previous_version_id is None]
    if len(roots) != 1:
        raise ValueError(f'{meta.regulation_id}: versions need exactly one first version, found {len(roots)}')
    successor = {}
    for version in meta.versions:
        if version.previous_version_id is None:
            continue
        if version.previous_version_id not in by_id:
            raise ValueError(f'{version.version_id} replaces unknown version {version.previous_version_id}')
        if version.previous_version_id in successor:
            raise ValueError(f'{version.previous_version_id} is replaced twice')
        successor[version.previous_version_id] = version
    chain = [roots[0]]
    while chain[-1].version_id in successor:
        chain.append(successor[chain[-1].version_id])
    if len(chain) != len(meta.versions):
        raise ValueError(f'{meta.regulation_id}: versions do not form one chain')
    return chain


# The dimensions a scope may constrain at each level. A product-level duty (a label rule) cannot
# depend on the facility or the entity: that would be a facility- or entity-level duty.
LEVEL_DIMENSIONS = {
    'LEGAL_ENTITY': ('entity_classes', 'activity_classes', 'license_classes', 'product_classes', 'product_attributes'),
    'FACILITY': ('facility_classes', 'activity_classes', 'entity_classes', 'license_classes', 'product_classes',
                 'product_attributes'),
    'ACTIVITY': ('activity_classes', 'entity_classes', 'license_classes', 'product_classes', 'product_attributes'),
    'PRODUCT': ('product_classes', 'product_attributes'),
}
DIMENSIONS = ('entity_classes', 'activity_classes', 'facility_classes', 'license_classes', 'product_classes',
              'product_attributes')


class ObligationScope(Strict):
    """Whom one regulation's duty lands on, as classes of a target.

    An empty dimension is not a constraint. alcohol_scope restricts the products the target
    handles. scope_status UNCLEAR marks a duty whose addressee the source leaves open: it is
    routed, and always answers UNKNOWN with REGULATORY_SCOPE_UNCLEAR."""
    scope_id: str = Field(pattern=r'^[A-Z][A-Z0-9_]+$')
    regulation_id: str = Field(pattern=REGULATION_ID)
    topic: str = Field(min_length=1)
    description_tr: str = Field(min_length=1)
    level: ScopeLevel
    provision_ref: str | None
    provision_status: Literal['RESOLVED', 'UNRESOLVED']
    scope_status: Literal['DEFINED', 'UNCLEAR'] = 'DEFINED'
    origin: Literal['PACK_METADATA', 'EXTRACTED'] = 'PACK_METADATA'
    alcohol_scope: AlcoholScope = 'ANY'
    entity_classes: list[str] = []
    activity_classes: list[str] = []
    facility_classes: list[str] = []
    license_classes: list[str] = []
    product_classes: list[str] = []
    product_attributes: list[str] = []
    # Clause-level narrowing of an EXTRACTED scope (extraction.py); pack metadata leaves them empty.
    #   excluded_product_classes  classes the clause or the regulation's scope article leaves out ("sporcu içeceklerini kapsamaz")
    #   conditions                predicates over a product fact the clause states ("hacmen % 1,2’den fazla alkol içeren"),
    #                             each {'fact', 'op', 'value', 'quote'}; all must hold
    #   exceptions                predicates that lift the duty ("ihraç amaçlı üretilenler hariç"), each
    #                             {'fact', 'op', 'value', 'quote', 'effect'}; an exception without a predicate is recorded
    #                             for the reviewer and never decides
    excluded_product_classes: list[str] = []
    conditions: list[dict] = []
    exceptions: list[dict] = []
    # The exact wording of the clause in the stored text, and the version it was read from.
    quote: str = ''
    version_id: str | None = None

    @model_validator(mode='after')
    def _consistent(self):
        if (self.conditions or self.exceptions or self.excluded_product_classes or self.quote) and self.origin != 'EXTRACTED':
            raise ValueError(f'{self.scope_id}: conditions, exceptions and quotes belong to an extracted scope')
        if self.origin == 'EXTRACTED' and not self.quote:
            raise ValueError(f'{self.scope_id}: an extracted scope quotes its clause')
        if (self.provision_ref is None) != (self.provision_status == 'UNRESOLVED'):
            raise ValueError(f'{self.scope_id}: provision_ref is set exactly when provision_status is RESOLVED')
        allowed = LEVEL_DIMENSIONS[self.level]
        for dimension in DIMENSIONS:
            if getattr(self, dimension) and dimension not in allowed:
                raise ValueError(f'{self.scope_id}: a {self.level} scope cannot constrain {dimension}')
        if self.origin == 'EXTRACTED' and self.provision_status != 'RESOLVED':
            raise ValueError(f'{self.scope_id}: an extracted scope names its provision')
        return self

    def constraints(self) -> dict[str, list[str]]:
        return {d: getattr(self, d) for d in DIMENSIONS if getattr(self, d)}

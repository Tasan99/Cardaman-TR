"""Regulation pack registry: sector packs as validated data over the one TR catalogue.

A PACK is selectable for a company (BEVERAGE_ALCOHOL_TR, BEVERAGE_NON_ALCOHOL_TR); a MODULE is
shared content packs include (TR_FOOD_BEVERAGE_COMMON: food law, labelling, packaging,
advertising and distance sales that bind every beverage business). Each regulation is owned by
exactly one pack or module, so shared law is never copied into two packs and cannot drift.

The registry refuses, instead of ignoring: an unknown vocabulary term, a pack naming a
regulation the catalogue lacks, an alcohol-only scope inside a non-alcohol pack, guidance that
reads binding text the pack does not carry, and the second-phase DECISION_PRECEDENT layer in
an active pack.
"""
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from ..pilot.schema import Strict
from .core import (MVP_LAYERS, SECOND_PHASE_LAYERS, TAG_FIELDS, VOCABULARY_OF, AlcoholScope, BindingStatus, ObligationScope,
                   RegulationMeta, Vocabulary)

DATA = Path(__file__).resolve().parent / 'data'
PACK_CLASS_FIELDS = ('product_classes', 'activity_classes', 'facility_classes', 'entity_classes', 'license_classes')


class RegistryError(ValueError):
    pass


class RegulationPack(Strict):
    format: Literal['cardaman-tr-pack/1']
    pack_id: str = Field(pattern=r'^[A-Z][A-Z0-9_]+$')
    kind: Literal['PACK', 'MODULE']
    title: str = Field(min_length=1)
    jurisdiction: Literal['TR']
    sector: str = Field(min_length=1)
    subsector: str = Field(min_length=1)
    alcohol_scope: AlcoholScope
    regulators: list[str] = Field(min_length=1)
    regulation_ids: list[str] = Field(min_length=1)
    includes: list[str] = []
    product_classes: list[str] = []
    activity_classes: list[str] = []
    facility_classes: list[str] = []
    entity_classes: list[str] = []
    license_classes: list[str] = []
    source_layers: list[BindingStatus] = list(MVP_LAYERS)
    status: Literal['ACTIVE', 'DRAFT', 'RETIRED']
    version: str = Field(pattern=r'^\d+\.\d+\.\d+$')
    notes: str = ''


def load_json(path: Path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


class Registry:
    def __init__(self, vocabulary: Vocabulary, regulations: list[RegulationMeta], scopes: list[ObligationScope],
                 packs: list[RegulationPack]):
        self.vocabulary = vocabulary
        self.regulations = self._unique(regulations, 'regulation_id', 'regulation')
        self.scopes = self._unique(scopes, 'scope_id', 'scope')
        self._packs = self._unique(packs, 'pack_id', 'pack')
        self.packs = list(self._packs.values())
        self._validate()

    @classmethod
    def from_data(cls, vocabulary, regulations, scopes, packs):
        try:
            return cls(Vocabulary.model_validate(vocabulary),
                       [RegulationMeta.model_validate(r) for r in regulations['regulations']],
                       [ObligationScope.model_validate(s) for s in scopes['scopes']],
                       [RegulationPack.model_validate(p) for p in packs])
        except ValidationError as exc:
            raise RegistryError(str(exc)) from exc

    @classmethod
    def load(cls, directory: Path = DATA):
        directory = Path(directory)
        return cls.from_data(load_json(directory / 'vocabulary.json'), load_json(directory / 'regulations.json'),
                             load_json(directory / 'scopes.json'),
                             [load_json(path) for path in sorted((directory / 'packs').glob('*.json'))])

    @staticmethod
    def _unique(items, key, what):
        out = {}
        for item in items:
            value = getattr(item, key)
            if value in out:
                raise RegistryError(f'duplicate {what} {value}')
            out[value] = item
        return out

    def pack(self, pack_id: str) -> RegulationPack:
        if pack_id not in self._packs:
            raise KeyError(pack_id)
        return self._packs[pack_id]

    def selectable_packs(self):
        return [p for p in self.packs if p.kind == 'PACK' and p.status == 'ACTIVE']

    def closure(self, pack_id: str) -> list[str]:
        """The pack and every module it includes, depth first, each once."""
        order, stack = [], [pack_id]
        while stack:
            current = stack.pop()
            if current in order:
                continue
            order.append(current)
            stack.extend(reversed(self.pack(current).includes))
        return order

    def regulation_ids(self, pack_id: str) -> list[str]:
        """The regulations a pack carries (its own, then its modules'), limited to its source layers."""
        layers = set(self.pack(pack_id).source_layers)
        out = []
        for member in self.closure(pack_id):
            for regulation_id in self.pack(member).regulation_ids:
                if regulation_id not in out and self.regulations[regulation_id].binding_status in layers:
                    out.append(regulation_id)
        return out

    def owner(self, regulation_id: str) -> str:
        return next(p.pack_id for p in self.packs if regulation_id in p.regulation_ids)

    def scopes_of(self, pack_id: str) -> list[ObligationScope]:
        carried = set(self.regulation_ids(pack_id))
        return [s for s in self.scopes.values() if s.regulation_id in carried]

    # -- validation ------------------------------------------------------------------------------
    def _terms(self, where: str, kind: str, values):
        unknown = self.vocabulary.unknown(kind, values)
        if unknown:
            raise RegistryError(f'{where}: unknown {kind} {unknown}')

    def _validate(self):
        vocab = self.vocabulary
        for reg in self.regulations.values():
            if reg.regulator not in vocab.regulators:
                raise RegistryError(f'{reg.regulation_id}: unknown regulator {reg.regulator}')
            for field in TAG_FIELDS:
                self._terms(reg.regulation_id, VOCABULARY_OF[field], getattr(reg, field))
            for target in reg.interprets:
                if target not in self.regulations or self.regulations[target].binding_status != 'BINDING':
                    raise RegistryError(f'{reg.regulation_id}: interprets {target}, which is not catalogued binding text')

        owners = {}
        for pack in self.packs:
            for field in PACK_CLASS_FIELDS:
                self._terms(pack.pack_id, VOCABULARY_OF[field], getattr(pack, field))
            unknown = [r for r in pack.regulators if r not in vocab.regulators]
            if unknown:
                raise RegistryError(f'{pack.pack_id}: unknown regulators {unknown}')
            if pack.status == 'ACTIVE':
                enabled = sorted(set(pack.source_layers) & set(SECOND_PHASE_LAYERS))
                if enabled:
                    raise RegistryError(f'{pack.pack_id}: {enabled} is a second-phase source layer, not enabled in an active pack')
            for regulation_id in pack.regulation_ids:
                if regulation_id not in self.regulations:
                    raise RegistryError(f'{pack.pack_id}: regulation {regulation_id} is not in the catalogue')
                if regulation_id in owners:
                    raise RegistryError(f'{regulation_id} is owned by more than one pack or module: '
                                        f'{owners[regulation_id]}, {pack.pack_id}')
                owners[regulation_id] = pack.pack_id
            if pack.alcohol_scope != 'ANY':
                for product_class in pack.product_classes:
                    if vocab.alcohol_of(product_class) != pack.alcohol_scope:
                        raise RegistryError(f'{pack.pack_id}: product class {product_class} is not {pack.alcohol_scope}')
            for included in pack.includes:
                if included not in self._packs:
                    raise RegistryError(f'{pack.pack_id}: includes unknown {included}')
                if self._packs[included].kind != 'MODULE':
                    raise RegistryError(f'{pack.pack_id}: includes {included}, which is not a MODULE')
        self._acyclic()
        orphans = sorted(set(self.regulations) - set(owners))
        if orphans:
            raise RegistryError(f'catalogued regulations in no pack or module: {orphans}')

        for pack in self.packs:
            carried = set(self.regulation_ids(pack.pack_id))
            for regulation_id in sorted(carried):
                reg = self.regulations[regulation_id]
                if reg.regulator not in pack.regulators:
                    raise RegistryError(f'{pack.pack_id}: regulator {reg.regulator} of {regulation_id} is not listed')
                missing = [t for t in reg.interprets if t not in carried]
                if missing:
                    raise RegistryError(f'{pack.pack_id}: {regulation_id} interprets {missing}, which the pack does not carry')

        for scope in self.scopes.values():
            if scope.regulation_id not in self.regulations:
                raise RegistryError(f'{scope.scope_id}: regulation {scope.regulation_id} is not in the catalogue')
            for dimension, values in scope.constraints().items():
                self._terms(scope.scope_id, VOCABULARY_OF[dimension], values)
            owner = self._packs[owners[scope.regulation_id]]
            if scope.alcohol_scope != 'ANY' and owner.alcohol_scope != scope.alcohol_scope:
                raise RegistryError(f'{scope.scope_id}: an {scope.alcohol_scope} scope sits in {owner.pack_id} '
                                    f'({owner.alcohol_scope})')
            for product_class in scope.product_classes:
                category = vocab.alcohol_of(product_class)
                if scope.alcohol_scope != 'ANY' and category != scope.alcohol_scope:
                    raise RegistryError(f'{scope.scope_id}: product class {product_class} is {category}, '
                                        f'the scope is {scope.alcohol_scope}')

    def _acyclic(self):
        state = {}

        def visit(pack_id, path):
            if state.get(pack_id) == 'done':
                return
            if state.get(pack_id) == 'open':
                raise RegistryError(f'include cycle: {" -> ".join(path + [pack_id])}')
            state[pack_id] = 'open'
            for included in self._packs[pack_id].includes:
                visit(included, path + [pack_id])
            state[pack_id] = 'done'

        for pack_id in self._packs:
            visit(pack_id, [])

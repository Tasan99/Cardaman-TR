"""Sector knowledge the engine reads through one seam.

The pilot engine's gates, entity reading and conflict lexicon were written on the Turkish AML corpus
(Kanun 5549, Tedbirler Yönetmeliği) and carried that sector's obliged-party kinds, regulator names,
act vocabulary and risk categories as module constants. Those tables are sector knowledge, not engine
logic: they live in a sector pack (regchain.packs.<pack>.knowledge) and reach the engine only through
knowledge() below. The engine names no pack; the default is registered by regchain.compat when the
regchain package is imported, so every existing call path resolves to the same tables it always read.

Only raw data crosses this seam (strings, tuples, sets). The engine compiles its patterns from them
exactly as before, so a registered pack's tables produce the same objects the constants produced.
"""
from dataclasses import dataclass, field
from pathlib import Path


class SectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class SectorKnowledge:
    pack_id: str
    # applicability.py: obliged-party kinds (key, Turkish label, list-side pattern, company-side pattern), in order.
    categories: tuple = ()
    financial: frozenset = frozenset()                 # the wide "financial business" reading
    generic_categories: dict = field(default_factory=dict)
    financial_institutions: frozenset = frozenset()
    dnfbp: frozenset = frozenset()
    plainly_non_financial: frozenset = frozenset()
    defined_terms: dict = field(default_factory=dict)  # generic category -> pattern of the defined term
    regulator_pattern: str = r'(?!x)x'
    financial_catch_all_pattern: str = r'(?!x)x'
    generic_finance_pattern: str = r'(?!x)x'
    finance_words_pattern: str = r'(?!x)x'
    non_obliged_pattern: str = r'(?!x)x'
    # entities.py: counterparties and obliged kinds a clause names, customer families a profile states.
    counterparties: tuple = ()
    suppresses: dict = field(default_factory=dict)
    obliged: tuple = ()
    qualifiers: frozenset = frozenset()
    customer_families: tuple = ()
    satisfies: dict = field(default_factory=dict)
    covered_by_generic_legal: frozenset = frozenset()
    never_mismatch: frozenset = frozenset()
    entity_labels: dict = field(default_factory=dict)
    # conflict.py: compliance acts, institution nouns, customer groups, authorities.
    acts: dict = field(default_factory=dict)
    institution_pattern: str = r'(?!x)x'
    groups: dict = field(default_factory=dict)
    authority_pattern: str = r'(?!x)x'
    # engine.py: the risk categories the enrichment step classifies a duty into.
    risk_categories: tuple = ()


_REGISTRY: dict[str, SectorKnowledge] = {}
_ROOTS: dict[str, Path] = {}
_DEFAULT: list[str] = []


def register(knowledge: SectorKnowledge, default: bool = False, data_root: Path | None = None) -> SectorKnowledge:
    """Register a pack's engine tables and, optionally, the data root (catalogue, scopes, pack manifest) the pack
    registry discovers alongside the built-in root."""
    if knowledge.pack_id in _REGISTRY and _REGISTRY[knowledge.pack_id] is not knowledge:
        raise SectorError(f'sector knowledge {knowledge.pack_id} is already registered')
    _REGISTRY[knowledge.pack_id] = knowledge
    if data_root is not None:
        _ROOTS[knowledge.pack_id] = Path(data_root)
    if default:
        if _DEFAULT and _DEFAULT[0] != knowledge.pack_id:
            raise SectorError(f'default sector knowledge is already {_DEFAULT[0]}')
        _DEFAULT[:] = [knowledge.pack_id]
    return knowledge


def registered() -> tuple:
    return tuple(_REGISTRY)


def data_roots() -> list[Path]:
    """Data roots of the registered packs, in registration order."""
    return [root for _, root in _ROOTS.items()]


def default_pack_id() -> str | None:
    return _DEFAULT[0] if _DEFAULT else None


def knowledge(pack_id: str | None = None) -> SectorKnowledge:
    """The registered knowledge of a pack, or the default when no pack is named."""
    wanted = pack_id or default_pack_id()
    if wanted is None:
        raise SectorError('no default sector knowledge is registered (regchain.compat registers it on import)')
    if wanted not in _REGISTRY:
        raise SectorError(f'sector knowledge {wanted} is not registered; registered: {sorted(_REGISTRY)}')
    return _REGISTRY[wanted]

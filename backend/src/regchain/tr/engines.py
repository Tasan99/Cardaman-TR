"""Sector engines over shared expert services: one engine per sector pack a profile falls under, one set of services.

A profile is read by the packs the registry selects for it (routing.select_packs: an integrated brewer with an
alcohol-free line falls under both the alcoholic and the non-alcoholic pack). Each SectorEngine reads the
regulations its pack carries - its own and its modules', limited to the pack's source layers - extracts their
obligations, routes them to the profile's targets and assesses them against the register. What the engines share is
the ExpertServices: the corpus, the registry, the recorded similarities, the adjudicator (the strong model, or a replay
of its records) and the verification of every decision against its sources. Nothing here changes what the single
functions do (extraction.extract_regulation, extraction.route, adjudicate.assess_profile); it only says which engine
read which duty, and keeps one run's numbers per engine.
"""
from pathlib import Path

from ..pilot.schema import Strict
from .adjudicate import Adjudicator, assess_profile, load_adjudications
from .compare import GapReport, Register, load_register
from .corpus import CorpusStore
from .extraction import ExtractedObligation, extract_regulation
from .packs import Registry
from .profile import EnterpriseProfile
from .routing import PackSelection, select_packs
from .semantic import SimilarityTable

ENGINES_VERSION = 'tr-engines-v1'


class ExpertServices:
    """What every sector engine uses and none owns: the texts, the vocabulary, the recorded similarities, the
    adjudicator and the verification. One instance per run."""

    def __init__(self, registry: Registry | None = None, store: CorpusStore | None = None, table: SimilarityTable | None = None,
                 adjudicator: Adjudicator | None = None):
        self.registry = registry or Registry.load()
        self.store = store or CorpusStore()
        self.table = table
        self.adjudicator = adjudicator

    @classmethod
    def recorded(cls, similarities: Path | None = None, adjudications: list[Path] = (), registry=None, store=None) -> 'ExpertServices':
        """Services that replay a recorded run: no model is asked."""
        records = {}
        for path in adjudications:
            records.update(load_adjudications(path)[1])
        table = SimilarityTable.load(similarities) if similarities else None
        return cls(registry, store, table, Adjudicator(recorded=records))


class EngineRun(Strict):
    pack_id: str
    selection: PackSelection
    regulation_ids: list[str]
    obligations: int
    statistics: dict
    summary: dict


class SectorEngine:
    """One pack's reading of one profile."""

    def __init__(self, pack_id: str, services: ExpertServices):
        self.pack_id = pack_id
        self.services = services
        self.pack = services.registry.pack(pack_id)
        self.regulation_ids = [rid for rid in services.registry.regulation_ids(pack_id) if services.store.versions(rid)]
        self._obligations: list[ExtractedObligation] | None = None

    def obligations(self) -> list[ExtractedObligation]:
        if self._obligations is None:
            self._obligations = [o for rid in self.regulation_ids
                                 for o in extract_regulation(rid, self.services.registry, self.services.store)[1]]
        return self._obligations

    def assess(self, profile: EnterpriseProfile, register: Register, selection: PackSelection) -> tuple[GapReport, list, EngineRun]:
        """(gap report, assessments, run record) of the profile against the register, for this pack's duties."""
        report, assessments, statistics = assess_profile(profile, self.obligations(), register, self.services.registry, self.services.store,
                                                         self.services.table, self.services.adjudicator)
        run = EngineRun(pack_id=self.pack_id, selection=selection, regulation_ids=self.regulation_ids,
                        obligations=len({o.obligation_id for o in self.obligations()}), statistics=statistics, summary=report.summary)
        return report, assessments, run


def engines_for(profile: EnterpriseProfile, services: ExpertServices) -> list[tuple[SectorEngine, PackSelection]]:
    """The engines a profile is read by: one per pack the registry selects for it. A pack whose selection is UNKNOWN
    (an incomplete product list) is read too and says so in its selection; a pack not selected is not."""
    out = []
    for selection in select_packs(profile, services.registry):
        if selection.status in ('SELECTED', 'UNKNOWN'):
            out.append((SectorEngine(selection.pack_id, services), selection))
    return out


def assess_by_engine(profile: EnterpriseProfile, services: ExpertServices, register: Register | None = None) -> dict:
    """Every engine of the profile on the register: {'engines': [EngineRun...], 'reports': {pack_id: GapReport},
    'assessments': {pack_id: [...]}}. A regulation two packs both carry is read by both; the rows say which pack."""
    register = register or load_register(profile.profile_id)
    runs, reports, assessments = [], {}, {}
    for engine, selection in engines_for(profile, services):
        report, items, run = engine.assess(profile, register, selection)
        runs.append(run)
        reports[engine.pack_id] = report
        assessments[engine.pack_id] = items
    return {'profile_id': profile.profile_id, 'engines': runs, 'reports': reports, 'assessments': assessments, 'version': ENGINES_VERSION}

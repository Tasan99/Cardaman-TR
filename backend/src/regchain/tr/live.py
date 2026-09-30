"""Live model runs on the corpus, recorded so that every comparison afterwards replays stored answers.

Two runs, both through the existing engine and the local models the ModelRouter names (nothing here talks to a model
directly, and nothing leaves the machine):

  read     each clause is put to the extraction model (extraction.pipeline.extract: schema, span repair, grounding and
           review gates included) and the result is written as one JSON line: the input, the status and reason, the
           grounded candidates, the diagnostics. ai.py compares these readings with the rule frames element by element.
  analyze  one legal entity of an enterprise profile is projected to the engine's Company and analysed against the
           policies in force for it (pilot.engine.analyze, the v19 coverage pipeline when the environment selects it).
           The packet is saved whole; ai.py joins its rows to the extracted obligations by their quotes.

A run is evidence of what a model answered on a given day with given inputs; it is never a label.
"""
import json
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from ..extraction.pipeline import extract
from ..extraction.providers import runtime_manifest
from ..model_router import ModelRouter, ModelScope, bind_model_scope
from .compare import REGISTERS, load_register
from .corpus import CorpusStore
from .frames import frames_of
from .packs import Registry
from .profile import policy_covers, to_company

READ_FORMAT = 'cardaman-tr-clause-readings/1'


def clause_inputs(selection: dict, registry: Registry, store: CorpusStore) -> list[dict]:
    """The clauses of the selected articles ({regulation id: [paragraph numbers]}) with the text a model is given: a
    bent is read with the lead-in and the closing predicate of its fıkra, as a person would read it."""
    out = []
    for regulation_id, articles in selection.items():
        version = store.head(regulation_id)
        wanted = {str(a) for a in articles}
        for section in store.sections(regulation_id):
            if section['paragraph_number'] not in wanted or 'DELETED_PROVISION' in section['quality_flags']:
                continue
            for frame in frames_of(section, regulation_id):
                text = ' '.join(part for part in (frame.chapeau, frame.text, frame.closing) if part)
                out.append({'ref': frame.ref, 'regulation_id': regulation_id, 'version_id': version.version_id, 'label': frame.label,
                            'heading': frame.heading, 'input': text, 'input_sha256': sha256(text.encode('utf-8')).hexdigest(),
                            'clause': frame.text, 'chapeau': frame.chapeau, 'closing': frame.closing})
    return out


def read_clauses(inputs: list[dict], provider, out_path: Path, progress=None) -> dict:
    """Put every clause to the extraction model; one JSON line per clause, appended as it is answered."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = {'format': READ_FORMAT, 'started_at': datetime.now(timezone.utc).isoformat(), 'runtime': runtime_manifest(provider),
              'clauses': len(inputs)}
    started = time.monotonic()
    with out_path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + '\n')
        for position, item in enumerate(inputs):
            if progress is not None:
                progress(position, len(inputs), item['ref'])
            began = time.monotonic()
            try:
                result = extract(item['input'], provider)
                record = {'status': result.output.status, 'reason': result.reason, 'attempts': result.attempts,
                          'candidates': [c.model_dump(mode='json') for c in result.output.obligations],
                          'diagnostics': [d for d in result.diagnostics if isinstance(d, dict)]}
            except Exception as exc:                        # a provider failure is a recorded outcome, not a crash of the run
                record = {'status': 'FAILED', 'reason': f'{type(exc).__name__}: {exc}'[:400], 'attempts': 0, 'candidates': [],
                          'diagnostics': []}
            record = {**item, **record, 'elapsed_ms': int((time.monotonic() - began) * 1000)}
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            handle.flush()
    return {**header, 'elapsed_s': round(time.monotonic() - started, 1), 'path': str(out_path)}


def load_readings(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines or lines[0].get('format') != READ_FORMAT:
        raise ValueError(f'{path} is not a {READ_FORMAT} file')
    return lines[0], lines[1:]


def policies_in_force(profile, entity_id: str, register) -> list[Path]:
    """The register documents that are the entity's to compare, as files the engine's policy reader opens."""
    root = REGISTERS / profile.profile_id
    return [root / d.file for d in register.documents
            if policy_covers(d.scope, profile, 'LEGAL_ENTITY', entity_id)['covers'] == 'YES']


def analyze_entity(profile, entity_id: str, regulation_id: str, articles, registry: Registry, store: CorpusStore, out_path: Path,
                   router: ModelRouter, retrieval: str = 'hybrid', progress=None) -> dict:
    """One engine analysis: the entity against the named articles of one stored regulation and its policies in force."""
    from ..pilot.engine import analyze, pipeline_settings
    from ..pilot.policies import read_policy
    register = load_register(profile.profile_id)
    company = to_company(profile, entity_id, registry.vocabulary)
    sections = [{k: v for k, v in row.items() if k != 'lines'} for row in store.sections(regulation_id)]
    wanted = {str(a) for a in articles}
    labels = [s['printed_label'] for s in sections if s['paragraph_number'] in wanted and 'DELETED_PROVISION' not in s['quality_flags']]
    files = policies_in_force(profile, entity_id, register)
    policies = [read_policy(path) for path in files]
    provider = router.extraction_provider()
    embedder = router.embedder() if retrieval == 'hybrid' else None
    started = time.monotonic()
    packet = analyze(company, policies, sections, provider, labels, embedder=embedder, judge=router.judge_provider(),
                     progress=progress, settings=pipeline_settings())
    record = {'format': 'cardaman-tr-engine-run/1', 'profile_id': profile.profile_id, 'entity_id': entity_id,
              'regulation_id': regulation_id, 'version_id': store.head(regulation_id).version_id, 'articles': sorted(wanted),
              'policy_files': [p.name for p in files], 'elapsed_s': round(time.monotonic() - started, 1), 'packet': packet}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, default=str)
    return {k: v for k, v in record.items() if k != 'packet'} | {'path': str(out_path), 'rows': len(packet['events'][0]['payload']['obligations'])}


def scope(company_id: str):
    """The model scope a local run is bound to (development mode)."""
    return bind_model_scope(ModelScope('local', company_id, uuid4().hex))

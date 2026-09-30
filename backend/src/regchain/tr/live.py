"""Live model runs on the corpus, recorded so that every comparison afterwards replays stored answers.

Two runs, both through the existing engine and the local models the ModelRouter names (nothing here talks to a model
directly, and nothing leaves the machine):

  read     each reading unit of the selected articles is put to the extraction model (extraction.pipeline.extract:
           schema, span repair, grounding and review gates included), after the same deterministic pre-classification
           and with the same reference context the engine gives a sub-paragraph. One JSON line per unit: the input,
           the class, the status and reason, the grounded candidates, the diagnostics. ai.py compares these readings
           with the rule frames clause by clause and names the layer of every disagreement.
  analyze  one legal entity of an enterprise profile is projected to the engine's Company and analysed against the
           policies in force for it (pilot.engine.analyze, the v19 coverage pipeline when the environment selects it).
           The packet is saved whole; ai.py joins its rows to the extracted obligations by their quotes.

What a unit is decides what a small model can answer. Measured on 30 September 2026 (run 20260930-beverage-live-2):
the engine cuts an article at numbered "(1) (2)" sub-paragraphs only, so Kanun 4250 md. 6 (eleven unnumbered
paragraphs, 4,171 characters, some thirty duty markers) went to the model as one unit and came back with no obligation
at all (GROUNDING_REJECTED); and a bent read together with a lead-in that carries its own duty word ("... aşağıdaki
kurallara da uyulmalıdır: a) ...") was rejected for the duty word the answer did not cover. Neither is a model failure:
the unit was wrong. units_of cuts where Turkish drafting cuts (clauses.py): a sentence is a unit, read without the
lead-in of its fıkra; an item that has no predicate of its own is read with its lead-in and closing line, one item at
a time. (Read as one unit, a whole list fails whole: in run 20260930-beverage-live-3 thirteen label items of
Yönetmelik 23282 md. 9 were lost together because two of them cite another article.)

A run is evidence of what a model answered on a given day with given inputs; it is never a label.
"""
import json
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from ..extraction.classify import classify
from ..extraction.grounding import MODAL
from ..extraction.pipeline import extract
from ..extraction.providers import runtime_manifest
from ..extraction.retrieval import retrieve
from ..model_router import ModelRouter, ModelScope, bind_model_scope
from .clauses import NOTE, split_clauses
from .compare import REGISTERS, load_register
from .corpus import CorpusStore
from .frames import LETTERS, frames_of
from .packs import Registry
from .profile import policy_covers, to_company

READ_FORMAT = 'cardaman-tr-clause-readings/2'
UNIT_RULES_VERSION = 'tr-units-v2'
# What a sentence unit drops from its front and its end: the paragraph number, the bent letter and editorial notes.
LEAD_IN = re.compile(r'(?:\s*(?:\(\d{1,2}\)|[%s]{1,2}\)|%s))+\s*' % (LETTERS, NOTE.pattern))
TRAILING_NOTE = re.compile(r'(?:\s*%s)+\s*$' % NOTE.pattern)
# The reference context of one unit, as the engine budgets it for a sub-paragraph (pilot.engine.CONTEXT_CHARS).
CONTEXT_CHARS = 20000


def units_of(section: dict, regulation_id: str, version_id: str = '') -> list[dict]:
    """The reading units of one article, in order; each made of exact spans of the article text ('segments').

    SENTENCE  one clause (a sentence of a fıkra or of a bent), without its number, letter and editorial notes: one span.
    ITEM      an item that has no predicate of its own, read with the lead-in and the closing line that carry it:
              "... aşağıdaki belgeleri almaları zorunludur: | Alkollü içkilerin perakende satışını yapabilmek için, ...",
              "Piyasaya sürenler; | ... kayıt olmakla ve talep edilen bilgi ve belgeleri vermekle, | yükümlüdürler."
    """
    if 'DELETED_PROVISION' in section.get('quality_flags', ()):
        return []
    text = section['text']
    clauses = split_clauses(section)
    frames = frames_of(section, regulation_id)
    units = []

    def trimmed(start, end):
        """The span without the number, letter and notes in front of it and the notes behind it."""
        lead = LEAD_IN.match(text, start, end)
        if lead:
            start = lead.end()
        tail = TRAILING_NOTE.search(text, start, end)
        if tail and tail.start() > start:
            end = tail.start()
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    for index, clause in enumerate(clauses):
        start, end = trimmed(clause.start, clause.end)
        if start >= end:
            continue
        carried = frames[index].marker.endswith(('(chapeau)', '(closing)'))
        segments = [(start, end)]
        if carried and clause.chapeau_start is not None:
            segments.insert(0, trimmed(clause.chapeau_start, clause.chapeau_start + len(clause.chapeau)))
        if carried and clause.closing_start is not None:
            segments.append((clause.closing_start, clause.closing_start + len(clause.closing)))
        body = ' '.join(text[a:b] for a, b in segments)
        units.append({'unit_id': clause.ref, 'mode': 'ITEM' if len(segments) > 1 else 'SENTENCE', 'refs': [clause.ref],
                      'regulation_id': regulation_id, 'version_id': version_id, 'label': section['printed_label'], 'heading': clause.heading,
                      'start': start, 'end': end, 'segments': [list(s) for s in segments], 'input': body,
                      'input_sha256': sha256(body.encode('utf-8')).hexdigest(),
                      'lead': '' if len(segments) > 1 else clause.chapeau, 'rules_version': UNIT_RULES_VERSION})
    return units


def select_units(selection: dict, registry: Registry, store: CorpusStore) -> list[dict]:
    """The reading units of the selected articles ({regulation id: [paragraph numbers]})."""
    out = []
    for regulation_id, articles in selection.items():
        version = store.head(regulation_id)
        wanted = {str(a) for a in articles}
        for section in store.sections(regulation_id):
            if section['paragraph_number'] in wanted:
                out += units_of(section, regulation_id, version.version_id)
    return out


def _plain(section: dict) -> dict:
    return {k: v for k, v in section.items() if k != 'lines'}


def read_units(units: list[dict], provider, store: CorpusStore, out_path: Path, progress=None) -> dict:
    """Put every unit to the extraction model the way the engine puts a sub-paragraph: classified by rule first (a
    definition, a scope statement, a delegation, a penalty, a permission or an exemption is recorded with its class and
    costs no model call; so does a unit with no duty marker at all), then read with the context its references name."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = {'format': READ_FORMAT, 'started_at': datetime.now(timezone.utc).isoformat(), 'runtime': runtime_manifest(provider),
              'units': len(units), 'unit_rules': UNIT_RULES_VERSION}
    started = time.monotonic()
    sections: dict[str, list[dict]] = {}
    with out_path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + '\n')
        for position, item in enumerate(units):
            if progress is not None:
                progress(position, len(units), item['unit_id'])
            began = time.monotonic()
            kind = classify(item['input'], item['heading'])
            record = {'classification': {'kind': kind.kind, 'marker': kind.marker, 'basis': kind.reason}, 'attempts': 0, 'candidates': [],
                      'diagnostics': [], 'context': None}
            if not kind.extractable:
                record.update(status='NOT_EXTRACTED', reason=f'CLASSIFIED_{kind.kind}')
            elif not MODAL.search(item['input']):
                record.update(status='NOT_EXTRACTED', reason='NO_DUTY_MARKER')
            else:
                try:
                    same = sections.setdefault(item['regulation_id'], [_plain(s) for s in store.sections(item['regulation_id'], item['version_id'] or None)])
                    article = next(s for s in same if s['printed_label'] == item['label'])
                    context = retrieve(dict(article, text=item['input']), same, max_chars=CONTEXT_CHARS)
                    result = extract(item['input'], provider, context)
                    record.update(status=result.output.status, reason=result.reason, attempts=result.attempts,
                                  candidates=[c.model_dump(mode='json') for c in result.output.obligations],
                                  diagnostics=[d for d in result.diagnostics if isinstance(d, dict)],
                                  context={k: v for k, v in context.manifest().items() if k != 'items'} | {
                                      'items': [{'printed_label': i['printed_label'], 'reason': i['reason'], 'required': i['required']}
                                                for i in context.items]})
                except Exception as exc:                    # a provider failure is a recorded outcome, not a crash of the run
                    record.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}'[:400])
            record = {**item, **record, 'elapsed_ms': int((time.monotonic() - began) * 1000)}
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            handle.flush()
    return {**header, 'elapsed_s': round(time.monotonic() - started, 1), 'path': str(out_path)}


def load_readings(path: Path) -> tuple[dict, list[dict]]:
    """(header, records) of a readings file. A format-1 file (one record per clause, the lead-in joined to the clause
    and no pre-classification) is read as it was written: its records have 'ref' instead of 'refs'."""
    lines = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines or not str(lines[0].get('format', '')).startswith('cardaman-tr-clause-readings/'):
        raise ValueError(f'{path} is not a clause readings file')
    records = [{**r, 'refs': r.get('refs') or [r['ref']]} for r in lines[1:]]
    return lines[0], records


def policies_in_force(profile, entity_id: str, register) -> list[Path]:
    """The register documents that are the entity's to compare, as files the engine's policy reader opens."""
    root = REGISTERS / profile.profile_id
    return [root / d.file for d in register.documents
            if policy_covers(d.scope, profile, 'LEGAL_ENTITY', entity_id)['covers'] == 'YES']


def unit_sections(sections: list[dict], regulation_id: str, articles, version_id: str = '') -> tuple[list[dict], list[str], dict]:
    """(sections, target labels, {unit label: unit}) for an engine run that reads the selected articles unit by unit.

    Each unit becomes a provision of its own, placed right after its article: its label is the place of its clause, its
    text the unit's reading. The article itself stays in the list, so a reference to it by number still resolves to one
    text (a unit is not a 'mevzuat_madde' and is never what an article number means)."""
    wanted = {str(a) for a in articles}
    out, labels, by_label = [], [], {}
    for section in sections:
        plain = _plain(section)
        out.append(plain)
        if section['paragraph_number'] not in wanted or 'DELETED_PROVISION' in section['quality_flags']:
            continue
        for item in units_of(section, regulation_id, version_id):
            label = item['refs'][0]
            if ' ' in label.split(' md. ', 1)[-1]:
                continue                                     # "md. Geçici 1": a label the engine's chapter reader does not take
            out.append({**plain, 'id': sha256(f"{plain['id']}|{label}".encode('utf-8')).hexdigest(), 'printed_label': label,
                        'text': item['input'], 'content_hash': sha256(item['input'].encode('utf-8')).hexdigest(),
                        'locator_kind': 'mevzuat_birim'})
            labels.append(label)
            by_label[label] = item
    return out, labels, by_label


def analyze_entity(profile, entity_id: str, regulation_id: str, articles, registry: Registry, store: CorpusStore, out_path: Path,
                   router: ModelRouter, retrieval: str = 'hybrid', progress=None, units: bool = True) -> dict:
    """One engine analysis: the entity against the named articles of one stored regulation and its policies in force.
    units=True hands the engine the reading units of those articles (units_of); False the articles as they are stored."""
    from ..pilot.engine import analyze, pipeline_settings
    from ..pilot.policies import read_policy
    register = load_register(profile.profile_id)
    company = to_company(profile, entity_id, registry.vocabulary)
    version = store.head(regulation_id)
    sections = [_plain(row) for row in store.sections(regulation_id)]
    wanted = {str(a) for a in articles}
    by_label = {}
    if units:
        sections, labels, by_label = unit_sections(store.sections(regulation_id), regulation_id, articles, version.version_id)
    else:
        labels = [s['printed_label'] for s in sections if s['paragraph_number'] in wanted and 'DELETED_PROVISION' not in s['quality_flags']]
    files = policies_in_force(profile, entity_id, register)
    policies = [read_policy(path) for path in files]
    provider = router.extraction_provider()
    embedder = router.embedder() if retrieval == 'hybrid' else None
    started = time.monotonic()
    packet = analyze(company, policies, sections, provider, labels, embedder=embedder, judge=router.judge_provider(),
                     progress=progress, settings=pipeline_settings())
    record = {'format': 'cardaman-tr-engine-run/2', 'profile_id': profile.profile_id, 'entity_id': entity_id,
              'regulation_id': regulation_id, 'version_id': version.version_id, 'articles': sorted(wanted),
              'units': {label: {'refs': item['refs'], 'mode': item['mode'], 'segments': item['segments'], 'label': item['label']} for label, item in by_label.items()} if units else None,
              'policy_files': [p.name for p in files], 'elapsed_s': round(time.monotonic() - started, 1), 'packet': packet}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, default=str)
    return {k: v for k, v in record.items() if k not in ('packet', 'units')} | {
        'path': str(out_path), 'targets': len(labels), 'rows': len(packet['events'][0]['payload']['obligations'])}


def scope(company_id: str):
    """The model scope a local run is bound to (development mode)."""
    return bind_model_scope(ModelScope('local', company_id, uuid4().hex))

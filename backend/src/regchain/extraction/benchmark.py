"""Measure a provider on the ingested consolidated sources, not on nine synthetic cases.

This reports how a model *behaves* on real FCA provisions: how often it produces a
candidate, how often the grounding gate rejects it, how often its own second pass
withdraws one, and where it disagrees with the deterministic baseline.

It does NOT report legal accuracy. Precision and recall against real law need an
expert-labelled set, which does not exist here. A high candidate rate is not a good
result and a low one is not a bad one; what a reviewer can act on is the disagreement
list and the rejection rates.

The benchmark is read-only. Persisted extraction runs come from `extraction.cli`, so
a measurement never enters the audit trail as if it were an analysis of record.
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from regchain.evidence import digest
from .pipeline import extract
from .providers import configured_provider, runtime_manifest
from .grounding import preflight
from .retrieval import retrieve
from .rules import RulesProvider
from .schema import SCHEMA_VERSION
from .service import CORPUS_QUERY, PROMPT_HASH, pinned_corpus

DOCUMENTS = '''SELECT DISTINCT ON (v.regulation_id) v.id, r.external_key, v.normalized_hash
    FROM regulation_versions v JOIN regulations r ON r.id = v.regulation_id
    WHERE EXISTS (SELECT 1 FROM regulation_sections s
                  WHERE s.version_id = v.id AND s.source_kind = 'CONSOLIDATED')
    ORDER BY v.regulation_id, v.version_no DESC'''

SECTIONS = '''SELECT * FROM regulation_sections WHERE version_id=%s AND source_kind='CONSOLIDATED'
    ORDER BY ordinal NULLS LAST,section_number,paragraph_number'''


def identity(candidate) -> tuple:
    return (candidate.subject, candidate.modality,
            candidate.required_action or candidate.prohibited_action)


def measure(conn: psycopg.Connection, provider, *, limit: int | None = None) -> dict:
    if limit is not None and limit<1:
        raise ValueError('limit must be positive')
    started = time.monotonic()
    baseline = RulesProvider()
    outcomes: Counter = Counter()
    by_type: dict[str, Counter] = {}
    agreement: Counter = Counter()
    disagreements: list[dict] = []
    dataset: list[dict] = []
    corpus_versions: set[str] = set()
    evaluated = complete = total = 0
    pool=[]
    selected_manifest=[]
    details=[]
    with conn.cursor(row_factory=dict_row) as cur:
        documents = cur.execute(DOCUMENTS).fetchall()
        for document in documents:
            cur.execute(SECTIONS, (document['id'],))
            sections = cur.fetchall()
            if not sections:
                continue
            cur.execute(CORPUS_QUERY, (document['id'],))
            corpus = cur.fetchall()
            corpus_versions.update(source['version_id'] for source in pinned_corpus(corpus))
            dataset.append({'external_key': document['external_key'],
                            'version_id': str(document['id']),
                            'normalized_hash': document['normalized_hash'],
                            'paragraphs': len(sections)})
            for section in sections:
                total += 1
                packet = retrieve(section, sections, corpus=corpus)
                if preflight(section['text'],packet):
                    continue
                complete += 1
                pool.append((document['external_key'],section,packet))
        # Interleave documents rather than taking all examples from the first UUID.
        groups={}
        for key,section,packet in sorted(pool,key=lambda row:(row[0],row[1]['printed_label'] or '',str(row[1]['id']))):
            groups.setdefault(key,[]).append((section,packet))
        ordered=[]
        while any(groups.values()):
            for key in sorted(groups):
                if groups[key]:
                    section,packet=groups[key].pop(0)
                    ordered.append((key,section,packet))
        for key,section,packet in ordered[:limit] if limit is not None else ordered:
                evaluated += 1
                selected_manifest.append({'external_key':key,'section_id':str(section['id']),
                    'version_id':str(section['version_id']),'printed_label':section['printed_label'],
                    'source_hash':section['content_hash'],'context_hash':digest(packet.manifest())})
                result = extract(section['text'], provider, packet)
                reference = extract(section['text'], baseline, packet)
                legal_type = section['legal_type'] or 'UNKNOWN'
                outcomes[result.reason] += 1
                details.append({'section_id':str(section['id']),'printed_label':section['printed_label'],
                    'reason':result.reason,'candidate_count':len(result.output.obligations),
                    'attempts':result.attempts,'diagnostics':list(result.diagnostics)})
                by_type.setdefault(legal_type, Counter())[result.reason] += 1
                model = {identity(c) for c in result.output.obligations}
                rules = {identity(c) for c in reference.output.obligations}
                if model and rules:
                    agreement['both_extracted'] += 1
                    agreement['identical' if model == rules else 'different'] += 1
                    if model != rules:
                        disagreements.append({
                            'printed_label': section['printed_label'],
                            'legal_type': legal_type,
                            'model': sorted(list(pair) for pair in model),
                            'baseline': sorted(list(pair) for pair in rules)})
                elif model:
                    agreement['model_only'] += 1
                    disagreements.append({'printed_label':section['printed_label'],'kind':'model_only'})
                elif rules:
                    agreement['baseline_only'] += 1
                    disagreements.append({'printed_label':section['printed_label'],'kind':'baseline_only'})
                else:
                    agreement['both_abstained'] += 1
    ratio = lambda numerator: numerator / evaluated if evaluated else None
    return {
        'measured_at': datetime.now(timezone.utc).date().isoformat(),
        'provider': provider.name, 'model_version': provider.model_version,
        'prompt_hash': PROMPT_HASH, 'schema_version': SCHEMA_VERSION,
        'baseline_version': baseline.model_version,'runtime_settings':runtime_manifest(provider),
        'sample_manifest':selected_manifest,'sample_hash':digest(selected_manifest),'details':details,
        'dataset': {'documents': len(dataset), 'paragraphs': total,
                    'corpus_versions': len(corpus_versions),
                    'dataset_hash': digest(dataset), 'sources': dataset},
        'paragraphs': {'consolidated': total, 'context_complete': complete, 'evaluated': evaluated},
        'outcomes': dict(outcomes),
        'rates': {
            'candidate_rate': ratio(outcomes.get('CANDIDATE_REQUIRES_LEGAL_REVIEW', 0)),
            'grounding_rejection_rate': ratio(outcomes.get('GROUNDING_REJECTED', 0)),
            'second_pass_rejection_rate': ratio(outcomes.get('SECOND_PASS_UNCERTAIN', 0)),
            'provider_failure_rate': ratio(outcomes.get('PROVIDER_FAILURE', 0)),
        },
        'by_legal_type': {name: dict(counts) for name, counts in sorted(by_type.items())},
        'baseline_agreement': dict(agreement),
        'disagreements': disagreements[:50],
        'seconds': round(time.monotonic() - started, 1),
        'limitations': 'Behavioural measurement on real sources. No expert labels, so no '
                       'precision, recall or legal-accuracy claim. Rates describe how the '
                       'provider behaves under the grounding gates, not whether its readings '
                       'of the law are correct. Every candidate still requires review.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure a provider on ingested consolidated sources (review only, no legal accuracy)')
    parser.add_argument('--provider', choices=['rules', 'ollama'], required=True)
    parser.add_argument('--limit', type=int, help='Evaluate at most this many context-complete paragraphs')
    parser.add_argument('--output', help='Write the report to this path as well as stdout')
    args = parser.parse_args()
    try:
        provider = configured_provider(args.provider)
        with psycopg.connect(host=os.getenv('PGHOST', 'db'), dbname=os.getenv('PGDATABASE', 'regchain'),
                             port=os.getenv('PGPORT', '5432'), user='regchain_extract',
                             password=os.environ['EXTRACT_DB_PASSWORD'], connect_timeout=5) as conn:
            report = measure(conn, provider, limit=args.limit)
    except (ValueError, KeyError, psycopg.Error) as exc:
        parser.exit(1, (str(exc) if isinstance(exc, ValueError)
                        else 'Benchmark database/configuration failure; check migrations and credentials') + '\n')
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()

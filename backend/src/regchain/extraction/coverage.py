"""Which consolidated sources the ingested corpus still needs.

An unresolved reference is not a model failure. It usually means the cited chapter has
not been ingested yet. This report names each missing chapter and the official URL that
would close it, so source coverage is an ingestion worklist rather than a silent gap.
"""
import argparse
import json
import os
from collections import Counter

import psycopg
from psycopg.rows import dict_row

from regchain.ingestion.sources import chapter_of, chapter_key, chapter_url
from .retrieval import retrieve
from .service import CORPUS_QUERY

LATEST = '''SELECT DISTINCT ON (v.regulation_id) v.id, r.external_key, v.version_no
            FROM regulation_versions v JOIN regulations r ON r.id = v.regulation_id
            ORDER BY v.regulation_id, v.version_no DESC'''


def report(conn: psycopg.Connection) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        versions = cur.execute(LATEST).fetchall()
        documents = []
        missing: Counter = Counter()
        ambiguous: Counter = Counter()
        unidentified: Counter = Counter()
        for version in versions:
            cur.execute('''SELECT * FROM regulation_sections WHERE version_id=%s
                           ORDER BY ordinal NULLS LAST,section_number,paragraph_number''', (version['id'],))
            sections = cur.fetchall()
            if not sections:
                continue
            cur.execute(CORPUS_QUERY, (version['id'],))
            corpus = cur.fetchall()
            blocked = Counter()
            for section in sections:
                # Redline never becomes an operative rule, so its context completeness
                # is not an ingestion problem to fix. Counting it with the rest would
                # send an operator after sources that no extraction will ever use.
                if section['source_kind'] == 'AMENDMENT':
                    blocked['amendment_not_extracted'] += 1
                    continue
                packet = retrieve(section, sections, corpus=corpus)
                for label in packet.missing_sources:
                    missing[chapter_of(label) or label] += 1
                for reference in packet.unresolved:
                    if '[ambiguous' in reference:
                        ambiguous[reference.split(' [')[0]] += 1
                    elif '[unidentified target]' in reference:
                        unidentified[reference.split(' [')[0].lower()] += 1
                if packet.missing_sources:
                    blocked['missing_source'] += 1
                elif packet.unresolved:
                    blocked['unresolved_reference'] += 1
                elif packet.truncated:
                    blocked['context_budget'] += 1
                else:
                    blocked['context_complete'] += 1
            documents.append({'external_key': version['external_key'], 'version_no': version['version_no'],
                              'paragraphs': len(sections), 'corpus_paragraphs': len(corpus),
                              'paragraph_context': dict(blocked)})
    worklist = []
    for chapter, count in missing.most_common():
        url = chapter_url(chapter)
        worklist.append({'chapter': chapter, 'references': count,
                         'external_key': chapter_key(chapter) if url else None, 'url': url,
                         'action': 'ingest' if url else 'manual review: locator is not a Handbook chapter'})
    return {'documents': documents, 'missing_chapters': worklist,
            'ambiguous_references': [{'reference': k, 'occurrences': v} for k, v in ambiguous.most_common()],
            'unidentified_cues': [{'cue': k, 'occurrences': v} for k, v in unidentified.most_common()],
            'limitations': 'Reference coverage only. It does not measure legal accuracy, '
                           'applicability or extraction quality, and a complete context '
                           'packet is not a claim that the obligation was read correctly.'}


def main() -> None:
    parser = argparse.ArgumentParser(description='Report consolidated sources the corpus still needs')
    parser.parse_args()
    try:
        with psycopg.connect(host=os.getenv('PGHOST', 'db'), dbname=os.getenv('PGDATABASE', 'regchain'),
                             port=os.getenv('PGPORT', '5432'), user='regchain_extract',
                             password=os.environ['EXTRACT_DB_PASSWORD'], connect_timeout=5) as conn:
            print(json.dumps(report(conn), ensure_ascii=False, indent=2))
    except (KeyError, psycopg.Error) as exc:
        parser.exit(1, 'Coverage report failed; check migrations and extraction credentials\n'
                    if isinstance(exc, psycopg.Error) else 'EXTRACT_DB_PASSWORD is not configured\n')


if __name__ == '__main__':
    main()

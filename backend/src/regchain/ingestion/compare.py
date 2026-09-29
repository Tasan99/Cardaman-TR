"""Separate a regulator's change from our own parser's change.

Ingestion refuses to emit a paragraph diff when a document's bytes and the parser
changed between two stored versions: such a diff would mix a legal change with a
processing change, and a reviewer could not tell which caused what.

This command resolves that state. It reparses the earlier version's retained bytes
with the current parser and diffs the result against the later version's stored
paragraphs. Both sides then share one parser, so whatever remains is attributable to
the source. Nothing is refetched: the comparison uses the bytes already archived.
"""
import argparse
import json
import os
import sys
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from .diff import paragraph_diff
from .models import Download, IngestionError, NORMALIZER_VERSION, PARSER_VERSION, Paragraph
from .worker import bounded_parse

# The most recent earlier version whose archived bytes actually differ. Parser-only
# reprocessing versions in between are skipped: they answer nothing about the source.
EARLIER = '''SELECT * FROM regulation_versions WHERE regulation_id=%s
    AND version_no < %s AND raw_hash <> %s ORDER BY version_no DESC LIMIT 1'''

# Latest version per regulation, produced by the current parser, whose byte change
# has not yet been compared like for like.
PENDING = '''SELECT v.id FROM regulation_versions v
    WHERE v.version_no = (SELECT max(version_no) FROM regulation_versions
                          WHERE regulation_id = v.regulation_id)
      AND v.parser_version = %(parser)s AND v.normalizer_version = %(normalizer)s
      AND EXISTS (SELECT 1 FROM regulation_versions p WHERE p.regulation_id = v.regulation_id
                  AND p.version_no < v.version_no AND p.raw_hash <> v.raw_hash)
      AND NOT EXISTS (SELECT 1 FROM source_comparisons c
                      WHERE c.new_version_id = v.id AND c.parser_version = %(parser)s
                        AND c.normalizer_version = %(normalizer)s)
    ORDER BY v.created_at'''


def stored_paragraphs(rows: list[dict]) -> tuple[Paragraph, ...]:
    return tuple(Paragraph(row['section_number'], row['paragraph_number'], row['text'],
                           tuple(row['heading_path']), row['page'], row['locator_kind'],
                           row['page_end'], row['printed_label'], row['source_kind'],
                           tuple(row['quality_flags']), tuple(row['source_spans']),
                           row['legal_type'], row['effective_from'], tuple(row['defined_terms']))
                 for row in rows)


def compare_version(conn: psycopg.Connection, version_id: UUID) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM regulation_versions WHERE id=%s', (version_id,))
        new = cur.fetchone()
        if not new:
            raise ValueError('Regulatory version not found')
        # Holding the parser constant means both sides must be the current parser.
        # A later version stored under an older parser has to be reparsed first.
        if new['parser_version'] != PARSER_VERSION or new['normalizer_version'] != NORMALIZER_VERSION:
            raise ValueError('This version was produced by a different parser; run ingest reparse '
                             'before comparing, so both sides share one parser')
        cur.execute(EARLIER, (new['regulation_id'], new['version_no'], new['raw_hash']))
        old = cur.fetchone()
        if not old:
            raise ValueError('No earlier version of this document has different archived bytes')
        cur.execute('SELECT body,media_type FROM regulatory_source_objects WHERE raw_hash=%s',
                    (old['raw_hash'],))
        archived = cur.fetchone()
        if not archived:
            raise ValueError('The previous version has no archived bytes; nothing can be reparsed')
        cur.execute('''SELECT * FROM regulation_sections WHERE version_id=%s
                       ORDER BY ordinal NULLS LAST,section_number,paragraph_number''', (version_id,))
        sections = cur.fetchall()
        if not sections:
            raise ValueError('Version has no stored paragraphs')
    # Reparsing happens outside any transaction, in the bounded subprocess parser.
    source = Download(old['source_url'], old['source_url'], bytes(archived['body']),
                      archived['media_type'], old['retrieved_at'],
                      {'processing': 'archived-source-comparison'})
    reparsed = bounded_parse(source)
    changes = paragraph_diff(reparsed.paragraphs, stored_paragraphs(sections))
    verdict = 'SOURCE_CHANGED' if changes else 'SOURCE_UNCHANGED'
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute('''INSERT INTO source_comparisons(regulation_id,old_version_id,new_version_id,
            parser_version,normalizer_version,old_raw_hash,new_raw_hash,old_reparsed_hash,
            new_normalized_hash,changed_paragraphs,verdict)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(old_version_id,new_version_id,parser_version,normalizer_version)
            DO NOTHING RETURNING id''',
            (new['regulation_id'], old['id'], new['id'], PARSER_VERSION, NORMALIZER_VERSION,
             old['raw_hash'], new['raw_hash'], reparsed.content_hash, new['normalized_hash'],
             len(changes), verdict))
        row = cur.fetchone()
        if row is None:
            return {'version_id': str(version_id), 'status': 'UNCHANGED',
                    'detail': 'This comparison already exists for the current parser'}
        comparison_id = row['id']
        for change in changes:
            old_paragraph = reparsed.paragraphs[change.old_index] if change.old_index is not None else None
            new_section = sections[change.new_index] if change.new_index is not None else None
            cur.execute('''INSERT INTO source_comparison_changes(comparison_id,kind,old_locator,
                old_text,new_version_id,new_section_id,diff_text) VALUES(%s,%s,%s,%s,%s,%s,%s)''',
                (comparison_id, change.kind,
                 f'{old_paragraph.section}/{old_paragraph.number}' if old_paragraph else None,
                 old_paragraph.text if old_paragraph else None,
                 version_id if new_section else None,
                 new_section['id'] if new_section else None, change.diff_text))
    return {'comparison_id': str(comparison_id), 'regulation_id': str(new['regulation_id']),
            'old_version_id': str(old['id']), 'new_version_id': str(new['id']),
            'parser_version': PARSER_VERSION, 'old_raw_hash': old['raw_hash'],
            'new_raw_hash': new['raw_hash'], 'verdict': verdict,
            'changed_paragraphs': len(changes),
            'kinds': {kind: sum(1 for c in changes if c.kind == kind)
                      for kind in ('ADDED', 'REMOVED', 'MODIFIED')},
            'human_review_required': True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reparse archived bytes with one parser to tell a source change from a parser change')
    parser.add_argument('version', help='Version UUID, or pending for every unresolved simultaneous change')
    args = parser.parse_args()
    try:
        with psycopg.connect(host=os.getenv('PGHOST', 'db'), port=os.getenv('PGPORT', '5432'),
                             dbname=os.getenv('PGDATABASE', 'regchain'), user='regchain_ingest',
                             password=os.environ['INGEST_DB_PASSWORD'], connect_timeout=5,
                             autocommit=True) as conn:
            if args.version == 'pending':
                versions = [row[0] for row in
                            conn.execute(PENDING, {'parser': PARSER_VERSION, 'normalizer': NORMALIZER_VERSION}).fetchall()]
                if not versions:
                    print(json.dumps({'status': 'NOTHING_PENDING',
                                      'detail': 'No unresolved simultaneous source and parser change'}))
                    return
            else:
                versions = [UUID(args.version)]
            for version in versions:
                print(json.dumps(compare_version(conn, version), ensure_ascii=False))
    except (IngestionError, ValueError, KeyError, psycopg.Error) as exc:
        message = (str(exc) if isinstance(exc, (IngestionError, ValueError))
                   else 'Comparison database/configuration failure; check migrations and ingestion credentials')
        print(json.dumps({'status': 'FAILED', 'error': message}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()

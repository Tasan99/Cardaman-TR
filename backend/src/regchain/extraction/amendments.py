"""Link an amendment instrument's paragraphs to the consolidated provisions they cite.

A policy statement's appendix is a legal instrument: it says which module is amended,
by which annex, and from when. It does not contain the resulting law. This module
matches each amendment paragraph's printed locator to a provision in the pinned
consolidated corpus, so a change can be followed from the instrument to the text that
is in force today.

A link is evidence, never a legal conclusion. It states that the instrument cites a
locator that exists in the consolidated corpus. Whether the consolidated text is the
result of *this* instrument is only corroborated by comparing the instrument's
commencement date with the date printed against the provision:

MATCH       the provision carries this instrument's commencement date.
SUPERSEDED  the provision has a later date: it was amended again after this
            instrument, so the text in force today is not this instrument's output.
PREDATES    the provision has an earlier date: this instrument did not produce the
            current text. The annex most likely reproduced it as context.
UNKNOWN     one of the two dates is missing; nothing is inferred.

None of these states is a finding of legal causation, and a reviewer decides.
"""
import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from regchain.evidence import digest
from regchain.ingestion.sources import chapter_of
from .retrieval import BOOKS, admissible_reading, build_index, clean_label, reference_keys
from .service import CORPUS_QUERY, pinned_corpus

MATCHER_VERSION = 'printed-locator-v1'
# The instrument's own commencement clause, not a commentary paragraph about timing.
COMMENCEMENT = re.compile(r'this instrument comes into force on\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})', re.I)
ANNEX_SOURCEBOOK = re.compile(r'^Annex\s+([A-Z])\b.*?\bsourcebook\s*\((' + BOOKS + r')\)', re.I)
ANNEX_GLOSSARY = re.compile(r'^Annex\s+([A-Z])\b.*?\bGlossary\b', re.I)
ANNEX_HEADING = re.compile(r'^Annex\s+([A-Z])\b')
MONTHS = {name.lower(): number for number, name in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], 1)}


@dataclass(frozen=True)
class Analysis:
    commencement: date | None
    annexes: dict
    links: tuple[dict, ...]
    unmatched: tuple[dict, ...]

    def manifest(self) -> dict:
        return {'matcher_version': MATCHER_VERSION,
                'commencement': str(self.commencement) if self.commencement else None,
                'annexes': self.annexes, 'links': list(self.links), 'unmatched': list(self.unmatched)}


def commencement_date(sections: list[dict]) -> date | None:
    """Read the commencement clause verbatim; never infer a date from prose."""
    for section in sections:
        match = COMMENCEMENT.search(section['text'])
        if match and match[2].lower() in MONTHS:
            try:
                return date(int(match[3]), MONTHS[match[2].lower()], int(match[1]))
            except ValueError:
                return None
    return None


def annex_books(sections: list[dict]) -> dict:
    """Map each annex letter to the sourcebook its own heading names."""
    books: dict[str, str] = {}
    for section in sections:
        text = section['text'][:200]
        sourcebook = ANNEX_SOURCEBOOK.match(text)
        if sourcebook:
            books.setdefault(sourcebook[1].upper(), sourcebook[2].upper())
        elif ANNEX_GLOSSARY.match(text):
            books.setdefault(ANNEX_GLOSSARY.match(text)[1].upper(), 'GLOSSARY')
    return books


def annex_of(section: dict) -> str | None:
    for heading in reversed(section.get('heading_path') or []):
        match = ANNEX_HEADING.match(str(heading))
        if match:
            return match[1].upper()
    return None


def analyse(sections: list[dict], corpus: list[dict]) -> Analysis:
    commencement = commencement_date(sections)
    books = annex_books(sections)
    provisions, _ = build_index(corpus)
    links: list[dict] = []
    unmatched: list[dict] = []
    for section in sections:
        if section.get('source_kind') != 'AMENDMENT' or not section.get('printed_label'):
            continue
        annex = annex_of(section)
        book = books.get(annex or '')
        label = clean_label(section['printed_label'])
        # A locator with only a chapter and a section names a heading inside the
        # annex, not a provision. Linking it would assert a change to every
        # provision under it.
        if not book or book == 'GLOSSARY' or len(label.split('.')) < 3:
            unmatched.append({'section_id': str(section['id']), 'printed_label': section.get('printed_label'),
                              'annex': annex, 'book': book,
                              'reason': 'NO_ANNEX_SOURCEBOOK' if not book else
                                        'GLOSSARY_AMENDMENT_NOT_LINKED' if book == 'GLOSSARY' else
                                        'SECTION_HEADING_NOT_A_PROVISION'})
            continue
        matches: list[dict] = []
        for key, letter in reference_keys(book, label):
            for hit in provisions.get(key, ()):
                if admissible_reading(hit, letter) and hit not in matches:
                    matches.append(hit)
        locator = f'{book} {label}'
        if len(matches) != 1:
            unmatched.append({'section_id': str(section['id']), 'printed_label': section.get('printed_label'),
                              'annex': annex, 'book': book, 'locator': locator,
                              'reason': 'AMBIGUOUS_IN_CONSOLIDATED_CORPUS' if matches else
                                        'NOT_IN_CONSOLIDATED_CORPUS',
                              'chapter': chapter_of(locator)})
            continue
        provision = matches[0]
        effective = provision.get('effective_from')
        # Comparing the instrument's commencement with the date printed against the
        # provision separates three very different situations for a reviewer.
        agreement = ('UNKNOWN' if not commencement or not effective else
                     'MATCH' if effective == commencement else
                     'SUPERSEDED' if effective > commencement else 'PREDATES')
        links.append({'instrument_section_id': str(section['id']),
                      'consolidated_version_id': str(provision['version_id']),
                      'consolidated_section_id': str(provision['id']),
                      'locator': locator, 'printed_label': provision.get('printed_label'),
                      'legal_type': provision.get('legal_type', 'UNKNOWN'),
                      'consolidated_effective_from': str(effective) if effective else None,
                      'date_agreement': agreement})
    return Analysis(commencement, books, tuple(links), tuple(unmatched))


def run_version(conn: psycopg.Connection, version_id: UUID) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM regulation_versions WHERE id=%s', (version_id,))
        if not cur.fetchone():
            raise ValueError('Regulatory version not found')
        cur.execute('''SELECT * FROM regulation_sections WHERE version_id=%s
                       ORDER BY ordinal NULLS LAST,section_number,paragraph_number''', (version_id,))
        sections = cur.fetchall()
        if not sections:
            raise ValueError('Version has no source paragraphs')
        if not any(s['source_kind'] == 'AMENDMENT' for s in sections):
            raise ValueError('This version contains no amendment paragraphs to link')
        cur.execute(CORPUS_QUERY, (version_id,))
        corpus = cur.fetchall()
        if not corpus:
            raise ValueError('No consolidated sources ingested; run the handbook ingest first')
        pinned = pinned_corpus(corpus)
        analysis = analyse(sections, corpus)
        manifest = {'version_id': str(version_id), 'corpus': pinned,
                    'sections': [{'id': str(s['id']), 'hash': s['content_hash']} for s in sections]}
        input_hash = digest(manifest)
        find = '''SELECT id FROM amendment_runs WHERE instrument_version_id=%s
                  AND matcher_version=%s AND input_hash=%s ORDER BY created_at DESC LIMIT 1'''
        cur.execute(find, (version_id, MATCHER_VERSION, input_hash))
        existing = cur.fetchone()
        if existing:
            return {'run_id': str(existing['id']), 'version_id': str(version_id), 'status': 'UNCHANGED'}
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', (input_hash,))
        cur.execute(find, (version_id, MATCHER_VERSION, input_hash))
        existing = cur.fetchone()
        if existing:
            return {'run_id': str(existing['id']), 'version_id': str(version_id), 'status': 'UNCHANGED'}
        cur.execute('''INSERT INTO amendment_runs(instrument_version_id,matcher_version,input_hash,
            output_hash,status,commencement_date,corpus) VALUES(%s,%s,%s,%s,'REVIEW_REQUIRED',%s,%s)
            RETURNING id''', (version_id, MATCHER_VERSION, input_hash, digest(analysis.manifest()),
                              analysis.commencement, Jsonb(pinned)))
        run_id = cur.fetchone()['id']
        for link in analysis.links:
            cur.execute('''INSERT INTO amendment_links(run_id,instrument_version_id,instrument_section_id,
                consolidated_version_id,consolidated_section_id,locator,match_basis,commencement_date,
                consolidated_effective_from,date_agreement)
                VALUES(%s,%s,%s,%s,%s,%s,'PRINTED_LOCATOR',%s,%s,%s)''',
                (run_id, version_id, link['instrument_section_id'], link['consolidated_version_id'],
                 link['consolidated_section_id'], link['locator'], analysis.commencement,
                 date.fromisoformat(link['consolidated_effective_from'])
                 if link['consolidated_effective_from'] else None, link['date_agreement']))
        for entry in analysis.unmatched:
            cur.execute('''INSERT INTO amendment_link_gaps(run_id,instrument_version_id,
                instrument_section_id,locator,reason) VALUES(%s,%s,%s,%s,%s)''',
                (run_id, version_id, entry['section_id'], entry.get('locator'), entry['reason']))
    agreement = {value: sum(1 for link in analysis.links if link['date_agreement'] == value)
                 for value in ('MATCH', 'SUPERSEDED', 'PREDATES', 'UNKNOWN')}
    return {'run_id': str(run_id), 'version_id': str(version_id), 'status': 'REVIEW_REQUIRED',
            'matcher_version': MATCHER_VERSION, 'corpus_versions': len(pinned),
            'commencement_date': str(analysis.commencement) if analysis.commencement else None,
            'annexes': analysis.annexes, 'links': len(analysis.links),
            'unlinked': len(analysis.unmatched), 'date_agreement': agreement,
            'human_review_required': True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Link amendment instrument paragraphs to consolidated provisions (review only)')
    parser.add_argument('version', help='Version UUID, or latest for the newest version of each regulation')
    args = parser.parse_args()
    try:
        with psycopg.connect(host=os.getenv('PGHOST', 'db'), dbname=os.getenv('PGDATABASE', 'regchain'),
                             port=os.getenv('PGPORT', '5432'), user='regchain_extract',
                             password=os.environ['EXTRACT_DB_PASSWORD'], connect_timeout=5,
                             autocommit=True) as conn:
            if args.version == 'latest':
                versions = [row[0] for row in conn.execute(
                    '''SELECT DISTINCT ON(v.regulation_id) v.id FROM regulation_versions v
                       WHERE EXISTS (SELECT 1 FROM regulation_sections s
                                     WHERE s.version_id=v.id AND s.source_kind='AMENDMENT')
                       ORDER BY v.regulation_id, v.version_no DESC''').fetchall()]
                if not versions:
                    raise ValueError('No ingested version contains amendment paragraphs')
            else:
                versions = [UUID(args.version)]
            for version in versions:
                print(json.dumps(run_version(conn, version), ensure_ascii=False))
    except (ValueError, KeyError, psycopg.Error) as exc:
        parser.exit(1, (str(exc) if isinstance(exc, ValueError)
                        else 'Amendment link database/configuration failure; check migrations and credentials') + '\n')


if __name__ == '__main__':
    main()

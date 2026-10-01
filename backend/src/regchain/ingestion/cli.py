import argparse
import json
import os
import sys
from psycopg.rows import dict_row

import psycopg

from .fetch import fetch, validate_url
from .models import IngestionError, Download
from .repository import record_rejected, save
from .sources import PUBLICATIONS, consolidated_sources, glossary_source
from .worker import bounded_parse

SOURCES = PUBLICATIONS


def reparse_latest() -> list[dict]:
    """Reprocess retained original bytes; never refetch or rewrite old evidence."""
    with psycopg.connect(host=os.getenv('PGHOST','db'),port=os.getenv('PGPORT','5432'),
        dbname=os.getenv('PGDATABASE','regchain'),user='regchain_ingest',
        password=os.environ['INGEST_DB_PASSWORD'],connect_timeout=5,autocommit=True,
        row_factory=dict_row) as conn:
        rows = conn.execute('''SELECT r.external_key,r.canonical_url,v.source_url,v.retrieved_at,
            o.body,o.media_type FROM regulations r JOIN LATERAL
            (SELECT * FROM regulation_versions WHERE regulation_id=r.id ORDER BY version_no DESC LIMIT 1) v ON true
            JOIN regulatory_source_objects o ON o.raw_hash=v.raw_hash ORDER BY r.external_key''').fetchall()
        results = []
        for row in rows:
            source = Download(row['canonical_url'],row['source_url'],bytes(row['body']),
                              row['media_type'],row['retrieved_at'],{'processing':'retained-source-reparse'})
            results.append(save(conn,row['external_key'],source,bounded_parse(source)))
        return results


def ingest(url: str, key: str) -> dict:
    source = fetch(url)
    # Parsing happens before any version transaction. Rejected fetches retain evidence.
    with psycopg.connect(
        host=os.getenv("PGHOST", "db"), port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE", "regchain"), user="regchain_ingest",
        password=os.environ["INGEST_DB_PASSWORD"], connect_timeout=5, autocommit=True,
    ) as conn:
        try:
            document = bounded_parse(source)
        except IngestionError as exc:
            record_rejected(conn, source, str(exc))
            raise
        return save(conn, key, source, document)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect official FCA HTML/PDF; never infer legal obligations")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("url", help="Ingest one allowlisted FCA URL")
    one.add_argument("url")
    one.add_argument("--key", help="Stable document identity across URL changes")
    commands.add_parser("sync", help="Fetch the curated FCA publication starter sources once")
    commands.add_parser("handbook", help="Fetch the curated consolidated Handbook chapters")
    terms = commands.add_parser("glossary", help="Fetch named Handbook glossary definitions")
    terms.add_argument("ids", nargs="+", metavar="G252", help="Glossary identifiers read from an ingested source")
    commands.add_parser('reparse',help='Reprocess latest retained source bytes with the current parser')
    args = parser.parse_args()
    if args.command == 'reparse':
        try:
            for result in reparse_latest():
                print(json.dumps(result,ensure_ascii=False))
        except (IngestionError,psycopg.Error,KeyError,ValueError) as exc:
            print(json.dumps({'status':'FAILED','error':str(exc) if isinstance(exc,(IngestionError,ValueError))
                else 'Database/configuration failure; check migration and ingestion credentials'}),file=sys.stderr)
            raise SystemExit(1)
        return
    if args.command == "handbook":
        sources = list(consolidated_sources())
    elif args.command == "glossary":
        try:
            sources = [glossary_source(identifier) for identifier in args.ids]
        except ValueError as exc:
            print(json.dumps({"status": "FAILED", "error": str(exc)}), file=sys.stderr)
            raise SystemExit(1)
    elif args.command == "url":
        sources = [(args.key or args.url, args.url)]
    else:
        sources = list(SOURCES)
    failed = False
    for key, url in sources:
        try:
            if key == url:
                key = validate_url(url)
            print(json.dumps(ingest(url, key), ensure_ascii=False))
        except (IngestionError, psycopg.Error, KeyError, ValueError) as exc:
            failed = True
            message = str(exc) if isinstance(exc, (IngestionError, ValueError)) else "Database/configuration failure; check migration and ingestion credentials"
            print(json.dumps({"status": "FAILED", "source": url, "error": message}), file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

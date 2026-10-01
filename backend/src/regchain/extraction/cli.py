import argparse
import json
import os
from uuid import UUID

import psycopg

from .providers import configured_provider
from .service import run_version


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract review-only, source-grounded obligation candidates')
    parser.add_argument('version',help='Version UUID or latest (latest version of each regulation)')
    parser.add_argument('--provider',choices=['rules','ollama'],default=os.getenv('LLM_PROVIDER') or None)
    args = parser.parse_args()
    if not args.provider:
        parser.error('Specify --provider rules for non-AI candidates or --provider ollama for a configured LLM')
    try:
        provider = configured_provider(args.provider)
        with psycopg.connect(host=os.getenv('PGHOST','db'),dbname=os.getenv('PGDATABASE','regchain'),
             port=os.getenv('PGPORT','5432'),user='regchain_extract',
             password=os.environ['EXTRACT_DB_PASSWORD'],connect_timeout=5,autocommit=True) as conn:
            versions = ([r[0] for r in conn.execute('''SELECT DISTINCT ON(regulation_id) id FROM regulation_versions
                         ORDER BY regulation_id,version_no DESC''').fetchall()] if args.version == 'latest' else [UUID(args.version)])
            if not versions:
                raise ValueError('No regulatory versions; run ingest sync first')
            failed = False
            for version in versions:
                result = run_version(conn,version,provider)
                print(json.dumps(result))
                failed |= result['status'] == 'FAILED'
            if failed:
                raise SystemExit(1)
    except (ValueError,KeyError,psycopg.Error) as exc:
        parser.exit(1, (str(exc) if isinstance(exc,ValueError) else 'Extraction database/configuration failure; check migrations and credentials') + '\n')


if __name__ == '__main__':
    main()

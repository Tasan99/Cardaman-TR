"""Transactional, serialized, checksum-verified SQL migrations."""
import hashlib
import os
from pathlib import Path
import psycopg
from psycopg import sql

def main() -> None:
    directory = Path(__file__).resolve().parents[2] / "migrations"
    if not directory.exists():
        directory = Path("/app/migrations")
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7263241)")
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, checksum text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())")
        for path in sorted(directory.glob("*.sql")):
            version = path.name.split("_", 1)[0]
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            existing = conn.execute("SELECT checksum FROM schema_migrations WHERE version=%s", (version,)).fetchone()
            if existing:
                if existing[0] != checksum:
                    raise RuntimeError(f"Applied migration {version} was modified")
                continue
            conn.execute(raw.decode("utf-8"))
            conn.execute("INSERT INTO schema_migrations(version,checksum) VALUES (%s,%s)", (version, checksum))
        password = os.environ["APP_DB_PASSWORD"]
        if len(password) < 16:
            raise ValueError("APP_DB_PASSWORD must contain at least 16 characters")
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname='regchain_app'").fetchone():
            conn.execute("CREATE ROLE regchain_app LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT")
        conn.execute(sql.SQL("ALTER ROLE regchain_app PASSWORD {}").format(sql.Literal(password)))
        conn.execute("GRANT USAGE ON SCHEMA public TO regchain_app")
        conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO regchain_app")
        ingest_password = os.environ["INGEST_DB_PASSWORD"]
        if len(ingest_password) < 16:
            raise ValueError("INGEST_DB_PASSWORD must contain at least 16 characters")
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname='regchain_ingest'").fetchone():
            conn.execute("CREATE ROLE regchain_ingest LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT")
        conn.execute(sql.SQL("ALTER ROLE regchain_ingest PASSWORD {}").format(sql.Literal(ingest_password)))
        conn.execute("GRANT USAGE ON SCHEMA public TO regchain_ingest")
        conn.execute("GRANT SELECT ON regulators TO regchain_ingest")
        conn.execute("""GRANT SELECT, INSERT ON regulations, regulation_versions, regulation_sections,
                     regulatory_changes, regulatory_chunks, regulatory_source_objects, regulatory_fetches,
                     source_comparisons, source_comparison_changes TO regchain_ingest""")
        # Ingestion has no grants on tenant/company data, UPDATE, DELETE or role management.
        extract_password = os.environ['EXTRACT_DB_PASSWORD']
        if len(extract_password) < 16:
            raise ValueError('EXTRACT_DB_PASSWORD must contain at least 16 characters')
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname='regchain_extract'").fetchone():
            conn.execute('CREATE ROLE regchain_extract LOGIN NOSUPERUSER NOBYPASSRLS NOINHERIT')
        conn.execute(sql.SQL('ALTER ROLE regchain_extract PASSWORD {}').format(sql.Literal(extract_password)))
        conn.execute('GRANT USAGE ON SCHEMA public TO regchain_extract')
        conn.execute('GRANT SELECT ON regulations,regulators,regulation_versions,regulation_sections TO regchain_extract')
        conn.execute('''GRANT SELECT,INSERT ON extraction_runs,obligations,extraction_reviews,
                     obligation_context_evidence,extraction_corpus_sources,
                     amendment_runs,amendment_links,amendment_link_gaps TO regchain_extract''')
    print("Migrations applied and verified")

if __name__ == "__main__":
    main()

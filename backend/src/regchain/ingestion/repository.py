from hashlib import sha256

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .diff import paragraph_diff
from .models import Document, Download, NORMALIZER_VERSION, PARSER_VERSION, Paragraph


def store_object(conn: Connection, source: Download) -> None:
    conn.execute("""INSERT INTO regulatory_source_objects(raw_hash,body,media_type)
                    VALUES(%s,%s,%s) ON CONFLICT(raw_hash) DO NOTHING""",
                 (source.raw_hash, source.body, source.media_type))


def record_rejected(conn: Connection, source: Download, error: str) -> None:
    with conn.transaction():
        store_object(conn, source)
        conn.execute("""INSERT INTO regulatory_fetches(requested_url,final_url,fetched_at,raw_hash,status,error_message,response_metadata)
                        VALUES(%s,%s,%s,%s,'REJECTED',%s,%s)""",
                     (source.requested_url, source.final_url, source.fetched_at, source.raw_hash, error[:1000], Jsonb(source.headers)))


def save(conn: Connection, key: str, source: Download, document: Document) -> dict:
    """Atomic append. A per-source lock serializes concurrent version allocation."""
    if not key.strip() or len(key) > 2048:
        raise ValueError("Regulation key must contain 1..2048 characters")
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", ("FCA:" + key,))
        cursor.execute("SELECT id FROM regulators WHERE code='FCA'")
        regulator_id = cursor.fetchone()["id"]
        cursor.execute("SELECT id FROM regulations WHERE regulator_id=%s AND external_key=%s", (regulator_id, key))
        row = cursor.fetchone()
        if row:
            regulation_id = row["id"]
        else:
            cursor.execute("""INSERT INTO regulations(regulator_id,external_key,title,canonical_url)
                              VALUES(%s,%s,%s,%s) RETURNING id""",
                           (regulator_id, key, document.title, source.requested_url))
            regulation_id = cursor.fetchone()["id"]
        store_object(conn, source)
        cursor.execute("SELECT * FROM regulation_versions WHERE regulation_id=%s ORDER BY version_no DESC LIMIT 1", (regulation_id,))
        latest = cursor.fetchone()
        unchanged = (latest and latest["normalized_hash"] == document.content_hash
                     and latest["parser_version"] == PARSER_VERSION
                     and latest["normalizer_version"] == NORMALIZER_VERSION)
        changes = []
        parser_changed = bool(latest and (latest['parser_version'] != PARSER_VERSION
                              or latest['normalizer_version'] != NORMALIZER_VERSION))
        origin = ('INITIAL' if not latest else 'PARSER_REPROCESSING' if parser_changed
                  and latest['raw_hash'] == source.raw_hash else 'SOURCE_AND_PARSER_CHANGE'
                  if parser_changed else 'SOURCE_CHANGE')
        if unchanged:
            version_id, version_no = latest["id"], latest["version_no"]
        else:
            version_no = latest["version_no"] + 1 if latest else 1
            cursor.execute("""INSERT INTO regulation_versions(regulation_id,version_no,retrieved_at,
                publication_date,effective_date,source_url,raw_hash,normalized_hash,object_key,
                parser_version,normalizer_version,normalized_text,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (regulation_id, version_no, source.fetched_at, document.publication_date, document.effective_date,
                 source.final_url, source.raw_hash, document.content_hash, "pg-source:" + source.raw_hash,
                 PARSER_VERSION, NORMALIZER_VERSION, document.normalized_text,
                 Jsonb({"title": document.title, "media_type": source.media_type,
                        "warnings": list(document.warnings), "legal_classification": "UNKNOWN",
                        "change_origin":origin,"previous_version_id":str(latest['id']) if latest else None})))
            version_id = cursor.fetchone()["id"]
            section_ids = []
            for ordinal, paragraph in enumerate(document.paragraphs, 1):
                cursor.execute("""INSERT INTO regulation_sections(version_id,section_number,paragraph_number,
                    page,heading_path,text,legal_type,content_hash,ordinal,locator_kind,
                    page_end,printed_label,source_kind,quality_flags,source_spans,
                    effective_from,defined_terms)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (version_id, paragraph.section, paragraph.number, paragraph.page,
                     Jsonb(list(paragraph.heading_path)), paragraph.text, paragraph.legal_type,
                     sha256(paragraph.text.encode('utf-8')).hexdigest(), ordinal, paragraph.locator_kind,
                     paragraph.page_end,paragraph.printed_label,paragraph.source_kind,
                     Jsonb(list(paragraph.quality_flags)),Jsonb(list(paragraph.source_spans)),
                     paragraph.effective_from,Jsonb(list(paragraph.defined_terms))))
                section_id = cursor.fetchone()["id"]
                section_ids.append(section_id)
                # Full-paragraph chunks preserve lists/exceptions; embeddings come in RAG phase.
                cursor.execute("""INSERT INTO regulatory_chunks(section_id,text,char_start,char_end,embedding_model,chunker_version)
                    VALUES(%s,%s,0,%s,'NOT_EMBEDDED','paragraph-v1')""", (section_id, paragraph.text, len(paragraph.text)))
            if latest and not parser_changed:
                cursor.execute("SELECT * FROM regulation_sections WHERE version_id=%s ORDER BY ordinal NULLS LAST,section_number,paragraph_number", (latest["id"],))
                old = cursor.fetchall()
                old_paragraphs = tuple(Paragraph(r["section_number"], r["paragraph_number"], r["text"],
                    tuple(r["heading_path"]), r["page"], r["locator_kind"]) for r in old)
                changes = paragraph_diff(old_paragraphs, document.paragraphs)
                for change in changes:
                    cursor.execute("""INSERT INTO regulatory_changes(regulation_id,old_version_id,new_version_id,
                        old_section_id,new_section_id,kind,diff_text) VALUES(%s,%s,%s,%s,%s,%s,%s)""",
                        (regulation_id, latest["id"], version_id,
                         old[change.old_index]["id"] if change.old_index is not None else None,
                         section_ids[change.new_index] if change.new_index is not None else None,
                         change.kind, change.diff_text))
        status = "UNCHANGED" if unchanged else "CREATED"
        cursor.execute("""INSERT INTO regulatory_fetches(regulation_id,version_id,requested_url,final_url,
            fetched_at,raw_hash,status,response_metadata) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (regulation_id, version_id, source.requested_url, source.final_url, source.fetched_at,
             source.raw_hash, status, Jsonb(source.headers)))
        fetch_id = cursor.fetchone()["id"]
        return {"regulation_id": str(regulation_id), "version_id": str(version_id), "version_no": version_no,
                "fetch_id": str(fetch_id), "status": status, "changed_paragraphs": len(changes),
                "change_origin":'UNCHANGED' if unchanged else origin,
                "raw_hash": source.raw_hash, "normalized_hash": document.content_hash,
                "warnings": list(document.warnings)}

from collections import Counter
import json
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from regchain.evidence import digest
from .classify import classify
from .pipeline import Result, extract
from .providers import PROMPT, CONTEXT_PROMPT, REVIEW_PROMPT, Provider, runtime_manifest
from .contract import CONTRACT_VERSION, EXAMPLE
from .retrieval import retrieve, RETRIEVER_VERSION
from .schema import ExtractionOutput, ModelOutput, SCHEMA_VERSION
from .structure import structure_of

# Consolidated provisions from every other regulation's latest version. A policy
# statement states how the law changed; the operative text lives in these sources.
CORPUS_QUERY = '''SELECT s.*, r.external_key, v.normalized_hash AS version_hash
    FROM regulation_sections s
    JOIN regulation_versions v ON v.id = s.version_id
    JOIN regulations r ON r.id = v.regulation_id
    WHERE s.source_kind = 'CONSOLIDATED' AND v.id <> %s
      AND v.version_no = (SELECT max(version_no) FROM regulation_versions
                          WHERE regulation_id = r.id)
    ORDER BY r.external_key, s.ordinal NULLS LAST, s.section_number, s.paragraph_number'''

# grounding_version names the gate logic in grounding.py, classify.py and the span repair before
# it; v3.2 = delegation sentences excluded from modal evidence, alignment hints, and the
# classification gate below; v3.3 (v0.19) = a truncated Turkish action is expanded to its clause
# before the gate (spans.py). The bump blocks reuse of packets that carry truncated actions.
# v3.4 (v0.19 t6, review 1 PROV-1) = every duty sentence of an accepted Turkish unit gets a candidate from that sentence
# (spans.cover_sentences, pipeline.cover_output): the same model answer now yields more duties, so an extraction run or a
# packet made by v3.3 is not reused as current.
# v3.5 (v0.19 t7, review 1 PROV-P2-1) = the review decides per candidate (pipeline.second_reading / apply_review): an
# UNSUPPORTED decision drops its own candidate only, an UNCERTAIN one keeps its candidate flagged, and a review failure ends
# the unit on record. The same review answer now yields other duties than v3.4 (which emptied the unit on any doubt), so an
# extraction run or a packet made by v3.4 is not reused as current.
PROMPT_HASH = digest({'prompt':PROMPT,'context_prompt':CONTEXT_PROMPT,'review_prompt':REVIEW_PROMPT,
                      'schema_json':json.dumps(ExtractionOutput.model_json_schema(),sort_keys=True,separators=(',',':'),allow_nan=False),
                      'grounding_version':'scoped-evidence-v3.5-candidate-review','retriever_version':RETRIEVER_VERSION,
                      'contract_version':CONTRACT_VERSION,'example':EXAMPLE,
                      'model_schema_json':json.dumps(ModelOutput.model_json_schema(),sort_keys=True,separators=(',',':'),allow_nan=False)})


def classified(section: dict):
    """(Classification, Result or None): a paragraph whose class carries no duty is not extracted.

    The same rule the pilot applies per sub-paragraph, here per stored paragraph: a
    definitions article, a scope statement, a delegation to a by-law, a penalty article or a
    permission costs no model call and is recorded under its class.
    """
    path = section.get('heading_path') or []
    heading = path[2] if isinstance(path, (list, tuple)) and len(path) >= 3 else ''
    kind = classify(section['text'], heading, section.get('legal_type'))
    if kind.extractable:
        return kind, None
    return kind, Result(ExtractionOutput(status='NO_EXPLICIT_OBLIGATION'), 'CLASSIFIED_' + kind.kind, 0,
                        diagnostics=({'stage': 'classification', 'code': 'CLASSIFIED_' + kind.kind, 'marker': kind.marker, 'reason': kind.reason},))


def structures(text: str, output: ExtractionOutput) -> list[dict]:
    """One structure-v1 side-car per candidate, stored in the review JSON beside the candidates.

    Deterministic, never raises and float-free, so the payload digest and the JSONB column take
    it as they are; Candidate itself (and so the obligations table) is unchanged.
    """
    actions = [c.prohibited_action if c.modality.endswith('_NOT') else c.required_action for c in output.obligations]
    return [structure_of(text, candidate.model_dump(mode='json'), 0, [a for j, a in enumerate(actions) if j != i])
            for i, candidate in enumerate(output.obligations)]


def pinned_corpus(corpus: list[dict]) -> list[dict]:
    """One row per corpus version, identified by external key and content hash."""
    pinned: dict[str,dict] = {}
    for section in corpus:
        key = str(section['version_id'])
        pinned.setdefault(key,{'version_id':key,'external_key':section['external_key'],
            'normalized_hash':section['version_hash'],
            'role':'GLOSSARY' if section['locator_kind'] == 'handbook_glossary' else 'CONSOLIDATED'})
    return [pinned[key] for key in sorted(pinned)]


def run_version(conn: Connection, version_id: UUID, provider: Provider) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM regulation_versions WHERE id=%s',(version_id,))
        version = cur.fetchone()
        if not version:
            raise ValueError('Regulatory version not found')
        cur.execute('SELECT * FROM regulation_sections WHERE version_id=%s ORDER BY ordinal NULLS LAST,section_number,paragraph_number',(version_id,))
        sections = cur.fetchall()
        if not sections:
            raise ValueError('Version has no source paragraphs')
        cur.execute(CORPUS_QUERY,(version_id,))
        corpus = cur.fetchall()
        pinned = pinned_corpus(corpus)
        # The pinned corpus is part of the input: resolving the same paragraph against
        # a different corpus is a different analysis and must not report UNCHANGED.
        manifest = {'version_id':str(version_id),'raw_hash':version['raw_hash'],
                    'normalized_hash':version['normalized_hash'],
                    'sections':[{'id':str(s['id']),'text':s['text'],'hash':s['content_hash']} for s in sections],
                    'corpus':pinned,'runtime':runtime_manifest(provider)}
        input_hash = digest(manifest)
        signature = (version_id,provider.name,provider.model_version,PROMPT_HASH,SCHEMA_VERSION,input_hash)
        find = '''SELECT id FROM extraction_runs WHERE version_id=%s AND provider=%s AND model_version=%s
                  AND prompt_hash=%s AND schema_version=%s AND input_hash=%s AND status='REVIEW_REQUIRED'
                  ORDER BY created_at DESC LIMIT 1'''
        cur.execute(find,signature)
        existing = cur.fetchone()
        if existing:
            return {'run_id':str(existing['id']),'version_id':str(version_id),'status':'UNCHANGED'}
    # No transaction or database locks are held during provider calls (CLI uses autocommit).
    results, kinds = [], []
    aborted = False
    contexts = [retrieve(section,sections,corpus=corpus) for section in sections]
    for section,context in zip(sections,contexts):
        kind, skipped = classified(section)
        kinds.append(kind)
        result = (skipped if skipped is not None else
                  Result(ExtractionOutput(status='INSUFFICIENT_EVIDENCE'),'RUN_ABORTED_AFTER_PROVIDER_FAILURE',0)
                  if aborted else extract(section['text'],provider,context))
        results.append(result)
        aborted |= result.reason == 'PROVIDER_FAILURE'
    payload = [{'section_id':str(s['id']),'output':r.output.model_dump(mode='json'),
                'reason':r.reason,'attempts':r.attempts,'human_review_required':True,'context':c.manifest(),
                'classification':{'kind':k.kind,'marker':k.marker,'basis':k.reason},
                'diagnostics':list(r.diagnostics),'structures':structures(s['text'],r.output)}
               for s,r,c,k in zip(sections,results,contexts,kinds)]
    counts = Counter(r.output.status for r in results)
    # Operational provider failure must not look like a completed regulatory analysis.
    failed = any(r.reason == 'PROVIDER_FAILURE' for r in results)
    status = 'FAILED' if failed else 'REVIEW_REQUIRED'
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',(digest([str(v) for v in signature]),))
        cur.execute(find,signature)
        existing = cur.fetchone()
        if existing:
            return {'run_id':str(existing['id']),'version_id':str(version_id),'status':'UNCHANGED'}
        cur.execute('''INSERT INTO extraction_runs(version_id,provider,model_version,prompt_hash,schema_version,
            input_hash,output_hash,status,error_code,runtime_manifest) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
            (*signature,digest(payload),status,'PROVIDER_FAILURE' if failed else None,Jsonb(runtime_manifest(provider))))
        run_id = cur.fetchone()['id']
        for source in pinned:
            cur.execute('''INSERT INTO extraction_corpus_sources(run_id,corpus_version_id,
                external_key,normalized_hash,role) VALUES(%s,%s,%s,%s,%s)''',
                (run_id,source['version_id'],source['external_key'],
                 source['normalized_hash'],source['role']))
        for section,result,entry in zip(sections,results,payload):
            cur.execute('''INSERT INTO extraction_reviews(version_id,section_id,run_id,status,reason,attempts,result)
                VALUES(%s,%s,%s,%s,%s,%s,%s)''',
                (version_id,section['id'],run_id,result.output.status,result.reason,result.attempts,Jsonb(entry)))
            # Do not publish partial obligations from a failed run.
            if failed:
                continue
            for candidate in result.output.obligations:
                cur.execute('''INSERT INTO obligations(version_id,section_id,extraction_run_id,source_quote,
                    quote_start,quote_end,subject,modality,required_action,prohibited_action,conditions,exceptions,
                    deadline,jurisdiction,affected_products,affected_entities,customer_types,risk_category,
                    confidence,human_review_required)
                    VALUES(%s,%s,%s,%s,0,%s,%s,%s,%s,%s,%s,%s,%s,'UK',%s,%s,%s,'UNKNOWN',%s,true) RETURNING id''',
                    (version_id,section['id'],run_id,candidate.source_quote,len(candidate.source_quote),
                     candidate.subject,candidate.modality,candidate.required_action,candidate.prohibited_action,
                     Jsonb(candidate.conditions),Jsonb(candidate.exceptions),candidate.deadline,
                     Jsonb(candidate.affected_products),Jsonb(candidate.affected_entities),Jsonb(candidate.customer_types),
                     candidate.confidence_score))
                obligation_id = cur.fetchone()['id']
                context_sources = {i['section_id']:i for i in entry['context']['items']}
                for evidence in candidate.supporting_evidence:
                    source = context_sources[evidence.section_id]
                    cur.execute('''INSERT INTO obligation_context_evidence(obligation_id,version_id,source_version_id,section_id,role,quote,source_hash)
                        VALUES(%s,%s,%s,%s,%s,%s,%s)''',
                        (obligation_id,version_id,source['version_id'],evidence.section_id,evidence.role,evidence.quote,source['source_hash']))
    return {'run_id':str(run_id),'version_id':str(version_id),'status':status,
            'provider':provider.name,'model_version':provider.model_version,'paragraphs':len(sections),
            'candidates':0 if failed else sum(len(r.output.obligations) for r in results),
            'paragraph_status_counts':dict(counts),'human_review_required':True,
            'corpus_versions':len(pinned),
            'missing_sources':sorted({label for c in contexts for label in c.missing_sources})}

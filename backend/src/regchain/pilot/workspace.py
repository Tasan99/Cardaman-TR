"""Single-operator loopback workspace; deliberately separate from the tenant API."""
import argparse
import base64
import binascii
import io
import json
import os
import re
import secrets
import shutil
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import Field, field_validator, model_validator

from regchain import __version__, display_version
from regchain.evidence import digest, verify_chain
from regchain.model_router import ModelPolicyError, ModelRouter, ModelScope, bind_model_scope, model_mode
from regchain.extraction.providers import ProviderFailure, configured_judge, configured_provider, usage_summary
from regchain.platform.audit import AuditLog
from regchain.platform.context import RequestContext, deterministic_request_id, new_request_id, REQUEST_ID
from regchain.platform.rbac import ADMIN, PermissionDenied, require
from regchain.platform.redaction import ALLOWED_CALL_FIELDS, RedactionPolicy, redact_call
from regchain.platform.retention import RetentionSettings, delete_runs, export_runs
from .artifacts import save_bundle, write_json
from .engine import DEFAULT_CATEGORIES, analyze, answer_question, autonomous_review, draft_clause, load_packet, pipeline_settings, UNAVAILABLE
from regchain.extraction.grounding import turkish
from .evaluation import evaluate, label_template
from .paths import extended
from .policies import MAX_BYTES, handbook_extract, mevzuat_extract, read_policy
from .rerank import configured_reranker
from .turkiye import gazette_scan, search_mevzuat
from regchain.ingestion.mevzuat import KINDS, document_label
from .report import render
from .review import apply_review
from .schema import Company, Review, Strict
from .semantic import configured_embedder
from .sources import chapter_of, load_sources, save_sources, select_targets
from .verification import verify_artifacts

MAX_UPLOAD = 24 * 1024 * 1024
MAX_REQUEST = 34 * 1024 * 1024
RUN_ID = re.compile(r'^[0-9a-f]{32}$')
HASH = re.compile(r'^[0-9a-f]{64}$')
# Printed FCA locators include "COBS 4.5A.4", "CONC 7.3.5-A" and "COBS 4.7.-2"; Turkish
# provisions are "Kanun 5549 md. 4" or "Yönetmelik 200713012 md. 9/A".
LABEL = re.compile(r'^[A-Z]{2,6} [0-9]{1,2}[A-Z]?(?:\.-?[0-9]{1,3}[A-Z]{0,2}(?:-[A-Z])?){2}$'
                   r'|^(?:Kanun|Yönetmelik|Tebliğ|CBK|KHK|Tüzük) [0-9]{1,12} md\. (?:Ek |Geçici )?[0-9]{1,3}(?:/[A-ZÇĞİÖŞÜ])?$')
ARTICLE_NUMBER = re.compile(r'^(?:Ek |Geçici )?[0-9]{1,3}(?:/[A-ZÇĞİÖŞÜ])?$')
# A local model needs roughly a minute per provision that yields a candidate.
MAX_TARGETS = 400
TERMINAL = {'COMPLETED', 'FAILED', 'INTERRUPTED'}


class WorkspaceBusy(ValueError):
    """Another process holds this workspace; distinct from corrupt retained evidence."""


def now():
    return datetime.now(timezone.utc).isoformat()


def filename(value):
    # Reject Windows separators, ADS, devices and trailing dots on every platform.
    if (not value or len(value) > 150 or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)
            or value != value.strip() or value.endswith('.') or value in ('.', '..')
            or value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))}
            or Path(value).suffix.lower() not in ('.txt', '.md', '.pdf', '.docx', '.csv')):
        raise ValueError('Policy adı yalnızca dosya adı olmalı; TXT, MD, PDF, DOCX ya da kontrol kaydı için CSV kullanın.')
    return value


class Upload(Strict):
    name: str
    content: str = Field(min_length=1, max_length=MAX_BYTES * 4 // 3 + 8)

    _name = field_validator('name')(filename)


class RunInput(Strict):
    company: Company
    policies: list[Upload] = Field(min_length=1, max_length=8)
    provider: Literal['ollama', 'rules'] = 'ollama'
    source_mode: Literal['refresh', 'reuse'] = 'refresh'
    # Explicit operator choice. Hybrid never degrades to lexical when the embedder fails.
    retrieval: Literal['lexical', 'hybrid'] = 'lexical'
    previous_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    # Any FCA Handbook chapter. 'rules' and 'all' take every provision of the chapter
    # (optionally only the named sections); 'labels' takes exactly the listed provisions.
    module: str = Field(default='CONC', pattern=r'^[A-Za-z]{2,6}$')
    chapter: str = Field(default='7', pattern=r'^[0-9]{1,2}[A-Za-z]?$')
    selection: Literal['labels', 'rules', 'all'] = 'labels'
    # Türkiye: one regulation of the Mevzuat Bilgi Sistemi, named by the identifiers its
    # search returns. 'sections' then lists article numbers; empty means every operative article.
    regulator: Literal['FCA', 'TR'] = 'FCA'
    mevzuat_kind: str = Field(default='', pattern=r'^[0-9]{0,2}$')
    mevzuat_number: str = Field(default='', pattern=r'^[0-9]{0,12}$')
    mevzuat_tertip: str = Field(default='5', pattern=r'^[0-9]$')
    # A Handbook export uploaded as a policy is refused unless the operator insists.
    allow_regulatory_text: bool = False
    # 'autonomous': the AI records its own review; proposals that pass the evidence gate become
    # decisions, the rest are routed to a person. judge_votes > 1 judges every passage that many
    # times and escalates any disagreement (each vote costs a full judgement pass).
    decision_mode: Literal['human', 'autonomous'] = 'human'
    judge_votes: int = Field(default=1, ge=1, le=3)
    # The company's own risk taxonomy; every duty is filed under one of these for orientation.
    risk_categories: list[str] = Field(default=list(DEFAULT_CATEGORIES), max_length=12)
    # For every duty the policies do not cover (or contradict), an AI-generated remediation
    # proposal: rule-set type/priority/action plus a drafted clause (one model call per gap).
    remediation: bool = True

    @field_validator('risk_categories')
    @classmethod
    def valid_categories(cls, values):
        cleaned = [v.strip() for v in values if v.strip()]
        if any(len(v) > 60 for v in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError('Risk kategorileri en çok 60 karakter ve benzersiz olmalı.')
        return cleaned
    sections: list[str] = Field(default=[], max_length=40)
    labels: list[str] = Field(default=['CONC 7.3.4', 'CONC 7.3.4B'], max_length=MAX_TARGETS)

    @field_validator('labels')
    @classmethod
    def valid_labels(cls, values):
        if len(values) != len(set(values)) or any(not LABEL.fullmatch(v) for v in values):
            raise ValueError('Provision\'lar "COBS 4.2.1" biçiminde ve benzersiz olmalı.')
        return values

    @field_validator('sections')
    @classmethod
    def valid_sections(cls, values):
        if any(not (re.fullmatch(r'(?:[A-Za-z]{2,6} )?(?:[0-9]{1,2}[A-Za-z]?\.)?[0-9]{1,2}[A-Za-z]?', v.strip())
                    or ARTICLE_NUMBER.fullmatch(v.strip())) for v in values):
            raise ValueError('Kısımlar "4.2" ya da "4.5A" biçiminde, maddeler "3" ya da "9/A" biçiminde yazılır.')
        return values

    @model_validator(mode='after')
    def turkish_identity(self):
        if self.regulator == 'TR':
            if self.mevzuat_kind not in KINDS or not self.mevzuat_number:
                raise ValueError('Türk mevzuatı için düzenlemeyi arayıp listeden seç (tür kodu ve numarası gerekli).')
            # The group code and number stand in for module and chapter everywhere below.
            self.module, self.chapter = KINDS[self.mevzuat_kind][0], self.mevzuat_number
            if self.selection == 'labels':
                self.selection, self.sections, self.labels = 'all', [s for s in self.sections], []
        return self


def metrics_of(data):
    """Cardamon's header numbers, counted from the AI proposals: applicable candidates, and
    of those, the ones with written policy support and the ones a control register row supports.
    v0.16 adds every applicability and coverage state, the proposal outcomes and the review queue,
    so the screen can show the real scope of the analysis (a count of proposals, not a compliance rate)."""
    rows = data['obligations']
    controls = {c['source_id'] for p in data['policies'] for c in p['chunks'] if c.get('locator') == 'control_row'}
    state = lambda field, value: [r for r in rows if r['proposal'][field] == value]
    applicable = state('applicability', 'APPLIES')
    possibly = state('applicability', 'POSSIBLY_APPLIES')
    covered = [r for r in applicable if r['proposal']['coverage'] == 'COVERS_TEXT']
    control_covered = [r for r in applicable if r['proposal'].get('control_coverage') == 'SUPPORTS'
                       or any(c['relation'] == 'SUPPORTS' and c['source_id'] in controls for c in r['proposal']['policy_checks'])]
    categories = {}
    for r in rows:
        categories[r.get('category') or 'Sınıflanmadı'] = categories.get(r.get('category') or 'Sınıflanmadı', 0) + 1
    remediations = [r for r in rows if r['proposal'].get('remediation')]
    draft_failed = [r for r in remediations if r['proposal']['remediation'].get('draft_status') == 'DRAFT_UNAVAILABLE']
    statuses = {}
    for r in rows:
        key = r['proposal'].get('remediation_status') or ('PROPOSED' if r['proposal'].get('remediation') else 'NOT_ASSESSED')
        statuses[key] = statuses.get(key, 0) + 1
    unavailable = [r for r in rows if any(str(r['proposal'][f]).startswith(UNAVAILABLE) for f in ('applicability_reason', 'coverage_reason'))]
    assessed = [r for r in rows if r['proposal'].get('coverage_assessed', True)]
    return {'candidates': len(rows), 'applicable': len(applicable), 'possibly_applicable': len(possibly), 'covered': len(covered),
            'control_rows': len(controls), 'control_covered': len(control_covered),
            'conflicts': sum(1 for r in rows if r['signals']), 'categories': categories,
            'remediations': len(remediations),
            'remediations_high': sum(1 for r in remediations if r['proposal']['remediation']['priority'] == 'HIGH'),
            'applicability': {key: len(state('applicability', key)) for key in ('APPLIES', 'POSSIBLY_APPLIES', 'DOES_NOT_APPLY', 'UNKNOWN')},
            # v0.16.1: clauses the entity gate ruled out are not coverage-assessed and are counted apart.
            'entity_gate_excluded': sum(1 for r in rows if r['proposal'].get('applicability_rule') == 'ENTITY_GATE'),
            'coverage': {key: sum(1 for r in assessed if r['proposal']['coverage'] == key) for key in ('COVERS_TEXT', 'PARTIAL', 'CONFLICT', 'NO_EVIDENCE', 'UNKNOWN')},
            'coverage_not_assessed': len(rows) - len(assessed),
            'proposal_status': statuses, 'proposal_failed': len(draft_failed), 'ai_unavailable': len(unavailable),
            'pending_human_review': sum(1 for r in rows if r.get('status') == 'HUMAN_REVIEW_REQUIRED'),
            'note': 'AI önerilerinin sayımıdır; uyum oranı değildir.'}


class AskInput(Strict):
    question: str = Field(min_length=3, max_length=500)


class DraftInput(Strict):
    obligation_id: str = Field(pattern=r'^[0-9a-f]{64}$')


class AssignInput(Strict):
    obligation_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    assignee: str = Field(max_length=120)


class Workspace:
    def __init__(self, root: Path, cached_sources=None, provider_factory=configured_provider,
                 source_fetcher=save_sources, embedder_factory=configured_embedder,
                 judge_factory=configured_judge, reranker_factory=configured_reranker):
        self.require_local_mode()
        # Every run path derives from this root, so one namespace covers inputs,
        # sources, partial results, reviews and the containment checks between them.
        self.root = extended(root)
        self.runs = self.root/'runs'
        self.runs.mkdir(parents=True, exist_ok=True)
        self.cached_sources = extended(cached_sources) if cached_sources else None
        self.provider_factory = provider_factory
        self.source_fetcher = source_fetcher
        self.embedder_factory = embedder_factory
        self.judge_factory = judge_factory
        self.reranker_factory = reranker_factory
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='conc-pilot')
        self.active = None
        self.stopping = set()
        self.closed = False
        self.process_lock = None
        # v0.17 platform baseline. The single-operator pilot acts under one identity read from
        # the environment, never from a request: the tenant, the operator name and the roles.
        # Every action that starts, decides, exports or deletes is written to an append-only
        # audit chain beside the runs; model call logs leave the machine at the configured
        # redaction level; the retention window is a setting, and a sweep is an audited delete.
        self.tenant_id = (os.getenv('TENANT_ID') or 'local').strip()
        self.operator = (os.getenv('OPERATOR_NAME') or 'local-operator').strip()
        self.roles = frozenset(r.strip().upper() for r in (os.getenv('OPERATOR_ROLES') or ADMIN).split(',') if r.strip())
        self.audit_log = AuditLog(self.root/'audit.jsonl')
        self.redaction = RedactionPolicy((os.getenv('AI_LOG_REDACTION') or 'standard').strip().lower())
        self.retention = RetentionSettings.from_env()

    def context(self, request_id=None):
        """The identity a request acts under (operator, tenant, roles) with its request id."""
        return RequestContext(new_request_id(request_id), self.tenant_id, self.operator, self.roles, now())

    def audit(self, action, subject, outcome, detail='', context=None):
        """One entry on the audit chain; without a request context the workspace's own identity signs."""
        return self.audit_log.record(action, subject, outcome, detail, context=context or self.context())

    def ai_calls(self, run_id, level=None):
        """The run's model call records at a redaction level no weaker than the configured one."""
        levels = list(ALLOWED_CALL_FIELDS)
        level = (level or self.redaction.level).strip().lower()
        if level not in levels:
            raise ValueError('Redaksiyon düzeyi minimal, standard veya strict olmalı.')
        if levels.index(level) < levels.index(self.redaction.level):
            raise ValueError(f'Yapılandırılan redaksiyon düzeyi {self.redaction.level}; daha zayıf bir düzey istenemez.')
        path = self.directory(run_id)/'ai-calls.jsonl'
        if not path.is_file():
            return []
        policy = RedactionPolicy(level)
        return [redact_call(json.loads(line), policy) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def redacted_calls_text(self, run_id):
        return ''.join(json.dumps(entry, ensure_ascii=False)+'\n' for entry in self.ai_calls(run_id))

    def expired_runs(self, moment=None):
        """Runs past the retention window (never the active one, never a reviewed one when kept)."""
        rows = self.retention.expired(self.list_runs(), moment or now())
        return [row['id'] for row in rows if row['id'] != self.active]

    def delete(self, run_ids, context):
        with self.lock:
            run_ids = list(run_ids)
            if self.active in run_ids:
                # Refused like a malformed id in delete_runs: a delete attempt is on the chain either way.
                self.audit('delete_run', self.active, 'REFUSED', 'run is active', context)
                raise HTTPException(409, 'Çalışan analiz silinemez; önce durdurun.')
            return delete_runs(self.runs, run_ids, self.audit_log, context)

    def export_all(self, context):
        """Every completed run as one zip (the tenant's data), built beside the runs and removed after it is read."""
        run_ids = [row['id'] for row in self.list_runs() if row['state'] == 'COMPLETED']
        if not run_ids:
            raise HTTPException(404, 'Dışa aktarılacak tamamlanmış analiz yok.')
        exports = self.root/'exports'
        exports.mkdir(exist_ok=True)
        destination = exports/(uuid4().hex+'.zip')
        try:
            # Model call logs leave at the configured level, as in the single-run export.
            manifest = export_runs(self.runs, run_ids, destination, self.redaction)
            data = destination.read_bytes()
        finally:
            if destination.exists():
                destination.unlink()
        self.audit('export_data', 'tenant:'+self.tenant_id, 'OK', f'{len(run_ids)} runs, {len(manifest["files"])} files', context)
        return data

    def start(self):
        if self.process_lock:
            return
        # OS-held lock survives neither a crash nor process exit. A second server
        # cannot mark another process's live jobs interrupted or race its writes.
        self.process_lock = (self.root/'workspace.lock').open('a+b')
        self.process_lock.seek(0, 2)
        if self.process_lock.tell() == 0:
            self.process_lock.write(b'0')
            self.process_lock.flush()
        self.process_lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.process_lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.process_lock.close()
            self.process_lock = None
            raise WorkspaceBusy('Bu çalışma klasörü başka bir Cardaman sürecinde açık.') from exc
        for row in self.list_runs():
            if row['state'] not in TERMINAL:
                self.update(row['id'], state='INTERRUPTED',
                    error='Sunucu analiz bitmeden kapandı. Yeni bir deneme başlatabilirsiniz.')
            elif row['state'] == 'COMPLETED':
                # Recover an event committed just before a crash interrupted the
                # non-authoritative job index update. Evidence remains immutable.
                saved = []
                for path in (self.result(row['id'])/'reviews').glob('*.json'):
                    if not HASH.fullmatch(path.stem):
                        continue
                    packet = load_packet(path)
                    review = packet['events'][-1]['payload']['review']
                    if packet['events'][0]['event_hash'] != row['head'] or digest(review) != path.stem:
                        raise ValueError('Saklanan incelemenin analiz bağlantısı doğrulanamadı.')
                    saved.append({'id': path.stem, 'head': packet['head'], 'count': packet['count'],
                        'reviewer': review['reviewer'], 'saved_at': packet['events'][-1]['payload']['created_at']})
                saved.sort(key=lambda r: r['saved_at'])
                if saved != row.get('reviews', []):
                    self.update(row['id'], reviews=saved)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=True)
        if self.process_lock:
            self.process_lock.close()
            self.process_lock = None

    def directory(self, run_id):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError('Geçersiz analiz kimliği.')
        candidate = self.runs/run_id
        path = candidate.resolve()
        if not path.is_relative_to(self.runs) or candidate.is_symlink():
            raise ValueError('Çalışma klasörü dışına erişim reddedildi.')
        return path

    def metadata(self, run_id):
        # Same lock as update(): on Windows a reader holding job.json makes the worker's
        # os.replace fail, and a read during the swap is denied. Both surfaced as
        # sporadic PermissionError while the UI polled a running analysis.
        with self.lock:
            path = self.directory(run_id)/'job.json'
            if not path.is_file():
                raise HTTPException(404, 'Analiz bulunamadı.')
            return json.loads(path.read_text(encoding='utf-8'))

    def update(self, run_id, **fields):
        with self.lock:
            path = self.directory(run_id)/'job.json'
            value = self.metadata(run_id)
            value.update(fields, updated_at=now())
            temp = path.with_name('job-'+uuid4().hex+'.tmp')
            write_json(temp, value)
            os.replace(temp, path)
            return value

    def list_runs(self):
        rows = []
        for directory in self.runs.iterdir():
            if RUN_ID.fullmatch(directory.name) and (directory/'job.json').is_file():
                rows.append(self.metadata(directory.name))
        return sorted(rows, key=lambda r: r['created_at'], reverse=True)

    def result(self, run_id):
        if self.metadata(run_id)['state'] != 'COMPLETED':
            raise HTTPException(409, 'Analiz henüz tamamlanmadı.')
        return self.directory(run_id)/'result'

    def packet(self, run_id):
        result = self.result(run_id)
        packet = load_packet(result/'packet.json')
        verify_artifacts(packet, result)
        return packet

    def submit(self, request: RunInput):
        self.require_local_mode()
        with self.lock:
            if self.closed or self.active:
                raise HTTPException(409, 'Başka bir analiz çalışıyor; tamamlanmasını bekleyin.')
            previous = self.packet(request.previous_id) if request.previous_id else None
            if previous and previous['events'][0]['payload']['company']['id'] != request.company.id:
                raise ValueError('Önceki analiz başka bir şirkete ait.')
            # Misconfiguration is refused here, before a job exists or a model runs.
            run_id = uuid4().hex
            with bind_model_scope(ModelScope(self.tenant_id, request.company.id, run_id)):
                embedder = self.models(request.provider).embedder() if request.retrieval == 'hybrid' else None
            files, names, hashes = [], set(), set()
            for upload in request.policies:
                try:
                    raw = base64.b64decode(upload.content, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ValueError('Policy içeriği geçerli base64 değil.') from exc
                if not raw or len(raw) > MAX_BYTES:
                    raise ValueError('Her policy 1 byte ile 12 MiB arasında olmalı.')
                from hashlib import sha256
                raw_hash = sha256(raw).hexdigest()
                if upload.name.casefold() in names or raw_hash in hashes:
                    raise ValueError('Tekrarlanan policy adı veya içeriği.')
                names.add(upload.name.casefold())
                hashes.add(raw_hash)
                files.append((upload.name, raw))
            if sum(len(raw) for _, raw in files) > MAX_UPLOAD:
                raise ValueError('Toplam policy boyutu 24 MiB sınırını aşıyor.')
            directory = self.directory(run_id)
            directory.mkdir()
            inputs = directory/'inputs'
            inputs.mkdir()
            for name, raw in files:
                (inputs/name).write_bytes(raw)
            configuration = request.model_dump(exclude={'policies'})
            configuration['policy_names'] = [name for name, _ in files]
            write_json(directory/'input.json', configuration)
            # What the operator asked for, recorded before anything runs: the article numbers
            # (Türkiye) or the sections/provisions (FCA). Seen live: a run meant for md. 3, 4 and 8
            # processed all 51 articles because the form sent an empty filter, and nothing on the
            # screen said so until it was over.
            requested = (list(request.sections) if request.regulator == 'TR' or request.selection != 'labels' else list(request.labels))
            row = {'id': run_id, 'created_at': now(), 'updated_at': now(), 'state': 'QUEUED',
                'company_name': request.company.name, 'company_id': request.company.id,
                'synthetic': request.company.synthetic, 'provider': request.provider,
                'regulation': (document_label(request.mevzuat_kind, request.mevzuat_number) if request.regulator == 'TR'
                               else request.module.upper()+' '+request.chapter.upper()), 'regulator': request.regulator,
                'selection': request.selection, 'targets_requested': requested,
                'source_mode': request.source_mode, 'retrieval': request.retrieval,
                'decision_mode': request.decision_mode, 'judge_votes': request.judge_votes,
                'previous_id': request.previous_id, 'reviews': [], 'error': None}
            write_json(directory/'job.json', row)
            self.active = run_id
            self.executor.submit(self.work, run_id, request, previous, embedder)
            return row

    def work(self, run_id, request, previous, embedder=None):
        # ThreadPoolExecutor does not propagate ContextVars. The local operator's
        # run scope is rebound explicitly; a future server worker must instead
        # resolve these identities from authenticated, persisted job ownership.
        with bind_model_scope(ModelScope(self.tenant_id, request.company.id, run_id)):
            self._work(run_id, request, previous, embedder)

    def _work(self, run_id, request, previous, embedder=None):
        try:
            self.require_local_mode()
            directory = self.directory(run_id)
            self.update(run_id, state='PARSING')
            paths = [directory/'inputs'/u.name for u in request.policies]
            stages = {}
            parse_started = time.monotonic()
            policies = [read_policy(path) for path in paths]
            stages['parse_ms'] = int((time.monotonic() - parse_started) * 1000)
            # Document quality is recorded per file (v0.17); a scanned PDF was already refused by the reader.
            self.update(run_id, policy_quality=[{'name': p['filename'], **(p.get('document_quality') or {})} for p in policies])
            for policy in policies:
                found = None if request.allow_regulatory_text else handbook_extract(policy)
                if found:
                    module, chapter = found.split(' ')
                    raise ValueError(f'"{policy["filename"]}" bir şirket policy\'si değil, FCA Handbook metni ({found}) gibi görünüyor. '
                        f'Regülasyon yüklenmez: 03 adımında modül = {module}, bölüm = {chapter} seç ve buraya şirketinin kendi iç '
                        'policy belgesini yükle. Belge gerçekten şirket policy\'siyse formdaki onay kutusunu işaretle.')
                articles = 0 if request.allow_regulatory_text else mevzuat_extract(policy)
                if articles:
                    raise ValueError(f'"{policy["filename"]}" bir şirket policy\'si değil, {articles} maddelik bir mevzuat metni gibi görünüyor. '
                        'Regülasyon yüklenmez: 03 adımında "Türkiye" seçip düzenlemeyi adıyla ara; buraya şirketinin kendi iç policy '
                        'belgesini (uyum politikası, prosedür, talimat) yükle. Belge gerçekten şirket policy\'siyse formdaki onay kutusunu işaretle.')
            if sum(len(c['text']) for p in policies for c in p['chunks']) > 1_000_000:
                raise ValueError('Policy metni yerel pilotun 1 milyon karakter sınırını aşıyor.')
            models = self.models(request.provider)
            provider = models.extraction_provider()
            sources = directory/'sources'
            self.update(run_id, state='FETCHING')
            fetch_started = time.monotonic()
            retained, note = (None, None) if request.source_mode == 'refresh' else self.retained_sources_for(request, previous)
            if retained is None:
                self.source_fetcher(sources, module=request.module, chapter=request.chapter,
                                    **({'mevzuat': (request.mevzuat_kind, request.mevzuat_number, request.mevzuat_tertip)}
                                       if request.regulator == 'TR' else {}))
            else:
                shutil.copytree(retained, sources)
            stages['download_ms'] = int((time.monotonic() - fetch_started) * 1000)
            if note:
                self.update(run_id, source_note=note, source_mode_effective='refresh' if retained is None else 'reuse')
            _, sections = load_sources(sources)
            labels = select_targets(sections, request.module, request.chapter, request.selection,
                                    request.sections, request.labels)
            if not labels:
                raise ValueError('Bu seçimle incelenecek provision çıkmadı; kapsamı "kurallar + rehber" yap ya da kısımları kontrol et.')
            if len(labels) > MAX_TARGETS:
                raise ValueError(f'Seçim {len(labels)} provision içeriyor; tek analizde en fazla {MAX_TARGETS}. Kısım seçerek daralt.')
            # The resolved targets are on the job record from the first second of the analysis.
            reranker, rerank_status = (self.reranker_factory() if request.retrieval == 'hybrid'
                                       else (None, {'status': 'off', 'reason': 'kelime eşleşmesi seçildi; yeniden sıralama anlamsal aramayla çalışır'}))
            self.update(run_id, state='ANALYZING', targets=labels, target_count=len(labels), reranker=rerank_status,
                        progress={'done': 0, 'total': len(labels), 'label': None, 'phase': 'extraction'})
            old = previous['events'][0]['payload'] if previous else None
            judge = models.judge_provider()
            try:
                packet = analyze(request.company, policies, sections, provider, labels,
                    [c['source'] for c in old['cases']] if old else None,
                    previous['head'] if previous else None, old, embedder=embedder,
                    progress=lambda done, total, label, phase: self.update(
                        run_id, progress={'done': done, 'total': total, 'label': label, 'phase': phase}),
                    should_stop=lambda: run_id in self.stopping, judge=judge, votes=request.judge_votes,
                    categories=request.risk_categories, remediation=request.remediation, reranker=reranker,
                    target_filter=self.metadata(run_id).get('targets_requested') or [])
            finally:
                # Every model call of the run, with task, model, prompt hash, tokens and latency
                # but no text: the audit trail of why the AI answered what it answered.
                self.write_ai_calls(run_id, provider, judge)
            regulation = packet['events'][0]['payload']['regulation']
            if not regulation['processed']:
                self.update(run_id, state='INTERRUPTED', error='Analiz ilk provision işlenmeden durduruldu; saklanacak sonuç yok.')
                return
            self.update(run_id, state='SAVING', progress={'done': regulation['processed'],
                        'total': regulation['selected'], 'label': None, 'phase': 'done'})
            save_bundle(directory/'result', packet, sources, paths)
            data = packet['events'][0]['payload']
            # "Completed" must not hide that the model never answered. A user read a run
            # as a finished AI analysis when both proposals were ContextBudgetError.
            unavailable = sum(1 for o in data['obligations'] if any(o['proposal'][field].startswith(UNAVAILABLE)
                                                                     for field in ('applicability_reason', 'coverage_reason')))
            checks = [c for o in data['obligations'] for c in o['proposal']['policy_checks']]
            started = datetime.fromisoformat(self.metadata(run_id)['created_at'])
            usage = self.metadata(run_id).get('ai_usage') or {}
            # Observability (v0.17): stage timings, calls by stage, retries, failures, cache and the slowest duties.
            observability = {'stages_ms': {**stages, **(data.get('timings') or {})},
                             'calls_by_stage': {task: row['calls'] for task, row in (usage.get('by_task') or {}).items()},
                             'retries': self.metadata(run_id).get('ai_retries', 0), 'failures': usage.get('failures', 0),
                             'cache_hits': usage.get('cache_hits', 0), 'calls': usage.get('calls', 0),
                             'tokens': {'prompt': usage.get('prompt_tokens', 0), 'output': usage.get('output_tokens', 0)},
                             'slowest_obligations': [{'label': o['source_label'], 'elapsed_ms': o.get('elapsed_ms', 0),
                                                      'applicability': o['proposal']['applicability'], 'coverage': o['proposal']['coverage']}
                                                     for o in sorted(data['obligations'], key=lambda o: -(o.get('elapsed_ms') or 0))[:5]],
                             'review_flags': sum(1 for o in data['obligations'] if o['proposal'].get('review_flags'))}
            self.update(run_id, state='COMPLETED', head=packet['head'], count=packet['count'], metrics=metrics_of(data),
                ai_unavailable=unavailable, stopped_early=regulation['stopped_early'],
                # What the judge did, in numbers a user can hold it to.
                passages_judged=len(checks), passages_unclear=sum(1 for c in checks if c['relation'] == 'UNCLEAR'),
                passages_filtered=sum(len(o['proposal'].get('filtered_passages') or []) for o in data['obligations']),
                conflicts_flagged=sum(1 for o in data['obligations'] if o['signals']),
                candidates=len(data['obligations']), provisions=len(data['cases']),
                # The real scope of the analysis: the targets, and what was read around them.
                targets=regulation.get('targets', []), target_count=regulation.get('target_provisions', len(data['cases'])),
                helper_provisions=regulation.get('helper_provisions', []), helper_count=regulation.get('helper_count', 0),
                duration_seconds=int((datetime.now(timezone.utc) - started).total_seconds()),
                observability=observability,
                # v0.17 change impact: what the compared analysis' snapshot changed for this company.
                impact=data.get('impact'),
                source_dates=sorted({c['source']['fetched_at'] for c in data['cases']}),
                policy_count=len(data['policies']))
            if request.decision_mode == 'autonomous' and data['obligations']:
                # The AI's own review is a separate retained event on the same chain; the
                # original analysis stays untouched and verifiable.
                review, summary = autonomous_review(packet)
                saved = self.save_review(run_id, review)
                self.update(run_id, autonomy={**summary, 'review_id': saved['id'], 'head': saved['head']})
        except Exception as exc:
            # No tracebacks, company contents or full filesystem paths in HTTP logs.
            # shutil.Error lists source/destination paths, so only its kind is reported.
            if isinstance(exc, ValueError):
                error = str(exc)[:600]
            elif isinstance(exc, OSError):
                code = getattr(exc, 'winerror', None) or exc.errno
                error = ('Dosya işlemi tamamlanamadı ('+type(exc).__name__+(f' {code}' if code else '')+'). '
                    'Disk dolu, dosya başka bir programda açık veya klasör erişilemez olabilir. '
                    'Ayrıntı bu analizin klasöründeki error.log dosyasında.')
            else:
                error = ('İşlem tamamlanamadı: '+type(exc).__name__+
                    '. Ayrıntı bu analizin klasöründeki error.log dosyasında.')
            self.diagnose(run_id, exc)
            self.update(run_id, state='FAILED', error=error)
        finally:
            with self.lock:
                self.active = None
                self.stopping.discard(run_id)

    def retained_sources_for(self, request, previous):
        """(retained snapshot directory or None, note): the snapshot 'reuse' means for this run.

        Seen live (23 September 2026): the operator chose "reuse" for Tedbirler Yönetmeliği and the
        only retained snapshot the server knew was the launcher's CONC 7 folder, so the run failed
        with an FCA-worded error. Reuse now means: the compared run's snapshot; else the newest
        completed run of the same regulation; else the launcher's cached folder if it holds that
        regulation; else a fresh download, recorded as such in the job.
        """
        wanted = (request.module.strip().upper(), request.chapter.strip().upper())

        def holds(directory):
            try:
                _, sections = load_sources(directory)
            except (ValueError, OSError, KeyError):
                return False
            return any(chapter_of(s['printed_label']) == wanted for s in sections)
        if previous:
            directory = self.result(request.previous_id)/'regulatory-sources'
            if holds(directory):
                return directory, None
            raise ValueError('Karşılaştırılan analizin saklanan kaynağı bu düzenlemeye ait değil; aynı düzenlemenin bir analizini seç '
                             'ya da güncel kaynağı indir.')
        label = (document_label(request.mevzuat_kind, request.mevzuat_number) if request.regulator == 'TR'
                 else request.module.upper()+' '+request.chapter.upper())
        for row in self.list_runs():
            if row['state'] == 'COMPLETED' and row.get('regulation') == label and row.get('regulator', 'FCA') == request.regulator:
                directory = self.directory(row['id'])/'result'/'regulatory-sources'
                if holds(directory):
                    return directory, f'Saklanan kaynak: {row["created_at"][:10]} tarihli "{label}" analizinin snapshot\'ı.'
        if self.cached_sources and holds(self.cached_sources):
            return self.cached_sources, 'Saklanan kaynak: başlatıcının kaynak klasörü.'
        where = 'mevzuat.gov.tr' if request.regulator == 'TR' else 'FCA'
        return None, f'"{label}" için saklanan kaynak yoktu; {where}\'dan güncel kaynak indirildi.'

    def stop(self, run_id):
        with self.lock:
            if self.active != run_id:
                raise HTTPException(409, 'Bu analiz şu an çalışmıyor.')
            # Honoured between provisions: the one in progress finishes, the rest are skipped.
            self.stopping.add(run_id)
            return self.metadata(run_id)

    def write_ai_calls(self, run_id, *providers):
        """ai-calls.jsonl beside the run and a usage summary in job.json; nothing if no model ran."""
        logs = []
        for provider in providers:
            for candidate in (provider, getattr(provider, 'quick', None)):
                log = getattr(candidate, 'call_log', None)
                if isinstance(log, list) and not any(log is seen for seen in logs):
                    logs.append(log)
        calls = [call for log in logs for call in log]
        if not calls:
            return
        try:
            with (self.directory(run_id)/'ai-calls.jsonl').open('a', encoding='utf-8') as handle:
                for call in calls:
                    handle.write(json.dumps(call, ensure_ascii=False)+'\n')
            self.update(run_id, ai_usage=usage_summary(logs), ai_retries=sum(int(c.get('retry_count') or 0) for c in calls))
        except OSError:
            pass

    def diagnose(self, run_id, exc):
        # Local operator diagnostics beside the inputs the run already retains. No
        # endpoint serves this file and it is never part of an exported bundle.
        try:
            with (self.directory(run_id)/'error.log').open('a', encoding='utf-8') as handle:
                handle.write(now()+'\n'+''.join(traceback.format_exception(exc))+'\n')
        except OSError:
            pass

    def retry(self, run_id):
        if self.metadata(run_id)['state'] not in ('FAILED', 'INTERRUPTED'):
            raise HTTPException(409, 'Yalnızca başarısız veya kesilmiş işler yeniden denenebilir.')
        directory = self.directory(run_id)
        config = json.loads((directory/'input.json').read_text(encoding='utf-8'))
        config['policies'] = [dict(name=filename(name), content=base64.b64encode(
            (directory/'inputs'/filename(name)).read_bytes()).decode()) for name in config.pop('policy_names')]
        return self.submit(RunInput.model_validate(config))

    def import_bundle(self, source: Path):
        # Only the operator CLI accepts local paths; the HTTP API never does.
        source = extended(source)
        packet = load_packet(source/'packet.json')
        verify_artifacts(packet, source)
        if packet['count'] != 1:
            raise ValueError('Başlangıç aktarımı özgün analiz paketini gerektirir.')
        if any(r.get('head') == packet['head'] for r in self.list_runs()):
            return
        data = packet['events'][0]['payload']
        run_id = uuid4().hex
        directory = self.directory(run_id)
        directory.mkdir()
        paths = [next((source/'policy-originals').glob(p['raw_hash']+'.*')) for p in data['policies']]
        save_bundle(directory/'result', packet, source/'regulatory-sources', paths)
        write_json(directory/'job.json', {'id': run_id, 'created_at': data['created_at'], 'updated_at': now(),
            'state': 'COMPLETED', 'company_name': data['company']['name'], 'company_id': data['company']['id'],
            'synthetic': data['company']['synthetic'], 'provider': data['runtime']['provider'],
            'source_mode': 'imported', 'previous_id': None, 'reviews': [], 'error': None,
            'regulation': ' '.join((data.get('regulation') or {}).get(key, default) for key, default in (('module', 'CONC'), ('chapter', '7'))),
            'retrieval': 'hybrid' if str((data.get('policy_retrieval') or {}).get('method', '')).startswith('hybrid') else 'lexical',
            'head': packet['head'], 'count': packet['count'], 'candidates': len(data['obligations']),
            'provisions': len(data['cases']), 'policy_count': len(data['policies']),
            'source_dates': sorted({c['source']['fetched_at'] for c in data['cases']})})

    @staticmethod
    def require_local_mode():
        if model_mode() != 'development':
            raise ModelPolicyError('The local workspace requires development mode; company API authentication is not implemented here')

    def models(self, provider_name='ollama'):
        self.require_local_mode()
        return ModelRouter(provider_name, provider_factory=self.provider_factory,
                           judge_factory=self.judge_factory, embedder_factory=self.embedder_factory)

    def ai(self):
        """The local judge model for on-demand questions and drafts; never a remote host."""
        return self.models().judge_provider()

    def ask(self, run_id, question):
        packet = self.packet(run_id)
        data = packet['events'][0]['payload']
        with bind_model_scope(ModelScope(self.tenant_id, data['company']['id'], run_id)):
            return self._ask(run_id, question, data)

    def _ask(self, run_id, question, data):
        _, sections = load_sources(self.result(run_id)/'regulatory-sources')
        embedder = None
        try:
            if os.getenv('EMBED_MODEL'):
                embedder = self.models().embedder()
        except ModelPolicyError:
            raise
        except ValueError:
            embedder = None
        result = answer_question(self.ai(), question, sections, data['policies'], data['obligations'], embedder)
        log = self.result(run_id)/'questions.jsonl'
        with log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'asked_at': now(), **result}, ensure_ascii=False) + '\n')
        return result

    def draft(self, run_id, obligation_id):
        data = self.packet(run_id)['events'][0]['payload']
        with bind_model_scope(ModelScope(self.tenant_id, data['company']['id'], run_id)):
            return self._draft(obligation_id, data)

    def _draft(self, obligation_id, data):
        row = next((o for o in data['obligations'] if o['id'] == obligation_id), None)
        if row is None:
            raise ValueError('Yükümlülük bu analizde yok.')
        sample = next((c['text'] for p in data['policies'] for c in p['chunks'] if len(c['text']) > 80), '')
        return {'obligation_id': obligation_id, 'source_label': row['source_label'], **draft_clause(self.ai(), row['candidate'], sample)}

    def assign(self, run_id, obligation_id, assignee):
        with self.lock:
            data = self.packet(run_id)['events'][0]['payload']
            if not any(o['id'] == obligation_id for o in data['obligations']):
                raise ValueError('Yükümlülük bu analizde yok.')
            assignments = dict(self.metadata(run_id).get('assignments') or {})
            if assignee.strip():
                assignments[obligation_id] = {'assignee': assignee.strip(), 'assigned_at': now()}
            else:
                assignments.pop(obligation_id, None)
            self.update(run_id, assignments=assignments)
            return assignments

    def save_review(self, run_id, review):
        with self.lock:
            packet = self.packet(run_id)
            result = self.result(run_id)
            review_id = digest(review.model_dump())
            directory = result/'reviews'
            directory.mkdir(exist_ok=True)
            path = directory/(review_id+'.json')
            if path.exists():
                reviewed = load_packet(path)
                if reviewed['events'][0]['event_hash'] != packet['head']:
                    raise ValueError('İnceleme başka analize ait.')
            else:
                reviewed = apply_review(packet, review)
                pending = directory/(uuid4().hex+'.tmp')
                write_json(pending, reviewed)
                pending.rename(path)
            row = {'id': review_id, 'head': reviewed['head'], 'count': reviewed['count'],
                'reviewer': review.reviewer, 'saved_at': reviewed['events'][-1]['payload']['created_at']}
            reviews = self.metadata(run_id).get('reviews', [])
            if not any(r['id'] == review_id for r in reviews):
                self.update(run_id, reviews=[*reviews, row])
            return row

    def reviewed_packet(self, run_id, review_id=None):
        base = self.packet(run_id)
        if not review_id:
            return base
        if not HASH.fullmatch(review_id):
            raise ValueError('Geçersiz inceleme kimliği.')
        path = self.result(run_id)/'reviews'/(review_id+'.json')
        if not path.exists():
            raise HTTPException(404, 'İnceleme bulunamadı.')
        result = load_packet(path)
        if result['events'][0]['event_hash'] != base['head']:
            raise ValueError('İnceleme başka analize ait.')
        return result

    def export(self, run_id, review_id=None):
        packet = self.reviewed_packet(run_id, review_id)
        root = self.result(run_id)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for folder in ('regulatory-sources', 'policy-originals'):
                for path in (root/folder).iterdir():
                    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                        raise ValueError('Arşiv dışındaki dosya reddedildi.')
                    if path.is_file():
                        archive.write(path, folder+'/'+path.name)
            archive.writestr('packet.json', json.dumps(packet, ensure_ascii=False, indent=2))
            archive.writestr('review.html', render(packet))
            archive.writestr('labels.json', json.dumps(label_template(packet), indent=2))
            archive.writestr('receipt.json', json.dumps({'head': packet['head'], 'count': packet['count'],
                'notice': 'Preserve head/count separately. No external timestamp, identity or legal assurance.'}))
            # Model call records travel at the configured redaction level: hashes, models, tokens,
            # timings and outcomes, never a prompt, a passage or an answer.
            calls = self.redacted_calls_text(run_id)
            if calls:
                archive.writestr('ai-calls.redacted.jsonl', calls)
            for path in (root/'evaluations').glob('*.json'):
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Arşiv dışındaki değerlendirme reddedildi.')
                envelope = json.loads(path.read_text(encoding='utf-8'))
                expected_hash = envelope.pop('content_hash')
                if digest(envelope) != expected_hash or envelope['analysis_head'] != packet['events'][0]['event_hash']:
                    raise ValueError('Uzman değerlendirmesi değiştirilmiş veya başka analize ait.')
                archive.write(path, 'evaluations/'+path.name)
        return buffer.getvalue()


SAMPLES = Path(__file__).resolve().parents[4] / 'samples'


def sample_scenarios():
    """Ready-made demo inputs (synthetic company, policies with known conflicts, regulation).

    Users uploaded the regulation as the "policy" four times running. One click now loads a
    complete, realistic scenario so the pipeline can be seen working before real documents
    are prepared. The files ship with the repository; nothing is generated here.
    """
    index = SAMPLES / 'scenarios.json'
    if not index.exists():
        return []
    scenarios = []
    for item in json.loads(index.read_text(encoding='utf-8')):
        folder = SAMPLES / item['folder']
        files = sorted(folder.glob('*.md')) + sorted(folder.glob('*.txt')) if folder.is_dir() else []
        if not files:
            continue
        scenarios.append({**item, 'policies': [{'name': f.name, 'content': base64.b64encode(f.read_bytes()).decode('ascii')} for f in files]})
    return scenarios


def create_app(workspace: Workspace, token: str, port=8767):
    hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
    origins = {'http://'+host for host in hosts}

    @asynccontextmanager
    async def lifespan(app):
        workspace.start()
        try:
            yield
        finally:
            workspace.close()

    app = FastAPI(title='Cardaman local workspace', version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.workspace = workspace

    @app.middleware('http')
    async def guard(request: Request, call_next):
        # Every response names its request: the client's X-Request-ID when well formed, else
        # an id derived from tenant, method, path, body hash and the moment (deterministic
        # for the same request at the same second, never a random value).
        started = now()
        supplied = request.headers.get('x-request-id')
        request_id = supplied if supplied and REQUEST_ID.fullmatch(supplied) else None

        def reject(status, detail):
            return JSONResponse({'detail': detail}, status_code=status,
                                headers={'Cache-Control': 'no-store', 'X-Request-ID': request_id or new_request_id()})
        if not request.client or request.client.host not in ('127.0.0.1', '::1'):
            return reject(403, 'Yalnızca bu bilgisayardan erişilebilir.')
        if request.headers.get('host') not in hosts:
            return reject(403, 'Host reddedildi.')
        if (request.headers.get('origin') and request.headers['origin'] not in origins
                or request.headers.get('sec-fetch-site') == 'cross-site'):
            return reject(403, 'Başka siteden gelen istek reddedildi.')
        if request.url.path.startswith('/api/') and not secrets.compare_digest(
                request.headers.get('authorization', ''), 'Bearer '+token):
            return reject(401, 'Başlatıcının verdiği oturum bağlantısını açın.')
        if request.method == 'POST':
            if request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return reject(415, 'JSON isteği gerekli.')
            body = bytearray()
            async for chunk in request.stream():
                if len(body)+len(chunk) > MAX_REQUEST:
                    return reject(413, 'Yükleme boyutu sınırı aşıldı.')
                body.extend(chunk)
            request._body = bytes(body)
        if request_id is None:
            from hashlib import sha256
            body_hash = sha256(getattr(request, '_body', b'') or b'').hexdigest()
            request_id = deterministic_request_id(workspace.tenant_id, request.method, request.url.path, body_hash, started)
        request.state.context = workspace.context(request_id)
        response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY',
            'Content-Security-Policy': "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self' data:; frame-src 'self' blob:; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'"})
        return response

    @app.exception_handler(PermissionDenied)
    async def permission_denied(request, exc):
        # A refused action is an audit entry too; the response never says what else the role could do.
        context = getattr(request.state, 'context', None)
        try:
            workspace.audit(getattr(exc, 'permission', 'permission'), request.url.path, 'DENIED', '', context)
        except (OSError, ValueError):
            pass
        return JSONResponse({'detail': 'Bu işlem için yetki yok: ' + str(getattr(exc, 'permission', ''))}, status_code=403)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({'detail': str(exc)[:800]}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic normally echoes invalid input, which could contain a whole policy.
        errors = [{'field': '.'.join(map(str, e['loc'])), 'message': e['msg']} for e in exc.errors()]
        return JSONResponse({'detail': errors}, status_code=422)

    @app.exception_handler(OSError)
    async def file_error(request, exc):
        return JSONResponse({'detail': 'Kanıt dosyası okunamadı veya kaydedilemedi.'}, status_code=409)

    @app.exception_handler(ProviderFailure)
    async def model_unavailable(request, exc):
        # An on-demand question or draft whose model call failed (timeout, truncated output, Ollama
        # down, circuit open) was an HTTP 500 with no message: the local AI is unavailable for the
        # moment, not the server broken, so 503 says which failure it was and that nothing was lost.
        code = getattr(exc, 'code', None) or getattr(exc, 'kind', None) or type(exc).__name__
        cause = str(exc)[:300].rstrip(' .') or type(exc).__name__
        return JSONResponse({'detail': f'Yerel AI modeli yanıt veremedi ({code}: {cause}). Kayıtlı analizler etkilenmedi; '
                             "Ollama'nın açık olduğunu kontrol edip birazdan tekrar deneyin.",
                             'failure_code': code}, status_code=503)

    @app.get('/', response_class=HTMLResponse)
    def index():
        return Path(__file__).with_name('workspace_ui.html').read_text(encoding='utf-8')

    @app.get('/api/state')
    def state():
        return {'runs': workspace.list_runs(), 'active': workspace.active, 'samples': [s['id'] for s in sample_scenarios()],
            'cached_sources': bool(workspace.cached_sources), 'model': os.getenv('LLM_MODEL', ''),
            'embed_model': os.getenv('EMBED_MODEL', ''),
            'judge_model': os.getenv('JUDGE_MODEL', ''), 'judge_thinking': os.getenv('JUDGE_THINKING', ''),
            'rerank_model': os.getenv('RERANK_MODEL', ''),
            'max_policy_bytes': MAX_BYTES, 'max_total_bytes': MAX_UPLOAD, **version_state()}

    @app.post('/api/runs', status_code=202)
    def submit(value: RunInput, request: Request):
        context = require('start_analysis', request.state.context)
        try:
            row = workspace.submit(value)
        except (ValueError, HTTPException) as exc:
            workspace.audit('start_analysis', value.company.id[:64], 'REFUSED', type(exc).__name__, context)
            raise
        workspace.audit('start_analysis', row['id'], 'OK', f'{row["regulator"]} {row["regulation"]}'[:200], context)
        return row

    @app.post('/api/runs/{run_id}/ask')
    def ask(run_id: str, value: AskInput):
        if not RUN_ID.fullmatch(run_id):
            raise HTTPException(404, 'Analiz yok.')
        return workspace.ask(run_id, value.question)

    @app.post('/api/runs/{run_id}/draft')
    def draft(run_id: str, value: DraftInput):
        if not RUN_ID.fullmatch(run_id):
            raise HTTPException(404, 'Analiz yok.')
        return workspace.draft(run_id, value.obligation_id)

    @app.post('/api/runs/{run_id}/assign')
    def assign(run_id: str, value: AssignInput):
        if not RUN_ID.fullmatch(run_id):
            raise HTTPException(404, 'Analiz yok.')
        return {'assignments': workspace.assign(run_id, value.obligation_id, value.assignee)}

    @app.get('/api/mevzuat/search')
    def mevzuat_search(q: str, kind: str = 'yonetmelik'):
        # The operator's search words go to the public catalogue; nothing else does.
        return {'results': search_mevzuat(q, kind)}

    @app.get('/api/gazette')
    def gazette(days: int = 7, q: str = ''):
        return gazette_scan(days, q)

    @app.get('/api/samples')
    def samples():
        return {'scenarios': sample_scenarios()}

    @app.post('/api/runs/{run_id}/retry', status_code=202)
    def retry(run_id: str):
        return workspace.retry(run_id)

    @app.post('/api/runs/{run_id}/stop', status_code=202)
    def stop(run_id: str):
        return workspace.stop(run_id)

    @app.get('/api/runs/{run_id}')
    def details(run_id: str):
        return workspace.metadata(run_id)

    @app.get('/api/runs/{run_id}/packet')
    def packet(run_id: str, review_id: str | None = None):
        return workspace.reviewed_packet(run_id, review_id)

    @app.get('/api/runs/{run_id}/report', response_class=HTMLResponse)
    def report(run_id: str, review_id: str | None = None):
        return render(workspace.reviewed_packet(run_id, review_id), hosted=not review_id)

    @app.post('/api/runs/{run_id}/reviews')
    def review(run_id: str, value: Review, request: Request):
        context = require('save_review', request.state.context)
        try:
            row = workspace.save_review(run_id, value)
        except (ValueError, HTTPException) as exc:
            workspace.audit('save_review', run_id if RUN_ID.fullmatch(run_id) else '?', 'REFUSED', type(exc).__name__, context)
            raise
        actions = {}
        for decision in value.decisions:
            actions[decision.action or 'LEGACY'] = actions.get(decision.action or 'LEGACY', 0) + 1
        workspace.audit('save_review', run_id, 'OK', f'review {row["id"][:12]} ' + ' '.join(f'{k}={v}' for k, v in sorted(actions.items())), context)
        return row

    @app.get('/api/runs/{run_id}/ai-calls')
    def ai_calls(run_id: str, request: Request, level: str | None = None):
        context = require('read_ai_logs', request.state.context)
        if not RUN_ID.fullmatch(run_id):
            raise ValueError('Geçersiz analiz kimliği.')
        entries = workspace.ai_calls(run_id, level)
        return {'run_id': run_id, 'level': (level or workspace.redaction.level), 'calls': entries,
                'note': 'Model çağrı kaydı: istem, pasaj ya da yanıt metni hiçbir düzeyde tutulmaz.'}

    @app.delete('/api/runs/{run_id}')
    def delete_run(run_id: str, request: Request):
        context = require('delete_data', request.state.context)
        return {'deleted': workspace.delete([run_id], context)}

    @app.get('/api/retention')
    def retention(request: Request):
        require('view_runs', request.state.context)
        return {'days': workspace.retention.days, 'keep_reviewed': workspace.retention.keep_reviews,
                'redaction_level': workspace.redaction.level, 'expired': workspace.expired_runs()}

    @app.post('/api/retention/sweep')
    def sweep(request: Request):
        context = require('delete_data', request.state.context)
        expired = workspace.expired_runs()
        if not expired:
            return {'deleted': [], 'days': workspace.retention.days}
        return {'deleted': workspace.delete(expired, context), 'days': workspace.retention.days}

    @app.get('/api/tenant/export')
    def export_tenant(request: Request):
        context = require('export_data', request.state.context)
        return Response(workspace.export_all(context), media_type='application/zip',
            headers={'Content-Disposition': f'attachment; filename="cardaman-{workspace.tenant_id}-export.zip"'})

    @app.get('/api/audit')
    def audit_entries(request: Request):
        require('view_runs', request.state.context)
        return {'entries': workspace.audit_log.entries(), 'chain_valid': workspace.audit_log.verify()}

    @app.get('/api/runs/{run_id}/labels')
    def labels(run_id: str):
        return label_template(workspace.packet(run_id))

    @app.post('/api/runs/{run_id}/evaluate')
    def evaluation(run_id: str, value: dict):
        # Labels are not promoted to human review or identity verification.
        try:
            result = evaluate(workspace.packet(run_id), value)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('Uzman etiket dosyasının yapısı geçersiz.') from exc
        envelope = {'analysis_head': value['analysis_head'], 'labels': value, 'metrics': result,
                    'created_at': now(), 'identity_assurance': 'LOCAL_OPERATOR_ASSERTION'}
        envelope['content_hash'] = digest(envelope)
        folder = workspace.result(run_id)/'evaluations'
        folder.mkdir(exist_ok=True)
        write_json(folder/(uuid4().hex+'.json'), envelope)
        return envelope

    @app.get('/api/runs/{run_id}/verify')
    def verify(run_id: str, review_id: str | None = None, expected_head: str | None = None,
               expected_count: int | None = None):
        packet = workspace.reviewed_packet(run_id, review_id)
        if (expected_head is None) != (expected_count is None):
            raise ValueError('Beklenen hash ve olay sayısı birlikte girilmeli.')
        if expected_head is not None and not verify_chain(packet['events'], expected_head, expected_count):
            raise ValueError('Paket ayrı saklanan hash / olay sayısıyla eşleşmiyor.')
        return {'integrity': 'VERIFIED', 'head': packet['head'], 'count': packet['count'],
            'external_reference_checked': expected_head is not None,
            'notice': 'Paket ve kaynak/policy dosyaları tutarlı. Kimlik, hukuki doğruluk veya harici zaman damgası doğrulanmadı.'}

    @app.get('/api/runs/{run_id}/export')
    def export(run_id: str, request: Request, review_id: str | None = None):
        context = require('export_data', request.state.context)
        data = workspace.export(run_id, review_id)
        workspace.audit('export_data', run_id, 'OK', f'{len(data)} bytes' + (f' review {review_id[:12]}' if review_id else ''), context)
        return Response(data, media_type='application/zip',
            headers={'Content-Disposition': f'attachment; filename="cardaman-{run_id}.zip"'})

    return app


def running_instance(root):
    """Operator hint when the lock is held. The marker never holds the session token."""
    try:
        row = json.loads((root/'server.json').read_text(encoding='utf-8'))
        return (f"Çalışan sunucu: PID {int(row['pid'])}, port {int(row['port'])}, başlangıç {row['started_at']}. "
            'O terminaldeki oturum bağlantısını kullan. Terminal kayıpsa süreci kapatıp yeniden başlat: '
            f"Stop-Process -Id {int(row['pid'])}")
    except (OSError, ValueError, KeyError, TypeError):
        return ('Çalışan sunucunun terminalindeki oturum bağlantısını kullan; '
            'terminal kayıpsa o python sürecini kapatıp yeniden başlat.')


# v0.19: the product pipeline is set by the server itself, so it no longer depends on which Cardaman.exe
# started it. The exe of 22 September 2026 was built before the v0.18 keys existed and silently ran the
# v0.17 pipeline (APPLICABILITY_CLEAR_MATCH and RELEVANCE_SCREEN unset). Code defaults keep the v0.18
# behaviour for tests and library callers; an explicit value (.env, shell, launcher) always wins.
PRODUCT_DEFAULTS = {'APPLICABILITY_CLEAR_MATCH': 'rule', 'RELEVANCE_SCREEN': 'on', 'COVERAGE_PIPELINE': 'v19'}
# v0.19 judge window (user directive of 24 September 2026; extraction.providers.fit_call): qwen3:8b judges
# at 8,192 tokens, all on the 8 GB GPU (at 16,384 a fifth of it ran on the CPU), and 'adaptive' sends only a
# prompt that does not fit even after compression to 16,384. Filled only when a judge model is configured.
# JUDGE_CTX_MODE awaits the 8-case fixed-8k vs adaptive comparison: 'fixed' or 'adaptive' is this one value.
JUDGE_WINDOW_DEFAULTS = {'JUDGE_NUM_CTX': '8192', 'JUDGE_NUM_CTX_LARGE': '16384', 'JUDGE_CTX_MODE': 'adaptive'}


def product_defaults(environ=None):
    """Fill the product switches that are unset or empty (the settings read an empty value as unset,
    `os.getenv(key) or default`, and the launchers drop empty .env values); returns the keys filled.
    The judge window defaults follow when JUDGE_MODEL is set; the adaptive mode is left out when an
    explicit JUDGE_NUM_CTX leaves nothing larger to fall back to (configured_judge would refuse it)."""
    environ = os.environ if environ is None else environ
    filled = [key for key in PRODUCT_DEFAULTS if not (environ.get(key) or '').strip()]
    for key in filled:
        environ[key] = PRODUCT_DEFAULTS[key]
    if (environ.get('JUDGE_MODEL') or '').strip():
        for key, value in JUDGE_WINDOW_DEFAULTS.items():
            if (environ.get(key) or '').strip():
                continue
            if key == 'JUDGE_CTX_MODE' and value == 'adaptive':
                try:
                    if int(environ['JUDGE_NUM_CTX']) >= int(environ['JUDGE_NUM_CTX_LARGE']):
                        continue
                except ValueError:
                    continue
            environ[key] = value
            filled.append(key)
    return filled


def version_state():
    """The version part of /api/state. Cardaman.exe passes its BuildInfo.Version as
    CARDAMAN_LAUNCHER_VERSION; no value (an exe older than v0.19, Run-Workspace.ps1, python -m) is
    not a mismatch, only a different one is. The pipeline switches are the ones a run started now uses."""
    launcher = os.getenv('CARDAMAN_LAUNCHER_VERSION', '').strip()
    try:
        settings = pipeline_settings()
    except ValueError as exc:
        settings = {'error': str(exc)[:300]}
    return {'version': __version__, 'version_display': display_version(), 'launcher_version': launcher,
            'version_mismatch': bool(launcher) and launcher != __version__, 'pipeline_settings': settings}


def main():
    parser = argparse.ArgumentParser(description='Local-only Cardaman company/policy workspace')
    parser.add_argument('--root', required=True)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--cached-sources')
    parser.add_argument('--import-bundle', action='append', default=[])
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be between 1024 and 65535')
    product_defaults()
    # Its own line, before the session link: the launcher's link pattern (^Cardaman:\s+http…) never
    # matches it, so launchers built before v0.19 still start this server.
    print(f'Cardaman version: {__version__} ({display_version()})', flush=True)
    workspace = Workspace(Path(args.root), args.cached_sources)
    # Imports occur under the same process lock used by the running app.
    try:
        workspace.start()
    except WorkspaceBusy as exc:
        # An expected operator situation, not a crash: name the instance, no traceback.
        parser.exit(1, str(exc)+'\n'+running_instance(workspace.root)+'\n')
    try:
        for source in args.import_bundle:
            workspace.import_bundle(Path(source))
    except Exception:
        workspace.close()
        raise
    token = secrets.token_urlsafe(32)
    marker = workspace.root/'server.json'
    marker.write_text(json.dumps({'pid': os.getpid(), 'port': args.port, 'started_at': now(), 'version': __version__}), encoding='utf-8')
    print(f'Cardaman: http://127.0.0.1:{args.port}/#token={token}', flush=True)
    print('Local session link. Stop with Ctrl+C; analyses are retained on disk.', flush=True)
    import uvicorn
    try:
        uvicorn.run(create_app(workspace, token, args.port), host='127.0.0.1', port=args.port,
                    access_log=False, proxy_headers=False)
    finally:
        marker.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

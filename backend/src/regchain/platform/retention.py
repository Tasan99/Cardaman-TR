"""Retention window, export of run directories and audited deletion.

A run directory is the unit of retention: inputs, sources, results, reviews and the AI call
log of one analysis live under ``runs/<32 hex>``. Export writes the parts that carry evidence
(job.json, input.json, result/, the AI call log) with a manifest of SHA-256 per file; the raw
``inputs/`` copy is already inside ``result/policy-originals`` and ``error.log`` is operator
diagnostics that never leaves the machine (``Workspace.diagnose``). Given a redaction policy,
the call log leaves as ``ai-calls.redacted.jsonl`` at that level, like the single-run export.
Deletion is audited per run and confined to ``runs_dir``.
"""
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from regchain.pilot.paths import extended
from .audit import AuditLog
from .context import RequestContext, now
from .redaction import RedactionPolicy, redact_call_log

RUN_ID = re.compile(r'^[0-9a-f]{32}$')
EXPORT_FORMAT = 'cardaman-retention-export-v1'
CALL_LOG = 'ai-calls.jsonl'
# The name the single-run export gives the redacted log; a tenant export uses the same one.
REDACTED_CALL_LOG = 'ai-calls.redacted.jsonl'
EXPORTED = ('job.json', 'input.json', CALL_LOG)
EXPORTED_DIRS = ('result',)
MAX_DAYS = 36500


@dataclass(frozen=True)
class RetentionSettings:
    """``days`` None keeps everything forever; ``keep_reviews`` exempts runs with a saved review,
    because a recorded decision is the record a regulator asks for, not the analysis under it."""
    days: int | None = None
    keep_reviews: bool = True

    def __post_init__(self):
        if self.days is not None and (not isinstance(self.days, int) or isinstance(self.days, bool)
                                      or not 1 <= self.days <= MAX_DAYS):
            raise ValueError(f'RETENTION_DAYS must be an integer between 1 and {MAX_DAYS}, or unset')

    @classmethod
    def from_env(cls, environ=None) -> 'RetentionSettings':
        environ = os.environ if environ is None else environ
        raw = (environ.get('RETENTION_DAYS') or '').strip()
        try:
            days = int(raw) if raw else None
        except ValueError as exc:
            raise ValueError('RETENTION_DAYS must be a whole number of days') from exc
        setting = (environ.get('RETENTION_KEEP_REVIEWED') or '').strip().lower()
        if setting not in ('', 'on', 'off'):
            raise ValueError('RETENTION_KEEP_REVIEWED must be on or off')
        return cls(days, setting != 'off')

    def expired(self, rows, now) -> list:
        return expired(rows, now, self)


def _moment(value) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def expired(rows, now, settings: RetentionSettings) -> list:
    """The run rows older than the window, in the given order. A row without a readable
    ``created_at`` is never selected (what cannot be dated is not deleted), and with
    ``keep_reviews`` neither is a row whose ``reviews`` list is non-empty."""
    if settings.days is None:
        return []
    moment = _moment(now)
    if moment is None:
        raise ValueError('now must be a datetime or an ISO 8601 string')
    cutoff = moment - timedelta(days=settings.days)
    selected = []
    for row in rows:
        created = _moment(row.get('created_at'))
        if created is None or created >= cutoff:
            continue
        if settings.keep_reviews and row.get('reviews'):
            continue
        selected.append(row)
    return selected


def _run_directory(runs_dir: Path, run_id: str) -> Path:
    """The run's directory under ``runs_dir`` or a ValueError; never a path outside it."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError('Geçersiz analiz kimliği.')
    candidate = runs_dir / run_id
    if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()) \
            or not candidate.resolve().is_relative_to(runs_dir):
        raise ValueError('Çalışma klasörü dışına erişim reddedildi.')
    return candidate


def _files_of(directory: Path, root: Path):
    """Regular files under ``directory`` in sorted order; a symlink or an escape refuses the export."""
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()) \
                or not path.resolve().is_relative_to(root):
            raise ValueError('Arşiv dışındaki dosya reddedildi.')
        if path.is_file():
            yield path


def export_runs(runs_dir, run_ids, destination_zip, redaction: RedactionPolicy | None = None) -> dict:
    """Write ``destination_zip`` with ``<run id>/...`` for every named run and a top-level
    manifest.json (format, created_at, run ids, redaction level, sha256 per archived file). The
    destination must not exist yet and must not be inside ``runs_dir``; the zip is built beside it
    and moved into place, so a failed export leaves no half-written archive under the final name.

    With ``redaction`` each run's ai-calls.jsonl is archived as ``ai-calls.redacted.jsonl`` at that
    level, the bytes the single-run export writes, and the manifest hashes those bytes. Without it
    the log is copied as it is: an operator's own copy on this machine. Anything that leaves the
    machine passes a policy; the workspace's tenant export passes its configured one."""
    runs_dir = extended(runs_dir)
    destination = extended(Path(destination_zip).parent) / Path(destination_zip).name
    if destination.exists():
        raise ValueError('Dışa aktarma dosyası zaten var.')
    if destination.resolve().is_relative_to(runs_dir):
        raise ValueError('Dışa aktarma dosyası analiz klasörünün içine yazılamaz.')
    if not run_ids:
        raise ValueError('Dışa aktarılacak analiz seçilmedi.')
    directories = {}
    for run_id in run_ids:
        directory = _run_directory(runs_dir, run_id)
        if not (directory / 'job.json').is_file():
            raise ValueError('Analiz bulunamadı.')
        directories[run_id] = directory
    manifest = {'format': EXPORT_FORMAT, 'created_at': now(), 'runs': list(directories),
                'ai_calls_redaction': redaction.level if redaction is not None else None, 'files': {}}
    pending = destination.with_name('pending-' + uuid4().hex + '.zip')
    try:
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as archive:
            for run_id, directory in directories.items():
                files = [directory / name for name in EXPORTED if (directory / name).is_file()]
                for folder in EXPORTED_DIRS:
                    if (directory / folder).is_dir():
                        files.extend(_files_of(directory / folder, directory))
                for path in files:
                    if path.is_symlink() or not path.resolve().is_relative_to(directory):
                        raise ValueError('Arşiv dışındaki dosya reddedildi.')
                    data = path.read_bytes()
                    arcname = run_id + '/' + path.relative_to(directory).as_posix()
                    if redaction is not None and path == directory / CALL_LOG:
                        data = redact_call_log(data.decode('utf-8'), redaction).encode('utf-8')
                        if not data:
                            continue
                        arcname = run_id + '/' + REDACTED_CALL_LOG
                    archive.writestr(arcname, data)
                    manifest['files'][arcname] = sha256(data).hexdigest()
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        os.replace(pending, destination)
    finally:
        if pending.exists():
            pending.unlink()
    return manifest


def delete_runs(runs_dir, run_ids, audit: AuditLog, context: RequestContext) -> list:
    """Remove the named run directories and return ``[{'id', 'outcome'}, ...]``.

    Every id is validated before anything is touched: one malformed id refuses the whole call
    (recorded as REFUSED) because it means a bug or an attacker, not a stale run. A run whose
    directory is a symlink or resolves outside ``runs_dir`` is skipped and recorded as REFUSED;
    a missing run is NOT_FOUND; a directory the OS would not remove is FAILED. Whether a run is
    still active is the workspace's knowledge, so the caller checks that before calling.
    """
    runs_dir = extended(runs_dir)
    run_ids = list(run_ids)
    for run_id in run_ids:
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            shown = re.sub(r'[^A-Za-z0-9._-]', '?', str(run_id))[:64]
            audit.record('delete_run', shown, 'REFUSED', 'invalid run id', context=context)
            raise ValueError('Geçersiz analiz kimliği.')
    deleted = []
    for run_id in run_ids:
        try:
            directory = _run_directory(runs_dir, run_id)
        except ValueError:
            audit.record('delete_run', run_id, 'REFUSED', 'path outside runs directory', context=context)
            deleted.append({'id': run_id, 'outcome': 'REFUSED'})
            continue
        if not directory.is_dir():
            audit.record('delete_run', run_id, 'NOT_FOUND', '', context=context)
            deleted.append({'id': run_id, 'outcome': 'NOT_FOUND'})
            continue
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            code = getattr(exc, 'winerror', None) or exc.errno
            audit.record('delete_run', run_id, 'FAILED', type(exc).__name__ + (f' {code}' if code else ''), context=context)
            deleted.append({'id': run_id, 'outcome': 'FAILED'})
            continue
        audit.record('delete_run', run_id, 'OK', 'run directory removed', context=context)
        deleted.append({'id': run_id, 'outcome': 'OK'})
    return deleted

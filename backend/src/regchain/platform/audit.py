"""Append-only audit log: one JSON line per action, hash-chained like the evidence packets.

Each entry carries the previous entry's hash and its own hash over the canonical JSON of
``regchain.evidence`` (same rules: no floats, sorted keys, UTF-8). Removing, editing or
reordering a line breaks ``verify()``. ``detail`` is a short operator string, never document
content: the audit log is meant to be readable by people who may not read the documents.
"""
import json
import os
import threading
from pathlib import Path

from regchain.evidence import GENESIS, digest
from .context import RequestContext, current_context, now

FIELDS = ('at', 'request_id', 'tenant_id', 'actor', 'action', 'subject', 'outcome', 'detail', 'previous_hash', 'entry_hash')
# A closed vocabulary keeps the log queryable; REFUSED is a request that failed validation
# (bad id, path escape), DENIED a permission failure, FAILED an error after admission.
OUTCOMES = frozenset({'OK', 'DENIED', 'REFUSED', 'FAILED', 'NOT_FOUND'})
DETAIL_LIMIT = 200
SUBJECT_LIMIT = 128


class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        # Appends within one process are serialised here. A second process appending to the
        # same file would fork the chain, and verify() is what detects that; run one writer.
        self.lock = threading.Lock()
        self._head = None

    def _load_head(self) -> str:
        if self._head is None:
            rows = self.entries()
            self._head = rows[-1]['entry_hash'] if rows else GENESIS
        return self._head

    def record(self, action: str, subject: str, outcome: str, detail: str = '',
               context: RequestContext | None = None, at: str | None = None) -> dict:
        """Append one entry and return it. Needs a request context (given or bound): an entry
        without an actor is not an audit entry, so a system job binds one with its own actor."""
        context = context or current_context()
        if context is None:
            raise ValueError('An audit entry needs a request context (bind_context or pass context=)')
        if not action or not isinstance(action, str) or len(action) > 64:
            raise ValueError('action must be a short string')
        if outcome not in OUTCOMES:
            raise ValueError(f'outcome must be one of {sorted(OUTCOMES)}')
        if not isinstance(subject, str) or not isinstance(detail, str):
            raise ValueError('subject and detail must be strings')
        with self.lock:
            previous = self._load_head()
            body = {'at': at or now(), 'request_id': context.request_id, 'tenant_id': context.tenant_id,
                    'actor': context.actor, 'action': action, 'subject': subject[:SUBJECT_LIMIT],
                    'outcome': outcome, 'detail': detail[:DETAIL_LIMIT], 'previous_hash': previous}
            entry = {**body, 'entry_hash': digest(body)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            self._head = entry['entry_hash']
            return entry

    def entries(self) -> list:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open('r', encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f'Denetim kaydı bozuk (satır {number}).') from exc
                rows.append(row)
        return rows

    def verify(self) -> bool:
        """True when every line is intact and chained from GENESIS to the last entry."""
        try:
            rows = self.entries()
        except (ValueError, OSError):
            return False
        previous = GENESIS
        for row in rows:
            if not isinstance(row, dict) or tuple(sorted(row)) != tuple(sorted(FIELDS)):
                return False
            if row['previous_hash'] != previous or row['outcome'] not in OUTCOMES:
                return False
            body = {key: value for key, value in row.items() if key != 'entry_hash'}
            try:
                if digest(body) != row['entry_hash']:
                    return False
            except ValueError:
                return False
            previous = row['entry_hash']
        return True

    @property
    def head(self) -> str:
        with self.lock:
            return self._load_head()

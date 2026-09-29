"""Tenant-prefixed, content-addressed document storage with containment checks.

The key of a document is the SHA-256 of its bytes plus a sanitised file name, so the same
upload is stored once, a key can be checked against the bytes it returns, and no client-chosen
string ever becomes a path on its own. A key is only meaningful under the tenant that wrote it:
``get(tenant_b, key_of_a)`` looks under tenant_b's directory and finds nothing.
"""
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from regchain.pilot.paths import extended

# Lower case only: Windows and macOS merge "Acme" and "acme" into one directory, which would
# merge two tenants.
TENANT_ID = re.compile(r'^[a-z0-9][a-z0-9_-]{0,63}$')
KEY = re.compile(r'^[0-9a-f]{64}-[A-Za-z0-9._-]{1,80}$')
SAFE_NAME = re.compile(r'[^A-Za-z0-9._-]+')
MAX_BYTES = 24 * 1024 * 1024


class StorageRefused(ValueError):
    """Validation or containment failure; distinct from a key that simply does not exist."""


class DocumentNotFound(LookupError):
    pass


class DocumentStore(Protocol):
    def put(self, tenant_id: str, name: str, data: bytes) -> str: ...
    def get(self, tenant_id: str, key: str) -> bytes: ...
    def delete(self, tenant_id: str, key: str) -> None: ...
    def list(self, tenant_id: str) -> list: ...


def sanitized_name(name: str) -> str:
    """A file name made of [A-Za-z0-9._-], at most 80 characters, never empty and never
    starting with a dot; the original name is metadata the caller keeps, not part of the path."""
    if not isinstance(name, str):
        raise StorageRefused('Belge adı metin olmalı.')
    cleaned = SAFE_NAME.sub('_', name.strip()).strip('._')[:80]
    return cleaned or 'document'


def content_key(name: str, data: bytes) -> str:
    return sha256(data).hexdigest() + '-' + sanitized_name(name)


def _escapes(path: Path, base: Path) -> bool:
    """True when ``path`` is a symlink or junction, or resolves outside ``base``."""
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        return True
    return not path.resolve().is_relative_to(base)


class LocalDocumentStore:
    def __init__(self, root):
        # Extended form on Windows so hash-named files under a deep root stay addressable, and
        # so every containment check compares paths of one namespace.
        self.root = extended(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def tenant_dir(self, tenant_id: str) -> Path:
        if not isinstance(tenant_id, str) or not TENANT_ID.fullmatch(tenant_id):
            raise StorageRefused('Geçersiz kuruluş kimliği.')
        directory = self.root / tenant_id
        if directory.exists() and _escapes(directory, self.root):
            raise StorageRefused('Depo dışına erişim reddedildi.')
        return directory

    def path_of(self, tenant_id: str, key: str) -> Path:
        directory = self.tenant_dir(tenant_id)
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise StorageRefused('Geçersiz belge anahtarı.')
        candidate = directory / key
        if candidate.exists() and _escapes(candidate, directory):
            raise StorageRefused('Depo dışına erişim reddedildi.')
        return candidate

    def put(self, tenant_id: str, name: str, data: bytes) -> str:
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise StorageRefused('Boş belge kaydedilmez.')
        if len(data) > MAX_BYTES:
            raise StorageRefused('Belge 24 MiB sınırını aşıyor.')
        key = content_key(name, bytes(data))
        path = self.path_of(tenant_id, key)
        if path.is_file():
            return key
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside its final name and moved into place, so a reader never sees a partial file.
        pending = path.with_name('pending-' + uuid4().hex + '.tmp')
        pending.write_bytes(bytes(data))
        os.replace(pending, path)
        return key

    def get(self, tenant_id: str, key: str) -> bytes:
        path = self.path_of(tenant_id, key)
        if not path.is_file():
            raise DocumentNotFound('Belge bulunamadı.')
        data = path.read_bytes()
        # The key names the bytes: a file edited on disk no longer matches and is refused.
        if sha256(data).hexdigest() != key[:64]:
            raise StorageRefused('Belge içeriği anahtarıyla eşleşmiyor.')
        return data

    def delete(self, tenant_id: str, key: str) -> None:
        path = self.path_of(tenant_id, key)
        if not path.is_file():
            raise DocumentNotFound('Belge bulunamadı.')
        path.unlink()

    def list(self, tenant_id: str) -> list:
        directory = self.tenant_dir(tenant_id)
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir()
                      if KEY.fullmatch(p.name) and p.is_file() and not _escapes(p, directory))

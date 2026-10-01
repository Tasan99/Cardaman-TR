"""Private model policy and lazy access to Cardaman's existing model roles.

This module has no eager provider imports: the chat and embedding adapters share
its validation without changing their extraction, judging or retrieval behavior.
Server mode is a model boundary; callers must supply an authenticated scope.
"""

import contextvars
import json
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import urlsplit


class ModelPolicyError(ValueError):
    """A private model boundary was violated; never contains document text."""


def model_mode(environ=None) -> str:
    """An absent mode preserves local development; unknown values fail closed."""
    env = os.environ if environ is None else environ
    mode = env.get('CARDAMAN_MODE', 'development')
    if mode not in ('development', 'server'):
        raise ModelPolicyError('CARDAMAN_MODE must be development or server')
    return mode


_LOCAL_AUTHORITY = re.compile(r'(localhost|127\.0\.0\.1|host\.docker\.internal|\[::1\])(?::([0-9]+))?', re.I)


def _unsafe_characters(value: str) -> bool:
    return any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)


def validate_ollama_endpoint(base_url: str, *, environ=None) -> str:
    """Return a local Ollama origin; an HTTPS URL never grants remote access."""
    model_mode(environ)
    if (not isinstance(base_url, str) or not base_url or _unsafe_characters(base_url)
            or any(character in base_url for character in ('\\', '?', '#'))):
        raise ModelPolicyError('Private models require a local Ollama origin without URL credentials or parameters')
    try:
        url = urlsplit(base_url)
        authority = _LOCAL_AUTHORITY.fullmatch(url.netloc)
        if (url.scheme not in ('http', 'https') or authority is None or url.path not in ('', '/')
                or url.username is not None or url.password is not None):
            raise ValueError('not a local origin')
        port = url.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError('invalid port')
    except ValueError as exc:
        raise ModelPolicyError('Private models require local Ollama; remote transfer is disabled') from exc
    host = authority.group(1).lower()
    return f'{url.scheme}://{host}' + (f':{port}' if port is not None else '')


_MODEL_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?(?::[A-Za-z0-9][A-Za-z0-9_.-]*)?')
_CLOUD_NAME = re.compile(r'(?:^|[/:._-])cloud(?:$|[/:._-])', re.I)
_SHA256 = re.compile(r'(?:sha256:)?[a-fA-F0-9]{64}')


def _validate_local_model_name(model):
    if (not isinstance(model, str) or not _MODEL_NAME.fullmatch(model) or _CLOUD_NAME.search(model)
            or ('/' in model and '.' in model.split('/', 1)[0])):
        raise ModelPolicyError('Model names must refer to a local model; cloud models and remote registry paths are disabled')


def validate_model(model: str, model_digest: str, *, kind='chat', environ=None) -> None:
    """Require pinned local models, and restrict server models to the approved set."""
    mode = model_mode(environ)
    if kind not in ('chat', 'embedding'):
        raise ModelPolicyError('Model kind must be chat or embedding')
    _validate_local_model_name(model)
    if not isinstance(model_digest, str) or not model_digest or _unsafe_characters(model_digest):
        raise ModelPolicyError('A model digest is required for reproducible local inference')
    if mode == 'server':
        approved = ('qwen3:4b', 'qwen3:8b') if kind == 'chat' else ('bge-m3', 'bge-m3:latest')
        if model not in approved:
            raise ModelPolicyError('Server mode requires the approved local Qwen3 or BGE-M3 model for this role')
        if not _SHA256.fullmatch(model_digest):
            raise ModelPolicyError('Server model digests must be complete SHA256 digests')


def model_names_match(actual, configured) -> bool:
    """Match local inventory names, including Ollama's implicit latest tag.

    Validate policy separately before accepting an inventory entry. This helper
    does not grant permission to use either name or any cloud-backed metadata.
    """
    if not isinstance(actual, str) or not isinstance(configured, str):
        return False
    if not _MODEL_NAME.fullmatch(actual) or not _MODEL_NAME.fullmatch(configured):
        return False
    actual = actual if ':' in actual else actual + ':latest'
    configured = configured if ':' in configured else configured + ':latest'
    return actual == configured


def model_digests_match(actual, configured) -> bool:
    """Normalize complete SHA256 identities, preserving legacy fixture equality."""
    if not isinstance(actual, str) or not isinstance(configured, str) or not actual or not configured:
        return False
    if _SHA256.fullmatch(actual) and _SHA256.fullmatch(configured):
        return actual.removeprefix('sha256:').lower() == configured.removeprefix('sha256:').lower()
    return actual == configured


def validate_inventory_entry(entry) -> None:
    """Reject runtime-advertised cloud aliases before sending any document text."""
    if not isinstance(entry, Mapping):
        raise ModelPolicyError('Ollama model inventory entry must be an object')
    pending = [entry]
    inspected = 0
    while pending:
        details = pending.pop()
        inspected += 1
        if inspected > 64:
            raise ModelPolicyError('Ollama model inventory metadata is malformed')
        for field, value in details.items():
            if field in ('remote_host', 'remote_model', 'cloud', 'is_cloud') and value not in (None, '', False):
                raise ModelPolicyError('Ollama cloud-backed models are disabled, including local aliases')
            if field in ('name', 'model') and value is not None:
                _validate_local_model_name(value)
            if isinstance(value, Mapping):
                pending.append(value)


@dataclass(frozen=True, slots=True)
class ModelScope:
    """Trusted caller context, never inferred from model input or public requests."""

    tenant_id: str
    company_id: str
    run_id: str

    def __post_init__(self):
        for name in ('tenant_id', 'company_id', 'run_id'):
            value = getattr(self, name)
            if (not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 256
                    or any(ord(character) < 32 or ord(character) == 127 for character in value)):
                raise ModelPolicyError(f'Model scope {name} must be a nonempty identifier of at most 256 characters')


_MODEL_SCOPE = contextvars.ContextVar('regchain_model_scope', default=None)


@contextmanager
def bind_model_scope(scope: ModelScope):
    """Bind scope for this execution context and restore it on every exit path."""
    if not isinstance(scope, ModelScope):
        raise ModelPolicyError('Model scope must be a ModelScope instance')
    token = _MODEL_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _MODEL_SCOPE.reset(token)


def current_model_scope() -> ModelScope | None:
    return _MODEL_SCOPE.get()


def cache_namespace(environ=None) -> str:
    """Cache data can be reused across runs only within the same tenant/company."""
    mode = model_mode(environ)
    scope = current_model_scope()
    if scope is None:
        if mode == 'server':
            raise ModelPolicyError('Server model calls require a tenant, company and run scope')
        return 'development-local'
    identity = json.dumps([scope.tenant_id, scope.company_id], ensure_ascii=False, separators=(',', ':'))
    return 'tenant-company-' + sha256(identity.encode('utf-8')).hexdigest()


def _configuration():
    return {key: value for key, value in os.environ.items()
            if key == 'CARDAMAN_MODE' or key.startswith(('LLM_', 'JUDGE_', 'FAST_', 'EMBED_', 'OLLAMA_'))}


_UNSET = object()


class ModelRouter:
    """Lazily build the established roles once, under one immutable run scope.

    Factories retain their existing signatures: provider(name), judge(provider),
    and embedder(). In particular a judge factory can return the extraction object
    itself, and all quick/wide/fast variants remain managed by those factories.
    """

    def __init__(self, provider_name='ollama', *, provider_factory=None, judge_factory=None, embedder_factory=None):
        self.provider_name = provider_name
        self._provider_name = provider_name
        self._mode = model_mode()
        self._scope = current_model_scope()
        self._configuration = _configuration()
        self._provider_factory = provider_factory
        self._judge_factory = judge_factory
        self._embedder_factory = embedder_factory
        self._provider = self._judge = self._embedder = _UNSET
        if provider_name not in ('rules', 'ollama'):
            raise ModelPolicyError('Choose rules or ollama explicitly; no silent provider fallback')
        if self._mode == 'server':
            cache_namespace()
            self._validate_server_roles()

    def _validate_server_roles(self):
        if self.provider_name != 'ollama':
            raise ModelPolicyError('Server mode requires Ollama; rules fallback is disabled')
        validate_ollama_endpoint(os.environ.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434'))
        for prefix, expected, kind in (('LLM', ('qwen3:4b',), 'chat'),
                                       ('JUDGE', ('qwen3:8b',), 'chat'),
                                       ('EMBED', ('bge-m3', 'bge-m3:latest'), 'embedding')):
            model = os.environ.get(prefix + '_MODEL', '')
            validate_model(model, os.environ.get(prefix + '_MODEL_DIGEST', ''), kind=kind)
            if model not in expected:
                raise ModelPolicyError(f'Server {prefix}_MODEL must use its approved model role')
        fast = os.environ.get('FAST_MODEL', '')
        if fast:
            validate_model(fast, os.environ.get('FAST_MODEL_DIGEST', ''))
            if fast != 'qwen3:4b':
                raise ModelPolicyError('Server FAST_MODEL must use qwen3:4b')

    def _check_context(self):
        if self.provider_name != self._provider_name:
            raise ModelPolicyError('Model provider selection changed; create a new router for this run')
        if model_mode() != self._mode or _configuration() != self._configuration:
            raise ModelPolicyError('Model configuration changed; create a new router for this run')
        if current_model_scope() != self._scope:
            raise ModelPolicyError('A model router cannot be reused outside its original tenant, company and run scope')
        if self._mode == 'server':
            cache_namespace()

    def extraction_provider(self):
        self._check_context()
        if self._provider is _UNSET:
            factory = self._provider_factory
            if factory is None:
                from regchain.extraction.providers import configured_provider
                factory = configured_provider
            self._provider = factory(self.provider_name)
        return self._provider

    def judge_provider(self):
        self._check_context()
        if self._judge is _UNSET:
            factory = self._judge_factory
            if factory is None:
                from regchain.extraction.providers import configured_judge
                factory = configured_judge
            self._judge = factory(self.extraction_provider())
        return self._judge

    def embedder(self):
        self._check_context()
        if self._embedder is _UNSET:
            factory = self._embedder_factory
            if factory is None:
                from regchain.pilot.semantic import configured_embedder
                factory = configured_embedder
            self._embedder = factory()
        return self._embedder

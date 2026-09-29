"""Per-request identity carried in a context variable, not in function arguments.

The AI providers already read their task name from a ``ContextVar`` (``AI_TASK``) so that
test doubles keep their signatures; the request context follows the same pattern: a middleware
binds it once, and audit entries, permission checks and storage calls read it wherever they run
on that request's thread or task.
"""
import contextvars
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from regchain.evidence import digest

# What a client may hand us as X-Request-ID: long enough to be unique, short enough for a log
# column, and nothing a log line could not carry verbatim (no spaces, quotes or control bytes).
REQUEST_ID = re.compile(r'^[A-Za-z0-9._-]{8,64}$')
SHA256_HEX = re.compile(r'^[0-9a-f]{64}$')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RequestContext:
    """One request's identity. ``roles`` is a frozenset of ``regchain.platform.rbac`` role names;
    ``started_at`` is an ISO 8601 UTC string, like every timestamp kept as evidence."""
    request_id: str
    tenant_id: str
    actor: str
    roles: frozenset
    started_at: str

    def __post_init__(self):
        if not REQUEST_ID.fullmatch(self.request_id or ''):
            raise ValueError('request_id must match ' + REQUEST_ID.pattern)
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError('tenant_id is required')
        if not isinstance(self.actor, str) or not self.actor.strip():
            raise ValueError('actor is required')
        # A frozen dataclass cannot assign in __post_init__; normalise through object.__setattr__.
        object.__setattr__(self, 'roles', frozenset(str(role) for role in (self.roles or ())))
        if not isinstance(self.started_at, str) or not self.started_at:
            raise ValueError('started_at must be an ISO 8601 string')


_CONTEXT = contextvars.ContextVar('regchain_request_context', default=None)


def current_context() -> RequestContext | None:
    """The bound context, or None outside a request (a CLI import, a unit test)."""
    return _CONTEXT.get()


@contextmanager
def bind_context(context: RequestContext | None = None, /, *, request_id=None, tenant_id=None,
                 actor=None, roles=(), started_at=None):
    """Bind a context for the block; nested binds restore the outer one on exit.

    Either pass a ready ``RequestContext`` or the fields; a missing ``request_id`` becomes a
    fresh random one and a missing ``started_at`` becomes now.
    """
    if context is None:
        context = RequestContext(new_request_id(request_id), tenant_id, actor, roles, started_at or now())
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


def new_request_id(existing=None) -> str:
    """The id for an ordinary inbound request.

    A well-formed client-supplied X-Request-ID is kept, so the gateway, the application log and
    the audit line share one id. Anything else (missing, too short, quotes, spaces, non-ASCII)
    is replaced by a uuid4 hex: a client must not be able to plant a log-injection string or a
    guessable id. Use this for every request; use ``deterministic_request_id`` only for the
    replayable operations described there.
    """
    if isinstance(existing, str) and REQUEST_ID.fullmatch(existing):
        return existing
    return uuid4().hex


def deterministic_request_id(tenant_id: str, method: str, path: str, body_sha256: str, timestamp_iso: str) -> str:
    """The id for an operation the client may resend (start analysis, save review, delete).

    A random id makes two identical POSTs two requests, so a retried "start analysis" runs the
    model twice. This id is a function of the tenant, the method, the path, the SHA-256 of the
    request body and a timestamp the CLIENT declares (an Idempotency-Key with the moment the user
    clicked, not the server clock): a resend maps to the same id and an idempotency table can
    answer with the first result. Never use it for reads, and never derive it from the server
    time, which would make every retry a new request again. The result is 32 hex characters,
    so it also satisfies the X-Request-ID grammar and can be echoed back to the client. It is
    not a run id, although both are 32 hex characters.
    """
    if not tenant_id or not isinstance(tenant_id, str):
        raise ValueError('tenant_id is required')
    if not SHA256_HEX.fullmatch(body_sha256 or ''):
        raise ValueError('body_sha256 must be 64 lowercase hex characters')
    if not timestamp_iso or not isinstance(timestamp_iso, str):
        raise ValueError('timestamp_iso is required')
    if not method or not path:
        raise ValueError('method and path are required')
    return digest({'tenant_id': tenant_id, 'method': method.upper(), 'path': path,
                   'body_sha256': body_sha256, 'timestamp': timestamp_iso})[:32]

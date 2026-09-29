"""Roles, permissions and tenant isolation for the production path.

Four roles are enough for the pilot's actions and keep the two separations that matter for an
audit: the person who starts an analysis is not the person who records the decision on it
(ANALYST versus REVIEWER), and deleting evidence or changing settings is a distinct duty
(ADMIN). ADMIN is the break-glass role and holds every permission; the audit log, not the
permission table, is the control on it.
"""
from .context import RequestContext, current_context

VIEWER = 'VIEWER'
REVIEWER = 'REVIEWER'
ANALYST = 'ANALYST'
ADMIN = 'ADMIN'
ROLES = (VIEWER, REVIEWER, ANALYST, ADMIN)

PERMISSIONS = {
    # Screen access to the run list, job records, packets and reports of the caller's tenant.
    'view_runs': frozenset({VIEWER, REVIEWER, ANALYST, ADMIN}),
    # Starts a model run; costs money and time, and shapes the evidence a reviewer will see.
    'start_analysis': frozenset({ANALYST, ADMIN}),
    # Records a human decision on the AI proposals; kept apart from start_analysis (four eyes).
    'save_review': frozenset({REVIEWER, ADMIN}),
    # The export zip carries the policy originals, so a screen-only viewer cannot take it.
    'export_data': frozenset({REVIEWER, ANALYST, ADMIN}),
    'delete_data': frozenset({ADMIN}),
    'manage_settings': frozenset({ADMIN}),
    # ai-calls.jsonl names models, hashes, token counts and failures: operational, not decision, data.
    'read_ai_logs': frozenset({ANALYST, ADMIN}),
}


class PermissionDenied(ValueError):
    """Raised by ``require`` and ``assert_tenant``; carries the permission name for the audit line."""

    def __init__(self, permission: str, message: str | None = None):
        super().__init__(message or f'Bu işlem için yetkiniz yok ({permission}).')
        self.permission = permission


def permitted(roles, permission: str) -> bool:
    """True when any of ``roles`` grants ``permission``. An unknown permission name is a
    programming error and raises, so a typo cannot pass as a silent denial (or a silent grant)."""
    if permission not in PERMISSIONS:
        raise ValueError(f'Unknown permission: {permission!r}')
    return not PERMISSIONS[permission].isdisjoint(set(roles or ()))


def require(permission: str, context: RequestContext | None = None) -> RequestContext:
    """The current (or given) context must hold ``permission``; no bound context is a denial too,
    because an unauthenticated code path must fail closed rather than run as nobody."""
    context = context or current_context()
    if context is None or not permitted(context.roles, permission):
        raise PermissionDenied(permission)
    return context


def same_tenant(context: RequestContext | None, tenant_id: str) -> bool:
    """A record belongs to the caller only when the ids are equal and both are set; an empty or
    missing tenant never matches anything, so a record without an owner is reachable by nobody."""
    return bool(context and tenant_id and isinstance(tenant_id, str) and context.tenant_id == tenant_id)


def assert_tenant(context: RequestContext | None, tenant_id: str, permission: str = 'view_runs') -> None:
    """Refuse a record of another tenant. The HTTP layer should answer 404, not 403, so the
    existence of another tenant's run id is not confirmed; the message here is for the audit line."""
    if not same_tenant(context, tenant_id):
        raise PermissionDenied(permission, 'Kayıt bulunamadı ya da başka bir kuruluşa ait.')

"""Who may do what in the staff tools: one table from role to permissions, and the checks that enforce it.

`security.require_admin` only answers "is anyone signed in". Everything the staff app touches (the review queues, report and
burn-ban detail and photos, approve/reject, the audit log, staff push) asks one of these instead, and the account is looked up
live so a role change, a disabled account or an expired reviewer takes effect on the next request.

Two forms, same rules:

    @router.post("/x")
    def x(email: str = Depends(require_permission("reports.moderate"))): ...     # new routes

    def y(..., token: Optional[str] = None):
        actor = authorize(token, "burnbans.moderate")                             # routes that already take `token`

Both return the admin's email, like `require_admin`, so they are drop-in replacements. `@requires(...)` marks a route that
calls `authorize` itself so `unguarded_admin_routes()` (and its test) can tell it apart from one with no check at all.

Not every admin route has a permission yet (the website's other admin pages still use `require_admin`);
`unguarded_admin_routes()` lists them so the gap stays visible while it is closed."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Iterable, Optional

from fastapi import HTTPException
from jose import jwt

from core import security

from . import users

STAFF_ACCESS = "staff.access"
REPORTS_VIEW = "reports.view"
REPORTS_MODERATE = "reports.moderate"
BURNBANS_VIEW = "burnbans.view"
BURNBANS_MODERATE = "burnbans.moderate"
AUDIT_VIEW = "audit.view"
PUSH_RECEIVE = "push.receive"
ACCOUNTS_MANAGE = "accounts.manage"

ALL_PERMISSIONS = frozenset({
    STAFF_ACCESS, REPORTS_VIEW, REPORTS_MODERATE, BURNBANS_VIEW, BURNBANS_MODERATE, AUDIT_VIEW, PUSH_RECEIVE, ACCOUNTS_MANAGE,
})
_VIEW = {STAFF_ACCESS, REPORTS_VIEW, BURNBANS_VIEW, AUDIT_VIEW}
_MODERATE = {REPORTS_MODERATE, BURNBANS_MODERATE}

ROLE_PERMISSIONS: dict[str, frozenset] = {
    "owner": ALL_PERMISSIONS,
    "admin": ALL_PERMISSIONS - {ACCOUNTS_MANAGE},
    "staff": ALL_PERMISSIONS - {ACCOUNTS_MANAGE},
    # Partners can look but not act.
    "partner": frozenset(_VIEW | {PUSH_RECEIVE}),
    # App-store reviewers get the same screens and actions as staff, against demo data only (core/sandbox.py), and no real push.
    "reviewer": frozenset(_VIEW | _MODERATE),
}
assert set(ROLE_PERMISSIONS) == set(users.ROLES), "every role needs a row in ROLE_PERMISSIONS"


def access_token_for(user: dict, session_id: str, *, client: str = "web", minutes: Optional[int] = None) -> str:
    """The signed access token for a session. Besides who (`sub`) and which session (`sid`, checked on every request), it
    carries the account id (`adm`), role (`rl`) and client (`cl`: "m" for the app) so the audit log can name the actor
    without a lookup. The claims are only labels: permission checks always read the account live."""
    claims = {"sub": user["email"], "sid": session_id, "adm": user["id"], "rl": user["role"]}
    if client == "mobile":
        claims["cl"] = "m"
    return security.create_access_token(claims, timedelta(minutes=minutes) if minutes else None)


def permissions_for(role: str) -> frozenset:
    return ROLE_PERMISSIONS.get(role, frozenset())


def has(user: dict, permission: str) -> bool:
    return permission in permissions_for(user["role"])


def roles_with(permission: str) -> tuple:
    """Roles that hold a permission (used to decide who may receive staff push)."""
    return tuple(role for role, held in ROLE_PERMISSIONS.items() if permission in held)


@dataclass(frozen=True)
class Actor:
    """Who is acting, for the audit log. `client` is "web", "mobile" or "sandbox"."""
    admin_id: Optional[int]
    email: str
    role: str
    client: str
    session_id: Optional[str]


def _claims(token: str) -> dict:
    try:
        return jwt.get_unverified_claims(token)
    except Exception:  # noqa: BLE001 - the token was already verified; a decode problem only costs the extra claims
        return {}


def actor_claims(email: str, user: Optional[dict], token: str) -> Actor:
    claims = _claims(token)
    role = (user or {}).get("role") or claims.get("rl") or ""
    client = "sandbox" if users.is_sandbox_role(role) else ("mobile" if claims.get("cl") == "m" else "web")
    admin_id = (user or {}).get("id") or claims.get("adm")
    return Actor(admin_id=admin_id, email=email, role=role, client=client, session_id=claims.get("sid"))


def resolve_actor(token: Optional[str] = None) -> Optional[Actor]:
    """The actor behind the current request's token (or the one passed), or None for an anonymous request or a system
    job. Reads the token's own claims and never queries the database, so it is safe to call in the middle of another
    transaction (the audit log does). Tokens minted before the claims existed fall back to a lookup."""
    token = token or security._request_token.get()
    email = security.verify_token(token) if token else None
    if not email:
        return None
    claims = _claims(token)
    if claims.get("rl") and claims.get("adm"):
        return actor_claims(email, None, token)
    return actor_claims(email, users.get_by_email(email), token)


def authorize_user(token: Optional[str], *permissions: str) -> dict:
    """The signed-in account, if it holds every permission asked for. 401 when nobody valid is signed in (or the account
    is disabled or expired), 403 when it lacks a permission."""
    email = security.verify_token(token)
    user = users.get_by_email(email) if email else None
    if user is None or user["disabled_at"] or users.is_expired(user):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if any(not has(user, permission) for permission in permissions):
        raise HTTPException(status_code=403, detail="Your account does not have access to that.")
    return user


def authorize(token: Optional[str] = None, *permissions: str) -> str:
    """Drop-in for `require_admin(token)`: the admin's email, after checking permissions."""
    return authorize_user(token, *permissions)["email"]


def require_permission(*permissions: str) -> Callable[[], str]:
    """FastAPI dependency for new routes. Takes no `token` query parameter: the token comes from the request's cookie or
    Bearer header."""
    unknown = set(permissions) - ALL_PERMISSIONS
    if unknown:
        raise ValueError(f"Unknown permission(s): {', '.join(sorted(unknown))}")

    def dependency() -> str:
        return authorize(None, *permissions)

    dependency.__required_permissions__ = tuple(permissions)  # type: ignore[attr-defined]
    return dependency


def requires(*permissions: str) -> Callable[[Callable], Callable]:
    """Mark a route function that calls `authorize(token, ...)` itself, for the unguarded-routes report."""
    unknown = set(permissions) - ALL_PERMISSIONS
    if unknown:
        raise ValueError(f"Unknown permission(s): {', '.join(sorted(unknown))}")

    def mark(function: Callable) -> Callable:
        function.__required_permissions__ = tuple(permissions)  # type: ignore[attr-defined]
        return function

    return mark


def _dependency_permissions(dependant: Any) -> list:
    found: list = []
    for sub in getattr(dependant, "dependencies", []):
        found += list(getattr(sub.call, "__required_permissions__", ()) or ())
        found += _dependency_permissions(sub)
    return found


def route_permissions(route: Any) -> tuple:
    """Permissions a route declares, either through `require_permission` or `@requires`."""
    declared = list(getattr(getattr(route, "endpoint", None), "__required_permissions__", ()) or ())
    dependant = getattr(route, "dependant", None)
    if dependant is not None:
        declared += _dependency_permissions(dependant)
    return tuple(dict.fromkeys(declared))


def unguarded_admin_routes(app: Any, prefix: str = "/api/admin") -> list:
    """(method, path) of every route under `prefix` with no declared permission. The test compares this with a checked-in
    list so a new admin route cannot be added without a permission (or an explicit entry)."""
    missing = []
    for route in getattr(app, "routes", []):
        path = getattr(route, "path", "")
        if not path.startswith(prefix) or not getattr(route, "methods", None):
            continue
        if not route_permissions(route):
            missing += [(method, path) for method in sorted(route.methods - {"HEAD", "OPTIONS"})]
    return sorted(missing)

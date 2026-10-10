# Copyright (c) 2026 Show Me Fire. PolyForm Internal Use License 1.0.0; see LICENSE-FIELD.md.
"""
ShowMeFire Field team collaboration API.

Lets a small team of Field app users share maps and, optionally, their live positions.

Authentication is per member: creating or joining a team returns a bearer token
(``Authorization: Bearer fld_...``); only its SHA-256 is stored. A team is created only with
the operator key in ``FIELD_TEAM_CREATE_KEY`` (creation is disabled when it is unset), and
others join with the team's join code. Every request is scoped to the caller's own team, and
another team's map answers exactly like a missing one.

Real-time updates use long polling: ``GET /api/field/v1/events`` returns as soon as there are
new map operations or teammate positions, or empty after ``wait`` seconds. It checks the
database every half second, so it works with any number of server processes.
"""
import asyncio
import hmac
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.routing import APIRoute
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator

from core import field_store as store
from core.field_accounts import AccountError
from core.security import verify_token

logger = logging.getLogger(__name__)


class _SizeLimitedRoute(APIRoute):
    """Refuse oversized bodies from the Content-Length header before FastAPI parses them."""

    def get_route_handler(self):
        inner = super().get_route_handler()

        async def handler(request: Request):
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request is too large.")
            return await inner(request)

        return handler


router = APIRouter(prefix="/api/field/v1", tags=["field"], route_class=_SizeLimitedRoute)

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_WAIT_SECONDS = 25
POLL_INTERVAL_SECONDS = 0.5
# Attempts per client address in the window for endpoints that accept secrets.
AUTH_ATTEMPTS = 10
AUTH_WINDOW_SECONDS = 600

_attempts: Dict[str, Deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    return (
        request.headers.get("cf-connecting-ip")
        or (request.client.host if request.client else "unknown")
    )


def _throttle(request: Request) -> None:
    """Process-local sliding window; enough to make guessing a join code impractical."""
    now = time.monotonic()
    q = _attempts[_client_ip(request)]
    while q and now - q[0] > AUTH_WINDOW_SECONDS:
        q.popleft()
    if len(q) >= AUTH_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")
    q.append(now)


def _as_http(exc: AccountError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.message)


def current_member(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    try:
        return store.authenticate(token)
    except AccountError as exc:
        raise _as_http(exc)


def _call(fn, *args):
    try:
        return fn(*args)
    except AccountError as exc:
        raise _as_http(exc)


# ------------------------------------------------------------------ models

def _clean(v: str) -> str:
    v = " ".join(v.split())
    if not v:
        raise ValueError("must not be empty")
    return v


class TeamCreate(BaseModel):
    team_name: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=40)

    _c1 = field_validator("team_name", "display_name")(_clean)


class TeamJoin(BaseModel):
    join_code: str = Field(min_length=4, max_length=40)
    display_name: str = Field(min_length=1, max_length=40)

    _c1 = field_validator("display_name")(_clean)


class MapCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    features: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    _c1 = field_validator("name")(_clean)


class Op(BaseModel):
    op: Literal["upsert", "delete"]
    uid: str = Field(min_length=8, max_length=64)
    feature: Optional[Dict[str, Any]] = None


class OpsBatch(BaseModel):
    ops: List[Op] = Field(min_length=1, max_length=store.MAX_OPS_PER_BATCH)


class Position(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    accuracy: Optional[float] = Field(default=None, ge=0, le=100000)
    speed: Optional[float] = Field(default=None, ge=0, le=500)
    course: Optional[float] = Field(default=None, ge=0, le=360)
    alt: Optional[float] = Field(default=None, ge=-1000, le=100000)


# ------------------------------------------------------------------ teams

@router.post("/teams", status_code=201)
def create_team(
    body: TeamCreate, request: Request, authorization: Optional[str] = Header(default=None)
):
    """Create a team. Needs a paid account token (``Authorization: Bearer fldacct_...``), or
    the operator key in ``X-Field-Create-Key`` for teams the operator makes directly."""
    _throttle(request)
    if authorization and authorization.lower().startswith("bearer "):
        account_token = authorization[7:].strip()
        return _call(store.create_team, body.team_name, body.display_name, account_token)
    key = os.getenv("FIELD_TEAM_CREATE_KEY", "").strip()
    supplied = request.headers.get("x-field-create-key", "")
    if not key or not supplied:
        raise HTTPException(
            status_code=401,
            detail="Creating a team needs a Show Me Fire account. Use your account token.",
        )
    if not hmac.compare_digest(supplied.encode("utf-8"), key.encode("utf-8")):
        raise HTTPException(status_code=403, detail="Team creation key is wrong.")
    return _call(store.create_team, body.team_name, body.display_name)


@router.post("/teams/join")
def join_team(body: TeamJoin, request: Request):
    _throttle(request)
    return _call(store.join_team, body.join_code, body.display_name)


@router.get("/me")
def me(member: Dict[str, Any] = Depends(current_member)):
    return {
        "member_id": member["id"],
        "display_name": member["display_name"],
        "is_owner": bool(member["is_owner"]),
        "team_id": member["team_id"],
        "team_name": member["team_name"],
        "members": store.list_members(member["team_id"]),
        "plan": store.team_plan(member),
    }


@router.post("/teams/join-code")
def new_join_code(member: Dict[str, Any] = Depends(current_member)):
    return {"join_code": _call(store.rotate_join_code, member)}


@router.delete("/members/{member_id}", status_code=204)
def remove_member(member_id: str, member: Dict[str, Any] = Depends(current_member)):
    _call(store.revoke_member, member, member_id)


# ------------------------------------------------------------------ maps

@router.get("/maps")
def list_maps(member: Dict[str, Any] = Depends(current_member)):
    return {"maps": store.list_maps(member)}


@router.post("/maps", status_code=201)
def create_map(body: MapCreate, member: Dict[str, Any] = Depends(current_member)):
    return _call(store.create_map, member, body.name, body.features)


@router.get("/maps/{map_id}")
def get_map(map_id: str, member: Dict[str, Any] = Depends(current_member)):
    return _call(store.get_map, member, map_id)


@router.delete("/maps/{map_id}", status_code=204)
def delete_map(map_id: str, member: Dict[str, Any] = Depends(current_member)):
    _call(store.delete_map, member, map_id)


@router.post("/maps/{map_id}/ops")
def post_ops(map_id: str, body: OpsBatch, member: Dict[str, Any] = Depends(current_member)):
    return _call(store.apply_ops, member, map_id, [o.model_dump() for o in body.ops])


# ------------------------------------------------------------------ positions and events

@router.post("/positions")
def post_position(body: Position, member: Dict[str, Any] = Depends(current_member)):
    return _call(store.report_position, member, body.model_dump())


@router.delete("/positions", status_code=204)
def delete_position(member: Dict[str, Any] = Depends(current_member)):
    store.clear_position(member)


@router.get("/events")
async def events(
    map_id: Optional[str] = None,
    since: int = 0,
    pos_since: int = 0,
    wait: int = 0,
    member: Dict[str, Any] = Depends(current_member),
):
    """New map operations (when ``map_id`` is given) and teammate positions.

    Pass the ``rev`` and ``pos_seq`` from the previous reply as ``since`` and ``pos_since``.
    With ``wait`` seconds (up to 25) the call holds open until something changes.
    """
    if since < 0 or pos_since < 0:
        raise HTTPException(status_code=422, detail="Cursors must not be negative.")
    deadline = time.monotonic() + max(0, min(wait, MAX_WAIT_SECONDS))
    while True:
        news = await run_in_threadpool(_call, store.has_news, member, map_id, since, pos_since)
        if news or time.monotonic() >= deadline:
            break
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    return await run_in_threadpool(_call, store.events_since, member, map_id, since, pos_since)


# ------------------------------------------------------------------ accounts and plans

class AccountCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    contact: str = Field(default="", max_length=200)
    plan: str = Field(min_length=1, max_length=20)
    expires_at: Optional[int] = Field(default=None, ge=0)
    notes: str = Field(default="", max_length=1000)
    limits: Optional[Dict[str, int]] = None

    _c1 = field_validator("name")(_clean)


class AccountUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    contact: Optional[str] = Field(default=None, max_length=200)
    plan: Optional[str] = None
    status: Optional[str] = None
    expires_at: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = Field(default=None, max_length=1000)
    limits: Optional[Dict[str, int]] = None


def admin_required(token: Optional[str] = Query(default=None)) -> str:
    """Show Me Fire administrator (the same JWT the other admin routes use)."""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


@router.get("/account")
def my_account(authorization: Optional[str] = Header(default=None)):
    """For an account holder: plan, status, limits and their teams' usage."""
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
    return _call(store.account_overview, token)


@router.post("/admin/accounts", status_code=201)
def admin_create_account(body: AccountCreate, _admin: str = Depends(admin_required)):
    """Grant an account. The token in the reply is shown only once."""
    return _call(
        store.admin_create_account,
        body.name, body.contact, body.plan, body.expires_at, body.notes, body.limits,
    )


@router.get("/admin/accounts")
def admin_list_accounts(_admin: str = Depends(admin_required)):
    return {"accounts": store.admin_list_accounts()}


@router.patch("/admin/accounts/{account_id}")
def admin_update_account(account_id: str, body: AccountUpdate, _admin: str = Depends(admin_required)):
    return _call(store.admin_update_account, account_id, body.model_dump(exclude_unset=True))


@router.post("/admin/accounts/{account_id}/token")
def admin_rotate_token(account_id: str, _admin: str = Depends(admin_required)):
    """Issue a new account token; the old one stops working."""
    return {"token": _call(store.admin_rotate_account_token, account_id)}


@router.get("/admin/usage")
def admin_usage(month: Optional[str] = Query(default=None, pattern=r"^\d{4}-\d{2}$"), _admin: str = Depends(admin_required)):
    """Per-team activity for a month (UTC, YYYY-MM), for reviewing server cost."""
    return store.admin_usage(month)

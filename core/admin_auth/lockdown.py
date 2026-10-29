"""Default-deny, and the sandbox, for reviewer (app-store review) sessions.

A reviewer account is an admin account, and almost every admin route only checks "is someone signed in", so on its own it
would reach real data through any of them. This sits in front of the whole app. Once a request carries a token that belongs to a
sandbox role:

  * only the staff API, the app's own sign-in/refresh/sign-out and "who am I" may be reached; everything else is refused with
    a 403 before it is routed;
  * a staff API request runs inside the sandbox scope (core/sandbox.py), so every query in it hits the synthetic sandbox
    database. If the sandbox is switched off or cannot be built the request fails with a 503; it never falls back to the real
    database.

Requests with no token (the public site and app) and every other role pass straight through.

It must be added to the app BEFORE AdminCookieMiddleware (so it runs inside it, after the request token has been read).
"""
from __future__ import annotations

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from core import sandbox, sandbox_data, security

from . import permissions, users

STAFF_PREFIX = "/api/staff/"
ALLOWED_PREFIXES = (STAFF_PREFIX, "/api/admin/mobile/")
ALLOWED_PATHS = ("/api/admin/me", "/api/admin/logout")


def path_allowed_for_sandbox(path: str) -> bool:
    return path.startswith(ALLOWED_PREFIXES) or path.rstrip("/") in ALLOWED_PATHS


class ReviewerLockdownMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        token = security._request_token.get()
        actor = await run_in_threadpool(permissions.resolve_actor, token) if token and token.count(".") == 2 else None
        if actor is None or not users.is_sandbox_role(actor.role):
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        if not path_allowed_for_sandbox(path):
            await JSONResponse({"detail": "Your account does not have access to that."}, status_code=403)(scope, receive, send)
            return
        if not path.startswith(STAFF_PREFIX):
            await self.app(scope, receive, send)  # sign-in, refresh, sign-out, /me: accounts live in the real database
            return
        if not sandbox_data.sandbox_enabled():
            await JSONResponse({"detail": "Reviewer access is not available right now."}, status_code=503)(scope, receive, send)
            return
        try:
            live = await run_in_threadpool(sandbox_data.ensure_sandbox)
        except Exception:  # noqa: BLE001 - fail closed: never serve a reviewer from the real database
            await JSONResponse({"detail": "The reviewer sandbox is not available right now."}, status_code=503)(scope, receive, send)
            return
        await run_in_threadpool(sandbox_data.touch_activity, scope["method"] not in ("GET", "HEAD", "OPTIONS"))
        with sandbox.sandbox_scope(live):
            await self.app(scope, receive, send)

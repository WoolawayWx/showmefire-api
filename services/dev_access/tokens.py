"""Signed session tokens for the development website gate.

The website (Nitro middleware, website/server/utils/devAccess.ts) checks these on every request without calling the API, so the
format is deliberately tiny and easy to verify with Web Crypto: ``v1.<base64url(json)>.<base64url(hmac-sha256)>``. The payload is
``{"sid": session id, "exp": unix expiry, "chk": unix time the API last confirmed the session}``. The website re-confirms with the
API once ``chk`` is a few minutes old, which is how a revoked code or ended session stops working without a lookup per request.

Both sides share DEV_ACCESS_SECRET (NUXT_DEV_ACCESS_SECRET on the website). Without it nothing can be issued or checked.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

VERSION = "v1"


def secret() -> str:
    return os.getenv("DEV_ACCESS_SECRET", "").strip()


def configured() -> bool:
    return len(secret()) >= 32


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(body: str) -> str:
    return _b64(hmac.new(secret().encode("utf-8"), f"{VERSION}.{body}".encode("ascii"), hashlib.sha256).digest())


def issue(session_id: int, expires_at: int, checked_at: Optional[int] = None) -> str:
    payload = {"sid": int(session_id), "exp": int(expires_at), "chk": int(checked_at if checked_at is not None else time.time())}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{VERSION}.{body}.{_sign(body)}"


def read(token: Optional[str], now: Optional[float] = None) -> Optional[dict]:
    """The payload of a well-signed, unexpired token, else None."""
    if not token or not configured() or len(token) > 1024:
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != VERSION:
        return None
    if not hmac.compare_digest(_sign(parts[1]), parts[2]):
        return None
    try:
        payload = json.loads(_unb64(parts[1]))
        sid, exp = int(payload["sid"]), int(payload["exp"])
    except (ValueError, KeyError, TypeError):
        return None
    if exp <= (now if now is not None else time.time()):
        return None
    return {"sid": sid, "exp": exp, "chk": int(payload.get("chk", 0))}

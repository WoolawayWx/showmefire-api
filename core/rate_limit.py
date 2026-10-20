"""Per-client rate limiting for the public read-only API routes.

Sliding window per client IP, held in process memory (like
``routers/field.py::_throttle``), so limits are per worker. Clients that send
a valid ``Authorization: Bearer <key>`` (keys listed in ``API_READ_KEYS``,
comma separated) get a higher limit, and an authenticated admin session is not
limited. Without a key the route still works for the website; it is just
capped so outside scrapers cannot pull the data in bulk.
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

from fastapi import HTTPException, Request

from core.security import verify_token

TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
WINDOW_SECONDS = 60
KEY_TIER_MULTIPLIER = 10
MAX_TRACKED_CLIENTS = 50_000


def client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")
        if forwarded and forwarded[0].strip():
            return forwarded[0].strip()
    return (request.client.host if request.client else "") or "unknown"


def _configured_keys() -> list[str]:
    return [k.strip() for k in os.getenv("API_READ_KEYS", "").split(",") if k.strip()]


def _supplied_key(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    return header[7:].strip() if header.lower().startswith("bearer ") else ""


# Keys issued through the application form (services/api_keys) live in the database. A lookup result is remembered for a minute so
# a busy client does not cost a query per request; revoking a key therefore takes up to a minute to bite. Uses are counted in memory
# and written back with each refresh.
DB_KEY_CACHE_SECONDS = 60
_db_keys: Dict[str, tuple] = {}
_db_key_uses: Dict[int, int] = {}
_db_key_lock = threading.Lock()


def _database_key(supplied: str) -> Optional[dict]:
    now = time.monotonic()
    with _db_key_lock:
        cached = _db_keys.get(supplied)
        if cached and cached[0] > now:
            row = cached[1]
            if row:
                _db_key_uses[row["id"]] = _db_key_uses.get(row["id"], 0) + 1
            return row
        if len(_db_keys) > 5000:
            _db_keys.clear()
    row = None
    try:
        from services.api_keys import store

        row = store.verify_key(supplied)
        with _db_key_lock:
            pending = _db_key_uses.pop(row["id"], 0) if row else 0
        if row:
            store.touch_key(row["id"], pending + 1)
    except Exception:  # noqa: BLE001 - a database problem must not take the public API down; the caller is just unkeyed
        row = None
    with _db_key_lock:
        _db_keys[supplied] = (now + DB_KEY_CACHE_SECONDS, row)
    return row


def clear_database_key_cache() -> None:
    """Forget remembered lookups (after a key is revoked or replaced). Uses not yet written back are kept for the next refresh."""
    with _db_key_lock:
        _db_keys.clear()


def key_multiplier(request: Request) -> int:
    """0 for no valid key, otherwise how many times the public limit this caller gets (see KEY_TIER_MULTIPLIER)."""
    supplied = _supplied_key(request)
    if not supplied:
        return 0
    if any(hmac.compare_digest(supplied.encode(), key.encode()) for key in _configured_keys()):
        return KEY_TIER_MULTIPLIER
    row = _database_key(supplied)
    return int(row["multiplier"]) if row else 0


def valid_api_key(request: Request) -> bool:
    return key_multiplier(request) > 0


def is_admin(request: Request) -> bool:
    try:
        return bool(verify_token())
    except Exception:
        return False


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


class RateLimiter:
    """FastAPI dependency factory: ``Depends(RateLimiter("verification", 60))``."""

    def __init__(self, scope: str, default_per_minute: int, env_var: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.scope = scope
        self.default_per_minute = default_per_minute
        self.env_var = env_var or f"RATE_LIMIT_{scope.upper()}_PER_MIN"
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def per_minute(self) -> int:
        return _env_int(self.env_var, self.default_per_minute)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        for key in [k for k, q in self._hits.items() if not q or q[-1] <= cutoff]:
            del self._hits[key]

    def __call__(self, request: Request) -> None:
        if is_admin(request):
            return
        multiplier = key_multiplier(request)
        keyed = multiplier > 0
        limit = self.per_minute * (multiplier if keyed else 1)
        bucket = f"{'key' if keyed else 'ip'}:{client_ip(request)}"
        now = self._clock()
        with self._lock:
            if len(self._hits) >= MAX_TRACKED_CLIENTS:
                self._prune(now)
            queue = self._hits.setdefault(bucket, deque())
            while queue and queue[0] <= now - WINDOW_SECONDS:
                queue.popleft()
            if len(queue) >= limit:
                retry_after = max(1, int(queue[0] + WINDOW_SECONDS - now) + 1)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Slow down or use an API key.",
                    headers={"Retry-After": str(retry_after)},
                )
            queue.append(now)


def require_station_data_access(request: Request) -> None:
    """Gate station-level rows behind a key/admin when VERIFICATION_STATION_DATA_REQUIRES_KEY=true."""
    if os.getenv("VERIFICATION_STATION_DATA_REQUIRES_KEY", "false").lower() != "true":
        return
    if is_admin(request) or valid_api_key(request):
        return
    raise HTTPException(status_code=401, detail="API key required for station-level data")

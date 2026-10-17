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


def valid_api_key(request: Request) -> bool:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        return False
    supplied = header[7:].strip().encode()
    return any(hmac.compare_digest(supplied, key.encode()) for key in _configured_keys())


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
        keyed = valid_api_key(request)
        limit = self.per_minute * (KEY_TIER_MULTIPLIER if keyed else 1)
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

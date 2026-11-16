"""Saved copies of what the website's pages need most, kept on the CDN (R2, bucket cdn-showmefire) under `fallback/`.

If the API hangs or is down, the website reads these instead of failing: it then shows the last good data and says so.
They live on the CDN, not behind the API, so they are served even when the API container is dead.

    fallback/<name>.json   the exact body the API returned for that endpoint at `at`
    fallback/meta.json     {"generated_at": newest success, "sources": {name: {"ok": bool, "at": iso, "error": str|None}}}

Rules that keep this safe:
  * A source that fails, times out, or returns an empty / non-JSON body is skipped: the previous good copy stays. An outage
    therefore never overwrites good data with an error page.
  * It reads the API over HTTP (`SMF_FALLBACK_SOURCE_URL`, default http://backend:8000) from the scheduler container, a
    different process from the API, so a hung API stops the refresh (the last copy stays) instead of taking this down too.
  * Skipped silently when R2 is not configured.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

logger = logging.getLogger(__name__)

PREFIX = "fallback"
CACHE = "public, max-age=60"
FETCH_TIMEOUT_S = 15.0

# name -> API path. These are the calls the home page, burn-ban pages and the app-style summary make on the server side.
SOURCES: dict[str, str] = {
    "opsbrief": "/api/opsbrief",
    "banner": "/api/banner",
    "burn-bans-active": "/api/burn-bans/active",
    "home-video": "/api/home-video",
    "homepage-card": "/api/homepage-card",
    "mobile-content": "/api/mobile/content",
}


def source_url() -> str:
    return os.getenv("SMF_FALLBACK_SOURCE_URL", "http://backend:8000").rstrip("/")


def _usable(body: Any) -> bool:
    """A body worth saving: a non-empty JSON object or array (an error page or `null` is not)."""
    return isinstance(body, (dict, list)) and bool(body)


def fetch_source(path: str, client: Optional[httpx.Client] = None) -> Any:
    """The parsed JSON body of one API path, or raise."""
    own = client is None
    client = client or httpx.Client(timeout=FETCH_TIMEOUT_S)
    try:
        response = client.get(source_url() + path)
        response.raise_for_status()
        return response.json()
    finally:
        if own:
            client.close()


def publish(put: Callable[[str, bytes], None], fetch: Callable[[str], Any] = fetch_source,
            now: Optional[datetime] = None, previous_meta: Optional[dict] = None) -> dict:
    """Refresh every source through `put(key, bytes)`. Returns the meta that was written."""
    now = now or datetime.now(timezone.utc)
    stamp = now.isoformat().replace("+00:00", "Z")
    sources: dict[str, dict] = {}
    previous = (previous_meta or {}).get("sources", {})
    for name, path in SOURCES.items():
        try:
            body = fetch(path)
            if not _usable(body):
                raise ValueError("empty or non-JSON body")
            put(f"{PREFIX}/{name}.json", json.dumps({"at": stamp, "data": body}, separators=(",", ":")).encode())
            sources[name] = {"ok": True, "at": stamp, "error": None}
        except Exception as error:  # noqa: BLE001 - one source failing must not stop the others
            was = previous.get(name, {})
            sources[name] = {"ok": False, "at": was.get("at"), "error": f"{type(error).__name__}: {error}"[:200]}
    ok_times = [s["at"] for s in sources.values() if s.get("at")]
    meta = {"generated_at": max(ok_times) if ok_times else None, "sources": sources}
    put(f"{PREFIX}/meta.json", json.dumps(meta, separators=(",", ":")).encode())
    return meta


def run() -> dict:
    """The scheduler job: publish to R2. A no-op without R2 credentials."""
    from services import cdn_datasets

    if not cdn_datasets.configured():
        return {"skipped": "r2 not configured"}
    client = cdn_datasets.make_client()

    def put(key: str, body: bytes) -> None:
        client.put_object(Bucket=cdn_datasets.BUCKET, Key=key, Body=body, ContentType="application/json", CacheControl=CACHE)

    previous = None
    try:
        previous = json.loads(client.get_object(Bucket=cdn_datasets.BUCKET, Key=f"{PREFIX}/meta.json")["Body"].read())
    except Exception:  # noqa: BLE001 - first run, or unreadable: start fresh
        pass
    with httpx.Client(timeout=FETCH_TIMEOUT_S) as http:
        meta = publish(put, lambda path: fetch_source(path, http), previous_meta=previous)
    failed = [name for name, s in meta["sources"].items() if not s["ok"]]
    if failed:
        # Quiet at WARNING: a single blip is normal; a source failing every 5 minutes is visible in meta.json and on the status page.
        logger.warning("Fallback snapshots: kept the previous copy of %s", ", ".join(failed))
    return meta

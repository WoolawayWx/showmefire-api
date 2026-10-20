"""Serve the large in-memory station feeds from bytes encoded once per update.

`/stations` (~1 MB) and `/stations/timeseries` (~1.5 MB) change every few minutes, but returning the dict from a route
makes FastAPI run `jsonable_encoder` + `json.dumps` on every request: ~95 ms and ~200 ms of CPU each, on the event loop,
so every other request waits. Here the body is built once when the feed changes and reused until it changes again.

The body is built with exactly what FastAPI does for a returned dict, so the bytes are identical to before. A feed counts
as changed when its `stations` object is replaced (every fetch builds a new list) or its `last_updated`/`error` changes.

`Cache-Control: private` on purpose: a shared cache (Cloudflare) would keep one website's echoed
`Access-Control-Allow-Origin` and hand it to the others, the problem core/public_cors.py describes for tiles.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Any, Optional

from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

CACHE_CONTROL = "private, max-age=60"

_lock = threading.Lock()
# name -> (fingerprint, body, etag)
_snapshots: dict[str, tuple[tuple, bytes, str]] = {}


def _fingerprint(data: dict) -> tuple:
    return (id(data.get("stations")), data.get("last_updated"), data.get("error"))


def snapshot(name: str, data: dict) -> tuple[bytes, str]:
    """(body, etag) for `data`, rebuilt only when the feed has changed since the last call for `name`."""
    fingerprint = _fingerprint(data)
    cached = _snapshots.get(name)
    if cached and cached[0] == fingerprint:
        return cached[1], cached[2]
    with _lock:
        # Another request may have built it while this one waited for the lock.
        cached = _snapshots.get(name)
        if cached and cached[0] == fingerprint:
            return cached[1], cached[2]
        body = JSONResponse(jsonable_encoder(data)).body
        etag = '"' + hashlib.blake2b(body, digest_size=16).hexdigest() + '"'
        _snapshots[name] = (fingerprint, body, etag)
        return body, etag


def _etag_matches(header: Optional[str], etag: str) -> bool:
    if not header:
        return False
    # Accept weak validators too: a proxy may have turned `"x"` into `W/"x"` when it compressed the response.
    tags = {tag.strip().removeprefix("W/") for tag in header.split(",")}
    return "*" in tags or etag in tags


def snapshot_response(name: str, data: dict[str, Any], request: Optional[Request] = None) -> Response:
    """The feed as a JSON response, or `304 Not Modified` when the caller already has this version."""
    body, etag = snapshot(name, data)
    headers = {"ETag": etag, "Cache-Control": CACHE_CONTROL}
    if request is not None and _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/json", headers=headers)

"""
Stable CORS headers for public, cacheable, credential-free resources (map tiles).

Why this exists: the normal ``CORSMiddleware`` echoes the caller's ``Origin`` back in
``Access-Control-Allow-Origin`` and only when the request carries one. A CDN such as Cloudflare
caches tile responses per URL and does not vary them by ``Origin``, so the first requester decides
what everyone gets: a copy cached for a request without an ``Origin`` header (a desktop app, curl)
has no CORS header at all, and a copy cached for one website's origin is rejected by every other
website. Browser clients (the ShowMeFire Field web app, embedded maps) then fail with CORS errors
even though the server is configured correctly.

Tiles are public data and are fetched without cookies, so they can safely say ``*`` on every
response, with or without an ``Origin`` header. This middleware must be added **after** the
``CORSMiddleware`` so that it runs outside it and has the last word.
"""
from typing import Callable, Iterable, Optional

from starlette.datastructures import MutableHeaders
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Map tiles, and the public burn check (other agencies call it from their own pages, with no credentials).
DEFAULT_PREFIXES = ("/tiles/", "/v1/burn-check", "/v1/spot-forecast")
# Tiles always answer "*". Everything else here (the burn check) is not shared by caches, so a caller whose origin the API
# already trusts is left to CORSMiddleware: our own sites send every API request with credentials, and a browser rejects
# "*" on a credentialed request.
ALWAYS_WILDCARD = ("/tiles/",)


class PublicTileCorsMiddleware:
    def __init__(self, app: ASGIApp, prefixes: Iterable[str] = DEFAULT_PREFIXES,
                 credentialed_origin: Optional[Callable[[str], bool]] = None) -> None:
        self.app = app
        self.prefixes = tuple(prefixes)
        self.credentialed_origin = credentialed_origin

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.prefixes):
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        origin = dict(scope.get("headers") or []).get(b"origin", b"").decode("latin-1")
        if (origin and self.credentialed_origin and self.credentialed_origin(origin)
                and not path.startswith(ALWAYS_WILDCARD)):
            await self.app(scope, receive, send)
            return

        # CORSMiddleware rejects a preflight (400) from any origin not on its credentialed list, which blocks
        # the public burn check from every other site as soon as the request has a header like Content-Type.
        # Answer the preflight here for public paths instead.
        if scope["method"] == "OPTIONS":
            request_headers = dict(scope.get("headers") or [])
            if b"access-control-request-method" in request_headers:
                allow_headers = request_headers.get(b"access-control-request-headers", b"*").decode("latin-1")
                response = Response(status_code=204, headers={
                    "access-control-allow-origin": "*",
                    "access-control-allow-methods": "GET, HEAD, OPTIONS",
                    "access-control-allow-headers": allow_headers,
                    "access-control-max-age": "600",
                })
                await response(scope, receive, send)
                return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["access-control-allow-origin"] = "*"
                # Credentials are never needed (and are forbidden together with "*").
                if "access-control-allow-credentials" in headers:
                    del headers["access-control-allow-credentials"]
                # The response no longer depends on the Origin, so caches may share it.
                if "vary" in headers:
                    kept = [v.strip() for v in headers["vary"].split(",") if v.strip().lower() != "origin"]
                    if kept:
                        headers["vary"] = ", ".join(kept)
                    else:
                        del headers["vary"]
            await send(message)

        await self.app(scope, receive, send_with_cors)

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
from typing import Iterable

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_PREFIXES = ("/tiles/",)


class PublicTileCorsMiddleware:
    def __init__(self, app: ASGIApp, prefixes: Iterable[str] = DEFAULT_PREFIXES) -> None:
        self.app = app
        self.prefixes = tuple(prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.prefixes):
            await self.app(scope, receive, send)
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

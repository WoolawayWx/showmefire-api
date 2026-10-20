"""Pure ASGI base for the session-cookie middlewares (core/admin_session.py, core/graphics_session.py).

They used to be `@app.middleware("http")` functions. Starlette runs those through BaseHTTPMiddleware, which hands the
rest of the app to a separate task and re-streams every response, on every request, including the public routes that
carry no session at all. This only looks at the cookies and, when a session was renewed, adds the new cookie to the
response's start message.
"""
from __future__ import annotations

from typing import Any, Optional

from starlette.datastructures import MutableHeaders
from starlette.requests import HTTPConnection
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class CookieRenewalMiddleware:
    """Subclasses say how to renew a session (`resolve`), where the request's token lives (`set_token`/`reset_token`)
    and how to write the renewed cookie (`set_cookie`)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def resolve(self, request: HTTPConnection) -> tuple[Optional[str], bool]:
        """(the access token for this request, whether it was just renewed)."""
        raise NotImplementedError

    def set_token(self, token: Optional[str]) -> Any:
        raise NotImplementedError

    def reset_token(self, context: Any) -> None:
        raise NotImplementedError

    def set_cookie(self, response: Response, access: str) -> None:
        raise NotImplementedError

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        access, renewed = self.resolve(HTTPConnection(scope))

        async def send_renewed(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Build the Set-Cookie header the same way the routes do, on a throwaway response, and copy it over.
                carrier = Response()
                self.set_cookie(carrier, access)
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "no-store"
                for value in carrier.headers.getlist("set-cookie"):
                    headers.append("set-cookie", value)
            await send(message)

        context = self.set_token(access)
        try:
            await self.app(scope, receive, send_renewed if renewed else send)
        finally:
            self.reset_token(context)

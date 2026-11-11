"""Public side of the QR code tool: what a scan of a tracked code lands on.

GET /q/<slug> records the scan and redirects. GET /qr/go/<slug> does the same but answers with JSON, for the website's own /q/<slug> route (which
passes the visitor's user agent, referrer and IP along, since the API would otherwise only see the website server)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from core.rate_limit import RateLimiter, client_ip
from services.qr import store

router = APIRouter(tags=["qr"])
scan_limiter = RateLimiter("qr_scan", 600)


def _live_code(slug: str) -> dict:
    code = store.get_code_by_slug(slug)
    if code is None or not code["active"] or not code["tracked"] or code["kind"] != "url":
        raise HTTPException(status_code=404, detail="This QR code is not active.")
    return code


@router.get("/q/{slug}", dependencies=[Depends(scan_limiter)], include_in_schema=False)
def scan_redirect(slug: str, request: Request):
    code = _live_code(slug)
    store.record_scan(code, user_agent=request.headers.get("user-agent", ""), referrer=request.headers.get("referer", ""), ip=client_ip(request))
    return RedirectResponse(store.target_url(code), status_code=302, headers={"Cache-Control": "no-store"})


@router.get("/qr/go/{slug}", dependencies=[Depends(scan_limiter)], include_in_schema=False)
def scan_resolve(slug: str, request: Request, x_scan_user_agent: Optional[str] = Header(None), x_scan_referrer: Optional[str] = Header(None)):
    code = _live_code(slug)
    store.record_scan(code, user_agent=x_scan_user_agent or "", referrer=x_scan_referrer or "", ip=client_ip(request))
    return {"url": store.target_url(code)}

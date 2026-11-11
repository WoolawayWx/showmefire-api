"""Admin side of the QR code tool: create and edit codes, download them as SVG or PNG, read scan counts, export scans as CSV.

The prefix is /admin/qr-codes. The public side (the scan redirect) is in routers/qr.py."""
from __future__ import annotations

import csv
import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from core.security import require_admin
from services.qr import store

router = APIRouter(prefix="/admin/qr-codes", tags=["qr-admin"])


class CreateBody(BaseModel):
    name: str = Field(..., max_length=120)
    destination: str = Field(..., max_length=store.MAX_CONTENT)
    kind: str = "url"
    tracked: bool = True
    slug: str = Field("", max_length=40)
    utm_source: str = ""
    utm_medium: str = ""
    utm_campaign: str = ""
    utm_term: str = ""
    utm_content: str = ""
    error_level: str = "M"
    fg: str = "#000000"
    bg: str = "#ffffff"
    border: int = 4


class UpdateBody(BaseModel):
    """Only the keys sent are changed."""
    name: Optional[str] = None
    destination: Optional[str] = None
    kind: Optional[str] = None
    tracked: Optional[bool] = None
    slug: Optional[str] = None
    active: Optional[bool] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_term: Optional[str] = None
    utm_content: Optional[str] = None
    error_level: Optional[str] = None
    fg: Optional[str] = None
    bg: Optional[str] = None
    border: Optional[int] = None


class PreviewBody(CreateBody):
    name: str = "preview"


def _bad(error: store.QRError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(error))


def _code_or_404(code_id: int) -> dict:
    code = store.get_code(code_id)
    if code is None:
        raise HTTPException(status_code=404, detail="QR code not found")
    return code


def _filename(code: dict, ext: str) -> str:
    stem = "".join(c if c.isalnum() else "-" for c in code["name"].lower()).strip("-") or code["slug"]
    return f"qr-{stem[:50]}.{ext}"


@router.get("")
def list_codes(_admin: str = Depends(require_admin)):
    return {"codes": store.list_codes(), "web_url": store.web_url()}


@router.post("", status_code=201)
def create_code(body: CreateBody, admin: str = Depends(require_admin)):
    try:
        values = body.model_dump()
        return {"success": True, "code": store.create_code(created_by=admin, **values)}
    except store.QRError as error:
        raise _bad(error) from None


@router.post("/preview")
def preview(body: PreviewBody, _admin: str = Depends(require_admin)):
    """Render a code that is not saved yet, so the form can show it as the options change. Nothing is stored."""
    try:
        draft = {**body.model_dump(), "slug": body.slug or "preview", "destination": store.clean_destination(body.kind, body.destination)}
        draft["tracked"] = bool(draft["tracked"]) and draft["kind"] == "url"
        text = store.encoded_text(draft)
        svg = store.render_svg(text, level=draft["error_level"], fg=draft["fg"], bg=draft["bg"], border=draft["border"])
    except store.QRError as error:
        raise _bad(error) from None
    return {"svg": svg, "encoded_text": text, "target_url": store.target_url(draft)}


@router.get("/{code_id}")
def get_code(code_id: int, _admin: str = Depends(require_admin)):
    return {"code": _code_or_404(code_id)}


@router.put("/{code_id}")
def update_code(code_id: int, body: UpdateBody, admin: str = Depends(require_admin)):
    try:
        code = store.update_code(code_id, body.model_dump(exclude_unset=True), admin)
    except store.QRError as error:
        raise _bad(error) from None
    if code is None:
        raise HTTPException(status_code=404, detail="QR code not found")
    return {"success": True, "code": code}


@router.delete("/{code_id}")
def delete_code(code_id: int, _admin: str = Depends(require_admin)):
    if not store.delete_code(code_id):
        raise HTTPException(status_code=404, detail="QR code not found")
    return {"success": True}


@router.get("/{code_id}/image")
def download_image(code_id: int, format: str = Query("svg", pattern="^(svg|png)$"), download: bool = False, _admin: str = Depends(require_admin)):
    code = _code_or_404(code_id)
    try:
        body, media_type = store.render_code(code, format)
    except store.QRError as error:
        raise _bad(error) from None
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{_filename(code, format)}"'
    return Response(content=body, media_type=media_type, headers=headers)


@router.get("/{code_id}/stats")
def code_stats(code_id: int, days: int = Query(30, ge=1, le=365), _admin: str = Depends(require_admin)):
    _code_or_404(code_id)
    return store.stats(code_id, days)


@router.get("/{code_id}/scans.csv")
def scans_csv(code_id: int, _admin: str = Depends(require_admin)):
    code = _code_or_404(code_id)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["scanned_at_utc", "device", "referrer_host"])
    writer.writerows(store.scans_csv_rows(code_id))
    return Response(content=out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{_filename(code, "csv").replace("qr-", "qr-scans-", 1)}"'})

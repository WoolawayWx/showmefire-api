"""Public side of the form builder: a published form's definition and the endpoint that takes a response (no /api prefix, like the other
new routes). Forms are made and edited by staff in routers/forms_admin.py.

Anonymous writes are guarded by a Cloudflare Turnstile check (per form, on by default), a hidden honeypot field and a per-IP daily cap.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from core.rate_limit import RateLimiter, client_ip
from services.discord_notifier import notify_staff_alert
from services.forms import schema, store
from services.turnstile import verify_turnstile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/forms", tags=["forms"])
form_limiter = RateLimiter("form_definition", 120)
submit_limiter = RateLimiter("form_response", 20)
DAILY_RESPONSES_PER_IP = 20


class ResponseBody(BaseModel):
    answers: dict[str, Any]
    turnstile_token: str = Field("", max_length=4096)
    website: str = Field("", max_length=200, description="Honeypot: real visitors leave it empty")


def public_definition(form: dict) -> dict:
    """What the website needs to draw a form: nothing about its owner, limits or responses."""
    return {"slug": form["slug"], "version": form["version"], "title": form["title"], "intro": form["intro"],
            "submit_label": form["submit_label"], "success_message": form["success_message"], "require_captcha": form["require_captcha"],
            "fields": form["fields"]}


def _published_or_404(slug: str) -> dict:
    form = store.get_form(slug=slug)
    if form is None or form["status"] == "draft":
        raise HTTPException(status_code=404, detail="Form not found")
    return form


@router.get("/{slug}", dependencies=[Depends(form_limiter)])
def get_form(slug: str, response: Response):
    """A published form, or 410 `{detail}` with the reason when it is closed, past its date or full."""
    form = _published_or_404(slug)
    is_open, reason = store.public_state(form)
    if not is_open:
        raise HTTPException(status_code=410, detail={"message": reason, "title": form["title"]})
    response.headers["Cache-Control"] = "public, max-age=60"
    return {"success": True, "form": public_definition(form)}


@router.post("/{slug}/responses", status_code=201, dependencies=[Depends(submit_limiter)])
def submit_response(slug: str, body: ResponseBody, request: Request):
    """Send a response. 422 `{errors: {field: message}}` when an answer is missing or wrong."""
    form = _published_or_404(slug)
    if body.website.strip():                                   # a bot filled the hidden field: look successful, store nothing
        return {"success": True, "reference": "FR-000000", "message": form["success_message"]}
    is_open, reason = store.public_state(form)
    if not is_open:
        raise HTTPException(status_code=410, detail={"message": reason, "title": form["title"]})
    answers, errors = schema.validate_answers(body.answers, form["fields"])
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    ip = client_ip(request)
    if form["require_captcha"]:
        ok, _verdict = verify_turnstile(body.turnstile_token, ip)
        if not ok:
            raise HTTPException(status_code=400, detail="The human check failed. Please try again.")
    if store.count_recent_from_ip(form["id"], ip) >= DAILY_RESPONSES_PER_IP:
        raise HTTPException(status_code=429, detail="Too many responses from this address today. Please try again tomorrow.")
    row = store.create_response(form, answers, ip)
    if form["notify_staff"]:
        try:
            notify_staff_alert(alert_type="form_response", title=f"New response to {form['title']} ({row['reference']})",
                               description=f"A new response arrived on the “{form['title']}” form.",
                               fields=[{"name": a["label"][:200], "value": (schema.answer_text(a) or "-")[:900], "inline": False} for a in answers[:6]],
                               admin_path=f"/admin/forms/{form['id']}", force=True)
        except Exception:  # noqa: BLE001 - the alert is a courtesy; the response is already saved
            logger.warning("Could not send the form response alert", exc_info=False)
    return {"success": True, "reference": row["reference"], "message": form["success_message"]}

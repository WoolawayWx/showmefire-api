"""Public API key application: the form definition and the submit endpoint (no /api prefix, like the other new routes).

A request is stored, staff are pinged on Discord with a link to the admin page, and the applicant gets a confirmation email. Staff approve
or deny it on the website (routers/api_keys_admin.py); approval creates the key and emails it. Anonymous writes are guarded by a Cloudflare
Turnstile check, a hidden honeypot field, a per-IP daily cap and one open request per email address.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from core.rate_limit import RateLimiter, client_ip
from services.api_keys import form as key_form
from services.api_keys import mail, store
from services.discord_notifier import notify_staff_alert
from services.turnstile import verify_turnstile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/api-keys", tags=["api-keys"])
form_limiter = RateLimiter("api_key_form", 60)
submit_limiter = RateLimiter("api_key_request", 10)
DAILY_REQUESTS_PER_IP = 5


class KeyRequestBody(BaseModel):
    answers: dict[str, Any]
    turnstile_token: str = Field("", max_length=4096)
    website: str = Field("", max_length=200, description="Honeypot: real visitors leave it empty")


@router.get("/form", dependencies=[Depends(form_limiter)])
def get_form(response: Response):
    """The application questions, as the website draws them (core/api_key_form.json)."""
    response.headers["Cache-Control"] = "public, max-age=300"
    return {"success": True, "form": key_form.public_form()}


def _announce(request_row: dict, answers: list[dict]) -> None:
    fields = [
        {"name": "Agency", "value": request_row["agency"], "inline": True},
        {"name": "Type", "value": key_form.answer_value(answers, "agency_type") or "-", "inline": True},
        {"name": "Contact", "value": f"{request_row['name']} <{request_row['email']}>", "inline": False},
        {"name": "Reason", "value": key_form.answer_value(answers, "reason") or "-", "inline": False},
        {"name": "Plan", "value": key_form.answer_value(answers, "plan") or "-", "inline": False},
    ]
    notify_staff_alert(alert_type="api_key_request", title=f"API key request {request_row['reference']}: {request_row['agency']}",
                       description="A new Burn Check API key request is waiting for review.", fields=fields,
                       admin_path="/admin/api-keys", force=True)


def _confirm(request_row: dict) -> None:
    try:
        if mail.configured():
            mail.send(request_row["email"], *mail.received(request_row["name"], request_row["reference"]))
    except Exception:  # noqa: BLE001 - the confirmation is a courtesy; the request is already saved
        logger.warning("Could not send the API key request confirmation", exc_info=False)


@router.post("/requests", status_code=201, dependencies=[Depends(submit_limiter)])
def submit_request(body: KeyRequestBody, request: Request):
    """Send an application. 422 `{errors: {field: message}}` when an answer is missing or wrong."""
    if body.website.strip():                                   # a bot filled the hidden field: look successful, store nothing
        return {"success": True, "reference": "KR-000000", "message": key_form.load_form().get("success_message", "")}
    form = key_form.load_form()
    answers, errors = key_form.validate_answers(body.answers, form)
    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors})
    ip = client_ip(request)
    ok, _verdict = verify_turnstile(body.turnstile_token, ip)
    if not ok:
        raise HTTPException(status_code=400, detail="The human check failed. Please try again.")
    if store.count_recent_from_ip(ip) >= DAILY_REQUESTS_PER_IP:
        raise HTTPException(status_code=429, detail="Too many requests from this address today. Please try again tomorrow.")
    email = key_form.answer_value(answers, "email")
    open_request = store.open_request_for_email(email)
    if open_request:
        raise HTTPException(status_code=409, detail=f"You already have a request waiting for review (reference {open_request['reference']}). "
                                                    "We will email you when there is a decision.")
    row = store.create_request(name=key_form.answer_value(answers, "name"), email=email, agency=key_form.answer_value(answers, "agency"),
                               answers=answers, form_version=int(form.get("version", 1)), ip=ip)
    threading.Thread(target=_confirm, args=(row,), daemon=True).start()
    _announce(row, answers)
    return {"success": True, "reference": row["reference"], "message": form.get("success_message", "")}

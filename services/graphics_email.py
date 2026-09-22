"""Transactional email delivery for the department graphics portal."""
from __future__ import annotations

import html
import os

import requests
from dotenv import load_dotenv

load_dotenv()

RESEND_ENDPOINT = "https://api.resend.com/emails"
RESEND_API_BASE = "https://api.resend.com"


def _resend_request(method: str, path: str, payload: dict | None = None) -> dict:
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Resend API configuration is incomplete")
    kwargs = {
        "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        "timeout": 15,
    }
    if payload is not None:
        kwargs["json"] = payload
    response = requests.request(method, f"{RESEND_API_BASE}{path}", **kwargs)
    response.raise_for_status()
    return response.json()


def _sender_settings() -> tuple[str, str]:
    sender = os.getenv("BULLETIN_EMAIL_FROM", "").strip() or os.getenv("GRAPHICS_EMAIL_FROM", "").strip()
    reply_to = os.getenv("BULLETIN_EMAIL_REPLY_TO", "").strip() or os.getenv("GRAPHICS_EMAIL_REPLY_TO", "").strip()
    if not sender:
        raise RuntimeError("Resend bulletin email configuration is incomplete")
    return sender, reply_to


def resend_audience_id() -> str:
    segment_id = os.getenv("RESEND_SEGMENT_ID", "").strip() or os.getenv("RESEND_AUDIENCE_ID", "").strip()
    if not segment_id:
        raise RuntimeError("RESEND_SEGMENT_ID is not configured")
    return segment_id


def upsert_audience_contact(email: str) -> str:
    """Create a contact in the configured Audience and return its provider ID."""
    segment_id = resend_audience_id()
    normalized = email.strip().lower()
    try:
        result = _resend_request(
            "POST",
            "/contacts",
            {"email": normalized, "segment_id": segment_id, "unsubscribed": False},
        )
    except requests.HTTPError as exc:
        if not exc.response or exc.response.status_code != 409:
            raise
        # An existing contact may have explicitly unsubscribed. Do not
        # silently resubscribe it when a preference update is retried.
        existing = _resend_request("GET", f"/contacts/{normalized}")
        return str(existing.get("id") or "")
    return str(result.get("id") or "")


def set_audience_contact_unsubscribed(email: str, unsubscribed: bool) -> None:
    """Synchronize the website's subscription state with Resend."""
    _resend_request(
        "PATCH",
        f"/contacts/{email.strip().lower()}",
        {"unsubscribed": bool(unsubscribed)},
    )


def send_bulletin_broadcast(subject: str, html_body: str, text_body: str) -> str:
    """Create and send a Resend broadcast to the configured Audience."""
    sender, reply_to = _sender_settings()
    payload = {
        "segment_id": resend_audience_id(),
        "from": sender,
        "subject": subject,
        "html": html_body if "RESEND_UNSUBSCRIBE_URL" in html_body else (
            f"{html_body}<p style=\"font-size:12px;color:#64748b\">"
            "{{{RESEND_UNSUBSCRIBE_URL}}}</p>"
        ),
        "text": text_body,
        "send": True,
    }
    if reply_to:
        payload["reply_to"] = reply_to
    created = _resend_request("POST", "/broadcasts", payload)
    broadcast_id = str(created.get("id") or "")
    if not broadcast_id:
        raise RuntimeError("Resend did not return a broadcast ID")
    return broadcast_id


def send_daily_forecast_email(
    recipient: str, subject: str, html_body: str, text_body: str,
) -> str:
    """Send one qualifying daily forecast email through Resend."""
    sender, reply_to = _sender_settings()
    payload = {
        "from": sender,
        "to": [recipient],
        "subject": subject,
        "html": html_body,
        "text": text_body,
    }
    if reply_to:
        payload["reply_to"] = reply_to
    result = _resend_request("POST", "/emails", payload)
    return str(result.get("id") or "")


def send_graphics_login_code(recipient: str, code: str) -> str:
    """Send a one-time graphics login code and return the provider message ID."""
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    sender = os.getenv("GRAPHICS_EMAIL_FROM", "").strip()
    reply_to = os.getenv("GRAPHICS_EMAIL_REPLY_TO", "").strip()
    if not api_key or not sender:
        raise RuntimeError("Resend graphics email configuration is incomplete")

    safe_code = html.escape(code)
    payload = {
        "from": sender,
        "to": [recipient],
        "subject": f"Your Show Me Fire sign-in code: {code}",
        "text": (
            f"Your Show Me Fire Department Graphics sign-in code is {code}.\n\n"
            "This code expires in 10 minutes and can be used once. If you did not "
            "request it, you can ignore this email."
        ),
        "html": (
            "<div style=\"font-family:Arial,sans-serif;max-width:560px;margin:auto;"
            "color:#172033\"><h1 style=\"font-size:22px\">Show Me Fire Department Graphics</h1>"
            "<p>Use this code to sign in:</p>"
            f"<p style=\"font-size:32px;font-weight:700;letter-spacing:8px\">{safe_code}</p>"
            "<p>This code expires in 10 minutes and can be used once.</p>"
            "<p style=\"color:#64748b\">If you did not request it, you can ignore this email.</p></div>"
        ),
    }
    if reply_to:
        payload["reply_to"] = reply_to
    response = requests.post(
        RESEND_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()
    return str(result.get("id") or "")

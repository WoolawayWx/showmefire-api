"""Transactional email delivery for the department graphics portal."""
from __future__ import annotations

import html
import os

import requests
from dotenv import load_dotenv

load_dotenv()

RESEND_ENDPOINT = "https://api.resend.com/emails"


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

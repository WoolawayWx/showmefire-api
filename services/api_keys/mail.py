"""Emails for the API key application: received, approved (with the key) and not approved. Sent through Resend like the other
transactional mail (services/graphics_email.py). Everything a person typed is HTML-escaped before it goes into a message."""
from __future__ import annotations

import html
import os
from typing import Optional

from services import graphics_email

WEB_URL = os.getenv("PUBLIC_WEB_URL", "https://showmefire.org").rstrip("/")
API_URL = os.getenv("PUBLIC_API_BASE_URL", "https://api.showmefire.org").rstrip("/")
DOCS_URL = f"{WEB_URL}/tools/burn-check-api"
TERMS_URL = f"{DOCS_URL}#terms"


def configured() -> bool:
    return bool(os.getenv("RESEND_API_KEY", "").strip() and _sender()[0])


def _sender() -> tuple[str, str]:
    sender = (os.getenv("API_KEY_EMAIL_FROM") or os.getenv("GRAPHICS_EMAIL_FROM") or os.getenv("BULLETIN_EMAIL_FROM") or "").strip()
    reply_to = (os.getenv("API_KEY_EMAIL_REPLY_TO") or os.getenv("GRAPHICS_EMAIL_REPLY_TO") or os.getenv("BULLETIN_EMAIL_REPLY_TO") or "").strip()
    return sender, reply_to


def send(to: str, subject: str, text: str, body_html: str) -> str:
    """Send one message and return the provider's message id. Raises if email is not set up or the provider refuses it."""
    sender, reply_to = _sender()
    if not (os.getenv("RESEND_API_KEY", "").strip() and sender):
        raise RuntimeError("Email is not configured (RESEND_API_KEY and a from address are needed)")
    payload = {"from": sender, "to": [to], "subject": subject, "text": text, "html": body_html}
    if reply_to:
        payload["reply_to"] = reply_to
    return str(graphics_email._resend_request("POST", "/emails", payload).get("id") or "")


def _page(heading: str, paragraphs: list[str], extra_html: str = "") -> str:
    body = "".join(f'<p style="margin:0 0 14px;line-height:1.5">{p}</p>' for p in paragraphs)
    return (
        '<div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;color:#172033">'
        f'<h1 style="font-size:22px;margin:0 0 16px">{html.escape(heading)}</h1>{body}{extra_html}'
        '<p style="color:#64748b;font-size:12px;margin-top:24px">Show Me Fire is a public information service. Burn Check is not a burn permit '
        'and not a guarantee, and Show Me Fire is not responsible for how it is used.</p></div>')


def received(name: str, reference: str) -> tuple[str, str, str]:
    subject = "We received your Burn Check API key request"
    text = (f"Hello {name},\n\nWe received your request for a Burn Check API key (reference {reference}). A staff member will review it "
            f"and email you when there is a decision.\n\nBefore then, you can read how the API works: {DOCS_URL}\n\nShow Me Fire")
    return subject, text, _page("We received your request", [f"Hello {html.escape(name)},", f"We received your request for a Burn Check API key "
                                f"(reference <b>{html.escape(reference)}</b>). A staff member will review it and email you when there is a decision.",
                                f'In the meantime, see how the API works: <a href="{DOCS_URL}">{DOCS_URL}</a>.'])


def approved(name: str, agency: str, key: str, multiplier: int, expires_at: Optional[str], note: str = "") -> tuple[str, str, str]:
    subject = "Your Burn Check API key"
    expiry = f"This key expires on {expires_at[:10]}." if expires_at else "This key does not have an expiry date."
    text = (f"Hello {name},\n\nYour request for a Burn Check API key for {agency} was approved.\n\nYour key (keep it private):\n\n{key}\n\n"
            f"Send it with each request as a header:\n  Authorization: Bearer {key}\n\nExample:\n  curl -H 'Authorization: Bearer {key}' "
            f"'{API_URL}/v1/burn-check?lat=38.9523&lon=-92.3738'\n\nWith this key your limits are {multiplier} times the public limits. {expiry}\n"
            + (f"\nNote from our staff: {note}\n" if note else "") +
            f"\nPlease keep to the terms of use, including always showing the disclaimer next to the answer: {TERMS_URL}\n"
            f"API and embed documentation: {DOCS_URL}\n\nIf this key is exposed, reply to this email and we will replace it.\n\nShow Me Fire")
    body = _page("Your Burn Check API key", [
        f"Hello {html.escape(name)},",
        f"Your request for a Burn Check API key for <b>{html.escape(agency)}</b> was approved.",
        "Your key (keep it private, and do not put it in a public web page):"],
        f'<p style="font-family:monospace;font-size:15px;background:#f1f5f9;padding:12px;word-break:break-all;margin:0 0 14px">{html.escape(key)}</p>'
        f'<p style="margin:0 0 14px;line-height:1.5">Send it with each request as a header: <code>Authorization: Bearer &lt;your key&gt;</code>.<br>'
        f'With this key your limits are <b>{multiplier} times</b> the public limits. {html.escape(expiry)}</p>'
        + (f'<p style="margin:0 0 14px;line-height:1.5"><b>Note from our staff:</b> {html.escape(note)}</p>' if note else "")
        + f'<p style="margin:0 0 14px;line-height:1.5">Please keep to the <a href="{TERMS_URL}">terms of use</a>, including always showing the '
          f'disclaimer next to the answer. API and embed documentation: <a href="{DOCS_URL}">{DOCS_URL}</a>.</p>'
          '<p style="margin:0 0 14px;line-height:1.5">If this key is exposed, reply to this email and we will replace it.</p>')
    return subject, text, body


def denied(name: str, note: str = "") -> tuple[str, str, str]:
    subject = "About your Burn Check API key request"
    text = (f"Hello {name},\n\nThank you for asking for a Burn Check API key. We are not able to approve this request right now.\n"
            + (f"\n{note}\n" if note else "") +
            f"\nThe Burn Check API and the embeddable page are still available without a key, within the public limits: {DOCS_URL}\n\nShow Me Fire")
    paragraphs = [f"Hello {html.escape(name)},", "Thank you for asking for a Burn Check API key. We are not able to approve this request right now."]
    if note:
        paragraphs.append(html.escape(note))
    paragraphs.append(f'The Burn Check API and the embeddable page are still available without a key, within the public limits: <a href="{DOCS_URL}">{DOCS_URL}</a>.')
    return subject, text, _page("About your request", paragraphs)

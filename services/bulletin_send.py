"""Sending a bulletin to its audience. Shared by the admin "Send now" button and the scheduled-send job."""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from urllib.parse import quote

from core.database import (
    claim_bulletin_send,
    get_bulletin,
    list_active_newsletter_recipients,
    mark_bulletin_sent,
    set_bulletin_error,
)
from services import newsletter_web
from services.graphics_email import send_bulletin_email

logger = logging.getLogger(__name__)


class BulletinSendError(Exception):
    """A send that did not (fully) happen. `status_code` is what the admin endpoint answers with."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def send_bulletin(bulletin_id: int) -> dict:
    bulletin = get_bulletin(bulletin_id)
    if not bulletin:
        raise BulletinSendError(404, "Bulletin not found")
    if bulletin["status"] != "draft":
        raise BulletinSendError(409, "Bulletin has already been sent")
    if not claim_bulletin_send(bulletin_id):
        raise BulletinSendError(409, "Bulletin send is already in progress")
    recipients = (
        list_active_newsletter_recipients(bulletin["audience_list_id"])
        if bulletin.get("audience_list_id")
        else []
    )
    recipient_emails = {item["email"].strip().lower() for item in recipients}
    for email in bulletin.get("direct_recipients", []):
        if email not in recipient_emails:
            recipients.append({"email": email, "manage_token": None})
            recipient_emails.add(email)
    if not recipients:
        set_bulletin_error(bulletin_id, "No subscribers or direct recipients were provided.")
        raise BulletinSendError(409, "There are no subscribers or direct recipients to send to.")
    sent = 0
    try:
        attachments = []
        for attachment in bulletin.get("attachments", []):
            path = Path(attachment["path"])
            if not path.is_file():
                raise RuntimeError(f"Attachment is missing: {attachment['filename']}")
            attachments.append({
                "filename": attachment["filename"],
                "content": base64.b64encode(path.read_bytes()).decode("ascii"),
                "content_type": attachment["content_type"],
            })
        for recipient in recipients:
            manage_base = os.getenv("PUBLIC_WEB_URL", "https://showmefire.org").rstrip("/")
            manage_url = (
                f"{manage_base}/comms/emails/manage?token={quote(recipient['manage_token'], safe='')}"
                if recipient.get("manage_token")
                else f"{manage_base}/comms/emails"
            )
            send_bulletin_email(
                recipient["email"],
                bulletin["subject"],
                bulletin["html_body"],
                bulletin["text_body"],
                manage_url,
                attachments,
                web_url=newsletter_web.web_url(bulletin),
            )
            sent += 1
        return mark_bulletin_sent(bulletin_id, f"targeted:{sent}")
    except Exception as exc:
        set_bulletin_error(bulletin_id, f"{exc} (sent {sent} of {len(recipients)})")
        raise BulletinSendError(
            502, f"Bulletin send failed after {sent} of {len(recipients)} recipients: {exc}"
        ) from exc


def deliver_scheduled_bulletins() -> int:
    """Scheduler job (every minute): send each draft whose scheduled time has passed. A failed send clears its schedule
    (see set_bulletin_error), so it is never retried on its own, and staff are told. Returns how many were sent."""
    from datetime import datetime, timezone

    from core.database import list_due_scheduled_bulletin_ids
    from services.discord_notifier import notify_staff_alert

    sent = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for bulletin_id in list_due_scheduled_bulletin_ids(now):
        bulletin = get_bulletin(bulletin_id) or {}
        try:
            send_bulletin(bulletin_id)
            sent += 1
            logger.info("Scheduled bulletin %s sent", bulletin_id)
        except BulletinSendError as exc:
            if exc.status_code == 409 and "in progress" in exc.detail:
                continue  # another worker has it; nothing to report
            logger.warning("Scheduled bulletin %s was not sent: %s", bulletin_id, exc.detail)
            notify_staff_alert(
                alert_type="system_issue",
                title=f"Scheduled email was not sent: {bulletin.get('subject', bulletin_id)}",
                description=exc.detail,
                fields=[],
                admin_path="/admin/email-studio",
            )
    return sent

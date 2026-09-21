"""Daily county forecast email delivery."""
from __future__ import annotations

import html
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from core.database import (
    claim_newsletter_delivery,
    complete_newsletter_delivery,
    list_matching_newsletter_forecasts,
)
from services.graphics_email import send_daily_forecast_email

logger = logging.getLogger(__name__)
DANGER_LABELS = ("Low", "Moderate", "Elevated", "Critical", "Extreme")
CENTRAL = ZoneInfo("America/Chicago")


def _render_email(rows: list[dict], forecast_date: str) -> tuple[str, str, str]:
    manage_base = os.getenv("PUBLIC_WEB_URL", "https://showmefire.org").rstrip("/")
    manage_token = rows[0].get("manage_token") or ""
    manage_url = f"{manage_base}/comms/emails/manage?token={manage_token}"
    labels = [
        (html.escape(row["county_fips"]), DANGER_LABELS[int(row["danger_level"])],
         html.escape(row.get("summary") or "Fire weather forecast available."))
        for row in rows
    ]
    subject = (
        f"FireWx forecast for {labels[0][0]}: {labels[0][1]}"
        if len(labels) == 1
        else f"FireWx forecast for {len(labels)} counties"
    )
    text_lines = [
        f"Show Me Fire daily forecast for {forecast_date}",
        "",
        *[
            f"{row['county_fips']}: {label}\n{row.get('summary') or 'Fire weather forecast available.'}"
            for row, (_, label, _) in zip(rows, labels)
        ],
        "",
        f"Manage or unsubscribe: {manage_url}",
    ]
    county_rows = "".join(
        f"<li><strong>{county}</strong>: {html.escape(label)}<br>{summary}</li>"
        for county, label, summary in labels
    )
    text = "\n".join(text_lines)
    body = (
        '<div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;color:#172033">'
        f"<h1>Show Me Fire daily forecast</h1><p><strong>Date:</strong> {html.escape(forecast_date)}</p>"
        f"<ul>{county_rows}</ul><p><a href=\"{html.escape(manage_url)}\">Manage or unsubscribe from email updates</a></p>"
        "</div>"
    )
    return subject, body, text


def run_daily_forecast_delivery(forecast_date: str | None = None) -> dict:
    """Send each qualifying county forecast once; provider failures remain retryable."""
    target_date = forecast_date or datetime.now(CENTRAL).date().isoformat()
    rows = list_matching_newsletter_forecasts(target_date)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["email"], []).append(row)
    sent = 0
    failed = 0
    skipped = 0
    for email, subscriber_rows in grouped.items():
        # A single daily digest claim prevents duplicate messages when
        # multiple selected counties qualify on the same forecast date.
        if not claim_newsletter_delivery(email, "__daily__", target_date):
            skipped += 1
            continue
        try:
            subject, html_body, text_body = _render_email(subscriber_rows, target_date)
            provider_id = send_daily_forecast_email(email, subject, html_body, text_body)
            complete_newsletter_delivery(
                email, "__daily__", target_date,
                "sent", provider_message_id=provider_id,
            )
            sent += 1
        except Exception as exc:
            complete_newsletter_delivery(
                email, "__daily__", target_date,
                "failed", error=str(exc),
            )
            failed += 1
            logger.warning("Daily forecast email failed for %s: %s", email, exc)
    return {
        "forecast_date": target_date,
        "matched": len(rows),
        "recipients": len(grouped),
        "sent": sent,
        "failed": failed,
        "skipped": skipped,
    }

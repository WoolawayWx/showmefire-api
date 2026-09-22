"""Daily county forecast email delivery."""
from __future__ import annotations

import html
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from core.database import (
    claim_newsletter_delivery,
    complete_newsletter_delivery,
    list_matching_newsletter_forecasts,
)
from services.graphics_email import send_daily_forecast_email
from services.mobile_content import county_catalog

logger = logging.getLogger(__name__)
DANGER_LABELS = ("Low", "Moderate", "Elevated", "Critical", "Extreme")
CENTRAL = ZoneInfo("America/Chicago")
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "emails"
SUBJECT_FORECAST_DATE = datetime.now(CENTRAL).date().strftime("%B %d, %Y")


def _render_email(rows: list[dict], forecast_date: str) -> tuple[str, str, str]:
    manage_base = os.getenv("PUBLIC_WEB_URL", "https://showmefire.org").rstrip("/")
    manage_token = rows[0].get("manage_token") or ""
    manage_url = f"{manage_base}/comms/emails/manage?token={manage_token}"
    county_names = {item["fips"]: item["name"] for item in county_catalog()}
    labels = [
        (
            html.escape(
                county_names.get(row["county_fips"], "Missouri County")
                if county_names.get(row["county_fips"], "").lower().endswith(" county")
                else f"{county_names.get(row['county_fips'], 'Missouri County')} County"
            ),
            DANGER_LABELS[int(row["danger_level"])],
            html.escape(row.get("summary") or "Fire weather forecast available."),
        )
        for row in rows
    ]
    subject = f"Fire Weather Forecast for {SUBJECT_FORECAST_DATE}"
    county_rows = (
        "<table border=\"1\" cellpadding=\"6\" cellspacing=\"0\">"
        "<thead><tr><th>County</th><th>Fire Danger</th></tr></thead>"
        "<tbody>"
        + "".join(
            f"<tr><td>{county}</td><td>{html.escape(label)} Fire Danger</td></tr>"
            for county, label, _summary in labels
        )
        + "</tbody></table>"
    )
    county_rows_text = "\n\n".join(
        f"{county}: {label} Fire Danger\n{summary}"
        for county, label, summary in labels
    )
    manage_url_html = (
        f'<a href="{html.escape(manage_url, quote=True)}" style="color:#315f9e;">'
        "Manage or unsubscribe from email updates</a>"
    )
    replacements = {
        "{{forecast_date}}": html.escape(SUBJECT_FORECAST_DATE),
        "{{county_rows_html}}": county_rows,
        "{{county_rows_text}}": county_rows_text,
        "{{manage_url}}": html.escape(manage_url, quote=True),
        "{{manage_link_html}}": manage_url_html,
    }

    def render_template(filename: str) -> str:
        rendered = (TEMPLATE_DIR / filename).read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
        return rendered

    return subject, render_template("daily_forecast.html"), render_template("daily_forecast.txt")


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

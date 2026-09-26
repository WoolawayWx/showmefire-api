import logging
import os
import threading
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request, error

from dotenv import load_dotenv

from core.database import get_discord_admin_settings

logger = logging.getLogger(__name__)

# Ensure direct script/cron executions see the same .env values as app startup.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _PROJECT_ROOT / ".env"
if _DOTENV_PATH.exists():
    load_dotenv(dotenv_path=_DOTENV_PATH)
else:
    load_dotenv()

DISCORD_EVENT_URL = os.getenv("DISCORD_EVENT_URL", "").strip()
DISCORD_EVENT_SECRET = os.getenv("DISCORD_EVENT_SECRET", os.getenv("DISCORD_WEBHOOK_SECRET", "")).strip()
PUBLIC_API_BASE_URL = os.getenv("PUBLIC_API_BASE_URL", "https://api.showmefire.org").rstrip("/")
PUBLIC_WEB_URL = os.getenv("PUBLIC_WEB_URL", "https://showmefire.org").rstrip("/")

# Internal staff pings, keyed by the value stored in staff_alert_types.
STAFF_ALERT_TYPES: dict[str, str] = {
    "burn_ban": "Burn ban submissions",
    "fire_report": "User-reported fires",
    "incident_feedback": "Fire incident feedback",
    "site_feedback": "Site feedback",
}
DISCORD_EVENT_TIMEOUT_SEC = float(os.getenv("DISCORD_EVENT_TIMEOUT_SEC", "10"))


def _parse_role_ids(value: str | None) -> list[str]:
    role_ids: list[str] = []
    for raw in str(value or "").split(","):
        role_id = raw.strip()
        if role_id and role_id not in role_ids:
            role_ids.append(role_id)
    return role_ids


def _get_event_routing(event_type: str) -> dict[str, Any]:
    try:
        settings = get_discord_admin_settings()
    except Exception:
        settings = {}

    default_channel_id = str(settings.get("channel_id") or "").strip()
    default_channel_name = str(settings.get("channel_name") or "").strip()

    if event_type == "forecast_ready":
        channel_id = str(settings.get("forecast_channel_id") or "").strip() or default_channel_id
        channel_name = str(settings.get("forecast_channel_name") or "").strip() or default_channel_name
        mention_role_ids = _parse_role_ids(str(settings.get("forecast_role_ids") or ""))
    elif event_type == "fire_alert":
        channel_id = str(settings.get("fire_alert_channel_id") or "").strip() or default_channel_id
        channel_name = str(settings.get("fire_alert_channel_name") or "").strip() or default_channel_name
        mention_role_ids = _parse_role_ids(str(settings.get("fire_alert_role_ids") or ""))
    else:
        channel_id = str(settings.get("outlook_channel_id") or "").strip() or default_channel_id
        channel_name = str(settings.get("outlook_channel_name") or "").strip() or default_channel_name
        mention_role_ids = _parse_role_ids(str(settings.get("outlook_role_ids") or ""))

    return {
        "target_channel_id": channel_id,
        "target_channel_name": channel_name,
        "mention_role_ids": mention_role_ids,
    }


def _send_event(payload: Dict[str, Any]) -> bool:
    try:
        settings = get_discord_admin_settings()
    except Exception:
        settings = {}

    effective_url = str(settings.get("event_url_override") or DISCORD_EVENT_URL or "").strip()
    effective_secret = str(settings.get("event_secret_override") or DISCORD_EVENT_SECRET or "").strip()

    if not effective_url:
        logger.warning("Discord event URL not configured; skipping event send")
        return False

    headers = {
        "Content-Type": "application/json",
    }
    if effective_secret:
        headers["x-showmefire-secret"] = effective_secret

    try:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(effective_url, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=DISCORD_EVENT_TIMEOUT_SEC) as resp:
            status_code = getattr(resp, "status", 200)
            response_text = resp.read().decode("utf-8", errors="replace")
        if status_code >= 400:
            logger.warning(
                "Discord notifier returned %s for event %s: %s",
                status_code,
                payload.get("event_type"),
                response_text[:300],
            )
            return False
        return True
    except error.HTTPError as exc:
        try:
            err_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_text = ""
        logger.warning(
            "Discord notifier returned %s for event %s: %s",
            exc.code,
            payload.get("event_type"),
            err_text[:300],
        )
        return False
    except Exception as exc:
        logger.warning("Discord notifier send failed for %s: %s", payload.get("event_type"), exc)
        return False


def notify_outlook_published(
    *,
    day: int,
    feature_count: int,
    published_at: str,
    issue_time: Optional[str],
    valid_date: Optional[str],
    outlook_text: str,
    image_version: Optional[str] = None,
    target_channel_id: Optional[str] = None,
    target_channel_name: Optional[str] = None,
    mention_role_ids: Optional[list[str]] = None,
) -> bool:
    routing = _get_event_routing("outlook_published")
    image_suffix = f"?v={image_version}" if image_version else ""
    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": "outlook_published",
        "day": day,
        "feature_count": int(feature_count),
        "published_at": published_at,
        "issue_time": issue_time,
        "valid_date": valid_date,
        "outlook_text": outlook_text or "",
        "image_png": f"{PUBLIC_API_BASE_URL}/images/mo-outlook-day{day}.png{image_suffix}",
        "image_webp": f"{PUBLIC_API_BASE_URL}/images/mo-outlook-day{day}.webp{image_suffix}",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "target_channel_id": str(target_channel_id or routing["target_channel_id"] or "").strip() or None,
        "target_channel_name": str(target_channel_name or routing["target_channel_name"] or "").strip() or None,
        "mention_role_ids": mention_role_ids if mention_role_ids is not None else routing["mention_role_ids"],
    }
    return _send_event(payload)


def notify_forecast_ready(
    *,
    title: str,
    discussion: str,
    valid_time: Optional[str],
    updated_at: Optional[str],
    url: Optional[str] = None,
    image_url: Optional[str] = None,
    image_urls: Optional[list[str]] = None,
    target_channel_id: Optional[str] = None,
    target_channel_name: Optional[str] = None,
    mention_role_ids: Optional[list[str]] = None,
) -> bool:
    routing = _get_event_routing("forecast_ready")
    normalized_image_urls: list[str] = []
    seen: set[str] = set()

    for raw_url in (image_urls or []):
        value = str(raw_url or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized_image_urls.append(value)

    fallback = str(image_url or "").strip()
    if fallback and fallback not in seen:
        normalized_image_urls.append(fallback)

    if not normalized_image_urls:
        logger.warning("Forecast event has no image URLs; sending text-only embed payload")

    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": "forecast_ready",
        "title": title,
        "discussion": discussion,
        "valid_time": valid_time,
        "updated_at": updated_at,
        "url": url or f"{PUBLIC_API_BASE_URL}/forecasts",
        "image_url": normalized_image_urls[0] if normalized_image_urls else None,
        "image_urls": normalized_image_urls,
        "target_channel_id": str(target_channel_id or routing["target_channel_id"] or "").strip() or None,
        "target_channel_name": str(target_channel_name or routing["target_channel_name"] or "").strip() or None,
        "mention_role_ids": mention_role_ids if mention_role_ids is not None else routing["mention_role_ids"],
    }

    logger.info(
        "Sending forecast Discord event with %s image URL(s)",
        len(normalized_image_urls),
    )
    return _send_event(payload)


def _enabled_staff_alert_types(settings: Dict[str, Any]) -> set[str]:
    raw = settings.get("staff_alert_types")
    if raw is None:
        return set(STAFF_ALERT_TYPES)
    return {item.strip() for item in str(raw).split(",") if item.strip() in STAFF_ALERT_TYPES}


def build_staff_alert_payload(
    *,
    alert_type: str,
    title: str,
    description: str = "",
    fields: Optional[list[dict[str, Any]]] = None,
    admin_path: str = "",
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build a staff_alert event, or None when that alert is off/unrouted.

    Staff alerts go only to the configured staff channel. There is deliberately
    no fallback to the public default channel.
    """
    try:
        settings = get_discord_admin_settings()
    except Exception:
        settings = {}
    if not force and alert_type not in _enabled_staff_alert_types(settings):
        return None
    channel_id = str(settings.get("staff_channel_id") or "").strip()
    channel_name = str(settings.get("staff_channel_name") or "").strip()
    if not channel_id and not channel_name:
        logger.info("Staff alert %s skipped: no staff channel configured", alert_type)
        return None

    clean_fields = []
    for field in fields or []:
        name = str(field.get("name") or "").strip()[:256]
        value = str(field.get("value") or "").strip()
        if not name or not value:
            continue
        if len(value) > 1024:
            value = value[:1021] + "..."
        clean_fields.append({"name": name, "value": value, "inline": bool(field.get("inline", True))})

    return {
        "event_id": str(uuid.uuid4()),
        "event_type": "staff_alert",
        "alert_type": alert_type,
        "alert_label": STAFF_ALERT_TYPES.get(alert_type, alert_type),
        "title": title[:256],
        "description": (description or "")[:2000],
        "fields": clean_fields[:10],
        "url": f"{PUBLIC_WEB_URL}{admin_path}" if admin_path else PUBLIC_WEB_URL,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "target_channel_id": channel_id or None,
        "target_channel_name": channel_name or None,
        "mention_role_ids": _parse_role_ids(str(settings.get("staff_role_ids") or "")),
    }


def notify_staff_alert(*, background: bool = True, **kwargs: Any) -> bool:
    """Ping staff on Discord. Never raises; by default sends off-thread so the
    public request that triggered it doesn't wait on the bot."""
    try:
        payload = build_staff_alert_payload(**kwargs)
    except Exception as exc:
        logger.warning("Failed to build staff alert: %s", exc)
        return False
    if payload is None:
        return False
    if not background:
        return _send_event(payload)
    threading.Thread(target=_send_event, args=(payload,), daemon=True).start()
    return True


def notify_fire_weather_alert(
    alert: Dict[str, Any],
    *,
    image_version: Optional[str] = None,
    target_channel_id: Optional[str] = None,
    target_channel_name: Optional[str] = None,
    mention_role_ids: Optional[list[str]] = None,
) -> bool:
    """Post one NWS fire weather alert (Red Flag Warning / Fire Weather Watch).

    ``alert`` is an item from services.mobile_content.active_fire_weather_alerts.
    Public content, so it falls back to the default channel like forecasts.
    """
    routing = _get_event_routing("fire_alert")
    image_suffix = f"?v={image_version}" if image_version else ""
    payload = {
        "event_id": f"fire_alert:{alert.get('id')}",
        "event_type": "fire_alert",
        "alert_id": str(alert.get("id") or ""),
        "event": str(alert.get("event") or "Fire Weather Alert"),
        "headline": str(alert.get("headline") or ""),
        "area_description": str(alert.get("areaDescription") or "Missouri"),
        "description": str(alert.get("description") or "")[:1500],
        "instruction": str(alert.get("instruction") or "")[:500],
        "severity": str(alert.get("severity") or ""),
        "onset": alert.get("onset"),
        "expires": alert.get("expires"),
        "sent": alert.get("sent"),
        "url": PUBLIC_WEB_URL,
        "image_url": f"{PUBLIC_API_BASE_URL}/images/mo-firewx-alerts.png{image_suffix}",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "target_channel_id": str(target_channel_id or routing["target_channel_id"] or "").strip() or None,
        "target_channel_name": str(target_channel_name or routing["target_channel_name"] or "").strip() or None,
        "mention_role_ids": mention_role_ids if mention_role_ids is not None else routing["mention_role_ids"],
    }
    return _send_event(payload)


def process_fire_weather_alerts_for_discord(alerts: list[Dict[str, Any]]) -> int:
    """Post alerts not yet sent to Discord; returns how many were posted.

    The first run only records what's already active. An alert is marked
    posted only after the bot accepts it, so a bot outage retries next poll.
    """
    from core.database import claim_unposted_discord_fire_alerts, mark_discord_fire_alert_posted

    by_id = {str(alert.get("id")): alert for alert in alerts if alert.get("id")}
    pending = claim_unposted_discord_fire_alerts(list(by_id))
    image_version = datetime.utcnow().strftime("%Y%m%d%H%M")
    posted = 0
    for alert_id in pending:
        if notify_fire_weather_alert(by_id[alert_id], image_version=image_version):
            mark_discord_fire_alert_posted(alert_id)
            posted += 1
    return posted

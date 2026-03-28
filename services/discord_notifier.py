import logging
import os
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

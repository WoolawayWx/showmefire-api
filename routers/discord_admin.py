import logging
import os
from datetime import datetime, timezone
from typing import Optional
from urllib import error, request
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.database import get_discord_admin_settings, update_discord_admin_settings
from core.security import verify_token
from services.discord_notifier import (
    DISCORD_EVENT_SECRET,
    DISCORD_EVENT_TIMEOUT_SEC,
    DISCORD_EVENT_URL,
    notify_forecast_ready,
    notify_outlook_published,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["discord-admin"])
_TEST_EVENT_LAST_SENT_AT: dict[str, datetime] = {}


class DiscordConfigUpdateRequest(BaseModel):
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    forecast_channel_id: Optional[str] = None
    forecast_channel_name: Optional[str] = None
    outlook_channel_id: Optional[str] = None
    outlook_channel_name: Optional[str] = None
    forecast_role_ids: Optional[str] = None
    outlook_role_ids: Optional[str] = None
    event_url_override: Optional[str] = None
    event_secret_override: Optional[str] = None
    clear_event_secret_override: Optional[bool] = False
    image_fetch_retries: Optional[int] = Field(default=None, ge=1, le=10)
    image_fetch_timeout_ms: Optional[int] = Field(default=None, ge=1000, le=30000)
    dedupe_ttl_hours: Optional[int] = Field(default=None, ge=1, le=48)

    @field_validator(
        "channel_id",
        "channel_name",
        "forecast_channel_id",
        "forecast_channel_name",
        "outlook_channel_id",
        "outlook_channel_name",
        "forecast_role_ids",
        "outlook_role_ids",
        "event_url_override",
        "event_secret_override",
        mode="before",
    )
    @classmethod
    def _normalize_optional_string(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return str(value).strip()


class DiscordTestEventRequest(BaseModel):
    event_type: str = "outlook_published"
    day: int = Field(default=2, ge=2, le=3)

    @field_validator("event_type")
    @classmethod
    def _validate_event_type(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized not in {"outlook_published", "forecast_ready"}:
            raise ValueError("event_type must be 'outlook_published' or 'forecast_ready'")
        return normalized


def _require_admin(token: str) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _normalize_role_csv(value: str | None) -> str:
    unique_ids: list[str] = []
    for raw in str(value or "").split(","):
        role_id = raw.strip()
        if not role_id:
            continue
        if role_id not in unique_ids:
            unique_ids.append(role_id)
    return ",".join(unique_ids)


def _build_health_url() -> str:
    override = os.getenv("DISCORD_BOT_HEALTH_URL", "").strip()
    if override:
        return override

    if DISCORD_EVENT_URL:
        parsed = urlparse(DISCORD_EVENT_URL)
        if parsed.scheme and parsed.netloc:
            path = parsed.path.rstrip("/")
            if path.endswith("/events"):
                health_path = f"{path[:-7]}/health"
            else:
                health_path = "/health"
            return parsed._replace(path=health_path, query="", fragment="").geturl()

    return "http://host.docker.internal:8787/health"


def _build_servers_url() -> str:
    override = os.getenv("DISCORD_BOT_SERVERS_URL", "").strip()
    if override:
        return override

    if DISCORD_EVENT_URL:
        parsed = urlparse(DISCORD_EVENT_URL)
        if parsed.scheme and parsed.netloc:
            path = parsed.path.rstrip("/")
            if path.endswith("/events"):
                servers_path = f"{path[:-7]}/servers"
            else:
                servers_path = "/servers"
            return parsed._replace(path=servers_path, query="", fragment="").geturl()

    return "http://host.docker.internal:8787/servers"


def _fetch_discord_health() -> dict:
    health_url = _build_health_url()
    req = request.Request(health_url, method="GET")
    try:
        with request.urlopen(req, timeout=4) as resp:
            status_code = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")
            if status_code >= 400:
                return {
                    "ok": False,
                    "reachable": False,
                    "url": health_url,
                    "error": f"health endpoint returned {status_code}",
                }
            import json

            payload = json.loads(body)
            return {
                "ok": bool(payload.get("ok", True)),
                "reachable": True,
                "url": health_url,
                "bot_ready": bool(payload.get("bot_ready")),
                "channel_resolved": bool(payload.get("channel_resolved")),
                "channel_id": payload.get("channel_id"),
                "uptime_sec": payload.get("uptime_sec"),
            }
    except error.HTTPError as exc:
        return {
            "ok": False,
            "reachable": False,
            "url": health_url,
            "error": f"HTTP {exc.code}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "reachable": False,
            "url": health_url,
            "error": str(exc),
        }


def _fetch_discord_servers() -> dict:
    servers_url = _build_servers_url()
    settings = get_discord_admin_settings()
    effective_secret = str(settings.get("event_secret_override") or DISCORD_EVENT_SECRET or "").strip()
    headers = {}
    if effective_secret:
        headers["x-showmefire-secret"] = effective_secret

    req = request.Request(servers_url, method="GET", headers=headers)
    try:
        with request.urlopen(req, timeout=6) as resp:
            status_code = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8", errors="replace")
            if status_code >= 400:
                return {
                    "ok": False,
                    "url": servers_url,
                    "servers": [],
                    "error": f"servers endpoint returned {status_code}",
                }
            import json

            payload = json.loads(body)
            return {
                "ok": bool(payload.get("ok", True)),
                "url": servers_url,
                "servers": payload.get("servers") or [],
            }
    except error.HTTPError as exc:
        return {
            "ok": False,
            "url": servers_url,
            "servers": [],
            "error": f"HTTP {exc.code}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": servers_url,
            "servers": [],
            "error": str(exc),
        }


@router.get("/api/admin/discord/config")
async def get_discord_config(token: str):
    _require_admin(token)
    settings = get_discord_admin_settings()

    return {
        "success": True,
        "config": {
            "channel_id": settings.get("channel_id") or "",
            "channel_name": settings.get("channel_name") or "",
            "forecast_channel_id": settings.get("forecast_channel_id") or "",
            "forecast_channel_name": settings.get("forecast_channel_name") or "",
            "outlook_channel_id": settings.get("outlook_channel_id") or "",
            "outlook_channel_name": settings.get("outlook_channel_name") or "",
            "forecast_role_ids": settings.get("forecast_role_ids") or "",
            "outlook_role_ids": settings.get("outlook_role_ids") or "",
            "event_url_override": settings.get("event_url_override") or "",
            "event_secret_override_set": bool(settings.get("event_secret_override") or ""),
            "image_fetch_retries": int(settings.get("image_fetch_retries") or 3),
            "image_fetch_timeout_ms": int(settings.get("image_fetch_timeout_ms") or 5000),
            "dedupe_ttl_hours": int((settings.get("dedupe_ttl_ms") or 21600000) / 3600000),
            "updated_by": settings.get("updated_by"),
            "updated_at": settings.get("updated_at"),
            "restart_required_fields": ["bot_token", "webhook_secret", "webhook_port"],
            "secrets": {
                "event_secret_configured": bool(DISCORD_EVENT_SECRET),
                "event_url_configured": bool(DISCORD_EVENT_URL),
            },
        },
    }


@router.post("/api/admin/discord/config")
async def update_discord_config(payload: DiscordConfigUpdateRequest, token: str):
    email = _require_admin(token)

    if payload.channel_id is not None and payload.channel_name is not None:
        if not payload.channel_id and not payload.channel_name:
            raise HTTPException(status_code=422, detail="Provide channel_id or channel_name")

    updated = update_discord_admin_settings(
        channel_id=payload.channel_id,
        channel_name=payload.channel_name,
        forecast_channel_id=payload.forecast_channel_id,
        forecast_channel_name=payload.forecast_channel_name,
        outlook_channel_id=payload.outlook_channel_id,
        outlook_channel_name=payload.outlook_channel_name,
        forecast_role_ids=_normalize_role_csv(payload.forecast_role_ids) if payload.forecast_role_ids is not None else None,
        outlook_role_ids=_normalize_role_csv(payload.outlook_role_ids) if payload.outlook_role_ids is not None else None,
        event_url_override=str(payload.event_url_override or "").strip() if payload.event_url_override is not None else None,
        event_secret_override=(
            ""
            if payload.clear_event_secret_override
            else (str(payload.event_secret_override or "").strip() if payload.event_secret_override is not None else None)
        ),
        image_fetch_retries=payload.image_fetch_retries,
        image_fetch_timeout_ms=payload.image_fetch_timeout_ms,
        dedupe_ttl_ms=(payload.dedupe_ttl_hours * 3600000) if payload.dedupe_ttl_hours is not None else None,
        updated_by=email,
    )

    applied_fields = []
    if payload.channel_id is not None:
        applied_fields.append("channel_id")
    if payload.channel_name is not None:
        applied_fields.append("channel_name")
    if payload.image_fetch_retries is not None:
        applied_fields.append("image_fetch_retries")
    if payload.image_fetch_timeout_ms is not None:
        applied_fields.append("image_fetch_timeout_ms")
    if payload.dedupe_ttl_hours is not None:
        applied_fields.append("dedupe_ttl_hours")
    if payload.forecast_channel_id is not None:
        applied_fields.append("forecast_channel_id")
    if payload.forecast_channel_name is not None:
        applied_fields.append("forecast_channel_name")
    if payload.outlook_channel_id is not None:
        applied_fields.append("outlook_channel_id")
    if payload.outlook_channel_name is not None:
        applied_fields.append("outlook_channel_name")
    if payload.forecast_role_ids is not None:
        applied_fields.append("forecast_role_ids")
    if payload.outlook_role_ids is not None:
        applied_fields.append("outlook_role_ids")
    if payload.event_url_override is not None:
        applied_fields.append("event_url_override")
    if payload.event_secret_override is not None:
        applied_fields.append("event_secret_override")
    if payload.clear_event_secret_override:
        applied_fields.append("clear_event_secret_override")

    logger.info("Discord admin config updated by %s: %s", email, ",".join(applied_fields) or "none")

    return {
        "success": True,
        "message": "Discord admin settings saved",
        "applied_fields": applied_fields,
        "config": {
            "channel_id": updated.get("channel_id") or "",
            "channel_name": updated.get("channel_name") or "",
            "forecast_channel_id": updated.get("forecast_channel_id") or "",
            "forecast_channel_name": updated.get("forecast_channel_name") or "",
            "outlook_channel_id": updated.get("outlook_channel_id") or "",
            "outlook_channel_name": updated.get("outlook_channel_name") or "",
            "forecast_role_ids": updated.get("forecast_role_ids") or "",
            "outlook_role_ids": updated.get("outlook_role_ids") or "",
            "event_url_override": updated.get("event_url_override") or "",
            "event_secret_override_set": bool(updated.get("event_secret_override") or ""),
            "image_fetch_retries": int(updated.get("image_fetch_retries") or 3),
            "image_fetch_timeout_ms": int(updated.get("image_fetch_timeout_ms") or 5000),
            "dedupe_ttl_hours": int((updated.get("dedupe_ttl_ms") or 21600000) / 3600000),
            "updated_by": updated.get("updated_by"),
            "updated_at": updated.get("updated_at"),
            "requires_restart": ["bot settings are env-based in production deployments"],
        },
    }


@router.get("/api/admin/discord/status")
async def get_discord_status(token: str):
    _require_admin(token)
    settings = get_discord_admin_settings()
    health = _fetch_discord_health()

    return {
        "success": True,
        "status": {
            "health": health,
            "event_url_configured": bool(DISCORD_EVENT_URL),
            "event_secret_configured": bool(DISCORD_EVENT_SECRET),
            "event_url_override_set": bool(settings.get("event_url_override") or ""),
            "event_secret_override_set": bool(settings.get("event_secret_override") or ""),
            "event_timeout_sec": DISCORD_EVENT_TIMEOUT_SEC,
            "saved_channel_id": settings.get("channel_id") or "",
            "saved_channel_name": settings.get("channel_name") or "",
            "saved_routing": {
                "forecast": {
                    "channel_id": settings.get("forecast_channel_id") or "",
                    "channel_name": settings.get("forecast_channel_name") or "",
                    "role_ids": settings.get("forecast_role_ids") or "",
                },
                "outlook": {
                    "channel_id": settings.get("outlook_channel_id") or "",
                    "channel_name": settings.get("outlook_channel_name") or "",
                    "role_ids": settings.get("outlook_role_ids") or "",
                },
            },
            "saved_delivery": {
                "image_fetch_retries": int(settings.get("image_fetch_retries") or 3),
                "image_fetch_timeout_ms": int(settings.get("image_fetch_timeout_ms") or 5000),
                "dedupe_ttl_hours": int((settings.get("dedupe_ttl_ms") or 21600000) / 3600000),
            },
        },
    }


@router.get("/api/admin/discord/servers")
async def get_discord_servers(token: str):
    _require_admin(token)
    payload = _fetch_discord_servers()
    return {
        "success": payload.get("ok", False),
        "url": payload.get("url"),
        "servers": payload.get("servers") or [],
        "error": payload.get("error"),
    }


@router.post("/api/admin/discord/test-event")
async def send_discord_test_event(payload: DiscordTestEventRequest, token: str):
    email = _require_admin(token)
    now_dt = datetime.now(timezone.utc)
    last_sent = _TEST_EVENT_LAST_SENT_AT.get(email)
    if last_sent and (now_dt - last_sent).total_seconds() < 15:
        raise HTTPException(status_code=429, detail="Please wait before sending another test event")

    _TEST_EVENT_LAST_SENT_AT[email] = now_dt
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        if payload.event_type == "forecast_ready":
            sent = notify_forecast_ready(
                title="Test Forecast Event",
                discussion="This is a test forecast notification from Show Me Fire admin panel.",
                valid_time=now_iso,
                updated_at=now_iso,
                url="https://showmefire.org/forecasts",
                image_url="https://api.showmefire.org/images/mo-forecastfiredanger.png",
            )
        else:
            sent = notify_outlook_published(
                day=payload.day,
                feature_count=0,
                published_at=now_iso,
                issue_time=now_iso,
                valid_date=now_iso[:10],
                outlook_text="Test outlook event payload from admin panel.",
                image_version=str(int(datetime.now(timezone.utc).timestamp())),
            )

        logger.info("Discord test event requested by %s type=%s day=%s sent=%s", email, payload.event_type, payload.day, sent)
        return {
            "success": True,
            "message": "Discord test event processed",
            "event_type": payload.event_type,
            "day": payload.day,
            "sent": bool(sent),
            "requested_by": email,
            "at": now_iso,
        }
    except Exception as exc:
        logger.error("Failed to send Discord test event for %s: %s", email, exc)
        raise HTTPException(status_code=500, detail="Failed to send Discord test event") from exc

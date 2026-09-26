import logging
import os
from datetime import datetime, timedelta, timezone
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
    STAFF_ALERT_TYPES,
    notify_fire_weather_alert,
    notify_forecast_ready,
    notify_outlook_published,
    notify_staff_alert,
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
    fire_alert_channel_id: Optional[str] = None
    fire_alert_channel_name: Optional[str] = None
    fire_alert_role_ids: Optional[str] = None
    staff_channel_id: Optional[str] = None
    staff_channel_name: Optional[str] = None
    staff_role_ids: Optional[str] = None
    staff_alert_types: Optional[list[str]] = None

    @field_validator("staff_alert_types")
    @classmethod
    def _validate_staff_alert_types(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        unknown = [item for item in value if item not in STAFF_ALERT_TYPES]
        if unknown:
            raise ValueError(f"unknown staff alert types: {', '.join(unknown)}")
        return [key for key in STAFF_ALERT_TYPES if key in value]

    @field_validator(
        "fire_alert_channel_id",
        "fire_alert_channel_name",
        "fire_alert_role_ids",
        "staff_channel_id",
        "staff_channel_name",
        "staff_role_ids",
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
        if normalized not in {"outlook_published", "forecast_ready", "fire_alert", "staff_alert"}:
            raise ValueError("event_type must be 'outlook_published', 'forecast_ready', 'fire_alert', or 'staff_alert'")
        return normalized


def _fire_alert_config(settings: dict) -> dict:
    return {
        "fire_alert_channel_id": settings.get("fire_alert_channel_id") or "",
        "fire_alert_channel_name": settings.get("fire_alert_channel_name") or "",
        "fire_alert_role_ids": settings.get("fire_alert_role_ids") or "",
    }


def _staff_config(settings: dict) -> dict:
    raw_types = settings.get("staff_alert_types")
    enabled = (
        list(STAFF_ALERT_TYPES)
        if raw_types is None
        else [item for item in str(raw_types).split(",") if item in STAFF_ALERT_TYPES]
    )
    return {
        "staff_channel_id": settings.get("staff_channel_id") or "",
        "staff_channel_name": settings.get("staff_channel_name") or "",
        "staff_role_ids": settings.get("staff_role_ids") or "",
        "staff_alert_types": enabled,
        "staff_alert_type_options": [{"key": key, "label": label} for key, label in STAFF_ALERT_TYPES.items()],
    }


def _require_admin(token: Optional[str] = None) -> str:
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
                "application_id": payload.get("application_id"),
                "guild_count": payload.get("guild_count"),
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


DISCORD_API_BASE = "https://discord.com/api/v10"
# View Channel, Send Messages, Embed Links, Attach Files, Mention @everyone/roles
BOT_INVITE_PERMISSIONS = 1024 | 2048 | 16384 | 32768 | 131072
_TEXT_CHANNEL_TYPES = {0, 5}  # guild text, announcement


def _discord_rest_get(path: str, bot_token: str):
    import json

    req = request.Request(
        f"{DISCORD_API_BASE}{path}",
        method="GET",
        headers={
            "Authorization": f"Bot {bot_token}",
            "User-Agent": "ShowMeFire (https://showmefire.org, 1.0)",
        },
    )
    with request.urlopen(req, timeout=8) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _discover_servers_via_rest(bot_token: str) -> dict:
    """Ask Discord directly with the bot token; works even if the bot process is down."""
    me = _discord_rest_get("/users/@me", bot_token)
    guilds = _discord_rest_get("/users/@me/guilds", bot_token)
    servers, errors = [], []
    for guild in guilds:
        try:
            channels = _discord_rest_get(f"/guilds/{guild['id']}/channels", bot_token)
            roles = _discord_rest_get(f"/guilds/{guild['id']}/roles", bot_token)
        except Exception as exc:
            errors.append({"id": guild["id"], "name": guild.get("name"), "error": str(exc)})
            continue
        servers.append({
            "id": guild["id"],
            "name": guild.get("name") or guild["id"],
            "channels": sorted(
                ({"id": c["id"], "name": c.get("name", "")} for c in channels if c.get("type") in _TEXT_CHANNEL_TYPES),
                key=lambda c: c["name"],
            ),
            "roles": sorted(
                (
                    {"id": r["id"], "name": r.get("name", "")}
                    for r in roles
                    if r.get("name") != "@everyone" and not r.get("managed")
                ),
                key=lambda r: r["name"],
            ),
        })
    servers.sort(key=lambda g: g["name"].lower())
    return {"ok": True, "source": "discord_api", "application_id": me.get("id"), "servers": servers, "errors": errors}


def _discover_servers_via_bot() -> dict:
    import json

    servers_url = _build_servers_url()
    settings = get_discord_admin_settings()
    effective_secret = str(settings.get("event_secret_override") or DISCORD_EVENT_SECRET or "").strip()
    headers = {"x-showmefire-secret": effective_secret} if effective_secret else {}
    req = request.Request(servers_url, method="GET", headers=headers)
    try:
        with request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8", errors="replace")).get("message") or ""
        except Exception:
            detail = ""
        hint = {401: "bot rejected the shared secret (DISCORD_EVENT_SECRET mismatch)",
                503: "bot is running but not connected to Discord"}.get(exc.code, "")
        raise RuntimeError(f"bot /servers returned HTTP {exc.code}" + (f": {hint or detail}" if hint or detail else "")) from exc
    except Exception as exc:
        raise RuntimeError(f"bot unreachable at {servers_url}: {exc}") from exc
    return {
        "ok": True,
        "source": "bot",
        "application_id": payload.get("application_id"),
        "servers": payload.get("servers") or [],
        "errors": payload.get("errors") or [],
    }


def _fetch_discord_servers() -> dict:
    """Discover servers/channels/roles, preferring the Discord API, then the
    bot, then the last good result so the admin page never goes blank."""
    from core.database import get_discord_server_cache, save_discord_server_cache

    attempts: list[str] = []
    bot_token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    discoverers = []
    if bot_token:
        discoverers.append(("Discord API", lambda: _discover_servers_via_rest(bot_token)))
    discoverers.append(("bot", _discover_servers_via_bot))

    for label, discover in discoverers:
        try:
            result = discover()
        except Exception as exc:
            attempts.append(f"{label}: {exc}")
            continue
        try:
            save_discord_server_cache({
                "servers": result["servers"],
                "application_id": result.get("application_id"),
                "source": result["source"],
            })
        except Exception as exc:
            logger.warning("Failed to cache Discord servers: %s", exc)
        return {**result, "stale": False, "synced_at": datetime.now(timezone.utc).isoformat(), "url": _build_servers_url()}

    cached = get_discord_server_cache()
    message = "; ".join(attempts)
    if cached:
        return {
            "ok": True,
            "source": "cache",
            "stale": True,
            "synced_at": cached.get("synced_at"),
            "application_id": cached.get("application_id"),
            "servers": cached.get("servers") or [],
            "errors": [],
            "error": message,
            "url": _build_servers_url(),
        }
    return {"ok": False, "stale": True, "servers": [], "errors": [], "error": message, "url": _build_servers_url()}


def _bot_application_id(servers_payload: Optional[dict] = None) -> str:
    configured = os.getenv("DISCORD_CLIENT_ID", "").strip()
    if configured:
        return configured
    if servers_payload and servers_payload.get("application_id"):
        return str(servers_payload["application_id"])
    health = _fetch_discord_health()
    if health.get("application_id"):
        return str(health["application_id"])
    from core.database import get_discord_server_cache
    cached = get_discord_server_cache() or {}
    return str(cached.get("application_id") or "")


def _bot_invite_url(application_id: str, guild_id: Optional[str] = None) -> str:
    from urllib.parse import urlencode

    params = {
        "client_id": application_id,
        "scope": "bot applications.commands",
        "permissions": str(BOT_INVITE_PERMISSIONS),
    }
    if guild_id:
        params["guild_id"] = guild_id
        params["disable_guild_select"] = "true"
    return f"https://discord.com/oauth2/authorize?{urlencode(params)}"


@router.get("/api/admin/discord/config")
async def get_discord_config(token: Optional[str] = None):
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
            **_fire_alert_config(settings),
            **_staff_config(settings),
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
async def update_discord_config(payload: DiscordConfigUpdateRequest, token: Optional[str] = None):
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
        fire_alert_channel_id=payload.fire_alert_channel_id,
        fire_alert_channel_name=payload.fire_alert_channel_name,
        fire_alert_role_ids=_normalize_role_csv(payload.fire_alert_role_ids) if payload.fire_alert_role_ids is not None else None,
        staff_channel_id=payload.staff_channel_id,
        staff_channel_name=payload.staff_channel_name,
        staff_role_ids=_normalize_role_csv(payload.staff_role_ids) if payload.staff_role_ids is not None else None,
        staff_alert_types=",".join(payload.staff_alert_types) if payload.staff_alert_types is not None else None,
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
    for staff_field in (
        "fire_alert_channel_id", "fire_alert_channel_name", "fire_alert_role_ids",
        "staff_channel_id", "staff_channel_name", "staff_role_ids", "staff_alert_types",
    ):
        if getattr(payload, staff_field) is not None:
            applied_fields.append(staff_field)

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
            **_fire_alert_config(updated),
            **_staff_config(updated),
            "updated_by": updated.get("updated_by"),
            "updated_at": updated.get("updated_at"),
            "requires_restart": ["bot settings are env-based in production deployments"],
        },
    }


@router.get("/api/admin/discord/status")
async def get_discord_status(token: Optional[str] = None):
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
                "fire_alert": {
                    "channel_id": settings.get("fire_alert_channel_id") or "",
                    "channel_name": settings.get("fire_alert_channel_name") or "",
                    "role_ids": settings.get("fire_alert_role_ids") or "",
                },
                "staff": {
                    "channel_id": settings.get("staff_channel_id") or "",
                    "channel_name": settings.get("staff_channel_name") or "",
                    "role_ids": settings.get("staff_role_ids") or "",
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
async def get_discord_servers(token: Optional[str] = None):
    _require_admin(token)
    payload = _fetch_discord_servers()
    application_id = _bot_application_id(payload)
    return {
        "success": payload.get("ok", False),
        "url": payload.get("url"),
        "source": payload.get("source"),
        "stale": payload.get("stale", False),
        "synced_at": payload.get("synced_at"),
        "servers": payload.get("servers") or [],
        "server_errors": payload.get("errors") or [],
        "error": payload.get("error"),
        "invite_url": _bot_invite_url(application_id) if application_id else None,
    }


@router.post("/api/admin/discord/test-event")
async def send_discord_test_event(payload: DiscordTestEventRequest, token: Optional[str] = None):
    email = _require_admin(token)
    now_dt = datetime.now(timezone.utc)
    last_sent = _TEST_EVENT_LAST_SENT_AT.get(email)
    if last_sent and (now_dt - last_sent).total_seconds() < 15:
        raise HTTPException(status_code=429, detail="Please wait before sending another test event")

    _TEST_EVENT_LAST_SENT_AT[email] = now_dt
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        if payload.event_type == "staff_alert":
            settings = get_discord_admin_settings()
            if not (settings.get("staff_channel_id") or settings.get("staff_channel_name")):
                raise HTTPException(status_code=400, detail="Save a staff alert channel before sending a test")
            sent = notify_staff_alert(
                background=False,
                force=True,
                alert_type="burn_ban",
                title="Test staff alert",
                description="This is a test staff alert from the Show Me Fire admin panel.",
                fields=[{"name": "Requested by", "value": email}],
                admin_path="/admin/discord",
            )
        elif payload.event_type == "fire_alert":
            later_iso = (now_dt + timedelta(hours=12)).isoformat()
            sent = notify_fire_weather_alert(
                {
                    # Unique per test so the bot's dedupe doesn't swallow repeats.
                    "id": f"test-{int(now_dt.timestamp())}",
                    "event": "Red Flag Warning",
                    "headline": "TEST: Red Flag Warning issued for central Missouri (admin panel test)",
                    "areaDescription": "Boone; Cole; Callaway",
                    "description": "This is a test fire weather alert from the Show Me Fire admin panel.",
                    "severity": "Severe",
                    "onset": now_iso,
                    "expires": later_iso,
                    "sent": now_iso,
                },
                image_version=str(int(now_dt.timestamp())),
            )
        elif payload.event_type == "forecast_ready":
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to send Discord test event for %s: %s", email, exc)
        raise HTTPException(status_code=500, detail="Failed to send Discord test event") from exc

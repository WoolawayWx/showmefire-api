"""
Direct Discord REST delivery using the bot token.

When DISCORD_BOT_TOKEN is set on the API, notifications are posted straight to
Discord's API, like a typical bot dashboard backend. That removes the
API -> bot webhook hop and its shared secret, which could drift out of sync
(e.g. a browser autofilling the admin page's secret override). Without the
token, services.discord_notifier falls back to the bot webhook.

Embeds mirror discord-bot/src/embeds/*.js so posts look the same either way.
"""
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
USER_AGENT = "ShowMeFire (https://showmefire.org, 1.0)"
TEXT_CHANNEL_TYPES = {0, 5}  # guild text, announcement

ORANGE = 0xE65100
GRAY = 0x6B7280
STAFF_COLORS = {
    "burn_ban": 0xE65100,
    "fire_report": 0xDC2626,
    "incident_feedback": 0xF59E0B,
    "site_feedback": 0x2563EB,
}
FIRE_ALERT_COLORS = {"Red Flag Warning": 0xDC2626, "Fire Weather Watch": 0xF59E0B}


class DiscordRestError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Discord API HTTP {status}: {message}")
        self.status = status


def bot_token() -> str:
    return os.getenv("DISCORD_BOT_TOKEN", "").strip()


def is_configured() -> bool:
    return bool(bot_token())


# --- HTTP -----------------------------------------------------------------

def _call(method: str, path: str, *, body: Optional[bytes] = None, content_type: Optional[str] = None,
          token: Optional[str] = None, auth_scheme: str = "Bot", timeout: float = 10.0) -> Any:
    """One Discord API call; retries once on a 429 using Discord's retry_after."""
    token = token or bot_token()
    headers = {"Authorization": f"{auth_scheme} {token}", "User-Agent": USER_AGENT}
    if content_type:
        headers["Content-Type"] = content_type
    for attempt in range(2):
        req = request.Request(f"{DISCORD_API_BASE}{path}", data=body, method=method, headers=headers)
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else None
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw)
            except ValueError:
                detail = {"message": raw[:200]}
            if exc.code == 429 and attempt == 0:
                time.sleep(min(float(detail.get("retry_after") or 1.0), 5.0))
                continue
            raise DiscordRestError(exc.code, str(detail.get("message") or raw[:200])) from exc
    raise DiscordRestError(429, "rate limited")


def get(path: str, **kwargs) -> Any:
    return _call("GET", path, **kwargs)


def _multipart(payload: Dict[str, Any], files: List[Tuple[str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = f"smf{uuid.uuid4().hex}"
    parts: List[bytes] = []

    def add(headers: str, content: bytes) -> None:
        parts.append(f"--{boundary}\r\n{headers}\r\n\r\n".encode() + content + b"\r\n")

    add('Content-Disposition: form-data; name="payload_json"\r\nContent-Type: application/json',
        json.dumps(payload).encode())
    for index, (filename, content, mime) in enumerate(files):
        add(f'Content-Disposition: form-data; name="files[{index}]"; filename="{filename}"\r\nContent-Type: {mime}',
            content)
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def send_message(channel_id: str, *, content: str = "", embeds: Optional[List[Dict]] = None,
                 mention_role_ids: Optional[List[str]] = None,
                 files: Optional[List[Tuple[str, bytes, str]]] = None) -> Dict:
    payload: Dict[str, Any] = {
        "content": content or None,
        "embeds": embeds or [],
        # Only the configured roles may be pinged; never @everyone or users.
        "allowed_mentions": {"parse": [], "roles": list(mention_role_ids or [])},
    }
    if files:
        payload["attachments"] = [{"id": i, "filename": name} for i, (name, _, _) in enumerate(files)]
        body, content_type = _multipart(payload, files)
    else:
        body, content_type = json.dumps(payload).encode(), "application/json"
    return _call("POST", f"/channels/{channel_id}/messages", body=body, content_type=content_type, timeout=20.0)


# --- Channel resolution ---------------------------------------------------

def _find_channel_by_name(name: str) -> Optional[str]:
    for guild in get("/users/@me/guilds") or []:
        try:
            channels = get(f"/guilds/{guild['id']}/channels") or []
        except DiscordRestError:
            continue
        for channel in channels:
            if channel.get("type") in TEXT_CHANNEL_TYPES and channel.get("name") == name:
                return channel["id"]
    return None


def resolve_channel(channel_id: Optional[str], channel_name: Optional[str], *,
                    default_id: str = "", default_name: str = "", strict: bool = False) -> str:
    """Mirror discord-bot ChannelResolver: id, then name, then (non-strict) default."""
    candidates = [(channel_id, channel_name)]
    if not strict:
        candidates.append((default_id, default_name))
    for cid, cname in candidates:
        cid, cname = str(cid or "").strip(), str(cname or "").strip()
        if cid:
            try:
                channel = get(f"/channels/{cid}")
                if channel and channel.get("type") in TEXT_CHANNEL_TYPES:
                    return cid
            except DiscordRestError as exc:
                logger.warning("Discord channel %s not accessible: %s", cid, exc)
        if cname:
            found = _find_channel_by_name(cname)
            if found:
                return found
    if strict:
        raise RuntimeError("Target channel for this event is not accessible; refusing to fall back to the default channel.")
    raise RuntimeError("Unable to resolve a Discord channel. Set a channel on the admin Discord page.")


# --- Embeds (mirroring discord-bot/src/embeds) ----------------------------

def _clip(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _discord_time(value: Any, style: str = "f") -> Optional[str]:
    parsed = _parse_time(value)
    return f"<t:{int(parsed.timestamp())}:{style}>" if parsed else None


def _iso(value: Optional[datetime]) -> str:
    return (value or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


def _forecast_embed(p: Dict) -> Dict:
    discussion = _clip(p.get("discussion"), 700)
    valid = _parse_time(p.get("valid_time") or p.get("updated_at"))
    fields = [
        {"name": "Forecast For", "value": valid.date().isoformat() if valid else str(p.get("valid_time") or "Unknown"), "inline": True},
        {"name": "Posted", "value": f"<t:{int(time.time())}:F>", "inline": True},
    ]
    if p.get("url"):
        fields.append({"name": "View The Full Forecast", "value": str(p["url"]), "inline": False})
    return {"title": _clip(p.get("title") or "Daily Forecast Update", 256),
            "description": discussion or "A forecast update is available.",
            "color": ORANGE, "fields": fields, "timestamp": _iso(None)}


def _outlook_embed(p: Dict) -> Dict:
    day = p.get("day") or "?"
    features = int(p.get("feature_count") or 0)
    published = _parse_time(p.get("published_at") or p.get("updated_at")) or datetime.now(timezone.utc)
    valid = _parse_time(p.get("valid_date") or p.get("issue_time") or p.get("published_at"))
    return {
        "title": f"Day {day} Outlook Updated",
        "description": ("A new fire weather outlook has been published." if features > 0
                        else "No outlook polygons were published for this update."),
        "color": ORANGE if features > 0 else GRAY,
        "fields": [
            {"name": "Day", "value": str(day), "inline": True},
            {"name": "Outlook For", "value": valid.date().isoformat() if valid else str(p.get("valid_date") or "Unknown"), "inline": True},
            {"name": "Published", "value": f"<t:{int(published.timestamp())}:F>", "inline": True},
        ],
        "timestamp": _iso(published),
    }


def _fire_alert_embed(p: Dict) -> Dict:
    event = _clip(p.get("event") or "Fire Weather Alert", 100)
    description = "\n\n".join(x for x in (_clip(p.get("headline"), 500), _clip(p.get("description"), 1400)) if x)
    fields = [
        {"name": "Begins", "value": _discord_time(p.get("onset")), "inline": True},
        {"name": "Expires", "value": _discord_time(p.get("expires")), "inline": True},
        {"name": "Severity", "value": _clip(p.get("severity"), 100), "inline": True},
        {"name": "Precautions", "value": _clip(p.get("instruction"), 1024), "inline": False},
    ]
    embed = {
        "title": _clip(f"{event}: {p.get('area_description') or 'Missouri'}", 256),
        "color": FIRE_ALERT_COLORS.get(p.get("event"), ORANGE),
        "footer": {"text": "National Weather Service via Show Me Fire"},
        "timestamp": _iso(_parse_time(p.get("sent") or p.get("updated_at"))),
        "fields": [f for f in fields if f["value"]],
    }
    if description:
        embed["description"] = _clip(description, 2000)
    if str(p.get("url") or "").startswith("http"):
        embed["url"] = p["url"]
    return embed


def _staff_alert_embed(p: Dict) -> Dict:
    embed = {
        "title": _clip(p.get("title") or "Staff alert", 256),
        "color": STAFF_COLORS.get(p.get("alert_type"), GRAY),
        "footer": {"text": f"Staff alert · {_clip(p.get('alert_label') or p.get('alert_type') or 'general', 100)}"},
        "timestamp": _iso(_parse_time(p.get("created_at"))),
        "fields": [
            {"name": _clip(f.get("name"), 256), "value": _clip(f.get("value"), 1024), "inline": f.get("inline", True) is not False}
            for f in (p.get("fields") or []) if f.get("name") and f.get("value")
        ][:10],
    }
    if p.get("description"):
        embed["description"] = _clip(p["description"], 2000)
    if str(p.get("url") or "").startswith("http"):
        embed["url"] = p["url"]
    return embed


EMBED_BUILDERS = {
    "forecast_ready": _forecast_embed,
    "outlook_published": _outlook_embed,
    "fire_alert": _fire_alert_embed,
    "staff_alert": _staff_alert_embed,
}


def _event_image_url(payload: Dict) -> Optional[str]:
    event_type = payload.get("event_type")
    if event_type == "outlook_published":
        return payload.get("image_png") or payload.get("image_webp")
    if event_type == "forecast_ready":
        return (payload.get("image_urls") or [None])[0] or payload.get("image_url")
    if event_type == "fire_alert":
        return payload.get("image_url")
    return None


def _fetch_image(url: str, *, cache_buster: str, retries: int, timeout_s: float) -> Optional[bytes]:
    sep = "&" if "?" in url else "?"
    fetch_url = f"{url}{sep}ev={cache_buster}"
    for attempt in range(1, max(1, retries) + 1):
        try:
            with request.urlopen(request.Request(fetch_url, headers={"User-Agent": USER_AGENT}), timeout=timeout_s) as resp:
                return resp.read()
        except Exception as exc:
            logger.warning("Discord image fetch attempt %s failed for %s: %s", attempt, fetch_url, exc)
            time.sleep(0.15 * attempt)
    return None


def deliver_event(payload: Dict, settings: Dict) -> bool:
    """Post a notifier event payload directly to Discord. Never raises."""
    event_type = payload.get("event_type")
    builder = EMBED_BUILDERS.get(event_type)
    if not builder:
        logger.warning("Unknown Discord event type for REST delivery: %s", event_type)
        return False
    try:
        channel_id = resolve_channel(
            payload.get("target_channel_id"), payload.get("target_channel_name"),
            default_id=str(settings.get("channel_id") or ""),
            default_name=str(settings.get("channel_name") or ""),
            strict=event_type == "staff_alert",
        )
        embed = builder(payload)
        files: List[Tuple[str, bytes, str]] = []
        image_url = _event_image_url(payload)
        if image_url:
            image = _fetch_image(
                image_url,
                cache_buster=str(payload.get("event_id") or int(time.time())),
                retries=int(settings.get("image_fetch_retries") or 3),
                timeout_s=float(settings.get("image_fetch_timeout_ms") or 5000) / 1000,
            )
            if image:
                ext = "webp" if image_url.lower().split("?")[0].endswith(".webp") else "png"
                filename = f"{event_type.replace('_', '-')}-{int(time.time())}.{ext}"
                files.append((filename, image, f"image/{ext}"))
                embed["image"] = {"url": f"attachment://{filename}"}
            else:
                embed["image"] = {"url": image_url}
        roles = [str(r) for r in (payload.get("mention_role_ids") or []) if str(r).strip()]
        send_message(
            channel_id,
            content=" ".join(f"<@&{r}>" for r in roles),
            embeds=[embed],
            mention_role_ids=roles,
            files=files,
        )
        logger.info("Discord %s posted directly to channel %s", event_type, channel_id)
        return True
    except Exception as exc:
        logger.warning("Direct Discord delivery failed for %s: %s", event_type, exc)
        return False

"""Routes unexpected API problems - crashed requests, failed scheduled jobs,
any logger.error() call anywhere in the app - to the same Discord staff-alert
channel used for burn-ban/feedback/site-feedback, instead of errors only
being visible in server logs nobody is watching in real time.

Two entry points feed this (both wired up in main.py):
  - A global FastAPI exception handler, for unhandled request-cycle errors.
  - DiscordAlertLogHandler, attached to the root logger, which forwards any
    ERROR-level log record app-wide. This is what catches core/scheduler.py's
    ~25 scheduled jobs for free - each already wraps its body in its own
    try/except and calls logger.error(..., exc_info=True) on failure, so no
    job wrapper needs to be touched individually.

Deliberately reuses the existing staff_alert_types gate and dedupe_ttl_ms
setting from the Discord admin page (core.database.get_discord_admin_settings)
rather than inventing new config - a crash loop shouldn't spam the channel
every few seconds just because nothing in this repo previously read that
field.
"""
from __future__ import annotations

import logging
import threading
import time

from services.discord_notifier import notify_staff_alert

logger = logging.getLogger(__name__)

_DEDUPE_LOCK = threading.Lock()
_LAST_SENT: dict[str, float] = {}
_DEFAULT_DEDUPE_SECONDS = 15 * 60

# Never let this module (or the Discord delivery path it calls into) trigger
# itself - a Discord outage must not become an infinite loop of "failed to
# alert about failing to alert".
_EXCLUDED_LOGGER_PREFIXES = (
    "services.discord_notifier",
    "services.discord_rest",
    "services.system_alerts",
)


def _dedupe_seconds() -> float:
    try:
        from core.database import get_discord_admin_settings

        ttl_ms = get_discord_admin_settings().get("dedupe_ttl_ms")
        if ttl_ms:
            return max(60.0, float(ttl_ms) / 1000.0)
    except Exception:
        pass
    return _DEFAULT_DEDUPE_SECONDS


def _should_send(key: str) -> bool:
    now = time.monotonic()
    window = _dedupe_seconds()
    with _DEDUPE_LOCK:
        last = _LAST_SENT.get(key)
        if last is not None and (now - last) < window:
            return False
        _LAST_SENT[key] = now
        # Opportunistic cleanup so this dict doesn't grow unbounded over a
        # long-running process that accumulates many distinct error messages.
        stale = [k for k, sent_at in _LAST_SENT.items() if (now - sent_at) > window * 4]
        for stale_key in stale:
            _LAST_SENT.pop(stale_key, None)
    return True


def notify_system_issue(source: str, message: str, *, detail: str = "") -> bool:
    """Alert staff that something broke. Deduplicated per (source, message)
    within the configured dedupe window; never raises."""
    key = f"{source}:{message}"[:300]
    if not _should_send(key):
        return False
    try:
        return notify_staff_alert(
            alert_type="system_issue",
            title=f"API issue: {source}"[:256],
            description=(detail or message)[:2000],
            fields=[
                {"name": "Source", "value": source[:256] or "unknown", "inline": True},
                {"name": "Message", "value": message[:1024] or "(no message)", "inline": False},
            ],
            admin_path="/admin",
        )
    except Exception:
        logger.warning("Failed to send system issue alert for %s", source, exc_info=True)
        return False


class DiscordAlertLogHandler(logging.Handler):
    """logging.Handler that forwards ERROR+ records app-wide to Discord.

    Attach to the root logger so it catches core/scheduler.py's existing
    per-job logger.error(..., exc_info=True) calls, plus any other
    logger.error anywhere in the app, without touching each call site.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_EXCLUDED_LOGGER_PREFIXES):
            return
        try:
            message = record.getMessage()
            detail = self.format(record)
        except Exception:
            return
        notify_system_issue(source=record.name, message=message, detail=detail)


_handler_installed = False


def install_discord_log_handler() -> None:
    """Attach DiscordAlertLogHandler to the root logger, once per process."""
    global _handler_installed
    if _handler_installed:
        return
    logging.getLogger().addHandler(DiscordAlertLogHandler())
    _handler_installed = True

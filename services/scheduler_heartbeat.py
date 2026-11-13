"""Heartbeat of the heavy scheduler worker (`python -m core.scheduler_worker`), and the API-side watch on it.

The worker writes `data/current_conditions/status/scheduler_heavy.json` every minute (groups, job ids with their next run
time, jobs running right now). Everything that needs to know about the worker reads that file instead of an in-process
scheduler object, because in a split deployment the API's own scheduler no longer knows the heavy jobs:

  * `/health/ready` and the admin Logs / model-schedule pages (`read_heartbeat`),
  * `run_watch`, a light job in the API group, which alerts staff (Discord + staff push) when the heartbeat stops,
    reminds every 6 hours, and says when it recovers. It runs in the API on purpose, so it still fires when the worker
    is dead. It does nothing in an unsplit deployment (the API runs the heavy group itself) or before a worker ever ran.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.product_health import PROJECT_DIR, STATUS_DIR, Product, plan_alerts

logger = logging.getLogger(__name__)

HEARTBEAT_PATH = STATUS_DIR / "scheduler_heavy.json"
STATE_PATH = PROJECT_DIR / "data" / "current_conditions" / "scheduler_worker_state.json"
STALE_AFTER_MINUTES = 5


def write_heartbeat(payload: dict[str, Any], path: Path | None = None) -> None:
    path = path or HEARTBEAT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
    os.replace(temporary, path)


def read_heartbeat(path: Path | None = None) -> dict[str, Any] | None:
    try:
        return json.loads((path or HEARTBEAT_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def heartbeat_age_minutes(now: datetime | None = None, path: Path | None = None) -> float | None:
    """Minutes since the worker last wrote its heartbeat, or None when it never has."""
    try:
        mtime = (path or HEARTBEAT_PATH).stat().st_mtime
    except OSError:
        return None
    return ((now or datetime.now(timezone.utc)).timestamp() - mtime) / 60


def worker_status(now: datetime | None = None) -> dict[str, Any] | None:
    """The worker's heartbeat when this process hands the heavy group to it and the heartbeat is fresh, else None."""
    if not _split_here():
        return None
    age = heartbeat_age_minutes(now)
    if age is None or age > STALE_AFTER_MINUTES:
        return None
    return read_heartbeat()


def products() -> list[Product]:
    return [Product("scheduler_heavy", "Heavy scheduler worker", [os.path.relpath(HEARTBEAT_PATH, PROJECT_DIR)],
                    STALE_AFTER_MINUTES, "show-me-fire-scheduler")]


def _split_here() -> bool:
    """True when this process does NOT run the heavy group (so the worker is a separate container)."""
    from core.job_groups import HEAVY, groups_from_env

    return HEAVY not in groups_from_env()


def _send(message: str, detail: str = "") -> bool:
    sent = False
    try:
        from services.system_alerts import notify_system_issue

        sent = bool(notify_system_issue("scheduler-worker", message, detail=detail))
    except Exception:  # noqa: BLE001
        logger.warning("Scheduler worker Discord alert failed", exc_info=True)
    try:
        from services import staff_push

        staff_push.dispatch({"alert_type": "feed_stale", "title": f"Scheduler: {message}"})
    except Exception:  # noqa: BLE001
        logger.warning("Scheduler worker staff push failed", exc_info=True)
    return sent


def run_watch(now: datetime | None = None) -> list[tuple[str, str]]:
    """Alert when the heavy worker's heartbeat goes stale / recovers. Returns the alerts sent."""
    now = now or datetime.now(timezone.utc)
    if not _split_here():
        return []
    age = heartbeat_age_minutes(now)
    if age is None:
        return []   # no worker has ever run: nothing to watch yet
    problems = [f"no heartbeat for {age:.0f} min (expected every minute)"] if age > STALE_AFTER_MINUTES else []
    report = {"products": [{"key": "scheduler_heavy", "label": "Heavy scheduler worker", "source": "show-me-fire-scheduler",
                            "status": "stale" if problems else "ok", "problems": problems}]}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    alerts, new_state = plan_alerts(report, state, now)
    for message, detail in alerts:
        _send(message, detail + " Forecast jobs are not running." if problems else detail)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return alerts

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

HEARTBEAT_PATH = STATUS_DIR / "scheduler_heavy.json"   # the heavy worker; the forecast worker writes next to it
FORECAST_HEARTBEAT_NAME = "scheduler_forecast.json"
LABELS = {"heavy": ("scheduler_heavy", "Heavy scheduler worker", "show-me-fire-scheduler"),
          "forecast": ("scheduler_forecast", "Forecast scheduler worker", "show-me-fire-forecast")}
STATE_PATH = PROJECT_DIR / "data" / "current_conditions" / "scheduler_worker_state.json"
STALE_AFTER_MINUTES = 5


def forecast_heartbeat_path() -> Path:
    return HEARTBEAT_PATH.with_name(FORECAST_HEARTBEAT_NAME)


def heartbeat_path_for(groups) -> Path:
    """Where a worker running `groups` writes its heartbeat: the heavy file when it runs the heavy group (a combined worker
    `heavy,forecast` included, which keeps working with the old single-worker setup), else the forecast file."""
    return HEARTBEAT_PATH if "heavy" in groups or "forecast" not in groups else forecast_heartbeat_path()


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


def _claims(group: str) -> list[tuple[float, dict[str, Any]]]:
    """(age in minutes, payload) of every heartbeat file whose worker runs `group`, freshest first."""
    out = []
    for native, path in (("heavy", HEARTBEAT_PATH), ("forecast", forecast_heartbeat_path())):
        payload = read_heartbeat(path)
        age = heartbeat_age_minutes(path=path)
        # A beat that names no groups is taken to run the group its file is named for.
        if payload is not None and age is not None and group in (payload.get("groups") or [native]):
            out.append((age, payload))
    return sorted(out, key=lambda item: item[0])


def group_age_minutes(group: str) -> float | None:
    """Minutes since the worker that runs `group` last reported, or None when none ever has."""
    claims = _claims(group)
    return claims[0][0] if claims else None


def worker_status(now: datetime | None = None, group: str = "heavy") -> dict[str, Any] | None:
    """The heartbeat of the worker running `group` when this process hands that group to it and the beat is fresh."""
    if not _split_here(group):
        return None
    claims = _claims(group)
    if not claims or claims[0][0] > STALE_AFTER_MINUTES:
        return None
    return claims[0][1]


def merged_jobs() -> list[dict[str, Any]]:
    """Every worker's job list (id + next run), for pages that show the schedule of jobs that run elsewhere."""
    jobs: list[dict[str, Any]] = []
    for path in (HEARTBEAT_PATH, forecast_heartbeat_path()):
        jobs += (read_heartbeat(path) or {}).get("jobs") or []
    return jobs


def products() -> list[Product]:
    out = []
    for group, (key, label, source) in LABELS.items():
        path = HEARTBEAT_PATH if group == "heavy" else forecast_heartbeat_path()
        if heartbeat_age_minutes(path=path) is not None:
            out.append(Product(key, label, [os.path.relpath(path, PROJECT_DIR)], STALE_AFTER_MINUTES, source))
    return out


def _split_here(group: str = "heavy") -> bool:
    """True when this process does NOT run `group` (so a worker container is expected to)."""
    from core.job_groups import groups_from_env

    return group not in groups_from_env()


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
    """Alert when a worker's heartbeat goes stale / recovers. Returns the alerts sent."""
    now = now or datetime.now(timezone.utc)
    rows = []
    for group, (key, label, source) in LABELS.items():
        if not _split_here(group):
            continue
        age = group_age_minutes(group)
        if age is None:
            continue   # no worker has ever run this group: nothing to watch yet
        problems = [f"no heartbeat for {age:.0f} min (expected every minute)"] if age > STALE_AFTER_MINUTES else []
        rows.append({"key": key, "label": label, "source": source, "status": "stale" if problems else "ok", "problems": problems})
    if not rows:
        return []
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    alerts, new_state = plan_alerts({"products": rows}, state, now)
    for message, detail in alerts:
        stale = any(row["problems"] for row in rows if row["label"] in message)
        _send(message, detail + " Its jobs are not running." if stale else detail)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    return alerts

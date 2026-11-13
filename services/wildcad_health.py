"""Staleness and problem alerts for the WildCAD feed (services/wildcad.py).

* Heartbeat files are written only after a *successful* fetch, so "stale" means "no good data", not "no attempt".
* Each feed counts as stale after 20 minutes (four missed 5-minute cycles); this is judged by `services/product_health`
  (so it also shows in GET /health/products) and alerted by `run_health_check`, run every 5 minutes by the scheduler.
* An alert goes out when a feed turns stale, again every 6 hours while it stays stale, and once when it recovers.
  Blocks (HTTP 403/429) and unreadable payloads alert immediately through `alert_problem`, once per outage.
* Alerts go to the staff Discord channel (as a system issue) and, when STAFF_PUSH_ENABLED is on, to staff phones.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from services.product_health import PROJECT_DIR, STATUS_DIR, Product, check_all, plan_alerts

logger = logging.getLogger(__name__)

STALE_AFTER_MINUTES = 20
STATE_PATH = PROJECT_DIR / "data" / "current_conditions" / "wildcad_health_state.json"
PROBLEM_REMINDER_HOURS = 6
LABELS = {"incidents": "WildCAD incidents", "resources": "WildCAD resources"}


def heartbeat_path(kind: str) -> Path:
    return STATUS_DIR / f"wildcad_{kind}.json"


def write_heartbeat(kind: str, now: datetime) -> None:
    path = heartbeat_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "last_success": now.astimezone(timezone.utc).isoformat()}), encoding="utf-8")


def products() -> list[Product]:
    return [
        Product(f"wildcad_{kind}", label, [os.path.relpath(heartbeat_path(kind), PROJECT_DIR)], STALE_AFTER_MINUTES,
                "MOMOC WildCAD-E")
        for kind, label in LABELS.items()
    ]


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def send(message: str, detail: str = "") -> bool:
    """Discord (system issue) plus a staff push. Neither can raise."""
    sent = False
    try:
        from services.system_alerts import notify_system_issue

        sent = bool(notify_system_issue("wildcad", message, detail=detail))
    except Exception:  # noqa: BLE001
        logger.warning("WildCAD Discord alert failed", exc_info=True)
    try:
        from services import staff_push

        staff_push.dispatch({"alert_type": "feed_stale", "title": f"WildCAD: {message}"})
    except Exception:  # noqa: BLE001
        logger.warning("WildCAD staff push failed", exc_info=True)
    return sent


def run_health_check(now: datetime | None = None) -> list[tuple[str, str]]:
    """Alert on stale/recovered WildCAD feeds. Returns the alerts it sent. Does nothing while the feed is switched off."""
    from services import wildcad

    now = now or datetime.now(timezone.utc)
    if not wildcad.enabled():
        return []
    report = check_all(now)
    report["products"] = [p for p in report["products"] if p["key"].startswith("wildcad_")]
    state = _load(STATE_PATH)
    mine = {key: value for key, value in state.items() if key.startswith("wildcad_")}
    alerts, new_mine = plan_alerts(report, mine, now)
    for message, detail in alerts:
        send(message, detail)
    state = {key: value for key, value in state.items() if not key.startswith("wildcad_")}
    state.update(new_mine)
    _save(STATE_PATH, state)
    return alerts


def alert_problem(kind: str, message: str, *, blocked: bool = False) -> bool:
    """Immediate alert for a block or an unreadable payload, once per outage (and every 6 h while it lasts)."""
    now = datetime.now(timezone.utc)
    state = _load(STATE_PATH)
    key = f"problem:{kind}:{'blocked' if blocked else 'format'}"
    last = state.get(key)
    if last:
        try:
            if (now - datetime.fromisoformat(last)).total_seconds() < PROBLEM_REMINDER_HOURS * 3600:
                return False
        except ValueError:
            pass
    state[key] = now.isoformat()
    _save(STATE_PATH, state)
    return send(message)


def clear_problem(kind: str) -> None:
    """Forget a past block/format problem after a good fetch, so the next one alerts again."""
    state = _load(STATE_PATH)
    changed = False
    for key in [k for k in state if k.startswith(f"problem:{kind}:")]:
        del state[key]
        changed = True
    if changed:
        _save(STATE_PATH, state)

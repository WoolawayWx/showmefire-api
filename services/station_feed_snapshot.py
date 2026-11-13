"""Share the live station feeds with other containers.

The API keeps the Synoptic station feed and the RAWS list in module-level variables (`services.synoptic.station_data`,
`core.scheduler.raws_station_data`) and serves them from memory. forecast_v1, the RTMA analysis and the spread-rate
pipeline need those stations too, and they run in the heavy scheduler worker, a different container where those variables
are always empty. So the fetchers write each successful update to a small JSON file on the shared volume and the heavy
jobs read the feed through `station_data_for_job()` / `raws_data_for_job()`: live in-memory data when there is any (the
API, an unsplit deployment, tests), otherwise the snapshot, but never one older than MAX_AGE (a job would rather skip a
tick than forecast from stale observations).

Writes are atomic (temp file + os.replace), best-effort (a full disk must not break a station refresh), and skipped under
pytest unless SMF_STATION_SNAPSHOT=1.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.logging_setup import running_tests

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_DIR / "data" / "current_conditions" / "status"
FILES = {"stations": "station_feed.json", "raws": "raws_feed.json"}
MAX_AGE = timedelta(minutes=30)


def _enabled() -> bool:
    flag = os.getenv("SMF_STATION_SNAPSHOT", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return flag in {"1", "true", "yes", "on"} or not running_tests()


def _path(kind: str) -> Path:
    return SNAPSHOT_DIR / FILES[kind]


def write_snapshot(kind: str, data: dict[str, Any], now: datetime | None = None) -> bool:
    """Persist one feed ("stations" | "raws"). Returns False when skipped or on any error (never raises)."""
    stations = data.get("stations")
    if not _enabled() or not stations:
        return False
    try:
        path = _path(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": (now or datetime.now(timezone.utc)).isoformat(),
            "last_updated": data.get("last_updated"),
            "stations": stations,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
        os.replace(temporary, path)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Could not write the %s feed snapshot", kind, exc_info=True)
        return False


def read_snapshot(kind: str, now: datetime | None = None) -> dict[str, Any] | None:
    """The snapshot, or None when missing, unreadable or older than MAX_AGE."""
    try:
        payload = json.loads(_path(kind).read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(str(payload["fetched_at"]))
    except (OSError, ValueError, KeyError):
        return None
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    if (now or datetime.now(timezone.utc)) - fetched > MAX_AGE:
        return None
    return payload


def station_data_for_job(now: datetime | None = None) -> dict[str, Any]:
    """The Synoptic station feed in the shape `services.synoptic.get_station_data()` returns."""
    from services.synoptic import get_station_data

    live = get_station_data()
    if live.get("stations"):
        return live
    snapshot = read_snapshot("stations", now)
    if snapshot:
        return {"stations": snapshot["stations"], "count": len(snapshot["stations"]),
                "last_updated": snapshot.get("last_updated"), "error": None, "from_snapshot": True}
    return live


def raws_data_for_job(now: datetime | None = None) -> dict[str, Any]:
    """The RAWS list in the shape of `core.scheduler.raws_station_data`."""
    from core import scheduler

    live = scheduler.raws_station_data
    if live.get("stations"):
        return live
    snapshot = read_snapshot("raws", now)
    if snapshot:
        return {"stations": snapshot["stations"], "last_updated": snapshot.get("last_updated"), "error": None,
                "from_snapshot": True}
    return live

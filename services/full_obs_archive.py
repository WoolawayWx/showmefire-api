"""Full-day, all-network Synoptic observation capture for the day archive.

The nightly verification pull (scripts/endOfDay.py) deliberately stays RAWS-only
and 10:00-21:00 Central. This captures the whole Central day for every network
the live maps use (NWS/FAA 1, RAWS 2, Missouri Mesonet 156, CWOP 65), so a day
can be reconstructed. One JSON file per Central day, written atomically.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

CENTRAL = ZoneInfo("America/Chicago")
DEFAULT_NETWORKS = "1,2,156,65"
DEFAULT_STATES = "MO"


def archive_dir() -> Path:
    root = Path(os.getenv("SMF_ARCHIVE_ROOT") or ("/app" if Path("/app").exists() else Path(__file__).resolve().parents[1]))
    return root / "archive" / "raw_data_full"


def output_path(day: str, directory: Path | None = None) -> Path:
    return (directory or archive_dir()) / f"raw_data_full_{day.replace('-', '')}.json"


def central_day_bounds_utc(day: str) -> tuple[datetime, datetime]:
    """Naive-UTC [start, end) of a Central calendar day (what the Synoptic helper expects)."""
    local_day = date.fromisoformat(day)
    start = datetime.combine(local_day, time.min, tzinfo=CENTRAL)
    end = datetime.combine(local_day + timedelta(days=1), time.min, tzinfo=CENTRAL)
    return (start.astimezone(timezone.utc).replace(tzinfo=None),
            end.astimezone(timezone.utc).replace(tzinfo=None))


def capture_full_observations(day: str, directory: Path | None = None, fetch=None) -> Path | None:
    """Fetch and save one completed Central day. Returns the file path, or None if nothing was saved."""
    target = output_path(day, directory)
    if target.exists():
        return target
    if fetch is None:
        from services.synoptic import fetch_historical_station_data as fetch
    start, end = central_day_bounds_utc(day)
    networks = [int(n) for n in os.getenv("SMF_FULL_OBS_NETWORKS", DEFAULT_NETWORKS).split(",") if n.strip()]
    states = [s.strip() for s in os.getenv("SMF_FULL_OBS_STATES", DEFAULT_STATES).split(",") if s.strip()]
    data = asyncio.run(fetch(states=states, networks=networks, start_time=start, end_time=end))
    if not data or not data.get("STATION"):
        logger.warning("Full observation capture for %s returned no stations", day)
        return None
    data["_archive"] = {"day_central": day, "utc_start": start.isoformat() + "Z", "utc_end": end.isoformat() + "Z",
                        "networks": networks, "states": states, "stations": len(data["STATION"])}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(target)
    logger.info("Full observations for %s: %d stations -> %s", day, len(data["STATION"]), target)
    return target


async def run_scheduled() -> None:
    """APScheduler entry point: capture yesterday (Central) once it has ended."""
    try:
        yesterday = (datetime.now(CENTRAL).date() - timedelta(days=1)).isoformat()
        await asyncio.to_thread(capture_full_observations, yesterday)
    except Exception as exc:  # noqa: BLE001 - never break the scheduler
        logger.error("Full observation capture failed: %s", exc, exc_info=True)

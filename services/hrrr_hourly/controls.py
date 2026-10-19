"""Admin controls for the hourly HRRR product: which cycles are built, a pause switch, and a short build history.

Both live as small JSON files beside the data (`<root>/settings.json`, `<root>/history.jsonl`, on the volume the API and
the worker share). Every scheduler tick runs `scripts/hrrr_hourly_forecast.py --auto` as a fresh process, which reads
the settings from disk, so a change made on the admin page takes effect on the next tick with no restart.

Precedence for the cycle hours: the settings file, then SMF_HRRR_HOURLY_CYCLE_HOURS, then all 24.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.jsonl"
HISTORY_KEEP = 500          # lines kept after a trim
HISTORY_TRIM_AT = 600       # trim when the file grows past this many lines


def _root(root: Optional[Path]) -> Path:
    from services.hrrr_hourly import root_dir

    return Path(root or root_dir())


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_settings() -> dict[str, Any]:
    return {"cycle_hours": None, "paused": False, "updated_at": None, "updated_by": None}


def load_settings(root: Optional[Path] = None) -> dict[str, Any]:
    """The saved settings, or the defaults when there is no usable file. Never raises."""
    settings = default_settings()
    try:
        saved = json.loads((_root(root) / SETTINGS_FILE).read_text())
    except (OSError, ValueError):
        return settings
    if not isinstance(saved, dict):
        return settings
    hours = saved.get("cycle_hours")
    if isinstance(hours, list):
        valid = sorted({h for h in hours if isinstance(h, int) and not isinstance(h, bool) and 0 <= h <= 23})
        settings["cycle_hours"] = valid or None
    settings["paused"] = bool(saved.get("paused", False))
    settings["updated_at"] = saved.get("updated_at")
    settings["updated_by"] = saved.get("updated_by")
    return settings


def validate_cycle_hours(hours: Optional[Iterable[int]]) -> Optional[list[int]]:
    """None (use the default) or a sorted list of distinct UTC hours 0-23. Raises ValueError for anything else."""
    if hours is None:
        return None
    chosen = []
    for hour in hours:
        if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
            raise ValueError(f"{hour!r} is not a UTC hour from 0 to 23")
        chosen.append(hour)
    if not chosen:
        raise ValueError("pick at least one hour, or pause the product instead")
    return sorted(set(chosen))


def save_settings(cycle_hours: Optional[Iterable[int]], paused: bool, updated_by: str,
                  root: Optional[Path] = None) -> dict[str, Any]:
    """Validate and write the settings atomically. `cycle_hours=None` clears the override (env or all 24 applies)."""
    settings = {"cycle_hours": validate_cycle_hours(cycle_hours), "paused": bool(paused),
                "updated_at": _now(), "updated_by": updated_by}
    path = _root(root) / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(settings, indent=1))
    os.replace(temporary, path)
    logger.info("hrrr_hourly settings saved by %s: cycle_hours=%s paused=%s", updated_by, settings["cycle_hours"],
                settings["paused"])
    return settings


def cycle_hours_source(root: Optional[Path] = None) -> str:
    """Where the effective cycle hours come from: "settings", "env" or "default"."""
    if load_settings(root)["cycle_hours"]:
        return "settings"
    if any(part.strip().isdigit() for part in os.environ.get("SMF_HRRR_HOURLY_CYCLE_HOURS", "").split(",")):
        return "env"
    return "default"


def is_paused(root: Optional[Path] = None) -> bool:
    return bool(load_settings(root)["paused"])


# --- build history -------------------------------------------------------------------------------------------------

def record_event(event: dict[str, Any], root: Optional[Path] = None) -> None:
    """Append one build outcome (cycle, outcome, runtime, error...). Best effort: never raises."""
    try:
        path = _root(root) / HISTORY_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": _now(), **event}) + "\n")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > HISTORY_TRIM_AT:
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text("\n".join(lines[-HISTORY_KEEP:]) + "\n", encoding="utf-8")
            os.replace(temporary, path)
    except OSError:
        logger.exception("hrrr_hourly: could not record a history event")


def recent_events(limit: int = 50, root: Optional[Path] = None) -> list[dict[str, Any]]:
    """The newest `limit` history events, newest first."""
    try:
        lines = (_root(root) / HISTORY_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in reversed(lines):
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
        if len(events) >= limit:
            break
    return events

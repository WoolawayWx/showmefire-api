"""Hourly bookkeeping for the fire-weather maps: which valid hours are
done (so each hour is processed exactly once), which missed hours to backfill,
and the per-hour archive used to track conditions over time."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from services.fire_weather.hrrr_profiles import hour_key
from services.fire_weather.summary import PROJECT_DIR, atomic_write_json

ROOT = PROJECT_DIR / "data" / "fire_weather"
LEDGER_PATH = ROOT / "processed.json"
HISTORY_DIR = ROOT / "history"
GRID_DIR = ROOT / "grids"
LEDGER_DAYS = 10
GRID_RETENTION_DAYS = 30
HISTORY_RETENTION_DAYS = 400
MAX_HOURS_PER_RUN = 3

# Fields kept in the hourly gridded archive (int16, value = stored * scale).
GRID_FIELDS = {
    "mixing_height_ft": 1.0,
    "holzworth_ft": 1.0,
    "transport_speed_mph": 0.1,
    "transport_dir_deg": 1.0,
    "ventilation_kt_ft": 10.0,
    "inversion_strength_f": 0.1,
    "lapse_rate_c_km": 0.1,
    "haines": 1.0,
    "hdw": 0.1,
}


def load_ledger(path: Optional[Path] = None) -> set[str]:
    try:
        return set(json.loads((path or LEDGER_PATH).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def mark_processed(key: str, path: Optional[Path] = None, now: Optional[datetime] = None) -> None:
    path = path or LEDGER_PATH
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=LEDGER_DAYS)).strftime("%Y%m%d%H")
    keys = {item for item in load_ledger(path) if item >= cutoff} | {key}
    atomic_write_json(path, sorted(keys))


def select_hours(available: Iterable[datetime], processed: set[str], budget: int = MAX_HOURS_PER_RUN) -> list[datetime]:
    """Valid hours to process this run, oldest first: always the newest unprocessed
    hour (so the maps stay current), plus the oldest missed ones up to `budget`."""
    missing = sorted(valid for valid in available if hour_key(valid) not in processed)
    if len(missing) <= budget:
        return missing
    return missing[: budget - 1] + missing[-1:]


def _record_path(valid: datetime) -> Path:
    return HISTORY_DIR / f"{valid:%Y}" / f"{valid:%m}" / f"{valid:%d}" / f"{valid:%H}z.json"


def write_record(valid: datetime, summary: dict[str, Any], stations: dict[str, Any]) -> Path:
    path = _record_path(valid)
    atomic_write_json(path, {**summary, "stations": stations})
    return path


def write_grids(valid: datetime, fields: dict[str, np.ndarray]) -> Path:
    """Compressed, int16-packed hourly grids (~1-2 MB/hour)."""
    import xarray as xr

    data_vars = {}
    for name, scale in GRID_FIELDS.items():
        values = fields[name]
        packed = np.where(np.isfinite(values), np.round(values / scale), -32768).clip(-32768, 32767).astype("int16")
        data_vars[name] = (("y", "x"), packed, {"scale_factor": scale, "_FillValue": -32768})
    dataset = xr.Dataset(
        data_vars,
        coords={"latitude": (("y", "x"), fields["lat"].astype("float32")),
                "longitude": (("y", "x"), fields["lon"].astype("float32"))},
        attrs={"valid_time": valid.strftime("%Y-%m-%dT%H:00:00Z"), "source": "NOAA HRRR 1-hour forecast (f01)"},
    )
    path = GRID_DIR / f"{valid:%Y%m%d}" / f"{valid:%H}z.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in GRID_FIELDS}
    temporary = path.with_suffix(".nc.tmp")
    dataset.to_netcdf(temporary, encoding=encoding)
    temporary.replace(path)
    return path


def read_records(hours: int = 48, now: Optional[datetime] = None) -> list[dict[str, Any]]:
    """Hourly records from the last `hours` hours, oldest first (missing hours skipped)."""
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    records = []
    for back in range(hours, -1, -1):
        try:
            records.append(json.loads(_record_path(now - timedelta(hours=back)).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


def prune(now: Optional[datetime] = None) -> None:
    """Drop gridded archives older than GRID_RETENTION_DAYS and records older than HISTORY_RETENTION_DAYS."""
    import shutil

    now = now or datetime.now(timezone.utc)
    grid_cutoff = (now - timedelta(days=GRID_RETENTION_DAYS)).strftime("%Y%m%d")
    if GRID_DIR.is_dir():
        for folder in GRID_DIR.iterdir():
            if folder.is_dir() and folder.name < grid_cutoff:
                shutil.rmtree(folder, ignore_errors=True)
    history_cutoff = now - timedelta(days=HISTORY_RETENTION_DAYS)
    if HISTORY_DIR.is_dir():
        for day in HISTORY_DIR.glob("*/*/*"):
            try:
                stamp = datetime.strptime("".join(day.parts[-3:]), "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if stamp < history_cutoff:
                shutil.rmtree(day, ignore_errors=True)

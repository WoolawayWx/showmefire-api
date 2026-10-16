"""Balloon soundings vs the hourly HRRR fire-weather grids at the same hour and place.

Each archived sounding (maps/sounding_maps.py) inside the HRRR grid is paired,
once, with the nearest grid cell of the hourly archive for its launch hour.
Mixing height is compared two ways because the methods differ: the balloon's
parcel method against HRRR's boundary-layer height (what the maps show), and
against HRRR's own parcel-method value (like for like). Pairs are appended to
data/verification/sounding_vs_hrrr.jsonl; summarize() gives bias (HRRR minus
balloon) and mean absolute error per metric and per site.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from services.fire_weather import history
from services.fire_weather.summary import PROJECT_DIR, atomic_write_json

VERIFY_DIR = PROJECT_DIR / "data" / "verification"
PAIRS_PATH = VERIFY_DIR / "sounding_vs_hrrr.jsonl"
DONE_PATH = VERIFY_DIR / "sounding_vs_hrrr_done.json"
SOUNDING_DIR = PROJECT_DIR / "data" / "soundings"
MAX_DISTANCE_KM = 10.0

# (pair name, balloon fire_weather key, HRRR grid field, units)
METRICS = [
    ("mixing_height_vs_hpbl", "mixing_height_ft", "mixing_height_ft", "ft"),
    ("mixing_height_parcel", "mixing_height_ft", "holzworth_ft", "ft"),
    ("transport_wind", "transport_speed_mph", "transport_speed_mph", "mph"),
    ("inversion_strength", "inversion_strength_f", "inversion_strength_f", "F"),
    ("lapse_rate_0_1km", "lapse_rate_c_km", "lapse_rate_c_km", "C/km"),
    ("haines", "haines", "haines", "index"),
    ("hdw", "hdw", "hdw", "hPa m/s"),
]


def _grid_values(valid: datetime, lat: float, lon: float) -> Optional[dict[str, float]]:
    import xarray as xr

    hour = (valid + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)  # nearest hour
    path = history.GRID_DIR / f"{hour:%Y%m%d}" / f"{hour:%H}z.nc"
    if not path.is_file():
        return None
    with xr.open_dataset(path) as dataset:
        lats = np.asarray(dataset["latitude"].values, dtype=float)
        lons = np.asarray(dataset["longitude"].values, dtype=float)
        distance = (lats - lat) ** 2 + ((lons - lon) * np.cos(np.radians(lat))) ** 2
        i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)
        if np.sqrt(distance[i, j]) * 111.0 > MAX_DISTANCE_KM:
            return None
        return {name: float(dataset[name].values[i, j]) for name in dataset.data_vars}


def _load_done() -> set[str]:
    try:
        return set(json.loads(DONE_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def verify_new(stations: dict[str, tuple], now: Optional[datetime] = None, days: int = 10) -> int:
    """Pair every not-yet-verified archived sounding from the last `days` days
    whose HRRR hour is archived. Returns the number of new pairs."""
    now = now or datetime.now(timezone.utc)
    done = _load_done()
    added = 0
    for path in sorted(SOUNDING_DIR.glob("*/*/*.json")):
        if path.stem in done:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            valid = datetime.fromisoformat(record["valid_time"])
        except (OSError, ValueError, KeyError):
            continue
        if now - valid > timedelta(days=days) or record["station"] not in stations:
            continue
        _name, lat, lon = stations[record["station"]]
        grid = _grid_values(valid, lat, lon)
        if grid is None:
            if now - valid > timedelta(hours=6):
                done.add(path.stem)  # the HRRR hour was never archived (or the site is off the grid)
            continue
        pair = {"station": record["station"], "valid_time": record["valid_time"], "special": record.get("special")}
        for name, balloon_key, grid_key, _units in METRICS:
            balloon = record["fire_weather"].get(balloon_key)
            model = grid.get(grid_key)
            if balloon is not None and model is not None and np.isfinite(model):
                pair[name] = {"balloon": round(float(balloon), 2), "hrrr": round(float(model), 2)}
        VERIFY_DIR.mkdir(parents=True, exist_ok=True)
        with PAIRS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(pair) + "\n")
        done.add(path.stem)
        added += 1
    atomic_write_json(DONE_PATH, sorted(done))
    return added


def _stats(pairs: list[dict]) -> dict[str, Any]:
    if not pairs:
        return {"count": 0}
    diff = np.array([p["hrrr"] - p["balloon"] for p in pairs], dtype=float)
    return {"count": int(diff.size), "bias": round(float(diff.mean()), 2),
            "mae": round(float(np.abs(diff).mean()), 2)}


def summarize(days: int = 30, now: Optional[datetime] = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    rows = []
    try:
        for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if now - datetime.fromisoformat(row["valid_time"]) <= timedelta(days=days):
                rows.append(row)
    except (OSError, ValueError):
        pass
    metrics = {}
    for name, _b, _g, units in METRICS:
        pairs = [row[name] for row in rows if name in row]
        by_station = {}
        for row in rows:
            if name in row:
                by_station.setdefault(row["station"], []).append(row[name])
        metrics[name] = {"units": units, "all": _stats(pairs),
                         "by_station": {station: _stats(items) for station, items in sorted(by_station.items())}}
    return {
        "days": days,
        "soundings": len(rows),
        "note": ("bias = HRRR minus balloon. mixing_height_vs_hpbl compares the balloon parcel method with "
                 "HRRR boundary-layer height (the mapped value); mixing_height_parcel compares like for like."),
        "metrics": metrics,
    }

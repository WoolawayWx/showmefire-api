"""Statewide/regional/station summaries of one hour of fire-weather fields."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ai.briefing import REGION_NAMES, region_for
from services.fire_weather import indices as ix

PROJECT_DIR = Path(__file__).resolve().parents[2]
LATEST_PATH = PROJECT_DIR / "data" / "current_conditions" / "fire_weather_latest.json"

# field -> (label, units, decimals) for the numeric summaries and station values.
METRICS = {
    "mixing_height_ft": ("Mixing height", "ft AGL", 0),
    "transport_speed_mph": ("Transport wind", "mph", 0),
    "ventilation_kt_ft": ("Ventilation rate", "knot-ft", 0),
    "inversion_strength_f": ("Surface inversion strength", "F", 1),
    "inversion_top_ft": ("Surface inversion top", "ft AGL", 0),
    "lapse_rate_c_km": ("0-1 km lapse rate", "C/km", 1),
    "haines": ("Haines Index (low elevation)", "2-6", 0),
    "hdw": ("Hot-Dry-Windy Index", "hPa m/s", 0),
    "holzworth_ft": ("Holzworth mixing height", "ft AGL", 0),
}
INVERSION_THRESHOLD_F = 1.0
SECTORS = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")


def missouri_mask(lat: np.ndarray, lon: np.ndarray, geometry=None) -> np.ndarray:
    import shapely

    if geometry is None:
        from maps import station_plot as sp

        geometry = sp._missouri_geometry()
    if geometry is None:
        return np.ones(lat.shape, dtype=bool)
    return shapely.contains_xy(geometry, lon, lat)


def region_grid(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.vectorize(region_for, otypes=[object])(lat, lon)


def _stats(values: np.ndarray, decimals: int) -> dict[str, Optional[float]]:
    values = values[np.isfinite(values)]
    if not values.size:
        return {"min": None, "median": None, "max": None}
    rounded = lambda value: round(float(value), decimals) if decimals else int(round(float(value)))  # noqa: E731
    return {"min": rounded(values.min()), "median": rounded(np.median(values)), "max": rounded(values.max())}


def _sector(u: np.ndarray, v: np.ndarray) -> Optional[str]:
    if not u.size:
        return None
    direction = float(ix.wind_direction_from(np.nanmean(u), np.nanmean(v)))
    return SECTORS[int(((direction + 22.5) % 360) // 45)]


def _area_summary(fields: dict[str, np.ndarray], cells: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {key: _stats(fields[key][cells], decimals) for key, (_l, _u, decimals) in METRICS.items()}
    categories = fields["ventilation_category"][cells]
    categories = categories[categories >= 0].astype(int)
    total = max(categories.size, 1)
    summary["ventilation_category_percent"] = {
        name: round(100.0 * float((categories == index).sum()) / total, 1)
        for index, (name, _low, _color) in enumerate(ix.VENTILATION_CATEGORIES)
    }
    present = [name for name, share in summary["ventilation_category_percent"].items() if share > 0]
    summary["ventilation_dominant"] = (
        max(summary["ventilation_category_percent"], key=summary["ventilation_category_percent"].get) if present else None
    )
    inversion = fields["inversion_strength_f"][cells]
    summary["surface_inversion_percent"] = round(
        100.0 * float((inversion >= INVERSION_THRESHOLD_F).sum()) / max(np.isfinite(inversion).sum(), 1), 1)
    summary["transport_direction"] = _sector(fields["transport_u_ms"][cells], fields["transport_v_ms"][cells])
    summary["cell_count"] = int(cells.sum())
    return summary


def summarize(fields: dict[str, np.ndarray], valid: datetime, in_state: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Statewide and per-region summaries over the Missouri grid cells."""
    lat, lon = fields["lat"], fields["lon"]
    in_state = missouri_mask(lat, lon) if in_state is None else in_state
    regions = region_grid(lat, lon)
    return {
        "valid_time": valid.strftime("%Y-%m-%dT%H:00:00Z"),
        "source": "NOAA HRRR 1-hour forecast (f01), 3 km",
        "units": {key: units for key, (_label, units, _d) in METRICS.items()},
        "statewide": _area_summary(fields, in_state),
        "regions": {
            REGION_NAMES[code]: _area_summary(fields, in_state & (regions == code))
            for code in REGION_NAMES
        },
    }


def station_values(fields: dict[str, np.ndarray], stations: Iterable[dict]) -> dict[str, dict[str, Any]]:
    """Values at the grid cell nearest each station (Missouri stations only)."""
    from scipy.spatial import cKDTree

    lat, lon = fields["lat"], fields["lon"]
    scale = np.cos(np.radians(38.5))
    tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel() * scale]))
    keys = list(METRICS) + ["transport_dir_deg", "ventilation_category"]
    result = {}
    for station in stations:
        stid, slat, slon = station.get("stid"), station.get("latitude"), station.get("longitude")
        if not stid or slat is None or slon is None or station.get("state") not in (None, "MO"):
            continue
        _distance, index = tree.query([slat, slon * scale])
        row = {}
        for key in keys:
            value = float(fields[key].ravel()[index])
            row[key] = None if not np.isfinite(value) else round(value, 1)
        result[stid] = {"name": station.get("name"), "county": station.get("county"), "lat": slat, "lon": slon, **row}
    return result


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_latest(summary: dict[str, Any], path: Optional[Path] = None) -> None:
    atomic_write_json(path or LATEST_PATH, summary)


def load_latest(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    try:
        return json.loads((path or LATEST_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import numpy as np
from zoneinfo import ZoneInfo

try:
    from core.config import REPORTS_DIR
except ModuleNotFoundError:
    # Allow running map scripts directly (python ./maps/<script>.py)
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    REPORTS_DIR = PROJECT_DIR / "reports"

CHICAGO_TZ = ZoneInfo("America/Chicago")
HISTORY_FILE = REPORTS_DIR / "fire_danger_counts_history.json"
CSV_FILENAME = "fire_danger_counts.csv"

REALTIME_DANGER_LABELS = ["Low", "Moderate", "Elevated", "Critical", "Extreme"]
LEGACY_TO_REALTIME_LABEL = {
    "High": "Elevated",
    "Very High": "Critical",
}
CLASS_VALUE_TO_LABEL = {
    0: "Low",
    1: "Moderate",
    2: "Elevated",
    3: "Critical",
    4: "Extreme",
}


def mph_to_kts(mph_value):
    try:
        return float(mph_value) * 0.868976
    except (TypeError, ValueError):
        return None


def calculate_station_fire_danger_label(fm_value, rh_value, wind_mph_value):
    """Classify station fire danger into Low/Moderate/High/Very High/Extreme."""
    try:
        fm = float(fm_value)
        rh = float(rh_value)
    except (TypeError, ValueError):
        return None

    wind_kts = mph_to_kts(wind_mph_value)
    if wind_kts is None:
        return None

    if fm < 7 and rh < 20 and wind_kts >= 30:
        return "Extreme"
    if fm < 9 and rh < 25 and wind_kts >= 15:
        return "Very High"
    if fm < 9 and (rh < 45 or wind_kts >= 10):
        return "High"
    if 9 <= fm < 15 and (rh < 50 and wind_kts >= 10):
        return "Moderate"
    return "Low"


def _empty_counts() -> Dict[str, int]:
    return {label: 0 for label in REALTIME_DANGER_LABELS}


def normalize_fire_danger_counts(raw_counts: Dict[str, Any] | None) -> Dict[str, int]:
    normalized = _empty_counts()
    if not isinstance(raw_counts, dict):
        return normalized

    for label in REALTIME_DANGER_LABELS:
        normalized[label] += int(raw_counts.get(label, 0) or 0)

    for legacy_label, realtime_label in LEGACY_TO_REALTIME_LABEL.items():
        normalized[realtime_label] += int(raw_counts.get(legacy_label, 0) or 0)

    return normalized


def empty_station_fire_danger_summary(stations: List[Dict[str, Any]], state_filter: str = "MO") -> Dict[str, Any]:
    filtered_stations = [s for s in stations if s.get("state") == state_filter]
    total = len(filtered_stations)
    return {
        "counts": _empty_counts(),
        "total_mo_stations": total,
        "classified_mo_stations": 0,
        "unclassified_mo_stations": total,
    }


def _extract_grid_axes(lon_mesh: Any, lat_mesh: Any) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    lon_arr = np.asarray(lon_mesh, dtype=float)
    lat_arr = np.asarray(lat_mesh, dtype=float)
    if lon_arr.ndim != 2 or lat_arr.ndim != 2:
        return None, None
    if lon_arr.shape != lat_arr.shape:
        return None, None
    return lon_arr[0, :], lat_arr[:, 0]


def _class_to_label(class_value: Any) -> str | None:
    if not np.isfinite(class_value):
        return None
    class_int = int(round(float(class_value)))
    return CLASS_VALUE_TO_LABEL.get(class_int)


def count_station_fire_danger_categories_from_grid(
    stations: List[Dict[str, Any]],
    grid_values: Any,
    lon_mesh: Any,
    lat_mesh: Any,
    state_filter: str = "MO",
) -> Dict[str, Any]:
    summary = empty_station_fire_danger_summary(stations, state_filter=state_filter)
    counts = summary["counts"]

    grid = np.asarray(grid_values, dtype=float)
    lon_axis, lat_axis = _extract_grid_axes(lon_mesh=lon_mesh, lat_mesh=lat_mesh)
    if lon_axis is None or lat_axis is None:
        return summary
    if grid.shape != (lat_axis.shape[0], lon_axis.shape[0]):
        return summary

    lon_min = float(np.nanmin(lon_axis))
    lon_max = float(np.nanmax(lon_axis))
    lat_min = float(np.nanmin(lat_axis))
    lat_max = float(np.nanmax(lat_axis))

    classified = 0
    unclassified = 0
    filtered_stations = [s for s in stations if s.get("state") == state_filter]

    for station in filtered_stations:
        lon = station.get("longitude")
        lat = station.get("latitude")
        if lon is None or lat is None:
            unclassified += 1
            continue

        try:
            lon = float(lon)
            lat = float(lat)
        except (TypeError, ValueError):
            unclassified += 1
            continue

        if lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max:
            unclassified += 1
            continue

        row = int(np.argmin(np.abs(lat_axis - lat)))
        col = int(np.argmin(np.abs(lon_axis - lon)))
        label = _class_to_label(grid[row, col])
        if label is None:
            unclassified += 1
            continue

        counts[label] += 1
        classified += 1

    return {
        "counts": counts,
        "total_mo_stations": len(filtered_stations),
        "classified_mo_stations": classified,
        "unclassified_mo_stations": unclassified,
    }


def count_station_fire_danger_categories(stations):
    # Legacy fallback from station observations (kept for compatibility callers).
    counts = _empty_counts()

    classified = 0
    unclassified = 0
    mo_stations = [s for s in stations if s.get("state") == "MO"]

    for station in mo_stations:
        obs = station.get("observations", {})
        fm = obs.get("fuel_moisture", {}).get("value")
        rh = obs.get("relative_humidity", {}).get("value")
        wind = obs.get("wind_speed", {}).get("value")
        label = calculate_station_fire_danger_label(fm, rh, wind)

        if label is None:
            unclassified += 1
            continue

        if label == "High":
            label = "Elevated"
        elif label == "Very High":
            label = "Critical"

        counts[label] += 1
        classified += 1

    return {
        "counts": counts,
        "total_mo_stations": len(mo_stations),
        "classified_mo_stations": classified,
        "unclassified_mo_stations": unclassified,
    }


def _parse_iso_utc(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _atomic_write_text(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=file_path.parent, encoding="utf-8") as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(file_path)


def load_fire_danger_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []
    return data


def _prune_history_entries(history: List[Dict[str, Any]], run_time_utc: datetime, keep_hours: int) -> List[Dict[str, Any]]:
    cutoff = run_time_utc - timedelta(hours=keep_hours)
    kept: List[Dict[str, Any]] = []
    for entry in history:
        ts = entry.get("timestamp_utc")
        if not ts:
            continue
        try:
            entry_dt = _parse_iso_utc(ts)
        except ValueError:
            continue
        if entry_dt >= cutoff:
            kept.append(entry)

    kept.sort(key=lambda e: e.get("timestamp_utc", ""))
    return kept


def append_fire_danger_snapshot(
    summary: Dict[str, Any],
    run_time_utc: datetime | None = None,
    keep_hours: int = 48,
) -> List[Dict[str, Any]]:
    run_time_utc = run_time_utc or datetime.now(timezone.utc)
    run_time_utc = run_time_utc.astimezone(timezone.utc)
    run_time_local = run_time_utc.astimezone(CHICAGO_TZ)

    counts = normalize_fire_danger_counts(summary.get("counts", {}))
    snapshot = {
        "timestamp_utc": run_time_utc.isoformat().replace("+00:00", "Z"),
        "timestamp_local": run_time_local.isoformat(),
        "date_local": run_time_local.strftime("%Y-%m-%d"),
        "counts": {
            "Low": int(counts.get("Low", 0) or 0),
            "Moderate": int(counts.get("Moderate", 0) or 0),
            "Elevated": int(counts.get("Elevated", 0) or 0),
            "Critical": int(counts.get("Critical", 0) or 0),
            "Extreme": int(counts.get("Extreme", 0) or 0),
        },
        "total_mo_stations": int(summary.get("total_mo_stations", 0) or 0),
        "classified_mo_stations": int(summary.get("classified_mo_stations", 0) or 0),
        "unclassified_mo_stations": int(summary.get("unclassified_mo_stations", 0) or 0),
    }

    history = load_fire_danger_history()
    history = [h for h in history if h.get("timestamp_utc") != snapshot["timestamp_utc"]]
    history.append(snapshot)
    history = _prune_history_entries(history, run_time_utc=run_time_utc, keep_hours=keep_hours)

    _atomic_write_text(HISTORY_FILE, json.dumps(history, indent=2))
    return history


def export_fire_danger_daily_csv(history: List[Dict[str, Any]], target_date: str | None = None) -> Path:
    if target_date is None:
        target_date = datetime.now(CHICAGO_TZ).strftime("%Y-%m-%d")

    rows = [entry for entry in history if entry.get("date_local") == target_date]

    day_dir = REPORTS_DIR / target_date
    csv_path = day_dir / CSV_FILENAME
    day_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "timestamp_local",
        "Low",
        "Moderate",
        "Elevated",
        "Critical",
        "Extreme",
        "total_mo_stations",
        "classified_mo_stations",
        "unclassified_mo_stations",
    ]

    with NamedTemporaryFile("w", delete=False, dir=day_dir, encoding="utf-8", newline="") as tmp_file:
        tmp_path = Path(tmp_file.name)
        writer = csv.DictWriter(tmp_file, fieldnames=fieldnames)
        writer.writeheader()
        for entry in rows:
            counts = normalize_fire_danger_counts(entry.get("counts", {}))
            writer.writerow(
                {
                    "timestamp_utc": entry.get("timestamp_utc"),
                    "timestamp_local": entry.get("timestamp_local"),
                    "Low": counts.get("Low", 0),
                    "Moderate": counts.get("Moderate", 0),
                    "Elevated": counts.get("Elevated", 0),
                    "Critical": counts.get("Critical", 0),
                    "Extreme": counts.get("Extreme", 0),
                    "total_mo_stations": entry.get("total_mo_stations", 0),
                    "classified_mo_stations": entry.get("classified_mo_stations", 0),
                    "unclassified_mo_stations": entry.get("unclassified_mo_stations", 0),
                }
            )

    tmp_path.replace(csv_path)
    return csv_path


def get_recent_fire_danger_history(last_hours: int = 48) -> List[Dict[str, Any]]:
    history = load_fire_danger_history()
    now_utc = datetime.now(timezone.utc)
    return _prune_history_entries(history, run_time_utc=now_utc, keep_hours=last_hours)
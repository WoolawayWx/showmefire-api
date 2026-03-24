import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List
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


def count_station_fire_danger_categories(stations):
    counts = {
        "Low": 0,
        "Moderate": 0,
        "High": 0,
        "Very High": 0,
        "Extreme": 0,
    }

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

    counts = summary.get("counts", {})
    snapshot = {
        "timestamp_utc": run_time_utc.isoformat().replace("+00:00", "Z"),
        "timestamp_local": run_time_local.isoformat(),
        "date_local": run_time_local.strftime("%Y-%m-%d"),
        "counts": {
            "Low": int(counts.get("Low", 0) or 0),
            "Moderate": int(counts.get("Moderate", 0) or 0),
            "High": int(counts.get("High", 0) or 0),
            "Very High": int(counts.get("Very High", 0) or 0),
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
        "High",
        "Very High",
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
            counts = entry.get("counts", {})
            writer.writerow(
                {
                    "timestamp_utc": entry.get("timestamp_utc"),
                    "timestamp_local": entry.get("timestamp_local"),
                    "Low": counts.get("Low", 0),
                    "Moderate": counts.get("Moderate", 0),
                    "High": counts.get("High", 0),
                    "Very High": counts.get("Very High", 0),
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
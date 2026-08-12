"""
Persists a running growing-degree-day (GDD) accumulator since March 1,
used by core.fire_danger.seasonal_dampening_cap to keep dead-fuel-stick
dryness from being reported as full fire danger during green-up, before
real fuel curing has plausibly occurred. See the seasonal fuel-moisture
plan for why: calculate_fire_danger has no live-fuel signal anywhere in
this project, and neither does any RTMA/HRRR/static input feeding it.

GDD-only for now, not KBDI: KBDI needs a verified per-county mean-annual-
precipitation climate normal, and none exists yet anywhere in this repo
(see model-training/risk_fusion/build_county_days.py's own documented
gap). Fabricating one here would repeat exactly the mistake that
module's docstring warns against. seasonal_dampening_cap already accepts
an optional kbdi_mm and will use it once a real normal is sourced -
nothing here blocks that later.

Every public function degrades to "no signal" on any failure rather than
raising - this must never break a scheduler job or a live forecast run.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RAW_ROOT = Path(os.getenv("SMF_RAW_OBSERVATION_ROOT", "archive/raw_data"))
STATE_PATH = Path(os.getenv("SMF_SEASONAL_FUEL_STATE_PATH") or
                  (Path(os.getenv("DATA_DIR", "data")) / "seasonal-fuel-state.json"))
GDD_BASE_TEMP_C = 10.0  # matches model-training/risk_fusion/features.py's GDD_BASE_TEMP_C
SEASON_START_MONTH_DAY = (3, 1)  # matches risk_fusion/features.py's GDD_SEASON_START

_CACHE_TTL_SECONDS = 300
_cache = {"loaded_at": None, "state": None}


def _season_start_for(day: date) -> date:
    month, day_of_month = SEASON_START_MONTH_DAY
    this_year_start = date(day.year, month, day_of_month)
    return this_year_start if day >= this_year_start else date(day.year - 1, month, day_of_month)


def _default_state(as_of: date) -> dict:
    return {
        "season_start": _season_start_for(as_of).isoformat(),
        "gdd_accum_since_mar1": 0.0,
        "last_updated_date": None,
    }


def _load_state() -> dict:
    try:
        if STATE_PATH.exists():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("seasonal_fuel_state: failed to read state, starting fresh: %s", error)
    return _default_state(datetime.now(timezone.utc).date())


def _persist_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = STATE_PATH.with_suffix(".tmp")
        temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp_path.replace(STATE_PATH)
    except Exception as error:
        logger.warning("seasonal_fuel_state: failed to persist state: %s", error)


def daily_mean_temp_c_from_archive(day: date, raw_root: Path = RAW_ROOT) -> Optional[float]:
    """Statewide daily mean air temperature (C) from that date's archived Synoptic raw_data JSON.

    Averages every station's own daily mean air_temp_set_1 (Synoptic
    reports in Fahrenheit, matching the convention already used in
    forecast/DailyForecast.py's own temp handling). Returns None if the
    archive file is missing or has no usable readings - callers must
    treat that as "no signal today", never as 0.
    """
    path = Path(raw_root) / f"raw_data_{day.strftime('%Y%m%d')}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.warning("seasonal_fuel_state: failed to read %s: %s", path, error)
        return None
    stations = payload.get("STATION", [])
    stations = [stations] if isinstance(stations, dict) else stations
    station_means = []
    for station in stations:
        observations = station.get("OBSERVATIONS", {}) if isinstance(station, dict) else {}
        temps_f = observations.get("air_temp_set_1") or observations.get("air_temp_value_1") or []
        valid = [float(value) for value in temps_f if value is not None]
        if valid:
            station_means.append(statistics.mean(valid))
    if not station_means:
        return None
    mean_temp_f = statistics.mean(station_means)
    return (mean_temp_f - 32.0) * 5.0 / 9.0


def update_daily_gdd(day: Optional[date] = None, raw_root: Path = RAW_ROOT) -> dict:
    """Advance the persisted GDD accumulator by one day of already-archived station observations.

    Meant to run once daily, shortly before the end-of-day archive job
    (core/scheduler.py's end_of_day_archive) removes that date's local
    raw_data JSON. Idempotent per calendar day: re-running for a date
    already applied is a no-op rather than double-counting. Resets the
    accumulator to 0 at each new March-1 season boundary.
    """
    day = day or datetime.now(timezone.utc).date()
    state = _load_state()
    season_start = _season_start_for(day)
    if state.get("season_start") != season_start.isoformat():
        state = _default_state(day)
    if state.get("last_updated_date") == day.isoformat():
        return state
    mean_temp_c = daily_mean_temp_c_from_archive(day, raw_root)
    if mean_temp_c is None:
        logger.warning("seasonal_fuel_state: no usable observations for %s; GDD accumulator not advanced", day)
        return state
    daily_gdd = max(0.0, mean_temp_c - GDD_BASE_TEMP_C)
    state["gdd_accum_since_mar1"] = float(state.get("gdd_accum_since_mar1", 0.0)) + daily_gdd
    state["last_updated_date"] = day.isoformat()
    _persist_state(state)
    return state


def current_gdd_accum(max_age_days: int = 3) -> Optional[float]:
    """Cached read of today's accumulated GDD for the hot forecast path.

    Never touches disk more than once per _CACHE_TTL_SECONDS. Returns
    None (no cap should be applied) when the state is missing or stale by
    more than max_age_days, rather than silently serving an outdated value.
    """
    now = datetime.now(timezone.utc)
    loaded_at = _cache["loaded_at"]
    if _cache["state"] is not None and loaded_at is not None and \
            (now - loaded_at).total_seconds() < _CACHE_TTL_SECONDS:
        state = _cache["state"]
    else:
        state = _load_state()
        _cache["state"], _cache["loaded_at"] = state, now
    last_updated = state.get("last_updated_date")
    if not last_updated:
        return None
    try:
        age_days = (now.date() - date.fromisoformat(last_updated)).days
    except Exception:
        return None
    if age_days > max_age_days:
        return None
    return float(state.get("gdd_accum_since_mar1", 0.0))

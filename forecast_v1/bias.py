"""Rolling bias correction from forecast_v1's own station verification.

The hourly Synoptic refresh (observations.update_forecast_verification) stores
forecast-minus-observed errors at stations. A run is corrected by minus the
mean error over the 30 days before its cycle, for the same lead bucket and
three-hour block of local time (the diurnal cycle carries most of the bias),
falling back to the whole lead bucket when a block is thin. The gate is the
one confidence.rolling_verification uses: 30 pairs from 3 stations.

Errors are measured on published, already corrected forecasts, so each run's
applied residuals are stored (forecast_bias_applied) and added back here;
otherwise the correction would cancel itself out after one window.

The same window also gives the spread of the errors around that bias per lead
bucket (error_sigma), which sizes the fire-danger scenarios in the engine.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr

from core.database import get_db_path
from .contracts import CORRECTION_LIMITS, LOCAL_TIMEZONE, correction_multiplier, utc_rfc3339
from .repository import ensure_schema

CORRECTED_VARIABLES = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gust_10m")
SIGMA_VARIABLES = ("relative_humidity_2m", "wind_speed_10m", "wind_gust_10m", "fuel_moisture_p50")
QUERIED_VARIABLES = tuple(dict.fromkeys(CORRECTED_VARIABLES + SIGMA_VARIABLES))
WINDOW_DAYS = 30
MIN_PAIRS = 30
MIN_STATIONS = 3
LOCAL_BLOCK_HOURS = 3


@dataclass(frozen=True)
class BiasCorrection:
    residuals: dict[str, xr.DataArray]
    summary: dict
    error_sigma: dict[str, xr.DataArray]


def lead_bucket(lead_hour: int) -> str:
    return "0-24" if lead_hour <= 24 else "25-48" if lead_hour <= 48 else "49-72"


def _local_block(utc: datetime) -> int:
    return utc.astimezone(ZoneInfo(LOCAL_TIMEZONE)).hour // LOCAL_BLOCK_HOURS


def _error_sums(database: Path, cycle: datetime) -> tuple[dict, dict]:
    """Raw (uncorrected) error sums per (variable, bucket, block) and per (variable, bucket)."""
    window = (utc_rfc3339(cycle - timedelta(days=WINDOW_DAYS)), utc_rfc3339(cycle))
    placeholders = ",".join("?" for _ in QUERIED_VARIABLES)
    # GEFS-only Day 3 (RRFS missing) is a different model; keep its errors out.
    filters = f"""
        FROM forecast_verification v
        LEFT JOIN forecast_runs r ON r.run_id = v.run_id
        LEFT JOIN forecast_bias_applied a
          ON a.run_id = v.run_id AND a.valid_time_utc = v.valid_time_utc AND a.variable = v.variable
        WHERE v.qc_eligible = 1 AND v.error IS NOT NULL
          AND v.valid_time_utc >= ? AND v.valid_time_utc < ?
          AND v.variable IN ({placeholders})
          AND NOT (v.lead_bucket = '49-72' AND COALESCE(r.warnings_json, '') LIKE '%coarse_synoptic_fallback%')"""
    parameters = (*window, *QUERIED_VARIABLES)
    blocks: dict[tuple[str, str, int], list[float]] = {}
    buckets: dict[tuple[str, str], list[float]] = {}
    with sqlite3.connect(database) as connection:
        hourly = connection.execute(
            f"""SELECT v.variable, v.lead_bucket, substr(v.valid_time_utc, 1, 13),
                       SUM(v.error - COALESCE(a.residual, 0)), COUNT(*),
                       SUM((v.error - COALESCE(a.residual, 0)) * (v.error - COALESCE(a.residual, 0)))
                {filters}
                GROUP BY v.variable, v.lead_bucket, substr(v.valid_time_utc, 1, 13)""",
            parameters,
        ).fetchall()
        stations = connection.execute(
            f"""SELECT v.variable, v.lead_bucket, COUNT(DISTINCT v.station_id)
                {filters}
                GROUP BY v.variable, v.lead_bucket""",
            parameters,
        ).fetchall()
    for variable, bucket, valid_hour, total, count, squares in hourly:
        block = _local_block(datetime.fromisoformat(valid_hour + ":00:00+00:00"))
        for key, store in (((variable, bucket, block), blocks), ((variable, bucket), buckets)):
            entry = store.setdefault(key, [0.0, 0, 0, 0.0])
            entry[0] += float(total)
            entry[1] += int(count)
            entry[3] += float(squares)
    for variable, bucket, count in stations:
        buckets.setdefault((variable, bucket), [0.0, 0, 0, 0.0])[2] = int(count)
    return blocks, buckets


def rolling_bias_correction(
    times: xr.DataArray, cycle: datetime, db_path: str | Path | None = None,
) -> BiasCorrection:
    """Per-hour residuals (added to the forecast), clipped to the correction caps."""
    cycle = cycle.astimezone(timezone.utc)
    database = Path(db_path or get_db_path())
    blocks: dict = {}
    buckets: dict = {}
    if database.exists():
        ensure_schema(database)
        blocks, buckets = _error_sums(database, cycle)
    valid_times = [
        datetime.fromisoformat(np.datetime_as_string(value, unit="s")).replace(tzinfo=timezone.utc)
        for value in times.values
    ]
    residuals: dict[str, xr.DataArray] = {}
    summary: dict = {"windowDays": WINDOW_DAYS, "minPairs": MIN_PAIRS, "minStations": MIN_STATIONS, "variables": {}}
    for variable in CORRECTED_VARIABLES:
        values = np.zeros(len(valid_times), dtype=np.float32)
        sources = {"block": 0, "bucket": 0, "none": 0}
        for lead, valid in enumerate(valid_times):
            bucket = lead_bucket(lead)
            total, count, station_count, _ = buckets.get((variable, bucket), (0.0, 0, 0, 0.0))
            if count < MIN_PAIRS or station_count < MIN_STATIONS:
                sources["none"] += 1
                continue
            block_total, block_count, _, _ = blocks.get((variable, bucket, _local_block(valid)), (0.0, 0, 0, 0.0))
            if block_count >= MIN_PAIRS:
                mean_error, source = block_total / block_count, "block"
            else:
                mean_error, source = total / count, "bucket"
            cap = CORRECTION_LIMITS[variable] * correction_multiplier(lead)
            values[lead] = float(np.clip(-mean_error, -cap, cap))
            sources[source] += 1
        summary["variables"][variable] = {
            "applied": bool(np.any(values != 0)),
            "leadHours": sources,
            "maxAbsResidual": round(float(np.max(np.abs(values))), 3),
            "buckets": {
                bucket: {"pairs": entry[1], "stations": entry[2], "meanError": round(entry[0] / entry[1], 3)}
                for (name, bucket), entry in sorted(buckets.items()) if name == variable and entry[1]
            },
        }
        if np.any(values != 0):
            residuals[variable] = xr.DataArray(values, dims="time", coords={"time": times.values})
    summary["applied"] = bool(residuals)
    error_sigma: dict[str, xr.DataArray] = {}
    for variable in SIGMA_VARIABLES:
        values = np.full(len(valid_times), np.nan, dtype=np.float32)
        for lead in range(len(valid_times)):
            total, count, station_count, squares = buckets.get((variable, lead_bucket(lead)), (0.0, 0, 0, 0.0))
            if count >= MIN_PAIRS and station_count >= MIN_STATIONS:
                values[lead] = float(np.sqrt(max(squares / count - (total / count) ** 2, 0.0)))
        if np.isfinite(values).any():
            error_sigma[variable] = xr.DataArray(values, dims="time", coords={"time": times.values})
    summary["errorSigma"] = {
        variable: {
            bucket: round(float(sigma.values[lead]), 3)
            for bucket, lead in (("0-24", 0), ("25-48", 25), ("49-72", 49))
            if lead < sigma.sizes["time"] and np.isfinite(sigma.values[lead])
        }
        for variable, sigma in error_sigma.items()
    }
    return BiasCorrection(residuals, summary, error_sigma)

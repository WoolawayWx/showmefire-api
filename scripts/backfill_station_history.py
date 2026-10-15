#!/usr/bin/env python3
"""Seed maps/station_history.py's database from Synoptic's timeseries archive.

The pressure, wind-gust and rainfall maps are driven by history the 5-minute
recorder (core/scheduler.py record_station_history_job) only started keeping
for these fields. Run this once after deploying so those maps work immediately
instead of after hours/days of accumulation:

    python scripts/backfill_station_history.py [--rain-days 45] [--obs-hours 48]

  * wind gusts + sea-level pressure: last --obs-hours of samples (for the 3-hour
    pressure change and the 1-hour/today/yesterday maximum gusts)
  * rainfall: per-day totals (max of "since local midnight") for --rain-days,
    for the consecutive-dry-days maps

Safe to re-run: samples are INSERT OR IGNORE and daily rows are replaced.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_DIR / ".env")

from maps import station_history as sh  # noqa: E402
from maps import station_plot as sp  # noqa: E402
from services.synoptic import SYNOPTIC_API_TOKEN  # noqa: E402

API = "https://api.synopticdata.com/v2/stations/timeseries"
RAIN_CHUNK_DAYS = 10  # ~13 MB / 15 s per chunk; keeps each response manageable


def _fetch(stids: list[str], start: datetime, end: datetime, variables: str) -> list[dict]:
    response = requests.get(API, params={
        "token": SYNOPTIC_API_TOKEN, "stid": ",".join(stids), "vars": variables,
        "start": start.strftime("%Y%m%d%H%M"), "end": end.strftime("%Y%m%d%H%M"),
        "units": "english", "obtimezone": "utc", "hfmetars": "0",
    }, timeout=240)
    response.raise_for_status()
    body = response.json()
    code = int(body.get("SUMMARY", {}).get("RESPONSE_CODE", 0))
    if code not in (1, 2):
        raise RuntimeError(f"Synoptic error: {body.get('SUMMARY')}")
    return body.get("STATION", [])


def _series(station: dict, name: str) -> tuple[list[str], list]:
    observations = station.get("OBSERVATIONS") or {}
    key = next((k for k in observations if k.startswith(name)), None)
    return (observations.get("date_time") or [], observations.get(key) or []) if key else ([], [])


def _parse(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)


def _accumulate(extreme_rows: dict, stid: str, field: str, when: datetime, value: float) -> None:
    """Fold one sample into the running per-(station, field, local date) min/max."""
    key = (stid, field, when.astimezone(sh.CHICAGO_TZ).strftime("%Y-%m-%d"))
    ts = sh._iso(when)
    current = extreme_rows.get(key)
    if current is None:
        extreme_rows[key] = {"min": (value, ts), "max": (value, ts)}
        return
    if value < current["min"][0]:
        current["min"] = (value, ts)
    if value > current["max"][0]:
        current["max"] = (value, ts)


def backfill_observations(stids: list[str], hours: float) -> int:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    stations = _fetch(stids, start, end, "wind_gust,sea_level_pressure")
    fields = {"wind_gust": "wind_gust_mph", "sea_level_pressure": "sea_level_pressure_mb"}
    retention_cutoff = sh._iso(end - timedelta(hours=sh.DEFAULT_RETENTION_HOURS))
    sample_rows, extreme_rows = [], {}
    for station in stations:
        stid = station["STID"]
        for variable, field in fields.items():
            stamps, values = _series(station, variable)
            for stamp, value in zip(stamps, values):
                if value is None:
                    continue
                when = _parse(stamp)
                ts = sh._iso(when)
                if ts >= retention_cutoff:
                    sample_rows.append((stid, field, ts, float(value)))
                _accumulate(extreme_rows, stid, field, when, float(value))
    _write(sample_rows, extreme_rows)
    return len(sample_rows)


def backfill_rain(stids: list[str], days: int) -> int:
    end = datetime.now(timezone.utc)
    extreme_rows = {}
    chunk_end = end
    remaining = days + 1
    while remaining > 0:
        span = min(RAIN_CHUNK_DAYS, remaining)
        chunk_start = chunk_end - timedelta(days=span)
        for station in _fetch(stids, chunk_start, chunk_end, "precip_accum_since_local_midnight"):
            stamps, values = _series(station, "precip_accum_since_local_midnight")
            for stamp, value in zip(stamps, values):
                if value is None:
                    continue
                _accumulate(extreme_rows, station["STID"], "precip_since_midnight_in", _parse(stamp), float(value))
        chunk_end = chunk_start
        remaining -= span
    _write([], extreme_rows)
    return len(extreme_rows)


def _write(sample_rows: list, extreme_rows: dict) -> None:
    conn = sh._connect()
    try:
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO station_obs (stid, field, ts_utc, value) VALUES (?, ?, ?, ?)", sample_rows,
            )
            conn.executemany(
                "INSERT OR REPLACE INTO station_daily_extreme "
                "(stid, field, date_local, min_value, min_ts_utc, max_value, max_ts_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (stid, field, date_local, e["min"][0], e["min"][1], e["max"][0], e["max"][1])
                    for (stid, field, date_local), e in extreme_rows.items()
                ],
            )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rain-days", type=int, default=45)
    parser.add_argument("--obs-hours", type=float, default=48)
    args = parser.parse_args()
    if not SYNOPTIC_API_TOKEN:
        sys.exit("SYNOPTIC_API_TOKEN is not set")

    stids = sorted({s["stid"] for s in sp.fetch_stations() if s.get("stid")})
    print(f"Backfilling {len(stids)} stations")
    started = time.time()
    print(f"gust/pressure samples written: {backfill_observations(stids, args.obs_hours)} ({time.time() - started:.0f}s)")
    started = time.time()
    print(f"daily rainfall rows written: {backfill_rain(stids, args.rain_days)} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()

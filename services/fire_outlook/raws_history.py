"""Causal RAWS 10-h fuel moisture for an outlook cycle, read from the station history the API records every 5 minutes
(maps/station_history.py: `station_obs` field `fuel_moisture_pct`, `station_meta` for locations).

Why this and not the `observations` table: nothing in production fills `observations` any more, so anchoring on it found
no stations. The history DB is filled all day by `record_station_history_job` and keeps 36 hours, enough for a cycle the
worker processes about 5 hours after it starts.

Causality. A history row's time is when the API recorded it, and the value was a RAWS reading at most 75 minutes old at
that moment (services/raws_calibration.fresh_station_fuel_moisture), so the reading itself is no later than the row time.
Only rows at or before the cycle are used, so nothing after the cycle can leak in. The row time is reported as the
observation time, which makes a reading look slightly younger than it is (by up to 75 minutes) and is why the lookback
below is the conditioning window (3 h) and not a tighter one.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FIELD = "fuel_moisture_pct"


def fuel_moisture_payload(cycle: datetime, max_age_hours: float = 3.0, db_path: Optional[Path] = None) -> dict:
    """RAWS-style payload for conditioning.causal_raws_points: the newest recorded reading per station in
    (cycle - max_age_hours, cycle]. Stations carry state "" (the history has none), so condition with raws_states=None.
    Stations with no recorded location are skipped. An unreadable or missing history DB returns no stations."""
    from maps.station_history import STATION_HISTORY_DB
    path = Path(db_path) if db_path is not None else STATION_HISTORY_DB
    if not path.is_file():
        return {"stations": []}
    cycle = cycle.astimezone(timezone.utc) if cycle.tzinfo else cycle.replace(tzinfo=timezone.utc)
    start = cycle - timedelta(hours=max_age_hours)
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15) as con:
            # Coarse text bounds only narrow the scan; the exact causal filter is applied after parsing below
            # (isoformat drops the fraction when it is zero, so text order is not trustworthy at the second).
            rows = con.execute(
                "SELECT o.stid, o.ts_utc, o.value, m.latitude, m.longitude FROM station_obs o "
                "JOIN station_meta m ON m.stid = o.stid WHERE o.field = ? AND o.ts_utc >= ? AND o.ts_utc <= ?",
                (FIELD, (start - timedelta(minutes=5)).isoformat(), (cycle + timedelta(minutes=5)).isoformat())).fetchall()
    except sqlite3.Error as error:
        logger.warning("station history unreadable for the RAWS anchor (%s): %s", path, error)
        return {"stations": []}
    if not rows:
        return {"stations": []}
    df = pd.DataFrame(rows, columns=["stid", "time", "fm10", "latitude", "longitude"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["time", "fm10"])
    df = df[(df["time"] <= pd.Timestamp(cycle)) & (df["time"] > pd.Timestamp(start))]
    latest = df.sort_values("time").groupby("stid").last()
    return {"stations": [
        {"stid": str(stid), "state": "", "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
         "observations": {"fuel_moisture": {"value": float(r["fm10"]), "time": r["time"].isoformat().replace("+00:00", "Z")}}}
        for stid, r in latest.iterrows()]}

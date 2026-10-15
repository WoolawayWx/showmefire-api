"""Per-station running daily extremes + short-term lookback history.

Generalizes observed_peak_history.py's day-rollover pattern (America/Chicago
local-midnight, keyed by date_local) from a single per-pixel raster to
per-station scalar values, backing the "Today's Max/Min X" and "N-hour X
Change" products in the current-conditions map catalog.

Fed by a dedicated 5-minute scheduler job (record_station_history_job,
wired in core/scheduler.py) that reuses the already-cached get_station_data()
feed from the existing fetch_synoptic job - this module never calls Synoptic
itself, so adding more tracked fields costs nothing in extra external API
calls. Storage is a small dedicated SQLite file (not the main app database;
N stations x M fields x many timestamps is write-heavy enough that it
shouldn't share a database with transactional app data), using
DATA_DIR from core.config like other small app-local datasets
(core.config.MISSOURI_FIRES_JSON etc).
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

try:
    from core.config import DATA_DIR
except ModuleNotFoundError:
    # Allow running map scripts directly (python ./maps/<script>.py)
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_DIR / "data"
    sys.path.append(str(PROJECT_DIR))

CHICAGO_TZ = ZoneInfo("America/Chicago")
STATION_HISTORY_DB = Path(DATA_DIR) / "station_history.sqlite"

DEFAULT_RETENTION_HOURS = 36
# How far from "exactly N hours ago" a sample may be and still count, given
# the 5-minute recording cadence plus any missed/late cycles.
DEFAULT_LOOKUP_TOLERANCE_MINUTES = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS station_obs (
    stid TEXT NOT NULL,
    field TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (stid, field, ts_utc)
);
CREATE INDEX IF NOT EXISTS idx_station_obs_field_ts ON station_obs(field, ts_utc);

CREATE TABLE IF NOT EXISTS station_daily_extreme (
    stid TEXT NOT NULL,
    field TEXT NOT NULL,
    date_local TEXT NOT NULL,
    min_value REAL,
    min_ts_utc TEXT,
    max_value REAL,
    max_ts_utc TEXT,
    PRIMARY KEY (stid, field, date_local)
);
"""


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else STATION_HISTORY_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def record_observations(
    stations: list[dict], fields: dict[str, Any], *, run_time_utc: Optional[datetime] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert one sample per (station, field) and upsert today's running
    min/max. `fields` maps a field name (e.g. 'air_temp_f') to a callable
    `station -> float|None` that extracts/derives that field's value for one
    station (raw passthrough or a services.derived_met computation). Returns
    the number of (station, field) samples recorded."""
    run_time_utc = run_time_utc or datetime.now(timezone.utc)
    if run_time_utc.tzinfo is None:
        run_time_utc = run_time_utc.replace(tzinfo=timezone.utc)
    ts_utc = _iso(run_time_utc)
    date_local = run_time_utc.astimezone(CHICAGO_TZ).strftime("%Y-%m-%d")

    recorded = 0
    conn = _connect(db_path)
    try:
        with conn:
            for station in stations:
                stid = station.get("stid")
                if not stid:
                    continue
                for field_name, extractor in fields.items():
                    try:
                        value = extractor(station)
                    except Exception:
                        continue
                    if value is None:
                        continue
                    value = float(value)
                    conn.execute(
                        "INSERT OR REPLACE INTO station_obs (stid, field, ts_utc, value) VALUES (?, ?, ?, ?)",
                        (stid, field_name, ts_utc, value),
                    )
                    _upsert_daily_extreme(conn, stid, field_name, date_local, value, ts_utc)
                    recorded += 1
    finally:
        conn.close()
    return recorded


def _upsert_daily_extreme(conn: sqlite3.Connection, stid: str, field: str, date_local: str, value: float, ts_utc: str) -> None:
    row = conn.execute(
        "SELECT min_value, min_ts_utc, max_value, max_ts_utc FROM station_daily_extreme WHERE stid = ? AND field = ? AND date_local = ?",
        (stid, field, date_local),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO station_daily_extreme (stid, field, date_local, min_value, min_ts_utc, max_value, max_ts_utc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (stid, field, date_local, value, ts_utc, value, ts_utc),
        )
        return
    min_value, min_ts, max_value, max_ts = row["min_value"], row["min_ts_utc"], row["max_value"], row["max_ts_utc"]
    if value < min_value:
        min_value, min_ts = value, ts_utc
    if value > max_value:
        max_value, max_ts = value, ts_utc
    conn.execute(
        "UPDATE station_daily_extreme SET min_value = ?, min_ts_utc = ?, max_value = ?, max_ts_utc = ? "
        "WHERE stid = ? AND field = ? AND date_local = ?",
        (min_value, min_ts, max_value, max_ts, stid, field, date_local),
    )


def get_today_extreme(
    stid: str, field: str, *, now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> Optional[dict]:
    """Today's (America/Chicago) running min/max for one station+field, or
    None if nothing has been recorded yet today."""
    now_utc = now_utc or datetime.now(timezone.utc)
    date_local = now_utc.astimezone(CHICAGO_TZ).strftime("%Y-%m-%d")
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT min_value, min_ts_utc, max_value, max_ts_utc FROM station_daily_extreme "
            "WHERE stid = ? AND field = ? AND date_local = ?",
            (stid, field, date_local),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"min": row["min_value"], "min_ts": row["min_ts_utc"], "max": row["max_value"], "max_ts": row["max_ts_utc"]}


def get_today_extremes_batch(
    field: str, *, now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Same as get_today_extreme, but for every station in one query/
    connection instead of one connection per station - the current-
    conditions map scripts each loop over every station, so this turns N
    SQLite connections per script run into 1."""
    now_utc = now_utc or datetime.now(timezone.utc)
    date_local = now_utc.astimezone(CHICAGO_TZ).strftime("%Y-%m-%d")
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT stid, min_value, min_ts_utc, max_value, max_ts_utc FROM station_daily_extreme "
            "WHERE field = ? AND date_local = ?",
            (field, date_local),
        ).fetchall()
    finally:
        conn.close()
    return {
        row["stid"]: {"min": row["min_value"], "min_ts": row["min_ts_utc"], "max": row["max_value"], "max_ts": row["max_ts_utc"]}
        for row in rows
    }


def get_extremes_for_date_batch(
    field: str, date_local: str, *, db_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Daily min/max for every station on one America/Chicago calendar date
    (YYYY-MM-DD) - e.g. yesterday's maximum wind gust."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT stid, min_value, max_value FROM station_daily_extreme WHERE field = ? AND date_local = ?",
            (field, date_local),
        ).fetchall()
    finally:
        conn.close()
    return {row["stid"]: {"min": row["min_value"], "max": row["max_value"]} for row in rows}


def get_daily_maxima_batch(
    field: str, days: int, *, now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> dict[str, dict[str, float]]:
    """{stid: {date_local: max_value}} for the last `days` America/Chicago
    dates including today. For a cumulative-since-midnight field (rainfall)
    the daily max is the day's total."""
    now_utc = now_utc or datetime.now(timezone.utc)
    start = (now_utc.astimezone(CHICAGO_TZ) - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT stid, date_local, max_value FROM station_daily_extreme "
            "WHERE field = ? AND date_local >= ? AND max_value IS NOT NULL",
            (field, start),
        ).fetchall()
    finally:
        conn.close()
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        result.setdefault(row["stid"], {})[row["date_local"]] = row["max_value"]
    return result


def get_rolling_extremes_batch(
    field: str, hours: float, *, now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> dict[str, dict]:
    """Batch form of get_rolling_extreme - one query for every station."""
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    window_start = _iso(now_utc - timedelta(hours=hours))
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT stid, MIN(value) AS min_value, MAX(value) AS max_value FROM station_obs "
            "WHERE field = ? AND ts_utc >= ? GROUP BY stid",
            (field, window_start),
        ).fetchall()
    finally:
        conn.close()
    return {row["stid"]: {"min": row["min_value"], "max": row["max_value"]} for row in rows}


def get_values_n_hours_ago_batch(
    field: str, hours: float, *, tolerance_minutes: float = DEFAULT_LOOKUP_TOLERANCE_MINUTES,
    now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> dict[str, float]:
    """Batch form of get_value_n_hours_ago - one query for every station,
    then the same nearest-sample-in-window selection done in Python."""
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    target = now_utc - timedelta(hours=hours)
    window_start = _iso(target - timedelta(minutes=tolerance_minutes))
    window_end = _iso(target + timedelta(minutes=tolerance_minutes))

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT stid, ts_utc, value FROM station_obs WHERE field = ? AND ts_utc BETWEEN ? AND ?",
            (field, window_start, window_end),
        ).fetchall()
    finally:
        conn.close()

    best_by_stid: dict[str, tuple[float, float]] = {}  # stid -> (seconds_off, value)
    for row in rows:
        seconds_off = abs((_parse_iso(row["ts_utc"]) - target).total_seconds())
        existing = best_by_stid.get(row["stid"])
        if existing is None or seconds_off < existing[0]:
            best_by_stid[row["stid"]] = (seconds_off, row["value"])
    return {stid: value for stid, (_, value) in best_by_stid.items()}


def get_changes_batch(
    field: str, hours: float, current_values: dict[str, float], *,
    tolerance_minutes: float = DEFAULT_LOOKUP_TOLERANCE_MINUTES,
    now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> dict[str, float]:
    """Batch form of get_change. `current_values` maps stid -> the caller's
    freshly-fetched current reading for `field` (required here - unlike
    get_change, this does not fall back to the most recent recorded sample,
    since that per-station fallback query would defeat the point of
    batching)."""
    previous = get_values_n_hours_ago_batch(field, hours, tolerance_minutes=tolerance_minutes, now_utc=now_utc, db_path=db_path)
    return {
        stid: current - previous[stid]
        for stid, current in current_values.items()
        if stid in previous
    }


def get_value_n_hours_ago(
    stid: str, field: str, hours: float, *, tolerance_minutes: float = DEFAULT_LOOKUP_TOLERANCE_MINUTES,
    now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> Optional[float]:
    """Nearest recorded sample within [target - tolerance, target + tolerance]
    of exactly `hours` ago, or None if nothing falls in that window (e.g. not
    enough history yet after a fresh deploy or an extended outage)."""
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    target = now_utc - timedelta(hours=hours)
    window_start = _iso(target - timedelta(minutes=tolerance_minutes))
    window_end = _iso(target + timedelta(minutes=tolerance_minutes))

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ts_utc, value FROM station_obs WHERE stid = ? AND field = ? AND ts_utc BETWEEN ? AND ?",
            (stid, field, window_start, window_end),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    best = min(rows, key=lambda row: abs((_parse_iso(row["ts_utc"]) - target).total_seconds()))
    return best["value"]


def get_change(
    stid: str, field: str, hours: float, *, current_value: Optional[float] = None,
    tolerance_minutes: float = DEFAULT_LOOKUP_TOLERANCE_MINUTES, now_utc: Optional[datetime] = None,
    db_path: Optional[Path] = None,
) -> Optional[float]:
    """current_value minus the value ~`hours` ago, or None if either side is
    unavailable. Pass `current_value` explicitly when the caller already has
    a freshly-fetched reading (preferred); otherwise falls back to this
    station/field's most recent recorded sample."""
    previous = get_value_n_hours_ago(stid, field, hours, tolerance_minutes=tolerance_minutes, now_utc=now_utc, db_path=db_path)
    if previous is None:
        return None
    if current_value is None:
        conn = _connect(db_path)
        try:
            row = conn.execute(
                "SELECT value FROM station_obs WHERE stid = ? AND field = ? ORDER BY ts_utc DESC LIMIT 1",
                (stid, field),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        current_value = row["value"]
    return current_value - previous


def get_rolling_extreme(
    stid: str, field: str, hours: float, *, now_utc: Optional[datetime] = None, db_path: Optional[Path] = None,
) -> Optional[dict]:
    """Min/max over a trailing window of the last `hours` hours (e.g. a true
    rolling 24h max, as opposed to get_today_extreme's America/Chicago
    local-midnight reset). Returns None if there are no samples in the
    window."""
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    window_start = _iso(now_utc - timedelta(hours=hours))
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT MIN(value) AS min_value, MAX(value) AS max_value FROM station_obs "
            "WHERE stid = ? AND field = ? AND ts_utc >= ?",
            (stid, field, window_start),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["min_value"] is None:
        return None
    return {"min": row["min_value"], "max": row["max_value"]}


def prune_old(retention_hours: float = DEFAULT_RETENTION_HOURS, *, now_utc: Optional[datetime] = None, db_path: Optional[Path] = None) -> int:
    """Delete station_obs rows older than retention_hours. Cheap/indexed;
    safe to call every recording cycle. station_daily_extreme rows are tiny
    (one row per station/field/day) and intentionally NOT pruned here - they
    can be cleaned up on a much longer cycle if ever needed."""
    now_utc = now_utc or datetime.now(timezone.utc)
    cutoff = _iso(now_utc - timedelta(hours=retention_hours))
    conn = _connect(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM station_obs WHERE ts_utc < ?", (cutoff,))
            return cursor.rowcount
    finally:
        conn.close()

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from core.database import get_db_path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id TEXT PRIMARY KEY,
    cycle_time_utc TEXT NOT NULL,
    issued_at_utc TEXT,
    completed_at_utc TEXT,
    status TEXT NOT NULL CHECK(status IN ('staging','complete','failed','superseded')),
    is_public INTEGER NOT NULL DEFAULT 0 CHECK(is_public IN (0,1)),
    grid_id TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL CHECK(horizon_hours BETWEEN 1 AND 240),
    schema_version TEXT NOT NULL,
    config_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_cycles_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(source_cycles_json)),
    manifest_key TEXT,
    manifest_checksum TEXT,
    warnings_json TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(warnings_json)),
    superseded_run_id TEXT,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY(superseded_run_id) REFERENCES forecast_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_forecast_runs_public_cycle ON forecast_runs(is_public, cycle_time_utc DESC);

CREATE TABLE IF NOT EXISTS source_cycles (
    model TEXT NOT NULL,
    initialization_time_utc TEXT NOT NULL,
    member_count INTEGER NOT NULL CHECK(member_count > 0),
    variables_json TEXT NOT NULL CHECK(json_valid(variables_json)),
    source_status TEXT NOT NULL,
    object_key TEXT NOT NULL,
    checksum TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
    acquisition_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(acquisition_json)),
    PRIMARY KEY(model, initialization_time_utc, checksum)
);

CREATE TABLE IF NOT EXISTS forecast_assets (
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    variable TEXT NOT NULL DEFAULT '',
    aggregation TEXT NOT NULL DEFAULT '',
    valid_start_utc TEXT,
    valid_end_utc TEXT,
    unit TEXT,
    storage_data_type TEXT NOT NULL,
    crs TEXT,
    grid_id TEXT,
    object_key TEXT,
    local_path TEXT,
    checksum TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
    status TEXT NOT NULL DEFAULT 'ready',
    PRIMARY KEY(run_id, kind, variable, aggregation),
    FOREIGN KEY(run_id) REFERENCES forecast_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_forecast_assets_lookup ON forecast_assets(run_id, variable, aggregation, status);

CREATE TABLE IF NOT EXISTS station_forecast_hours (
    run_id TEXT NOT NULL,
    station_id TEXT NOT NULL,
    valid_time_utc TEXT NOT NULL,
    lead_hour INTEGER NOT NULL CHECK(lead_hour BETWEEN 0 AND 240),
    temperature_c REAL, dewpoint_c REAL, relative_humidity REAL,
    wind_u_ms REAL, wind_v_ms REAL, wind_speed_ms REAL, wind_gust_ms REAL,
    precipitation_mm REAL, solar_wm2 REAL, cloud_cover_pct REAL, mixing_height_m REAL,
    fuel_moisture_p10 REAL, fuel_moisture_p50 REAL, fuel_moisture_p90 REAL,
    danger_class INTEGER CHECK(danger_class BETWEEN 0 AND 4 OR danger_class IS NULL),
    danger_score REAL,
    meteorological_confidence INTEGER CHECK(meteorological_confidence BETWEEN 0 AND 100 OR meteorological_confidence IS NULL),
    category_confidence INTEGER CHECK(category_confidence BETWEEN 0 AND 100 OR category_confidence IS NULL),
    probability_rh_le_25 INTEGER CHECK(probability_rh_le_25 BETWEEN 0 AND 100 OR probability_rh_le_25 IS NULL),
    probability_gust_ge_30mph INTEGER CHECK(probability_gust_ge_30mph BETWEEN 0 AND 100 OR probability_gust_ge_30mph IS NULL),
    probability_concurrent INTEGER CHECK(probability_concurrent BETWEEN 0 AND 100 OR probability_concurrent IS NULL),
    source_mask INTEGER NOT NULL DEFAULT 0,
    quality_mask INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(run_id, station_id, valid_time_utc),
    FOREIGN KEY(run_id) REFERENCES forecast_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_station_hours_station_time ON station_forecast_hours(station_id, valid_time_utc);

CREATE TABLE IF NOT EXISTS station_forecast_days (
    run_id TEXT NOT NULL,
    station_id TEXT NOT NULL,
    local_date TEXT NOT NULL,
    day_index INTEGER NOT NULL CHECK(day_index BETWEEN 1 AND 3),
    temperature_low_c REAL, temperature_high_c REAL, minimum_rh REAL,
    maximum_wind_ms REAL, maximum_gust_ms REAL, precipitation_total_mm REAL,
    minimum_fuel_moisture_p50 REAL, fuel_moisture_p10 REAL, fuel_moisture_p90 REAL,
    peak_category INTEGER CHECK(peak_category BETWEEN 0 AND 4 OR peak_category IS NULL),
    critical_window_start_utc TEXT, critical_window_end_utc TEXT,
    concurrent_threshold_hours INTEGER NOT NULL DEFAULT 0,
    category_confidence INTEGER CHECK(category_confidence BETWEEN 0 AND 100 OR category_confidence IS NULL),
    meteorological_confidence INTEGER CHECK(meteorological_confidence BETWEEN 0 AND 100 OR meteorological_confidence IS NULL),
    PRIMARY KEY(run_id, station_id, local_date),
    FOREIGN KEY(run_id) REFERENCES forecast_runs(run_id)
);

CREATE TABLE IF NOT EXISTS station_observations_v2 (
    station_id TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    temperature_c REAL, dewpoint_c REAL, relative_humidity REAL,
    wind_u_ms REAL, wind_v_ms REAL, wind_speed_ms REAL, wind_gust_ms REAL,
    precipitation_mm REAL, solar_wm2 REAL, fuel_moisture REAL,
    network TEXT NOT NULL, source TEXT NOT NULL,
    qc_state TEXT NOT NULL, qc_reasons_json TEXT NOT NULL DEFAULT '[]' CHECK(json_valid(qc_reasons_json)),
    ingested_at_utc TEXT NOT NULL,
    PRIMARY KEY(station_id, observed_at_utc)
);

CREATE TABLE IF NOT EXISTS forecast_verification (
    run_id TEXT NOT NULL,
    station_id TEXT NOT NULL,
    valid_time_utc TEXT NOT NULL,
    variable TEXT NOT NULL,
    forecast_value REAL,
    observation_value REAL,
    error REAL,
    lead_bucket TEXT NOT NULL,
    model_version TEXT NOT NULL,
    config_version TEXT NOT NULL,
    qc_eligible INTEGER NOT NULL CHECK(qc_eligible IN (0,1)),
    PRIMARY KEY(run_id, station_id, valid_time_utc, variable),
    FOREIGN KEY(run_id) REFERENCES forecast_runs(run_id)
);

CREATE TABLE IF NOT EXISTS forecast_station_metadata (
    station_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    network_type TEXT NOT NULL CHECK(network_type IN ('RAWS','ASOS','AWOS','other')),
    elevation_m REAL,
    timezone TEXT NOT NULL DEFAULT 'America/Chicago',
    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
    sensor_capabilities_json TEXT NOT NULL DEFAULT '{}' CHECK(json_valid(sensor_capabilities_json)),
    updated_at_utc TEXT NOT NULL
);
"""


def ensure_schema(db_path: str | Path | None = None) -> None:
    path = Path(db_path or get_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_SQL)


@contextmanager
def transaction(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path or get_db_path())
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def upsert_run(connection: sqlite3.Connection, run: dict[str, Any]) -> None:
    columns = (
        "run_id", "cycle_time_utc", "issued_at_utc", "completed_at_utc", "status", "is_public",
        "grid_id", "horizon_hours", "schema_version", "config_version", "model_version",
        "source_cycles_json", "manifest_key", "manifest_checksum", "warnings_json",
        "superseded_run_id", "created_at_utc",
    )
    values = [run.get(name) for name in columns]
    connection.execute(
        f"INSERT INTO forecast_runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
        "ON CONFLICT(run_id) DO UPDATE SET completed_at_utc=excluded.completed_at_utc,status=excluded.status,"
        "is_public=excluded.is_public,manifest_key=excluded.manifest_key,manifest_checksum=excluded.manifest_checksum,"
        "warnings_json=excluded.warnings_json,superseded_run_id=excluded.superseded_run_id",
        values,
    )


def replace_assets(connection: sqlite3.Connection, run_id: str, assets: Iterable[dict[str, Any]]) -> None:
    connection.execute("DELETE FROM forecast_assets WHERE run_id=?", (run_id,))
    columns = (
        "run_id", "kind", "variable", "aggregation", "valid_start_utc", "valid_end_utc", "unit",
        "storage_data_type", "crs", "grid_id", "object_key", "local_path", "checksum", "byte_size", "status",
    )
    connection.executemany(
        f"INSERT INTO forecast_assets ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        ([asset.get(name) for name in columns] for asset in assets),
    )


def resolve_run(connection: sqlite3.Connection, run_id: str | None = None) -> sqlite3.Row | None:
    if run_id:
        return connection.execute("SELECT * FROM forecast_runs WHERE run_id=? AND status IN ('complete','superseded')", (run_id,)).fetchone()
    return connection.execute(
        "SELECT * FROM forecast_runs WHERE is_public=1 AND status='complete' ORDER BY cycle_time_utc DESC LIMIT 1"
    ).fetchone()


def run_assets(connection: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return connection.execute("SELECT * FROM forecast_assets WHERE run_id=? AND status='ready' ORDER BY kind,variable,aggregation", (run_id,)).fetchall()


def asset_for_layer(connection: sqlite3.Connection, run_id: str, variable: str, aggregation: str = "hourly") -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM forecast_assets WHERE run_id=? AND kind='raster' AND variable=? AND aggregation=? AND status='ready'",
        (run_id, variable, aggregation),
    ).fetchone()


def prune_hot_storage(forecast_root: str | Path, *, archive_verified: bool, now: datetime | None = None) -> dict[str, int]:
    """Enforce 7/30/90-day local retention without deleting immutable metadata."""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    point_cutoff = (current - timedelta(days=90)).isoformat(timespec="seconds").replace("+00:00", "Z")
    asset_cutoff = (current - timedelta(days=30)).isoformat(timespec="seconds").replace("+00:00", "Z")
    root = Path(forecast_root).resolve()
    removed_files = removed_cache = 0
    with transaction() as connection:
        old_runs = [row[0] for row in connection.execute("SELECT run_id FROM forecast_runs WHERE cycle_time_utc<? AND is_public=0", (point_cutoff,))]
        for run_id in old_runs:
            connection.execute("DELETE FROM station_forecast_hours WHERE run_id=?", (run_id,))
            connection.execute("DELETE FROM station_forecast_days WHERE run_id=?", (run_id,))
        if archive_verified:
            assets = connection.execute(
                "SELECT a.rowid,a.local_path FROM forecast_assets a JOIN forecast_runs r ON r.run_id=a.run_id WHERE r.cycle_time_utc<? AND r.is_public=0 AND a.local_path IS NOT NULL AND a.object_key IS NOT NULL",
                (asset_cutoff,),
            ).fetchall()
            for asset in assets:
                path = Path(asset["local_path"]).resolve()
                if path != root and root in path.parents and path.is_file():
                    path.unlink()
                    removed_files += 1
                connection.execute("UPDATE forecast_assets SET local_path=NULL WHERE rowid=?", (asset["rowid"],))
    cache_root = root / "download-cache"
    cache_cutoff = (current - timedelta(days=7)).timestamp()
    if cache_root.is_dir():
        for path in cache_root.rglob("*"):
            if path.is_file() and path.stat().st_mtime < cache_cutoff:
                path.unlink()
                removed_cache += 1
    return {"pointRunsPruned": len(old_runs), "artifactFilesRemoved": removed_files, "cacheFilesRemoved": removed_cache}

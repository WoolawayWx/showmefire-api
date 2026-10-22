import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecast_v1.repository import ensure_schema

API_ROOT = Path(__file__).resolve().parents[1]
CENTRAL_OFFSET = timedelta(hours=5)  # CDT in October


def _z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _observe_day(connection, station: str, local_date: str, fuel: float, rh: float, wind_ms: float):
    midnight = datetime.fromisoformat(local_date).replace(tzinfo=timezone.utc) + CENTRAL_OFFSET
    for hour in range(24):
        connection.execute(
            """INSERT INTO station_observations_v2(station_id,observed_at_utc,relative_humidity,wind_speed_ms,
               fuel_moisture,network,source,qc_state,ingested_at_utc) VALUES (?,?,?,?,?,?,?,?,?)""",
            (station, _z(midnight + timedelta(hours=hour, minutes=10)), rh, wind_ms, fuel, "RAWS", "synoptic", "pass", "x"),
        )


def test_backtest_scores_day_2_3_peaks_against_observed_rule_and_persistence(tmp_path):
    db = tmp_path / "backtest.db"
    ensure_schema(db)
    cycle = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    with sqlite3.connect(db) as connection:
        connection.execute(
            """INSERT INTO forecast_runs(run_id,cycle_time_utc,status,grid_id,horizon_hours,schema_version,
               config_version,model_version,created_at_utc) VALUES ('R',?,?,?,?,?,?,?,?)""",
            (_z(cycle), "complete", "g", 72, "s", "beta-2", "m", _z(cycle)),
        )
        _observe_day(connection, "RAWS1", "2026-10-09", 20.0, 70.0, 2.0)  # day before the run: Low
        _observe_day(connection, "RAWS1", "2026-10-11", 7.0, 30.0, 8.0)  # Elevated (15.5 kt, RH 30)
        _observe_day(connection, "RAWS1", "2026-10-12", 20.0, 70.0, 2.0)  # Low
        for day_index, local_date, peak in ((2, "2026-10-11", 2), (3, "2026-10-12", 2)):
            connection.execute(
                "INSERT INTO station_forecast_days(run_id,station_id,local_date,day_index,peak_category) VALUES ('R','RAWS1',?,?,?)",
                (local_date, day_index, peak),
            )
        valid = datetime(2026, 10, 11, 20, tzinfo=timezone.utc)  # 15 CDT on Day 2, obs at :10
        connection.execute(
            "INSERT INTO station_forecast_hours(run_id,station_id,valid_time_utc,lead_hour,fuel_moisture_p50) VALUES ('R','RAWS1',?,?,?)",
            (_z(valid), 32, 9.0),
        )
        connection.execute(
            """INSERT INTO forecast_verification(run_id,station_id,valid_time_utc,variable,forecast_value,
               observation_value,error,lead_bucket,model_version,config_version,qc_eligible)
               VALUES ('R','RAWS1',?,'relative_humidity_2m',36,30,6,'25-48','m','beta-2',1)""",
            (_z(valid),),
        )
    output = tmp_path / "result.json"
    completed = subprocess.run(
        [sys.executable, str(API_ROOT / "scripts" / "forecast_v1_backtest.py"), "--db", str(db), "--since", "2026-10-01", "--json", str(output)],
        cwd=API_ROOT, capture_output=True, text=True, check=True,
    )
    assert "Peak fire danger at RAWS" in completed.stdout
    results = json.loads(output.read_text())
    assert results["weather"] == [{"config": "beta-2", "variable": "relative_humidity_2m", "day": "Day 2", "pairs": 1, "stations": 1, "mae": 6.0, "bias": 6.0}]
    assert results["fuel"] == [{"config": "beta-2", "day": "Day 2", "pairs": 1, "mae": 2.0, "bias": 2.0}]
    day_2, day_3 = results["danger"]
    assert (day_2["day"], day_2["forecast"]["pod"], day_2["forecast"]["far"]) == ("Day 2", 1.0, 0.0)
    assert day_2["persistence"]["pod"] == 0.0  # Low the day before missed the Elevated day
    assert (day_3["forecast"]["far"], day_3["forecast"]["observedElevatedPlus"]) == (1.0, 0)
    with sqlite3.connect(db) as connection:  # read-only: nothing written
        assert connection.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0] == 1

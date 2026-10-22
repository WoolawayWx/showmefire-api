import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from forecast_v1.bias import MIN_PAIRS, rolling_bias_correction
from forecast_v1.engine import apply_corrections
from forecast_v1.repository import ensure_schema

CYCLE = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)  # 07 CDT
TIMES = xr.DataArray(
    np.array([np.datetime64(CYCLE.replace(tzinfo=None) + timedelta(hours=lead), "ns") for lead in range(73)]), dims="time",
)


def _z(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _verification(db, rows):
    with sqlite3.connect(db) as connection:
        connection.executemany(
            """INSERT INTO forecast_verification
               (run_id,station_id,valid_time_utc,variable,forecast_value,observation_value,error,
                lead_bucket,model_version,config_version,qc_eligible) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [(run, station, _z(valid), variable, 0.0, 0.0, error, bucket, "m", "c", 1)
             for run, station, valid, variable, error, bucket in rows],
        )


def _run(db, run_id, warnings="[]"):
    with sqlite3.connect(db) as connection:
        connection.execute(
            """INSERT INTO forecast_runs(run_id,cycle_time_utc,status,grid_id,horizon_hours,schema_version,
               config_version,model_version,warnings_json,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (run_id, _z(CYCLE - timedelta(days=5)), "complete", "g", 72, "s", "c", "m", warnings, _z(CYCLE)),
        )


def _days_at(utc_hour: int, days: int, stations: int):
    for day in range(1, days + 1):
        valid = (CYCLE - timedelta(days=day)).replace(hour=utc_hour)
        for station in range(stations):
            yield valid, f"S{station}"


def test_residual_is_minus_the_local_time_block_bias_with_bucket_fallback(tmp_path):
    db = tmp_path / "bias.db"
    ensure_schema(db)
    # Afternoon (19Z = 14 CDT, block 12-14 CDT): forecasts 2 C too warm, 40 pairs.
    rows = [("r", station, valid, "temperature_2m", 2.0, "0-24") for valid, station in _days_at(19, 10, 4)]
    # Night (07Z = 02 CDT): 1 C too cold, only 10 pairs - too thin for its own block.
    rows += [("r", station, valid, "temperature_2m", -1.0, "0-24") for valid, station in _days_at(7, 10, 1)]
    # RH from only two stations fails the station gate entirely.
    rows += [("r", station, valid, "relative_humidity_2m", 5.0, "0-24") for valid, station in _days_at(19, 20, 2)]
    _verification(db, rows)

    bias = rolling_bias_correction(TIMES, CYCLE, db)
    temperature = bias.residuals["temperature_2m"].values
    assert temperature[7] == pytest.approx(-2.0)  # lead 7 = 19Z = 14 CDT, its own block
    assert temperature[19] == pytest.approx(-1.4)  # 07Z: thin block -> bucket mean (40*2 - 10) / 50
    assert temperature[0] == pytest.approx(-1.4)  # a block with no pairs -> bucket mean
    assert not temperature[25:].any()  # no 25-48 / 49-72 history -> no correction
    assert "relative_humidity_2m" not in bias.residuals
    summary = bias.summary["variables"]["temperature_2m"]
    assert summary["applied"] and summary["buckets"]["0-24"] == {"pairs": 50, "stations": 4, "meanError": 1.4}
    assert summary["leadHours"] == {"block": 3, "bucket": 22, "none": 48}  # leads 5-7 are 12-14 CDT
    assert bias.summary["variables"]["relative_humidity_2m"]["applied"] is False


def test_previous_corrections_are_added_back_so_bias_does_not_fade(tmp_path):
    db = tmp_path / "bias.db"
    ensure_schema(db)
    pairs = list(_days_at(19, 10, 4))
    # The corrected forecasts verified perfectly because -2 C was applied.
    _verification(db, [("corrected", station, valid, "temperature_2m", 0.0, "0-24") for valid, station in pairs])
    with sqlite3.connect(db) as connection:
        connection.executemany(
            "INSERT INTO forecast_bias_applied(run_id,valid_time_utc,variable,residual) VALUES (?,?,?,?)",
            sorted({("corrected", _z(valid), "temperature_2m", -2.0) for valid, _ in pairs}),
        )
    assert rolling_bias_correction(TIMES, CYCLE, db).residuals["temperature_2m"].values[7] == pytest.approx(-2.0)


def test_day_3_bias_ignores_gefs_only_fallback_runs(tmp_path):
    db = tmp_path / "bias.db"
    ensure_schema(db)
    _run(db, "fallback", '["coarse_synoptic_fallback:gefs:49-72"]')
    _run(db, "rrfs")
    rows = [("fallback", station, valid, "wind_gust_10m", 3.0, "49-72") for valid, station in _days_at(19, 10, 4)]
    _verification(db, rows)
    assert "wind_gust_10m" not in rolling_bias_correction(TIMES, CYCLE, db).residuals
    _verification(db, [("rrfs", station, valid, "wind_gust_10m", 1.0, "49-72") for valid, station in _days_at(19, 10, 4)])
    residual = rolling_bias_correction(TIMES, CYCLE, db).residuals["wind_gust_10m"].values
    assert residual[55] == pytest.approx(-1.0)  # lead 55 = 19Z, only the RRFS run's errors count
    assert MIN_PAIRS == 30


def test_residuals_are_capped_like_apply_corrections(tmp_path):
    db = tmp_path / "bias.db"
    ensure_schema(db)
    _verification(db, [(f"r{bucket}", station, valid, "temperature_2m", 9.0, bucket)
                       for bucket in ("0-24", "25-48", "49-72") for valid, station in _days_at(19, 10, 4)])
    residual = rolling_bias_correction(TIMES, CYCLE, db).residuals["temperature_2m"].values
    assert (residual[7], residual[31], residual[55]) == pytest.approx((-2.8, -2.1, -1.4))


def test_no_history_means_no_correction(tmp_path):
    bias = rolling_bias_correction(TIMES, CYCLE, tmp_path / "missing.db")
    assert bias.residuals == {} and bias.summary["applied"] is False


def test_wind_speed_correction_scales_components_and_keeps_direction():
    shape = (73, 1, 2)
    dataset = xr.Dataset(
        {
            "wind_u_10m": (("time", "y", "x"), np.tile(np.array([3.0, 0.1], dtype=np.float32), (73, 1, 1))),
            "wind_v_10m": (("time", "y", "x"), np.tile(np.array([4.0, 0.1], dtype=np.float32), (73, 1, 1))),
            "relative_humidity_2m": (("time", "y", "x"), np.full(shape, 40.0, dtype=np.float32)),
            "quality_mask": (("time", "y", "x"), np.zeros(shape, dtype=np.uint16)),
        },
        coords={"time": TIMES.values},
    )
    dataset["wind_speed_10m"] = np.hypot(dataset.wind_u_10m, dataset.wind_v_10m)
    residual = xr.DataArray(np.full(73, -1.0, dtype=np.float32), dims="time", coords={"time": TIMES.values})
    residual[12] = -10.0  # beyond the 2.25 m/s cap
    corrected = apply_corrections(dataset, {"wind_speed_10m": residual}).dataset
    assert float(corrected.wind_speed_10m.isel(time=0, x=0)) == pytest.approx(4.0)
    assert float(corrected.wind_speed_10m.isel(time=12, x=0)) == pytest.approx(2.75)
    assert float(corrected.wind_direction_10m.isel(time=0, x=0)) == pytest.approx(float(np.degrees(np.arctan2(3, 4)) + 180))
    assert float(corrected.wind_speed_10m.isel(time=0, x=1)) == pytest.approx(float(np.hypot(0.1, 0.1)), rel=1e-5)  # calm

"""The "what happened" RTMA analysis days (services/rtma_analysis.py) and their API (routers/rtma_analysis.py)."""
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services import cdn_datasets, rtma_analysis as ra
from services.hrrr_hourly import live_store

_real_raws_observations = ra.raws_observations

DAY = date(2026, 10, 19)                       # a CDT day: 05Z on the 19th to 05Z on the 20th
LOV, LATIN = 265.0, 25.0


def write_hour(cache, hour_utc, temp_k=293.15, rh=40.0, u=4.0, v=0.0, apcp=0.0, missing_precip=False):
    """A cached RTMA hour shaped like rtma_capture writes it (2-D lat/lon, grid-relative winds)."""
    lat, lon = np.meshgrid(np.array([37.0, 38.0, 39.0]), np.array([-94.0, -95.0, -92.0, -93.0]), indexing="ij")
    shape = lat.shape
    wind_attrs = {"GRIB_LoVInDegrees": LOV, "GRIB_Latin1InDegrees": LATIN, "GRIB_uvRelativeToGrid": 1}
    dataset = xr.Dataset(
        {"t2m": (("y", "x"), np.full(shape, temp_k, "float32")), "d2m": (("y", "x"), np.full(shape, temp_k - 10, "float32")),
         "r2": (("y", "x"), np.full(shape, rh, "float32")),
         "u10": (("y", "x"), np.full(shape, u, "float32"), wind_attrs), "v10": (("y", "x"), np.full(shape, v, "float32"), wind_attrs),
         "apcp": (("y", "x"), np.full(shape, apcp, "float32"), {"missing": int(missing_precip)})},
        coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon + 360)})   # RTMA stores 0-360
    dataset.to_netcdf(cache / f"rtma_{hour_utc:%Y%m%d_%H}z.nc")


@pytest.fixture
def dirs(tmp_path):
    cache, root = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    return cache, root


@pytest.fixture(autouse=True)
def no_raws(monkeypatch):
    """No RAWS readings unless a test supplies them (the real archives and station history stay out of it)."""
    monkeypatch.setattr(ra, "raws_observations", lambda day, hours, now: {})


def test_a_day_is_its_local_hours_including_the_clock_change_days():
    assert len(ra.day_hours(DAY)) == 24
    assert ra.day_hours(DAY)[0] == datetime(2026, 10, 19, 5, tzinfo=timezone.utc)
    fall_back = ra.day_hours(date(2026, 11, 1))
    assert len(fall_back) == 25
    assert [ra.step_label(h, fall_back) for h in fall_back[:3]] == ["12 AM", "1 AM CDT", "1 AM CST"]
    assert len(ra.day_hours(date(2026, 3, 8))) == 23


def test_derived_fields_match_the_hourly_hrrr_definitions(dirs):
    cache, _ = dirs
    hour = datetime(2026, 10, 19, 20, tzinfo=timezone.utc)
    write_hour(cache, hour, temp_k=303.15, rh=20.0, u=0.0, v=10.0)     # 86 F, 20 %, 22 mph wind from grid-south
    with xr.open_dataset(cache / "rtma_20261019_20z.nc") as dataset:
        out = ra.derive_hour(dataset.load())
    assert out["temp_f"][0, 0] == pytest.approx(86.0, abs=0.01)
    assert out["fuel_moisture_pct"][0, 0] == pytest.approx(3.0 + 0.25 * 20.0)
    assert out["wind_speed_mph"][0, 0] == pytest.approx(22.37, abs=0.01)
    # At the grid's central longitude (-95 E = 265) grid-relative is earth-relative: wind from due south.
    assert out["wind_dir_deg"][1, 1] == pytest.approx(180.0, abs=0.01)
    # East of it the grid is rotated, so the same grid-relative wind comes from a different true direction.
    assert abs(out["wind_dir_deg"][1, 2] - 180.0) > 0.5
    assert out["fire_danger_est"][0, 0] >= 1


def test_today_fills_in_and_yesterday_is_complete(dirs):
    cache, root = dirs
    hours = ra.day_hours(DAY)
    for hour in hours[:3]:
        write_hour(cache, hour)
    write_hour(cache, hours[5], missing_precip=True)

    partial = ra.build_day(DAY, root=root, cache_dir=cache, now=hours[6] + timedelta(minutes=50))
    assert partial["complete"] is False
    assert partial["forecast_hours"] == [0, 1, 2, 5]
    # 3 and 4 are over two hours old, so they are missing, not pending; 6 (just ended) onwards can still arrive.
    assert partial["pending_hours"] == list(range(6, 24))
    assert "precip_mm_1h" in partial["variables"] and 5 not in partial["variables"]["precip_mm_1h"]["hours_present"]
    assert partial["step_names"][0] == "12 AM" and len(partial["step_names"]) == 24
    assert json.loads((root / "latest.json").read_text())["cycle"] == "20261019"

    done = ra.build_day(DAY, root=root, cache_dir=cache, now=hours[-1] + timedelta(hours=4))
    assert done["complete"] is True and done["pending_hours"] == []
    assert done["missing_hours"] == [i for i in range(24) if i not in (0, 1, 2, 5)]

    series = live_store.point_series(root / "20261019" / "forecast.zarr", done, 38.0, -95.0, 10.0)
    assert series["lead_hours"] == [0, 1, 2, 5]
    assert series["series"]["temp_f"][0] == pytest.approx(68.0, abs=0.25)   # stored at the archive's 0.5 F step


def test_a_day_with_no_hours_is_not_built(dirs):
    cache, root = dirs
    assert ra.build_day(DAY, root=root, cache_dir=cache, now=datetime(2026, 10, 21, tzinfo=timezone.utc)) is None
    assert not (root / "20261019").exists() and not (root / ".20261019.tmp").exists()


def test_refresh_publishes_a_finished_day_once(dirs, monkeypatch):
    cache, root = dirs
    for hour in ra.day_hours(DAY):
        write_hour(cache, hour)
    calls = []
    monkeypatch.setattr(cdn_datasets, "try_publish_run", lambda *a, **k: calls.append((a, k)) or {"uploaded": 1})
    now = datetime(2026, 10, 20, 16, tzinfo=timezone.utc)          # 11 AM CDT the next day

    first = ra.refresh(now=now, root=root, cache_dir=cache)
    assert first["built"]["20261019"] == "24/24 hours" and first["published"] == ["20261019"]
    assert calls[0][0][0] == "rtma-analysis" and calls[0][1]["keep_days"] == ra.keep_days()
    assert ra.refresh(now=now, root=root, cache_dir=cache)["published"] == []    # not again
    assert len(calls) == 1


def test_prune_keeps_the_window(dirs):
    _, root = dirs
    for name in ("20260101", "20261018", "notaday"):
        (root / name).mkdir(parents=True)
    assert ra.prune(root, now=datetime(2026, 10, 20, tzinfo=timezone.utc), days=30) == ["20260101"]
    assert (root / "20261018").exists() and (root / "notaday").exists()


def test_api_serves_runs_manifest_zarr_and_points(dirs, monkeypatch):
    cache, root = dirs
    for hour in ra.day_hours(DAY)[:4]:
        write_hour(cache, hour)
    ra.build_day(DAY, root=root, cache_dir=cache, now=datetime(2026, 10, 21, tzinfo=timezone.utc))
    monkeypatch.setenv("SMF_RTMA_ANALYSIS_DIR", str(root))
    from routers import rtma_analysis as router_module

    app = FastAPI()
    app.include_router(router_module.router)
    client = TestClient(app)

    assert client.get("/rtma-analysis/status").json()["serving_cycle"] == "20261019"
    runs = client.get("/rtma-analysis/runs").json()
    assert runs["latest"] == "20261019" and runs["runs"][0]["forecast_hours"] == [0, 1, 2, 3]
    assert client.get("/rtma-analysis/runs/latest").json()["kind"] == "analysis-day"
    assert client.get("/rtma-analysis/runs/latest/zarr/.zmetadata").status_code == 200
    # The HTTP client would collapse "../", so the traversal guard is called directly.
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as refused:
        router_module.get_zarr_file("20261019", "../../latest.json")
    assert refused.value.status_code == 404
    assert client.get("/rtma-analysis/runs/20261019_12z").status_code == 400
    point = client.get("/rtma-analysis/runs/20261019/point", params={"lat": 38.0, "lon": -95.0}).json()
    assert point["units"]["temp_f"] == "degF" and len(point["series"]["rh_pct"]) == 4
    assert client.get("/rtma-analysis/runs/20261019/point", params={"lat": 45.0, "lon": -80.0}).status_code == 404


def _raws_payload(hour_utc, value):
    """Three RAWS in a Synoptic timeseries payload, each reading `value` 7 minutes before `hour_utc`."""
    when = (hour_utc - timedelta(minutes=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"STATION": [{"STID": f"RAWS{i}", "LATITUDE": str(lat), "LONGITUDE": str(lon),
                         "OBSERVATIONS": {"date_time": [when], "fuel_moisture_set_1": [value]}}
                        for i, (lat, lon) in enumerate([(37.0, -94.0), (38.0, -95.0), (39.0, -92.0)])]}


def test_calibrated_fuel_moisture_layer_where_raws_reported(dirs, monkeypatch):
    cache, root = dirs
    hours = ra.day_hours(DAY)
    for hour in hours[:3]:
        write_hour(cache, hour, rh=40.0)                  # estimate 3 + 0.25 x 40 = 13 %
    from services.raws_calibration import synoptic_fuel_observations
    readings = {i: synoptic_fuel_observations(_raws_payload(hours[i], 19.0), hours[i]) for i in (0, 1)}
    monkeypatch.setattr(ra, "raws_observations", lambda day, hours, now: readings)

    manifest = ra.build_day(DAY, root=root, cache_dir=cache, now=hours[-1] + timedelta(hours=4))

    assert manifest["variables"]["fuel_moisture_cal_pct"]["hours_present"] == [0, 1]
    assert manifest["variables"]["fuel_moisture_pct"]["hours_present"] == [0, 1, 2]
    assert manifest["calibration"]["hours"] == {"0": {"stations": 3, "mean_residual": 6.0},
                                                "1": {"stations": 3, "mean_residual": 6.0}}
    store = root / "20261019" / "forecast.zarr"
    calibrated = live_store.read_step(store, "fuel_moisture_cal_pct", 0)
    estimate = live_store.read_step(store, "fuel_moisture_pct", 0)
    assert estimate[0, 0] == pytest.approx(13.0)
    assert calibrated[0, 0] == pytest.approx(19.0, abs=0.5)    # the RAWS at 37 N, 94 W
    assert live_store.read_step(store, "fuel_moisture_cal_pct", 2) is None


def test_raws_readings_prefer_the_full_day_archive_then_fill_today_from_history(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "raws_observations", _real_raws_observations)
    hours = ra.day_hours(DAY)
    monkeypatch.setenv("SMF_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setattr("core.config.ARCHIVE_RAW_DATA_DIR", str(tmp_path / "archive" / "raw_data"))
    history_calls = []
    monkeypatch.setattr(ra, "_station_coordinates", lambda: {"RAWS0": (-94.0, 37.0)})
    monkeypatch.setattr(ra, "_history_observations",
                        lambda hour, coordinates: history_calls.append(hour) or [{"fuel_moisture": 9.0}] * 3)

    # Nothing archived yet: hours of the last 36 h come from the station history, later hours are skipped.
    now = hours[3] + timedelta(minutes=30)
    found = ra.raws_observations(DAY, hours, now)
    assert sorted(found) == [0, 1, 2, 3] and history_calls == hours[:4]

    # The whole-day archive covers every hour, so the history is not consulted.
    from services import full_obs_archive
    path = full_obs_archive.output_path(DAY.isoformat())
    path.parent.mkdir(parents=True)
    payload = {"STATION": []}
    for hour in hours:
        for station in _raws_payload(hour, 15.0)["STATION"]:
            entry = next((s for s in payload["STATION"] if s["STID"] == station["STID"]), None)
            if entry is None:
                payload["STATION"].append(station)
            else:
                for key in ("date_time", "fuel_moisture_set_1"):
                    entry["OBSERVATIONS"][key] += station["OBSERVATIONS"][key]
    path.write_text(json.dumps(payload))
    history_calls.clear()
    found = ra.raws_observations(DAY, hours, hours[-1] + timedelta(hours=4))
    assert sorted(found) == list(range(24)) and history_calls == []
    assert {reading["fuel_moisture"] for reading in found[12]} == {15.0}

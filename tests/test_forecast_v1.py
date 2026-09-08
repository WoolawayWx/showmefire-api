from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forecast_v1.adapters import DatasetAdapter, SourceCube
from forecast_v1.acquisition import AcquisitionSpec, acquire_source, latest_publishable_12z
from forecast_v1.artifacts import write_cog, write_netcdf, write_points_parquet
from forecast_v1.contracts import PUBLIC_GRID, QUALITY_BITS, REQUIRED_VARIABLES
from forecast_v1.contracts import GridDefinition
from forecast_v1.engine import _classify_arrays, apply_corrections, blend_sources, build_forecast_cube, local_day_slices
from core.fire_danger import calculate_fire_danger, MPS_TO_KNOTS
from forecast_v1.repository import ensure_schema
from forecast_v1.pipeline import initialize_fuel_moisture, publish_run, source_member_count, summarize_ensemble_for_public
from forecast_v1.verification import evaluate_rrfs_refs_promotion


def test_latest_publishable_cycle_waits_for_extended_source_age():
    assert latest_publishable_12z(datetime(2026, 9, 7, 17, tzinfo=timezone.utc)) == datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    assert latest_publishable_12z(datetime(2026, 9, 7, 18, tzinfo=timezone.utc)) == datetime(2026, 9, 7, 12, tzinfo=timezone.utc)


def test_herbie_acquisition_normalizes_clips_and_checks_hours(tmp_path):
    cycle = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    steps = np.array([0, 1], dtype="timedelta64[h]")
    coordinates = {
        "step": steps, "latitude": ("y", [36.0, 37.0]), "longitude": ("x", [-94.0, -93.0]),
        "time": np.datetime64(cycle.replace(tzinfo=None)),
    }
    shape = (2, 2, 2)
    variables = {
        "t2m": (("step", "y", "x"), np.full(shape, 295.0, dtype=np.float32), {"units": "K"}),
        "r2": (("step", "y", "x"), np.full(shape, 30.0, dtype=np.float32), {"units": "%"}),
        "u10": (("step", "y", "x"), np.full(shape, 4.0, dtype=np.float32), {"units": "m s-1"}),
        "v10": (("step", "y", "x"), np.full(shape, 2.0, dtype=np.float32), {"units": "m s-1"}),
        "gust": (("step", "y", "x"), np.full(shape, 8.0, dtype=np.float32), {"units": "m s-1"}),
        "tp": (("step", "y", "x"), np.zeros(shape, dtype=np.float32), {"units": "mm"}),
        "dswrf": (("step", "y", "x"), np.full(shape, 500.0, dtype=np.float32), {"units": "W m-2"}),
        "tcc": (("step", "y", "x"), np.full(shape, 20.0, dtype=np.float32), {"units": "%"}),
        "hpbl": (("step", "y", "x"), np.full(shape, 1000.0, dtype=np.float32), {"units": "m"}),
        "soilw": (("step", "y", "x"), np.full(shape, .2, dtype=np.float32), {"units": "1"}),
        "weasd": (("step", "y", "x"), np.zeros(shape, dtype=np.float32), {"units": "mm"}),
    }
    raw = xr.Dataset(variables, coords=coordinates)

    class FakeFastHerbie:
        def __init__(self, **kwargs): self.calls = 0
        def xarray(self, search, **kwargs):
            self.calls += 1
            if "700" in search: raise RuntimeError("optional unavailable")
            return raw

    spec = AcquisitionSpec("hrrr", "hrrr", "sfc", (0, 1), (None,), required=True)
    progress = []
    cube = acquire_source(spec, cycle, tmp_path, fast_herbie_factory=FakeFastHerbie, progress_callback=progress.append)
    assert cube.dataset.sizes == {"member": 1, "time": 2, "y": 2, "x": 2}
    assert cube.dataset.attrs["acquisition"] == "herbie-indexed-grib"
    assert float(cube.dataset.temperature_2m.mean()) == pytest.approx(21.85)
    assert progress == [{"event": "member_completed", "member": "deterministic", "completed": 1, "total": 1}]


def test_hrrr_warm_season_missing_swe_uses_flagged_zero_fallback(tmp_path, monkeypatch):
    cycle = datetime(2026, 9, 7, 12, tzinfo=timezone.utc)
    base = source_cube("hrrr", 20, cycle=cycle, hours=2).dataset.isel(member=0, drop=True)
    aliases = {
        "temperature_2m": "t2m", "relative_humidity_2m": "r2", "wind_u_10m": "u10",
        "wind_v_10m": "v10", "wind_gust_10m": "gust", "precipitation_increment": "tp",
        "shortwave_down": "dswrf", "cloud_cover": "tcc", "mixing_height": "hpbl",
        "soil_moisture": "soilw",
    }
    raw = base.drop_vars("snow_water_equivalent").rename(aliases)
    raw = raw.assign_coords(
        latitude=("y", [36.0, 37.0]), longitude=("x", [-94.0, -93.0, -92.0]),
    )

    class MissingSweHerbie:
        def __init__(self, **kwargs): pass
        def xarray(self, search, **kwargs):
            if "WEASD" in search:
                raise RuntimeError("message unavailable")
            return raw

    monkeypatch.setenv("SMF_HERBIE_QUERY_ATTEMPTS", "1")
    spec = AcquisitionSpec("hrrr", "hrrr", "sfc", (0, 1), (None,), required=True)
    cube = acquire_source(spec, cycle, tmp_path, fast_herbie_factory=MissingSweHerbie)

    assert float(cube.dataset.snow_water_equivalent.max()) == 0
    assert "seasonal_swe_assumed_zero" in cube.quality_flags


def source_cube(model: str, value: float, *, cycle: datetime | None = None, hours: int = 73) -> SourceCube:
    cycle = cycle or datetime(2026, 9, 6, 12, tzinfo=timezone.utc)
    coords = {
        "member": ["control"],
        "time": np.array([np.datetime64(cycle.replace(tzinfo=None) + timedelta(hours=i), "ns") for i in range(hours)]),
        "y": [4_000_500.0, 3_997_500.0],
        "x": [500_500.0, 503_500.0, 506_500.0],
    }
    variables = {}
    for name in REQUIRED_VARIABLES:
        current = value
        if name == "relative_humidity_2m": current = 30.0
        elif name in {"wind_u_10m", "wind_v_10m"}: current = 4.0
        elif name == "wind_gust_10m": current = 8.0
        elif name == "shortwave_down": current = 500.0
        elif name == "soil_moisture": current = 0.2
        elif name == "cloud_cover": current = 20.0
        variables[name] = (("member", "time", "y", "x"), np.full((1, hours, 2, 3), current, dtype=np.float32))
    return SourceCube(model, cycle, xr.Dataset(variables, coords=coords), ("control",))


def test_public_ensemble_summary_preserves_mean_and_spread_with_two_members():
    cube = source_cube("gefs", 20, hours=3)
    member_values = np.arange(31, dtype=np.float32)
    dataset = xr.concat(
        [cube.dataset.isel(member=0, drop=True) + value for value in member_values],
        dim=xr.IndexVariable("member", [f"p{index:02d}" for index in range(31)]),
    )
    dataset.attrs["source_member_count"] = 31
    ensemble = SourceCube("gefs", cube.cycle_time, dataset, tuple(dataset.member.values))

    summary = summarize_ensemble_for_public(ensemble)

    assert summary.dataset.sizes["member"] == 2
    assert source_member_count(summary) == 31
    for name in REQUIRED_VARIABLES:
        original = ensemble.dataset[name]
        reduced = summary.dataset[name]
        xr.testing.assert_allclose(reduced.mean("member"), original.mean("member"))
        xr.testing.assert_allclose(reduced.std("member"), original.std("member"))


def test_lead_weights_are_renormalized_when_optional_source_is_missing():
    blended = blend_sources({"hrrr": source_cube("hrrr", 10), "refs": source_cube("refs", 40), "rrfs": source_cube("rrfs", 20)})
    assert float(blended.temperature_2m.isel(time=0, y=0, x=0)) == pytest.approx(17.0)
    assert float(blended.temperature_2m.isel(time=50, y=0, x=0)) == pytest.approx(25.0)

    degraded = blend_sources({"hrrr": source_cube("hrrr", 10), "refs": source_cube("refs", 40)})
    assert float(degraded.temperature_2m.isel(time=0, y=0, x=0)) == pytest.approx(15.0)
    assert np.isnan(float(degraded.temperature_2m.isel(time=50, y=0, x=0)))


def test_gefs_mean_is_degraded_fallback_only_after_hrrr_horizon():
    hrrr = source_cube("hrrr", 10, hours=49)
    gefs = source_cube("gefs", 30)
    blended = blend_sources({"hrrr": hrrr, "gefs": gefs})
    assert float(blended.temperature_2m.isel(time=48, y=0, x=0)) == pytest.approx(10)
    assert float(blended.temperature_2m.isel(time=49, y=0, x=0)) == pytest.approx(30)
    fallback_mask = int(blended.quality_mask.isel(time=49, y=0, x=0))
    assert fallback_mask & QUALITY_BITS["coarse_synoptic_fallback"]

    forecast, _ = build_forecast_cube({"hrrr": hrrr, "gefs": gefs}, initial_fuel_moisture=10)
    assert int(forecast.category_confidence.isel(time=49, y=0, x=0)) <= 49
    assert not int(forecast.quality_mask.isel(time=48, y=0, x=0)) & QUALITY_BITS["coarse_synoptic_fallback"]


def test_corrections_respect_variable_and_lead_caps():
    blended = blend_sources({"hrrr": source_cube("hrrr", 10), "rrfs": source_cube("rrfs", 10)})
    residual = xr.full_like(blended.temperature_2m, 20.0)
    corrected = apply_corrections(blended, {"temperature_2m": residual}).dataset
    assert float(corrected.temperature_2m.isel(time=12, y=0, x=0)) == pytest.approx(12.8)
    assert float(corrected.temperature_2m.isel(time=36, y=0, x=0)) == pytest.approx(12.1)
    assert float(corrected.temperature_2m.isel(time=60, y=0, x=0)) == pytest.approx(11.4)


def test_dst_local_day_boundaries_keep_23_hour_day():
    cycle = datetime(2026, 3, 7, 12, tzinfo=timezone.utc)
    times = xr.DataArray(np.array([np.datetime64(cycle.replace(tzinfo=None) + timedelta(hours=i), "ns") for i in range(73)]), dims="time")
    days = local_day_slices(times)
    assert [len(indexes) for _, indexes in days] == [18, 23, 24]


def test_adapter_converts_kelvin_fraction_and_cumulative_precip_resets():
    cube = source_cube("hrrr", 10, hours=4)
    dataset = cube.dataset.drop_vars(list(cube.dataset.data_vars))
    for name in REQUIRED_VARIABLES:
        base = np.ones((1, 4, 2, 3), dtype=np.float32)
        attrs = {}
        if name == "temperature_2m": base *= 300; attrs["units"] = "K"
        elif name in {"relative_humidity_2m", "cloud_cover"}: base *= .5; attrs["units"] = "1"
        elif name == "precipitation_increment":
            base[:] = np.array([0, 1, 3, .5], dtype=np.float32)[None, :, None, None]
            attrs = {"units": "mm", "accumulation_semantics": "cumulative"}
        dataset[name] = xr.DataArray(base, dims=("member", "time", "y", "x"), coords=cube.dataset.coords, attrs=attrs)
    normalized = DatasetAdapter("hrrr").normalize(dataset, cube.cycle_time).dataset
    assert float(normalized.temperature_2m.isel(member=0, time=0, y=0, x=0)) == pytest.approx(26.85, abs=.01)
    assert float(normalized.relative_humidity_2m.isel(member=0, time=0, y=0, x=0)) == 50
    assert normalized.precipitation_increment.isel(member=0, y=0, x=0).values.tolist() == pytest.approx([0, 1, 2, .5])


def test_synthetic_73_hour_cube_preserves_quantile_order_and_unavailable_cells():
    hrrr = source_cube("hrrr", 25)
    rrfs = source_cube("rrfs", 25)
    hrrr.dataset["temperature_2m"][:, 10, 0, 0] = np.nan
    rrfs.dataset["temperature_2m"][:, 10, 0, 0] = np.nan
    forecast, daily = build_forecast_cube({"hrrr": hrrr, "rrfs": rrfs}, 12.0)
    assert forecast.sizes["time"] == 73
    assert daily.sizes["day"] == 3
    assert float((forecast.fuel_moisture_p10 - forecast.fuel_moisture_p50).max(skipna=True)) <= 0
    assert float((forecast.fuel_moisture_p50 - forecast.fuel_moisture_p90).max(skipna=True)) <= 0
    assert int(forecast.fire_danger.isel(time=10, y=0, x=0)) == 255


def test_storage_schema_and_packed_netcdf_roundtrip(tmp_path):
    db = tmp_path / "forecast.db"
    ensure_schema(db)
    with sqlite3.connect(db) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"forecast_runs", "forecast_assets", "station_forecast_hours", "forecast_verification"} <= tables

    source = source_cube("hrrr", 22, hours=2).dataset.isel(member=0)
    path = write_netcdf(source, tmp_path / "packed.nc")
    with xr.open_dataset(path) as reopened:
        assert float(reopened.temperature_2m.isel(time=0, y=0, x=0)) == pytest.approx(22, abs=.01)


def test_multiband_cog_contains_band_metadata_and_category_overviews(tmp_path):
    values = xr.DataArray(
        np.zeros((3, PUBLIC_GRID.height, PUBLIC_GRID.width), dtype=np.uint8),
        dims=("time", "y", "x"), attrs={"units": "category"},
    )
    path = write_cog(values, tmp_path / "danger.tif", variable="fire_danger", band_times=["a", "b", "c"], categorical=True)
    import rasterio
    with rasterio.open(path) as source:
        assert source.count == 3
        assert source.tags(2)["lead_hour"] == "1"
        assert source.overviews(1)


def test_station_parquet_is_zstd_dictionary_encoded_and_sorted(tmp_path):
    path = write_points_parquet([
        {"station_id": "B", "model": "rrfs", "member": "2", "valid_time_utc": "2026-09-06T13:00:00Z", "temperature_c": 20.0},
        {"station_id": "A", "model": "hrrr", "member": "control", "valid_time_utc": "2026-09-06T12:00:00Z", "temperature_c": None},
    ], tmp_path / "points.parquet")
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    assert table.column("station_id").to_pylist() == ["A", "B"]
    assert table.column("temperature_c").to_pylist()[0] is None
    assert {pq.ParquetFile(path).metadata.row_group(0).column(i).compression for i in range(table.num_columns)} == {"ZSTD"}


def test_recent_qc_raws_fuel_initialization_overrides_spatial_analysis():
    cube = blend_sources({"hrrr": source_cube("hrrr", 20), "rrfs": source_cube("rrfs", 20)})
    cube = cube.assign_coords(x=[500_500.0, 503_500.0, 506_500.0], y=[4_000_500.0, 3_997_500.0])
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(PUBLIC_GRID.crs, "EPSG:4326", always_xy=True).transform(500_500, 4_000_500)
    result = initialize_fuel_moisture(
        cube, datetime(2026, 9, 6, 12, tzinfo=timezone.utc),
        [{"network": "RAWS", "qc_state": "pass", "observed_at_utc": "2026-09-06T10:00:00Z", "fuel_moisture": 7.5, "longitude": lon, "latitude": lat}],
        xr.full_like(cube.temperature_2m.isel(time=0), 14.0), station_radius_m=2_000,
    )
    assert float(result.values.isel(y=0, x=0)) == pytest.approx(7.5)
    assert float(result.values.isel(y=1, x=2)) == pytest.approx(14.0)


def test_station_api_uses_null_safe_nested_contract(tmp_path, monkeypatch):
    db = tmp_path / "api.db"
    ensure_schema(db)
    monkeypatch.setattr("forecast_v1.repository.get_db_path", lambda: db)
    with sqlite3.connect(db) as connection:
        connection.execute("INSERT INTO forecast_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "20260906T1200Z-v1", "2026-09-06T12:00:00Z", "2026-09-06T13:00:00Z", "2026-09-06T13:01:00Z",
            "complete", 1, PUBLIC_GRID.id, 72, "forecast-v1", "beta-1", "physics-1", "{}", None, None, "[]", None, "2026-09-06T13:00:00Z",
        ))
        connection.execute("INSERT INTO forecast_station_metadata VALUES (?,?,?,?,?,?,?,?,?,?)", (
            "TEST", "Test RAWS", 38.5, -92.5, "RAWS", 250.0, "America/Chicago", 1, '{"fuelMoisture":true}', "2026-09-06T13:00:00Z",
        ))
        columns = [row[1] for row in connection.execute("PRAGMA table_info(station_forecast_hours)")]
        row = {name: None for name in columns}
        row.update(run_id="20260906T1200Z-v1", station_id="TEST", valid_time_utc="2026-09-06T12:00:00Z", lead_hour=0, temperature_c=25.0, relative_humidity=24.0, wind_speed_ms=8.0, wind_gust_ms=14.0, fuel_moisture_p10=5.0, fuel_moisture_p50=7.0, fuel_moisture_p90=9.0, danger_class=3, meteorological_confidence=None, category_confidence=82, probability_rh_le_25=70, probability_gust_ge_30mph=60, probability_concurrent=55, source_mask=3, quality_mask=32)
        connection.execute(f"INSERT INTO station_forecast_hours ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", [row[name] for name in columns])
    from routers.forecast_v1 import router
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/forecast/v1/stations/TEST")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hours"][0]["confidence"] == {"meteorological": None, "category": 82}
    assert payload["hours"][0]["risk"]["label"] == "Critical"
    assert "insufficient_confidence" in payload["hours"][0]["qualityFlags"]


def test_forecast_admin_status_and_run_controls_require_admin(tmp_path, monkeypatch):
    from routers import forecast_v1_admin

    db = tmp_path / "admin.db"
    ensure_schema(db)
    monkeypatch.setattr("forecast_v1.repository.get_db_path", lambda: db)
    app = FastAPI()
    app.include_router(forecast_v1_admin.router)
    client = TestClient(app)

    monkeypatch.setattr(forecast_v1_admin, "verify_token", lambda token=None: None)
    assert client.get("/api/admin/forecast-v1/status").status_code == 401
    assert client.get("/api/admin/forecast-v1/job").status_code == 401
    assert client.post("/api/admin/forecast-v1/run").status_code == 401

    monkeypatch.setattr(forecast_v1_admin, "verify_token", lambda token=None: "admin@example.org")
    monkeypatch.setattr(forecast_v1_admin, "get_forecast_v1_job_status", lambda: {"status": "idle"})
    monkeypatch.setattr(forecast_v1_admin, "trigger_forecast_v1", lambda email: {"status": "queued", "requested_by": email})
    status_response = client.get("/api/admin/forecast-v1/status")
    assert status_response.status_code == 200
    assert status_response.json()["schedule"]["policy"].startswith("HRRR through hour 48")
    assert client.get("/api/admin/forecast-v1/job").json() == {"status": "idle"}
    run_response = client.post("/api/admin/forecast-v1/run")
    assert run_response.status_code == 202
    assert run_response.json()["requested_by"] == "admin@example.org"


def test_forecast_worker_memory_limit_is_reversible(monkeypatch):
    import resource
    from services.forecast_v1_job import _run_with_memory_limit

    previous = resource.getrlimit(resource.RLIMIT_AS)
    monkeypatch.setenv("SMF_FORECAST_V1_MEMORY_LIMIT_GB", "8")
    assert _run_with_memory_limit(lambda: "completed") == "completed"
    assert resource.getrlimit(resource.RLIMIT_AS) == previous


def test_forecast_job_progress_records_phase_details_and_history(tmp_path, monkeypatch):
    from services import forecast_v1_job

    monkeypatch.setattr(forecast_v1_job, "JOB_STATE_PATH", tmp_path / "admin-job.json")
    forecast_v1_job._write_job({"job_id": "test", "status": "queued", "requested_at": "2026-09-07T12:00:00Z"})
    forecast_v1_job._report_progress(
        "acquiring_sources", "Downloading GEFS member p01 (2/31)", 42,
        model="GEFS", current=2, total=31,
    )
    status = forecast_v1_job.get_forecast_v1_job_status()

    assert status["phase_label"] == "Downloading forecast sources"
    assert status["progress_percent"] == 42
    assert status["progress"] == {"model": "GEFS", "current": 2, "total": 31}
    assert status["events"][-1]["message"] == "Downloading GEFS member p01 (2/31)"
    assert status["memory_limit_gb"] == 8


def test_rrfs_refs_promotion_requires_every_documented_gate():
    metric = {
        "complete": True, "cycle_hour": 12, "required_hour_availability": .97,
        "mae": {"temperature": .9, "relative_humidity": .9, "wind": .9, "fuel_moisture": .9},
        "baseline_mae": {"temperature": 1, "relative_humidity": 1, "wind": 1, "fuel_moisture": 1},
        "elevated_plus_mae": {"relative_humidity": .9, "wind": .9, "fuel_moisture": .9},
        "baseline_elevated_plus_mae": {"relative_humidity": 1, "wind": 1, "fuel_moisture": 1},
        "brier_skill": .05, "max_false_negative_increase": 1,
    }
    assert evaluate_rrfs_refs_promotion([metric.copy() for _ in range(30)]).eligible
    failing = [metric.copy() for _ in range(30)]
    failing[-1] = {**failing[-1], "required_hour_availability": 0}
    decision = evaluate_rrfs_refs_promotion(failing)
    assert not decision.eligible
    assert not decision.gates["requiredHourAvailabilityAtLeast95Percent"]


def test_vectorized_public_categories_match_authoritative_rule_contract():
    rng = np.random.default_rng(7)
    fm = xr.DataArray(rng.uniform(1, 25, 500), dims="point")
    rh = xr.DataArray(rng.uniform(5, 100, 500), dims="point")
    wind = xr.DataArray(rng.uniform(0, 20, 500), dims="point")
    actual = _classify_arrays(fm, rh, wind).values
    expected = np.array([calculate_fire_danger(float(f), float(r), float(w) * MPS_TO_KNOTS) for f, r, w in zip(fm.values, rh.values, wind.values)], dtype=np.uint8)
    assert np.array_equal(actual, expected)


def test_publication_failure_keeps_previous_run_public(tmp_path):
    db = tmp_path / "failure.db"
    ensure_schema(db)
    with sqlite3.connect(db) as connection:
        connection.execute("INSERT INTO forecast_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            "previous-v1", "2026-09-05T12:00:00Z", "2026-09-05T13:00:00Z", "2026-09-05T13:01:00Z",
            "complete", 1, PUBLIC_GRID.id, 72, "forecast-v1", "beta-1", "physics-1", "{}", None, None, "[]", None, "2026-09-05T13:00:00Z",
        ))
    zero_cycle = datetime(2026, 9, 6, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="12Z"):
        publish_run(
            {"hrrr": source_cube("hrrr", 20, cycle=zero_cycle), "rrfs": source_cube("rrfs", 20, cycle=zero_cycle)},
            [], initial_fuel_moisture=12.0, publish_root=tmp_path / "products", db_path=db, make_public=True,
        )
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT is_public FROM forecast_runs WHERE run_id='previous-v1'").fetchone()[0] == 1
        assert connection.execute("SELECT status FROM forecast_runs WHERE run_id='20260906T0000Z-v1'").fetchone()[0] == "failed"


def test_synthetic_publication_writes_73_hour_three_day_contract(tmp_path, monkeypatch):
    import forecast_v1.artifacts as artifacts_module
    import forecast_v1.pipeline as pipeline_module

    grid = GridDefinition("test-grid", "EPSG:32615", 3, 2, 3000, 499_000, 4_002_000)
    monkeypatch.setattr(artifacts_module, "PUBLIC_GRID", grid)
    monkeypatch.setattr(pipeline_module, "PUBLIC_GRID", grid)

    def fake_parquet(rows, path):
        target = tmp_path / Path(path).relative_to(tmp_path) if not Path(path).is_absolute() else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"PAR1-test")
        return target

    def fake_graphic(data, png_path, webp_path, **kwargs):
        png, webp = Path(png_path), Path(webp_path)
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"png")
        webp.write_bytes(b"webp")
        return png, webp

    from pathlib import Path
    monkeypatch.setattr(pipeline_module, "write_points_parquet", fake_parquet)
    monkeypatch.setattr(pipeline_module, "write_static_graphic", fake_graphic)
    db = tmp_path / "published.db"
    monkeypatch.setattr("forecast_v1.repository.get_db_path", lambda: db)
    class OfflineArchive:
        configured = False
    manifest = publish_run(
        {"hrrr": source_cube("hrrr", 20), "rrfs": source_cube("rrfs", 22), "refs": source_cube("refs", 24)},
        [{"station_id": "TEST", "name": "Test", "latitude": 36.2, "longitude": -93.0, "network_type": "RAWS"}],
        initial_fuel_moisture=10.0, publish_root=tmp_path / "products", db_path=db, make_public=True, r2_store=OfflineArchive(),
    )
    assert manifest["timeCount"] == 73
    assert len(manifest["staticAssets"]) == 32
    assert {asset["day"] for asset in manifest["staticAssets"]} == {2, 3}
    assert manifest["legacyAliases"] == {}
    hourly_asset = next(layer for layer in manifest["layers"] if layer["variable"] == "fire_danger" and layer["aggregation"] == "hourly")
    assert "{lead_hour}" in hourly_asset["tileUrl"]
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM station_forecast_hours").fetchone()[0] == 73
        assert connection.execute("SELECT COUNT(*) FROM station_forecast_days").fetchone()[0] == 3
        assert connection.execute("SELECT is_public FROM forecast_runs WHERE run_id=?", (manifest["runId"],)).fetchone()[0] == 1
    from routers.tiles import _forecast_tile_sync
    tile = _forecast_tile_sync(manifest["runId"], "fire_danger", 36, 6, 15, 24)
    assert tile.status_code == 200
    assert tile.media_type == "image/png"

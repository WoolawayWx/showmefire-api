import sqlite3
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyproj
import pytest
import xarray as xr

from forecast_v1.acquisition import AcquisitionSpec, SourceDiagnostic, _clip_and_project_axes, _sanitized_error, acquire_cycle, acquire_source
from forecast_v1.adapters import SourceCube
from forecast_v1.confidence import build_confidence_inputs, cycle_consistency, rolling_verification
from forecast_v1.contracts import GridDefinition, QUALITY_BITS, REQUIRED_VARIABLES
from forecast_v1.engine import add_risk_and_confidence, blend_sources, build_forecast_cube
from forecast_v1.observations import persist_station_observations, update_forecast_verification
from forecast_v1.repository import ensure_schema


CYCLE = datetime(2026, 9, 6, 12, tzinfo=timezone.utc)


def test_pinned_herbie_distribution_and_operational_template():
    api_root = Path(__file__).resolve().parents[1]
    for requirements in ("requirements.txt", "requirements.lock.txt"):
        lines = (api_root / requirements).read_text(encoding="utf-8").splitlines()
        assert "herbie-data==2025.12.0" in lines
        assert not any(line.startswith("herbie==") for line in lines)
    dockerfile = (api_root / "Dockerfile").read_text(encoding="utf-8")
    assert 'version("herbie-data") == "2025.12.0"' in dockerfile
    assert 'version("herbie")' in dockerfile
    assert "rotated_latitude_longitude" in dockerfile
    template = runpy.run_path(str(api_root / "patches" / "rrfs.py"))["rrfs"]
    probe = SimpleNamespace(product="2dfld.13km", date=CYCLE, fxx=1, get_remoteFileName="probe.grib2")
    template.template(probe)
    assert "noaa-rrfs-ops-pds" in probe.SOURCES["aws"]


def cube(model: str, value: float, members: int = 1, hours: int = 73) -> SourceCube:
    coords = {
        "member": [f"m{i}" for i in range(members)],
        "time": [np.datetime64(CYCLE.replace(tzinfo=None) + timedelta(hours=i), "ns") for i in range(hours)],
        "y": [4_000_500.0], "x": [500_500.0],
    }
    shape = (members, hours, 1, 1)
    variables = {}
    for name in REQUIRED_VARIABLES:
        current = 30.0 if name == "relative_humidity_2m" else 4.0 if name in {"wind_u_10m", "wind_v_10m"} else 8.0 if name == "wind_gust_10m" else value
        data = np.full(shape, current, dtype=np.float32)
        if members > 1:
            data += np.arange(members, dtype=np.float32)[:, None, None, None]
        variables[name] = (("member", "time", "y", "x"), data)
    return SourceCube(model, CYCLE, xr.Dataset(variables, coords=coords), tuple(coords["member"]))


def test_confidence_bootstraps_from_agreement_and_spread_and_marks_partial(tmp_path):
    cubes = {"hrrr": cube("hrrr", 20), "rrfs": cube("rrfs", 21), "refs": cube("refs", 20, members=3)}
    atmosphere = blend_sources(cubes)
    inputs = build_confidence_inputs(cubes, atmosphere, CYCLE, db_path=tmp_path / "empty.db")
    assert {"model_agreement", "ensemble_spread"} <= set(inputs.available_components)
    forecast, _ = build_forecast_cube(cubes, 10.0, confidence_components=inputs.components)
    assert int(forecast.meteorological_confidence.isel(time=0, y=0, x=0)) != 255
    assert int(forecast.quality_mask.isel(time=0, y=0, x=0)) & QUALITY_BITS["partial_confidence_inputs"]


def test_model_agreement_requires_two_valid_sources_at_each_cell(tmp_path):
    hrrr = cube("hrrr", 20)
    rrfs = cube("rrfs", 21)
    rrfs.dataset["temperature_2m"][:, 0, 0, 0] = np.nan
    for variable in ("relative_humidity_2m", "wind_u_10m", "wind_v_10m", "wind_gust_10m"):
        rrfs.dataset[variable][:, 0, 0, 0] = np.nan
    cubes = {"hrrr": hrrr, "rrfs": rrfs}
    agreement = build_confidence_inputs(cubes, blend_sources(cubes), CYCLE, db_path=tmp_path / "empty.db").components["model_agreement"]
    assert np.isnan(float(agreement.isel(time=0, y=0, x=0)))


def test_observed_rrfs_rotation_clips_and_reprojects_over_missouri(monkeypatch):
    import forecast_v1.pipeline as pipeline

    rotation = pyproj.CRS.from_proj4(
        "+proj=ob_tran +o_proj=longlat +lon_0=247 +o_lon_p=0 +o_lat_p=35 +R=6371229 +no_defs"
    )
    to_rotated = pyproj.Transformer.from_crs("EPSG:4326", rotation, always_xy=True)
    to_geographic = pyproj.Transformer.from_crs(rotation, "EPSG:4326", always_xy=True)
    center_x, center_y = to_rotated.transform(-92.5, 38.5)
    rotated_x, rotated_y = np.meshgrid(center_x + np.array([-1.0, 0.0, 1.0]), center_y + np.array([-1.0, 0.0, 1.0]))
    longitude, latitude = to_geographic.transform(rotated_x, rotated_y)
    raw = xr.Dataset(
        {"t2m": (("y", "x"), np.full((3, 3), 293.15), {
            "GRIB_shapeOfTheEarth": 6, "GRIB_gridType": "rotated_ll",
            "GRIB_longitudeOfSouthernPoleInDegrees": 247.0,
            "GRIB_latitudeOfSouthernPoleInDegrees": -35.0,
        })},
        coords={"latitude": (("y", "x"), latitude), "longitude": (("y", "x"), longitude)},
    )
    clipped = _clip_and_project_axes(raw)
    assert clipped.sizes["x"] >= 2 and clipped.sizes["y"] >= 2
    assert pyproj.CRS(clipped.attrs["crs"]).to_proj4().find("ob_tran") >= 0
    utm_x, utm_y = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True).transform(-92.5, 38.5)
    test_grid = GridDefinition("missouri-test", "EPSG:32615", 3, 3, 3000, utm_x - 4500, utm_y + 4500)
    monkeypatch.setattr(pipeline, "PUBLIC_GRID", test_grid)
    dataset = xr.Dataset(
        {name: clipped.t2m.expand_dims(member=["control"], time=[np.datetime64(CYCLE.replace(tzinfo=None))])
         for name in REQUIRED_VARIABLES}, attrs=clipped.attrs,
    )
    source = SourceCube("rrfs", CYCLE, dataset, ("control",))
    regridded = pipeline.regrid_to_public(source)
    assert np.isfinite(regridded.dataset.temperature_2m.values).any()


def test_confidence_renormalizes_and_requires_realtime_component():
    current = blend_sources({"hrrr": cube("hrrr", 20), "rrfs": cube("rrfs", 20)})
    current["fuel_moisture_p10"] = xr.full_like(current.temperature_2m, 8.0)
    current["fuel_moisture_p50"] = xr.full_like(current.temperature_2m, 10.0)
    current["fuel_moisture_p90"] = xr.full_like(current.temperature_2m, 12.0)
    agreement = xr.full_like(current.temperature_2m, 80.0)
    result = add_risk_and_confidence(current, {"model_agreement": agreement})
    assert int(result.meteorological_confidence.isel(time=0, y=0, x=0)) == 85
    unsupported = add_risk_and_confidence(current, {"cycle_consistency": xr.full_like(agreement, 100.0)})
    assert int(unsupported.meteorological_confidence.isel(time=0, y=0, x=0)) == 255
    assert int(unsupported.quality_mask.isel(time=0, y=0, x=0)) & QUALITY_BITS["insufficient_confidence"]


def test_gefs_fallback_caps_real_meteorological_confidence():
    cubes = {"hrrr": cube("hrrr", 20, hours=49), "gefs": cube("gefs", 20, members=3)}
    confidence = {"ensemble_spread": xr.full_like(blend_sources(cubes).temperature_2m, 100.0)}
    forecast, _ = build_forecast_cube(cubes, 10.0, confidence_components=confidence)
    assert int(forecast.meteorological_confidence.isel(time=49, y=0, x=0)) == 49


def test_source_diagnostic_contract_is_additive_and_null_safe():
    diagnostic = SourceDiagnostic("fv3hires", "shadow", "degraded", "info", "field_completeness", "missing", ("cloud_cover",))
    assert diagnostic.as_dict()["missingFields"] == ["cloud_cover"]
    assert diagnostic.as_dict()["affectedLeadHours"] == []
    assert "secret" not in _sanitized_error("failed token=secret Authorization: Bearer secret")


def test_existing_run_database_gains_empty_diagnostics(tmp_path):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TABLE forecast_runs (
            run_id TEXT PRIMARY KEY, cycle_time_utc TEXT, issued_at_utc TEXT, completed_at_utc TEXT,
            status TEXT, is_public INTEGER, grid_id TEXT, horizon_hours INTEGER, schema_version TEXT,
            config_version TEXT, model_version TEXT, source_cycles_json TEXT, manifest_key TEXT,
            manifest_checksum TEXT, warnings_json TEXT, superseded_run_id TEXT, created_at_utc TEXT
        )""")
        connection.execute("""INSERT INTO forecast_runs
            (run_id,cycle_time_utc,status,is_public,grid_id,horizon_hours,schema_version,config_version,
             model_version,source_cycles_json,warnings_json,created_at_utc)
            VALUES ('older','2026-09-05T12:00:00Z','complete',0,'grid',72,'v1','config','model','{}','[]',
                    '2026-09-05T12:00:00Z')""")
    ensure_schema(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT source_diagnostics_json FROM forecast_runs WHERE run_id='older'").fetchone()[0] == "[]"


def test_cycle_consistency_uses_matching_valid_times():
    current = blend_sources({"hrrr": cube("hrrr", 20), "rrfs": cube("rrfs", 20)})
    previous = current.copy(deep=True)
    previous = previous.assign_coords(time=previous.time + np.timedelta64(24, "h"))
    score = cycle_consistency(current, previous)
    assert np.isnan(float(score.isel(time=0, y=0, x=0)))
    assert float(score.isel(time=24, y=0, x=0)) == pytest.approx(100.0)


def test_rolling_verification_requires_30_pairs_and_three_stations(tmp_path):
    db = tmp_path / "history.db"
    ensure_schema(db)
    template = cube("hrrr", 20).dataset.temperature_2m.isel(member=0)
    rows = []
    for index in range(30):
        valid = (CYCLE - timedelta(hours=30 - index)).isoformat().replace("+00:00", "Z")
        rows.append(("prior", f"S{index % 3}", valid, "temperature_2m", 21.0, 20.0, 1.0,
                     "0-24", "model", "config", 1))
    with sqlite3.connect(db) as connection:
        connection.executemany(
            """INSERT INTO forecast_verification
               (run_id,station_id,valid_time_utc,variable,forecast_value,observation_value,error,
                lead_bucket,model_version,config_version,qc_eligible) VALUES (?,?,?,?,?,?,?,?,?,?,?)""", rows[:-1],
        )
    assert np.isnan(float(rolling_verification(template, CYCLE, db).isel(time=0, y=0, x=0)))
    with sqlite3.connect(db) as connection:
        connection.execute(
            """INSERT INTO forecast_verification
               (run_id,station_id,valid_time_utc,variable,forecast_value,observation_value,error,
                lead_bucket,model_version,config_version,qc_eligible) VALUES (?,?,?,?,?,?,?,?,?,?,?)""", rows[-1],
        )
    assert float(rolling_verification(template, CYCLE, db).isel(time=0, y=0, x=0)) == pytest.approx(
        100.0 * np.exp(-0.5 * (1.0 / 3.0) ** 2)
    )


def test_rrfs_f000_missing_apcp_is_treated_as_zero(tmp_path, monkeypatch):
    coords = {
        "step": np.array([0, 1], dtype="timedelta64[h]"), "time": np.datetime64(CYCLE.replace(tzinfo=None)),
        "latitude": ("y", [36.0]), "longitude": ("x", [-93.0]),
    }
    shape = (2, 1, 1)
    variables = {
        "t2m": (("step", "y", "x"), np.full(shape, 293.15), {"units": "K"}),
        "r2": (("step", "y", "x"), np.full(shape, 40.0), {"units": "%"}),
        "u10": (("step", "y", "x"), np.full(shape, 3.0), {"units": "m s-1"}),
        "v10": (("step", "y", "x"), np.full(shape, 2.0), {"units": "m s-1"}),
        "gust": (("step", "y", "x"), np.full(shape, 6.0), {"units": "m s-1"}),
        "tp": (("step", "y", "x"), np.array([np.nan, 1.0]).reshape(shape), {"units": "mm"}),
        "dswrf": (("step", "y", "x"), np.full(shape, 400.0), {"units": "W m-2"}),
        "tcc": (("step", "y", "x"), np.full(shape, 25.0), {"units": "%"}),
        "hpbl": (("step", "y", "x"), np.full(shape, 1000.0), {"units": "m"}),
        "soilw": (("step", "y", "x"), np.full(shape, 0.2), {"units": "1"}),
        "weasd": (("step", "y", "x"), np.zeros(shape), {"units": "mm"}),
    }
    raw = xr.Dataset(variables, coords=coords)

    class Herbie:
        def __init__(self, **kwargs): pass
        def xarray(self, *args, **kwargs): return raw

    result = acquire_source(AcquisitionSpec("rrfs", "rrfs", "prslev", (0, 1), ("control",)), CYCLE, tmp_path, fast_herbie_factory=Herbie)
    assert result.dataset.precipitation_increment.isel(member=0, y=0, x=0).values.tolist() == pytest.approx([0.0, 1.0])


def test_acquisition_warnings_are_operational_and_errors_remain_actionable(monkeypatch, tmp_path):
    specs = (
        AcquisitionSpec("hrrr", "hrrr", "sfc", (0,), (None,), required=True),
        AcquisitionSpec("rrfs", "rrfs", "prslev", tuple(range(73)), ("control",)),
        AcquisitionSpec("gefs", "gefs", "atmos.5", tuple(range(73)), (0,), role="fallback"),
        AcquisitionSpec("fv3hires", "fv3", "sfc", (0,), (None,), role="shadow"),
    )

    def fake_acquire(spec, *args, **kwargs):
        if spec.public_name == "rrfs":
            raise ValueError("rotated_ll conversion failed at /cache/member.grib2")
        if spec.public_name == "fv3hires":
            raise RuntimeError("shadow field missing")
        return cube(spec.public_name, 20)

    monkeypatch.setattr("forecast_v1.acquisition.acquire_source", fake_acquire)
    monkeypatch.setattr("forecast_v1.registry.mark_acquired", lambda model: None)
    result = acquire_cycle(CYCLE, tmp_path, specs=specs)
    assert result.warnings == ("source_unavailable:rrfs:ValueError", "coarse_synoptic_fallback:gefs:49-72")
    rrfs = next(item for item in result.diagnostics if item.source == "rrfs")
    shadow = next(item for item in result.diagnostics if item.source == "fv3hires")
    assert "rotated_ll conversion failed" in rrfs.message
    assert rrfs.affected_lead_hours == (0, 72)
    assert shadow.severity == "info"


def test_shadow_missing_fields_never_become_run_warnings(monkeypatch, tmp_path):
    specs = (
        AcquisitionSpec("hrrr", "hrrr", "sfc", (0,), (None,), required=True),
        AcquisitionSpec("rrfs", "rrfs", "prslev", tuple(range(73)), ("control",)),
        AcquisitionSpec("fv3hires", "fv3", "sfc", (0,), (None,), role="shadow"),
    )

    def fake_acquire(spec, *args, **kwargs):
        source = cube(spec.public_name, 20)
        if spec.role == "shadow":
            return SourceCube(source.model, source.cycle_time, source.dataset, source.member_ids,
                              ("required_field_unavailable:cloud_cover", "optional_field_missing:temperature_700hpa"))
        return source

    monkeypatch.setattr("forecast_v1.acquisition.acquire_source", fake_acquire)
    monkeypatch.setattr("forecast_v1.registry.mark_acquired", lambda model: None)
    result = acquire_cycle(CYCLE, tmp_path, specs=specs)
    assert result.warnings == ()
    diagnostic = next(item for item in result.diagnostics if item.source == "fv3hires")
    assert diagnostic.severity == "info" and diagnostic.code == "field_completeness"
    assert set(diagnostic.missing_fields) == {"cloud_cover", "temperature_700hpa"}


def test_synoptic_observations_normalize_and_match_completed_forecast(tmp_path):
    db = tmp_path / "verification.db"
    ensure_schema(db)
    with sqlite3.connect(db) as connection:
        connection.execute(
            """INSERT INTO forecast_runs
               (run_id,cycle_time_utc,status,is_public,grid_id,horizon_hours,schema_version,config_version,
                model_version,source_cycles_json,warnings_json,created_at_utc,source_diagnostics_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("run", "2026-09-06T12:00:00Z", "complete", 0, "grid", 72, "v1", "config", "model", "{}", "[]", "2026-09-06T12:00:00Z", "[]"),
        )
        connection.execute(
            """INSERT INTO station_forecast_hours
               (run_id,station_id,valid_time_utc,lead_hour,temperature_c,relative_humidity,wind_speed_ms,wind_gust_ms)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("run", "TEST", "2026-09-06T13:00:00Z", 1, 20.0, 40.0, 4.0, 6.0),
        )
    station = {"stid": "TEST", "network": "RAWS", "qc_flagged": False, "observations": {
        "air_temp": {"value": 68.0, "time": "2026-09-06T13:10:00Z"},
        "relative_humidity": {"value": 42.0, "time": "2026-09-06T13:20:00Z"},
        "wind_speed": {"value": 10.0, "time": "2026-09-06T13:10:00Z"},
        "wind_gust": {"value": 15.0, "time": "2026-09-06T13:10:00Z"},
    }}
    assert persist_station_observations([station], db) == 2
    matched = update_forecast_verification(db, now=datetime(2026, 9, 6, 14, tzinfo=timezone.utc))
    assert matched == 4
    with sqlite3.connect(db) as connection:
        row = connection.execute("SELECT temperature_c,wind_speed_ms FROM station_observations_v2").fetchone()
        assert row[0] == pytest.approx(20.0)
        assert row[1] == pytest.approx(4.4704)
        assert connection.execute("SELECT COUNT(*) FROM forecast_verification WHERE qc_eligible=1").fetchone()[0] == 4

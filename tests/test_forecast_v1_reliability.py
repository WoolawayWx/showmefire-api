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


def _lead_dataset(leads):
    """What FastHerbie.xarray really returns (checked against HRRR 2026-10-20 12Z):
    several leads give a `step` dimension with `valid_time` along it; a single
    lead is squeezed, so `step` and `valid_time` are scalars and there is no
    time-like dimension at all."""
    init = np.datetime64(CYCLE.replace(tzinfo=None), "ns")
    steps = np.array([lead * 3600 for lead in leads], dtype="timedelta64[s]").astype("timedelta64[ns]")
    spatial = {"latitude": ("y", [36.0, 37.0]), "longitude": ("x", [-94.0, -93.0])}
    if len(leads) == 1:
        coords = {"time": init, "step": steps[0], "valid_time": init + steps[0], **spatial}
        return xr.Dataset({"tp": (("y", "x"), np.ones((2, 2), dtype=np.float32))}, coords=coords)
    coords = {"time": init, "step": ("step", steps), "valid_time": ("step", init + steps), **spatial}
    return xr.Dataset({"tp": (("step", "y", "x"), np.ones((len(leads), 2, 2), dtype=np.float32))}, coords=coords)


def test_query_leads_chunks_and_skips_hours_without_the_field(monkeypatch):
    from forecast_v1.acquisition import _query_leads

    monkeypatch.setenv("SMF_HERBIE_LEAD_CHUNK", "3")
    requested = []

    def make_client(leads):
        requested.append(tuple(leads))

        class Client:
            def xarray(self, search, **kwargs):
                if 0 in leads:  # field is not published at f000 -> Herbie leaves no subset file
                    raise FileNotFoundError("subset_x__f000")
                return _lead_dataset(leads)
        return Client()

    result = _query_leads(make_client, tuple(range(7)), ":APCP:surface:", CYCLE)

    assert result.sizes["time"] == 6  # f001..f006; f000 skipped instead of failing the query
    assert max(len(batch) for batch in requested) == 3  # never asked for all hours at once


def test_query_leads_does_not_ask_again_for_hours_already_found_missing(monkeypatch):
    # Each GEFS member used to rediscover that f000 has no APCP by failing a chunk and retrying it hour by hour.
    from forecast_v1.acquisition import _query_leads

    monkeypatch.setenv("SMF_HERBIE_LEAD_CHUNK", "3")
    requested = []

    def make_client(leads):
        requested.append(tuple(leads))

        class Client:
            def xarray(self, search, **kwargs):
                if 0 in leads:
                    raise FileNotFoundError("subset_x__f000")
                return _lead_dataset(leads)
        return Client()

    known_missing: set[int] = set()
    _query_leads(make_client, tuple(range(7)), ":APCP:surface:", CYCLE, known_missing=known_missing)
    assert known_missing == {0}

    requested.clear()
    second = _query_leads(make_client, tuple(range(7)), ":APCP:surface:", CYCLE, known_missing=known_missing)

    assert second.sizes["time"] == 6
    assert all(0 not in batch for batch in requested)
    assert len(requested) == 2  # f001-f003 and f004-f006, no hour-by-hour retries


def test_fetch_member_does_not_retry_a_field_that_is_not_published(monkeypatch):
    from forecast_v1 import acquisition as aq

    calls = []

    def fake_query(make_client, leads, search, cycle, unnamed=None, known_missing=None):
        calls.append(search)
        raise FileNotFoundError(f"{search} matched no usable fields")

    monkeypatch.setattr(aq, "_query_leads", fake_query)
    monkeypatch.setattr(aq.time, "sleep", lambda seconds: None)
    spec = aq.AcquisitionSpec("hrrr", "hrrr", "sfc", (0, 1), (None,), searches=(":ABC:",), required=True)

    try:
        aq._fetch_member(spec, CYCLE, None, Path("."), lambda **kwargs: None)
    except RuntimeError as error:
        assert "returned no forecast-v1 surface fields" in str(error)
    else:
        raise AssertionError("expected the member fetch to fail")
    assert calls == [":ABC:"]  # one attempt, not three


def test_query_leads_joins_a_single_lead_last_chunk(monkeypatch):
    # 49 HRRR leads in chunks of 8 leave f048 alone; that chunk used to make every run fail.
    from forecast_v1.acquisition import _query_leads

    monkeypatch.setenv("SMF_HERBIE_LEAD_CHUNK", "8")

    def make_client(leads):
        class Client:
            def xarray(self, search, **kwargs):
                return _lead_dataset(leads)
        return Client()

    result = _query_leads(make_client, tuple(range(49)), ":TMP:2 m above ground", CYCLE)

    expected = np.datetime64(CYCLE.replace(tzinfo=None), "ns") + np.arange(49) * np.timedelta64(1, "h")
    assert result.sizes["time"] == 49
    np.testing.assert_array_equal(result.time.values, expected)
    assert "valid_time" not in result.coords and "step" not in result.coords


def test_query_leads_raises_when_no_hour_has_the_field(monkeypatch):
    from forecast_v1.acquisition import _query_leads

    class Client:
        def xarray(self, search, **kwargs):
            raise FileNotFoundError("subset_x")

    with pytest.raises(FileNotFoundError):
        _query_leads(lambda leads: Client(), (0, 1, 2), ":APCP:surface:", CYCLE)


# APCP entries as Herbie's inventory()["search_this"] shows them (checked for
# HRRR 2026-10-01 12Z and the NOAA .idx files for 2026-10-21 12Z). There is
# no trailing colon unless the line has an extra field, as GEFS lines do.
HRRR_RRFS_APCP_RUN_TOTALS = (
    ":APCP:surface:0-0 day acc fcst",       # HRRR f00
    ":APCP:surface:0-1 hour acc fcst",      # HRRR/RRFS f01 (the only message)
    ":APCP:surface:0-5 hour acc fcst",      # HRRR/RRFS f05
    ":APCP:surface:0-1 day acc fcst",       # HRRR f24
    ":APCP:surface:0-2 day acc fcst",       # HRRR f48
    ":APCP:surface:0-50 hour acc fcst",     # RRFS f50
)
HRRR_RRFS_APCP_LAST_HOUR = (
    ":APCP:surface:4-5 hour acc fcst",
    ":APCP:surface:10-11 hour acc fcst",
    ":APCP:surface:47-48 hour acc fcst",
    ":APCP:surface:49-50 hour acc fcst",
)
GEFS_APCP = (
    ":APCP:surface:0-3 hour acc fcst:ENS=low-res ctl",
    ":APCP:surface:0-6 hour acc fcst:ENS=low-res ctl",
    ":APCP:surface:6-9 hour acc fcst:ENS=low-res ctl",
    ":APCP:surface:6-12 hour acc fcst:ENS=low-res ctl",
    ":APCP:surface:48-54 hour acc fcst:ENS=low-res ctl",
)


def test_apcp_searches_pick_the_run_total_for_hrrr_rrfs_and_every_gefs_bucket():
    import re
    from forecast_v1.acquisition import APCP_RUN_TOTAL_SEARCH, GEFS_SURFACE_SEARCHES, default_specs

    assert all(re.search(APCP_RUN_TOTAL_SEARCH, line) for line in HRRR_RRFS_APCP_RUN_TOTALS)
    assert not any(re.search(APCP_RUN_TOTAL_SEARCH, line) for line in HRRR_RRFS_APCP_LAST_HOUR)
    gefs_apcp = next(search for search in GEFS_SURFACE_SEARCHES if "APCP" in search)
    assert all(re.search(gefs_apcp, line) for line in GEFS_APCP)
    specs = {spec.public_name: spec for spec in default_specs()}
    assert APCP_RUN_TOTAL_SEARCH in specs["hrrr"].searches and APCP_RUN_TOTAL_SEARCH in specs["rrfs"].searches
    assert APCP_RUN_TOTAL_SEARCH not in specs["gefs"].searches


def test_cloud_and_sunshine_searches_keep_one_message_per_model():
    # The old TCDC pattern excluded "... hour ave fcst:" with a trailing colon:
    # Herbie's RRFS text has none, so both messages matched; GEFS's text has
    # one (":ENS=..."), so its only cloud message was rejected every run.
    import re
    from forecast_v1.acquisition import DSWRF_INSTANT_SEARCH, GEFS_SURFACE_SEARCHES, TCDC_INSTANT_SEARCH

    rrfs_cloud = (
        ":TCDC:entire atmosphere (considered as a single layer):5 hour fcst",
        ":TCDC:entire atmosphere (considered as a single layer):4-5 hour ave fcst",
    )
    rrfs_sun = (":DSWRF:surface:4-5 hour ave fcst", ":DSWRF:surface:5 hour fcst")
    assert [line for line in rrfs_cloud if re.search(TCDC_INSTANT_SEARCH, line)] == [rrfs_cloud[0]]
    assert [line for line in rrfs_sun if re.search(DSWRF_INSTANT_SEARCH, line)] == [rrfs_sun[1]]
    assert re.search(TCDC_INSTANT_SEARCH, ":TCDC:entire atmosphere:5 hour fcst")  # HRRR
    assert re.search(DSWRF_INSTANT_SEARCH, ":DSWRF:surface:anl")  # HRRR f00
    gefs = (
        ":TCDC:entire atmosphere:0-6 hour ave fcst:ENS=low-res ctl",
        ":DSWRF:surface:0-6 hour ave fcst:ENS=low-res ctl",
    )
    assert all(any(re.search(search, line) for search in GEFS_SURFACE_SEARCHES) for line in gefs)


def test_hrrr_run_total_precipitation_becomes_hourly_rain(tmp_path):
    # The 2026-10-01 run kept the run total: one cell read 0, 16, 20, 25 ... 180 mm "per hour".
    run_total = np.array([0.0, 16.2, 19.9, 25.0, 25.0, 31.4])
    coords = {
        "step": np.arange(6).astype("timedelta64[h]"), "time": np.datetime64(CYCLE.replace(tzinfo=None)),
        "latitude": ("y", [36.0]), "longitude": ("x", [-93.0]),
    }
    shape = (6, 1, 1)
    variables = {
        "t2m": (("step", "y", "x"), np.full(shape, 293.15), {"units": "K"}),
        "r2": (("step", "y", "x"), np.full(shape, 40.0), {"units": "%"}),
        "u10": (("step", "y", "x"), np.full(shape, 3.0), {"units": "m s-1"}),
        "v10": (("step", "y", "x"), np.full(shape, 2.0), {"units": "m s-1"}),
        "gust": (("step", "y", "x"), np.full(shape, 6.0), {"units": "m s-1"}),
        "tp": (("step", "y", "x"), run_total.reshape(shape), {"units": "kg m**-2"}),
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

    result = acquire_source(AcquisitionSpec("hrrr", "hrrr", "sfc", tuple(range(6)), (None,)), CYCLE, tmp_path, fast_herbie_factory=Herbie)
    hourly = result.dataset.precipitation_increment.isel(member=0, y=0, x=0).values
    assert hourly.tolist() == pytest.approx([0.0, 16.2, 3.7, 5.1, 0.0, 6.4], abs=1e-4)
    assert hourly.sum() == pytest.approx(run_total[-1], abs=1e-4)


def test_packing_dips_in_a_run_total_are_not_bucket_resets():
    from forecast_v1.adapters import _precip_to_increments

    # Real HRRR cell, 2026-10-01 12Z f00-f04: 3.085 -> 3.080 is GRIB packing noise.
    total = xr.DataArray(
        np.array([0.0, 3.085, 3.080, 3.12, 6.95, 0.4]), dims="time",
        coords={"time": np.arange(6)}, attrs={"accumulation_semantics": "cumulative"},
    )
    hourly = _precip_to_increments(total).values
    assert hourly.tolist() == pytest.approx([0.0, 3.085, 0.0, 0.04, 3.83, 0.4], abs=1e-6)  # last step is a real reset


def _gefs_buckets(three_hour_values: np.ndarray, *, average: bool) -> np.ndarray:
    """Lay out 3-hour values the way GEFS publishes them: f03 0-3, f06 0-6, f09 6-9, ..."""
    published = []
    for index, value in enumerate(three_hour_values):
        if index % 2 == 0:
            published.append(value)
        else:
            pair = three_hour_values[index - 1:index + 1]
            published.append(pair.mean() if average else pair.sum())
    return np.array(published)


def test_gefs_six_hour_buckets_are_not_double_counted():
    from forecast_v1.acquisition import _hourly_gefs

    rng = np.random.default_rng(3)
    rain_3h = rng.uniform(0, 6, size=24)  # mm in each 3-hour interval, f00-f72
    flux_3h = rng.uniform(0, 800, size=24)  # mean W m-2 in each interval
    cloud_3h = rng.uniform(0, 100, size=24)  # mean % in each interval
    times = [np.datetime64(CYCLE.replace(tzinfo=None) + timedelta(hours=3 * i), "ns") for i in range(25)]
    nan = np.array([np.nan])  # GEFS publishes no APCP or DSWRF at f000
    dataset = xr.Dataset(
        {
            "t2m": (("time", "y", "x"), np.full((25, 1, 1), 290.0)),
            "tp": (("time", "y", "x"), np.concatenate([nan, _gefs_buckets(rain_3h, average=False)]).reshape(25, 1, 1)),
            "dswrf": (("time", "y", "x"), np.concatenate([nan, _gefs_buckets(flux_3h, average=True)]).reshape(25, 1, 1)),
            "tcc": (("time", "y", "x"), np.concatenate([nan, _gefs_buckets(cloud_3h, average=True)]).reshape(25, 1, 1)),
        },
        coords={"time": times, "y": [0.0], "x": [0.0]},
    )
    hourly = _hourly_gefs(dataset, CYCLE)
    rain = hourly.tp.isel(y=0, x=0).values
    assert rain[0] == 0
    assert rain[1:].tolist() == pytest.approx(np.repeat(rain_3h / 3, 3).tolist())
    assert rain.sum() == pytest.approx(rain_3h.sum())
    assert hourly.dswrf.isel(y=0, x=0).values[1:].tolist() == pytest.approx(np.repeat(flux_3h, 3).tolist())
    assert hourly.tcc.isel(y=0, x=0).values[1:].tolist() == pytest.approx(np.repeat(cloud_3h, 3).tolist())


def test_gefs_lead_off_the_three_hour_grid_fails_loudly():
    from forecast_v1.acquisition import _hourly_gefs

    times = [np.datetime64(CYCLE.replace(tzinfo=None) + timedelta(hours=hour), "ns") for hour in (0, 3, 4)]
    dataset = xr.Dataset(
        {"tp": (("time", "y", "x"), np.array([np.nan, 1.0, 1.0]).reshape(3, 1, 1))},
        coords={"time": times, "y": [0.0], "x": [0.0]},
    )
    with pytest.raises(ValueError, match="off the 3-hour bucket grid"):
        _hourly_gefs(dataset, CYCLE)


def test_widespread_heavy_hourly_rain_flags_the_run_but_one_extreme_cell_does_not():
    from forecast_v1.pipeline import precipitation_suspect

    rain = xr.DataArray(np.full((49, 10, 10), 0.5), dims=("time", "y", "x"))
    rain[13, 4, 4] = 103.6  # a real HRRR cell on 2026-10-01
    assert not precipitation_suspect(xr.Dataset({"precipitation_increment": rain}))
    rain[20:, :, :5] = 40.0  # a run total that stays high after the rain, as beta-1 published
    assert precipitation_suspect(xr.Dataset({"precipitation_increment": rain}))

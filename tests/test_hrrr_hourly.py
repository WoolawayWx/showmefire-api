import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.rate_limit import RateLimiter
from routers import hrrr_hourly as router_module
from services.hrrr_hourly import BY_NAME, ELEMENTS, derive, fetch, job, render, store

CYCLE = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
CYCLE_ID = "20261016_12z"
NY, NX = 6, 7


def grid():
    lat, lon = np.meshgrid(np.linspace(37.0, 38.0, NY), np.linspace(-93.0, -92.0, NX), indexing="ij")
    return lat, lon


def fields(offset=0.0):
    out = {}
    for index, element in enumerate(ELEMENTS):
        out[element.name] = np.full((NY, NX), (index + 1) * 10.0 + offset, dtype="float32")
    out["cape_jkg"][0, 0] = np.nan
    return out


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_HRRR_HOURLY_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SMF_HRRR_HOURLY_IMAGES", str(tmp_path / "images"))
    return tmp_path / "data"


def write(root, hours=(0, 1, 2), cycle=CYCLE, expected=range(0, 19)):
    lat, lon = grid()
    return store.write_cycle(cycle, {h: fields(h * 10) for h in hours}, lat, lon, {"ny": NY, "nx": NX}, {},
                             expected, issued_at=CYCLE + timedelta(hours=2, minutes=40), root=root)


# --- derive -------------------------------------------------------------------------------------

def test_every_element_has_registry_entry_and_unique_name():
    names = [e.name for e in ELEMENTS]
    assert len(names) == len(set(names))
    assert all(e.scale > 0 for e in ELEMENTS)


def test_derive_hour_converts_units_and_omits_missing(monkeypatch):
    shape = (NY, NX)
    base = {name: np.zeros(shape) for name in (
        "mixing_height_ft", "holzworth_ft", "transport_speed_mph", "transport_dir_deg", "ventilation_kt_ft",
        "ventilation_category", "inversion_strength_f", "inversion_top_ft", "lapse_rate_c_km", "haines", "hdw")}
    monkeypatch.setattr(derive.hp, "derive_fields", lambda p: base)
    lat, lon = grid()
    profiles = SimpleNamespace(t2_c=np.full(shape, 20.0), td2_c=np.full(shape, 10.0), u10=np.full(shape, 3.0),
                               v10=np.full(shape, 4.0), lon=lon, central_longitude=lon.mean(), standard_parallel=38.5)
    extras = {"gust": np.full(shape, 10.0), "cape": np.full(shape, 250.0), "smoke_surface": np.full(shape, 2e-8),
              "smoke_column": np.full(shape, 3e-6)}
    out = derive.derive_hour(profiles, extras)
    assert np.allclose(out["temp_f"], 68.0)
    assert np.allclose(out["lcl_m"], 1250.0)               # 125 m per degree of dewpoint depression
    assert np.allclose(out["fuel_moisture_pct"], 3.0 + 0.25 * out["rh_pct"])
    assert 45 < float(out["rh_pct"][0, 0]) < 60             # computed from T/Td when HRRR RH is absent
    assert np.allclose(out["wind_speed_mph"], 5.0 * 2.23694, atol=0.01)
    assert np.allclose(out["wind_gust_mph"], 22.3694, atol=0.01)
    assert np.allclose(out["smoke_surface_ugm3"], 20.0) and np.allclose(out["smoke_column_mgm2"], 3.0)
    assert "cin_jkg" not in out and "precip_mm_1h" not in out  # not fetched -> not invented


# --- store --------------------------------------------------------------------------------------

def test_write_cycle_labels_run_and_round_trips(root):
    manifest = write(root)
    assert manifest["label"] == "Forecast from the 12z HRRR run"
    assert manifest["cycle_time_utc"] == "2026-10-16T12:00:00Z"
    assert manifest["issued_at_utc"] == "2026-10-16T14:40:00Z"
    assert manifest["forecast_hours"] == [0, 1, 2] and manifest["missing_hours"] == list(range(3, 19))
    assert manifest["valid_times_utc"][2] == "2026-10-16T14:00:00Z"
    assert manifest["variables"]["mixing_height_ft"]["units"] == "ft AGL"
    assert json.loads((root / "latest.json").read_text())["cycle"] == CYCLE_ID

    import xarray as xr
    nc = xr.open_dataset(store.nc_path(CYCLE_ID, root))
    assert nc.attrs["cycle"] == "12z" and "12z HRRR run" in nc.attrs["note"]
    expected = fields(10)["mixing_height_ft"]
    assert np.allclose(nc["mixing_height_ft"].values[1], expected, atol=BY_NAME["mixing_height_ft"].scale)
    assert np.isnan(nc["cape_jkg"].values[0, 0, 0])
    if "zarr" in manifest["files"]:
        z = xr.open_zarr(root / CYCLE_ID / "forecast.zarr")
        assert np.allclose(z["mixing_height_ft"].values, nc["mixing_height_ft"].values, equal_nan=True)
        assert z["mixing_height_ft"].encoding["chunks"] == (1, NY, NX)


def test_rewrite_replaces_only_its_own_cycle(root):
    write(root)
    other = write(root, cycle=CYCLE - timedelta(days=1))
    write(root, hours=(0, 1))
    assert store.list_cycles(root) == [CYCLE_ID, "20261015_12z"]
    assert store.read_manifest(CYCLE_ID, root)["forecast_hours"] == [0, 1]
    assert store.read_manifest("20261015_12z", root)["forecast_hours"] == other["forecast_hours"]
    assert not list(root.glob(".*.tmp")) and not list(root.glob(".*.old"))


def test_point_series_and_outside_grid(root):
    write(root)
    result = store.point_series(CYCLE_ID, 37.4, -92.5, root)
    assert result["lead_hours"] == [0, 1, 2]
    assert result["series"]["mixing_height_ft"] == [pytest.approx(10.0), pytest.approx(20.0), pytest.approx(30.0)]
    assert store.point_series(CYCLE_ID, 45.0, -92.5, root)["error"] == "outside_grid"


def test_prune_removes_old_cycles_everywhere(root):
    write(root)
    write(root, cycle=CYCLE - timedelta(days=20))
    removed = store.prune(now=CYCLE + timedelta(hours=5), days=14, root=root)
    assert removed == ["20260926_12z"]
    assert not store.nc_path("20260926_12z", root).exists() and store.nc_path(CYCLE_ID, root).exists()


# --- job ----------------------------------------------------------------------------------------

def fake_profiles():
    lat, lon = grid()
    return SimpleNamespace(lat=lat, lon=lon, central_longitude=-97.5, standard_parallel=38.5)


def test_build_cycle_resumes_from_work_files_and_survives_a_failed_hour(root):
    calls = []

    def fetch_hour(cycle_time, lead):
        calls.append(lead)
        if lead == 1:
            raise RuntimeError("download failed")
        return fake_profiles(), {}

    manifest = job.build_cycle(CYCLE, hours=(0, 1, 2), render=False, fetch_hour=fetch_hour,
                               derive_hour=lambda p, e: fields(), root=root)
    assert manifest["forecast_hours"] == [0, 2] and manifest["failed_hours"] == [1]
    assert manifest["missing_hours"] == [1]
    assert manifest["series"]["mixing_height_ft"][0]["median"] == pytest.approx(10.0) or manifest["series"] == {}
    assert not (root / ".work" / CYCLE_ID).exists()


def test_run_auto_skips_when_built_unpublished_or_forecast_running(root):
    kwargs = dict(now=CYCLE + timedelta(hours=3), root=root, cycle_time=CYCLE, hours=(0, 1, 2))
    built, seen = [], []

    def build(cycle_time, hours, root, on_progress=None, wait=None):
        built.append(cycle_time)
        on_progress("fetching", 1, 3)
        seen.append(job.read_status(root))
        on_progress("rendering", 2, 3)
        return write(root, hours=hours, expected=hours)

    go = dict(ready=lambda c, h: True, lock_free=lambda: True, forecast_done=lambda c: True, build=build)
    assert job.run_auto(**{**go, "ready": lambda c, h: False}, **kwargs)["reason"] == "cycle not fully published"
    assert job.read_status(root)["state"] == "waiting_for_hrrr"
    assert job.run_auto(**{**go, "lock_free": lambda: False}, **kwargs)["reason"] == "forecast running"
    assert not built
    assert job.run_auto(**go, **kwargs)["action"] == "built"
    again = job.run_auto(**go, **kwargs)
    assert again == {"action": "skip", "reason": "already built", "cycle": CYCLE_ID} and len(built) == 1
    status = job.read_status(root)
    assert status["state"] == "complete" and status["cycle"] == CYCLE_ID and status["serving_cycle"] == CYCLE_ID
    assert status["updating"] is False and status["pending"] is False
    assert seen[0]["state"] == "building" and seen[0]["updating"] is True and seen[0]["hours_total"] == 3
    assert seen[0]["started_at"]


def test_hourly_waits_for_the_daily_forecast_then_proceeds(root):
    built = []

    def build(cycle_time, hours, root, on_progress=None, wait=None):
        built.append(1)
        return write(root, hours=hours, expected=hours)

    base = dict(root=root, cycle_time=CYCLE, hours=(0, 1, 2), ready=lambda c, h: True, lock_free=lambda: True,
                build=build)
    # Forecast not done yet (e.g. a tick just before 14:30 UTC): wait, even though HRRR is published.
    early = job.run_auto(now=CYCLE + timedelta(hours=2), forecast_done=lambda c: False, **base)
    assert early["reason"] == "daily forecast not finished" and not built
    state = job.read_status(root)
    assert state["state"] == "waiting_for_forecast" and state["pending"] is True and state["updating"] is False
    # Forecast finished (and its notifications sent, lock released): build.
    assert job.run_auto(now=CYCLE + timedelta(hours=3), forecast_done=lambda c: True, **base)["action"] == "built"


def test_hourly_proceeds_after_grace_if_forecast_never_reports(root):
    result = job.run_auto(now=CYCLE + timedelta(hours=7), root=root, cycle_time=CYCLE, hours=(0,),
                          ready=lambda c, h: True, lock_free=lambda: True, forecast_done=lambda c: False,
                          build=lambda c, hours, root, on_progress=None, wait=None: write(root, hours=hours, expected=hours))
    assert result["action"] == "built"


def test_forecast_complete_reads_the_daily_forecast_status(tmp_path):
    path = tmp_path / "status.json"
    assert not job.forecast_complete(CYCLE, path)
    path.write_text(json.dumps({"ForecastFireDanger": {"status": "updated", "model_run": "2026-10-15 12Z"}}))
    assert not job.forecast_complete(CYCLE, path)       # yesterday's run
    path.write_text(json.dumps({"ForecastFireDanger": {"status": "updated", "model_run": "2026-10-16 12Z"}}))
    assert job.forecast_complete(CYCLE, path)
    path.write_text(json.dumps({"ForecastFireDanger": {"status": "error", "model_run": "2026-10-16 12Z"}}))
    assert not job.forecast_complete(CYCLE, path)


def test_run_auto_records_failure(root):
    def boom(cycle_time, hours, root, on_progress=None, wait=None):
        raise RuntimeError("no hours")

    with pytest.raises(RuntimeError):
        job.run_auto(now=CYCLE + timedelta(hours=3), root=root, cycle_time=CYCLE, ready=lambda c, h: True,
                     lock_free=lambda: True, forecast_done=lambda c: True, build=boom)
    status = job.read_status(root)
    assert status["state"] == "failed" and status["updating"] is False and "no hours" in status["error"]


def test_forecast_lock_free_detects_exclusive_holder(tmp_path):
    import fcntl
    lock = tmp_path / "forecast.lock"
    assert job.forecast_lock_free(lock)
    with open(lock, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        assert not job.forecast_lock_free(lock)
    assert job.forecast_lock_free(lock)


def test_latest_cycle_and_ready_checks():
    assert fetch.latest_cycle(datetime(2026, 10, 16, 11, 59, tzinfo=timezone.utc)) == CYCLE - timedelta(days=1)
    assert fetch.latest_cycle(datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc)) == CYCLE
    urls = []
    assert fetch.cycle_ready(CYCLE, 18, head=lambda u: urls.append(u) or 200)
    assert all("t12z" in u and "f18" in u for u in urls) and len(urls) == 2
    assert not fetch.cycle_ready(CYCLE, 18, head=lambda u: 404)


def test_job_never_touches_existing_forecast_outputs(root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    protected = [Path("images/mo-forecastmaxtemp.png"), Path("archive/forecasts/station_forecasts_20261016_12.json"),
                 Path("gis/peak_fire_danger.tif"), Path("cache/hrrr/hrrr_20261016_12z_f04-15.nc")]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    job.build_cycle(CYCLE, hours=(0,), render=False, fetch_hour=lambda c, l: (fake_profiles(), {}),
                    derive_hour=lambda p, e: fields(), root=root)
    assert all(path.read_bytes() == b"original" for path in protected)


# --- render + router ----------------------------------------------------------------------------

def test_image_paths_are_namespaced(tmp_path):
    path = render.image_path(CYCLE_ID, "mixing_height_ft", 9, tmp_path / "images")
    assert path == tmp_path / "images" / CYCLE_ID / "mixing_height_ft_f09.png"
    assert "mo-" not in path.name


def _client():
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app)


def test_router_serves_manifest_zarr_point_and_blocks_traversal(root, tmp_path):
    manifest = write(root)
    image = render.image_path(CYCLE_ID, "mixing_height_ft", 1)
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG")
    client = _client()
    assert client.get("/hrrr-hourly/status").json()["state"] == "idle"
    job.set_state(root, "building", CYCLE_ID, hours_total=19, hours_done=4)
    live = client.get("/hrrr-hourly/status").json()
    assert live["updating"] is True and live["hours_done"] == 4 and live["serving_cycle"] == CYCLE_ID
    assert client.get("/hrrr-hourly/runs").json()["latest"] == CYCLE_ID
    assert client.get("/hrrr-hourly/runs/latest").json()["cycle"] == CYCLE_ID
    assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}").json()["label"] == "Forecast from the 12z HRRR run"
    assert client.get("/hrrr-hourly/runs/not-a-cycle").status_code == 400
    assert client.get("/hrrr-hourly/runs/20260101_12z").status_code == 404
    point = client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/point", params={"lat": 37.4, "lon": -92.5}).json()
    assert point["units"]["mixing_height_ft"] == "ft AGL" and point["label"].startswith("Forecast from the 12z")
    assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/point", params={"lat": 45, "lon": -92.5}).status_code == 404
    assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/netcdf").status_code == 200
    assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/image/mixing_height_ft/1").content == b"\x89PNG"
    assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/image/mixing_height_ft/99").status_code == 404
    assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/image/../1").status_code == 404
    if "zarr" in manifest["files"]:
        meta = client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/zarr/.zmetadata")
        assert meta.status_code == 200 and meta.headers["content-type"] == "application/json"
        assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/zarr/mixing_height_ft/1.0.0").status_code == 200
    for evil in ("../manifest.json", "%2e%2e/manifest.json", "..%2fmanifest.json"):
        assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/zarr/{evil}").status_code in (400, 404)


def test_point_route_is_rate_limited(root):
    write(root)
    limiter = router_module.point_limiter
    limiter.reset()
    client = _client()
    statuses = [client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/point", params={"lat": 37.4, "lon": -92.5}).status_code
                for _ in range(limiter.per_minute + 1)]
    limiter.reset()
    assert statuses[-1] == 429 and statuses[0] == 200


# --- priorities: forecast > 15-minute maps > this product ---------------------------------------

def _hold(path):
    import fcntl
    handle = open(path, "a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def test_pauses_for_forecast_then_maps_then_memory(tmp_path, monkeypatch):
    import fcntl
    forecast, maps = tmp_path / "forecast.lock", tmp_path / "maps.lock"
    held = {"forecast": _hold(forecast), "maps": _hold(maps)}
    memory = iter([100, 100, 9000])
    waits = []

    def sleep(_seconds):
        # each poll "ends" one blocker: forecast first, then maps
        if "forecast" in held:
            fcntl.flock(held.pop("forecast"), fcntl.LOCK_UN)
        elif "maps" in held:
            fcntl.flock(held.pop("maps"), fcntl.LOCK_UN)

    job.wait_for_priority_work(tmp_path, on_wait=waits.append, sleep=sleep, forecast_lock=forecast, maps_lock=maps,
                               memory_mb=lambda: next(memory))
    assert waits[0] == "daily forecast" and "15-minute maps" in waits and waits[-1] == "free memory"
    assert waits.index("daily forecast") < waits.index("15-minute maps") < waits.index("free memory")


def test_pause_is_capped_so_the_build_cannot_stall(tmp_path):
    maps = tmp_path / "maps.lock"
    handle = _hold(maps)
    ticks = iter(range(0, 100000, 400))
    job.wait_for_priority_work(tmp_path, sleep=lambda s: None, clock=lambda: next(ticks),
                               forecast_lock=tmp_path / "none.lock", maps_lock=maps, memory_mb=lambda: 9000)
    handle.close()  # returned despite the lock still being held


def test_refuses_to_build_on_a_nearly_full_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "free_disk_mb", lambda path: 10)
    with pytest.raises(RuntimeError, match="free disk"):
        job.wait_for_priority_work(tmp_path)


def test_status_shows_why_a_build_is_paused(root):
    seen = []

    def build(cycle_time, hours, root, on_progress=None, wait=None):
        wait(root)
        seen.append(job.read_status(root))
        return write(root, hours=hours, expected=hours)

    original = job.wait_for_priority_work
    job.wait_for_priority_work = lambda r, on_wait=None, **k: on_wait("15-minute maps")
    try:
        job.run_auto(now=CYCLE + timedelta(hours=3), root=root, cycle_time=CYCLE, hours=(0,), ready=lambda c, h: True,
                     lock_free=lambda: True, forecast_done=lambda c: True, build=build)
    finally:
        job.wait_for_priority_work = original
    assert seen[0]["phase"] == "paused" and seen[0]["paused_for"] == "15-minute maps" and seen[0]["updating"]


def test_scheduler_subprocess_runs_at_lowest_priority(monkeypatch):
    from services.hrrr_hourly import scheduler_job
    captured = {}
    monkeypatch.setattr(scheduler_job.subprocess, "run",
                        lambda *a, **k: captured.update(k) or SimpleNamespace(returncode=0, stdout="{}", stderr=""))
    scheduler_job._run()
    assert captured["preexec_fn"] is scheduler_job._lower_priority
    assert captured["env"]["OMP_NUM_THREADS"] == "1"

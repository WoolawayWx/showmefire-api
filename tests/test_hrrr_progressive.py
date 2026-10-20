"""Progressive HRRR build: each forecast hour is added to the store and published as soon as it is processed."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.fire_danger import calculate_fire_danger
from core.rate_limit import RateLimiter
from routers import hrrr_fire_danger_peak as peak_router
from routers import hrrr_hourly as router_module
from services import cdn_datasets
from services.hrrr_hourly import ELEMENTS, danger, job, live_store, peak, progressive, store

CYCLE = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
CYCLE_ID = "20261019_12z"
NY, NX = 6, 7
ALL_HOURS = tuple(range(19))


def grid():
    lat, lon = np.meshgrid(np.linspace(37.0, 38.0, NY), np.linspace(-93.0, -92.0, NX), indexing="ij")
    return lat, lon


def hour_fields(lead):
    """Distinguishable values per hour: temp_f rises with the hour, rh falls, so peaks and minimums are predictable."""
    out = {}
    for index, element in enumerate(ELEMENTS):
        out[element.name] = np.full((NY, NX), (index + 1) * 10.0 + lead, dtype="float32")
    out["temp_f"][:] = 60.0 + lead
    out["rh_pct"][:] = 50.0 - lead
    out["fire_danger_est"][:] = min(lead // 4, 4)
    out["fuel_moisture_pct"][:] = 10.0
    out["wind_speed_mph"][:] = 5.0 + lead
    out["wind_gust_mph"][:] = 10.0 + lead
    out["hdw"][:] = 20.0 + lead
    return out


def fake_profiles():
    lat, lon = grid()
    return SimpleNamespace(lat=lat, lon=lon, central_longitude=-97.5, standard_parallel=38.5,
                           t2_c=np.full((NY, NX), 20.0), u10=np.full((NY, NX), 3.0), v10=np.full((NY, NX), -4.0))


class FakeR2:
    """An in-memory bucket that remembers each object's cache header."""

    def __init__(self, fail_times=0):
        self.objects = {}
        self.cache = {}
        self.fail_times = fail_times
        self.calls = 0

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.calls += 1
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("R2 is down")
        self.objects[key] = Path(filename).read_bytes()
        self.cache[key] = (ExtraArgs or {}).get("CacheControl")

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
        self.cache[Key] = kwargs.get("CacheControl")

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            error = Exception("missing")
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error
        return {"Body": SimpleNamespace(read=lambda: self.objects[Key])}

    def get_paginator(self, name):
        raise AssertionError("nothing should be pruned")

    def json(self, key):
        return json.loads(self.objects[key])



@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_HRRR_HOURLY_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SMF_HRRR_HOURLY_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("SMF_HRRR_PEAK_DIR", str(tmp_path / "peak"))
    monkeypatch.setattr(cdn_datasets, "configured", lambda: False)
    monkeypatch.setattr(cdn_datasets, "make_client", lambda: pytest.fail("a test tried to reach R2"))
    monkeypatch.setattr(progressive, "LATEST_AFTER_HOURS", 3)
    return SimpleNamespace(root=tmp_path / "data", peak=tmp_path / "peak", r2=FakeR2())


class Hrrr:
    """A pretend HRRR: hours 0..published-1 are out, and every fetch is counted."""

    def __init__(self, published=0, fail=()):
        self.published, self.fail, self.fetched = published, set(fail), []

    def ready(self, run, lead):
        return lead < self.published

    def fetch(self, cycle_time, lead):
        self.fetched.append(lead)
        if lead in self.fail:
            raise RuntimeError("download failed")
        return fake_profiles(), {}


def make_tick(env, hrrr):
    """advance() with a derive step that knows which hour it is deriving."""
    def derive(profiles, extras):
        return hour_fields(hrrr.fetched[-1])

    def run(hours=ALL_HOURS, minutes_after_cycle=60, **kwargs):
        return progressive.advance(
            now=CYCLE + timedelta(minutes=minutes_after_cycle), hours=hours, root=env.root, peak_root=env.peak,
            fetch_hour=hrrr.fetch, derive_hour=derive, hour_ready=hrrr.ready, wait=lambda root: None,
            candidates=[CYCLE], client=env.r2, **kwargs)
    return run


# --- the fire danger estimate ---------------------------------------------------------------------

def test_vectorised_fire_danger_matches_the_canonical_rule():
    rng = np.random.default_rng(7)
    fm = rng.uniform(2, 20, 4000)
    rh = rng.uniform(5, 70, 4000)
    wind = rng.uniform(0, 35, 4000)
    expected = [calculate_fire_danger(a, b, c) for a, b, c in zip(fm, rh, wind)]
    assert danger.fire_danger_category(fm, rh, wind).astype(int).tolist() == expected


def test_fire_danger_is_nan_when_an_input_is_missing_and_reaches_the_product():
    out = danger.fire_danger_category(np.array([5.0, np.nan]), np.array([15.0, 15.0]), np.array([30.0, 30.0]))
    assert out[0] == 4 and np.isnan(out[1])
    assert "fire_danger_est" in {e.name for e in ELEMENTS}


# --- the peak window ---------------------------------------------------------------------------------

def test_peak_days_follow_the_10am_to_9pm_central_window():
    days = peak.plan(CYCLE, ALL_HOURS)
    assert len(days) == 1 and days[0].date == "2026-10-19"
    assert days[0].window_leads == tuple(range(3, 15))            # 10 AM CDT is 15Z, 9 PM CDT is 02Z
    overnight = peak.plan(datetime(2026, 10, 19, 0, tzinfo=timezone.utc), ALL_HOURS)
    assert [d.date for d in overnight] == ["2026-10-18", "2026-10-19"]
    assert peak.plan(datetime(2026, 10, 19, 6, tzinfo=timezone.utc), ALL_HOURS)[0].window_leads[0] == 9


def test_peak_aggregate_takes_the_worst_hour_and_when_it_happened():
    day = peak.plan(CYCLE, ALL_HOURS)[0]
    fields = {lead: hour_fields(lead) for lead in (3, 4, 5, 6)}
    out = peak.aggregate(day, fields)
    assert out["peak_fire_danger"][0, 0] == 1 and out["peak_hour_local"][0, 0] == 11   # first hour at the max: f04, 11 AM CDT
    assert out["min_rh_pct"][0, 0] == 44.0 and out["max_temp_f"][0, 0] == 66.0 and out["max_wind_mph"][0, 0] == 11.0
    assert peak.aggregate(day, {}) == {}


# --- progressive build ---------------------------------------------------------------------------------

def test_each_hour_is_published_as_it_arrives_and_latest_waits_for_enough_of_them(env):
    hrrr = Hrrr(published=2)
    run = make_tick(env, hrrr)
    result = run()
    assert result["action"] == "advanced" and result["advanced"] == [0, 1]
    manifest = store.read_manifest(CYCLE_ID, env.root)
    assert manifest["forecast_hours"] == [0, 1] and manifest["complete"] is False
    assert manifest["pending_hours"] == list(range(2, 19)) and manifest["missing_hours"] == []
    assert manifest["zarr_leads"] == list(ALL_HOURS)
    # listed on the CDN, but not "latest" yet: 2 hours is fewer than LATEST_AFTER_HOURS
    runs = env.r2.json("datasets/hrrr-hourly/runs.json")
    assert runs["runs"][0]["id"] == CYCLE_ID and runs["runs"][0]["ready"] is False and runs["latest"] is None
    assert "datasets/hrrr-hourly/latest.json" not in env.r2.objects
    assert not (env.root / "latest.json").exists()
    # the manifest of a run in progress is cached for a minute; finished chunks for good
    assert env.r2.cache[f"datasets/hrrr-hourly/runs/{CYCLE_ID}/manifest.json"] == cdn_datasets.LIVE_CACHE
    chunk = f"datasets/hrrr-hourly/runs/{CYCLE_ID}/forecast.zarr/temp_f/1.0.0"
    assert env.r2.cache[chunk] == cdn_datasets.IMMUTABLE_CACHE
    assert f"datasets/hrrr-hourly/runs/{CYCLE_ID}/forecast.zarr/.zmetadata" in env.r2.objects
    assert f"datasets/hrrr-hourly/runs/{CYCLE_ID}/forecast.zarr/temp_f/2.0.0" not in env.r2.objects

    hrrr.published = 4
    assert run()["advanced"] == [2, 3]
    assert hrrr.fetched == [0, 1, 2, 3]                            # hours already done are not fetched again
    assert env.r2.json("datasets/hrrr-hourly/latest.json")["run"] == CYCLE_ID
    assert json.loads((env.root / "latest.json").read_text())["cycle"] == CYCLE_ID
    assert job.read_status(env.root)["state"] == "building"
    assert job.read_status(env.root)["hours_done"] == 4


def test_the_store_is_readable_while_partial_and_missing_hours_are_empty(env):
    import xarray as xr

    hrrr = Hrrr(published=3)
    make_tick(env, hrrr)()
    ds = xr.open_zarr(env.root / CYCLE_ID / "forecast.zarr")
    assert ds["lead_hour"].values.tolist() == list(ALL_HOURS)
    assert ds["valid_time"].values[0] == np.datetime64("2026-10-19T12:00:00")
    assert float(ds["temp_f"].isel(lead_hour=2).values[0, 0]) == pytest.approx(62.0)
    assert np.isnan(ds["temp_f"].isel(lead_hour=3).values).all()


def test_finishing_writes_the_netcdf_marks_complete_and_caches_for_good(env):
    hrrr = Hrrr(published=19)
    result = make_tick(env, hrrr)()
    assert result["action"] == "built" and result["missing_hours"] == []
    manifest = store.read_manifest(CYCLE_ID, env.root)
    assert manifest["complete"] is True and manifest["pending_hours"] == [] and manifest["forecast_hours"] == list(ALL_HOURS)
    assert store.nc_path(CYCLE_ID, env.root).is_file() and manifest["files"]["netcdf"]
    assert not (env.root / ".work" / CYCLE_ID).exists()
    assert env.r2.cache[f"datasets/hrrr-hourly/runs/{CYCLE_ID}/manifest.json"] == cdn_datasets.IMMUTABLE_CACHE
    assert env.r2.json("datasets/hrrr-hourly/runs.json")["runs"][0]["complete"] is True
    assert job.read_status(env.root)["state"] == "complete"
    # nothing left to do
    assert make_tick(env, hrrr)()["reason"] == "already built"
    # series and the NetCDF agree with the stored hours
    assert manifest["series"]["temp_f"][5]["median"] == pytest.approx(65.0)
    point = store.point_series(CYCLE_ID, 37.4, -92.5, env.root)
    assert point["lead_hours"] == list(ALL_HOURS) and point["series"]["temp_f"][4] == pytest.approx(64.0)


def test_a_cycle_in_progress_answers_point_queries_from_the_zarr_store(env):
    make_tick(env, Hrrr(published=3))()
    point = store.point_series(CYCLE_ID, 37.4, -92.5, env.root)
    assert point["lead_hours"] == [0, 1, 2] and point["series"]["temp_f"] == [60.0, 61.0, 62.0]
    assert store.point_series(CYCLE_ID, 10.0, -92.5, env.root)["error"] == "outside_grid"
    some = store.point_series(CYCLE_ID, 37.4, -92.5, env.root, names=("temp_f",))
    assert set(some["series"]) == {"temp_f"} and some["series"]["temp_f"] == [60.0, 61.0, 62.0]


def test_an_hour_that_keeps_failing_is_given_up_on_and_the_rest_are_kept(env):
    hrrr = Hrrr(published=5, fail=(2,))
    run = make_tick(env, hrrr)
    hours = tuple(range(5))
    for _ in range(progressive.MAX_HOUR_ATTEMPTS - 1):
        assert run(hours=hours)["advanced"] in ([0, 1, 3, 4], [])
    result = run(hours=hours)
    assert result["action"] == "built" and result["missing_hours"] == [2]
    manifest = store.read_manifest(CYCLE_ID, env.root)
    assert manifest["forecast_hours"] == [0, 1, 3, 4] and manifest["missing_hours"] == [2]
    assert hrrr.fetched.count(2) == progressive.MAX_HOUR_ATTEMPTS and hrrr.fetched.count(0) == 1


def test_hours_that_never_appear_are_given_up_on_after_a_few_hours(env):
    hrrr = Hrrr(published=3)
    run = make_tick(env, hrrr)
    hours = tuple(range(6))
    assert run(hours=hours)["action"] == "advanced"
    assert run(hours=hours, minutes_after_cycle=120)["reason"] == "waiting for the next hour"
    result = run(hours=hours, minutes_after_cycle=60 * 4)
    assert result["action"] == "built" and result["missing_hours"] == [3, 4, 5]


def test_nothing_is_published_until_an_hour_exists(env):
    result = make_tick(env, Hrrr(published=0))()
    assert result["action"] == "skip" and not (env.root / CYCLE_ID).exists()
    assert job.read_status(env.root)["state"] == "waiting_for_hrrr"


def test_a_failed_upload_is_retried_with_the_next_hour(env):
    env.r2.fail_times = 1000                                           # R2 is down for the first tick
    hrrr = Hrrr(published=2)
    run = make_tick(env, hrrr)
    assert run()["advanced"] == [0, 1]                                 # the build carries on; only the CDN is behind
    assert not env.r2.objects
    env.r2.fail_times = 0
    hrrr.published = 3
    run()
    key = f"datasets/hrrr-hourly/runs/{CYCLE_ID}/forecast.zarr/temp_f"
    assert {f"{key}/0.0.0", f"{key}/1.0.0", f"{key}/2.0.0"} <= set(env.r2.objects)   # everything that was missed goes up


def test_without_r2_credentials_the_build_still_runs(env):
    hrrr = Hrrr(published=2)
    out = progressive.advance(now=CYCLE + timedelta(hours=1), hours=ALL_HOURS, root=env.root, peak_root=env.peak,
                              fetch_hour=hrrr.fetch, derive_hour=lambda p, e: hour_fields(hrrr.fetched[-1]),
                              hour_ready=hrrr.ready, wait=lambda r: None, candidates=[CYCLE])
    assert out["advanced"] == [0, 1]


def test_the_scheduler_flag_and_pause_switch(env, monkeypatch):
    monkeypatch.setenv("SMF_HRRR_HOURLY_PROGRESSIVE", "0")
    assert not progressive.enabled()
    monkeypatch.delenv("SMF_HRRR_HOURLY_PROGRESSIVE")
    assert progressive.enabled()
    from services.hrrr_hourly import controls
    controls.save_settings(None, True, "test", env.root)
    hrrr = Hrrr(published=3)
    assert make_tick(env, hrrr)()["reason"] == "paused" and hrrr.fetched == []


def test_a_one_shot_build_does_not_mistake_a_partial_progressive_cycle_for_done(env):
    make_tick(env, Hrrr(published=2))()
    assert job.prefetch_cycle(CYCLE, hours=(0, 1), root=env.root, hour_ready=lambda c, h: True, wait=lambda r: None,
                              fetch_hour=lambda c, h: (fake_profiles(), {}), derive_hour=lambda p, e: hour_fields(0)
                              )["reason"] != "already built"


# --- peak fire danger product ------------------------------------------------------------------------------

def test_the_peak_day_fills_in_and_is_cached_for_good_once_its_window_is_complete(env):
    import xarray as xr

    hrrr = Hrrr(published=6)                                           # f03..f05 of the window (3 of its 12 hours)
    run = make_tick(env, hrrr)
    run()
    manifest = store.read_manifest(CYCLE_ID, env.peak)
    assert manifest["product"] == "hrrr_fire_danger_peak" and manifest["estimate"] is True
    assert manifest["forecast_hours"] == [0] and manifest["step_labels"] == ["Mon Oct 19"]
    assert manifest["volatile_steps"] == [0] and manifest["days"][0]["window_present"] == 3
    assert manifest["days"][0]["window_total"] == 12 and manifest["complete"] is False
    peak_key = f"datasets/hrrr-fire-danger-peak/runs/{CYCLE_ID}/forecast.zarr/peak_fire_danger/0.0.0"
    assert env.r2.cache[peak_key] == cdn_datasets.LIVE_CACHE           # may still change: short cache
    ds = xr.open_zarr(env.peak / CYCLE_ID / "forecast.zarr")
    assert float(ds["max_temp_f"].isel(day=0).values[0, 0]) == pytest.approx(65.0)   # f05

    hrrr.published = 19
    run()
    manifest = store.read_manifest(CYCLE_ID, env.peak)
    assert manifest["complete"] is True and manifest["volatile_steps"] == [] and manifest["days"][0]["complete"] is True
    assert env.r2.cache[peak_key] == cdn_datasets.IMMUTABLE_CACHE      # final: cached for good
    ds = xr.open_zarr(env.peak / CYCLE_ID / "forecast.zarr")
    assert float(ds["max_temp_f"].isel(day=0).values[0, 0]) == pytest.approx(74.0)   # f14, the last window hour
    assert float(ds["peak_fire_danger"].isel(day=0).values[0, 0]) == 3                # min(14 // 4, 4)
    assert float(ds["peak_hour_local"].isel(day=0).values[0, 0]) == 19.0              # f12 (7 PM CDT) is the first hour at 3
    assert env.r2.json("datasets/hrrr-fire-danger-peak/latest.json")["run"] == CYCLE_ID


def test_peak_router_serves_the_manifest_zarr_and_a_point(env, monkeypatch):
    make_tick(env, Hrrr(published=19))()
    app = FastAPI()
    for name in ("chunk_limiter", "data_limiter", "point_limiter", "status_limiter"):
        monkeypatch.setattr(peak_router, name, RateLimiter(f"t_{name}", 10_000))
    app.include_router(peak_router.router)
    client = TestClient(app)
    runs = client.get("/hrrr-fire-danger-peak/runs").json()
    assert runs["latest"] == CYCLE_ID and runs["runs"][0]["step_labels"] == ["Mon Oct 19"]
    assert client.get("/hrrr-fire-danger-peak/runs/latest").json()["cycle"] == CYCLE_ID
    assert client.get(f"/hrrr-fire-danger-peak/runs/{CYCLE_ID}/zarr/.zmetadata").status_code == 200
    assert client.get(f"/hrrr-fire-danger-peak/runs/{CYCLE_ID}/zarr/../../manifest.json").status_code in (400, 404)
    point = client.get(f"/hrrr-fire-danger-peak/runs/{CYCLE_ID}/point", params={"lat": 37.4, "lon": -92.5}).json()
    assert point["lead_hours"] == [0] and point["step_labels"] == ["Mon Oct 19"] and "peak_fire_danger" in point["series"]
    assert client.get("/hrrr-fire-danger-peak/status").json()["serving_cycle"] == CYCLE_ID


def test_hourly_router_lists_a_partial_run_but_latest_waits(env, monkeypatch):
    make_tick(env, Hrrr(published=2))()
    app = FastAPI()
    for name in ("chunk_limiter", "data_limiter", "point_limiter", "status_limiter"):
        monkeypatch.setattr(router_module, name, RateLimiter(f"h_{name}", 10_000))
    app.include_router(router_module.router)
    client = TestClient(app)
    runs = client.get("/hrrr-hourly/runs").json()
    assert runs["latest"] is None or runs["latest"] != CYCLE_ID
    assert runs["runs"][0]["cycle"] == CYCLE_ID and runs["runs"][0]["complete"] is False
    assert client.get(f"/hrrr-hourly/runs/{CYCLE_ID}").json()["forecast_hours"] == [0, 1]
    live = client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/zarr/temp_f/1.0.0")
    assert live.status_code == 200 and live.headers["cache-control"] == "public, max-age=30"
    point = client.get(f"/hrrr-hourly/runs/{CYCLE_ID}/point", params={"lat": 37.4, "lon": -92.5})
    assert point.status_code == 200 and point.json()["lead_hours"] == [0, 1]


# --- the CDN index ------------------------------------------------------------------------------------------

def test_an_unready_run_never_becomes_latest_and_does_not_move_the_pointer(tmp_path):
    r2 = FakeR2()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}")
    kwargs = dict(title="t", kind="k", schema_version=1, stable=[], changing=[], client=r2)
    older = {"id": "a", "time_utc": "2026-10-19T00:00:00Z", "ready": True}
    cdn_datasets.publish_update("d", "a", run_dir, older, complete=True, **kwargs)
    newer = {"id": "b", "time_utc": "2026-10-19T01:00:00Z", "ready": False}
    cdn_datasets.publish_update("d", "b", run_dir, newer, complete=False, **kwargs)
    assert r2.json("datasets/d/latest.json")["run"] == "a" and r2.json("datasets/d/runs.json")["latest"] == "a"
    assert [row["id"] for row in r2.json("datasets/d/runs.json")["runs"]] == ["b", "a"]
    cdn_datasets.publish_update("d", "b", run_dir, {**newer, "ready": True}, complete=False, **kwargs)
    assert r2.json("datasets/d/latest.json")["run"] == "b"
    # a run that is not ready and has nothing before it is listed without a pointer
    fresh = FakeR2()
    cdn_datasets.publish_update("d", "c", run_dir, {"id": "c", "time_utc": "2026-10-19T02:00:00Z", "ready": False},
                                complete=False, **{**kwargs, "client": fresh})
    assert "datasets/d/latest.json" not in fresh.objects and fresh.json("datasets/d/runs.json")["latest"] is None


def test_live_store_round_trip_and_metadata_list(tmp_path):
    lat, lon = grid()
    element = ELEMENTS[0]
    path = tmp_path / "forecast.zarr"
    live_store.create(path, [element], 4, lat, lon, np.arange(4), np.arange(4))
    assert live_store.exists(path)
    chunk = live_store.write_step(path, element.name, 2, np.full((NY, NX), 12.34), element.scale)
    assert chunk == f"{element.name}/2.0.0"
    assert live_store.read_step(path, element.name, 2)[0, 0] == pytest.approx(round(12.34 / element.scale) * element.scale, rel=1e-5)
    assert live_store.read_step(path, element.name, 0) is None            # never written: no chunk file
    files = live_store.metadata_files(path, [element.name])
    assert ".zmetadata" in files and f"{element.name}/.zarray" in files and not any(f.endswith("/2.0.0") for f in files if f.startswith(element.name))


def test_the_daily_forecasts_shared_cache_is_built_once_when_its_hours_are_in(env, monkeypatch):
    from services import hrrr_shared_cache

    built = []
    monkeypatch.setattr(hrrr_shared_cache, "required_hours", lambda: (1, 2))
    monkeypatch.setattr(hrrr_shared_cache, "build_shared_cache", lambda cycle_time, root: built.append(cycle_time) or {"ok": 1})
    hrrr = Hrrr(published=2)
    run = make_tick(env, hrrr)
    run()
    assert built == []                                              # f02 is not in yet
    hrrr.published = 4
    run()
    run()
    assert built == [CYCLE]                                         # once, and only for the 12z run
    older = datetime(2026, 10, 19, 6, tzinfo=timezone.utc)
    progressive.advance(now=older + timedelta(hours=1), hours=ALL_HOURS, root=env.root, peak_root=env.peak,
                        fetch_hour=hrrr.fetch, derive_hour=lambda p, e: hour_fields(hrrr.fetched[-1]),
                        hour_ready=hrrr.ready, wait=lambda r: None, candidates=[older], client=env.r2)
    assert built == [CYCLE]


def test_a_leftover_prefetch_cron_makes_one_quick_pass(env, monkeypatch, capsys):
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "hrrr_hourly_forecast.py"
    spec = importlib.util.spec_from_file_location("hrrr_hourly_forecast_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(progressive, "advance", lambda **kwargs: calls.append(kwargs) or {"action": "skip"})
    assert module._prefetch(CYCLE, ALL_HOURS) == 0
    assert calls == [{"hours": ALL_HOURS, "candidates": [CYCLE]}]
    monkeypatch.setenv("SMF_HRRR_HOURLY_PROGRESSIVE", "0")
    monkeypatch.setattr(job, "prefetch_cycle", lambda **kwargs: {"action": "prefetched"})
    assert module._prefetch(CYCLE, ALL_HOURS) == 0 and len(calls) == 1      # switched off: the old polling prefetch

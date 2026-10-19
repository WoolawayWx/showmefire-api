from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from services import hrrr_shared_cache as sc

CYCLE = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)
CYCLE_ID = "20261016_12z"
HOURS = tuple(range(4, 16))
NY, NX = 6, 7


def grid():
    lat, lon = np.meshgrid(np.linspace(37.0, 38.0, NY), np.linspace(-93.0, -92.0, NX), indexing="ij")
    return lat, lon


def write_work_files(root, hours=HOURS, skip_key=None, gust=True):
    """Per-hour work files shaped like job._save_work_file writes them (raw_* keys beside the derived ones)."""
    work = root / ".work" / CYCLE_ID
    work.mkdir(parents=True, exist_ok=True)
    lat, lon = grid()
    for lead in hours:
        raw = {"raw_t2m_k": np.full((NY, NX), 280.0 + lead, dtype="float32"),
               "raw_rh2": np.full((NY, NX), 40.0 + lead, dtype="float32"),
               "raw_u10": np.full((NY, NX), 2.0 + lead, dtype="float32"),
               "raw_v10": np.full((NY, NX), -1.0 - lead, dtype="float32")}
        if gust:
            raw["raw_gust"] = np.full((NY, NX), 8.0 + lead, dtype="float32")
        if skip_key:
            raw.pop(skip_key, None)
        np.savez(work / f"f{lead:02d}.npz", lat=lat, lon=lon, temp_f=np.zeros((NY, NX), dtype="float32"), **raw)
    return work


def fake_precip(hours=HOURS, shift=0.0):
    """What DailyForecast's `FH.xarray(':APCP:')` gives: (step, y, x) with 2-D lat/lon and an accumulation attr."""
    lat, lon = grid()
    data = np.cumsum(np.full((len(hours), NY, NX), 0.1 + shift, dtype="float32"), axis=0)
    return xr.Dataset(
        {"tp": (("step", "y", "x"), data, {"units": "kg m**-2", "GRIB_stepType": "accum"})},
        coords={"step": ("step", np.array(hours, dtype="timedelta64[h]").astype("timedelta64[ns]")),
                "latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon),
                "time": np.datetime64("2026-10-16T12:00:00", "ns")})


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_HRRR_SHARED_CACHE_DIR", str(tmp_path / "shadow"))
    monkeypatch.setenv("SMF_HRRR_CACHE_DIR", str(tmp_path / "live"))
    monkeypatch.delenv("FORECAST_USE_SHARED_HRRR", raising=False)
    monkeypatch.delenv("FORECAST_START_HOUR", raising=False)
    monkeypatch.delenv("FORECAST_END_HOUR", raising=False)
    return tmp_path


def test_hours_and_file_name_match_what_dailyforecast_looks_for(dirs):
    assert sc.required_hours() == HOURS
    assert sc.cache_name(CYCLE, HOURS) == "hrrr_20261016_12z_f04-15.nc"


def test_stack_raw_fields_builds_one_cube_per_variable(dirs):
    work = write_work_files(dirs)
    lat, lon, stacks = sc.stack_raw_fields(work, HOURS)
    assert lat.shape == (NY, NX) and set(stacks) == {"t2m", "r2", "u10", "v10", "i10fg"}
    assert stacks["t2m"].shape == (12, NY, NX) and stacks["t2m"][0, 0, 0] == 284.0 and stacks["t2m"][-1, 0, 0] == 295.0


def test_gust_is_optional_but_the_four_main_fields_are_not(dirs):
    _, _, stacks = sc.stack_raw_fields(write_work_files(dirs, gust=False), HOURS)
    assert "i10fg" not in stacks and "t2m" in stacks
    with pytest.raises(sc.SharedCacheUnavailable, match="raw_u10"):
        sc.stack_raw_fields(write_work_files(dirs, skip_key="raw_u10"), HOURS)


def test_a_missing_hour_is_unavailable(dirs):
    work = write_work_files(dirs, hours=tuple(h for h in HOURS if h != 9))
    with pytest.raises(sc.SharedCacheUnavailable, match="f09"):
        sc.stack_raw_fields(work, HOURS)


def test_assembled_file_has_what_dailyforecast_needs_to_treat_it_as_a_cache_hit(dirs):
    write_work_files(dirs)
    result = sc.build_shared_cache(CYCLE, dirs, live=False, precip_loader=lambda run, hours: fake_precip())
    path = dirs / "shadow" / "hrrr_20261016_12z_f04-15.nc"
    assert result["path"] == str(path) and result["live"] is False and path.is_file()
    with xr.open_dataset(path, decode_cf=False) as ds:        # opened exactly as DailyForecast opens a cache
        for name in ("t2m", "r2", "u10", "v10", "i10fg", "tp"):
            assert name in ds.variables
        assert ds.sizes["step"] == 12 and ds["t2m"].dims == ("step", "y", "x")
        assert ds["t2m"].attrs["units"] == "K" and ds["r2"].attrs["units"] == "%"
        assert ds["latitude"].shape == (NY, NX) and ds["tp"].attrs["units"] == "kg m**-2"
        assert float(ds["u10"][0, 0, 0]) == 6.0


def test_live_mode_writes_the_path_dailyforecast_reads_and_leaves_no_temp_file(dirs):
    write_work_files(dirs)
    result = sc.build_shared_cache(CYCLE, dirs, live=True, precip_loader=lambda run, hours: fake_precip())
    assert result["live"] is True and (dirs / "live" / "hrrr_20261016_12z_f04-15.nc").is_file()
    assert not list((dirs / "live").glob("*.tmp"))


def test_switch_defaults_from_the_environment(dirs, monkeypatch):
    write_work_files(dirs)
    assert sc.build_shared_cache(CYCLE, dirs, precip_loader=lambda r, h: fake_precip())["live"] is False
    monkeypatch.setenv("FORECAST_USE_SHARED_HRRR", "true")
    assert sc.build_shared_cache(CYCLE, dirs, precip_loader=lambda r, h: fake_precip())["live"] is True


def test_nothing_is_written_when_precipitation_cannot_be_fetched_or_does_not_line_up(dirs):
    write_work_files(dirs)

    def boom(run, hours):
        raise RuntimeError("HRRR down")

    with pytest.raises(sc.SharedCacheUnavailable, match="precipitation download failed"):
        sc.build_shared_cache(CYCLE, dirs, live=True, precip_loader=boom)
    with pytest.raises(sc.SharedCacheUnavailable, match="steps"):
        sc.build_shared_cache(CYCLE, dirs, live=True, precip_loader=lambda r, h: fake_precip(hours=tuple(range(4, 15))))
    wrong_grid = fake_precip()
    wrong_grid["latitude"] = wrong_grid["latitude"] + 0.5
    with pytest.raises(sc.SharedCacheUnavailable, match="grid"):
        sc.build_shared_cache(CYCLE, dirs, live=True, precip_loader=lambda r, h: wrong_grid)
    assert not (dirs / "live").exists() or not list((dirs / "live").glob("*.nc"))


def test_old_shadow_files_are_pruned(dirs):
    shadow = dirs / "shadow"
    shadow.mkdir()
    old, new = shadow / "hrrr_20261001_12z_f04-15.nc", shadow / "hrrr_20261016_12z_f04-15.nc"
    old.write_bytes(b"x"), new.write_bytes(b"x")
    import os, time
    stale = time.time() - 5 * 86400
    os.utime(old, (stale, stale))
    assert sc.prune_shadow(shadow) == [old.name] and new.exists() and not old.exists()


# --- compare_caches -----------------------------------------------------------------------------------------------

def make_pair(dirs, change=None):
    """(shared file, 'DailyForecast's own' file). The second is the same data, with `change` applied to it."""
    work = write_work_files(dirs)
    lat, lon, stacks = sc.stack_raw_fields(work, HOURS)
    name = sc.cache_name(CYCLE, HOURS)
    shared_ds = sc.assemble(fake_precip(), lat, lon, stacks, HOURS)
    real_ds = sc.assemble(fake_precip(), lat, lon, stacks, HOURS)
    if change:
        change(real_ds)
    shared, real = dirs / "shadow" / name, dirs / "live" / name
    sc.write_atomic(shared_ds, shared)
    sc.write_atomic(real_ds, real)
    return shared, real


def test_identical_caches_match(dirs):
    shared, real = make_pair(dirs)
    report = sc.compare_caches(shared, real)
    assert report["ok"] and not report["problems"]
    assert report["variables"]["t2m"]["max_abs_diff"] == 0 and set(report["variables"]) >= {"t2m", "r2", "u10", "v10", "tp"}


def test_small_float_noise_is_allowed_but_a_real_difference_is_not(dirs):
    def noise(ds):   # float32 rounding scale, with the units kept
        ds["t2m"] = (ds["t2m"].dims, ds["t2m"].values + np.float32(5e-4), dict(ds["t2m"].attrs))
    assert sc.compare_caches(*make_pair(dirs, noise))["ok"]

    def shifted(ds):
        ds["t2m"] = ds["t2m"] + 0.5
    report = sc.compare_caches(*make_pair(dirs, shifted))
    assert not report["ok"] and any("t2m differs" in problem for problem in report["problems"])


def test_precipitation_must_be_identical_and_units_must_agree(dirs):
    def different_precip(ds):
        ds["tp"] = ds["tp"] + 1e-3
    assert any("tp differs" in p for p in sc.compare_caches(*make_pair(dirs, different_precip))["problems"])

    def different_units(ds):
        ds["r2"].attrs["units"] = "percent"
    assert any("units" in p for p in sc.compare_caches(*make_pair(dirs, different_units))["problems"])


def test_a_variable_missing_from_one_file_is_reported(dirs):
    report = sc.compare_caches(*make_pair(dirs, lambda ds: ds.__delitem__("v10")))
    assert not report["ok"] and any("v10 is only in the shared file" in p for p in report["problems"])

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forecast import ExtendedForecast as ef
from forecast import extended_verify as ev

CYCLE = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
STATIONS = [{"station_id": "AAA", "lat": 38.0, "lon": -92.0}, {"station_id": "BBB", "lat": 37.0, "lon": -93.0},
            {"station_id": "FAR", "lat": 10.0, "lon": -50.0}]


def model(base=290.0, rain=(0.0, 0.0, 1.0, 1.0), size=31):
    """A tiny HRRR-like dataset: 4 hourly leads (f28..f31 of the 12z run) on a 0.1 degree lat/lon-like grid."""
    lon = np.linspace(-94, -91, size)
    lat = np.linspace(36, 39, size)
    lon2, lat2 = np.meshgrid(lon, lat)
    times = np.datetime64("2026-11-20T16") + np.arange(4).astype("timedelta64[h]")
    ds = xr.Dataset(coords={"time": times})
    shape = (4, size, size)
    ds["t2m"] = (("time", "y", "x"), np.full(shape, base) + np.arange(4)[:, None, None])        # warms 1 K per hour
    ds["r2"] = (("time", "y", "x"), np.full(shape, 40.0) - np.arange(4)[:, None, None] * 2)    # dries 2 % per hour
    ds["u10"] = (("time", "y", "x"), np.full(shape, 6.0))
    ds["v10"] = (("time", "y", "x"), np.full(shape, 8.0))                                      # 10 m/s
    ds["tp"] = (("time", "y", "x"), np.broadcast_to(np.array(rain).reshape(4, 1, 1), shape).copy())
    return ds, lon2, lat2


def snap(ds, lon, lat, model_name="hrrr"):
    return ev.snapshot(ds, (28, 31), CYCLE, STATIONS, day=2, model=model_name, lon=lon, lat=lat,
                       precip_increments=ef.precip_increments, precip_name=ef.precip_name)


def run_snapshot(*names):
    """The combined, one-file-per-run dataset for the named models (all from the same fake model)."""
    ds, lon, lat = model()
    return ev.combine({name: snap(ds, lon, lat, name) for name in names})


def test_snapshot_uses_display_units_and_drops_stations_far_from_the_grid():
    ds, lon, lat = model()
    data = snap(ds, lon, lat)
    assert list(data.station_id.values) == ["AAA", "BBB"]                    # FAR is thousands of km away
    assert list(data.lead_hour.values) == [28, 29, 30, 31]
    assert np.allclose(data.temp_f.values[0], (290.0 - 273.15) * 9 / 5 + 32)
    assert np.allclose(data.wind_speed_mph.values, 10 * ev.MPH_PER_MS)
    assert np.allclose(data.precip_mm_1h.values[:, 0], [0.0, 0.0, 1.0, 0.0])  # run total differenced into hours
    assert np.isnan(data.wind_gust_mph.values).all()                         # no gust field in this dataset


def test_models_share_one_packed_file_that_round_trips_within_each_elements_step(tmp_path):
    ds, lon, lat = model()
    both = run_snapshot("hrrr", "rrfs")
    assert list(both.model.values) == ["hrrr", "rrfs"] and both.temp_f.dims == ("model", "lead_hour", "station")
    path = ev.write_snapshot(both, tmp_path, 2, CYCLE)
    assert path.name == "day2_20261119_12z.nc"
    back = ev.read_snapshot(path)
    assert np.abs(back.temp_f.values - both.temp_f.values).max() <= 0.05 + 1e-9        # step 0.1 F
    assert np.abs(back.rh_pct.values - both.rh_pct.values).max() <= 0.25 + 1e-9        # step 0.5 %
    assert np.isnan(back.wind_gust_mph.values).all()                                     # NaN survives the int16 packing
    assert back.attrs["cycle_utc"] == "2026-11-19T12:00:00Z" and "model" not in back.attrs
    one = ev.write_snapshot(run_snapshot("hrrr"), tmp_path / "single", 2, CYCLE)
    assert path.stat().st_size < 2 * one.stat().st_size                                  # the per-file overhead is paid once


def test_combine_keeps_only_stations_every_model_has_and_skips_missing_models():
    ds, lon, lat = model()
    a, b = snap(ds, lon, lat, "hrrr"), snap(ds, lon, lat, "rrfs")
    both = ev.combine({"hrrr": a, "rrfs": b.isel(station=[0]), "none": None})
    assert list(both.station_id.values) == ["AAA"] and list(both.model.values) == ["hrrr", "rrfs"]
    assert ev.combine({"hrrr": None}) is None and ev.record({"hrrr": None}, "unused", 2, CYCLE) == []


def observations(offset_temp_c=0.0):
    """Hourly obs for both stations over the window: temp rises 1 C/h from 17 C, RH falls 2 %/h from 40."""
    rows = []
    for station in ("AAA", "BBB"):
        for h in range(4):
            rows.append({"station_id": station, "hour": pd.Timestamp(CYCLE + timedelta(hours=28 + h)),
                         "fm10": np.nan, "temp_c": 17.0 + h + offset_temp_c, "rh": 40.0 - 2 * h, "wind_ms": 9.0,
                         "precip_mm": 0.5 if h == 2 else 0.0})
    return pd.DataFrame(rows)


def test_scoring_waits_for_the_window_plus_latency_then_compares_window_statistics():
    data = run_snapshot("hrrr", "rrfs")
    provider = lambda ids, start, end: observations()
    window_end = CYCLE + timedelta(hours=31)
    assert ev.score_snapshot(data, provider, window_end + timedelta(hours=2)) is None          # observations still arriving
    rows = ev.score_snapshot(data, provider, window_end + timedelta(hours=7))
    assert len(rows) == 2 * 2 * len(ev.VARIABLES) and set(rows.model) == {"hrrr", "rrfs"}
    aaa = rows[(rows.station_id == "AAA") & (rows.model == "hrrr")].set_index("variable")
    assert aaa.loc["rh_min_pct", "forecast"] == pytest.approx(34.0, abs=0.3) and aaa.loc["rh_min_pct", "observed"] == 34.0
    assert aaa.loc["wind_max_mph", "error"] == pytest.approx(10 * ev.MPH_PER_MS - 9 * ev.MPH_PER_MS, abs=0.1)
    assert aaa.loc["precip_mm", "forecast"] == pytest.approx(1.0) and aaa.loc["precip_mm", "observed"] == 0.5


def test_an_observed_statistic_needs_most_of_the_window_and_is_never_zero_filled():
    data = run_snapshot("hrrr")
    obs = observations()
    sparse = obs[obs["hour"] < pd.Timestamp("2026-11-20T18:00:00Z")]                    # only 2 of 4 hours
    rows = ev.score_snapshot(data, lambda *_: sparse, CYCLE + timedelta(hours=60))
    assert rows["observed"].isna().all() and rows["error"].isna().all()
    missing_hour = obs.assign(precip_mm=np.where(obs["hour"].dt.hour == 18, np.nan, 0.0))
    rows = ev.score_snapshot(data, lambda *_: missing_hour, CYCLE + timedelta(hours=60))
    assert np.isnan(rows[(rows.station_id == "AAA") & (rows.variable == "precip_mm")]["observed"].iloc[0])   # a missing hour: no total
    assert rows[(rows.station_id == "AAA") & (rows.variable == "temp_max_f")]["observed"].notna().all()      # extremes still fine


def test_score_matured_scores_each_snapshot_once(tmp_path):
    ev.write_snapshot(run_snapshot("hrrr", "rrfs"), tmp_path, 2, CYCLE)
    now = CYCLE + timedelta(hours=60)
    assert ev.score_matured(tmp_path, lambda *_: observations(), now) == 1
    assert ev.score_matured(tmp_path, lambda *_: observations(), now) == 0               # already scored
    summary = ev.write_summary(tmp_path, now)
    assert {m["model"] for m in summary["metrics"]} == {"hrrr", "rrfs"} and (tmp_path / "summary.json").exists()


def scores(hrrr_error, rrfs_error, days=10, stations=20):
    rows = []
    for d in range(days):
        cycle = f"2026-11-{10 + d:02d}T12:00:00Z"
        for s in range(stations):
            for variable in ("temp_max_f", "rh_min_pct", "wind_max_mph"):
                for model_name, error in (("hrrr", hrrr_error), ("rrfs", rrfs_error)):
                    rows.append({"day": 2, "model": model_name, "cycle_utc": cycle, "valid_date": f"2026-11-{11 + d:02d}",
                                 "station_id": f"S{s}", "variable": variable, "forecast": 0.0, "observed": 0.0, "error": error})
    return pd.DataFrame(rows)


def test_shadow_weights_favour_the_better_model_but_stay_within_30_to_70():
    better = ev.propose_weights(scores(hrrr_error=1.0, rrfs_error=3.0))             # HRRR much better: capped at 70
    assert better["overall"] == 0.7 and better["status"] == "shadow" and better["applied"] is False
    worse = ev.propose_weights(scores(hrrr_error=3.0, rrfs_error=1.0))              # RRFS much better: floor of 30
    assert worse["overall"] == 0.3
    close = ev.propose_weights(scores(hrrr_error=2.0, rrfs_error=2.2))              # about even: near 50
    assert 0.45 < close["overall"] < 0.55
    assert all(v["primary_share"] is not None for v in close["per_variable"].values())


def test_shadow_weights_keep_the_prior_until_there_is_enough_record():
    assert ev.propose_weights(pd.DataFrame())["overall"] == ev.WEIGHT_PRIOR
    thin = ev.propose_weights(scores(1.0, 3.0, days=2, stations=5))                 # 10 pairs per variable, 2 days
    assert thin["overall"] == ev.WEIGHT_PRIOR and thin["reason"] and all(v["primary_share"] is None for v in thin["per_variable"].values())


def test_old_snapshots_and_scores_are_pruned(tmp_path):
    old = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
    ev.write_snapshot(run_snapshot("hrrr"), tmp_path, 2, old)
    ev.write_snapshot(run_snapshot("hrrr"), tmp_path, 2, CYCLE)
    assert ev.prune_snapshots(tmp_path, CYCLE + timedelta(days=1), keep_days=60) == 1
    assert [p.name for p in (tmp_path / "forecast").glob("*.nc")] == ["day2_20261119_12z.nc"]


def test_record_snapshots_saves_one_file_with_every_model_and_never_raises(tmp_path, monkeypatch):
    ds, lon, lat = model()
    ds["longitude"], ds["latitude"] = (("y", "x"), lon), (("y", "x"), lat)
    monkeypatch.setattr(ev, "station_list", lambda *a, **k: STATIONS)
    cfg = ef.config_for(2, 12)
    ef.record_snapshots({"hrrr": ds, "rrfs": ds.copy(deep=True), "blend": ds.copy(deep=True)}, cfg, CYCLE, tmp_path)
    files = list((tmp_path / "forecast").glob("*.nc"))
    assert [f.name for f in files] == ["day2_20261119_12z.nc"]
    assert list(ev.read_snapshot(files[0]).model.values) == ["hrrr", "rrfs", "blend"]
    # a failure inside (here: no stations table) is logged, not raised: the forecast must not depend on this
    monkeypatch.setattr(ev, "station_list", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no database")))
    ef.record_snapshots({"hrrr": ds}, cfg, CYCLE, tmp_path / "elsewhere")
    assert not (tmp_path / "elsewhere").exists()


def test_peak_block_scores_only_the_hours_the_map_shows_and_the_window_scores_all_of_them():
    ds, lon, lat = model()
    data = ev.combine({"hrrr": snap(ds, lon, lat)}, {"peak_start_lead": 30, "peak_end_lead": 31})
    rows = ev.score_snapshot(data, lambda *_: observations(), CYCLE + timedelta(hours=60))
    assert set(rows.scope) == {"window", "peak"}
    rh = rows[(rows.station_id == "AAA") & (rows.variable == "rh_min_pct")].set_index("scope")
    assert rh.loc["window", "observed"] == 34.0 and rh.loc["peak", "observed"] == 34.0      # RH bottoms out in the last hour
    t = rows[(rows.station_id == "AAA") & (rows.variable == "temp_max_f")].set_index("scope")
    assert t.loc["peak", "observed"] == pytest.approx((17 + 3) * 9 / 5 + 32)                  # leads 30-31 are obs hours 2-3
    one_hour_missing = observations()
    one_hour_missing = one_hour_missing[one_hour_missing["hour"] != pd.Timestamp(CYCLE + timedelta(hours=31))]
    rows = ev.score_snapshot(data, lambda *_: one_hour_missing, CYCLE + timedelta(hours=60))
    peak = rows[(rows.scope == "peak") & (rows.variable == "temp_max_f")]
    window = rows[(rows.scope == "window") & (rows.variable == "temp_max_f")]
    assert peak["observed"].isna().all() and window["observed"].notna().all()               # a 2-hour block needs both hours


def test_snapshots_without_a_peak_block_score_the_window_only():
    rows = ev.score_snapshot(run_snapshot("hrrr"), lambda *_: observations(), CYCLE + timedelta(hours=60))
    assert set(rows.scope) == {"window"}


def test_shadow_weights_ignore_the_noisy_peak_block_scores():
    table = scores(hrrr_error=1.0, rrfs_error=3.0).assign(scope="window")
    peak = scores(hrrr_error=9.0, rrfs_error=0.1).assign(scope="peak")                       # would flip the weights if it counted
    assert ev.propose_weights(pd.concat([table, peak], ignore_index=True))["overall"] == 0.7
    assert {m["scope"] for m in ev.metrics(pd.concat([table, peak], ignore_index=True))} == {"window", "peak"}
    assert {m["scope"] for m in ev.metrics(scores(1.0, 3.0))} == {"window"}                  # old tables had no scope column

"""Tests for services/fire_outlook/driver.py with synthetic member datasets."""
from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from services.fire_outlook import driver, moisture
from services.rule_uncertainty import MISSING_CATEGORY, category_vectorized

CYCLE = datetime(2026, 10, 29, 0, tzinfo=timezone.utc)
GRID = (3, 4)


def _thresholds():
    from core.fire_danger import RULE_SPEC
    return RULE_SPEC["thresholds"]


def categorize(fm, rh, w):
    return category_vectorized(fm, rh, w, _thresholds())


def member(temp=25.0, rh=60.0, wind=3.0, rain=0.0, swe=0.0, n=169):
    times = np.datetime64("2026-10-29T00", "h") + np.arange(n).astype("timedelta64[h]")
    shape = (n,) + GRID

    def full(v):
        return (("time", "y", "x"), np.full(shape, v, dtype="float32"))
    return xr.Dataset({
        "temperature_2m": full(temp), "relative_humidity_2m": full(rh),
        "wind_u_10m": full(wind), "wind_v_10m": full(0.0),
        "precipitation_increment": full(rain), "snow_water_equivalent": full(swe),
    }, coords={"time": times})


def init(ds, anchor=None):
    f = driver._fields(ds, CYCLE)
    return driver.seed_state(f, anchor)[0]


def test_calm_humid_member_is_low_everywhere():
    ds = member()
    out = driver.run_member(ds, CYCLE, init(ds), categorize)
    assert out["max_category"].shape == (28,) + GRID
    assert out["max_category"].max() == 0
    assert out["fm10_min"].shape == (28,) + GRID


def test_dry_windy_member_reaches_critical_and_counts_hours():
    th = _thresholds()
    wind_ms = (th["critical_wind"] + 6) / driver.MPS_TO_KNOTS
    ds = member(temp=30.0, rh=th["critical_rh"] - 3, wind=wind_ms)
    out = driver.run_member(ds, CYCLE, init(ds), categorize)
    assert out["max_category"][0].min() == 3
    assert (out["hours_critical"][0] == 6).all()


def test_wind_factor_changes_category():
    th = _thresholds()
    wind_ms = (th["critical_wind"] + 2) / driver.MPS_TO_KNOTS
    ds = member(temp=30.0, rh=th["critical_rh"] - 3, wind=wind_ms)
    full = driver.run_member(ds, CYCLE, init(ds), categorize, wind_factor=1.0)
    reduced = driver.run_member(ds, CYCLE, init(ds), categorize, wind_factor=0.5)
    assert full["max_category"].max() >= 3 and reduced["max_category"].max() < 3


def test_snow_forces_low_unknown_snow_does_not():
    th = _thresholds()
    wind_ms = (th["critical_wind"] + 6) / driver.MPS_TO_KNOTS
    kwargs = dict(temp=30.0, rh=th["critical_rh"] - 3, wind=wind_ms)
    snowy = member(swe=5.0, **kwargs)
    assert driver.run_member(snowy, CYCLE, init(snowy), categorize)["max_category"].max() == 0
    unknown = member(swe=np.nan, **kwargs)
    assert driver.run_member(unknown, CYCLE, init(unknown), categorize)["max_category"].max() == 3


def test_light_coarse_rain_wets_and_sums_precip():
    ds = member(temp=25.0, rh=30.0)
    dry = driver.run_member(ds, CYCLE, init(ds), categorize)
    rainy = member(temp=25.0, rh=30.0, rain=0.4)  # 1.2 mm per 3 h interval, 0.4 mm/h
    wet = driver.run_member(rainy, CYCLE, init(rainy), categorize, precip_interval_hours=3)
    assert wet["fm10_min"][0].mean() > dry["fm10_min"][0].mean()
    np.testing.assert_allclose(wet["precip_6h_mm"][0], 0.4 * 6, rtol=1e-5)


def test_missing_rain_is_nodata_not_dry():
    ds = member()
    ds["precipitation_increment"][50, 0, 0] = np.nan
    out = driver.run_member(ds, CYCLE, init(ds), categorize)
    w = 49 // 6
    assert out["max_category"][w, 0, 0] == MISSING_CATEGORY
    assert np.isnan(out["precip_6h_mm"][w, 0, 0])
    assert out["max_category"][w, 1, 1] != MISSING_CATEGORY


def test_incomplete_members_are_rejected():
    with pytest.raises(driver.IncompleteMember):
        driver.run_member(member(n=100), CYCLE, init(member()), categorize)
    ds = member().drop_vars("relative_humidity_2m")
    with pytest.raises(driver.IncompleteMember):
        driver.run_member(ds, CYCLE, init(member()), categorize)
    shifted = member().assign_coords(time=member().time + np.timedelta64(1, "h"))
    with pytest.raises(driver.IncompleteMember):
        driver.run_member(shifted, CYCLE, init(member()), categorize)


def test_anchor_and_quality_label():
    f = driver._fields(member(), CYCLE)
    assert driver.seed_state(f)[1] == "emc_seed"
    anchor = np.full(GRID, 6.0)
    state, label = driver.seed_state(f, anchor)
    assert label == "anchored_emc_seed" and (state["fm10"] == 6.0).all()


def test_ensemble_agreement_uses_full_member_denominator():
    th = _thresholds()
    wind_ms = (th["critical_wind"] + 6) / driver.MPS_TO_KNOTS
    hot = member(temp=30.0, rh=th["critical_rh"] - 3, wind=wind_ms)
    calm = member()
    members = [hot] * 10 + [calm] * 21
    out = driver.run_ensemble(iter(members), CYCLE, 31, GRID, init, categorize)
    np.testing.assert_allclose(out["agreement_elevated"][0], 10 / 31, rtol=1e-6)
    assert (out["agreement_critical"] <= out["agreement_elevated"] + 1e-9).all()
    assert out["n_members"] == 31
    assert out["fm10_min_p10"].shape == (28,) + GRID


def test_ensemble_refuses_partial_runs():
    with pytest.raises(ValueError):
        driver.run_ensemble(iter([member()] * 30), CYCLE, 31, GRID, init, categorize)
    with pytest.raises(driver.IncompleteMember):
        driver.run_ensemble(iter([member()] * 5 + [member(n=50)]), CYCLE, 6, GRID, init, categorize)

"""Tests for services/fire_outlook (moisture advance, windows, agreement)."""
import numpy as np
import pytest

from services.fire_outlook import moisture, windows
from services.rule_uncertainty import MISSING_CATEGORY, category_vectorized

GRID = (3, 4)


def _thresholds():
    from core.fire_danger import RULE_SPEC
    return RULE_SPEC["thresholds"]


# --- moisture ---------------------------------------------------------------

def test_advance_matches_existing_conditioning_step():
    from services import spread_rate_moisture as ref
    rng = np.random.default_rng(1)
    state = {k: rng.uniform(3, 25, GRID) for k in ("fm1", "fm10", "fm100")}
    temp, rh = rng.uniform(-5, 35, GRID), rng.uniform(5, 100, GRID)
    precip = rng.choice([0.0, 0.2, 0.6, 3.0], GRID)
    got = moisture.advance(state, temp, rh, precip)
    want = ref._advance_moisture(state, temp, rh, precip)
    for k in want:
        np.testing.assert_allclose(got[k], want[k])
    np.testing.assert_allclose(moisture.nelson_emc(temp, rh), ref.nelson_emc(temp, rh))


def test_rain_then_drying_and_continuity():
    n = 48
    temp = np.full((n,) + GRID, 25.0)
    rh = np.full((n,) + GRID, 30.0)
    rain = np.zeros((n,) + GRID)
    rain[5:8] = 4.0
    init = moisture.initial_state(temp[0], rh[0])
    series = moisture.run_member(init, temp, rh, rain)["fm10"]
    assert series[8].mean() > series[4].mean() + 1.0       # wetted
    assert series[-1].mean() < series[8].mean() - 1.0      # then dried
    # one continuous run equals two chained runs (no restart at the boundary)
    first = moisture.run_member(init, temp[:24], rh[:24], rain[:24])
    state = {k: v[-1] for k, v in first.items()}
    second = moisture.run_member(state, temp[24:], rh[24:], rain[24:])
    np.testing.assert_allclose(np.concatenate([first["fm10"], second["fm10"]]), series)


def test_overnight_recovery():
    n = 24
    temp = np.where(np.arange(n) < 12, 28.0, 8.0)[:, None, None] * np.ones((n,) + GRID)
    rh = np.where(np.arange(n) < 12, 25.0, 85.0)[:, None, None] * np.ones((n,) + GRID)
    series = moisture.run_member(moisture.initial_state(temp[0], rh[0]), temp, rh, np.zeros((n,) + GRID))["fm1"]
    assert series[-1].mean() > series[11].mean() + 5.0


def test_coarse_interval_rain_still_wets_fuel():
    """1.2 mm over six hours is 0.2 mm/h; the hourly > 0.5 rule alone would ignore it."""
    totals = np.full((1,) + GRID, 1.2)
    hourly, wet = moisture.allocate_precip(totals, 6)
    assert hourly.shape == (6,) + GRID and wet.all()
    np.testing.assert_allclose(hourly.sum(axis=0), 1.2)
    init = moisture.initial_state(np.full(GRID, 20.0), np.full(GRID, 40.0))
    temp, rh = np.full((6,) + GRID, 20.0), np.full((6,) + GRID, 40.0)
    plain = moisture.run_member(init, temp, rh, hourly)["fm1"][-1]
    wetted = moisture.run_member(init, temp, rh, hourly, wet)["fm1"][-1]
    assert (wetted > plain).all()


def test_missing_inputs_are_nodata_not_zero():
    state = moisture.initial_state(np.full(GRID, 20.0), np.full(GRID, 40.0))
    precip = np.zeros(GRID)
    precip[0, 0] = np.nan
    rh = np.full(GRID, 40.0)
    rh[1, 1] = np.nan
    out = moisture.advance(state, np.full(GRID, 20.0), rh, precip)
    assert np.isnan(out["fm10"][0, 0]) and np.isnan(out["fm10"][1, 1])
    assert np.isfinite(out["fm10"][2, 2])


def test_anchor_overrides_fm10_only_where_finite():
    anchor = np.full(GRID, np.nan)
    anchor[0, 0] = 7.0
    s = moisture.initial_state(np.full(GRID, 20.0), np.full(GRID, 40.0), anchor)
    assert s["fm10"][0, 0] == 7.0 and s["fm10"][1, 1] == s["fm1"][1, 1]


def test_interpolate_hourly_length_and_endpoints():
    v = np.array([0.0, 6.0, 12.0])[:, None, None] * np.ones((3,) + GRID)
    out = moisture.interpolate_hourly(v, 3)
    assert out.shape[0] == 7
    np.testing.assert_allclose(out[:, 0, 0], [0, 2, 4, 6, 8, 10, 12])


# --- windows ----------------------------------------------------------------

def test_28_gap_free_windows():
    b = windows.window_bounds()
    assert len(b) == 28 and b[0] == (0, 6) and b[-1] == (162, 168)
    assert all(b[i][1] == b[i + 1][0] for i in range(27))


def _hourly(value=0):
    return np.full((168,) + GRID, value, dtype="int8")


def test_boundary_assignment():
    cat = _hourly(0)
    cat[6, 0, 0] = 2      # first hour of window 1
    cat[5, 1, 1] = 2      # last hour of window 0
    s = windows.window_summary(cat)
    assert s["max_category"][0, 1, 1] == 2 and s["max_category"][0, 0, 0] == 0
    assert s["max_category"][1, 0, 0] == 2 and s["max_category"][1, 1, 1] == 0
    assert s["hours_elevated"][1, 0, 0] == 1


def test_concurrent_threshold_not_per_variable_max():
    th = _thresholds()
    fm_dry, rh_dry, wind_hi = th["elevated_fm"] - 1, th["critical_rh"] - 1, th["critical_wind"] + 5
    n = 168
    fm = np.full((n,) + GRID, fm_dry)
    rh = np.full((n,) + GRID, 90.0)          # humid...
    wind = np.full((n,) + GRID, 0.0)         # ...and calm...
    rh[3], wind[4] = rh_dry, wind_hi         # dry hour and windy hour never coincide
    cat = windows.hourly_categories(fm, rh, wind, lambda a, b, c: category_vectorized(a, b, c, th))
    assert windows.window_summary(cat)["max_category"][0].max() < 2
    rh[4] = rh_dry                           # now they coincide
    cat = windows.hourly_categories(fm, rh, wind, lambda a, b, c: category_vectorized(a, b, c, th))
    assert windows.window_summary(cat)["max_category"][0].min() == 3


def test_snow_forces_low_but_keeps_nodata():
    cat_in = _hourly(3)
    cat_in[0, 0, 0] = MISSING_CATEGORY
    snow = np.ones(GRID, dtype=bool)
    out = windows.hourly_categories(np.zeros((168,) + GRID), np.zeros((168,) + GRID), np.zeros((168,) + GRID),
                                    lambda *_: cat_in, snow_mask=snow)
    assert out[1, 0, 0] == 0 and out[0, 0, 0] == MISSING_CATEGORY


def test_missing_hour_makes_window_nodata():
    cat = _hourly(1)
    cat[2, 0, 0] = MISSING_CATEGORY
    s = windows.window_summary(cat)
    assert s["max_category"][0, 0, 0] == MISSING_CATEGORY and s["hours_elevated"][0, 0, 0] == 255
    assert s["max_category"][1, 0, 0] == 1


def test_agreement_denominator_and_critical_le_elevated():
    acc = windows.AgreementAccumulator(31, GRID)
    for m in range(31):
        mc = np.zeros((28,) + GRID, dtype="int8")
        if m < 10:
            mc[:, 0, 0] = 2      # 10 members reach Elevated
        if m < 4:
            mc[:, 0, 0] = 3      # 4 of those reach Critical
        if m < 31:
            mc[:, 1, 1] = 3      # all reach Critical
        acc.add(mc)
    out = acc.finalize()
    np.testing.assert_allclose(out["elevated"][0, 0, 0], 10 / 31, rtol=1e-6)
    np.testing.assert_allclose(out["critical"][0, 0, 0], 4 / 31, rtol=1e-6)
    assert out["elevated"][5, 1, 1] == 1.0
    assert (out["critical"] <= out["elevated"] + 1e-9).all()


def test_agreement_refuses_partial_and_marks_missing_member_nodata():
    acc = windows.AgreementAccumulator(3, GRID)
    acc.add(np.zeros((28,) + GRID, dtype="int8"))
    with pytest.raises(ValueError):
        acc.finalize()
    bad = np.zeros((28,) + GRID, dtype="int8")
    bad[4, 2, 2] = MISSING_CATEGORY
    acc.add(bad)
    acc.add(np.zeros((28,) + GRID, dtype="int8"))
    out = acc.finalize()
    assert np.isnan(out["elevated"][4, 2, 2]) and out["elevated"][4, 0, 0] == 0.0
    with pytest.raises(ValueError):
        acc.add(np.zeros((28,) + GRID, dtype="int8"))

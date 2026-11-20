from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from forecast import ExtendedForecast as ef


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_latest_cycle_waits_for_the_run_to_be_old_enough():
    assert ef.latest_cycle(utc(2026, 11, 19, 15, 59), min_age_hours=4) == utc(2026, 11, 18, 12)
    assert ef.latest_cycle(utc(2026, 11, 19, 16, 0), min_age_hours=4) == utc(2026, 11, 19, 12)
    # RRFS Day 3 needs a longer wait than HRRR Day 2.
    assert ef.latest_cycle(utc(2026, 11, 19, 17, 0), min_age_hours=7) == utc(2026, 11, 18, 12)


def test_day_windows_are_the_day_one_afternoon_one_and_two_days_later():
    day2, day3 = ef.DAYS[2], ef.DAYS[3]
    assert (day2.source, day2.start, day2.end) == ("hrrr", 28, 39)
    assert (day3.source, day3.start, day3.end) == ("rrfs", 52, 63)
    # Day 1 is f04-f15 (16z-03z); each later day is exactly 24 h further on.
    assert day2.start - 4 == 24 and day3.start - 4 == 48 and day2.end - day2.start == 11 == day3.end - day3.start


def test_window_label_is_local_time_in_both_standard_and_daylight_time():
    cfg = ef.DAYS[2]
    label, start = ef.window_label(utc(2026, 11, 19, 12), cfg)      # CST: 12z + 28 h = 16z next day = 10:00
    assert label == "10:00–21:00 CT" and start.strftime("%Y-%m-%d") == "2026-11-20"
    label, _ = ef.window_label(utc(2026, 7, 1, 12), cfg)            # CDT: 16z = 11:00
    assert label == "11:00–22:00 CT"


def test_source_leads_cover_warmup_plus_one_anchor_lead():
    leads = ef.source_leads(ef.DAYS[2])
    assert leads[0] == 28 - ef.WARMUP_HOURS - 1 and leads[-1] == 39
    assert ef.source_leads(ef.DAYS[3])[0] == 27


def test_precip_increments_difference_a_run_total_and_handle_resets():
    total = np.array([[[0.0]], [[0.0]], [[2.0]], [[2.0]], [[3.5]], [[0.4]], [[0.9]]])
    increments = ef.precip_increments(total)[:, 0, 0]
    # first lead anchors only; 2 mm falls in hour 2; the drop 3.5 -> 0.4 is a bucket reset (new total), then +0.5
    assert increments.tolist() == pytest.approx([0.0, 0.0, 2.0, 0.0, 1.5, 0.4, 0.5])
    # a tiny dip is packing noise, not a reset
    assert ef.precip_increments(np.array([[[5.0]], [[4.95]]]))[:, 0, 0].tolist() == pytest.approx([0.0, 0.0])


def test_danger_grid_only_evaluates_cells_the_rule_could_raise_above_low():
    calls = []

    def rule(fm, rh, wind):
        calls.append((fm, rh, wind))
        return 3

    fm = np.array([[5.0, 20.0], [np.nan, 5.0]])
    rh = np.full((2, 2), 20.0)
    wind = np.full((2, 2), 20.0)
    snow = np.array([[False, False], [False, True]])
    risk = ef.danger_grid(fm, rh, wind, snow, calculate=rule, low_fm=15.0)
    assert len(calls) == 1                       # wet cell, missing cell and snow cell never reach the rule
    assert risk[0, 0] == 3 and risk[0, 1] == 0 and np.isnan(risk[1, 0]) and risk[1, 1] == 0


def test_danger_grid_matches_the_canonical_rule_with_no_shortcut_errors():
    from core.fire_danger import calculate_fire_danger
    rng = np.random.default_rng(1)
    fm = rng.uniform(2, 30, (40, 40))
    rh = rng.uniform(8, 80, (40, 40))
    wind = rng.uniform(0, 35, (40, 40))
    snow = np.zeros_like(fm, dtype=bool)
    risk = ef.danger_grid(fm, rh, wind, snow, calculate=calculate_fire_danger, low_fm=15.0)
    expected = np.array([[calculate_fire_danger(a, b, c) for a, b, c in zip(*row)] for row in zip(fm, rh, wind)], dtype=float)
    assert np.array_equal(risk, expected)


def _dataset(spacing_m: int) -> xr.Dataset:
    x = np.arange(0, 10 * spacing_m, spacing_m, dtype="float64")
    y = np.arange(10 * spacing_m, 0, -spacing_m, dtype="float64")      # descending, as the acquired grids are
    ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y)})
    ds.attrs["crs"] = "EPSG:32615"
    return ds


def test_a_13km_source_is_interpolated_to_3km_and_a_3km_one_is_left_alone():
    coarse = _dataset(13000)
    lon = np.zeros((10, 10))
    field = np.arange(100, dtype="float64").reshape(10, 10)
    fields, new_lon, new_lat = ef.to_display_grid({"a": field, "none": None}, coarse, lon, lon)
    assert fields["none"] is None
    assert fields["a"].shape == new_lon.shape == new_lat.shape and fields["a"].shape[0] > 30
    assert np.nanmin(fields["a"]) >= field.min() - 1e-9 and np.nanmax(fields["a"]) <= field.max() + 1e-9
    fine = _dataset(3000)
    same, same_lon, _ = ef.to_display_grid({"a": field}, fine, lon, lon)
    assert same["a"] is field and same_lon is lon

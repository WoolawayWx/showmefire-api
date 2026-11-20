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


# --- run selection: every synoptic run that can reach the window ------------------------------------------------

def test_each_cycle_shifts_the_window_to_the_same_afternoon():
    # window = 16z of (run date + day - 1): the hours move with the run, the afternoon does not
    assert [(h, ef.config_for(2, h).start, ef.config_for(2, h).end) for h in (6, 12, 18)] == [(6, 34, 45), (12, 28, 39), (18, 22, 33)]
    assert ef.config_for(2, 0) is None                        # f40-f51 is past HRRR's f48
    assert ef.cycle_hours(2) == (6, 12, 18)
    # RRFS Day 3 is limited by SMF_EXTENDED_RRFS_MAX_LEAD (63): only runs whose f(end) is within it
    assert ef.cycle_hours(3) == (12, 18) and ef.config_for(3, 18).start == 46


def test_latest_cycle_picks_the_newest_supported_run_that_is_old_enough():
    hours = ef.cycle_hours(2)
    assert ef.latest_cycle(utc(2026, 11, 19, 22, 30), min_age_hours=4, hours=hours) == utc(2026, 11, 19, 18)
    assert ef.latest_cycle(utc(2026, 11, 19, 21, 59), min_age_hours=4, hours=hours) == utc(2026, 11, 19, 12)
    assert ef.latest_cycle(utc(2026, 11, 19, 3, 0), min_age_hours=4, hours=hours) == utc(2026, 11, 18, 18)   # 00z is not supported


def test_an_early_window_cuts_the_warmup_at_f00_and_is_flagged():
    assert ef.source_leads(ef.config_for(2, 18))[0] == 0      # 22 - 25 would be negative
    assert ef.source_leads(ef.config_for(2, 6))[0] == 34 - 25


def test_already_done_ignores_a_degraded_result(tmp_path):
    cycle = utc(2026, 11, 19, 12)
    path = tmp_path / "extended-forecast-day2.json"
    path.write_text('{"cycle_utc": "2026-11-19T12:00:00Z", "flags": []}')
    assert ef.already_done(tmp_path, 2, cycle) is True
    assert ef.already_done(tmp_path, 2, utc(2026, 11, 19, 18)) is False                   # a newer run exists
    path.write_text('{"cycle_utc": "2026-11-19T12:00:00Z", "flags": ["rrfs_blend_missing"]}')
    assert ef.already_done(tmp_path, 2, cycle) is False                                   # try again for the blend later
    assert ef.already_done(tmp_path / "nothing", 2, cycle) is False


# --- blend -------------------------------------------------------------------------------------------------------

def _model(spacing_m, size, fields, times=3):
    """A small model dataset on a UTM grid with the variables the blend and model read."""
    import pyproj
    x = np.arange(size) * spacing_m + 500000.0
    y = np.arange(size) * spacing_m + 4200000.0
    ds = xr.Dataset(coords={"x": ("x", x), "y": ("y", y), "time": np.arange(times).astype("timedelta64[h]") + np.datetime64("2026-11-19T12")})
    ds.attrs["crs"] = "EPSG:32615"
    lon, lat = pyproj.Transformer.from_crs("EPSG:32615", "EPSG:4326", always_xy=True).transform(*np.meshgrid(x, y))
    ds["longitude"], ds["latitude"] = (("y", "x"), lon), (("y", "x"), lat)
    for name, value in fields.items():
        ds[name] = (("time", "y", "x"), np.broadcast_to(np.asarray(value, dtype="float64").reshape(-1, 1, 1), (times, size, size)).copy())
    return ds


def test_blend_mixes_by_weight_and_blends_wind_as_speed():
    primary = _model(3000, 30, {"t2m": [280, 280, 280], "r2": [30, 30, 30], "u10": [10, 10, 10], "v10": [0, 0, 0], "tp": [0, 1, 2]})
    partner = _model(13000, 8, {"t2m": [290, 290, 290], "r2": [50, 50, 50], "u10": [0, 0, 0], "v10": [10, 10, 10], "tp": [0, 3, 4]})
    out = ef.blend_sources(primary, partner, 0.7)
    inside = (slice(None), slice(8, 20), slice(8, 20))              # well inside the coarser grid
    assert np.allclose(out.t2m.values[inside], 283.0) and np.allclose(out.r2.values[inside], 36.0)
    assert np.allclose(out.u10.values[inside], 10.0) and np.allclose(out.v10.values[inside], 0.0)   # speeds 10 and 10 -> 10, not 7.07
    assert np.allclose(out.tp.values[:, 15, 15], [0.0, 0.7 * 1 + 0.3 * 3, 0.7 * 2 + 0.3 * 4])
    assert np.allclose(primary.t2m.values, 280.0)                    # the input dataset is not modified


def test_blend_keeps_the_primary_where_the_partner_has_no_value():
    primary = _model(3000, 30, {"t2m": [280, 280, 280], "r2": [30, 30, 30], "u10": [5, 5, 5], "v10": [0, 0, 0], "tp": [0, 0, 0]})
    partner = _model(13000, 8, {"t2m": [290, 290, 290], "r2": [50, 50, 50], "u10": [5, 5, 5], "v10": [0, 0, 0], "tp": [0, 0, 0]})
    partner["t2m"].values[:, :, :] = np.nan
    out = ef.blend_sources(primary, partner, 0.7)
    assert np.allclose(out.t2m.values, 280.0)                       # nothing to mix: unchanged
    assert np.allclose(out.r2.values[:, 15, 15], 0.7 * 30 + 0.3 * 50)


# --- peak block --------------------------------------------------------------------------------------------------

def _hours(values):
    return [np.full((2, 2), v, dtype="float64") for v in values]


def test_the_peak_block_is_the_two_hours_with_the_highest_statewide_danger():
    inside = np.ones((2, 2), dtype=bool)
    risk = _hours([0, 1, 2, 3, 2, 1, 0, 0])                     # danger peaks at hour 3
    rh = _hours([40, 38, 35, 30, 28, 33, 40, 45])
    # (2, 3) and (3, 4) both total 5 danger points; the drier RH at hour 4 makes (3, 4) the winner
    assert ef.choose_peak_block(risk, rh, inside, 2) == 3
    assert ef.choose_peak_block(risk, rh, inside, 1) == 3


def test_an_all_low_day_still_gets_its_driest_stretch():
    inside = np.ones((2, 2), dtype=bool)
    risk = _hours([0] * 6)
    rh = _hours([55, 50, 42, 40, 47, 60])
    assert ef.choose_peak_block(risk, rh, inside, 2) == 2       # RH 42 and 40 are the driest pair


def test_only_cells_inside_missouri_decide_the_peak():
    inside = np.array([[True, False], [False, False]])
    outside_hot = [np.array([[0.0, 4.0], [4.0, 4.0]]), np.array([[1.0, 0.0], [0.0, 0.0]])]   # hour 0: hot outside, hour 1: some danger inside
    rh = _hours([30, 30])
    assert ef.choose_peak_block(outside_hot, rh, inside, 1) == 1


def test_peak_label_is_local_time_with_daylight_saving():
    assert ef.peak_label(utc(2026, 11, 19, 12), 32, 2) == "2:00 PM–3:00 PM CT"        # 12z + 32 h = 20z = 2 PM CST
    assert ef.peak_label(utc(2026, 7, 1, 12), 32, 2) == "3:00 PM–4:00 PM CT"          # 20z = 3 PM CDT
    assert ef.peak_label(utc(2026, 11, 19, 12), 28, 2) == "10:00 AM–11:00 AM CT"


# --- map layout --------------------------------------------------------------------------------------------------

def test_long_titles_break_after_the_day_number_and_short_ones_do_not():
    assert ef.map_title(3, "Minimum Relative Humidity Forecast") == ("Missouri Day 3\nMinimum Relative Humidity Forecast", True)
    assert ef.map_title(2, "Minimum Fuel Moisture Forecast")[1] is True                      # 45 characters: too close to the state
    assert ef.map_title(3, "Peak Fire Danger Forecast") == ("Missouri Day 3 Peak Fire Danger Forecast", False)
    assert ef.map_title(2, "Forecast Precipitation")[1] is False


def test_the_regridded_data_reaches_beyond_the_source_edge_so_it_covers_the_state_border():
    coarse = _dataset(13000)
    lon = np.zeros((10, 10))
    field = np.arange(100, dtype="float64").reshape(10, 10)
    fields, new_lon, _ = ef.to_display_grid({"a": field}, coarse, lon, lon)
    out = fields["a"]
    assert np.isfinite(out).all()                                         # no ragged NaN rim around the coarse grid
    extent_cells = (new_lon.shape[1] - 1) * ef.DISPLAY_SPACING_M / 13000
    assert extent_cells >= (10 - 1) + 2 * ef.EDGE_PAD_CELLS - 0.3         # wider than the source by the padding on each side
    assert np.isclose(out[0, 0], field[-1, 0]) or np.isclose(out[0, 0], field[0, 0])      # the corner repeats the edge cell (y is flipped)
    assert out.min() >= field.min() - 1e-9 and out.max() <= field.max() + 1e-9            # nothing is invented beyond the data

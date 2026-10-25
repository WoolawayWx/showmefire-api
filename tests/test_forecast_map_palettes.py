import ast
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from forecast import map_palettes as palettes
from forecast_v1.pipeline import DAILY_GRAPHIC_EXCLUDED, DAILY_RASTER_MAP

DAILY_FORECAST = Path(__file__).resolve().parents[1] / "forecast" / "DailyForecast.py"
SNOW = "snow_water_equivalent"


def _day1_list(name):
    """A list literal assigned to `name` in the Day 1 map code, so these tables cannot drift from it."""
    match = re.search(rf"^\s*{name}\s*=\s*(\[.*?\])", DAILY_FORECAST.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"{name} not found in DailyForecast.py"
    return ast.literal_eval(match.group(1))


def test_every_day_2_3_graphic_has_a_house_table_except_the_fire_danger_category():
    graphic_variables = {public for daily, public in DAILY_RASTER_MAP.items() if daily not in DAILY_GRAPHIC_EXCLUDED}
    graphic_variables.add(SNOW)
    missing = sorted(name for name in graphic_variables if name != "fire_danger" and palettes.style_for(name) is None)
    assert not missing
    assert palettes.style_for("something_else") is None


def test_every_style_builds_and_has_one_color_per_band():
    for name in ("relative_humidity_2m", "fuel_moisture_p50", "wind_speed_10m", "wind_gust_10m", "temperature_2m",
                 "precipitation_increment", SNOW, "probability_elevated_plus", "probability_critical_plus"):
        style = palettes.style_for(name)
        cmap, norm = style.build()
        # Day 1's snow-water palette has 26 colors for 25 bands (matplotlib spreads them); it is copied as is.
        if style.kind in ("discrete", "probability") and name != SNOW:
            assert len(style.colors) == len(style.levels) - 1, name
        assert cmap is not None
        if style.tick_labels is not None:
            assert len(style.tick_labels) == len(style.ticks), name


def test_tables_match_the_day_1_maps():
    rh = palettes.style_for("relative_humidity_2m")
    assert list(rh.colors) == _day1_list("rh_colors") and list(rh.levels) == _day1_list("rh_levels")
    wind = palettes.style_for("wind_speed_10m")
    assert list(wind.colors) == _day1_list("wind_colors") and list(wind.levels) == _day1_list("wind_levels")
    rain = palettes.style_for("precipitation_increment")
    assert list(rain.colors) == _day1_list("rain_colors") and list(rain.levels) == _day1_list("rain_levels")


def test_values_are_converted_to_the_units_the_day_1_maps_show():
    assert palettes.style_for("wind_speed_10m").convert(np.array([10.0]))[0] == pytest.approx(19.44, abs=0.01)
    assert palettes.style_for("temperature_2m").convert(np.array([0.0, 100.0])).tolist() == [32.0, 212.0]
    assert palettes.style_for("precipitation_increment").convert(np.array([25.4]))[0] == pytest.approx(1.0)
    assert palettes.style_for("relative_humidity_2m").convert(np.array([40.0]))[0] == 40.0


def test_gusts_use_the_sustained_wind_colors_with_room_for_stronger_values():
    wind, gust = palettes.style_for("wind_speed_10m"), palettes.style_for("wind_gust_10m")
    assert gust.colors == wind.colors and gust.levels[:-1] == wind.levels[:-1] and gust.levels[-1] > wind.levels[-1]


@pytest.mark.parametrize("variable", ["relative_humidity_2m", "fuel_moisture_p50", "probability_critical_plus", "mystery_variable"])
def test_static_graphic_renders_with_and_without_a_house_table(tmp_path, variable):
    from forecast_v1 import artifacts

    lon, _ = artifacts._grid_lonlat()
    field = np.broadcast_to(np.linspace(5, 95, lon.shape[1], dtype="float32"), lon.shape).copy()
    png, webp = artifacts.write_static_graphic(
        xr.DataArray(field, dims=("y", "x")), tmp_path / "a.png", tmp_path / "a.webp", title="Day 2 Test",
        valid_period="2026-10-26 America/Chicago", run_time="2026-10-24T12:00:00Z", status="ok", variable=variable,
        value_range=(0, 100))
    assert png.stat().st_size > 10_000 and webp.stat().st_size > 1_000

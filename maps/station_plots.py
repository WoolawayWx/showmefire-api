"""Station-model plots (no interpolated fill), rendered together in one process:

  * Current Conditions - temperature, dewpoint, sea-level pressure and wind barb
  * Current Fire Weather Conditions - temperature, relative humidity, wind speed
    and gust, wind barb and (where reported) fuel moisture
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps import station_products as spx

MIN_STATION_SPACING_DEG = 0.42  # plots hold 3-4 numbers each, so they need more room than a single label
TEMP_COLOR, DEWPOINT_COLOR, RH_COLOR, FUEL_COLOR = "#C62828", "#2E7D32", "#1D4ED8", "#8D5524"
STROKE = [sp.path_effects.withStroke(linewidth=2, foreground="white")]


def _value(station: dict, name: str, low: float, high: float):
    value = spx.obs_value(station, name)
    return value if isinstance(value, (int, float)) and low <= value <= high else None


def select_stations(stations: list[dict], required: tuple[str, ...]) -> list[dict]:
    """Stations with all `required` observations, richest reports first so the
    spacing filter keeps the most complete station in each neighborhood."""
    candidates = [
        s for s in stations
        if s.get("longitude") is not None and s.get("latitude") is not None
        and all(spx.obs_value(s, name) is not None for name in required)
    ]
    candidates.sort(key=lambda s: -sum(1 for v in (s.get("observations") or {}).values() if (v or {}).get("value") is not None))
    kept: list[dict] = []
    positions: list[np.ndarray] = []
    for station in candidates:
        position = np.array([station["longitude"], station["latitude"]])
        if all(np.linalg.norm(position - other) > MIN_STATION_SPACING_DEG for other in positions):
            kept.append(station)
            positions.append(position)
    return kept


def _annotate(ax, data_crs, station, text, offset, *, color, ha, va, bold=True, fontsize=9.5):
    x, y = ax.projection.transform_point(station["longitude"], station["latitude"], data_crs)
    ax.annotate(
        text, (x, y), xytext=offset, textcoords="offset points", fontsize=fontsize, color=color, ha=ha, va=va, zorder=11,
        fontweight="bold" if bold else "normal", path_effects=STROKE,
    )


def _barbs(ax, data_crs, stations: list[dict]) -> None:
    """All wind barbs in one call; calm stations get a ring instead."""
    lons, lats, us, vs, calm_lons, calm_lats = [], [], [], [], [], []
    for station in stations:
        speed, direction = _value(station, "wind_speed", 0, 120), _value(station, "wind_direction", 0, 360)
        if speed is None or direction is None:
            continue
        if speed < 1:
            calm_lons.append(station["longitude"])
            calm_lats.append(station["latitude"])
            continue
        lons.append(station["longitude"])
        lats.append(station["latitude"])
        us.append(-speed * np.sin(np.radians(direction)))
        vs.append(-speed * np.cos(np.radians(direction)))
    if lons:
        ax.barbs(np.array(lons), np.array(lats), np.array(us), np.array(vs), transform=data_crs, length=6.5, zorder=10, color="black", linewidth=1)
    if calm_lons:
        ax.scatter(calm_lons, calm_lats, transform=data_crs, s=80, facecolors="none", edgecolors="black", linewidths=1, zorder=10)


def _dot(ax, data_crs, station):
    ax.scatter(station["longitude"], station["latitude"], transform=data_crs, s=14, color="black", edgecolor="white", linewidth=0.8, zorder=10)


def current_conditions_overlay(fig, ax, data_crs, stations):
    count = 0
    chosen = select_stations(stations, ("air_temp", "wind_speed"))
    _barbs(ax, data_crs, chosen)
    for station in chosen:
        temp = _value(station, "air_temp", -60, 140)
        dewpoint = _value(station, "dew_point_temperature", -60, 100)
        pressure = _value(station, "sea_level_pressure", 940, 1070)
        if temp is None:
            continue
        _dot(ax, data_crs, station)
        _annotate(ax, data_crs, station, f"{temp:.0f}", (-7, 6), color=TEMP_COLOR, ha="right", va="bottom")
        if dewpoint is not None:
            _annotate(ax, data_crs, station, f"{dewpoint:.0f}", (-7, -6), color=DEWPOINT_COLOR, ha="right", va="top")
        if pressure is not None:
            # Meteorological shorthand: tenths of mb, last three digits (1012.9 -> 129).
            _annotate(ax, data_crs, station, f"{int(round(pressure * 10)) % 1000:03d}", (7, 6), color="#222222", ha="left", va="bottom", fontsize=8.5)
        count += 1
    return count


def fire_weather_overlay(fig, ax, data_crs, stations):
    count = 0
    chosen = select_stations(stations, ("air_temp", "relative_humidity", "wind_speed"))
    _barbs(ax, data_crs, chosen)
    for station in chosen:
        temp = _value(station, "air_temp", -60, 140)
        rh = _value(station, "relative_humidity", 0, 100)
        speed = _value(station, "wind_speed", 0, 120)
        gust = _value(station, "wind_gust", 0, 150)
        fuel = _value(station, "fuel_moisture", 0, 60)
        if None in (temp, rh):
            continue
        _dot(ax, data_crs, station)
        _annotate(ax, data_crs, station, f"{temp:.0f}", (-7, 6), color=TEMP_COLOR, ha="right", va="bottom")
        _annotate(ax, data_crs, station, f"{rh:.0f}%", (-7, -6), color=RH_COLOR, ha="right", va="top")
        wind_text = f"{speed:.0f}G{gust:.0f}" if gust is not None and gust >= speed + 3 else f"{speed:.0f}"
        _annotate(ax, data_crs, station, wind_text, (7, 6), color="#222222", ha="left", va="bottom", fontsize=8.5)
        if fuel is not None:
            _annotate(ax, data_crs, station, f"FM{fuel:.0f}", (7, -6), color=FUEL_COLOR, ha="left", va="top", fontsize=8.5)
        count += 1
    return count


def main() -> None:
    stations = sp.fetch_stations()
    spx.run_products([
        ("current conditions", lambda: spx.render_map(
            stations, product="station_plot_current", png_name="mo-stationplot-current.png",
            title="Current Conditions", formatter=None, overlay=current_conditions_overlay,
            description=(
                "Station plots. Around each station:\n"
                "temperature (F, red, upper left)\n"
                "dewpoint (F, green, lower left)\n"
                "sea-level pressure (upper right; tenths of mb,\n"
                "last 3 digits, so 129 = 1012.9 mb)\n"
                "and a wind barb (each full flag = 10 kt).\n"
                "A ring marks calm wind."
            ),
        )),
        ("fire weather conditions", lambda: spx.render_map(
            stations, product="station_plot_firewx", png_name="mo-stationplot-firewx.png",
            title="Current Fire Weather Conditions", formatter=None, overlay=fire_weather_overlay,
            description=(
                "Station plots. Around each station:\n"
                "temperature (F, red, upper left)\n"
                "relative humidity (%, blue, lower left)\n"
                "wind speed and gust (mph, upper right)\n"
                "10-hr fuel moisture (FM, %, lower right)\n"
                "and a wind barb (each full flag = 10 kt).\n"
                "Low RH, strong wind and dry fuels favor fire spread."
            ),
        )),
    ], label="Station plots")


if __name__ == "__main__":
    main()

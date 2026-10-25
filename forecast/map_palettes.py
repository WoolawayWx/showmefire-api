"""House color tables for the public Missouri forecast maps, keyed by forecast-v1 variable name.

The levels, colors, tick labels and legend text are copied from the Day 1 maps in forecast/DailyForecast.py (which still
holds its own copies; change both together) so that a color means the same thing on Day 1, Day 2 and Day 3. Probability
maps use the ensemble fire danger ramps. Data arrive in forecast-v1 units (degC, m/s, mm), so every style carries the
conversion to the units the Day 1 maps show (degF, knots, inches).

    style = style_for("wind_speed_10m")        # None for a variable with no house table
    cmap, norm = style.build()
    values = style.convert(array)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

MS_TO_KNOTS = 1.943844
MM_TO_INCHES = 1 / 25.4


@dataclass(frozen=True)
class FieldStyle:
    label: str                                   # colorbar label, in display units
    levels: tuple
    convert: Callable[[np.ndarray], np.ndarray] = lambda values: values
    colors: Optional[tuple] = None               # discrete bands (one color per pair of levels)
    kind: str = "discrete"                       # discrete | fuel_moisture | temperature | probability
    ticks: Optional[tuple] = None
    tick_labels: Optional[tuple] = None
    extend: str = "neither"
    alpha: float = 0.7
    line_levels: Optional[tuple] = None          # faint contour lines drawn over the fill, as on the Day 1 maps
    legend: str = ""                             # threshold text added above the standard description
    min_shown: Optional[float] = None            # values below this are left unfilled (probability maps)

    def build(self):
        """(colormap, norm) for this style; norm is None where the colormap's own range is the levels."""
        from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

        if self.kind == "fuel_moisture":
            cmap = LinearSegmentedColormap.from_list("fm_enhanced", list(FUEL_MOISTURE_STOPS), N=512)
            return cmap, BoundaryNorm(self.levels, cmap.N)
        if self.kind == "temperature":
            from core.temperature_palette import load_wsi_temperature_palette

            return load_wsi_temperature_palette()[0], None
        cmap = ListedColormap(self.colors)
        return cmap, BoundaryNorm(self.levels, len(self.colors))


FUEL_MOISTURE_STOPS = (
    (0.0, "#4D0000"), (0.1, "#8B0000"), (0.2, "#DC143C"), (0.233, "#FF4500"), (0.267, "#FF6347"), (0.30, "#FF8C00"),
    (0.333, "#FFA500"), (0.367, "#FFB347"), (0.40, "#FFD700"), (0.50, "#FFED4E"), (0.60, "#F0E68C"), (0.70, "#C8E6C9"),
    (0.80, "#81C784"), (0.90, "#4CAF50"), (1.0, "#2E7D32"),
)

_RH_LEVELS = (0, 15, 25, 35, 50, 65, 100)
_WIND_LEVELS = (0, 10, 15, 20, 25, 30)
_WIND_COLORS = ("#90EE90", "#FFED4E", "#FFA500", "#FF6347", "#FF0000", "#8B0000")
_RAIN_LEVELS = (0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
_SWE_LEVELS = tuple(round(i * 0.1, 1) for i in range(26))


def _swe_colors() -> tuple:
    colors = ["#ffffff"]
    for i in range(1, 26):
        shade = 255 - int((i / 25) * 200)
        colors.append(f"#{shade:02x}{shade:02x}ff")
    return tuple(colors)


def _wind_style(label: str, top: float, legend: str) -> FieldStyle:
    levels = _WIND_LEVELS + (top,)
    return FieldStyle(
        label, levels, lambda ms: ms * MS_TO_KNOTS, _WIND_COLORS, ticks=(5, 12.5, 17.5, 22.5, 27.5, (30 + top) / 2),
        tick_labels=("<10 kt\n(<12 mph)", "10-15 kt\n(12-17 mph)", "15-20 kt\n(17-23 mph)", "20-25 kt\n(23-29 mph)",
                     "25-30 kt\n(29-35 mph)", ">30 kt\n(>35 mph)"),
        line_levels=levels[1:-1], legend=legend)


_WIND_LEGEND = (
    "Critical Thresholds:\n< 10 kts: Light winds - Normal fire behavior\n10-15 kts: Moderate - Increased fire activity\n"
    "15-20 kts: Strong - Red Flag criteria with low RH\n20-25 kts: Very Strong - Critical fire weather\n"
    "> 25 kts: Extreme - Dangerous fire conditions"
)


def _probability_style(category: int, label: str, legend: str) -> FieldStyle:
    from services.ensemble_fire_danger.render import PROB_FILL_LEVELS, PROB_RAMPS

    return FieldStyle(
        label, tuple(PROB_FILL_LEVELS), colors=tuple(PROB_RAMPS[category]), kind="probability",
        ticks=tuple(PROB_FILL_LEVELS), tick_labels=tuple(f"{value}%" for value in PROB_FILL_LEVELS), alpha=1.0,
        line_levels=tuple(PROB_FILL_LEVELS[:-1]), legend=legend, min_shown=PROB_FILL_LEVELS[0])


def style_for(variable: str) -> Optional[FieldStyle]:
    """The house style for a forecast-v1 variable, or None when the Day 1 maps have no equivalent."""
    if variable == "relative_humidity_2m":
        return FieldStyle(
            "Relative Humidity (%)", _RH_LEVELS, colors=("#8B0000", "#FF0000", "#FFA500", "#FFED4E", "#90EE90", "#228B22"),
            ticks=(7.5, 20, 30, 42.5, 57.5, 82.5), tick_labels=("<15%", "15-25%", "25-35%", "35-50%", "50-65%", ">65%"),
            line_levels=_RH_LEVELS[1:-1],
            legend="Critical Thresholds:\n< 15%: Extremely Dry - Critical fire conditions\n"
                   "15-25%: Very Dry - Red Flag criteria with wind\n25-35%: Dry - Elevated fire danger possible\n"
                   "35-50%: Moderate - Normal fire activity\n> 50%: Moist - Fire spread limited")
    if variable == "fuel_moisture_p50":
        bounds = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 21, 24, 27, 30)
        return FieldStyle(
            "Fuel Moisture (%)", bounds, kind="fuel_moisture", ticks=(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30),
            extend="both", alpha=0.75,
            legend="Critical Thresholds:\n< 7%: Extremely dry - potential for extreme fire behavior\n"
                   "7-9%: Very dry - critical fire behavior likely\n9-15%: Dry - elevated fire behavior expected\n"
                   "15-20%: Moderate - some fire activity possible\n> 20%: Moist - fuels much less receptive to fire")
    if variable == "wind_speed_10m":
        return _wind_style("Wind Speed (knots)", 50, _WIND_LEGEND)
    if variable == "wind_gust_10m":
        # Day 1 has no gust map; gusts use the sustained-wind colors so a color means the same speed everywhere.
        return _wind_style("Wind Gust (knots)", 60, _WIND_LEGEND.replace("Light winds", "Light gusts"))
    if variable == "temperature_2m":
        from core.temperature_palette import load_wsi_temperature_palette

        levels = tuple(load_wsi_temperature_palette()[1])
        return FieldStyle(
            "Temperature (°F)", levels, lambda celsius: celsius * 9 / 5 + 32, kind="temperature", ticks=levels,
            extend="both", line_levels=tuple(range(10, 90, 10)),
            legend="Temperature influences fire behavior:\nHigher temperatures increase fuel dryness\nand fire spread rates.")
    if variable == "precipitation_increment":
        return FieldStyle(
            "Total Precipitation (inches)", _RAIN_LEVELS, lambda mm: mm * MM_TO_INCHES,
            ("#FFFFFF", "#E0F0FF", "#B0D4FF", "#6AB4FF", "#0080FF", "#0050D0", "#003080"), ticks=_RAIN_LEVELS,
            tick_labels=tuple(f'{x:.2f}" / {x * 25.4:.1f} mm' if x < 1 else f'{x:.1f}" / {x * 25.4:.0f} mm' for x in _RAIN_LEVELS),
            alpha=0.8, line_levels=_RAIN_LEVELS[1:-1],
            legend='Precipitation reduces fire danger by:\n- Increasing fuel moisture\n- Raising relative humidity\n'
                   '- Potentially preventing ignition\n\n> 0.5": Significant fire danger reduction\n> 1.0": Major impact on fire activity')
    if variable == "snow_water_equivalent":
        return FieldStyle(
            "Snow Water Equivalent (inches)", _SWE_LEVELS, lambda mm: mm * MM_TO_INCHES, _swe_colors(),
            ticks=tuple(round(i * 0.5, 1) for i in range(6)), alpha=0.8,
            legend="Snow water equivalent is the water held in the snowpack,\nwhich suppresses fire danger.")
    if variable == "probability_elevated_plus":
        return _probability_style(2, "Chance of Elevated or Worse (%)", "Chance of Elevated (High) fire danger or higher.")
    if variable == "probability_critical_plus":
        return _probability_style(3, "Chance of Critical or Worse (%)", "Chance of Critical (Very High) fire danger or higher.")
    return None

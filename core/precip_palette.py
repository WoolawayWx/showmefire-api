"""Shared precipitation color table (AccuWeather QPF, inches) used by every forecast precipitation map.

The table is assets/AccuWeather_QPF_in.pal: one line per step, `inches R G B`, where the color applies from that value up
to the next line. Dry cells (below the first step) stay white, as on the earlier maps, and the last line's color is used
for everything at or above it (draw with extend="max"). Colors are discrete bands, not a gradient, because the table's
neighbouring colors jump (grey to light green at 0.1", yellow to orange at 1.25").

    qpf = load_qpf_palette()
    cs = ax.contourf(lon, lat, inches, levels=qpf.levels, cmap=qpf.cmap(), norm=qpf.norm(), extend="max")
    cbar.set_ticks(qpf.ticks); cbar.set_ticklabels(qpf.tick_labels)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PALETTE_PATH = Path(__file__).resolve().parents[1] / "assets" / "AccuWeather_QPF_in.pal"

# Colorbar ticks and faint contour lines: a handful of round values (the table itself has 41 steps, too many to label).
# The legend is uniform in steps, not inches, so these sit about 4-6 steps apart: room for a two-line label.
TICK_VALUES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0)
LINE_VALUES = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
MM_PER_INCH = 25.4


def tick_label(inches: float) -> str:
    """'0.25"' over '6.3 mm', on two lines so the legend stays narrow and clear of the state."""
    if inches < 1:
        return f'{inches:.2f}"\n{inches * MM_PER_INCH:.1f} mm'
    return f'{inches:.1f}"\n{inches * MM_PER_INCH:.0f} mm'


@dataclass(frozen=True)
class PrecipPalette:
    levels: tuple          # band edges in inches: 0, then every table value
    colors: tuple          # one hex color per band between levels (white for 0 up to the first step)
    over: str              # color of the last table line, for values at or above the top level
    ticks: tuple
    tick_labels: tuple
    line_levels: tuple

    def cmap(self):
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(self.colors)
        cmap.set_over(self.over)
        return cmap

    def norm(self):
        from matplotlib.colors import BoundaryNorm
        return BoundaryNorm(self.levels, len(self.colors))


def _hex(red: float, green: float, blue: float) -> str:
    return "#{:02X}{:02X}{:02X}".format(int(red), int(green), int(blue))


@lru_cache(maxsize=1)
def load_qpf_palette() -> PrecipPalette:
    steps = []
    with PALETTE_PATH.open("r", encoding="utf-8") as palette_file:
        for line_number, line in enumerate(palette_file, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Invalid precipitation palette line {line_number}: {line!r}")
            value, red, green, blue = (float(field) for field in fields)
            if not all(0 <= channel <= 255 for channel in (red, green, blue)):
                raise ValueError(f"Colour out of range on precipitation palette line {line_number}: {line!r}")
            steps.append((value, _hex(red, green, blue)))
    if len(steps) < 2:
        raise ValueError(f"Precipitation palette needs at least two steps: {PALETTE_PATH}")
    values = [value for value, _ in steps]
    if values != sorted(values) or len(set(values)) != len(values) or values[0] <= 0:
        raise ValueError(f"Precipitation palette steps must be positive and strictly increasing: {PALETTE_PATH}")
    # Band edges are 0 and every step; the band starting at step i is coloured by step i, except the last step, which
    # has no upper edge and becomes the "over" colour.
    levels = (0.0, *values)
    colors = ("#FFFFFF", *(color for _, color in steps[:-1]))
    return PrecipPalette(levels=levels, colors=colors, over=steps[-1][1],
                         ticks=TICK_VALUES, tick_labels=tuple(tick_label(v) for v in TICK_VALUES), line_levels=LINE_VALUES)

"""Shared temperature color table used by forecast map generation."""

from functools import lru_cache
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap


PALETTE_PATH = Path(__file__).resolve().parents[1] / "assets" / "WSI_Temperature.pal"


@lru_cache(maxsize=1)
def load_wsi_temperature_palette():
    """Return the WSI temperature colormap and its source range in °F.

    The palette uses repeated temperature entries to describe hard color
    transitions. Those repeated entries are intentionally retained when
    constructing the colormap.
    """
    points = []
    with PALETTE_PATH.open("r", encoding="utf-8") as palette_file:
        for line_number, line in enumerate(palette_file, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Invalid temperature palette line {line_number}: {line!r}")
            value, red, green, blue = (float(field) for field in fields)
            points.append((value, (red / 255, green / 255, blue / 255)))

    if len(points) < 2:
        raise ValueError(f"Temperature palette must contain at least two entries: {PALETTE_PATH}")

    minimum = points[0][0]
    maximum = points[-1][0]
    if maximum <= minimum:
        raise ValueError(f"Temperature palette range is invalid: {minimum} to {maximum}")

    normalized_points = [
        ((value - minimum) / (maximum - minimum), color)
        for value, color in points
    ]
    colormap = LinearSegmentedColormap.from_list(
        "wsi_temperature",
        normalized_points,
        N=256,
    )
    levels = tuple(sorted({value for value, _ in points}))
    return colormap, levels

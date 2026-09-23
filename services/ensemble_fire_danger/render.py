"""
The five ensemble fire danger graphics, in the public forecast house style.

Canvas, projection, extent, boundaries, title block, fonts, logo and
colorbar placement all come from forecast/map_style.py - the same helpers
forecast/DailyForecast.py draws the public "Missouri Peak Fire Danger
Forecast" with - so the framing cannot drift from the public maps.

- mo-forecast-ens-firedanger.png: the categorical ensemble forecast, with
  the production map's exact bins / colors / dual labels / alpha.
- mo-forecast-ens-prob-{moderate,elevated,critical,extreme}.png: chance
  of that category OR HIGHER within ~20 km (neighborhood probability).

Probability palettes: one hue per category (the category's own public
color family), 9 filled bands 10-20 ... 90-100%, stepped in OKLCH and
validated with the dataviz skill's validate_palette.js --ordinal against
the #E8E8E8 map background (all four PASS: monotone lightness, adjacent
dL >= 0.06, light end >= 2:1 contrast vs background, single hue). A 10th
filled band for 5-10% cannot pass - there is not enough lightness range
between a light end that still reads against the grey background and
black - so the 5% level is drawn as a dashed outline instead (the usual
"marginal signal" convention on SPC-style probability graphics). Fills
are opaque, unlike the categorical map's 0.7 alpha, so the validated
colors are the colors on screen; every level is also labelled on its
contour line, so the value never depends on color alone.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

API_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = API_ROOT / "forecast"  # map_style resolves fonts/logo relative to this

PIXEL_W, PIXEL_H, MAP_DPI = 2048, 1152, 144
EXTENT = (-95.8, -89.1, 35.8, 40.8)

CATEGORICAL_COLORS = ["#90EE90", "#FFED4E", "#FFA500", "#FF0000", "#8B0000"]
CATEGORICAL_LABELS = ["Low", "Moderate", "Elevated \nHigh", "Critical \n Very High", "Extreme"]
CATEGORICAL_BINS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]

PROB_FILL_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PROB_OUTLINE_LEVEL = 5
PROB_RAMPS = {
    1: ["#c2a301", "#ac9008", "#977e03", "#826c00", "#6d5b03", "#594a01", "#463a03", "#342a01", "#231b00"],
    2: ["#f08b00", "#d57b07", "#bb6b04", "#a15c02", "#884d01", "#703e00", "#593002", "#432301", "#2e1600"],
    3: ["#ff7969", "#fe4b3c", "#f01712", "#d20103", "#b20203", "#930202", "#750202", "#590201", "#3f0000"],
    4: ["#fe6dae", "#ec529b", "#d63d88", "#c02475", "#a90263", "#8b0251", "#6f0240", "#55002f", "#3b001f"],
}
CATEGORY_TITLES = {
    1: "Moderate",
    2: "Elevated (High)",
    3: "Critical (Very High)",
    4: "Extreme",
}
FILE_KEYS = {1: "moderate", 2: "elevated", 3: "critical", 4: "extreme"}

CRITERIA_TEXT = (
    "Fire Danger Criteria:\n"
    "Moderate:  FM < 15% AND (RH < 45% OR Wind ≥ 10 kts)\n"
    "Elevated:  FM < 9% WITH (RH < 35% and Wind >= 12) or (RH < 25% and Wind >= 5)\n"
    "Critical:  FM < 9% WITH (RH < 25% AND Wind >= 15 kts)\n"
    "Extreme:  FM < 7% WITH (RH < 20% AND Wind >= 25 kts)\n"
)


def _missouri_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    import geopandas as gpd
    import shapely

    border = gpd.read_file(API_ROOT / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp").to_crs("EPSG:4326")
    geom = border.geometry.union_all().buffer(0.01) if hasattr(border.geometry, "union_all") \
        else border.geometry.unary_union.buffer(0.01)
    return shapely.contains_xy(geom, lon, lat)


class Renderer:
    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.lat, self.lon = lat, lon
        self.mask = _missouri_mask(lat, lon)

    def _base(self):
        import cartopy.crs as ccrs
        import matplotlib
        matplotlib.use("Agg")
        from forecast.map_style import create_base_map

        data_crs = ccrs.PlateCarree()
        map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
        fig, ax = create_base_map(EXTENT, map_crs, data_crs, PIXEL_W, PIXEL_H, MAP_DPI)
        return fig, ax, data_crs

    def _finish(self, fig, ax, data_crs, title: str, subtitle: str, description: str, out_path: Path,
                run_date, county_zorder: float = 5) -> Path:
        import matplotlib.pyplot as plt
        from forecast.map_style import add_boundaries, add_title_and_branding

        add_boundaries(ax, data_crs, API_ROOT, county_zorder=county_zorder)
        ax.set_anchor("W")
        plt.subplots_adjust(left=0.05)
        add_title_and_branding(fig, title, subtitle, description, run_date, SCRIPT_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{out_path.stem}.", suffix=".png", dir=out_path.parent)
        os.close(fd)
        try:
            fig.savefig(tmp, dpi=MAP_DPI, bbox_inches=None, pad_inches=0)
            os.replace(tmp, out_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
            plt.close(fig)
        _write_webp(out_path)
        return out_path

    def categorical(self, grid: np.ndarray, out_path: Path, *, subtitle: str, description: str, run_date) -> Path:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        fig, ax, data_crs = self._base()
        field = np.where(self.mask, grid, np.nan)
        cmap = ListedColormap(CATEGORICAL_COLORS)
        norm = BoundaryNorm(CATEGORICAL_BINS, len(CATEGORICAL_COLORS))
        cs = ax.contourf(self.lon, self.lat, field, transform=data_crs, levels=CATEGORICAL_BINS, cmap=cmap,
                         norm=norm, alpha=0.7, zorder=7, antialiased=True)
        ax.contour(self.lon, self.lat, field, transform=data_crs, levels=CATEGORICAL_BINS[1:-1], colors="black",
                   linewidths=0.3, alpha=0.2, zorder=8)
        cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
        cbar = plt.colorbar(cs, cax=cax, label="Fire Danger Level")
        cbar.set_ticks([0, 1, 2, 3, 4])
        cbar.set_ticklabels(CATEGORICAL_LABELS)
        return self._finish(fig, ax, data_crs, "Missouri Ensemble Fire Danger Forecast (Beta)", subtitle,
                            description, out_path, run_date)

    def probability(self, k: int, prob_fraction: np.ndarray, out_path: Path, *, subtitle: str, description: str,
                    run_date) -> Path:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap

        fig, ax, data_crs = self._base()
        pct = np.where(self.mask, np.clip(prob_fraction * 100.0, 0.0, 100.0), np.nan)
        # 100% is a legitimate value; nudge so it falls inside the last band.
        pct = np.where(pct >= 100.0, 99.999, pct)
        colors = PROB_RAMPS[k]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(PROB_FILL_LEVELS, len(colors))
        filled = np.where(pct >= PROB_FILL_LEVELS[0], pct, np.nan)
        has_fill = bool(np.isfinite(filled).any())
        if has_fill:
            cs = ax.contourf(self.lon, self.lat, pct, transform=data_crs, levels=PROB_FILL_LEVELS, cmap=cmap,
                             norm=norm, zorder=7, antialiased=True)
            lines = ax.contour(self.lon, self.lat, pct, transform=data_crs, levels=PROB_FILL_LEVELS[:-1],
                               colors="black", linewidths=0.5, alpha=0.45, zorder=8)
            _halo(ax.clabel(lines, fmt="%g%%", fontsize=8, inline=True))
        if np.nanmax(np.nan_to_num(pct, nan=0.0)) >= PROB_OUTLINE_LEVEL:
            outline = ax.contour(self.lon, self.lat, pct, transform=data_crs, levels=[PROB_OUTLINE_LEVEL],
                                 colors="#333333", linewidths=1.3, linestyles="dashed", zorder=8)
            _halo(ax.clabel(outline, fmt="%g%%", fontsize=8, inline=True))

        cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
        from matplotlib.cm import ScalarMappable
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable, cax=cax, label="Probability (%)")
        cbar.set_ticks(PROB_FILL_LEVELS)
        cbar.set_ticklabels([f"{v}%" for v in PROB_FILL_LEVELS])
        if not has_fill and np.nanmax(np.nan_to_num(pct, nan=0.0)) < PROB_OUTLINE_LEVEL:
            fig.text(0.42, 0.5, f"Less than {PROB_OUTLINE_LEVEL}% chance statewide", ha="center", va="center",
                     fontsize=18, color="#555555", fontname="Montserrat",
                     bbox=dict(boxstyle="round,pad=0.6", facecolor="#E6E6E6", edgecolor="#9E9E9E"))
        title = f"Chance of {CATEGORY_TITLES[k]} or Greater Fire Danger (Beta)"
        # Opaque fills (unlike the categorical map's alpha 0.7) would hide the
        # county lines, so they go above the fill here.
        return self._finish(fig, ax, data_crs, title, subtitle, description, out_path, run_date,
                            county_zorder=7.5)


def _halo(labels) -> None:
    """White outline on contour labels so they stay legible on the darkest bands."""
    import matplotlib.patheffects as path_effects

    for label in labels or []:
        label.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])


def _write_webp(png_path: Path) -> None:
    try:
        from PIL import Image
        with Image.open(png_path) as image:
            image.convert("RGB").save(png_path.with_suffix(".webp"), "WEBP", quality=86)
    except Exception as error:  # WebP is a convenience copy - never fail the PNG over it
        logger.warning("WebP copy of %s failed: %s", png_path.name, error)


def descriptions(*, window_label: str, members_text: str, calibrated_text: str, track_label: str) -> Dict[str, str]:
    footer = (f"Data Source: {track_label}\n"
              "BETA – experimental guidance, not an official forecast\n"
              "For More Info, Visit ShowMeFire.org")
    categorical = (
        f"Ensemble Peak Fire Danger ({window_label})\n\n"
        f"{members_text}\n"
        "Categorical = calibrated ensemble consensus: the highest\n"
        "category the ensemble reaches with enough agreement\n"
        f"({calibrated_text}).\n\n"
        f"{CRITERIA_TEXT}\n"
        f"{footer}"
    )
    probability = (
        f"Ensemble Peak Fire Danger ({window_label})\n\n"
        "Chance this category OR HIGHER occurs within\n"
        "~20 km (12 mi) of a point during the peak window.\n"
        "Dashed line = 5% (marginal signal).\n\n"
        f"{members_text}\n"
        f"Probabilities: {calibrated_text}\n\n"
        f"{CRITERIA_TEXT}\n"
        f"{footer}"
    )
    return {"categorical": categorical, "probability": probability}

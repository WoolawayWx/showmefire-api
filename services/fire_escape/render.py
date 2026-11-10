"""The fire escape maps, in the house style of every other Missouri graphic (forecast/map_style.py).

Five maps and one overview sheet:
  pfdi          Physical Fire Danger Index level (grid)
  fire_10ac     if a fire starts here: chance it reaches 10 acres (grid)
  fire_100ac    if a fire starts here: chance it reaches 100 acres (grid)
  county_10ac   chance a fire of 10 acres or more is reported in the county today (county)
  county_100ac  the same for 100 acres or more (county)

The probability scales are fixed, not stretched to the day, so the same colour means the same odds on every
map of a kind; a quiet day looks quiet.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from core.fire_danger import CATEGORY_HEX_COLORS, CATEGORY_LABELS

API_ROOT = Path(__file__).resolve().parent.parent.parent
NOT_FOR_OPERATIONS = "EXPERIMENTAL - NOT FOR OPERATIONS"
EXTENT = (-95.8, -89.1, 35.8, 40.8)
LEVEL_BINS = (-0.5, 0.5, 1.5, 2.5, 3.5, 4.5)
# Percent. Chosen from the spread the models produce: the fire model's answers run from a few percent to over half;
# the county model's are small, because a large fire in one county on one day is rare.
SCALES = {
    "fire_10ac": (2, 5, 10, 15, 20, 30, 40, 50, 65, 100),
    "fire_100ac": (0.5, 1, 2, 3, 4, 6, 8, 12, 20, 100),
    "county_10ac": (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 100),
    "county_100ac": (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 100),
}
PROBABILITY_COLORS = ("#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026")
BELOW_SCALE_COLOR = "#f4f4ec"
MAPS = {
    "pfdi": {
        "title": "Physical Fire Danger Index (Experimental)",
        "text": ("Flame length a fire would have in open cured grass at\nthis afternoon's peak wind, from physics alone:\n"
                 "Low under 1 ft, Moderate 1-4 ft, Elevated 4-8 ft,\nCritical 8-11 ft, Extreme over 11 ft.\n\n"
                 "It reads higher than the public fire danger on ordinary\ndry days. It does not know real fuels, snow or green-up\n"
                 "beyond the date, and it is not trained on fire records.")},
    "fire_10ac": {
        "title": "If a Fire Starts Here: Chance It Reaches 10 Acres",
        "text": ("Trained on 23,000 Missouri wildfires (1992-2020).\nGiven that a fire is reported at this spot today, the\n"
                 "chance its final size is 10 acres or more.\n\nIt says nothing about whether a fire will start.\n"
                 "It ranks fires fairly, not sharply (about 0.7 on a\n0.5-1 scale), and leans on the fuels around the spot.")},
    "fire_100ac": {
        "title": "If a Fire Starts Here: Chance It Reaches 100 Acres",
        "text": ("Trained on 23,000 Missouri wildfires (1992-2020).\nGiven that a fire is reported at this spot today, the\n"
                 "chance its final size is 100 acres or more.\n\nIt says nothing about whether a fire will start.\n"
                 "It ranks fires fairly, not sharply (about 0.7 on a\n0.5-1 scale), and leans on the fuels around the spot.")},
    "county_10ac": {
        "title": "Chance of a 10+ Acre Fire in the County Today",
        "text": ("Trained on every Missouri county-day, 1992-2020.\nThe chance that a wildfire of 10 acres or more is\n"
                 "reported in the county today: that one starts AND\ngrows. Weather only.\n\n"
                 "Large fires are rare, so the numbers are small;\nread it as which counties and days stand out.\n"
                 "Scored about 0.9 (0.5-1 scale) on years and fire-\ndepartment records it never saw.")},
    "county_100ac": {
        "title": "Chance of a 100+ Acre Fire in the County Today",
        "text": ("Trained on every Missouri county-day, 1992-2020.\nThe chance that a wildfire of 100 acres or more is\n"
                 "reported in the county today: that one starts AND\ngrows. Weather only.\n\n"
                 "Large fires are rare, so the numbers are small;\nread it as which counties and days stand out.\n"
                 "Scored about 0.9 (0.5-1 scale) on years and fire-\ndepartment records it never saw.")},
}
FOOTER = "\n\nData: HRRR 12z forecast, gridMET | ShowMeFire\nExperimental output for comparison only."


def odds(probability: float) -> str:
    """'1 in 40' for 0.025; 'under 1 in 10,000' below that."""
    if not np.isfinite(probability) or probability <= 0:
        return "n/a"
    if probability < 1e-4:
        return "under 1 in 10,000"
    return f"1 in {max(1, int(round(1.0 / probability))):,}"


def percent(probability: float) -> str:
    """'15%', '2.5%', '0.28%': whole numbers from 10 % up, two significant figures below."""
    value = 100.0 * probability
    return f"{value:.0f}%" if value >= 9.95 else f"{value:.2g}%"


def _percent_label(value: float) -> str:
    return f"{value:g}%"


def _scale(kind: str):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    levels = SCALES[kind]
    cmap = ListedColormap(PROBABILITY_COLORS)
    cmap.set_under(BELOW_SCALE_COLOR)
    return levels, cmap, BoundaryNorm(levels, cmap.N)


def _clip_to_state(ax, data_crs, artists) -> None:
    """Vector clip to Missouri's boundary (a NaN mask leaves a notched edge at the grid spacing)."""
    from cartopy.mpl.path import shapely_to_path
    from matplotlib.patches import PathPatch
    import geopandas as gpd

    state = gpd.read_file(API_ROOT / "maps" / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp").to_crs("EPSG:4326")
    clip = PathPatch(shapely_to_path(ax.projection.project_geometry(state.geometry.union_all(), data_crs)), transform=ax.transData)
    for artist in artists:
        for collection in (getattr(artist, "collections", None) or [artist]):
            collection.set_clip_path(clip)


def _county_shapes():
    import geopandas as gpd

    counties = gpd.read_file(API_ROOT / "maps" / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp").to_crs("EPSG:4326")
    counties["fips"] = counties["COUNTYFIPS"].astype(str).str.zfill(3).radd("29")
    return counties


def _draw(ax, data_crs, kind: str, result: dict, fips: list[str]):
    """Fill one map's data on `ax`. Returns (mappable for the colorbar, tick positions, tick labels)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    lat, lon = result["lat"], result["lon"]
    if kind == "pfdi":
        cmap, norm = ListedColormap(CATEGORY_HEX_COLORS), BoundaryNorm(LEVEL_BINS, len(CATEGORY_HEX_COLORS))
        level = np.ma.masked_invalid(result["grid"]["pfdi_level"])
        fill = ax.contourf(lon, lat, level, transform=data_crs, levels=LEVEL_BINS, cmap=cmap, norm=norm, alpha=0.75, zorder=7)
        _clip_to_state(ax, data_crs, [fill])
        return fill, [0, 1, 2, 3, 4], list(CATEGORY_LABELS)
    levels, cmap, norm = _scale(kind)
    if kind.startswith("fire_"):
        percent = np.ma.masked_invalid(100.0 * result["grid"]["fire_p10" if kind == "fire_10ac" else "fire_p100"])
        fill = ax.contourf(lon, lat, percent, transform=data_crs, levels=levels, cmap=cmap, norm=norm, extend="min", alpha=0.8, zorder=7)
        _clip_to_state(ax, data_crs, [fill])
    else:
        values = dict(zip(fips, 100.0 * np.asarray(result["county"]["p_10ac" if kind == "county_10ac" else "p_100ac"])))
        for row in _county_shapes().itertuples():
            percent = values.get(row.fips, np.nan)
            color = "#d9d9d9" if not np.isfinite(percent) else cmap(norm(percent)) if percent >= levels[0] else BELOW_SCALE_COLOR
            ax.add_geometries([row.geometry], crs=data_crs, facecolor=color, edgecolor="none", alpha=0.85, zorder=7)
        fill = ScalarMappable(norm=norm, cmap=cmap)
    return fill, list(levels[:-1]), [_percent_label(v) for v in levels[:-1]]


def _save(fig, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".fire_escape.", suffix=".png", dir=out_path.parent)
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=dpi, bbox_inches=None, pad_inches=0)
        Path(temporary).replace(out_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def summary_line(kind: str, result: dict) -> str:
    """The day in one line: the highest value on the map, as odds where it is a probability."""
    if kind == "pfdi":
        flame = result["grid"]["pfdi_flame_ft"]
        return f"Highest flame length: {np.nanmax(flame):.1f} ft | statewide median {np.nanmedian(flame):.1f} ft"
    values = (result["grid"]["fire_p10" if kind == "fire_10ac" else "fire_p100"] if kind.startswith("fire_")
              else np.asarray(result["county"]["p_10ac" if kind == "county_10ac" else "p_100ac"]))
    return (f"Highest: {percent(float(np.nanmax(values)))} ({odds(float(np.nanmax(values)))}) | "
            f"median {percent(float(np.nanmedian(values)))} ({odds(float(np.nanmedian(values)))})")


def render_map(kind: str, result: dict, fips: list[str], day: date, out_path: Path, gridmet_lag_days: int = 1) -> Path:
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from forecast.map_style import add_boundaries, add_title_and_branding, create_base_map

    pixel_width, pixel_height, dpi = 2048, 1152, 144
    data_crs = ccrs.PlateCarree()
    fig, ax = create_base_map(EXTENT, ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3), data_crs,
                              pixel_width, pixel_height, dpi)
    try:
        mappable, ticks, labels = _draw(ax, data_crs, kind, result, fips)
        cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
        if kind == "pfdi":
            bar = plt.colorbar(mappable, cax=cax, label="Danger level")
        else:
            bar = plt.colorbar(mappable, cax=cax, label="Chance", extend="min", spacing="uniform")
        bar.set_ticks(ticks)
        bar.set_ticklabels(labels)
        add_boundaries(ax, data_crs, API_ROOT, county_zorder=8)     # above the fills, so county lines show on the county maps
        ax.set_anchor("W")
        plt.subplots_adjust(left=0.05)
        fig.text(0.99, 0.835, NOT_FOR_OPERATIONS, fontsize=18, fontweight="bold", ha="right", va="top", color="#B00000")
        stale = "" if gridmet_lag_days <= 1 else f" | fuel moisture {gridmet_lag_days} days old"
        add_title_and_branding(fig, MAPS[kind]["title"], f"Valid: {day:%a %b %d, %Y}{stale} | {summary_line(kind, result)}",
                               MAPS[kind]["text"] + FOOTER, None, API_ROOT / "forecast")
        _save(fig, out_path, dpi)
    finally:
        plt.close(fig)
    return out_path


def render_overview(result: dict, fips: list[str], day: date, out_path: Path) -> Path:
    """All five on one sheet, for a quick look."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd

    data_crs = ccrs.PlateCarree()
    projection = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    state = gpd.read_file(API_ROOT / "maps" / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp").to_crs("EPSG:4326")
    counties = _county_shapes()
    fig = plt.figure(figsize=(18, 11.5), dpi=110, facecolor="white")
    try:
        for position, kind in zip((1, 2, 3, 5, 6), MAPS):
            ax = fig.add_subplot(2, 3, position, projection=projection)
            ax.set_extent(EXTENT, crs=data_crs)
            ax.set_frame_on(False)
            mappable, ticks, labels = _draw(ax, data_crs, kind, result, fips)
            ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#b6b6b6", facecolor="none", linewidth=0.3, zorder=8)
            ax.add_geometries(state.geometry, crs=data_crs, edgecolor="black", facecolor="none", linewidth=1.0, zorder=9)
            ax.set_title(MAPS[kind]["title"].replace(" (Experimental)", ""), fontsize=12, fontweight="bold")
            bar = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.02, fraction=0.045, shrink=0.9,
                               **({} if kind == "pfdi" else {"extend": "min", "spacing": "uniform"}))
            bar.set_ticks(ticks)
            bar.set_ticklabels(labels, fontsize=8)
            ax.text(0.5, -0.24, summary_line(kind, result), transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        note = fig.add_subplot(2, 3, 4)
        note.axis("off")
        note.text(0.0, 0.95, f"Fire escape models\n{day:%A %B %d, %Y}", fontsize=18, fontweight="bold", va="top")
        note.text(0.0, 0.70, NOT_FOR_OPERATIONS, fontsize=12, fontweight="bold", color="#B00000", va="top")
        note.text(0.0, 0.60, ("Top row: what the weather would do to a fire.\n"
                              "  1. Physics only: flame length in cured grass.\n"
                              "  2-3. Trained: if a fire starts at that spot,\n       the chance it reaches 10 / 100 acres.\n\n"
                              "Bottom row: the chance a fire of that size is\nreported in the county today (one starts AND\ngrows). "
                              "These are small numbers: large fires\nare rare.\n\n"
                              "Colours are fixed scales, the same every day."), fontsize=10.5, va="top", linespacing=1.35)
        fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.09, wspace=0.08, hspace=0.32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    finally:
        plt.close(fig)
    return out_path


def render_all(result: dict, fips: list[str], day: date, out_dir: Path, gridmet_lag_days: int = 1) -> dict[str, Path]:
    """Every map for the day as <out_dir>/<YYYYMMDD>_<kind>.png, plus the overview."""
    out = {kind: render_map(kind, result, fips, day, Path(out_dir) / f"{day:%Y%m%d}_{kind}.png", gridmet_lag_days) for kind in MAPS}
    out["overview"] = render_overview(result, fips, day, Path(out_dir) / f"{day:%Y%m%d}_overview.png")
    return out

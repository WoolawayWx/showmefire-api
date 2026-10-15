"""Shared "station plot on a Missouri basemap" renderer.

Extracted from the basemap/branding/declutter/wind-barb code that
`rhmap-fil.py`, `windmap-fil.py`, and `fuelmoisturemap.py` each duplicate in
full. New current-conditions products (api/maps/*.py) should import from here
instead of copy-pasting another basemap. The three existing scripts are left
as-is for now (refactoring them to use this module is a safe, optional
follow-up, not required for new products to ship).

One intentional improvement over the siblings: boundaries drawn by
`redraw_boundaries` use a zorder (8/9) high enough to always render on top of
a filled contour/pcolormesh layer (zorder 7 in the existing scripts) -
`rhmap-fil.py`/`windmap-fil.py` redraw boundaries at the SAME zorder (5/6)
used in `generate_basemap()`, which is lower than their contourf's zorder=7
and so has no visible effect; that redundant second draw is not replicated
here.
"""
from __future__ import annotations

import json
from functools import lru_cache
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm, to_rgb
from matplotlib.ticker import MaxNLocator
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
import shapely

MAPS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MAPS_DIR.parent

PIXEL_WIDTH, PIXEL_HEIGHT, MAP_DPI = 2048, 1152, 144
MAP_BACKGROUND = "#E8E8E8"
MO_EXTENT = (-95.8, -89.1, 35.8, 40.8)
MAP_CENTRAL_LON, MAP_CENTRAL_LAT = -92.45, 38.3

COUNTY_SHAPEFILE = MAPS_DIR / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
STATE_SHAPEFILE = MAPS_DIR / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp"

LABEL_DECLUTTER_MIN_DIST = 0.12

VALID_NETWORKS = {"RAWS", "ASOS/AWOS", "MOCOMAGNET", "Missouri Mesonet"}

# scripts/maps.sh runs ~25+ current-conditions scripts back-to-back as
# separate processes every cycle; each independently hitting /stations would
# mean 25+ redundant HTTP round-trips and JSON re-parses of the identical
# payload every cycle. A short-lived file cache (one real fetch, the rest of
# the run's scripts reuse it) turns that into effectively one fetch per
# maps.sh cycle without needing to merge these scripts into one process.
STATION_CACHE_PATH = PROJECT_DIR / "data" / "current_conditions" / "station_cache.json"
STATION_CACHE_MAX_AGE_SECONDS = 120


#  (field_path, neighbor_count, min_abs_deviation, mad_multiplier) - thresholds
# are in each field's native units and sized to its normal local variability
# (RH/wind swing more between neighbors than temperature does), so a single
# shared multiplier wouldn't fit all of them.
QC_RAW_FIELDS = [
    ("observations.air_temp.value", 5, 15.0, 6.0),
    ("observations.dew_point_temperature.value", 5, 15.0, 6.0),
    ("observations.relative_humidity.value", 5, 25.0, 6.0),
    ("observations.wind_speed.value", 5, 20.0, 6.0),
]


# Derived "extreme" fields (running max/min from maps/station_history.py) are
# built from stored history, not the live observation, so QC_RAW_FIELDS can't
# protect them: a stuck sensor's stored 36F "max" sits among neighbors in the
# 80s. Same neighbor-outlier check, keyed by the field's leaf name
# (neighbor_count, min_abs_deviation, mad_multiplier). Applied automatically
# by every plot_* function below.
QC_DERIVED_FIELDS = {
    "air_temp_max_today": (5, 15.0, 6.0),
    "air_temp_min_today": (5, 15.0, 6.0),
    "feels_like_max_today": (5, 15.0, 6.0),
    "feels_like_min_today": (5, 15.0, 6.0),
    "wbgt_max_today": (5, 10.0, 6.0),
    "dewpoint_max_24hr": (5, 15.0, 6.0),
    # Wind gusts (raw and the history-derived maxima) swing a lot between
    # neighbors in a storm, so the absolute floor is wide; this only catches
    # a stuck sensor reading 90 mph among stations in the teens.
    "wind_gust": (5, 25.0, 6.0),
    "gust_max_1hr": (5, 25.0, 6.0),
    "gust_max_today": (5, 25.0, 6.0),
    "gust_max_yesterday": (5, 25.0, 6.0),
}


def qc_derived_field(stations: list[dict], field_path: str) -> int:
    """Run the neighbor-outlier QC on `field_path` if it is a known derived
    extreme field (see QC_DERIVED_FIELDS); no-op for anything else. Mutates
    `stations` in place and returns the number of values nulled."""
    parts = field_path.split(".")
    leaf = parts[-2] if len(parts) >= 2 and parts[-1] == "value" else parts[-1]
    params = QC_DERIVED_FIELDS.get(leaf)
    if params is None:
        return 0
    neighbor_count, min_abs_deviation, mad_multiplier = params
    return qc_filter_field(
        stations, field_path,
        neighbor_count=neighbor_count, min_abs_deviation=min_abs_deviation, mad_multiplier=mad_multiplier,
    )


def fetch_stations(*, valid_networks: Optional[set] = None, max_age_seconds: float = STATION_CACHE_MAX_AGE_SECONDS) -> list[dict]:
    """Return the current station list, reusing a recent on-disk cache
    instead of re-fetching /stations when another script already did so
    within `max_age_seconds`. Falls back to a live fetch on any cache miss
    or read failure - never raises due to a stale/corrupt cache file.

    Raw observation fields are QC'd against each station's local neighbors
    (see qc_filter_field) before being returned, so every map built on this
    - and every value derived from these fields, like feels-like/WBGT/theta-E
    - is protected from a single stuck/miscalibrated sensor (e.g. a lone 36F
    reading in a field of 70s) without each script needing its own check."""
    cached = _read_station_cache(max_age_seconds)
    if cached is None:
        cached = _fetch_and_cache_stations()
    networks = valid_networks if valid_networks is not None else VALID_NETWORKS
    stations = [s for s in cached if s.get("network") in networks]
    for field_path, neighbor_count, min_abs_deviation, mad_multiplier in QC_RAW_FIELDS:
        qc_filter_field(
            stations, field_path,
            neighbor_count=neighbor_count, min_abs_deviation=min_abs_deviation, mad_multiplier=mad_multiplier,
        )
    return stations


def qc_filter_field(
    stations: list[dict], field_path: str, *,
    neighbor_count: int = 5, min_abs_deviation: float = 15.0, mad_multiplier: float = 6.0,
) -> int:
    """Null out a field's value at stations whose reading is a statistical
    outlier versus its nearest neighbors - catches a stuck/miscalibrated
    sensor (e.g. a lone 36F reading surrounded by stations in the 70s) before
    it reaches a dot plot or an interpolated fill, rather than needing a human
    to notice it on a published map.

    For each station, compares its value to the median of its
    `neighbor_count` nearest (by lat/lon) stations that also report the
    field, then flags it as an outlier if that residual exceeds BOTH
    `min_abs_deviation` (an absolute floor, so the check doesn't fire on tiny
    deviations when stations happen to be unusually uniform) and
    `mad_multiplier` times the statewide median absolute deviation of all
    stations' local residuals (so the threshold adapts to how noisy this
    field normally is day to day). Mutates `stations` in place, setting
    flagged values to None (identical to the field never having been
    reported). Returns the number of stations flagged."""
    entries = []
    for station in stations:
        value = _get_path(station, field_path)
        lon, lat = station.get("longitude"), station.get("latitude")
        if value is None or lon is None or lat is None:
            continue
        entries.append((station, value, lon, lat))
    if len(entries) < neighbor_count + 1:
        return 0

    positions = np.array([(lon, lat) for _, _, lon, lat in entries], dtype=float)
    values = np.array([value for _, value, _, _ in entries], dtype=float)

    residuals = np.empty(len(entries))
    for i in range(len(entries)):
        dists = np.linalg.norm(positions - positions[i], axis=1)
        nearest_idx = np.argsort(dists)[1:neighbor_count + 1]  # [0] is always itself (dist 0)
        residuals[i] = values[i] - np.median(values[nearest_idx])

    robust_scale = 1.4826 * (np.median(np.abs(residuals - np.median(residuals))) or 0.0)
    threshold = max(min_abs_deviation, mad_multiplier * robust_scale)

    flagged = 0
    parts = field_path.split(".")
    for i, (station, _, _, _) in enumerate(entries):
        if abs(residuals[i]) <= threshold:
            continue
        node = station
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None
        flagged += 1
    return flagged


def buffer_anchor_points(field_path: str, stations: Optional[list[dict]] = None) -> list[tuple[float, float, float]]:
    """(lon, lat, value) for neighbor-state stations (/stations/buffer) reporting
    `field_path`, to pass as plot_filled_field(extra_points=...) so the fill is
    anchored at the Missouri border instead of extrapolated past it. A raw
    field with a QC rule (QC_RAW_FIELDS) gets the same neighbor-outlier check
    here, run together with `stations` so border stations are compared across
    the line. Returns [] if the buffer feed is unavailable."""
    from maps.station_rbf import buffer_stations

    buffer = buffer_stations()
    for qc_path, neighbor_count, min_abs_deviation, mad_multiplier in QC_RAW_FIELDS:
        if qc_path == field_path:
            qc_filter_field(
                list(stations or []) + buffer, field_path,
                neighbor_count=neighbor_count, min_abs_deviation=min_abs_deviation, mad_multiplier=mad_multiplier,
            )
    points = []
    for station in buffer:
        value = _get_path(station, field_path)
        lon, lat = station.get("longitude"), station.get("latitude")
        if value is not None and lon is not None and lat is not None:
            points.append((lon, lat, value))
    return points


def _read_station_cache(max_age_seconds: float) -> Optional[list[dict]]:
    try:
        if time.time() - STATION_CACHE_PATH.stat().st_mtime > max_age_seconds:
            return None
        payload = json.loads(STATION_CACHE_PATH.read_text(encoding="utf-8"))
        return payload["stations"]
    except (OSError, ValueError, KeyError):
        return None


def _fetch_and_cache_stations() -> list[dict]:
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    port = os.getenv("PORT", "8000")
    response = requests.get(f"http://localhost:{port}/stations", timeout=30)
    response.raise_for_status()
    stations = response.json()["stations"]
    try:
        _atomic_json_write(STATION_CACHE_PATH, {"stations": stations})
    except OSError:
        pass  # Caching is an optimization, not a correctness requirement.
    return stations


def generate_basemap():
    """Missouri Lambert Conformal basemap, matching rhmap-fil.py/windmap-fil.py's
    existing look exactly (facecolor, extent, dpi, projection)."""
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=MAP_CENTRAL_LON, central_latitude=MAP_CENTRAL_LAT)

    fig = plt.figure(
        figsize=(PIXEL_WIDTH / MAP_DPI, PIXEL_HEIGHT / MAP_DPI),
        dpi=MAP_DPI,
        facecolor=MAP_BACKGROUND,
    )
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(MO_EXTENT, crs=data_crs)
    # MO_EXTENT's aspect ratio (~1.09) doesn't match this 16:9 canvas (~1.78),
    # so the equal-aspect GeoAxes needs to pad one dimension to fit. The
    # default anchor='C' centers the state in the frame, splitting that pad
    # into equal left/right gutters - which eats into the right-side
    # title/description text. Anchoring west keeps the state flush against
    # the left edge (matching the existing branding layout, which already
    # reserves the right side for text) and pushes all the padding to the
    # right instead.
    ax.set_anchor("W")
    # Station dots/labels plotted after this (some sit right at the state
    # border) would otherwise trigger Matplotlib autoscale and silently
    # re-fit the view, shifting the whole map sideways under the equal-aspect
    # GeoAxes constraint.
    ax.set_autoscale_on(False)

    redraw_boundaries(ax, data_crs)
    return fig, ax, data_crs, map_crs, MAP_DPI


@lru_cache(maxsize=8)
def _projected_boundary(path: str, proj4: str):
    """Boundary geometries reprojected once into the map projection.

    Cartopy re-projects lat/lon geometries on every draw, which dominated
    render time (~5s per basemap for the dense county/state outlines);
    pre-projecting here and drawing with the map CRS skips that and renders
    pixel-identically. Cached so scripts drawing several maps in one process
    (e.g. the advisory maps) pay for it once.
    """
    return gpd.read_file(path).to_crs(proj4).geometry


def redraw_boundaries(ax, data_crs, *, county_zorder: int = 8, state_zorder: int = 9) -> None:
    """County + state outlines. Default zorders are high enough to stay
    visible on top of a filled contour/pcolormesh layer underneath."""
    map_crs = ax.projection
    if COUNTY_SHAPEFILE.is_file():
        ax.add_geometries(
            _projected_boundary(str(COUNTY_SHAPEFILE), map_crs.proj4_init), crs=map_crs,
            edgecolor="#B6B6B6", facecolor="none", linewidth=1, zorder=county_zorder,
        )
    if STATE_SHAPEFILE.is_file():
        ax.add_geometries(
            _projected_boundary(str(STATE_SHAPEFILE), map_crs.proj4_init), crs=map_crs,
            edgecolor="#000000", facecolor="none", linewidth=1.5, zorder=state_zorder,
        )


_FONTS_LOADED = False


def _ensure_fonts_loaded() -> None:
    global _FONTS_LOADED
    if _FONTS_LOADED:
        return
    font_paths = [
        PROJECT_DIR / "assets/Montserrat/static/Montserrat-Regular.ttf",
        PROJECT_DIR / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        PROJECT_DIR / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Montserrat"
    _FONTS_LOADED = True


def add_branding(fig, ax, *, title: str, subtitle: str, description: str) -> None:
    """Title/subtitle/description block + "ShowMeFire.org" wordmark + logo,
    matching the look of rhmap-fil.py/windmap-fil.py/fuelmoisturemap.py."""
    _ensure_fonts_loaded()
    fig.text(0.99, 0.97, title, fontsize=26, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans")
    fig.text(0.99, 0.90, subtitle, fontsize=16, ha="right", va="top", fontname="Montserrat")
    fig.text(0.99, 0.6, description, fontsize=12, ha="right", va="top", fontname="Montserrat", linespacing=1.4)
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat")

    svg_path = PROJECT_DIR / "assets/LightBackGroundLogo.svg"
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(url=str(svg_path))
        image = mpimg.imread(BytesIO(png_bytes), format="png")
    except (ImportError, FileNotFoundError, OSError):
        image = None
    if image is not None:
        imagebox = OffsetImage(image, zoom=0.03)
        annotation = AnnotationBbox(
            imagebox, (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0),
        )
        ax.add_artist(annotation)


def _get_path(station: dict, path: str) -> Any:
    """Resolve a dotted path like 'observations.air_temp.value' against a
    station dict, returning None at any missing/None step."""
    value: Any = station
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def _declutter(stations: list[dict], field_paths: list[str], min_dist: float) -> list[tuple[dict, list[Any]]]:
    """Yield (station, [value_for_each_field_path]) pairs for stations where
    every field_path resolves to a non-None value, skipping any station too
    close to an already-placed label (same de-dup logic as rhmap-fil.py's
    label_positions/min_dist check)."""
    placed: list[np.ndarray] = []
    accepted: list[tuple[dict, list[Any]]] = []
    for station in stations:
        values = [_get_path(station, path) for path in field_paths]
        if any(value is None for value in values):
            continue
        lon, lat = station.get("longitude"), station.get("latitude")
        if lon is None or lat is None:
            continue
        pos = np.array([lon, lat])
        if any(np.linalg.norm(pos - placed_pos) <= min_dist for placed_pos in placed):
            continue
        placed.append(pos)
        accepted.append((station, values))
    return accepted


def plot_station_values(
    ax, data_crs, stations: list[dict], *, field_path: str,
    formatter: Callable[[Any], str] = lambda v: f"{v:.0f}",
    min_label_dist: float = LABEL_DECLUTTER_MIN_DIST, color: str = "black",
) -> int:
    """Dot + text label at each station for one numeric field, decluttered
    the same way rhmap-fil.py does. Returns the number of stations plotted."""
    qc_derived_field(stations, field_path)
    plotted = 0
    for station, (value,) in _declutter(stations, [field_path], min_label_dist):
        lon, lat = station["longitude"], station["latitude"]
        ax.scatter(lon, lat, transform=data_crs, color=color, s=20, zorder=10, edgecolor="white", linewidth=1)
        ax.text(
            lon, lat + 0.03, formatter(value), transform=data_crs, fontsize=9, color=color, zorder=11,
            ha="center", va="bottom",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )
        plotted += 1
    return plotted


def plot_station_dual_values(
    ax, data_crs, stations: list[dict], *, top_field_path: str, bottom_field_path: str,
    top_formatter: Callable[[Any], str] = lambda v: f"{v:.0f}",
    bottom_formatter: Callable[[Any], str] = lambda v: f"{v:.0f}",
    top_color: str = "#8B0000", bottom_color: str = "#1D4ED8",
    min_label_dist: float = LABEL_DECLUTTER_MIN_DIST,
) -> int:
    """Dot + two stacked text labels per station (e.g. today's max over min,
    or RH over dewpoint) - the "two numbers at one dot" capability none of
    the existing maps/*.py scripts have."""
    qc_derived_field(stations, top_field_path)
    qc_derived_field(stations, bottom_field_path)
    plotted = 0
    for station, (top_value, bottom_value) in _declutter(stations, [top_field_path, bottom_field_path], min_label_dist):
        lon, lat = station["longitude"], station["latitude"]
        ax.scatter(lon, lat, transform=data_crs, color="black", s=20, zorder=10, edgecolor="white", linewidth=1)
        ax.text(
            lon, lat + 0.09, top_formatter(top_value), transform=data_crs, fontsize=9, color=top_color, zorder=11,
            fontweight="bold", ha="center", va="bottom",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )
        ax.text(
            lon, lat - 0.03, bottom_formatter(bottom_value), transform=data_crs, fontsize=9, color=bottom_color, zorder=11,
            fontweight="bold", ha="center", va="top",
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )
        plotted += 1
    return plotted


# Fine grid + many levels + a matching blur keep the filled contours smooth
# (a coarse 200-cell grid with 41 levels showed visibly stepped, jagged edges).
@lru_cache(maxsize=1)
def _missouri_geometry():
    """Missouri outline in lon/lat, read once per process (every filled map
    needs it for its mask)."""
    if not STATE_SHAPEFILE.is_file():
        return None
    state = gpd.read_file(STATE_SHAPEFILE).to_crs("EPSG:4326")
    return None if state.empty else state.geometry.iloc[0]


_FILL_GRID_CACHE: dict = {}


def _fill_grid(grid_size: int, map_crs, data_crs):
    """(lon mesh, lat mesh, in-state mask, projected x, projected y) for the
    fill grid, built once per process and reused by every map it draws.

    Contouring in the map's own projected coordinates (rather than handing
    cartopy lon/lat with transform=PlateCarree) skips cartopy re-projecting
    every contour polygon on each draw - the single biggest cost of a filled
    map (~7s of ~11s) - and renders the same picture."""
    key = (grid_size, map_crs.proj4_init)
    cached = _FILL_GRID_CACHE.get(key)
    if cached is None:
        grid_lon = np.linspace(MO_EXTENT[0], MO_EXTENT[1], grid_size)
        grid_lat = np.linspace(MO_EXTENT[2], MO_EXTENT[3], grid_size)
        lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)
        geometry = _missouri_geometry()
        mask = shapely.contains_xy(geometry, lon_mesh, lat_mesh) if geometry is not None else None
        projected = map_crs.transform_points(data_crs, lon_mesh, lat_mesh)
        cached = (lon_mesh, lat_mesh, mask, projected[..., 0], projected[..., 1])
        _FILL_GRID_CACHE[key] = cached
    return cached


FILL_GRID_SIZE = 400
FILL_ALPHA = 0.8
FILL_BLUR_SIGMA = 4
FILL_LEVEL_COUNT = 161


def opaque_blend_cmap(cmap_name: str, alpha: float = FILL_ALPHA, background: str = MAP_BACKGROUND) -> ListedColormap:
    """`cmap_name` pre-blended with the (uniform) map background at `alpha`, so
    the fill can be drawn fully opaque. Semi-transparent contourf bands overlap
    slightly where they meet, which shows up as a thin darker stroke along every
    contour edge; opaque bands with identical on-screen color don't."""
    base = plt.get_cmap(cmap_name)
    bg = np.array(to_rgb(background))

    def blend(rgba):
        return tuple(alpha * np.array(rgba[:3]) + (1 - alpha) * bg) + (1.0,)

    # Index by integer so the blended map keeps the original's N: categorical
    # maps (e.g. 5 fire-danger colors) rely on a BoundaryNorm built from cmap.N.
    blended = ListedColormap([blend(base(i)) for i in range(base.N)], name=f"{getattr(base, 'name', 'cmap')}_opaque")
    blended.set_under(blend(base(0)))
    blended.set_over(blend(base(base.N - 1)))
    return blended


def plot_filled_field(
    fig, ax, data_crs, stations: list[dict], *, field_path: str,
    cmap: str = "RdYlBu_r", diverging: bool = False, label: str = "",
    vmin: Optional[float] = None, vmax: Optional[float] = None,
    grid_size: int = FILL_GRID_SIZE,
    levels: Optional[list] = None, colors: Optional[list] = None,
    tick_labels: Optional[list] = None, clip_min: Optional[float] = None,
    extra_points: Optional[list[tuple[float, float, float]]] = None,
) -> int:
    """RBF-interpolated filled contour of a numeric field across Missouri,
    masked to the state outline, with a colorbar - the same
    interpolate/mask/contourf approach rhmap-fil.py and moisture_convergence.py
    already use, generalized to any station field. Drawn at zorder=7, below
    redraw_boundaries' default county/state zorders (8/9) and
    plot_station_values'/plot_wind_barbs' dot/label/barb zorders (10/11), so
    callers can call this before or after those - the stacking order is
    zorder-driven, not call-order-driven. Returns the number of stations used
    for the interpolation (0 if there weren't enough to fit one).

    `vmin`/`vmax` set a FIXED color scale (for `diverging=True`, pass only
    `vmax` - the fixed scale runs -vmax to +vmax) so the same value always
    maps to the same color across runs, instead of each render rescaling to
    that cycle's min/max (which makes maps impossible to compare over time
    and lets one bad reading stretch the whole scale). Callers should always
    pass fixed bounds sized to the field's realistic range; values outside
    the range still render, clipped to the end-cap color via
    `extend='both'`, so an out-of-range reading is visible as "pegged" rather
    than silently absent.

    `extra_points` are (lon, lat, value) anchors outside the plotted station
    list (e.g. neighbor-state stations from /stations/buffer): they steer the
    interpolation near the border but get no markers or labels, and the fill is
    still masked to Missouri."""
    qc_derived_field(stations, field_path)
    points_lon, points_lat, points_val = [], [], []
    for lon, lat, value in extra_points or []:
        points_lon.append(lon)
        points_lat.append(lat)
        points_val.append(value)
    for station in stations:
        value = _get_path(station, field_path)
        lon, lat = station.get("longitude"), station.get("latitude")
        if value is None or lon is None or lat is None:
            continue
        points_lon.append(lon)
        points_lat.append(lat)
        points_val.append(value)
    if len(points_lon) < 4:  # Rbf needs enough points for a stable fit
        return 0

    field_grid = interpolate_to_grid(points_lon, points_lat, points_val, ax, data_crs, grid_size=grid_size)
    return draw_filled_grid(
        fig, ax, data_crs, field_grid, cmap=cmap, diverging=diverging, label=label, vmin=vmin, vmax=vmax,
        grid_size=grid_size, levels=levels, colors=colors, tick_labels=tick_labels, clip_min=clip_min,
        point_count=len(points_lon),
    )


def interpolate_to_grid(points_lon, points_lat, values, ax, data_crs, *, grid_size: int = FILL_GRID_SIZE, sigma: float = FILL_BLUR_SIGMA) -> np.ndarray:
    """RBF-interpolate scattered station values onto the fill grid (lon/lat
    mesh from _fill_grid, rows = latitude) and blur, unmasked."""
    lon_mesh, lat_mesh, _mask, _x, _y = _fill_grid(grid_size, ax.projection, data_crs)
    return gaussian_filter(
        Rbf(points_lon, points_lat, values, function="multiquadric", smooth=0.5)(lon_mesh, lat_mesh), sigma=sigma,
    )


def draw_filled_grid(
    fig, ax, data_crs, field_grid: np.ndarray, *, cmap: str = "RdYlBu_r", diverging: bool = False, label: str = "",
    vmin: Optional[float] = None, vmax: Optional[float] = None, grid_size: int = FILL_GRID_SIZE,
    levels: Optional[list] = None, colors: Optional[list] = None, tick_labels: Optional[list] = None,
    clip_min: Optional[float] = None, point_count: int = 0,
) -> int:
    """Mask `field_grid` (from interpolate_to_grid, or computed from one - e.g.
    a divergence) to Missouri and draw it as a filled contour + colorbar.
    Returns point_count, or 0 if nothing finite remained to draw."""
    _lon_mesh, _lat_mesh, mask, x_mesh, y_mesh = _fill_grid(grid_size, ax.projection, data_crs)
    field_grid = np.array(field_grid, dtype=float)
    if clip_min is not None:
        field_grid = np.maximum(field_grid, clip_min)
    if mask is not None:
        field_grid[~mask] = np.nan

    finite = field_grid[np.isfinite(field_grid)]
    if not finite.size:
        return 0
    if levels is not None:
        norm = BoundaryNorm(levels, ncolors=len(levels) - 1, clip=True) if colors is not None else None
        draw_levels = levels
    elif diverging:
        if vmax is None:
            raise ValueError("plot_filled_field(diverging=True) requires a fixed vmax")
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        draw_levels = np.linspace(-vmax, vmax, FILL_LEVEL_COUNT)
    else:
        if vmin is None or vmax is None:
            raise ValueError("plot_filled_field requires fixed vmin/vmax (pass the field's realistic range)")
        norm = None
        draw_levels = np.linspace(vmin, vmax, FILL_LEVEL_COUNT)

    if colors is not None:
        fill_cmap = ListedColormap(colors)
    else:
        fill_cmap = opaque_blend_cmap(cmap)
    mesh = ax.contourf(
        x_mesh, y_mesh, field_grid,
        levels=draw_levels, cmap=fill_cmap, norm=norm, zorder=7, extend="both" if levels is None else "neither",
        antialiased=False,
    )
    # Missouri's own western border angles inward below the NW panhandle (see
    # generate_basemap's west anchor), leaving a wedge of background open
    # from about x=40px down to a steady ~230px between roughly y=400-950px
    # (in the 2048x1152 canvas) - the same wedge the forecast graphics' own
    # legend sits in. Put the colorbar there instead of over the map.
    cax = fig.add_axes([0.025, 0.15, 0.018, 0.5])
    # Round tick values regardless of how many contour levels there are.
    if tick_labels is not None:
        # Categorical fill: one tick per class, centered in its band.
        colorbar = fig.colorbar(mesh, cax=cax, label=label, ticks=[(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)])
        colorbar.ax.set_yticklabels(tick_labels)
        colorbar.ax.tick_params(length=0)
    else:
        fig.colorbar(mesh, cax=cax, label=label, ticks=MaxNLocator(nbins=8))
    return point_count


def plot_wind_barbs(
    ax, data_crs, stations: list[dict], *,
    speed_path: str = "observations.wind_speed.value",
    dir_path: str = "observations.wind_direction.value",
    gust_path: Optional[str] = None, length: float = 6, color: str = "black",
) -> int:
    """Wind barbs at each station with both speed and direction, converting
    meteorological "from" direction to a u/v vector the same way
    windmap-fil.py does. gust_path is currently unused by ax.barbs (matplotlib
    has no native gust-barb variant) but accepted so callers can still filter
    on/report gust data alongside the barb."""
    lons, lats, us, vs = [], [], [], []
    for station, (speed, direction) in _declutter(stations, [speed_path, dir_path], min_dist=0.0):
        lons.append(station["longitude"])
        lats.append(station["latitude"])
        us.append(-speed * np.sin(np.radians(direction)))
        vs.append(-speed * np.cos(np.radians(direction)))
    if not lons:
        return 0
    ax.barbs(np.array(lons), np.array(lats), np.array(us), np.array(vs), transform=data_crs, length=length, zorder=10, color=color)
    return len(lons)


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_and_record_status(fig, out_png_path: Path, status_path: Path, status_payload: dict, *, dpi: int = MAP_DPI) -> None:
    """Save the figure (atomically, matching spread_rate.py's tmp-file +
    os.replace pattern) and write a dedicated per-product status JSON rather
    than clobbering the shared root status.json - with ~25 products using
    this helper, one shared file being read-modify-written by all of them
    would be fragile."""
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{out_png_path.stem}.", suffix=".png", dir=out_png_path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        fig.savefig(temp_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
        os.replace(temp_path, out_png_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        # Each script is its own process in production (maps.sh), so this
        # doesn't matter there - but closing matches the existing sibling
        # scripts' convention and avoids a figure leak if these are ever
        # run in-process together (as the test suite does).
        plt.close(fig)
    _atomic_json_write(status_path, status_payload)

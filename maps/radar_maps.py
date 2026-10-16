"""Radar maps: Missouri and National base reflectivity.

Both images come from NOAA's RIDGE 1x1 km base-reflectivity mosaic service as
one transparent PNG each (fetched in parallel), then drawn over the standard
Missouri basemap / a CONUS basemap. Radar changes every few minutes, so this
is the script worth running most often; it is fast (~3s for both maps).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import geopandas as gpd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from rasterio.features import rasterize
from rasterio.transform import from_bounds

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps import station_products as spx

SERVICE = "https://mapservices.weather.noaa.gov/eventdriven/rest/services/radar/radar_base_reflectivity/MapServer"
IMAGE_LAYER = 3
MISSOURI_BBOX, MISSOURI_SIZE = (-95.8, 35.8, -89.1, 40.8), (1600, 1194)
US_BBOX, US_SIZE = (-126.0, 23.0, -66.0, 50.0), (2400, 1080)
US_EXTENT = (-124.2, -67.5, 24.0, 49.6)
SOURCE = "Data Source: NOAA/NWS RIDGE Radar Mosaic\nBase Reflectivity (1x1 km)"

# The mosaic service draws a continuous color ramp and publishes no dBZ table for it, so
# the legend is built from the colors the service actually returns (sampled from its
# output, light to intense) rather than a numeric key that would not match the image.
INTENSITY_COLORS = ["#506ca5", "#5289b8", "#58beb9", "#43d484", "#0da912", "#297509", "#ccc202", "#ffbf00", "#ff7400", "#fd2f00"]
INTENSITY_LABELS = [(0.08, "Light"), (0.40, "Moderate"), (0.72, "Heavy"), (0.95, "Intense")]


def fetch_radar(bbox: tuple, size: tuple) -> np.ndarray:
    response = requests.get(f"{SERVICE}/export", params={
        "bbox": ",".join(str(v) for v in bbox), "bboxSR": "4326", "imageSR": "4326",
        "size": f"{size[0]},{size[1]}", "format": "png32", "transparent": "true",
        "layers": f"show:{IMAGE_LAYER}", "f": "image",
    }, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("radar service did not return a PNG")
    return mpimg.imread(BytesIO(response.content), format="png")


def _add_intensity_legend(fig, x: float, y: float, width: float, height: float) -> None:
    """Boxed horizontal color bar (light -> intense) with its lower-left corner at (x, y) in
    figure fractions; drawn above the map and sized so it never needs to sit on the data."""
    pad_x, pad_top, pad_bottom = 0.012, 0.045, 0.045
    box_height = height + pad_top + pad_bottom + 0.012
    fig.add_artist(Rectangle(
        (x - pad_x, y - pad_bottom), width + 2 * pad_x, box_height, transform=fig.transFigure,
        facecolor="white", edgecolor="#999999", linewidth=0.8, zorder=40,
    ))
    fig.text(x, y + height + 0.012, "Radar Intensity (base reflectivity)", fontsize=11, fontweight="bold",
             ha="left", va="bottom", color="#222222", zorder=60)
    bar = fig.add_axes((x, y, width, height), zorder=50)
    ramp = LinearSegmentedColormap.from_list("radar_intensity", INTENSITY_COLORS)
    bar.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=ramp, extent=(0, 1, 0, 1))
    bar.set_xticks([]); bar.set_yticks([])
    for spine in bar.spines.values():
        spine.set_edgecolor("#555555"); spine.set_linewidth(0.6)
    for position, label in INTENSITY_LABELS:
        fig.text(x + position * width, y - 0.01, label, fontsize=10, ha="center", va="top", color="#222222", zorder=60)


def _mask_to_state(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """Make radar transparent outside Missouri's outline (bbox is lon/lat west, south, east, north)."""
    state = sp._missouri_geometry()
    if state is None:
        return image
    height, width = image.shape[:2]
    mask = rasterize([(state, 1)], out_shape=(height, width), transform=from_bounds(*bbox, width, height), fill=0, dtype="uint8").astype(bool)
    masked = image.copy()
    masked[~mask, 3] = 0.0
    return masked


def _logo():
    return sp.logo_image()


def render_missouri(image: np.ndarray) -> dict:
    started = time.time()
    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    bbox = MISSOURI_BBOX
    ax.imshow(_mask_to_state(image, bbox), origin="upper", extent=(bbox[0], bbox[2], bbox[1], bbox[3]), transform=data_crs, interpolation="nearest", zorder=3)
    _add_intensity_legend(fig, x=0.745, y=0.22, width=0.225, height=0.028)
    sp.add_branding(
        fig, ax, title="Missouri Radar", subtitle=f"Valid Time: {spx.valid_time_label()}",
        description=f"{SOURCE}\n\nLatest composite base reflectivity, shown\nover Missouri only. Colors show precipitation\nintensity: blue-teal is light, green moderate,\nyellow-orange heavy and red intense.",
    )
    status = {"product": "radar_missouri", "status": "updated", "last_update": spx.valid_time_label(), "runtime_sec": round(time.time() - started, 2)}
    sp.save_and_record_status(fig, spx.IMAGES_DIR / "mo-radar.png", spx.STATUS_DIR / "radar_missouri.json", status)
    print(f"Missouri radar updated ({status['runtime_sec']:.2f}s)")
    return status


@lru_cache(maxsize=2)
def _us_boundaries(proj4: str):
    states = gpd.read_file(shapereader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces_lakes"))
    us_states = states[states["iso_a2"] == "US"].to_crs(proj4).geometry
    others = states[states["iso_a2"] != "US"].to_crs(proj4).geometry
    return us_states, others, sp._projected_boundary(str(sp.STATE_SHAPEFILE), proj4)


def render_national(image: np.ndarray) -> dict:
    started = time.time()
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0, standard_parallels=(33, 45))
    fig = plt.figure(figsize=(sp.PIXEL_WIDTH / sp.MAP_DPI, sp.PIXEL_HEIGHT / sp.MAP_DPI), dpi=sp.MAP_DPI, facecolor=sp.MAP_BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 0.9], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_extent(US_EXTENT, crs=data_crs)
    ax.set_autoscale_on(False)

    bbox = US_BBOX
    ax.imshow(image, origin="upper", extent=(bbox[0], bbox[2], bbox[1], bbox[3]), transform=data_crs, interpolation="nearest", zorder=3)
    us_states, others, missouri = _us_boundaries(map_crs.proj4_init)
    ax.add_geometries(others, crs=map_crs, edgecolor="#9A9A9A", facecolor="none", linewidth=0.5, zorder=5)
    ax.add_geometries(us_states, crs=map_crs, edgecolor="#555555", facecolor="none", linewidth=0.6, zorder=6)
    ax.add_geometries(missouri, crs=map_crs, edgecolor="#202020", facecolor="none", linewidth=1.4, zorder=8)

    sp._ensure_fonts_loaded()
    fig.text(0.015, 0.975, "National Radar", fontsize=26, fontweight="bold", ha="left", va="top", fontname="Plus Jakarta Sans")
    fig.text(0.015, 0.925, f"Valid Time: {spx.valid_time_label()}  |  {SOURCE.splitlines()[0]}", fontsize=13, ha="left", va="top", fontname="Montserrat")
    _add_intensity_legend(fig, x=0.03, y=0.085, width=0.24, height=0.03)
    fig.text(0.99, 0.015, "ShowMeFire.org", fontsize=16, fontweight="bold", ha="right", va="bottom", fontname="Montserrat")
    logo = _logo()
    if logo is not None:
        ax.add_artist(AnnotationBbox(OffsetImage(logo, zoom=0.024), (0.985, 0.93), frameon=False, xycoords="figure fraction", box_alignment=(1, 1)))

    status = {"product": "radar_national", "status": "updated", "last_update": spx.valid_time_label(), "runtime_sec": round(time.time() - started, 2)}
    sp.save_and_record_status(fig, spx.IMAGES_DIR / "mo-radar-national.png", spx.STATUS_DIR / "radar_national.json", status)
    print(f"National radar updated ({status['runtime_sec']:.2f}s)")
    return status


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        missouri = pool.submit(fetch_radar, MISSOURI_BBOX, MISSOURI_SIZE)
        national = pool.submit(fetch_radar, US_BBOX, US_SIZE)
        spx.run_products([
            ("Missouri radar", lambda: render_missouri(missouri.result())),
            ("National radar", lambda: render_national(national.result())),
        ], label="Radar maps")


if __name__ == "__main__":
    main()

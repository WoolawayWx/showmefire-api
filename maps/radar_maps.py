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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch

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

# The standard NWS reflectivity key the mosaic is drawn with.
DBZ_COLORS = [
    ("5", "#04E9E7"), ("10", "#019FF4"), ("15", "#0300F4"), ("20", "#02FD02"), ("25", "#01C501"),
    ("30", "#008E00"), ("35", "#FDF802"), ("40", "#E5BC00"), ("45", "#FD9500"), ("50", "#FD0000"),
    ("55", "#D40000"), ("60", "#BC0000"), ("65", "#F800FD"), ("70", "#9854C6"),
]


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


def _dbz_legend_handles():
    return [Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=f"{label}") for label, color in DBZ_COLORS]


@lru_cache(maxsize=1)
def _logo():
    try:
        import cairosvg

        return mpimg.imread(BytesIO(cairosvg.svg2png(url=str(PROJECT_DIR / "assets/LightBackGroundLogo.svg"))), format="png")
    except Exception:  # noqa: BLE001 - the logo is decorative
        return None


def render_missouri(image: np.ndarray) -> dict:
    started = time.time()
    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    bbox = MISSOURI_BBOX
    ax.imshow(image, origin="upper", extent=(bbox[0], bbox[2], bbox[1], bbox[3]), transform=data_crs, interpolation="nearest", zorder=3)
    legend = fig.legend(
        handles=_dbz_legend_handles(), title="Reflectivity (dBZ)", loc="center left", bbox_to_anchor=(0.012, 0.42),
        frameon=True, framealpha=1.0, facecolor="white", edgecolor="#999999", fontsize=9, title_fontsize=10, handlelength=1.6, handleheight=1.0, labelspacing=0.5,
    )
    legend.set_zorder(20)
    sp.add_branding(
        fig, ax, title="Missouri Radar", subtitle=f"Valid Time: {spx.valid_time_label()}",
        description=f"{SOURCE}\n\nLatest composite base reflectivity. Colors\nshow precipitation intensity; green is light\nrain, yellow-orange heavy, red-purple intense\nor hail.",
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
    legend = fig.legend(
        handles=_dbz_legend_handles(), title="Reflectivity (dBZ)", loc="lower left", bbox_to_anchor=(0.012, 0.03), ncol=7,
        frameon=True, facecolor="white", edgecolor="#999999", fontsize=9, title_fontsize=10, handlelength=1.4, handleheight=0.9, columnspacing=1.0,
    )
    legend.set_zorder(20)
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

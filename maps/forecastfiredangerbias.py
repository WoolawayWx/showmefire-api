"""Forecast-vs-realtime fire danger bias map.

Diffs the forecast fire-danger class valid nearest to *now* (read from the
GIS publisher's canonical EPSG:32615 catalog, written hourly by
forecast/DailyForecast.py) against the current realtime/observed fire-danger
class (RGBA GeoTIFF written by realtimefiredanger.py), regridded onto the
same canonical grid so the two can be subtracted directly.

Positive bias = forecast ran hotter (higher danger class) than what's
actually being observed right now; negative = forecast ran cooler.

Runs every 15 min via scripts/maps.sh (auto-discovered — no cron change
needed), publishing images/mo-forecastfiredangerbias.png which upload_cdn.py
then pushes to cdn.showmefire.org/latest/ like every other map script here.
"""
import json
from station_plot import opaque_blend_cmap
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pyproj import Transformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from core.config import GIS_DIR  # noqa: E402
from core.fire_danger import CATEGORY_RGBA_COLORS  # noqa: E402
from services.gis_publisher import CRS as CANONICAL_CRS, canonical_grid, regrid_lonlat  # noqa: E402
from forecast.map_style import add_boundaries, create_base_map  # noqa: E402

FORECAST_PRODUCT = "forecast_fire_danger"
REALTIME_TIF = PROJECT_DIR / "gis/realtime/realtime_fire_danger.tif"
OUTPUT_PNG = PROJECT_DIR / "images/mo-forecastfiredangerbias.png"

# Same five-class colors realtimefiredanger.py / observed_peak_history.py use
# to paint the realtime RGBA GeoTIFF - needed here to decode color back to class.
REALTIME_CLASS_COLORS = CATEGORY_RGBA_COLORS

# -5..+5 diverging scale: blue = forecast under-predicted, red = over-predicted.
BIAS_BINS = np.arange(-5.5, 6.5, 1.0)
BIAS_LABELS = [str(v) for v in range(-5, 6)]
BIAS_COLORS = plt.get_cmap("RdBu_r")(np.linspace(0, 1, len(BIAS_LABELS)))
BIAS_CMAP = ListedColormap(BIAS_COLORS)
BIAS_NORM = BoundaryNorm(BIAS_BINS, BIAS_CMAP.N)


def _load_forecast_catalog_item():
    """Return the forecast_fire_danger catalog item whose valid_time is nearest now."""
    catalog_path = GIS_DIR / "catalog.json"
    if not catalog_path.is_file():
        return None
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    items = catalog.get("products", {}).get(FORECAST_PRODUCT, {}).get("items", [])
    if not items:
        return None

    now = datetime.now(timezone.utc)

    def _age_seconds(item):
        stamp = item.get("valid_time")
        if not stamp:
            return float("inf")
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return abs((parsed - now).total_seconds())

    return min(items, key=_age_seconds)


def _read_forecast_class(relative_path):
    with rasterio.open(GIS_DIR / relative_path) as src:
        band = src.read(1).astype(float)
        if src.nodata is not None:
            band = np.where(band == src.nodata, np.nan, band)
        if band.shape != (canonical_grid()["height"], canonical_grid()["width"]):
            raise ValueError(f"forecast raster shape {band.shape} does not match canonical grid")
        return band


def _read_realtime_class():
    with rasterio.open(REALTIME_TIF) as src:
        red, green, blue, alpha = src.read(1), src.read(2), src.read(3), src.read(4)
        rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing="ij")
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
        lon_mesh = np.array(xs).reshape(rows.shape)
        lat_mesh = np.array(ys).reshape(rows.shape)

    classes = np.full(red.shape, np.nan, dtype=float)
    for value, color in REALTIME_CLASS_COLORS.items():
        match = (alpha > 0) & (red == color[0]) & (green == color[1]) & (blue == color[2])
        classes[match] = value
    return classes, lon_mesh, lat_mesh


def _canonical_lonlat_mesh():
    grid = canonical_grid()
    west, _, _, north = grid["bounds"]
    res = grid["resolution"]
    xs = west + (np.arange(grid["width"]) + 0.5) * res
    ys = north - (np.arange(grid["height"]) + 0.5) * res
    xx, yy = np.meshgrid(xs, ys)
    transformer = Transformer.from_crs(CANONICAL_CRS, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return lon, lat


def main():
    if not REALTIME_TIF.is_file():
        print(f"No realtime fire danger raster yet at {REALTIME_TIF}; skipping bias map.")
        return

    forecast_item = _load_forecast_catalog_item()
    if forecast_item is None:
        print("No forecast_fire_danger catalog items published yet; skipping bias map.")
        return

    forecast_class = _read_forecast_class(forecast_item["path"])
    realtime_class_raw, realtime_lon, realtime_lat = _read_realtime_class()
    realtime_class = regrid_lonlat(realtime_class_raw, realtime_lon, realtime_lat, categorical=True)

    if realtime_class.shape != forecast_class.shape:
        raise ValueError(
            f"Regridded realtime shape {realtime_class.shape} != forecast shape {forecast_class.shape}"
        )

    bias = np.clip(forecast_class - realtime_class, -5, 5)
    bias = np.where(np.isfinite(forecast_class) & np.isfinite(realtime_class), bias, np.nan)

    lon_mesh, lat_mesh = _canonical_lonlat_mesh()

    pixelw, pixelh, mapdpi = 2048, 1152, 144
    extent = (-95.8, -89.1, 35.8, 40.8)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)

    cs = ax.contourf(
        lon_mesh, lat_mesh, bias, transform=data_crs,
        levels=BIAS_BINS, cmap=opaque_blend_cmap(BIAS_CMAP, 0.75), norm=BIAS_NORM, zorder=7, antialiased=False,
    )
    ax.contour(
        lon_mesh, lat_mesh, bias, transform=data_crs,
        levels=BIAS_BINS[1:-1], colors='black', linewidths=0.3, alpha=0.15, zorder=8,
    )

    add_boundaries(ax, data_crs, PROJECT_DIR)

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Forecast Bias (classes)')
    cbar.set_ticks(np.arange(-5, 6))
    cbar.set_ticklabels(BIAS_LABELS)

    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)

    valid_time = forecast_item.get("valid_time")
    forecast_valid_ct = (
        pd.Timestamp(valid_time).tz_convert('America/Chicago').strftime('%Y-%m-%d %H:%M CT')
        if valid_time else "unknown"
    )
    now_ct = pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')

    font_paths = [
        str(PROJECT_DIR / 'assets/Montserrat/static/Montserrat-Regular.ttf'),
        str(PROJECT_DIR / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf'),
        str(PROJECT_DIR / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf'),
    ]
    import matplotlib.font_manager as font_manager
    for font_path in font_paths:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Montserrat'

    fig.text(0.99, 0.97, "Missouri Fire Danger Forecast Bias", fontsize=26, fontweight='bold',
              ha='right', va='top', fontname='Plus Jakarta Sans')
    fig.text(
        0.99, 0.90,
        f"Forecast valid: {forecast_valid_ct}  |  Compared to realtime: {now_ct}",
        fontsize=14, ha='right', va='top', fontname='Montserrat',
    )
    fig.text(
        0.99, 0.62,
        "Bias = Forecast class − Realtime observed class\n\n"
        "Positive (red): forecast ran hotter than what's\n"
        "actually being observed right now.\n\n"
        "Negative (blue): forecast ran cooler than\n"
        "what's actually being observed right now.\n\n"
        "Classes: 0=Low, 1=Moderate, 2=Elevated,\n"
        "3=Critical, 4=Extreme\n\n"
        "For More Info, Visit ShowMeFire.org",
        fontsize=10, ha='right', va='top', linespacing=1.6, fontname='Montserrat',
    )
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight='bold', ha='left', va='bottom', fontname='Montserrat')

    svg_path = str(PROJECT_DIR / 'assets/LightBackGroundLogo.svg')
    try:
        import cairosvg  # lazy: a missing cairo native lib only drops the logo
        png_bytes = cairosvg.svg2png(url=svg_path)
        image = mpimg.imread(BytesIO(png_bytes), format='png')
        imagebox = OffsetImage(image, zoom=0.03)
        ab = AnnotationBbox(imagebox, (0.99, 0.01), frameon=False, xycoords='figure fraction', box_alignment=(1, 0))
        ax.add_artist(ab)
    except (ImportError, FileNotFoundError, OSError):
        pass

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"Forecast fire danger bias map updated at {now_ct} (forecast valid {forecast_valid_ct})")


if __name__ == "__main__":
    main()

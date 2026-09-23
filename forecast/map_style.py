"""House style shared by the public Missouri forecast maps.

Moved verbatim out of forecast/DailyForecast.py (which re-imports these
names, so `from forecast.DailyForecast import add_boundaries, ...` keeps
working for analysis/fd_analysis.py) so products that must look identical
to the public forecast - the ensemble fire danger graphics in
services/ensemble_fire_danger/render.py - can reuse the exact same code
without importing DailyForecast, which loads the stable fuel-moisture
model and opens its log file at import time.
"""
import logging
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

logger = logging.getLogger(__name__)


def create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi):
    logger.info(f"Creating base map with extent={extent}, size=({pixelw}x{pixelh}), dpi={mapdpi}")
    """Create base map figure and axes."""
    figsize_width = pixelw / mapdpi
    figsize_height = pixelh / mapdpi

    fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=mapdpi, facecolor='#E8E8E8')
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)

    return fig, ax


def add_boundaries(ax, data_crs, PROJECT_DIR, county_zorder=5, state_zorder=9):
    logger.info("Adding county and state boundaries to map.")
    """Add county and state boundaries to map."""
    counties = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp')
    if counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6",
                      facecolor='none', linewidth=1, zorder=county_zorder)

    missouriborder = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000",
                      facecolor='none', linewidth=1.5, zorder=state_zorder)


def add_title_and_branding(fig, title, subtitle, description, RUN_DATE, SCRIPT_DIR):
    logger.info(f"Adding title and branding: {title}")
    """Add title, description, and branding to figure."""
    font_paths = [
        str(SCRIPT_DIR.parent / 'assets/Montserrat/static/Montserrat-Regular.ttf'),
        str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf'),
        str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf')
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Montserrat'

    fig.text(0.99, 0.97, title, fontsize=26, fontweight='bold', ha='right', va='top', fontname='Plus Jakarta Sans')
    fig.text(0.99, 0.90, subtitle, fontsize=16, ha='right', va='top', fontname='Montserrat')
    fig.text(0.99, 0.62, description, fontsize=10, ha='right', va='top', linespacing=1.6, fontname='Montserrat')
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight='bold', ha='left', va='bottom', fontname='Montserrat')

    # Add logo
    svg_path = str(SCRIPT_DIR.parent / 'assets/LightBackGroundLogo.svg')
    try:
        import cairosvg  # lazy: a missing optional dep only drops the logo (the except below)
        png_bytes = cairosvg.svg2png(url=svg_path)
        image = mpimg.imread(BytesIO(png_bytes), format='png')
        imagebox = OffsetImage(image, zoom=0.03)
        ab = AnnotationBbox(imagebox, (0.99, 0.01), frameon=False, xycoords='figure fraction', box_alignment=(1, 0))
        plt.gca().add_artist(ab)
    except (ImportError, FileNotFoundError, OSError):  # OSError: cairo native lib absent
        pass

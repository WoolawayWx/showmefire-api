"""Render a static Missouri county burn-ban map image."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch

from core.config import IMAGES_DIR
from core.database import list_active_burn_bans

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNTIES_SHP = PROJECT_ROOT / "maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp"
STATE_SHP = PROJECT_ROOT / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp"
BURN_BAN_PNG = Path(IMAGES_DIR) / "mo-burnban.png"

MAP_EXTENT = (-95.8, -89.1, 35.8, 40.8)
PIXEL_W = 2048
PIXEL_H = 1152
MAP_DPI = 144
BACKGROUND_COLOR = "#E8E8E8"
BURN_BAN_COLOR = "#B91C1C"
INACTIVE_COLOR = "#FFFFFF"
COUNTY_EDGE_COLOR = "#B6B6B6"
STATE_EDGE_COLOR = "#000000"
CENTRAL_TZ = ZoneInfo("America/Chicago")


def burn_ban_map_public_meta() -> dict:
    updated_at = None
    if BURN_BAN_PNG.exists():
        updated_at = datetime.fromtimestamp(
            BURN_BAN_PNG.stat().st_mtime, tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "image_path": "mo-burnban.png",
        "url": "/images/mo-burnban.png",
        "updated_at": updated_at,
    }


def ensure_burn_ban_map() -> dict:
    """Generate the static map if it has never been built."""
    if not BURN_BAN_PNG.is_file():
        generate_burn_ban_map()
    return burn_ban_map_public_meta()


def _format_central_timestamp(value: datetime | None = None) -> str:
    base = value.astimezone(CENTRAL_TZ) if value else datetime.now(CENTRAL_TZ)
    return base.strftime("%Y-%m-%d %H:%M CT")


def _load_fonts() -> None:
    font_paths = [
        PROJECT_ROOT / "assets/Montserrat/static/Montserrat-Regular.ttf",
        PROJECT_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        PROJECT_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Montserrat"


def _create_base_figure():
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    fig = plt.figure(
        figsize=(PIXEL_W / MAP_DPI, PIXEL_H / MAP_DPI),
        dpi=MAP_DPI,
        facecolor=BACKGROUND_COLOR,
    )
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(MAP_EXTENT, crs=data_crs)
    ax.set_anchor("W")
    plt.subplots_adjust(left=0.05)
    return fig, ax, data_crs


def _read_counties(data_crs) -> gpd.GeoDataFrame:
    counties = gpd.read_file(COUNTIES_SHP)
    if counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    counties["fips"] = counties["COUNTYFIPS"].astype(str).str.zfill(3).radd("29")
    return counties


def _add_county_fills(ax, counties: gpd.GeoDataFrame, active_fips: set[str], data_crs) -> None:
    inactive = counties[~counties["fips"].isin(active_fips)]
    active = counties[counties["fips"].isin(active_fips)]

    if not inactive.empty:
        ax.add_geometries(
            inactive.geometry,
            crs=data_crs,
            facecolor=INACTIVE_COLOR,
            edgecolor="none",
            zorder=6,
        )
    if not active.empty:
        ax.add_geometries(
            active.geometry,
            crs=data_crs,
            facecolor=BURN_BAN_COLOR,
            edgecolor="none",
            zorder=7,
        )


def _add_boundaries(ax, counties: gpd.GeoDataFrame, data_crs) -> None:
    ax.add_geometries(
        counties.geometry,
        crs=data_crs,
        edgecolor=COUNTY_EDGE_COLOR,
        facecolor="none",
        linewidth=1,
        zorder=9,
    )

    if STATE_SHP.exists():
        state = gpd.read_file(STATE_SHP)
        if state.crs != data_crs.proj4_init:
            state = state.to_crs(data_crs.proj4_init)
        ax.add_geometries(
            state.geometry,
            crs=data_crs,
            edgecolor=STATE_EDGE_COLOR,
            facecolor="none",
            linewidth=1.5,
            zorder=10,
        )


def _add_legend(fig) -> None:
    _load_fonts()
    legend_handles = [
        Patch(
            facecolor=BURN_BAN_COLOR,
            edgecolor=STATE_EDGE_COLOR,
            linewidth=0.9,
            label="Active burn ban",
        ),
        Patch(
            facecolor=INACTIVE_COLOR,
            edgecolor=COUNTY_EDGE_COLOR,
            linewidth=0.9,
            label="No active burn ban",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.78),
        frameon=True,
        fancybox=False,
        edgecolor="#444444",
        facecolor="#FFFFFF",
        fontsize=12,
        handlelength=1.6,
        handleheight=1.2,
        borderpad=0.9,
        labelspacing=0.85,
        title="Burn Ban Status",
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(0.96)
    legend.get_title().set_fontsize(13)
    legend.get_title().set_fontweight("bold")


def _add_branding(fig, ax, active_count: int, updated_at: datetime) -> None:
    _load_fonts()

    county_label = "county" if active_count == 1 else "counties"
    fig.text(
        0.99,
        0.97,
        "Missouri County Burn Bans",
        fontsize=26,
        fontweight="bold",
        ha="right",
        va="top",
        fontname="Plus Jakarta Sans",
    )
    fig.text(
        0.99,
        0.90,
        f"{active_count} active {county_label} | Updated: {_format_central_timestamp(updated_at)}",
        fontsize=16,
        ha="right",
        va="top",
        fontname="Montserrat",
    )
    fig.text(
        0.99,
        0.48,
        "Confirmed county burn bans submitted by officials and the public,\n"
        "then reviewed by Show Me Fire staff.\n\n"
        "For more info, visit ShowMeFire.org/burn-bans",
        fontsize=10,
        ha="right",
        va="top",
        linespacing=1.6,
        fontname="Montserrat",
    )
    fig.text(
        0.02,
        0.01,
        "ShowMeFire.org",
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="bottom",
        fontname="Montserrat",
    )

    svg_path = PROJECT_ROOT / "assets/LightBackGroundLogo.svg"
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(url=str(svg_path))
        image = mpimg.imread(BytesIO(png_bytes), format="png")
        image_box = OffsetImage(image, zoom=0.03)
        logo = AnnotationBbox(
            image_box,
            (0.99, 0.01),
            frameon=False,
            xycoords="figure fraction",
            box_alignment=(1, 0),
        )
        ax.add_artist(logo)
    except Exception:
        pass


def generate_burn_ban_map() -> dict:
    active = list_active_burn_bans()
    active_fips = {item["county_fips"] for item in active}
    updated_at = datetime.now(timezone.utc)

    fig, ax, data_crs = _create_base_figure()
    counties = _read_counties(data_crs)
    _add_county_fills(ax, counties, active_fips, data_crs)
    _add_boundaries(ax, counties, data_crs)
    _add_branding(fig, ax, len(active_fips), updated_at)
    _add_legend(fig)

    BURN_BAN_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BURN_BAN_PNG, dpi=MAP_DPI, bbox_inches=None, pad_inches=0, facecolor=BACKGROUND_COLOR)
    plt.close(fig)

    updated_at_str = datetime.fromtimestamp(
        BURN_BAN_PNG.stat().st_mtime, tz=timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("Burn-ban map generated: %s counties active", len(active_fips))
    return {
        "active_counties": len(active_fips),
        "image_path": "mo-burnban.png",
        "updated_at": updated_at_str,
    }

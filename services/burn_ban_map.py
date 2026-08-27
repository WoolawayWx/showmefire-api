"""Render a static Missouri county burn-ban map image."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from core.config import IMAGES_DIR
from core.database import list_active_burn_bans

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COUNTIES_SHP = PROJECT_ROOT / "maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp"
STATE_SHP = PROJECT_ROOT / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp"
BURN_BAN_PNG = Path(IMAGES_DIR) / "mo-burnban.png"
BURN_BAN_COLOR = "#B91C1C"
INACTIVE_COLOR = "#F3F4F6"
BORDER_COLOR = "#6B7280"


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


def generate_burn_ban_map() -> dict:
    active = list_active_burn_bans()
    active_fips = {item["county_fips"] for item in active}

    counties = gpd.read_file(COUNTIES_SHP)
    counties["fips"] = counties["COUNTYFIPS"].astype(str).str.zfill(3).radd("29")
    counties["fill_color"] = counties["fips"].apply(
        lambda fips: BURN_BAN_COLOR if fips in active_fips else INACTIVE_COLOR
    )

    fig, ax = plt.subplots(figsize=(12, 7), dpi=144)
    counties.plot(ax=ax, color=counties["fill_color"], edgecolor=BORDER_COLOR, linewidth=0.6)

    if STATE_SHP.exists():
        state = gpd.read_file(STATE_SHP)
        state.boundary.plot(ax=ax, color="#111827", linewidth=1.4)

    ax.set_axis_off()
    ax.set_title("Missouri County Burn Bans", fontsize=18, fontweight="bold", pad=12)
    legend_items = [
        Patch(facecolor=BURN_BAN_COLOR, edgecolor=BORDER_COLOR, label="Active burn ban"),
        Patch(facecolor=INACTIVE_COLOR, edgecolor=BORDER_COLOR, label="No active burn ban"),
    ]
    ax.legend(handles=legend_items, loc="lower left", frameon=True)
    subtitle = f"{len(active_fips)} active {'county' if len(active_fips) == 1 else 'counties'}"
    ax.text(0.99, 0.02, subtitle, transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
    fig.tight_layout()

    BURN_BAN_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(BURN_BAN_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    updated_at = datetime.fromtimestamp(
        BURN_BAN_PNG.stat().st_mtime, tz=timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("Burn-ban map generated: %s counties active", len(active_fips))
    return {
        "active_counties": len(active_fips),
        "image_path": "mo-burnban.png",
        "updated_at": updated_at,
    }

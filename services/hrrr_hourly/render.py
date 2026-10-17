"""Per-hour map images for the hourly HRRR product, written under images/hrrr_hourly/<cycle>/.

Uses the same basemap, colorbar and branding helpers as the fire-weather maps, but
never writes to images/mo-*.png.
"""
from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from services.hrrr_hourly import BY_NAME, ELEMENTS, Element, cycle_id, images_dir

logger = logging.getLogger(__name__)

SOURCE = "Data Source: NOAA HRRR (3 km)"
CATEGORY_CLASSES = {
    "haines": [("2  Very Low", "#1a9850"), ("3  Very Low", "#91cf60"), ("4  Low", "#fee08b"),
               ("5  Moderate", "#fc8d59"), ("6  High", "#d73027")],
}


def image_name(element: str, lead: int) -> str:
    return f"{element}_f{lead:02d}.png"


def image_path(cycle: str, element: str, lead: int, root: Optional[Path] = None) -> Path:
    return Path(root or images_dir()) / cycle / image_name(element, lead)


def _classes(element: Element):
    from services.fire_weather import indices as ix

    if element.name == "ventilation_category":
        return [(name, color) for name, _low, color in ix.VENTILATION_CATEGORIES]
    return CATEGORY_CLASSES[element.name]


def _labels(cycle_time: datetime, lead: int):
    from maps import station_products as spx

    valid = (cycle_time + timedelta(hours=lead)).astimezone(spx.CENTRAL)
    title_valid = "Valid Time: " + spx.valid_time_label(valid)
    run = f"Forecast from the {cycle_time:%H}z HRRR run ({cycle_time:%b %d}, f{lead:02d})"
    return title_valid, run


def render_image(element_name: str, values: np.ndarray, lat: np.ndarray, lon: np.ndarray, cycle_time: datetime,
                 lead: int, root: Optional[Path] = None) -> Path:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.patches import Patch

    from maps import raster_maps as rm
    from maps import station_plot as sp

    element = BY_NAME[element_name]
    subtitle, run_label = _labels(cycle_time, lead)
    fig, ax, _data_crs, _map_crs, _dpi = sp.generate_basemap()
    masked = np.ma.masked_invalid(values)
    if element.categorical:
        classes = _classes(element)
        offset = 2 if element.name == "haines" else 0
        colors = ListedColormap([color for _label, color in classes])
        norm = BoundaryNorm(np.arange(-0.5, len(classes)), len(classes))
        drawn = ax.pcolormesh(lon, lat, masked - offset, transform=ccrs.PlateCarree(), shading="nearest", zorder=7,
                              cmap=colors, norm=norm)
        rm.clip_to_state(ax, drawn)
        rm.category_legend(fig, [Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=label)
                                 for label, color in classes], element.long_name.split(" (")[0], y=0.36, fontsize=12)
    else:
        colors = rm.blend_cmap(element.cmap)
        norm = Normalize(vmin=element.vmin, vmax=element.vmax, clip=False)
        drawn = ax.pcolormesh(lon, lat, masked, transform=ccrs.PlateCarree(), shading="nearest", zorder=7,
                              cmap=colors, norm=norm)
        rm.clip_to_state(ax, drawn)
        rm.left_colorbar(fig, colors, norm, f"{element.long_name} ({element.units})", ticks=element.ticks or None,
                         extend="both" if element.vmin < 0 else "max")
    sp.add_branding(fig, ax, title=f"Missouri {element.long_name.split('(')[0].strip()}", subtitle=subtitle,
                    description=f"{SOURCE}\n{run_label}\n\nSource: {element.source}")
    path = image_path(cycle_id(cycle_time), element_name, lead, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".png", dir=path.parent)
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=sp.MAP_DPI, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        Path(temporary).unlink(missing_ok=True)
    return path


def render_hour(fields: dict[str, np.ndarray], lat, lon, cycle_time: datetime, lead: int,
                elements: Optional[Iterable[str]] = None, root: Optional[Path] = None) -> list[str]:
    """Render every imaged element present in `fields`. A failing element is logged and skipped."""
    wanted = list(elements) if elements else [e.name for e in ELEMENTS if e.image]
    done = []
    for name in wanted:
        if name not in fields:
            continue
        try:
            render_image(name, fields[name], lat, lon, cycle_time, lead, root)
            done.append(name)
        except Exception:
            logger.exception("hrrr_hourly image %s f%02d failed", name, lead)
    return done


def prune_images(keep_cycles: Iterable[str], root: Optional[Path] = None) -> list[str]:
    import shutil

    root = Path(root or images_dir())
    keep = set(keep_cycles)
    removed = []
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name not in keep:
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child.name)
    return removed

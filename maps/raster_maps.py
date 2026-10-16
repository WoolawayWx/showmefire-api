"""Shared helpers for gridded (raster) maps on the standard Missouri basemap.

Imported by drought_indices.py, satellite_maps.py and fire_weather_maps.py;
never run directly (scripts/maps.sh skips it). Keeps every gridded product on
the same look: fills masked to the state, categorical legends on the right,
continuous colorbars in the open wedge west of the state like the station fills.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from cartopy.mpl.path import shapely_to_path
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import PathPatch

from maps import station_plot as sp

COLORBAR_AXES = [0.025, 0.15, 0.018, 0.5]  # same spot as station_plot.draw_filled_grid


def state_clip(ax) -> PathPatch:
    """Missouri's outline as a clip patch in map coordinates; set it as an
    artist's clip path to mask a fill or image to the state."""
    outline = sp._projected_boundary(str(sp.STATE_SHAPEFILE), ax.projection.proj4_init).union_all()
    return PathPatch(shapely_to_path(outline), transform=ax.transData, facecolor="none", edgecolor="none")


def clip_to_state(ax, artist) -> None:
    if sp.STATE_SHAPEFILE.is_file():
        artist.set_clip_path(state_clip(ax))


def blend_cmap(cmap, alpha: float = sp.FILL_ALPHA):
    """station_plot.opaque_blend_cmap for a registered name or a colormap object,
    so gridded fills get the same softened colors as the station fills."""
    if isinstance(cmap, str):
        return sp.opaque_blend_cmap(cmap, alpha)
    name = f"_smf_{cmap.name}"
    if name not in colormaps:
        colormaps.register(cmap, name=name)
    return sp.opaque_blend_cmap(name, alpha)


def category_legend(fig, handles: list, title: str, y: float = 0.38, fontsize: float = 11) -> None:
    """Category key under the description block on the right (two columns past six entries)."""
    legend = fig.legend(
        handles=handles, title=title, loc="upper right", bbox_to_anchor=(0.99, y), bbox_transform=fig.transFigure,
        fontsize=fontsize, title_fontsize=fontsize + 1, frameon=False, labelspacing=0.55, handlelength=1.6,
        handleheight=1.1, ncol=2 if len(handles) > 6 else 1, columnspacing=1.2,
    )
    legend.set_zorder(20)


def left_colorbar(fig, cmap: Colormap, norm: Normalize, label: str, ticks=None, tick_labels: Optional[list] = None,
                  extend: str = "both") -> None:
    """Vertical colorbar in the open wedge west of the state."""
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes(COLORBAR_AXES), label=label,
                            ticks=ticks, extend=extend)
    if tick_labels is not None:
        colorbar.ax.set_yticklabels(tick_labels)


def finite_or(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    return np.where(np.isfinite(values), values, fill)

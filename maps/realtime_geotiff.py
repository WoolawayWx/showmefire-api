from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.transform import from_bounds


def _base_rgba_array(grid_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create an empty RGBA stack and finite-data mask."""
    data = np.asarray(grid_values, dtype=float)
    rows, cols = data.shape
    rgba = np.zeros((4, rows, cols), dtype=np.uint8)
    valid_mask = np.isfinite(data)
    return rgba, valid_mask


def _write_rgba_geotiff(
    rgba_data: np.ndarray,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    out_path: Path,
    description: str,
    source: str,
    legend: str | None = None,
) -> bool:
    """Write RGBA data to an EPSG:4326 GeoTIFF (north-up)."""
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Input grid rows are south->north; GeoTIFF rows must be north->south.
        rgba_north_up = np.flip(rgba_data, axis=1)
        _, rows, cols = rgba_north_up.shape

        lon_min, lon_max = float(np.nanmin(lon_mesh)), float(np.nanmax(lon_mesh))
        lat_min, lat_max = float(np.nanmin(lat_mesh)), float(np.nanmax(lat_mesh))
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=4,
            dtype=rasterio.uint8,
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            photometric="RGB",
        ) as dst:
            dst.write(rgba_north_up)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")
            dst.set_band_description(4, "Alpha")
            tags = {
                "DESCRIPTION": description,
                "SOURCE": source,
                "COLOR_INTERPRETATION": "Red, Green, Blue, Alpha",
                "CREATED": datetime.now(timezone.utc).isoformat(),
            }
            if legend:
                tags["LEGEND"] = legend
            dst.update_tags(**tags)

        return True
    except Exception:
        return False


def export_continuous_rgba_geotiff(
    grid_values: np.ndarray,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    out_path: Path,
    cmap_name: str,
    vmin: float,
    vmax: float,
    description: str,
    source: str,
    legend: str | None = None,
) -> bool:
    """Color continuous values with a matplotlib colormap and export RGBA GeoTIFF."""
    rgba, valid_mask = _base_rgba_array(grid_values)
    if not np.any(valid_mask):
        return False

    data = np.asarray(grid_values, dtype=float)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)
    colorized = cmap(norm(data[valid_mask]))

    rgba[0][valid_mask] = np.round(colorized[:, 0] * 255).astype(np.uint8)
    rgba[1][valid_mask] = np.round(colorized[:, 1] * 255).astype(np.uint8)
    rgba[2][valid_mask] = np.round(colorized[:, 2] * 255).astype(np.uint8)
    rgba[3][valid_mask] = 255

    return _write_rgba_geotiff(rgba, lon_mesh, lat_mesh, out_path, description, source, legend)


def export_discrete_rgba_geotiff(
    grid_values: np.ndarray,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    out_path: Path,
    class_colors: Mapping[int, tuple[int, int, int, int]],
    description: str,
    source: str,
    legend: str | None = None,
) -> bool:
    """Color discrete class values and export RGBA GeoTIFF."""
    rgba, valid_mask = _base_rgba_array(grid_values)
    if not np.any(valid_mask):
        return False

    class_values = np.zeros(np.asarray(grid_values).shape, dtype=int)
    class_values[valid_mask] = np.rint(np.asarray(grid_values, dtype=float)[valid_mask]).astype(int)
    for value, color in class_colors.items():
        class_mask = valid_mask & (class_values == value)
        if np.any(class_mask):
            rgba[0][class_mask] = color[0]
            rgba[1][class_mask] = color[1]
            rgba[2][class_mask] = color[2]
            rgba[3][class_mask] = color[3]

    return _write_rgba_geotiff(rgba, lon_mesh, lat_mesh, out_path, description, source, legend)

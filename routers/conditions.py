"""Current-conditions map support: layer styling + point readouts.

The RTMA-derived rasters under gis/latest/realtime_*.tif are small (3 km grid),
so a point lookup is a cheap array index. Arrays are cached per file mtime, so
the many requests a hovering mouse generates never touch disk once warm.
"""
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException, Query, Response
from pyproj import Transformer

from core.config import GIS_DIR
from core.fire_events import MO_LAT_MAX, MO_LAT_MIN, MO_LON_MAX, MO_LON_MIN
from rio_tiler.colormap import cmap as rio_cmap
from routers.tiles import FIRE_DANGER_COLORS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/conditions", tags=["conditions"])


@dataclass(frozen=True)
class Product:
    key: str            # response key and layer id
    label: str
    units: str          # display units
    filename: str       # relative to GIS_DIR, usable as the tile `filename` param
    colormap: str
    rescale: tuple[float, float]
    decimals: int = 0


PRODUCTS: tuple[Product, ...] = (
    Product("fire_danger", "Fire danger", "category", "latest/realtime_fire_danger.tif", "fire_danger", (0, 4)),
    Product("temperature", "Temperature", "°F", "latest/realtime_temperature.tif", "turbo", (10, 105)),
    Product("rh", "Relative humidity", "%", "latest/realtime_rh.tif", "rdylbu", (0, 100)),
    Product("wind", "Wind speed", "kt", "latest/realtime_wind.tif", "viridis", (0, 30), 1),
    Product("fuel_moisture", "Fuel moisture", "%", "latest/realtime_fuel_moisture.tif", "rdylgn", (2, 30), 1),
    Product("precipitation", "Precipitation", "mm", "latest/realtime_precipitation.tif", "blues", (0, 25), 1),
    Product("spread_rate", "Spread rate", "ch/hr", "latest/realtime_spread_rate.tif", "magma", (0, 60), 1),
)

@dataclass(frozen=True)
class FillProduct:
    """Station-interpolated RGBA fill written by maps/*.py; already coloured, so tiles ignore colormap/rescale."""
    key: str
    label: str
    units: str
    filename: str
    mpl_cmap: str
    vmin: float
    vmax: float
    classes: Optional[tuple[tuple[str, str], ...]] = None  # (label, hex) for discrete products


FILL_PRODUCTS: tuple[FillProduct, ...] = (
    # Same station-based analysis as the static page's mo-realtimefiredanger.png.
    FillProduct("station_fire_danger", "Fire danger (station analysis)", "category", "realtime/realtime_fire_danger.tif", "", 0, 4,
                (("Low", "#90ee90"), ("Moderate", "#ffed4e"), ("Elevated", "#ffa500"), ("Critical", "#ff0000"), ("Extreme", "#8b0000"))),
    FillProduct("station_rh", "Relative humidity", "%", "realtime/relative_humidity.tif", "RdYlGn", 0, 100),
    FillProduct("station_fuel_moisture", "Fuel moisture", "%", "realtime/fuel_moisture.tif", "RdYlGn", 0, 30),
    FillProduct("station_wind", "Wind speed", "mph", "realtime/wind_speed.tif", "RdYlGn_r", 0, 51),
)


def _fill_layer(product: FillProduct) -> dict:
    import matplotlib

    path = (GIS_DIR / product.filename).resolve()
    observed_at = None
    available = path.is_file()
    if available:
        try:
            with rasterio.open(path) as src:
                observed_at = src.tags().get("CREATED")
        except Exception:
            available = False
    legend = []
    if product.classes:
        legend = [{"value": i, "color": color, "label": label} for i, (label, color) in enumerate(product.classes)]
    cmap = matplotlib.colormaps[product.mpl_cmap] if not product.classes else None
    for step in range(0 if product.classes else 6):
        fraction = step / 5
        legend.append({"value": round(product.vmin + (product.vmax - product.vmin) * fraction, 1),
                       "color": _hex([v * 255 for v in cmap(fraction)])})
    return {
        "id": product.key, "label": product.label, "units": product.units, "filename": product.filename,
        "colormap": "rgba", "rescale": [product.vmin, product.vmax], "group": "station_fill",
        "available": available, "observed_at": observed_at, "categories": [label for label, _ in product.classes] if product.classes else None, "legend": legend,
    }


FIRE_DANGER_LABELS = ("Low", "Moderate", "Elevated", "Critical", "Extreme")


@dataclass
class _Grid:
    mtime: float
    array: np.ndarray
    transform: rasterio.Affine
    crs: str
    nodata: Optional[float]
    observed_at: Optional[str]
    to_grid: Transformer


_cache: dict[str, _Grid] = {}
_lock = threading.Lock()


def _grid(product: Product) -> Optional[_Grid]:
    path = (GIS_DIR / product.filename).resolve()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    cached = _cache.get(product.key)
    if cached and cached.mtime == mtime:
        return cached
    with _lock:
        cached = _cache.get(product.key)
        if cached and cached.mtime == mtime:
            return cached
        try:
            with rasterio.open(path) as src:
                grid = _Grid(
                    mtime=mtime,
                    array=src.read(1),
                    transform=src.transform,
                    crs=str(src.crs),
                    nodata=src.nodata,
                    observed_at=src.tags().get("OBSERVATION_TIME"),
                    to_grid=Transformer.from_crs("EPSG:4326", src.crs, always_xy=True),
                )
        except Exception:
            logger.warning("conditions: unable to read %s", path, exc_info=True)
            return None
        _cache[product.key] = grid
        return grid


def _sample(grid: _Grid, lat: float, lon: float) -> Optional[float]:
    x, y = grid.to_grid.transform(lon, lat)
    col, row = ~grid.transform * (x, y)
    row, col = int(np.floor(row)), int(np.floor(col))
    if not (0 <= row < grid.array.shape[0] and 0 <= col < grid.array.shape[1]):
        return None
    value = grid.array[row, col]
    if not np.isfinite(value) or (grid.nodata is not None and value == grid.nodata):
        return None
    return float(value)


def _hex(color) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in color[:3]))


def _legend(product: Product) -> list[dict]:
    """Evenly spaced [value, color] stops matching what the tile endpoint paints."""
    if product.colormap == "fire_danger":
        return [{"value": level, "color": _hex(FIRE_DANGER_COLORS[level]), "label": FIRE_DANGER_LABELS[level]}
                for level in range(5)]
    colors = rio_cmap.get(product.colormap)
    lo, hi = product.rescale
    stops = []
    for step in range(6):
        fraction = step / 5
        stops.append({"value": round(lo + (hi - lo) * fraction, 1), "color": _hex(colors[int(round(fraction * 255))])})
    return stops


@router.get("/layers")
def conditions_layers(response: Response):
    """Style + freshness for each realtime raster; the single source of truth for tile params and legends."""
    response.headers["Cache-Control"] = "public, max-age=60"
    layers = []
    for product in PRODUCTS:
        grid = _grid(product)
        layers.append({
            "id": product.key,
            "label": product.label,
            "units": product.units,
            "filename": product.filename,
            "colormap": product.colormap,
            "rescale": list(product.rescale),
            "group": "rtma",
            "available": grid is not None,
            "observed_at": grid.observed_at if grid else None,
            "categories": list(FIRE_DANGER_LABELS) if product.key == "fire_danger" else None,
            "legend": _legend(product),
        })
    layers.extend(_fill_layer(product) for product in FILL_PRODUCTS)
    return {"success": True, "layers": layers}


@router.get("/at")
def conditions_at(
    response: Response,
    lat: float = Query(..., description="WGS84 latitude"),
    lon: float = Query(..., description="WGS84 longitude"),
):
    """Every realtime raster value at one point (null where a layer has no data there)."""
    if not (MO_LAT_MIN <= lat <= MO_LAT_MAX and MO_LON_MIN <= lon <= MO_LON_MAX):
        raise HTTPException(status_code=422, detail="Point is outside Missouri.")
    values: dict[str, Optional[float]] = {}
    observed_at: Optional[str] = None
    for product in PRODUCTS:
        grid = _grid(product)
        raw = _sample(grid, lat, lon) if grid else None
        values[product.key] = None if raw is None else round(raw, product.decimals) if product.decimals else int(round(raw))
        if grid and grid.observed_at and (observed_at is None or grid.observed_at < observed_at):
            observed_at = grid.observed_at
    danger = values.get("fire_danger")
    response.headers["Cache-Control"] = "public, max-age=60"
    return {
        "success": True,
        "lat": lat,
        "lon": lon,
        "observed_at": observed_at,
        "values": values,
        "fire_danger_label": FIRE_DANGER_LABELS[danger] if danger is not None and 0 <= danger < 5 else None,
        "units": {product.key: product.units for product in PRODUCTS},
    }

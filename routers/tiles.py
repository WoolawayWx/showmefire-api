"""
Tile Server Router
──────────────────
Generate map tiles from GeoTIFF files using rio-tiler.
Provides COG (Cloud Optimized GeoTIFF) endpoints for MapLibre GL.
"""

import asyncio
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response
import numpy as np
import rasterio
from PIL import Image
from rio_tiler.io import Reader
from rio_tiler.colormap import cmap as rio_cmap
from rio_tiler.models import ImageData
from rasterio.features import rasterize, shapes as raster_shapes
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import box, shape, mapping
from shapely.ops import unary_union

from core.config import GIS_DIR
from forecast_v1.repository import asset_for_layer, ensure_schema as ensure_forecast_v1_schema, transaction as forecast_v1_transaction
from forecast_v1.contracts import PUBLIC_LAYER_STYLES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tiles", tags=["tiles"])

FIRE_DANGER_COLORS = {
    0: (144, 238, 144, 255),
    1: (255, 237, 78, 255),
    2: (255, 165, 0, 255),
    3: (255, 0, 0, 255),
    4: (139, 0, 0, 255),
    255: (0, 0, 0, 0),
}

def _safe_gis_path(filename: str) -> Path:
    base = Path(GIS_DIR).resolve()
    path = (base / filename).resolve()
    if base != path and base not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid GeoTIFF filename")
    return path


def _bounds_list(bounds, source_crs=None) -> list[float]:
    if bounds is None:
        return [-95.8, 35.8, -89.1, 40.8]
    if hasattr(bounds, "left"):
        values = (float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top))
        if source_crs and str(source_crs).upper() not in {"EPSG:4326", "CRS84"}:
            try:
                return [float(value) for value in transform_bounds(source_crs, "EPSG:4326", *values, densify_pts=21)]
            except Exception:
                logger.warning("Unable to transform raster bounds from %s to EPSG:4326", source_crs, exc_info=True)
        return list(values)
    if isinstance(bounds, dict):
        return [
            float(bounds.get("left", bounds.get("west"))),
            float(bounds.get("bottom", bounds.get("south"))),
            float(bounds.get("right", bounds.get("east"))),
            float(bounds.get("top", bounds.get("north"))),
        ]
    return [float(value) for value in list(bounds)[:4]]


def _render_classified_png(img: ImageData) -> bytes:
    data = np.asarray(img.data)
    arr = data[0] if data.ndim == 3 else data
    mask = np.asarray(img.mask) if getattr(img, "mask", None) is not None else None
    if mask is None or mask.shape != arr.shape:
        valid = np.ones(arr.shape, dtype=bool)
    elif mask.dtype == bool:
        valid = mask
    else:
        valid = mask > 0

    rounded = np.full(arr.shape, 255, dtype=np.int16)
    finite = np.isfinite(arr)
    rounded[finite] = np.clip(np.rint(arr[finite]), 0, 255).astype(np.int16)

    rgba = np.zeros(arr.shape + (4,), dtype=np.uint8)
    for value, color in FIRE_DANGER_COLORS.items():
        rgba[valid & (rounded == value)] = color
    rgba[~valid] = (0, 0, 0, 0)

    with BytesIO() as buf:
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()


def _render_multiband_png(img: ImageData) -> bytes:
    """Fallback PNG encoder for RGB/RGBA tiles using data+mask arrays."""
    data = img.data
    if data.shape[0] < 3:
        raise ValueError("Multiband fallback requires at least 3 bands")

    rgb = np.moveaxis(data[:3], 0, -1).astype(np.uint8)
    # Preserve native alpha when available (RGBA source); otherwise use rio-tiler mask.
    if data.shape[0] >= 4:
        alpha = data[3].astype(np.uint8)
    else:
        alpha = img.mask.astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    with BytesIO() as buf:
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()


MISSOURI_BORDER_PATH = Path(__file__).resolve().parent.parent / "assets" / "missouri_border.geojson"
_MASK_SUPERSAMPLE = 4
_mo_border_cache: dict[str, object] = {}


def _missouri_border_wgs84():
    """Missouri outline in lon/lat, loaded once."""
    geom = _mo_border_cache.get("wgs84")
    if geom is None:
        import json
        with open(MISSOURI_BORDER_PATH) as fh:
            geom = shape(json.load(fh)["features"][0]["geometry"])
        _mo_border_cache["wgs84"] = geom
    return geom


def _missouri_border_3857():
    """Missouri outline in Web Mercator, loaded once."""
    geom = _mo_border_cache.get("geom")
    if geom is None:
        import json
        with open(MISSOURI_BORDER_PATH) as fh:
            collection = json.load(fh)
        wgs84 = shape(collection["features"][0]["geometry"])
        geom = shape(transform_geom("EPSG:4326", "EPSG:3857", mapping(wgs84)))
        _mo_border_cache["geom"] = geom
    return geom


def _apply_missouri_mask(png: bytes, img: ImageData) -> bytes:
    """Zero the alpha of every pixel outside Missouri (anti-aliased at the border)."""
    bounds = tuple(img.bounds)
    tile_box = box(*bounds)
    clipped = _missouri_border_3857().intersection(tile_box)
    rgba = Image.open(BytesIO(png)).convert("RGBA")
    width, height = rgba.size
    if clipped.is_empty:
        coverage = np.zeros((height, width), dtype=np.float32)
    elif clipped.equals(tile_box):
        return png
    else:
        scale = _MASK_SUPERSAMPLE
        fine = rasterize(
            [(mapping(clipped), 1)],
            out_shape=(height * scale, width * scale),
            transform=from_bounds(*bounds, width * scale, height * scale),
            fill=0,
            dtype="uint8",
        )
        coverage = fine.reshape(height, scale, width, scale).mean(axis=(1, 3)).astype(np.float32)
    arr = np.asarray(rgba).copy()
    arr[..., 3] = (arr[..., 3].astype(np.float32) * coverage).astype(np.uint8)
    with BytesIO() as buf:
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()


FIRE_DANGER_LABELS = {0: "Low", 1: "Moderate", 2: "Elevated", 3: "Critical", 4: "Extreme"}
# ~400 m: well under the raster's own cell size, so it only thins vertices.
_POLYGON_SIMPLIFY_DEGREES = 0.004
_polygon_cache: dict[tuple[str, int], dict] = {}


def _fire_danger_polygons_sync(filename: str) -> dict:
    tif_path = _safe_gis_path(filename)
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF {filename} not found")

    cache_key = (str(tif_path), tif_path.stat().st_mtime_ns)
    cached = _polygon_cache.get(cache_key)
    if cached is not None:
        return cached

    with rasterio.open(tif_path) as src:
        if src.count != 1:
            raise HTTPException(status_code=400, detail="Polygons are only available for single-band classified rasters")
        classes = src.read(1)
        valid = src.dataset_mask() > 0
        transform = src.transform
        crs = src.crs

    needs_reproject = bool(crs) and str(crs).upper() not in {"EPSG:4326", "CRS84"}

    classes = np.nan_to_num(classes, nan=255).astype(np.int16)
    valid &= (classes >= 0) & (classes <= 4)
    by_level: dict[int, list] = {}
    for geometry, value in raster_shapes(classes.astype(np.uint8), mask=valid, transform=transform, connectivity=4):
        if needs_reproject:
            geometry = transform_geom(crs, "EPSG:4326", geometry)
        by_level.setdefault(int(value), []).append(shape(geometry))

    border = _missouri_border_wgs84()
    features = []
    for level in sorted(by_level):
        clipped = unary_union(by_level[level]).intersection(border)
        if clipped.is_empty:
            continue
        clipped = clipped.simplify(_POLYGON_SIMPLIFY_DEGREES, preserve_topology=True)
        features.append({
            "type": "Feature",
            "properties": {"level": level, "label": FIRE_DANGER_LABELS.get(level, str(level))},
            "geometry": mapping(clipped),
        })

    result = {"type": "FeatureCollection", "features": features}
    _polygon_cache.clear()
    _polygon_cache[cache_key] = result
    return result


@router.get("/cog/polygons")
async def cog_polygons(filename: str = Query("peak_fire_danger.tif", description="Classified GeoTIFF filename")):
    """
    Fire-danger classes as GeoJSON polygons clipped to the Missouri border.

    One feature per danger level (`level` 0-4, `label`), for clients that draw the
    layer as native vector overlays instead of raster tiles.
    """
    return await asyncio.to_thread(_fire_danger_polygons_sync, filename)


def _transparent_tile_png(size: int = 256) -> bytes:
    """Return a transparent PNG tile for out-of-bounds requests."""
    with BytesIO() as buf:
        Image.new("RGBA", (size, size), (0, 0, 0, 0)).save(buf, format="PNG")
        return buf.getvalue()


def _cog_info_sync(filename: str) -> dict:
    tif_path = _safe_gis_path(filename)

    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF {filename} not found")

    try:
        with Reader(str(tif_path)) as src:
            info = src.info()

            return {
                # MapLibre expects [west, south, east, north] in degrees;
                # operational rasters are commonly stored in EPSG:32615.
                "bounds": _bounds_list(src.bounds, src.dataset.crs),
                "minzoom": 4,
                "maxzoom": max(int(src.maxzoom or 11), 11),
                "band_metadata": info.band_metadata,
                "band_descriptions": info.band_descriptions,
                "width": src.dataset.width,
                "height": src.dataset.height,
                "count": src.dataset.count,
                "nodata": src.dataset.nodata,
            }
    except Exception as e:
        logger.error(f"Error reading GeoTIFF info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cog/info")
async def cog_info(filename: str = "peak_fire_danger.tif"):
    """
    Get GeoTIFF metadata and bounds.

    Query params:
    - filename: Name of the GeoTIFF file (default: peak_fire_danger.tif)

    Returns: Metadata including bounds, zoom levels, band info
    """
    return await asyncio.to_thread(_cog_info_sync, filename)


def _parse_rescale(value: str):
    try:
        lo, hi = (float(part) for part in value.split(","))
    except ValueError:
        return None
    return (lo, hi) if hi > lo else None


def _cog_tile_sync(z: int, x: int, y: int, filename: str, colormap: str, rescale: str, mask: Optional[str] = None) -> Response:
    tif_path = _safe_gis_path(filename)

    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF {filename} not found")

    try:
        with Reader(str(tif_path)) as src:
            # Read tile data. If source is RGB/RGBA, render bands directly.
            band_count = src.dataset.count
            if band_count >= 4:
                img: ImageData = src.tile(x, y, z, indexes=(1, 2, 3, 4))
            elif band_count >= 3:
                img = src.tile(x, y, z, indexes=(1, 2, 3))
            else:
                img = src.tile(x, y, z)

            if band_count >= 4:
                png_data = _render_multiband_png(img)
            elif band_count >= 3:
                try:
                    png_data = img.render(img_format="PNG")
                except Exception:
                    png_data = _render_multiband_png(img)
            elif colormap == "fire_danger":
                png_data = _render_classified_png(img)
            else:
                try:
                    colormap_dict = rio_cmap.get(colormap)
                except KeyError:
                    colormap_dict = rio_cmap.get("rdylgn_r")
                # Continuous (float) rasters must be scaled to 0-255 before a
                # colormap applies; `rescale` was accepted but never used.
                lo_hi = _parse_rescale(rescale)
                if lo_hi is not None and img.array.dtype != np.uint8:
                    img.rescale(in_range=(lo_hi,))
                png_data = img.render(img_format="PNG", colormap=colormap_dict)

            if mask == "missouri":
                png_data = _apply_missouri_mask(png_data, img)

            return Response(
                content=png_data,
                media_type="image/png",
                headers={
                    "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    "Content-Type": "image/png"
                }
            )
            
    except Exception as e:
        # Out-of-bounds tiles are normal around map edges/zooms; return transparent tile.
        if "outside bounds" in str(e).lower():
            return Response(
                content=_transparent_tile_png(),
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=3600", "Content-Type": "image/png"},
            )
        logger.error(f"Error generating tile {z}/{x}/{y}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cog/tiles/{z}/{x}/{y}.png")
async def cog_tile(
    z: int,
    x: int,
    y: int,
    filename: str = Query("peak_fire_danger.tif", description="GeoTIFF filename"),
    colormap: str = Query("fire_danger", description="Colormap name"),
    rescale: str = Query("0,4", description="Min,max values for rescaling"),
    mask: Optional[str] = Query(None, description="Set to 'missouri' to make pixels outside the state transparent"),
):
    """
    Generate a map tile from GeoTIFF.

    Path params:
    - z: Zoom level
    - x: Tile X coordinate
    - y: Tile Y coordinate

    Query params:
    - filename: GeoTIFF filename (default: peak_fire_danger.tif)
    - colormap: Color ramp to apply (default: fire_danger)
    - rescale: Min,max values for data rescaling (default: 0,4)
    - mask: 'missouri' to clip the tile to the state border (default: no clipping)

    Returns: PNG tile image
    """
    return await asyncio.to_thread(_cog_tile_sync, z, x, y, filename, colormap, rescale, mask)


def _cog_preview_sync(filename: str, colormap: str, rescale: str, max_size: int) -> Response:
    tif_path = _safe_gis_path(filename)

    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF {filename} not found")

    try:
        with Reader(str(tif_path)) as src:
            band_count = src.dataset.count
            # Read overview/preview. If source is RGB/RGBA, render bands directly.
            if band_count >= 4:
                img = src.preview(max_size=max_size, indexes=(1, 2, 3, 4))
            elif band_count >= 3:
                img = src.preview(max_size=max_size, indexes=(1, 2, 3))
            else:
                img = src.preview(max_size=max_size)
            
            if band_count >= 4:
                png_data = _render_multiband_png(img)
            elif band_count >= 3:
                try:
                    png_data = img.render(img_format="PNG")
                except Exception:
                    png_data = _render_multiband_png(img)
            elif colormap == "fire_danger":
                png_data = _render_classified_png(img)
            else:
                try:
                    colormap_dict = rio_cmap.get(colormap)
                except KeyError:
                    colormap_dict = rio_cmap.get("rdylgn_r")
                png_data = img.render(img_format="PNG", colormap=colormap_dict)
            
            return Response(
                content=png_data,
                media_type="image/png",
                headers={
                    "Cache-Control": "public, max-age=3600"
                }
            )
            
    except Exception as e:
        logger.error(f"Error generating preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cog/preview.png")
async def cog_preview(
    filename: str = Query("peak_fire_danger.tif", description="GeoTIFF filename"),
    colormap: str = Query("fire_danger", description="Colormap name"),
    rescale: str = Query("0,4", description="Min,max values"),
    max_size: int = Query(512, description="Max dimension in pixels")
):
    """
    Generate a preview image of the entire GeoTIFF.

    Query params:
    - filename: GeoTIFF filename (default: peak_fire_danger.tif)
    - colormap: Color ramp to apply (default: fire_danger)
    - rescale: Min,max values (default: 0,4)
    - max_size: Maximum dimension in pixels (default: 512)

    Returns: PNG preview image
    """
    return await asyncio.to_thread(_cog_preview_sync, filename, colormap, rescale, max_size)


def _forecast_tile_sync(run_id: str, variable: str, lead_hour: int, z: int, x: int, y: int, mask: Optional[str] = None) -> Response:
    if variable not in PUBLIC_LAYER_STYLES:
        raise HTTPException(status_code=404, detail="Forecast layer not found")
    if not 0 <= lead_hour <= 72:
        raise HTTPException(status_code=422, detail="lead_hour must be between 0 and 72")
    ensure_forecast_v1_schema()
    with forecast_v1_transaction() as connection:
        asset = asset_for_layer(connection, run_id, variable, "hourly")
    if not asset or not asset["local_path"]:
        raise HTTPException(status_code=404, detail="Forecast raster not found")
    path = Path(asset["local_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Forecast raster is not available on this server")
    style = PUBLIC_LAYER_STYLES[variable]
    colormap_name, scale = str(style["colormap"]), tuple(style["rescale"])
    try:
        with Reader(str(path)) as src:
            image = src.tile(x, y, z, indexes=lead_hour + 1)
            if variable == "fire_danger":
                png = _render_classified_png(image)
            else:
                image.rescale(in_range=(scale,))
                try:
                    color_map = rio_cmap.get(colormap_name)
                except KeyError:
                    color_map = rio_cmap.get("viridis")
                png = image.render(img_format="PNG", colormap=color_map)
            if mask == "missouri":
                png = _apply_missouri_mask(png, image)
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})
    except HTTPException:
        raise
    except Exception as exc:
        if "outside bounds" in str(exc).lower():
            return Response(content=_transparent_tile_png(), media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})
        logger.exception("Unable to render forecast-v1 tile")
        raise HTTPException(status_code=500, detail="Unable to render forecast tile") from exc


@router.get("/forecast/{run_id}/{variable}/{lead_hour}/{z}/{x}/{y}.png")
async def forecast_tile(
    run_id: str,
    variable: str,
    lead_hour: int,
    z: int,
    x: int,
    y: int,
    mask: Optional[str] = Query(None, description="Set to 'missouri' to clip to the state border"),
):
    """Render one allow-listed band from an immutable forecast-v1 COG."""
    return await asyncio.to_thread(_forecast_tile_sync, run_id, variable, lead_hour, z, x, y, mask)

"""
Tile Server Router
──────────────────
Generate map tiles from GeoTIFF files using rio-tiler.
Provides COG (Cloud Optimized GeoTIFF) endpoints for MapLibre GL.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response
from rio_tiler.io import Reader
from rio_tiler.colormap import cmap as rio_cmap
from rio_tiler.models import ImageData

from core.config import GIS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tiles", tags=["tiles"])


@router.get("/cog/info")
async def cog_info(filename: str = "peak_fire_danger.tif"):
    """
    Get GeoTIFF metadata and bounds.
    
    Query params:
    - filename: Name of the GeoTIFF file (default: peak_fire_danger.tif)
    
    Returns: Metadata including bounds, zoom levels, band info
    """
    tif_path = Path(GIS_DIR) / filename
    
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF {filename} not found")
    
    try:
        with Reader(str(tif_path)) as src:
            info = src.info()
            
            return {
                "bounds": src.bounds,
                "minzoom": src.minzoom,
                "maxzoom": src.maxzoom,
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


@router.get("/cog/tiles/{z}/{x}/{y}.png")
async def cog_tile(
    z: int,
    x: int,
    y: int,
    filename: str = Query("peak_fire_danger.tif", description="GeoTIFF filename"),
    colormap: str = Query("fire_danger", description="Colormap name"),
    rescale: str = Query("0,4", description="Min,max values for rescaling")
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
    
    Returns: PNG tile image
    """
    tif_path = Path(GIS_DIR) / filename
    
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF {filename} not found")
    
    try:
        with Reader(str(tif_path)) as src:
            # Read tile data
            img: ImageData = src.tile(x, y, z)
            
            # Parse rescale values
            try:
                vmin, vmax = map(float, rescale.split(","))
            except ValueError:
                vmin, vmax = 0, 4
            
            # Select colormap
            if colormap == "fire_danger":
                # Custom fire danger colormap
                colormap_dict = {
                    0: (144, 238, 144, 255),  # Low - Light green
                    1: (255, 237, 78, 255),   # Moderate - Yellow
                    2: (255, 165, 0, 255),    # Elevated - Orange
                    3: (255, 0, 0, 255),      # Critical - Red
                    4: (139, 0, 0, 255),      # Extreme - Dark red
                    255: (0, 0, 0, 0),        # NoData - Transparent
                }
            else:
                # Try to get from rio-tiler built-in colormaps
                try:
                    colormap_dict = rio_cmap.get(colormap)
                except KeyError:
                    # Fallback to rdylgn_r (red-yellow-green reversed)
                    colormap_dict = rio_cmap.get("rdylgn_r")
            
            # Render to PNG
            png_data = img.render(
                img_format="PNG",
                colormap=colormap_dict,
            )
            
            return Response(
                content=png_data,
                media_type="image/png",
                headers={
                    "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    "Content-Type": "image/png"
                }
            )
            
    except Exception as e:
        logger.error(f"Error generating tile {z}/{x}/{y}: {e}")
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
    tif_path = Path(GIS_DIR) / filename
    
    if not tif_path.exists():
        raise HTTPException(status_code=404, detail=f"GeoTIFF {filename} not found")
    
    try:
        with Reader(str(tif_path)) as src:
            # Read overview/preview
            img = src.preview(max_size=max_size)
            
            # Select colormap (same logic as tiles)
            if colormap == "fire_danger":
                colormap_dict = {
                    0: (144, 238, 144, 255),
                    1: (255, 237, 78, 255),
                    2: (255, 165, 0, 255),
                    3: (255, 0, 0, 255),
                    4: (139, 0, 0, 255),
                    255: (0, 0, 0, 0),
                }
            else:
                try:
                    colormap_dict = rio_cmap.get(colormap)
                except KeyError:
                    colormap_dict = rio_cmap.get("rdylgn_r")
            
            # Render to PNG
            png_data = img.render(
                img_format="PNG",
                colormap=colormap_dict,
            )
            
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

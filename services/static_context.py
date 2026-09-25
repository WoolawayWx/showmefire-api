"""Point-samples static land-cover/fuel-context rasters for detection
ingestion (services/fire_ingest.py), so FIRMS (MODIS/VIIRS) detections get
comparable land-cover context to what NGFS detections already carry via
their raw `land_cover` percentage string.

Reads the same statewide source rasters model-training/static_features/
downloads for the fire-behavior spread-rate bundles (data/static/source/),
by direct point sample rather than the 256x256-per-fire bundle pipeline
those use - that pipeline is built for spread modeling over a small area
around one fire, not a cheap single-point lookup anywhere in the state.

Degrades to (None, None) if the source rasters aren't present on this
deployment - this is enrichment, not a hard ingest dependency.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_DIAGNOSTICS = {"available": False, "fallback_reason": "not initialized"}


def diagnostics() -> dict:
    return dict(_DIAGNOSTICS)


def _candidate_dirs():
    override = os.getenv("SMF_STATIC_SOURCE_DIR")
    if override:
        yield Path(override)
    # api/services/static_context.py -> parents[2] is the repo root, which
    # holds data/static/source as a sibling of api/ (not under api/data).
    here = Path(__file__).resolve()
    yield here.parents[2] / "data" / "static" / "source"
    yield Path("/app/data/static/source")


@lru_cache(maxsize=1)
def _source_dir() -> Optional[Path]:
    for candidate in _candidate_dirs():
        if candidate.is_dir():
            return candidate
    return None


@lru_cache(maxsize=4)
def _open_raster(path_str: str):
    import rasterio

    return rasterio.open(path_str)


def _sample(path: Path, latitude: float, longitude: float) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        dataset = _open_raster(str(path))
        row, col = dataset.index(longitude, latitude)
        if row < 0 or col < 0 or row >= dataset.height or col >= dataset.width:
            return None
        value = dataset.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
        if dataset.nodata is not None and value == dataset.nodata:
            return None
        return float(value)
    except Exception:
        logger.exception("static_context: failed to sample %s at (%s, %s)", path, latitude, longitude)
        return None


def sample_static_context(latitude: float, longitude: float) -> Tuple[Optional[int], Optional[float]]:
    """Returns (fuel_model_fbfm40, canopy_cover_pct) at a point, or (None,
    None) if the static source rasters aren't available on this deployment."""
    source_dir = _source_dir()
    if source_dir is None:
        _DIAGNOSTICS.update(available=False, fallback_reason="static source rasters not found")
        return None, None
    _DIAGNOSTICS.update(available=True, fallback_reason=None)

    fuel_model = _sample(source_dir / "fbfm40.tif", latitude, longitude)
    canopy_cover = _sample(source_dir / "canopy_cover.tif", latitude, longitude)
    return (
        int(fuel_model) if fuel_model is not None else None,
        canopy_cover,
    )

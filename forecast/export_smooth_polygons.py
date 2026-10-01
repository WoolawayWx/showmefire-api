"""
export_smooth_polygons.py
─────────────────────────
Additional, high-resolution vector product: peak fire danger as smooth polygons.

The production ``peak_fire_danger_polygons.geojson`` is built by dissolving the
square 3 km model cells, so its boundaries are stair-stepped.  This product
instead contours the *continuous* smoothed risk field (the same field the
forecast graphic draws) on a fine grid, then clips to the true Missouri border:

  1. interpolate the continuous field onto a regular EPSG:32615 grid
     (default 500 m) and fill the thin strip the model grid misses at the border,
  2. trace filled contours at the class thresholds (marching squares),
  3. clip to the Missouri state boundary and lightly simplify.

Because the bands come from one contour pass they tile the state exactly
(no gaps, no overlaps).  Detail is still limited by the 3 km source data; the
smoothness is cosmetic, not extra information.

Output is a separate file (``peak_fire_danger_smooth_polygons[suffix].geojson``);
nothing existing is read back or modified.

Disable with SMF_SMOOTH_POLYGONS=0.  Resolution: SMF_SMOOTH_POLYGON_RES (metres).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from shapely.geometry import mapping
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

UTM = "EPSG:32615"
BINS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
DEFAULT_RESOLUTION_M = 500
PRECISION = 5  # decimal degrees in the output (~1 m)


def enabled() -> bool:
    return os.getenv("SMF_SMOOTH_POLYGONS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _helpers():
    """The shared constants/helpers live in the production exporter; import lazily
    (it is imported both as ``forecast.export_fire_danger_gis`` and as a bare
    module depending on how the forecast scripts are launched)."""
    try:
        from forecast import export_fire_danger_gis as base
    except ImportError:  # pragma: no cover - script-style import
        import export_fire_danger_gis as base
    return base


def _fine_field(values: np.ndarray, lon: np.ndarray, lat: np.ndarray, resolution: float):
    """Continuous field on a regular UTM grid, with nearest-value fill at the edge."""
    lon = np.where(np.asarray(lon) > 180, np.asarray(lon) - 360, np.asarray(lon))
    x, y = Transformer.from_crs("EPSG:4326", UTM, always_xy=True).transform(lon, np.asarray(lat))
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4:
        raise ValueError("not enough valid cells to contour")
    points = np.column_stack((x[valid], y[valid]))

    pad = resolution * 4
    xs = np.arange(np.floor((points[:, 0].min() - pad) / resolution) * resolution,
                   points[:, 0].max() + pad, resolution)
    ys = np.arange(np.floor((points[:, 1].min() - pad) / resolution) * resolution,
                   points[:, 1].max() + pad, resolution)
    xx, yy = np.meshgrid(xs, ys)
    targets = np.column_stack((xx.ravel(), yy.ravel()))

    field = griddata(points, values[valid], (xx, yy), method="linear")

    # Cells whose centres sit just outside the masked model footprint (the border
    # strip) have no triangle to interpolate from: take the nearest value, but only
    # within ~2 source cells so nothing is invented far from real data.
    tree = cKDTree(points)
    spacing = float(np.median(tree.query(points, k=2)[0][:, 1]))
    distance, index = tree.query(targets)
    nearest = values[valid][index].reshape(xx.shape)
    near = (distance <= 2.0 * spacing).reshape(xx.shape)
    field = np.where(np.isfinite(field), field, np.where(near, nearest, np.nan))

    # Remove the faint facets left by linear interpolation of a coarser grid.
    sigma = max(1.0, 0.5 * spacing / resolution)
    filled = np.where(np.isfinite(field), field, 0.0)
    weight = gaussian_filter(np.isfinite(field).astype(float), sigma)
    smooth = gaussian_filter(filled, sigma) / np.maximum(weight, 1e-9)
    field = np.where(np.isfinite(field), smooth, np.nan)
    return xs, ys, field


def build_smooth_danger_regions(peak_risk_smooth: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                                run_date=None, resolution_m: float | None = None) -> list[dict]:
    """One dict per non-empty danger level with a WGS84 shapely geometry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = _helpers()
    resolution = float(resolution_m or os.getenv("SMF_SMOOTH_POLYGON_RES", DEFAULT_RESOLUTION_M))
    xs, ys, field = _fine_field(peak_risk_smooth, lon, lat, resolution)
    # Keep the extremes inside the first/last band instead of dropping them.
    field = np.clip(field, BINS[0] + 1e-6, BINS[-1] - 1e-6)

    fig, ax = plt.subplots()
    try:
        cs = ax.contourf(xs, ys, np.ma.masked_invalid(field), levels=BINS)
        paths = list(cs.get_paths())
    finally:
        plt.close(fig)

    state = gpd.read_file(base.STATE_BOUNDARY_SHP).to_crs(UTM)
    state_union = unary_union(state.geometry)
    to_wgs84 = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    run_str = run_date.strftime("%Y-%m-%dT%H:%M:%SZ") if run_date else None

    regions = []
    for level, path in enumerate(paths):
        geom = base._polygon_from_contourf_path(path)
        if geom is None:
            continue
        geom = geom.intersection(state_union)
        if geom.is_empty:
            continue
        area_km2 = geom.area / 1e6
        geom = geom.simplify(resolution / 10.0, preserve_topology=True)
        from shapely.ops import transform
        geom = transform(lambda x, y, z=None: to_wgs84.transform(x, y), geom)
        meta = base.DANGER_LEVELS[level]
        regions.append({
            "danger_level": level,
            "label": meta["label"],
            "color": meta["color"],
            "model_run": run_str,
            "resolution_m": int(resolution),
            "area_km2": round(area_km2, 1),
            "geometry": geom,
        })
    return regions


def _round_coords(obj):
    if isinstance(obj, (list, tuple)):
        return [_round_coords(item) for item in obj]
    if isinstance(obj, float):
        return round(obj, PRECISION)
    return obj


def export_smooth_geojson_polygons(peak_risk_smooth: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                                   out_path: Path, run_date=None) -> bool:
    """Write the smooth polygons to ``out_path`` atomically.  Never raises."""
    temporary = None
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        regions = build_smooth_danger_regions(peak_risk_smooth, lon, lat, run_date)
        if not regions:
            logger.warning("Smooth polygons: no danger-level regions produced")
            return False
        features = []
        for region in regions:
            geometry = mapping(region.pop("geometry"))
            geometry["coordinates"] = _round_coords(geometry["coordinates"])
            features.append({"type": "Feature", "geometry": geometry, "properties": region})
        payload = {
            "type": "FeatureCollection",
            "name": "Missouri Peak Fire Danger (smooth polygons)",
            "metadata": {
                "model_run": regions[0]["model_run"],
                "created": datetime.now(timezone.utc).isoformat(),
                "product": "smooth_polygons",
                "note": "Contours of the continuous risk field on a fine grid, clipped to Missouri.",
            },
            "features": features,
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{out_path.name}.", dir=out_path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        # mkstemp() creates 0600; the qgis-server container reads this tree as another user.
        os.chmod(temporary, 0o644)
        os.replace(temporary, out_path)
        temporary = None
        logger.info("Smooth polygons saved → %s (%.0f KB, %d features)",
                    out_path, out_path.stat().st_size / 1024, len(features))
        return True
    except Exception:
        logger.error("Smooth polygon export failed (non-fatal)", exc_info=True)
        return False
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)

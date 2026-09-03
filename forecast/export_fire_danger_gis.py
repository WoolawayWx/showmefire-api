"""
export_fire_danger_gis.py
─────────────────────────
Drop-in replacement for the GIS export block in forecastedfiredanger.py.

Exports peak fire danger as:
  1. GeoTIFF  – single-band uint8, EPSG:32615 on the canonical Missouri grid
  2. GeoJSON  – polygon contour regions (best for MapLibre fill layers)
  3. GeoJSON  – point grid  (best for QGIS spot checks / agency sharing)

Usage inside generate_complete_forecast():
    from export_fire_danger_gis import export_all_gis_formats
    export_all_gis_formats(
        peak_risk_smooth, lon, lat,
        run_date=RUN_DATE,
        out_dir=PROJECT_DIR / 'gis'
    )
"""

import json
import logging
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import rasterio
from rasterio.crs import CRS

from scipy.ndimage import label as nd_label
from shapely.geometry import mapping, shape, MultiPolygon, Polygon
from shapely.ops import unary_union
import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────

DANGER_LEVELS = {
    0: {"label": "Low",      "color": "#90EE90"},
    1: {"label": "Moderate", "color": "#FFED4E"},
    2: {"label": "Elevated", "color": "#FFA500"},
    3: {"label": "Critical", "color": "#FF0000"},
    4: {"label": "Extreme",  "color": "#8B0000"},
}

NODATA_UINT8 = 255   # sentinel for NaN / outside-Missouri cells

CHICAGO_TZ = ZoneInfo("America/Chicago")


def forecast_peak_local_date(run_date=None) -> str:
    """Local (America/Chicago) calendar date this forecast run belongs to.

    Used to key the forecast-peak archive so it lines up with the same
    local date used by observed_peak_history.py and endOfDayReport.py.
    """
    if run_date is None:
        run_time_utc = datetime.now(timezone.utc)
    elif run_date.tzinfo is None:
        run_time_utc = run_date.replace(tzinfo=timezone.utc)
    else:
        run_time_utc = run_date.astimezone(timezone.utc)
    return run_time_utc.astimezone(CHICAGO_TZ).strftime("%Y-%m-%d")


def archive_forecast_peak_tif(tif_path: Path, out_dir: Path, run_date=None) -> Path | None:
    """Copy the just-written peak_fire_danger.tif into a per-date archive.

    Mirrors gis/observed_peak/archive/{date}.tif so verification can compare
    a given date's forecast peak against that same date's observed peak,
    instead of always reading whatever the current live tif happens to be.
    Last forecast run of the local day wins (overwrites earlier runs).
    """
    try:
        date_local = forecast_peak_local_date(run_date)
        archive_dir = Path(out_dir) / "forecast_peak" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / f"{date_local}.tif"
        shutil.copy2(tif_path, archive_path)
        return archive_path
    except Exception as e:
        logger.error(f"Failed to archive forecast peak GeoTIFF: {e}", exc_info=True)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GeoTIFF  (single-band uint8, canonical EPSG:32615 grid)
# ══════════════════════════════════════════════════════════════════════════════

def export_geotiff(peak_risk_smooth: np.ndarray,
                   lon: np.ndarray,
                   lat: np.ndarray,
                   out_path: Path,
                   run_date=None) -> bool:
    """
    Write a single-band uint8 GeoTIFF in EPSG:32615.

    Why single-band instead of RGBA?
      • RGBA GeoTIFFs embed colours that clash with QGIS/MapLibre symbology.
      • A single uint8 band (0-4 = danger level, 255 = nodata) is universally
        understood: apply any colour ramp you like in the viewer.

    Why regrid to the canonical projected grid?
      • HRRR data comes on Lambert Conformal grid with 2D coordinate arrays
      • from_bounds assumes regular spacing in lat/lon, which causes distortion
      • direct coordinate-aware interpolation preserves geographic placement
    """
    try:
        from services.gis_publisher import canonical_grid, regrid_lonlat
        
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Bin continuous smooth values → 0-4 danger categories ─────────────
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        risk_binned = np.digitize(peak_risk_smooth, bins, right=False) - 1
        risk_binned = np.clip(risk_binned, 0, 4).astype(np.uint8)

        # Replace NaN locations with nodata sentinel
        nan_mask = np.isnan(peak_risk_smooth)
        risk_binned[nan_mask] = NODATA_UINT8

        source = risk_binned.astype(float)
        source[risk_binned == NODATA_UINT8] = np.nan
        projected = regrid_lonlat(source, lon, lat, categorical=True)
        regridded = np.where(np.isfinite(projected), projected, NODATA_UINT8).astype(np.uint8)
        grid = canonical_grid()
        rows, cols = regridded.shape
        transform = grid['transform']

        run_str = run_date.strftime('%Y-%m-%d %HZ') if run_date else 'unknown'

        with rasterio.open(
            out_path, 'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=rasterio.uint8,
            crs=CRS.from_epsg(32615),
            transform=transform,
            nodata=NODATA_UINT8,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(regridded, 1)
            dst.update_tags(
                BAND_1='Peak fire danger level: 0=Low 1=Moderate 2=Elevated 3=Critical 4=Extreme 255=NoData',
                MODEL_RUN=run_str,
                SOURCE='HRRR + ShowMeFire ML model + RAWS observations',
                CREATED=datetime.now(timezone.utc).isoformat(),
            )
            # Human-readable band description
            dst.set_band_description(1, 'Peak Fire Danger (0=Low … 4=Extreme)')

        logger.info(f"GeoTIFF saved → {out_path}")
        return True

    except Exception as e:
        logger.error(f"GeoTIFF export failed: {e}", exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GeoJSON – polygon contour regions
# ══════════════════════════════════════════════════════════════════════════════

def export_geojson_polygons(peak_risk_smooth: np.ndarray,
                            lon: np.ndarray,
                            lat: np.ndarray,
                            out_path: Path,
                            run_date=None) -> bool:
    """
    Convert the raster fire-danger grid to filled polygon regions.

    Each polygon feature represents a contiguous area sharing the same
    danger level.  This is the best format for:
      • MapLibre GL fill layers   (use 'danger_level' property for paint rules)
      • Sharing with agencies     (readable, self-describing, no special tools)
      • QGIS vector editing

    Strategy
    ────────
    1. Bin the smooth raster to uint8 danger levels (same as GeoTIFF).
    2. Create cell polygons using actual 2D coordinate grids (handles projection warping).
    3. Dissolve shapes per level with shapely unary_union.
    4. Write as a FeatureCollection with metadata properties.
    
    Note: HRRR data uses Lambert Conformal projection with 2D coordinate meshes.
    We must use the actual cell coordinates, not assume a regular lat/lon grid.
    """
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows, cols = peak_risk_smooth.shape

        # ── Bin values ────────────────────────────────────────────────────────
        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        risk_binned = np.digitize(peak_risk_smooth, bins, right=False) - 1
        risk_binned = np.clip(risk_binned, 0, 4).astype(np.uint8)
        nan_mask = np.isnan(peak_risk_smooth)
        risk_binned[nan_mask] = NODATA_UINT8

        # ── Create polygons using actual 2D coordinates ───────────────────────
        # For each grid cell, create a polygon from its corner coordinates
        features = []
        
        # Group cells by danger level
        level_cells = {level: [] for level in DANGER_LEVELS.keys()}
        
        for i in range(rows - 1):
            for j in range(cols - 1):
                level = risk_binned[i, j]
                if level == NODATA_UINT8:
                    continue
                
                # Get the 4 corners of this cell (i,j), (i,j+1), (i+1,j+1), (i+1,j)
                corners = [
                    (float(lon[i, j]), float(lat[i, j])),
                    (float(lon[i, j+1]), float(lat[i, j+1])),
                    (float(lon[i+1, j+1]), float(lat[i+1, j+1])),
                    (float(lon[i+1, j]), float(lat[i+1, j])),
                    (float(lon[i, j]), float(lat[i, j]))  # close the ring
                ]
                
                try:
                    poly = Polygon(corners)
                    if poly.is_valid and not poly.is_empty:
                        level_cells[level].append(poly)
                except Exception:
                    continue
        
        # Dissolve polygons per danger level
        for level, meta in DANGER_LEVELS.items():
            polys = level_cells.get(level, [])
            if not polys:
                continue

            merged = unary_union(polys)
            # Simplify to reduce file size while preserving topology
            merged = merged.simplify(0.001, preserve_topology=True)

            features.append({
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "danger_level": level,
                    "label":        meta["label"],
                    "color":        meta["color"],
                    "model_run":    run_date.strftime('%Y-%m-%dT%H:%M:%SZ') if run_date else None,
                }
            })

        run_str = run_date.strftime('%Y-%m-%d %HZ') if run_date else 'unknown'
        geojson = {
            "type": "FeatureCollection",
            "name": "Missouri Peak Fire Danger",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "metadata": {
                "model_run": run_str,
                "created":   datetime.now(timezone.utc).isoformat(),
                "source":    "HRRR + ShowMeFire ML + RAWS",
                "legend": {str(k): v for k, v in DANGER_LEVELS.items()},
            },
            "features": features
        }

        with open(out_path, 'w') as f:
            json.dump(geojson, f, separators=(',', ':'))   # compact – no whitespace

        size_kb = out_path.stat().st_size / 1024
        logger.info(f"GeoJSON polygons saved → {out_path}  ({size_kb:.0f} KB, {len(features)} features)")
        return True

    except Exception as e:
        logger.error(f"GeoJSON polygon export failed: {e}", exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GeoJSON – point grid  (one point per grid cell)
# ══════════════════════════════════════════════════════════════════════════════

def export_geojson_points(peak_risk_smooth: np.ndarray,
                          lon: np.ndarray,
                          lat: np.ndarray,
                          out_path: Path,
                          run_date=None,
                          stride: int = 1) -> bool:
    """
    Export the grid as a GeoJSON point FeatureCollection.

    Parameters
    ──────────
    stride : int
        Sample every N-th grid cell in both directions.
        stride=1  → every cell (default, full HRRR resolution ~3km)
        stride=3  → every 3rd cell (~9 km spacing)
        stride=5  → every 5th cell (smallest file)

    Each point carries the raw smoothed risk value AND the binned level so
    downstream tools can apply their own thresholds if needed.

    Best for
    ────────
    • QGIS spot checks / manual QA
    • Sharing tabular data with agencies who prefer spreadsheets
      (QGIS can export this to CSV trivially)
    • Debugging – easy to inspect individual cell values
    """
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        risk_binned = np.digitize(peak_risk_smooth, bins, right=False) - 1
        risk_binned = np.clip(risk_binned, 0, 4)

        run_str = run_date.strftime('%Y-%m-%dT%H:%M:%SZ') if run_date else None

        features = []
        rows_idx = range(0, peak_risk_smooth.shape[0], stride)
        cols_idx = range(0, peak_risk_smooth.shape[1], stride)

        for ii in rows_idx:
            for jj in cols_idx:
                raw = peak_risk_smooth[ii, jj]
                if np.isnan(raw):
                    continue   # skip outside-Missouri cells

                level = int(risk_binned[ii, jj])
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [round(float(lon[ii, jj]), 5),
                                        round(float(lat[ii, jj]), 5)]
                    },
                    "properties": {
                        "danger_level": level,
                        "label":        DANGER_LEVELS[level]["label"],
                        "color":        DANGER_LEVELS[level]["color"],
                        "risk_smooth":  round(float(raw), 3),
                        "model_run":    run_str,
                    }
                })

        geojson = {
            "type": "FeatureCollection",
            "name": "Missouri Peak Fire Danger (point grid)",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "metadata": {
                "model_run":   run_str,
                "created":     datetime.now(timezone.utc).isoformat(),
                "stride":      stride,
                "source":      "HRRR + ShowMeFire ML + RAWS",
                "legend":      {str(k): v for k, v in DANGER_LEVELS.items()},
            },
            "features": features
        }

        with open(out_path, 'w') as f:
            json.dump(geojson, f, separators=(',', ':'))

        size_kb = out_path.stat().st_size / 1024
        logger.info(f"GeoJSON points saved → {out_path}  ({size_kb:.0f} KB, {len(features)} points)")
        return True

    except Exception as e:
        logger.error(f"GeoJSON point export failed: {e}", exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Convenience wrapper – call this from generate_complete_forecast()
# ══════════════════════════════════════════════════════════════════════════════

def export_all_gis_formats(peak_risk_smooth: np.ndarray,
                           lon: np.ndarray,
                           lat: np.ndarray,
                           run_date=None,
                           out_dir: Path = Path('gis')) -> dict:
    """
    Export peak fire danger in all three GIS formats.

    Returns a dict of {format: path_or_None} so callers can log/upload selectively.

    Typical usage in generate_complete_forecast()
    ─────────────────────────────────────────────
        from export_fire_danger_gis import export_all_gis_formats

        # Replace the existing GeoTIFF block with:
        gis_files = export_all_gis_formats(
            peak_risk_smooth, lon, lat,
            run_date=RUN_DATE,
            out_dir=PROJECT_DIR / 'gis'
        )
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    date_tag = run_date.strftime('%Y%m%d_%H') if run_date else 'unknown'

    results = {}

    # ── GeoTIFF ───────────────────────────────────────────────────────────────
    tif_path = out_dir / 'peak_fire_danger.tif'
    ok = export_geotiff(peak_risk_smooth, lon, lat, tif_path, run_date)
    results['geotiff'] = tif_path if ok else None
    if ok:
        results['geotiff_archive'] = archive_forecast_peak_tif(tif_path, out_dir, run_date)

    # ── GeoJSON polygons ──────────────────────────────────────────────────────
    poly_path = out_dir / 'peak_fire_danger_polygons.geojson'
    ok = export_geojson_polygons(peak_risk_smooth, lon, lat, poly_path, run_date)
    results['geojson_polygons'] = poly_path if ok else None

    # ── GeoJSON points ────────────────────────────────────────────────────────
    pts_path = out_dir / 'peak_fire_danger_points.geojson'
    ok = export_geojson_points(peak_risk_smooth, lon, lat, pts_path, run_date, stride=1)
    results['geojson_points'] = pts_path if ok else None

    # ── Summary ───────────────────────────────────────────────────────────────
    for fmt, path in results.items():
        if path:
            size_kb = Path(path).stat().st_size / 1024
            logger.info(f"  ✓ {fmt:22s} → {path.name}  ({size_kb:.0f} KB)")
        else:
            logger.warning(f"  ✗ {fmt:22s} → FAILED")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MapLibre GL usage notes (written as comments for agency/dev reference)
# ══════════════════════════════════════════════════════════════════════════════
#
# GeoTIFF via raster source:
# ──────────────────────────
# Serve peak_fire_danger.tif via a tile server (e.g. titiler, rio-tiler) or
# convert to PMTiles / MBTiles with gdal2tiles / tippecanoe.
# In MapLibre:
#   map.addSource('fire-danger', { type: 'raster', url: '...' });
#   map.addLayer({ id: 'fire', type: 'raster', source: 'fire-danger' });
#
# GeoJSON polygons (recommended for web):
# ────────────────────────────────────────
# map.addSource('fire-danger', { type: 'geojson', data: '/gis/peak_fire_danger_polygons.geojson' });
# map.addLayer({
#   id: 'fire-fill', type: 'fill', source: 'fire-danger',
#   paint: {
#     'fill-color': ['match', ['get', 'danger_level'],
#       0, '#90EE90',   // Low
#       1, '#FFED4E',   // Moderate
#       2, '#FFA500',   // Elevated
#       3, '#FF0000',   // Critical
#       4, '#8B0000',   // Extreme
#       '#cccccc'       // fallback
#     ],
#     'fill-opacity': 0.7
#   }
# });

"""
export_fire_danger_gis.py
─────────────────────────
Drop-in replacement for the GIS export block in forecastedfiredanger.py.

Exports peak fire danger as:
  1. GeoTIFF   – single-band uint8, EPSG:32615 on the canonical Missouri grid
  2. GeoJSON   – polygon contour regions (best for MapLibre fill layers)
  3. Shapefile – zipped .shp/.shx/.dbf/.prj bundle + QGIS/SLD styles, for
                 desktop GIS tools and other applications.

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
import re
import shutil
import tempfile
import zipfile
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_BOUNDARY_SHP = PROJECT_ROOT / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp"
POLYGON_BUFFER_METERS = 250.0


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


def archive_forecast_peak_tif(
    tif_path: Path,
    out_dir: Path,
    run_date=None,
    archive_name: str = "forecast_peak",
) -> Path | None:
    """Copy the just-written peak_fire_danger.tif into a per-date archive.

    Mirrors gis/observed_peak/archive/{date}.tif so verification can compare
    a given date's forecast peak against that same date's observed peak,
    instead of always reading whatever the current live tif happens to be.
    Last forecast run of the local day wins (overwrites earlier runs).
    """
    try:
        date_local = forecast_peak_local_date(run_date)
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", archive_name):
            raise ValueError(f"Invalid forecast peak archive name: {archive_name}")
        archive_dir = Path(out_dir) / archive_name / "archive"
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

def _build_danger_level_regions(peak_risk_smooth: np.ndarray,
                                lon: np.ndarray,
                                lat: np.ndarray,
                                run_date=None) -> list[dict]:
    """
    Shared dissolve/buffer/clip/simplify pipeline used by every vector
    export (GeoJSON polygons, Shapefile). Returns one dict per non-empty
    danger level with a WGS84 shapely geometry plus the properties every
    exporter needs.

    Strategy
    ────────
    1. Bin the smooth raster to uint8 danger levels (same as GeoTIFF).
    2. Create cell polygons using actual 2D coordinate grids (handles projection warping).
    3. Dissolve shapes per level with shapely unary_union.
    4. Buffer 250m and clip to the Missouri state boundary (in a projected CRS).

    Note: HRRR data uses Lambert Conformal projection with 2D coordinate meshes.
    We must use the actual cell coordinates, not assume a regular lat/lon grid.
    """
    rows, cols = peak_risk_smooth.shape

    # ── Bin values ────────────────────────────────────────────────────────
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    risk_binned = np.digitize(peak_risk_smooth, bins, right=False) - 1
    risk_binned = np.clip(risk_binned, 0, 4).astype(np.uint8)
    nan_mask = np.isnan(peak_risk_smooth)
    risk_binned[nan_mask] = NODATA_UINT8

    # ── Create polygons using actual 2D coordinates ───────────────────────
    # For each grid cell, create a polygon from its corner coordinates
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

    run_str = run_date.strftime('%Y-%m-%dT%H:%M:%SZ') if run_date else None
    state = gpd.read_file(STATE_BOUNDARY_SHP).to_crs("EPSG:32615")
    state_union = unary_union(state.geometry)

    regions = []
    for level, meta in DANGER_LEVELS.items():
        polys = level_cells.get(level, [])
        if not polys:
            continue

        merged = unary_union(polys)
        # Close small grid seams, then trim the result to Missouri. Work
        # in a projected CRS so the buffer is measured in meters.
        projected = gpd.GeoSeries([merged], crs="EPSG:4326").to_crs("EPSG:32615")
        buffered = projected.iloc[0].buffer(POLYGON_BUFFER_METERS, join_style=2)
        clipped = buffered.intersection(state_union)
        merged = gpd.GeoSeries([clipped], crs="EPSG:32615").to_crs("EPSG:4326").iloc[0]
        # Simplify to reduce file size while preserving topology.
        merged = merged.simplify(0.0005, preserve_topology=True)

        regions.append({
            "danger_level": level,
            "label": meta["label"],
            "color": meta["color"],
            "model_run": run_str,
            "buffer_meters": POLYGON_BUFFER_METERS,
            "clipped_to": "Missouri state boundary",
            "geometry": merged,
        })

    return regions


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
    """
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        regions = _build_danger_level_regions(peak_risk_smooth, lon, lat, run_date)

        features = [
            {
                "type": "Feature",
                "geometry": mapping(r["geometry"]),
                "properties": {
                    "danger_level": r["danger_level"],
                    "label":        r["label"],
                    "color":        r["color"],
                    "model_run":    r["model_run"],
                    "buffer_meters": r["buffer_meters"],
                    "clipped_to":   r["clipped_to"],
                }
            }
            for r in regions
        ]

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
# 4.  Shapefile – zipped .shp/.shx/.dbf/.prj bundle + QGIS/SLD style files
# ══════════════════════════════════════════════════════════════════════════════

def _qml_categorized_polygon_style(field_name: str = "level") -> str:
    """
    QGIS layer style (.qml) with a categorized renderer matching DANGER_LEVELS.
    Load in QGIS via Layer Properties → Symbology → Style → Load Style.
    """
    def rgba(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"{r},{g},{b},255"

    categories = []
    symbols = []
    for level, meta in DANGER_LEVELS.items():
        categories.append(
            f'      <category value="{level}" symbol="{level}" label="{meta["label"]}" render="true"/>'
        )
        symbols.append(f'''      <symbol type="fill" name="{level}" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <prop k="color" v="{rgba(meta["color"])}"/>
          <prop k="outline_color" v="35,35,35,255"/>
          <prop k="outline_width" v="0.26"/>
          <prop k="outline_style" v="solid"/>
          <prop k="style" v="solid"/>
        </layer>
      </symbol>''')

    return f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <renderer-v2 type="categorizedSymbol" attr="{field_name}" forceraster="0" enableorderby="0" symbollevels="0">
    <categories>
{chr(10).join(categories)}
    </categories>
    <symbols>
{chr(10).join(symbols)}
    </symbols>
  </renderer-v2>
</qgis>
'''


def _sld_categorized_polygon_style(layer_name: str, field_name: str = "level") -> str:
    """
    OGC Styled Layer Descriptor (.sld) with the same categorized colors,
    for GIS software that doesn't read QGIS .qml files (GeoServer, ArcGIS, etc).
    """
    rules = []
    for level, meta in DANGER_LEVELS.items():
        rules.append(f'''    <Rule>
      <Name>{meta["label"]}</Name>
      <ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
        <ogc:PropertyIsEqualTo>
          <ogc:PropertyName>{field_name}</ogc:PropertyName>
          <ogc:Literal>{level}</ogc:Literal>
        </ogc:PropertyIsEqualTo>
      </ogc:Filter>
      <PolygonSymbolizer>
        <Fill>
          <CssParameter name="fill">{meta["color"]}</CssParameter>
          <CssParameter name="fill-opacity">0.7</CssParameter>
        </Fill>
        <Stroke>
          <CssParameter name="stroke">#232323</CssParameter>
          <CssParameter name="stroke-width">0.5</CssParameter>
        </Stroke>
      </PolygonSymbolizer>
    </Rule>''')

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor version="1.0.0"
    xmlns="http://www.opengis.net/sld"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.opengis.net/sld http://schemas.opengis.net/sld/1.0.0/StyledLayerDescriptor.xsd">
  <NamedLayer>
    <Name>{layer_name}</Name>
    <UserStyle>
      <Title>Missouri Peak Fire Danger</Title>
      <FeatureTypeStyle>
{chr(10).join(rules)}
      </FeatureTypeStyle>
    </UserStyle>
  </NamedLayer>
</StyledLayerDescriptor>
'''


def _package_shapefile_zip(
    gdf: "gpd.GeoDataFrame",
    out_path: Path,
    run_str: str,
    *,
    base: str = "peak_fire_danger",
    title: str = "Missouri Peak Fire Danger",
    field_doc: str = (
        "  level      0=Low 1=Moderate 2=Elevated 3=Critical 4=Extreme\n"
        "  label      Danger level name\n"
        "  color      Hex fill color matching the operational legend\n"
        "  model_run  Forecast run timestamp (UTC)\n"
        "  buffer_m   Polygon buffer distance in meters\n"
        "  clipped    Clip boundary description\n"
    ),
) -> bool:
    """
    Write a GeoDataFrame with a "level" field (0-4) as a zipped Esri
    Shapefile bundle plus QGIS (.qml) and OGC (.sld) style files matching
    the operational danger-level legend.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        shp_path = tmp_dir / f"{base}.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

        (tmp_dir / f"{base}.qml").write_text(_qml_categorized_polygon_style(), encoding="utf-8")
        (tmp_dir / f"{base}.sld").write_text(
            _sld_categorized_polygon_style(base), encoding="utf-8"
        )
        (tmp_dir / "README.txt").write_text(
            f"{title} — daily shapefile export\n"
            f"Model run: {run_str}\n"
            "CRS: EPSG:4326 (WGS84)\n\n"
            "Fields:\n"
            f"{field_doc}\n"
            f"{base}.qml — QGIS layer style "
            "(Layer Properties > Symbology > Style > Load Style)\n"
            f"{base}.sld — OGC Styled Layer Descriptor for other GIS software\n",
            encoding="utf-8",
        )

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(tmp_dir.iterdir()):
                zf.write(f, arcname=f.name)

    size_kb = out_path.stat().st_size / 1024
    logger.info(f"Shapefile bundle saved → {out_path}  ({size_kb:.0f} KB, {len(gdf)} features)")
    return True


def export_shapefile(peak_risk_smooth: np.ndarray,
                     lon: np.ndarray,
                     lat: np.ndarray,
                     out_path: Path,
                     run_date=None) -> bool:
    """
    Package today's peak fire danger polygons as a zipped Esri Shapefile
    (.shp/.shx/.dbf/.prj/.cpg) plus QGIS (.qml) and OGC (.sld) style files,
    for use in desktop GIS tools (QGIS, ArcGIS Pro) and other applications.

    Attribute fields (10-char DBF limit):
      level     – integer danger level, 0-4 (0=Low … 4=Extreme)
      label     – danger level name
      color     – hex fill color matching the operational legend
      model_run – forecast run timestamp (UTC)
      buffer_m  – polygon buffer distance in meters
      clipped   – clip boundary description
    """
    try:
        regions = _build_danger_level_regions(peak_risk_smooth, lon, lat, run_date)
        if not regions:
            logger.warning("Shapefile export: no danger-level regions to write")
            return False

        gdf = gpd.GeoDataFrame(
            [{
                "level": int(r["danger_level"]),
                "label": r["label"],
                "color": r["color"],
                "model_run": r["model_run"],
                "buffer_m": r["buffer_meters"],
                "clipped": "MO state boundary",
            } for r in regions],
            geometry=[r["geometry"] for r in regions],
            crs="EPSG:4326",
        )

        run_str = run_date.strftime('%Y-%m-%d %HZ') if run_date else 'unknown'
        return _package_shapefile_zip(gdf, out_path, run_str)

    except Exception as e:
        logger.error(f"Shapefile export failed: {e}", exc_info=True)
        return False


def export_shapefile_from_geojson(geojson_path: Path, out_path: Path) -> bool:
    """
    Rebuild the shapefile bundle from an already-published polygons GeoJSON
    (e.g. api/gis/peak_fire_danger_polygons.geojson) instead of recomputing
    the forecast. Lets today's shapefile be regenerated (or a stale one
    repaired) on demand, without re-running the full ML/HRRR pipeline.

    See scripts/regenerate_shapefile.py for a runnable CLI wrapper.
    """
    try:
        geojson_path = Path(geojson_path)
        with open(geojson_path) as f:
            data = json.load(f)

        run_str = (data.get("metadata") or {}).get("model_run", "unknown")

        rows = []
        geoms = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            rows.append({
                "level": int(props.get("danger_level", -1)),
                "label": props.get("label", ""),
                "color": props.get("color", ""),
                "model_run": props.get("model_run"),
                "buffer_m": props.get("buffer_meters"),
                "clipped": props.get("clipped_to", ""),
            })
            geoms.append(shape(feature["geometry"]))

        if not rows:
            logger.warning(f"Shapefile export: no features in {geojson_path}")
            return False

        gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
        return _package_shapefile_zip(gdf, out_path, run_str)

    except Exception as e:
        logger.error(f"Shapefile export from GeoJSON failed: {e}", exc_info=True)
        return False


def export_shapefile_from_raster(tif_path: Path, out_path: Path, band_index: int = 1) -> bool:
    """
    Build the shapefile bundle by directly vectorizing an already-published
    danger-level GeoTIFF (0-4 categories) rather than recomputing polygons
    from a separate in-memory array. Works for both a single-band legacy
    raster (e.g. gis/latest/forecast_peak_fire_danger.tif) and a multi-band
    one where each band is a forecast day (e.g. forecast_v1's
    rasters/daily/peak_fire_danger.tif, band 1 = Day 1/today, band 2 = Day 2,
    ...) - pick the day with band_index.

    This guarantees the shapefile matches whatever raster is actually being
    served/displayed as the operational forecast pixel-for-pixel, instead of
    depending on a second, parallel dissolve/buffer/clip pipeline that can
    silently drift out of sync with it.
    """
    try:
        import rasterio
        from rasterio.features import shapes as raster_shapes

        tif_path = Path(tif_path)
        with rasterio.open(tif_path) as src:
            band = src.read(band_index)
            nodata = src.nodata
            transform = src.transform
            crs = src.crs
            tags = src.tags()
            band_tags = src.tags(band_index)

        run_str = (
            tags.get("VALID_TIME") or tags.get("RUN_TIME") or tags.get("MODEL_RUN")
            or band_tags.get("valid_time") or "unknown"
        )
        mask = band != nodata if nodata is not None else None

        level_polys = {level: [] for level in DANGER_LEVELS.keys()}
        for geom, value in raster_shapes(band, mask=mask, transform=transform):
            level = int(value)
            if level in level_polys:
                level_polys[level].append(shape(geom))

        rows = []
        geoms = []
        for level, meta in DANGER_LEVELS.items():
            polys = level_polys.get(level, [])
            if not polys:
                continue
            rows.append({
                "level": level,
                "label": meta["label"],
                "color": meta["color"],
                "model_run": run_str,
                "buffer_m": 0.0,
                "clipped": "vectorized from published raster",
            })
            geoms.append(unary_union(polys))

        if not rows:
            logger.warning(f"Shapefile export: no danger-level regions found in {tif_path}")
            return False

        gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=crs).to_crs("EPSG:4326")
        return _package_shapefile_zip(gdf, out_path, run_str)

    except Exception as e:
        logger.error(f"Shapefile export from raster failed: {e}", exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Convenience wrapper – call this from generate_complete_forecast()
# ══════════════════════════════════════════════════════════════════════════════

def export_all_gis_formats(peak_risk_smooth: np.ndarray,
                           lon: np.ndarray,
                           lat: np.ndarray,
                           run_date=None,
                           out_dir: Path = Path('gis'),
                           filename_suffix: str = '') -> dict:
    """
    Export peak fire danger in all three GIS formats.

    Returns a dict of {format: path_or_None} so callers can log/upload selectively.

    filename_suffix (default '', i.e. today's peak_fire_danger.* filenames
    unchanged): set e.g. '_09z' so a secondary-cycle run writes its own
    peak_fire_danger_09z.tif / _polygons_09z.geojson / _points_09z.geojson
    instead of overwriting the live operational files that map tiles/the
    frontend read. Dated archives use the same suffix (for example,
    forecast_peak_09z/archive) so each cycle can be verified independently.

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
    tif_path = out_dir / f'peak_fire_danger{filename_suffix}.tif'
    ok = export_geotiff(peak_risk_smooth, lon, lat, tif_path, run_date)
    results['geotiff'] = tif_path if ok else None
    if ok:
        archive_name = f"forecast_peak{filename_suffix}" if filename_suffix else "forecast_peak"
        results['geotiff_archive'] = archive_forecast_peak_tif(
            tif_path, out_dir, run_date, archive_name=archive_name,
        )

    # ── GeoJSON polygons ──────────────────────────────────────────────────────
    poly_path = out_dir / f'peak_fire_danger_polygons{filename_suffix}.geojson'
    ok = export_geojson_polygons(peak_risk_smooth, lon, lat, poly_path, run_date)
    results['geojson_polygons'] = poly_path if ok else None

    # ── Shapefile bundle (zipped .shp/.shx/.dbf/.prj + .qml/.sld styles) ───────
    # Vectorized directly from the GeoTIFF just written above, so it's
    # guaranteed to match that file (and the map/download built from it)
    # pixel-for-pixel instead of drifting from a second recomputation.
    shp_zip_path = out_dir / f'peak_fire_danger_shapefile{filename_suffix}.zip'
    if results.get('geotiff'):
        ok = export_shapefile_from_raster(tif_path, shp_zip_path)
    else:
        ok = export_shapefile(peak_risk_smooth, lon, lat, shp_zip_path, run_date)
    results['shapefile'] = shp_zip_path if ok else None

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

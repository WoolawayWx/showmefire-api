import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

# Danger levels: 0=Low, 1=Moderate, 2=Elevated, 3=Critical, 4=Extreme
# 255 = nodata in the GeoTIFF
DANGER_LABELS = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
AREA_THRESHOLD = 0.20  # A level must cover ≥10% of the county to count

COUNTY_SHAPEFILE = 'maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp'
FIRE_DANGER_TIF  = '/app/gis/peak_fire_danger.tif'
OUTPUT_JSON      = '/app/gis/dangerbycounty.json'

# Shapefile COUNTYNAME → website GeoJSON NAME mapping for mismatched entries
NAME_OVERRIDES = {
    'Dekalb':        'DeKalb',
    'St Charles':    'St. Charles',
    'St Clair':      'St. Clair',
    'St Francois':   'St. Francois',
    'St Louis':      'St. Louis',
    'St Louis City': 'St. Louis City',
    'Ste Genevieve': 'Ste. Genevieve',
}


def normalize_county_name(name: str) -> str:
    """Normalise shapefile county name to match the website GeoJSON NAME field."""
    return NAME_OVERRIDES.get(name, name)


def classify_county(values: np.ndarray) -> int | None:
    """Return the highest danger level that covers ≥ AREA_THRESHOLD of the county.

    `values` is a 1-D array of valid (non-nodata) grid-cell danger levels (0-4).
    Returns an int 0-4 or None if no valid cells exist.
    """
    if len(values) == 0:
        return None

    total = len(values)
    # Walk from highest level down; first one ≥ threshold wins
    for level in range(4, -1, -1):
        count_at_or_above = np.sum(values >= level)
        if count_at_or_above / total >= AREA_THRESHOLD:
            return int(level)

    # Fallback (shouldn't happen — level 0 always covers 100%)
    return 0


def classify_counties(tif_path: str = FIRE_DANGER_TIF, county_shapefile: str = COUNTY_SHAPEFILE):
    """Classify every Missouri county against a danger-level raster.

    Returns (counties_gdf, results, raster_tags) where counties_gdf is
    reprojected to the raster's CRS and its row order matches `results`
    (one dict per county: county name + max_fire_danger level or None).
    Shared by main() (writes dangerbycounty.json) and export_county_shapefile
    (writes a styled per-county shapefile) so both use the identical
    area-threshold classification.
    """
    counties = gpd.read_file(county_shapefile)

    with rasterio.open(tif_path) as src:
        fire_danger = src.read(1)
        raster_crs = src.crs
        transform = src.transform
        nodata = src.nodata  # typically 255
        tags = src.tags()

    if counties.crs != raster_crs:
        counties = counties.to_crs(raster_crs)

    results = []
    for idx, row in counties.iterrows():
        county_name = normalize_county_name(row['COUNTYNAME'])
        geom = row['geometry']

        # Boolean mask: True where the county intersects the raster
        mask = geometry_mask([geom], transform=transform, invert=True,
                             out_shape=fire_danger.shape, all_touched=True)

        # Extract valid (non-nodata) cells within the county
        county_cells = fire_danger[mask]
        if nodata is not None:
            county_cells = county_cells[county_cells != nodata]

        level = classify_county(county_cells)
        results.append({'county': county_name, 'max_fire_danger': level})

    return counties, results, tags


def export_county_shapefile(out_path, tif_path: str = FIRE_DANGER_TIF,
                            county_shapefile: str = COUNTY_SHAPEFILE) -> bool:
    """Package the per-county danger classification as a styled shapefile.

    One polygon per Missouri county (not per contiguous danger-level
    region), colored by the same area-threshold classification as
    dangerbycounty.json / the public county map - so the shapefile matches
    that map exactly rather than the finer-grained dissolved-raster
    polygons in export_fire_danger_gis.export_shapefile_from_raster.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_fire_danger_gis import DANGER_LEVELS, _package_shapefile_zip

    counties, results, tags = classify_counties(tif_path, county_shapefile)
    run_str = tags.get('VALID_TIME') or tags.get('RUN_TIME') or tags.get('MODEL_RUN') or 'unknown'

    rows = []
    geoms = []
    for row, result in zip(counties.itertuples(), results):
        level = result['max_fire_danger']
        if level is None:
            continue
        meta = DANGER_LEVELS[level]
        rows.append({
            'county': result['county'],
            'level': level,
            'label': meta['label'],
            'color': meta['color'],
            'model_run': run_str,
        })
        geoms.append(row.geometry)

    if not rows:
        print("No classified counties to write to shapefile")
        return False

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=counties.crs).to_crs('EPSG:4326')
    return _package_shapefile_zip(
        gdf, out_path, run_str,
        base='peak_fire_danger_by_county',
        title='Missouri Peak Fire Danger by County',
        field_doc=(
            "  county     Missouri county name\n"
            "  level      0=Low 1=Moderate 2=Elevated 3=Critical 4=Extreme "
            f"(area-threshold ≥{AREA_THRESHOLD:.0%} of county)\n"
            "  label      Danger level name\n"
            "  color      Hex fill color matching the operational legend\n"
            "  model_run  Forecast run timestamp (UTC)\n"
        ),
    )


def main():
    counties, results, _tags = classify_counties()

    for result in results:
        level = result['max_fire_danger']
        county_name = result['county']
        print(f"  {county_name}: {DANGER_LABELS[level] if level is not None else 'N/A'}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} counties to {OUTPUT_JSON}")

    shapefile_path = str(Path(OUTPUT_JSON).parent / 'peak_fire_danger_by_county.zip')
    if export_county_shapefile(shapefile_path):
        print(f"Wrote county shapefile to {shapefile_path}")


if __name__ == '__main__':
    main()

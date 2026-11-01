import json
import os
import sys
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from datetime import datetime
from zoneinfo import ZoneInfo

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import init_database, upsert_county_forecast_day

# Danger levels: 0=Low, 1=Moderate, 2=Elevated, 3=Critical, 4=Extreme
# 255 = nodata in the GeoTIFF
DANGER_LABELS = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
AREA_THRESHOLD = 0.20  # A level must cover at least 20% of the county to count

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


def compute_county_danger(tif_path=FIRE_DANGER_TIF, shapefile_path=COUNTY_SHAPEFILE):
    """Classify every county from the peak fire danger raster.

    Returns [{'county', 'county_fips', 'max_fire_danger', 'cells'}]. Pure
    computation: nothing is written, so a run that must not publish county
    products (the 9z forecast) can still read today's county classes.
    """
    counties = gpd.read_file(shapefile_path)

    with rasterio.open(tif_path) as src:
        fire_danger = src.read(1)
        raster_crs = src.crs
        transform = src.transform
        nodata = src.nodata  # typically 255

    # Reproject counties to raster CRS if needed
    if counties.crs != raster_crs:
        counties = counties.to_crs(raster_crs)

    results = []
    for _idx, row in counties.iterrows():
        county_name = normalize_county_name(row['COUNTYNAME'])
        mask = geometry_mask([row['geometry']], transform=transform, invert=True,
                             out_shape=fire_danger.shape, all_touched=True)
        county_cells = fire_danger[mask]
        if nodata is not None:
            county_cells = county_cells[county_cells != nodata]

        level = classify_county(county_cells)
        raw_fips = str(row.get('COUNTYFIPS') or '').strip()
        county_fips = raw_fips if len(raw_fips) == 5 else f"29{raw_fips.zfill(3)}"
        results.append({
            'county': county_name,
            'county_fips': county_fips,
            'max_fire_danger': level,
            'cells': county_cells,
        })
    return results


def main():
    init_database()
    results = []
    for entry in compute_county_danger():
        county_name = entry['county']
        county_fips = entry['county_fips']
        level = entry['max_fire_danger']
        county_cells = entry.pop('cells')

        results.append(entry)

        if level is not None:
            upsert_county_forecast_day(
                datetime.now(ZoneInfo("America/Chicago")).date().isoformat(),
                county_fips,
                level,
                f"{DANGER_LABELS[level]} fire danger forecast for {county_name} County.",
                datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d"),
            )

        pct_str = ''
        if level is not None and len(county_cells) > 0:
            pct = np.sum(county_cells >= level) / len(county_cells) * 100
            pct_str = f' ({pct:.0f}% ≥ {DANGER_LABELS[level]})'
        print(f"  {county_name}: {DANGER_LABELS[level] if level is not None else 'N/A'}{pct_str}"
              f"  [{len(county_cells)} cells]")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} counties to {OUTPUT_JSON}")


if __name__ == '__main__':
    main()

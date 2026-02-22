import json
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

# Danger levels: 0=Low, 1=Moderate, 2=Elevated, 3=Critical, 4=Extreme
# 255 = nodata in the GeoTIFF
DANGER_LABELS = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
AREA_THRESHOLD = 0.10  # A level must cover ≥10% of the county to count

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


def main():
    counties = gpd.read_file(COUNTY_SHAPEFILE)

    with rasterio.open(FIRE_DANGER_TIF) as src:
        fire_danger = src.read(1)
        raster_crs = src.crs
        transform = src.transform
        nodata = src.nodata  # typically 255

    # Reproject counties to raster CRS if needed
    if counties.crs != raster_crs:
        counties = counties.to_crs(raster_crs)

    print(f"Raster shape: {fire_danger.shape}, CRS: {raster_crs}, nodata: {nodata}")
    print(f"County CRS: {counties.crs}, bounds: {counties.total_bounds}")

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

        results.append({
            'county': county_name,
            'max_fire_danger': level,
        })

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

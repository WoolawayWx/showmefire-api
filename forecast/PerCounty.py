import json
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry import shape
from rasterio.transform import from_bounds

def main():
    # Paths
    county_shapefile_path = 'maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp'
    fire_danger_npy_path = 'data/fire_danger_grid.npy'
    fire_danger_json_path = 'data/county_max_firedanger.json'

    # Load counties from shapefile
    counties = gpd.read_file(county_shapefile_path).to_crs("EPSG:4326")

    # Load fire danger grid and lon/lat grids
    fire_danger = np.load(fire_danger_npy_path)
    lon = np.load('data/lon_grid.npy')
    lat = np.load('data/lat_grid.npy')

    # Mask for Missouri bounding box
    mo_mask = (
        (lon >= -95.8) & (lon <= -89.1) &
        (lat >= 35.8) & (lat <= 40.8)
    )

    # Find the bounding box indices for the crop
    rows, cols = np.where(mo_mask)
    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError("No grid cells found in Missouri bounding box.")

    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    # Crop arrays
    fire_danger_crop = fire_danger[row_min:row_max+1, col_min:col_max+1]
    lon_crop = lon[row_min:row_max+1, col_min:col_max+1]
    lat_crop = lat[row_min:row_max+1, col_min:col_max+1]

    # Calculate new transform for the cropped grid
    lon_min, lon_max = lon_crop.min(), lon_crop.max()
    lat_min, lat_max = lat_crop.min(), lat_crop.max()
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, fire_danger_crop.shape[1], fire_danger_crop.shape[0])

    print("Cropped grid bounds:", lon_min, lon_max, lat_min, lat_max)
    print("County CRS:", counties.crs)
    print("County bounds:", counties.total_bounds)

    results = []
    for idx, row in counties.iterrows():
        if idx == 0:
            print(counties.columns)
        county_name = row['COUNTYNAME']
        geom = row['geometry']

        # Create mask for this county on the cropped grid
        mask = geometry_mask([geom], transform=transform, invert=True,
                             out_shape=fire_danger_crop.shape, all_touched=True)
        county_values = np.where(mask, fire_danger_crop, np.nan)
        if np.all(np.isnan(county_values)):
            max_danger = None
        else:
            max_danger = float(np.nanmax(county_values))
        results.append({
            'county': county_name,
            'max_fire_danger': max_danger
        })

    with open(fire_danger_json_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()

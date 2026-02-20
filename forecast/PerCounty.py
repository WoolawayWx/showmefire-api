import json
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry import shape
from rasterio.transform import from_bounds

def main():

    # Paths
    county_shapefile_path = 'maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp'
    fire_danger_tif_path = '/app/gis/peak_fire_danger.tif' # Update this path if needed
    fire_danger_json_path = '/app/gis/dangerbycounty.json'

    # Load counties from shapefile
    counties = gpd.read_file(county_shapefile_path)

    import rasterio
    with rasterio.open(fire_danger_tif_path) as src:
        fire_danger = src.read(1)
        raster_crs = src.crs
        transform = src.transform

    # Reproject counties to raster CRS if needed
    if counties.crs != raster_crs:
        counties = counties.to_crs(raster_crs)

    print("Raster CRS:", raster_crs)
    print("County CRS:", counties.crs)
    print("County bounds:", counties.total_bounds)

    results = []
    for idx, row in counties.iterrows():
        if idx == 0:
            print(counties.columns)
        county_name = row['COUNTYNAME']
        geom = row['geometry']

        # Create mask for this county on the raster grid
        mask = geometry_mask([geom], transform=transform, invert=True,
                             out_shape=fire_danger.shape, all_touched=True)
        county_values = np.where(mask, fire_danger, np.nan)
        if np.all(np.isnan(county_values)):
            max_danger = None
        else:
            max_danger = float(np.nanpercentile(county_values[~np.isnan(county_values)], 95))
        results.append({
            'county': county_name,
            'max_fire_danger': max_danger
        })

    with open(fire_danger_json_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()

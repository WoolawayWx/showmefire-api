"""
Export Missouri state boundary to GeoJSON for web mapping.
Creates both normal boundary and an inverse mask for clipping.
"""

import json
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, Polygon

# Paths
PROJECT_DIR = Path(__file__).parent.parent
SHAPEFILE_PATH = PROJECT_DIR / 'maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp'
GIS_DIR = PROJECT_DIR / 'gis'

def export_missouri_boundary():
    """Export Missouri boundary as GeoJSON"""
    
    # Read shapefile
    missouri = gpd.read_file(SHAPEFILE_PATH)
    
    # Simplify to reduce file size (tolerance in degrees)
    missouri_simplified = missouri.simplify(0.001, preserve_topology=True)
    
    # Convert to EPSG:4326 if not already
    if missouri.crs != 'EPSG:4326':
        missouri_simplified = missouri_simplified.to_crs('EPSG:4326')
    
    # Export as GeoJSON
    output_path = GIS_DIR / 'missouri_boundary.geojson'
    missouri_simplified.to_file(output_path, driver='GeoJSON')
    
    print(f"✓ Exported Missouri boundary to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path


def create_inverse_mask():
    """
    Create an inverse mask - a polygon covering the world except Missouri.
    This is used to dim/hide areas outside Missouri on the map.
    """
    
    # Read Missouri boundary
    missouri = gpd.read_file(SHAPEFILE_PATH)
    
    if missouri.crs != 'EPSG:4326':
        missouri = missouri.to_crs('EPSG:4326')
    
    # Get Missouri geometry (dissolve all features into one)
    mo_geom = missouri.unary_union
    
    # Create a bounding box covering a larger area around Missouri
    # Missouri bounds approximately: -95.8 to -89.1 longitude, 35.9 to 40.6 latitude
    # Add padding for visual context
    bbox = box(-100, 33, -85, 43)
    
    # Subtract Missouri from the bounding box (creates a polygon with a hole)
    inverse_mask = bbox.difference(mo_geom)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'name': ['Outside Missouri']},
        geometry=[inverse_mask],
        crs='EPSG:4326'
    )
    
    # Export as GeoJSON
    output_path = GIS_DIR / 'missouri_inverse_mask.geojson'
    gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"✓ Created inverse mask: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path


if __name__ == '__main__':
    print("Exporting Missouri boundary files...")
    GIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Export both versions
    boundary_path = export_missouri_boundary()
    mask_path = create_inverse_mask()
    
    print(f"\n✓ Done! Files ready for web mapping:")
    print(f"  - {boundary_path.name}")
    print(f"  - {mask_path.name}")

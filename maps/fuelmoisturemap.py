import cartopy.crs as ccrs
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
from shapely.ops import unary_union
from shapely.affinity import translate
from shapely.geometry import box, Point
import json
import matplotlib.patheffects as path_effects
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from pathlib import Path
import requests
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
from io import BytesIO
from dotenv import load_dotenv
import os
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import geojson
from shapely.geometry import LineString, mapping
from pathlib import Path

def generate_extent(center_lon, center_lat, zoom_width, zoom_height):
    lon_min = center_lon - zoom_width / 2
    lon_max = center_lon + zoom_width / 2
    lat_min = center_lat - zoom_height / 2
    lat_max = center_lat + zoom_height / 2
    return (lon_min, lon_max, lat_min, lat_max)

def add_hillshade(ax, hillshade_path, extent, data_crs, alpha=0.3, cmap='gray'):
    with rasterio.open(hillshade_path) as src:
        hillshade = src.read(1)
        
        # Get the bounds from the raster
        bounds = src.bounds
        hillshade_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # Display the hillshade
        ax.imshow(hillshade, 
                  extent=hillshade_extent,
                  transform=data_crs,
                  cmap=cmap,
                  alpha=alpha,
                  zorder=1,  # Behind everything else
                  interpolation='lanczos')

def generate_basemap():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent

    # Map Parameters
    pixelw = 2048
    pixelh = 1152
    mapdpi = 144

    center_lon = -92.5
    center_lat = 38.5
    # Missouri bounds with margin: lon -96.27 to -88.6, lat 35.49 to 41.11
    extent = (-95.8, -89.1, 35.8, 40.8)

    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    figsize_width = pixelw / mapdpi
    figsize_height = pixelh / mapdpi

    fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=mapdpi, facecolor='#E8E8E8')
    ax = plt.axes([0,0,1,1], projection=map_crs)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_extent(extent, crs=data_crs)

    # hillshade_path = PROJECT_DIR / 'assets/hillshade.tif'
    # if hillshade_path.exists():
    #    add_hillshade(ax, hillshade_path, extent, data_crs, alpha=0.3, cmap='gray')
    
    map_extent = ax.get_extent(crs=data_crs)  
    bbox = box(*map_extent)
    
    counties = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp')
    if counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", 
                    facecolor='none', linewidth=1, zorder=5)

    
    missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", 
                    facecolor='none', linewidth=1, zorder=6)

    
    return fig, ax, data_crs, map_crs, mapdpi

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Fetch stations from the API
load_dotenv()
port = os.getenv('PORT', '8000')
response = requests.get(f'http://localhost:{port}/stations')
data = response.json()
stations = data['stations']

# Filter to RAWS network
raws_stations = [s for s in stations if s.get('network') == 'RAWS']

# Collect fuel moisture data points
points = []
for s in raws_stations:
    if 'observations' in s and 'fuel_moisture' in s['observations']:
        fm = s['observations']['fuel_moisture']
        if fm.get('value') is not None:
            points.append((s['longitude'], s['latitude'], fm['value']))

# Generate the blank map
fig, ax, data_crs, map_crs, mapdpi = generate_basemap()

# Create grid for interpolation
lon_min, lon_max, lat_min, lat_max = -95.8, -89.1, 35.8, 40.8
grid_lon = np.linspace(lon_min, lon_max, 200)
grid_lat = np.linspace(lat_min, lat_max, 200)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

if points:
    points_lon = [p[0] for p in points]
    points_lat = [p[1] for p in points]
    values = [p[2] for p in points]
    
    # Interpolate fuel moisture onto the grid
    grid_values = griddata((points_lon, points_lat), values, (grid_lon_mesh, grid_lat_mesh), method='linear')
    
    # Fill any NaN with nearest neighbor
    nan_mask = np.isnan(grid_values)
    if np.any(nan_mask):
        nearest_values = griddata((points_lon, points_lat), values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
        grid_values[nan_mask] = nearest_values[nan_mask]
    
    # Mask to Missouri boundary
    missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor='none', 
                    facecolor='none', linewidth=2, zorder=6)
    if not missouriborder.empty:
        missouri_geom = missouriborder.geometry.iloc[0]
        # Create GeoSeries of grid points
        grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
        within_mask = gpd.GeoSeries(grid_points).within(missouri_geom)
        within_mask = within_mask.values.reshape(grid_lon_mesh.shape)
        # Set outside to NaN
        grid_values[~within_mask] = np.nan
    
    # Plot the interpolated grid with higher resolution and more levels for smoother gradient
    cs = ax.contourf(
        grid_lon_mesh, grid_lat_mesh, grid_values, transform=data_crs,
        levels=np.linspace(0, 30, 101), cmap='RdYlGn', alpha=0.7, zorder=7, antialiased=True
    )
    
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])  # [left, bottom, width, height] in figure coordinates
    cbar = plt.colorbar(cs, cax=cax, label='Fuel Moisture (%)', ticks=np.arange(0, 32, 2))

    # Add a dotted contour line at 9%
    contour_9 = ax.contour(
        grid_lon_mesh, grid_lat_mesh, grid_values, levels=[9], colors='black',
        linestyles='dotted', linewidths=2, transform=data_crs, zorder=8
    )
    cbar.ax.axhline(y=9, color='black', linewidth=2.5, linestyle='--', zorder=10)

# Plot RAWS stations
if raws_stations:
    lats = [s['latitude'] for s in raws_stations]
    lons = [s['longitude'] for s in raws_stations]
    ax.scatter(lons, lats, transform=data_crs, color='black', s=20, zorder=10, edgecolor='white', linewidth=1)
    # Add fuel moisture labels at each RAWS site
    for s in raws_stations:
        fm = s.get('observations', {}).get('fuel_moisture', {}).get('value')
        if fm is not None:
            ax.text(
                s['longitude'], s['latitude']+0.05, f"{fm:.1f}%", 
                transform=data_crs, fontsize=9, color='black', zorder=11,
                ha='center', va='bottom',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')]
            )

counties = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp')
if counties.crs != data_crs.proj4_init:
    counties = counties.to_crs(data_crs.proj4_init)
ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor='none', linewidth=1, zorder=5)
    # counties_filtered = counties[counties.within(coverage_geom)]
    #ax.add_geometries(counties_filtered.geometry, crs=data_crs, edgecolor="#616161", 
    #                facecolor='none', linewidth=1.5, zorder=6)
    
missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
if missouriborder.crs != data_crs.proj4_init:
    missouriborder = missouriborder.to_crs(data_crs.proj4_init)
ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", facecolor='none', linewidth=1.5, zorder=6)



# Force the map to the left side of its subplot area
ax.set_anchor('W') 

# Optional: Remove figure-level margins to move it even further left
plt.subplots_adjust(left=0.05) 



# Register custom fonts
font_paths = [
    str(SCRIPT_DIR.parent / 'assets/Montserrat/static/Montserrat-Regular.ttf'),
    str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf'),
    str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf')
]
for font_path in font_paths:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Montserrat'

# Add annotation box to the right of the map
fig.text(
    0.99, 0.97,
    "Missouri Statewide Fuel Moisture Map",
    fontsize=26,
    fontweight='bold',
    ha='right',
    va='top',
    fontname='Plus Jakarta Sans'
)
fig.text(
    0.99, 0.6,
    "Data Source: USFS RAWS stations from Synoptic Weather\n\n"
    "The 9% fuel moisture (black dotted line) is the critical threshold\n"
    "in place across much of Missouri, as defined in the Fire Weather AOP\n"
    "by NWS LSX, EAX, SGF, and PAH.\n\n"
    "Due to the low number of stations reporting fuel moisture,\n"
    "data may not be accurate, and local variations may occur.",
    fontsize=12,
    ha='right',
    va='top',
    fontname='Montserrat'
)
fig.text(
    0.99, 0.90,
    "Valid Time: {date}".format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')),
    fontsize=16,
    ha='right',
    va='top',
    fontname='Montserrat'
)
fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight='bold', ha='left', va='bottom', fontname='Montserrat')


# Path to your SVG logo
svg_path = str(SCRIPT_DIR.parent / 'assets/LightBackGroundLogo.svg')

# Read SVG as image using matplotlib (requires cairosvg or pillow with svg support)
try:
    # Convert SVG to PNG in memory
    png_bytes = cairosvg.svg2png(url=svg_path)
    image = mpimg.imread(BytesIO(png_bytes), format='png')
except ImportError:
    # If cairosvg is not available, skip logo
    image = None

if image is not None:
    # Create an OffsetImage and place it on the figure
    imagebox = OffsetImage(image, zoom=0.03)
    ab = AnnotationBbox(
        imagebox, (0.99, 0.01), frameon=False,
        xycoords='figure fraction', box_alignment=(1,0)
    )
    ax.add_artist(ab)

# Save the figure as an image
fig.savefig('images/fuelmoisturemap.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)



print(f"Fuel Moisture updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")

# Define contour levels: every 1% from 0-20, then every 5% from 25-30
contour_levels = list(range(0, 21, 1)) + list(range(25, 31, 5))

#Generate contours
contour_set = ax.contour(
    grid_lon_mesh, grid_lat_mesh, grid_values,
    levels=contour_levels,
    colors='none',  # Don't draw, just extract
    linewidths=0.1,
    transform=data_crs
)

features = []
# Use allsegs - it's a list of lists of arrays (one list per contour level)
for i, level_segs in enumerate(contour_set.allsegs):
    for seg in level_segs:
        if len(seg) > 1:  # Need at least 2 points for a line
            line = LineString(seg)
            features.append(geojson.Feature(
                geometry=mapping(line),
                properties={"fuel_moisture": contour_levels[i]}
            ))

# Save as GeoJSON
geojson_obj = geojson.FeatureCollection(features)
with open("gis/fuelmoisture_contours.geojson", "w") as f:
    geojson.dump(geojson_obj, f)
    
# Close the figure to free memory
plt.close(fig)

status_file = Path(__file__).parent.parent / 'status.json'
# Load existing status or create empty dict
if status_file.exists():
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
    except json.JSONDecodeError:
        # File exists but is empty or invalid JSON; start with empty dict
        status = {}
else:
    status = {}

# Update the status for this map (change 'rh_map' to the appropriate key)
status['RealTimeFuelMoisture'] = {
    'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT'),
    'status': 'updated'
}

# Save back to status.json
with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
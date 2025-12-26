import cartopy.crs as ccrs
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
from shapely.ops import unary_union
from shapely.affinity import translate
from shapely.geometry import box, Point, LineString, mapping
import json
import matplotlib.patheffects as path_effects
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from pathlib import Path
import requests
from scipy.interpolate import griddata, Rbf  # Add this import at the top if not present
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
from io import BytesIO
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
from dotenv import load_dotenv
import os
import geojson
import rasterio.features
import time
import logging

start_time = time.time()

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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

load_dotenv()
port = os.getenv('PORT', '8000')
response = requests.get(f'http://localhost:{port}/stations')
data = response.json()
stations = data['stations']

valid_networks = {'RAWS', 'ASOS/AWOS', 'MOCOMAGNET', 'Missouri Mesonet'}
filtered_stations = [s for s in stations if s.get('network') in valid_networks] 

points = []
for s in filtered_stations:
    if 'observations' in s and 'relative_humidity' in s['observations']:
        rh = s['observations']['relative_humidity']
        if rh.get('value') is not None:
            capped_rh = min(rh['value'], 100)  # Cap at 100%
            points.append((s['longitude'], s['latitude'], capped_rh))

fig, ax, data_crs, map_crs, mapdpi = generate_basemap()

lon_min, lon_max, lat_min, lat_max = -95.8, -89.1, 35.8, 40.8
grid_lon = np.linspace(lon_min, lon_max, 400)
grid_lat = np.linspace(lat_min, lat_max, 400)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

if points:
    points_lon = [p[0] for p in points]
    points_lat = [p[1] for p in points]
    values = [p[2] for p in points]

    # Interpolation and smoothing (same as fuel moisture)
    rbf = Rbf(points_lon, points_lat, values, function='multiquadric', smooth=0.005)
    grid_values = rbf(grid_lon_mesh, grid_lat_mesh)
    grid_values = np.minimum(grid_values, 100)  # Cap all values at 100
    grid_values = gaussian_filter(grid_values, sigma=0)

    # Masking (same as fuel moisture)
    missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)

    if not missouriborder.empty:
        missouri_geom = missouriborder.geometry.iloc[0]
        grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
        within_mask = gpd.GeoSeries(grid_points).within(missouri_geom).values.reshape(grid_lon_mesh.shape)
        grid_values[~within_mask] = np.nan

    # Plotting (same as fuel moisture)
    cs = ax.contourf(
        grid_lon_mesh, grid_lat_mesh, grid_values, transform=data_crs,
        levels=np.linspace(0, 101, 256),
        cmap='RdYlGn', alpha=0.75, zorder=7, antialiased=True
    )

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Relative Humidity (%)', ticks=np.arange(0, 102, 5))

if filtered_stations:
    lats = []
    lons = []
    for s in filtered_stations:
        rh = s.get('observations', {}).get('relative_humidity', {}).get('value')
        if rh is not None:
            lats.append(s['latitude'])
            lons.append(s['longitude'])
    
    label_positions = []
    min_dist = 0.12
    for s in filtered_stations:
        rh = s.get('observations', {}).get('relative_humidity', {}).get('value')
        if rh is not None:
            pos = np.array([s['longitude'], s['latitude']])
            if all(np.linalg.norm(pos - np.array(lp)) > min_dist for lp in label_positions):
                ax.scatter(
                    s['longitude'], s['latitude'],
                    transform=data_crs, color='black', s=20, zorder=10,
                    edgecolor='white', linewidth=1
                )
                ax.text(
                    s['longitude'], s['latitude']+0.03, f"{int(round(rh))}%",
                    transform=data_crs, fontsize=9, color='black', zorder=11,
                    ha='center', va='bottom',
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='white')]
                )
                label_positions.append(pos)

counties = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp')
if counties.crs != data_crs.proj4_init:
    counties = counties.to_crs(data_crs.proj4_init)
ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor='none', linewidth=1, zorder=5)
    
missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
if missouriborder.crs != data_crs.proj4_init:
    missouriborder = missouriborder.to_crs(data_crs.proj4_init)
ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", facecolor='none', linewidth=1.5, zorder=6)

ax.set_anchor('W') 

plt.subplots_adjust(left=0.05) 

font_paths = [
    str(SCRIPT_DIR.parent / 'assets/Montserrat/static/Montserrat-Regular.ttf'),
    str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf'),
    str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf')
]
for font_path in font_paths:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Montserrat'

fig.text(
    0.99, 0.97,
    "Missouri Statewide Relative Humidity Map",
    fontsize=26,
    fontweight='bold',
    ha='right',
    va='top',
    fontname='Plus Jakarta Sans'
)
fig.text(
    0.99, 0.6,
    "Data Source: Offical Sites Only from Synoptic Weather\n"
    "AWOS, RAWS, Missouri Mesonet\n\n"
    "Relative humidity is a key factor in fire weather and fuel drying.\n"
    "Values interpolated from available stations; local variations may occur.",
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

svg_path = str(SCRIPT_DIR.parent / 'assets/LightBackGroundLogo.svg')

try:
    png_bytes = cairosvg.svg2png(url=svg_path)
    image = mpimg.imread(BytesIO(png_bytes), format='png')
except ImportError:
    image = None

if image is not None:
    imagebox = OffsetImage(image, zoom=0.03)
    ab = AnnotationBbox(
        imagebox, (0.99, 0.01), frameon=False,
        xycoords='figure fraction', box_alignment=(1,0)
    )
    ax.add_artist(ab)

fig.savefig('images/mo-rh.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

runtime_sec = time.time() - start_time


print(f"RH% Filtered Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
print(f"Script runtime: {runtime_sec:.2f} seconds")

logging.basicConfig(filename='logs/rhmap-fil.log', level=logging.INFO)
logging.info(f"RH% Filtered Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
logging.info(f"Script runtime: {runtime_sec:.2f} seconds")

plt.close(fig)

status_file = Path(__file__).parent.parent / 'status.json'
if status_file.exists():
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
    except json.JSONDecodeError:
        status = {}
else:
    status = {}

status['HumidityFiltered'] = {
    'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT'),
    'status': 'updated',
    'runtime_sec': round(runtime_sec, 2),
    'log': [
        f"RH% Filtered Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}",
        f"Script runtime: {runtime_sec:.2f} seconds"
    ]
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
import cartopy.crs as ccrs
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf  # Added for Option 1
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
from io import BytesIO
from dotenv import load_dotenv
import os
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import geojson
import time
import logging
from logging.handlers import RotatingFileHandler
from shapely.geometry import Polygon, MultiPolygon

from realtime_geotiff import export_continuous_rgba_geotiff


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
    pixelw, pixelh, mapdpi = 2048, 1152, 144
    center_lon, center_lat = -92.5, 38.5
    extent = (-95.8, -89.1, 35.8, 40.8)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    fig = plt.figure(figsize=(pixelw / mapdpi, pixelh / mapdpi), dpi=mapdpi, facecolor='#E8E8E8')
    ax = plt.axes([0,0,1,1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)
    return fig, ax, data_crs, map_crs, mapdpi

start_time = time.time()

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv()
port = os.getenv('PORT', '8000')
response = requests.get(f'http://localhost:{port}/stations/raws')
stations = response.json()['stations']

points = []
for s in stations:
    fm = s.get('observations', {}).get('fuel_moisture', {}).get('value')
    if fm is not None:
        points.append((s['longitude'], s['latitude'], fm))

fig, ax, data_crs, map_crs, mapdpi = generate_basemap()

lon_min, lon_max, lat_min, lat_max = -95.8, -89.1, 35.8, 40.8
grid_lon = np.linspace(lon_min, lon_max, 400) 
grid_lat = np.linspace(lat_min, lat_max, 400)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

if points:
    points_lon = [p[0] for p in points]
    points_lat = [p[1] for p in points]
    values = [p[2] for p in points]
    
    rbf = Rbf(points_lon, points_lat, values, function='multiquadric', smooth=0.005)
    grid_values = rbf(grid_lon_mesh, grid_lat_mesh)
    grid_values = gaussian_filter(grid_values, sigma=0)

    missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    
    if not missouriborder.empty:
        missouri_geom = missouriborder.geometry.iloc[0]
        grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
        within_mask = gpd.GeoSeries(grid_points).within(missouri_geom).values.reshape(grid_lon_mesh.shape)
        grid_values[~within_mask] = np.nan

    geotiff_ok = export_continuous_rgba_geotiff(
        grid_values=grid_values,
        lon_mesh=grid_lon_mesh,
        lat_mesh=grid_lat_mesh,
        out_path=SCRIPT_DIR.parent / 'gis/realtime/fuel_moisture.tif',
        cmap_name='RdYlGn',
        vmin=0,
        vmax=30,
        description='Missouri realtime fuel moisture (RGBA)',
        source='Synoptic RAWS observations + ShowMeFire interpolation',
        legend='Fuel Moisture (%) mapped with RdYlGn, transparent outside Missouri',
    )
    if not geotiff_ok:
        print("Warning: Failed to export fuel moisture GeoTIFF")
    
    cs = ax.contourf(
        grid_lon_mesh, grid_lat_mesh, grid_values, transform=data_crs,
        levels=np.linspace(0, 30, 256),
        cmap='RdYlGn', alpha=0.75, zorder=7, antialiased=True
    )
    
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Fuel Moisture (%)', ticks=np.arange(0, 32, 2))
    contour_9 = ax.contour(
        grid_lon_mesh, grid_lat_mesh, grid_values, levels=[9], colors='black',
        linestyles='dotted', linewidths=2, transform=data_crs, zorder=8
    )
    cbar.ax.axhline(y=9, color='black', linewidth=2.5, linestyle='--', zorder=10)

mo_stations = [s for s in stations if s.get('state') == 'MO']
lats = [s['latitude'] for s in mo_stations]
lons = [s['longitude'] for s in mo_stations]
ax.scatter(lons, lats, transform=data_crs, color='black', s=20, zorder=10, edgecolor='white', linewidth=1)

for s in stations:
    if s.get('state') == 'MO':
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
    "data may not be accurate, and local variations may occur.\n"
    "RAWS stations outside of Missouri are used in order\n to create a proper plot.",
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

fig.savefig('images/mo-fuelmoisture.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

runtime_sec = time.time() - start_time

print(f"Fuel Moisture updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
print(f"Script runtime: {runtime_sec:.2f} seconds")

# Set up rotating log handler (max 5MB per file, keep 5 backup files)
log_file = Path(__file__).parent.parent / 'logs/fuelmoisturemap.log'
log_file.parent.mkdir(exist_ok=True)
logger = logging.getLogger('fuelmoisturemap')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
logger.info(f"Fuel Moisture updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
logger.info(f"Script runtime: {runtime_sec:.2f} seconds")

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

status['RealTimeFuelMoisture'] = {
    'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT'),
    'status': 'updated',
    'runtime_sec': round(runtime_sec, 2),
    'log': [
        f"Fuel Moisture updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}",
        f"Script runtime: {runtime_sec:.2f} seconds"
    ]
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)

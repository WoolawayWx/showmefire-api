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
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg

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

response = requests.get('http://localhost:8000/stations')
data = response.json()
stations = data['stations']

valid_networks = {'RAWS', 'ASOS/AWOS', 'MOCOMAGNET', 'Missouri Mesonet'}
filtered_stations = [s for s in stations if s.get('network') in valid_networks] 

points = []
for s in filtered_stations:
    if 'observations' in s and 'relative_humidity' in s['observations']:
        rh = s['observations']['relative_humidity']
        if rh.get('value') is not None:
            points.append((s['longitude'], s['latitude'], rh['value']))

fig, ax, data_crs, map_crs, mapdpi = generate_basemap()

lon_min, lon_max, lat_min, lat_max = -95.8, -89.1, 35.8, 40.8
grid_lon = np.linspace(lon_min, lon_max, 200)
grid_lat = np.linspace(lat_min, lat_max, 200)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

if points:
    filtered_points = [p for p in points if 5 <= p[2] <= 100]
    if not filtered_points:
        print("No valid relative humidity data points found.")
    else:
        points_lon = [p[0] for p in filtered_points]
        points_lat = [p[1] for p in filtered_points]
        values = [p[2] for p in filtered_points]
        
        grid_values = griddata((points_lon, points_lat), values, (grid_lon_mesh, grid_lat_mesh), method='linear')
        
        nan_mask = np.isnan(grid_values)
        if np.any(nan_mask):
            nearest_values = griddata((points_lon, points_lat), values, (grid_lon_mesh, grid_lat_mesh), method='nearest')
            grid_values[nan_mask] = nearest_values[nan_mask]
        
        missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
        if missouriborder.crs != data_crs.proj4_init:
            missouriborder = missouriborder.to_crs(data_crs.proj4_init)
        ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor='none', 
                        facecolor='none', linewidth=2, zorder=6)
        if not missouriborder.empty:
            missouri_geom = missouriborder.geometry.iloc[0]
            grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
            within_mask = gpd.GeoSeries(grid_points).within(missouri_geom)
            within_mask = within_mask.values.reshape(grid_lon_mesh.shape)
            grid_values[~within_mask] = np.nan
        
        cs = ax.contourf(
            grid_lon_mesh, grid_lat_mesh, grid_values, transform=data_crs,
            levels=np.linspace(0, 100, 101), cmap='RdYlGn', alpha=0.7, zorder=7, antialiased=True
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

fig.savefig('images/rh-filtered.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

plt.close(fig)

print(f"RH% Filtered Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
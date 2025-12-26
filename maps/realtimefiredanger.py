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
from scipy.interpolate import griddata, Rbf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
from io import BytesIO
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
from dotenv import load_dotenv
import os
import time
import logging

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

def estimate_fuel_moisture(relative_humidity, air_temp=None):
    """
    Estimate 10-hour fuel moisture from relative humidity.
    Based on empirical relationships used in fire weather.
    
    Simple estimation: FM ≈ 0.03 + 0.25 * RH (as decimal)
    This gives roughly: RH 20% → FM 8%, RH 40% → FM 13%
    
    Returns: estimated fuel moisture percentage
    """
    if relative_humidity is None:
        return None
    
    # Convert RH from percentage to decimal
    rh_decimal = relative_humidity / 100.0
    
    # Simple linear estimation
    fm_estimate = 3 + 0.25 * relative_humidity
    
    # Clamp to reasonable range (3-30%)
    fm_estimate = max(3, min(30, fm_estimate))
    
    return fm_estimate

def calculate_fire_danger(fuel_moisture, relative_humidity, wind_speed_knots):
    """
    Calculate fire danger level based on NWCG standards with extended scale.
    
    Parameters:
    - fuel_moisture: 10-hour fuel moisture percentage
    - relative_humidity: 2-meter relative humidity percentage
    - wind_speed_knots: 10-meter sustained wind speed in knots
    
    Returns: danger level (0=Low, 1=Moderate, 2=Elevated, 3=Critical, 4=Extreme)
    """
    
    # If fuel moisture >= 15%, conditions are generally safe
    if fuel_moisture >= 15:
        return 0  # Low
    
    # Fuel moisture 10-15% - MODERATE (heads up, starting to get dry)
    if fuel_moisture >= 10:
        return 1  # Moderate - heads up that fuels are drying
    
    # Fuel moisture < 10% - now check RH and wind for increasing danger levels
    # Following the NWCG table exactly
    
    # EXTREME conditions (Level 4): Very dry fuels + high winds + very low RH
    # FM < 6%, Wind >= 25kts, RH < 15%
    if fuel_moisture < 6 and wind_speed_knots >= 25 and relative_humidity < 15:
        return 4  # Extreme
    
    # Also extreme: FM < 7%, Wind >= 30kts, RH < 20%
    if fuel_moisture < 7 and wind_speed_knots >= 30 and relative_humidity < 20:
        return 4  # Extreme
    
    # CRITICAL conditions (Level 3): Red Flag Warning criteria per NWCG table
    # 15-19kts with RH < 20% = Generally Critical (Red Flag)
    if wind_speed_knots >= 15 and wind_speed_knots < 20 and relative_humidity < 20:
        return 3  # Critical/Red Flag
    # 20-24kts with RH < 25% = Red Flag Warning
    elif wind_speed_knots >= 20 and wind_speed_knots < 25 and relative_humidity < 25:
        return 3  # Critical/Red Flag
    # >= 25kts with RH < 25% = Red Flag Warning (detailed below in table)
    elif wind_speed_knots >= 25 and relative_humidity < 25:
        return 3  # Critical/Red Flag
    # Also critical if FM is very low even with slightly higher RH
    elif fuel_moisture < 7 and wind_speed_knots >= 15 and relative_humidity < 30:
        return 3  # Critical
    
    # ELEVATED conditions (Level 2): Following NWCG table exactly
    # 5-9kts with RH < 20% = Elevated
    if wind_speed_knots >= 5 and wind_speed_knots < 10 and relative_humidity < 20:
        return 2  # Elevated
    # 10-14kts with RH < 35% = Elevated
    elif wind_speed_knots >= 10 and wind_speed_knots < 15 and relative_humidity < 35:
        return 2  # Elevated
    # 15-19kts: RH 20-34% = Elevated
    elif wind_speed_knots >= 15 and wind_speed_knots < 20 and relative_humidity >= 20 and relative_humidity < 35:
        return 2  # Elevated
    # 20-24kts: RH 25-44% = Elevated  
    elif wind_speed_knots >= 20 and wind_speed_knots < 25 and relative_humidity >= 25 and relative_humidity < 45:
        return 2  # Elevated
    # >= 25kts: RH 25-44% = Elevated
    elif wind_speed_knots >= 25 and relative_humidity >= 25 and relative_humidity < 45:
        return 2  # Elevated
    
    # If FM < 10% but doesn't meet elevated criteria, still moderate
    if fuel_moisture < 10:
        return 1  # Moderate
    
    return 0  # Low

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

# --- Use RAWS endpoint for fuel moisture ---
response = requests.get(f'http://localhost:{port}/stations/raws')
raws_stations = response.json()['stations']

# Use all stations for RH and wind, but only RAWS for fuel moisture
response_all = requests.get(f'http://localhost:{port}/stations')
all_stations = response_all.json()['stations']

# Define grid
lon_min, lon_max, lat_min, lat_max = -95.8, -89.1, 35.8, 40.8
grid_lon = np.linspace(lon_min, lon_max, 400)
grid_lat = np.linspace(lat_min, lat_max, 400)
grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

fig, ax, data_crs, map_crs, mapdpi = generate_basemap()

# Collect data for interpolation
rh_points = []
wind_points = []
fuel_points_measured = []  # RAWS stations with actual fuel moisture

for s in all_stations:
    obs = s.get('observations', {})
    rh = obs.get('relative_humidity', {}).get('value')
    if rh is not None and 0 <= rh <= 100:
        rh_points.append((s['longitude'], s['latitude'], rh))
    ws = obs.get('wind_speed', {}).get('value')
    if ws is not None and ws >= 0:
        wind_points.append((s['longitude'], s['latitude'], ws))

# Only use RAWS for fuel moisture
for s in raws_stations:
    fm = s.get('observations', {}).get('fuel_moisture', {}).get('value')
    if fm is not None and fm > 0:
        fuel_points_measured.append((s['longitude'], s['latitude'], fm))

fuel_points = fuel_points_measured

print(f"Data summary: {len(rh_points)} RH, {len(wind_points)} wind, "
      f"{len(fuel_points_measured)} measured FM (RAWS only)")

start_time = time.time()

if rh_points and wind_points and fuel_points:
    # Interpolate RH
    rh_lon = [p[0] for p in rh_points]
    rh_lat = [p[1] for p in rh_points]
    rh_values = [p[2] for p in rh_points]
    rh_rbf = Rbf(rh_lon, rh_lat, rh_values, function='multiquadric', smooth=0.01)
    rh_grid = rh_rbf(grid_lon_mesh, grid_lat_mesh)
    rh_grid = gaussian_filter(rh_grid, sigma=0.7)

    # Interpolate wind
    wind_lon = [p[0] for p in wind_points]
    wind_lat = [p[1] for p in wind_points]
    wind_values = [p[2] for p in wind_points]
    wind_rbf = Rbf(wind_lon, wind_lat, wind_values, function='multiquadric', smooth=0.01)
    wind_grid = wind_rbf(grid_lon_mesh, grid_lat_mesh)
    wind_grid = gaussian_filter(wind_grid, sigma=0.7)

    # Interpolate fuel moisture with priority given to measured values
    fuel_lon = [p[0] for p in fuel_points]
    fuel_lat = [p[1] for p in fuel_points]
    fuel_values = [p[2] for p in fuel_points]
    fuel_rbf = Rbf(fuel_lon, fuel_lat, fuel_values, function='multiquadric', smooth=0.01)
    fuel_grid = fuel_rbf(grid_lon_mesh, grid_lat_mesh)
    fuel_grid = gaussian_filter(fuel_grid, sigma=0.7)

    # If we have measured fuel moisture, blend with estimated as before
    if fuel_points_measured:
        measured_lon = [p[0] for p in fuel_points_measured]
        measured_lat = [p[1] for p in fuel_points_measured]
        measured_values = [p[2] for p in fuel_points_measured]
        measured_rbf = Rbf(measured_lon, measured_lat, measured_values, function='multiquadric', smooth=0.005)
        measured_grid = measured_rbf(grid_lon_mesh, grid_lat_mesh)
        measured_grid = gaussian_filter(measured_grid, sigma=0)

        # Distance-based influence
        influence_grid = np.zeros_like(grid_lon_mesh)
        for mlon, mlat in zip(measured_lon, measured_lat):
            dist = np.sqrt((grid_lon_mesh - mlon)**2 + (grid_lat_mesh - mlat)**2)
            influence = np.exp(-dist / 0.5)
            influence_grid = np.maximum(influence_grid, influence)
        fuel_grid = (influence_grid * measured_grid + (1 - influence_grid) * fuel_grid)

    # Calculate fire danger using NWCG standards
    grid_values = np.zeros_like(rh_grid)
    for i in range(grid_values.shape[0]):
        for j in range(grid_values.shape[1]):
            grid_values[i, j] = calculate_fire_danger(
                fuel_grid[i, j],
                rh_grid[i, j],
                wind_grid[i, j]
            )

    # Define fire danger categories (0=Low, 1=Moderate, 2=Elevated, 3=Critical, 4=Extreme)
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    labels = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
    colors = ['#90EE90', '#FFED4E', '#FFA500', '#FF0000', '#8B0000']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, len(colors))

    # Mask to Missouri
    missouriborder = gpd.read_file(SCRIPT_DIR / 'shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    if not missouriborder.empty:
        missouri_geom = missouriborder.geometry.iloc[0]
        grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
        within_mask = gpd.GeoSeries(grid_points).within(missouri_geom).values.reshape(grid_lon_mesh.shape)
        grid_values[~within_mask] = np.nan

    cs = ax.contourf(
        grid_lon_mesh, grid_lat_mesh, grid_values, transform=data_crs,
        levels=bins, cmap=cmap, norm=norm, alpha=0.7, zorder=7, antialiased=True
    )

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Fire Danger Level')
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(labels)
else:
    print("Insufficient data for fire danger calculation.")

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
    "Missouri Fire Danger Assessment",
    fontsize=26,
    fontweight='bold',
    ha='right',
    va='top',
    fontname='Plus Jakarta Sans'
)
fig.text(
    0.99, 0.6,
    "Data Source: AWOS, RAWS, Missouri Mesonet, CWOP Stations\n\n"
    "Based on NWCG fire danger criteria with 5-level scale:\n"
    "• Low: FM≥15% (fuels adequately moist)\n"
    "• Moderate: FM 10-15% (heads up - fuels drying)\n"
    "• Elevated: FM<10%, following NWCG RH/Wind matrix\n"
    "• Critical: FM<10%, Red Flag criteria\n"
    "• Extreme: FM<7%, severe conditions\n\n"
    "RAWS stations: direct fuel moisture | Others: RH-based estimation \n\n"
    "For More Info, Vist ShowMeFire.org",
    fontsize=10,
    ha='right',
    va='top',
    fontname='Montserrat'
)
fig.text(
    0.99, 0.90,
    "Observations Analysis | Valid Time: {date}".format(date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')),
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

fig.savefig('images/mo-realtimefiredanger.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

runtime_sec = time.time() - start_time

print(f"Fire Danger Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
print(f"Script runtime: {runtime_sec:.2f} seconds")

logging.basicConfig(filename='logs/realtimefiredanger.log', level=logging.INFO)
logging.info(f"Fire Danger Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
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

status['RealtimeFireDanger'] = {
    'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT'),
    'status': 'updated',
    'runtime_sec': round(runtime_sec, 2),
    'log': [
        f"Fire Danger Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}",
        f"Script runtime: {runtime_sec:.2f} seconds"
    ]
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
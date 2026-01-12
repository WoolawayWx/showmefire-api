import cartopy.crs as ccrs
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import logging
import os
import json
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cairosvg
from io import BytesIO
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
from dotenv import load_dotenv
import time
import warnings
from herbie import FastHerbie
import xarray as xr
import requests
from scipy.interpolate import Rbf
import pickle
import gc
import psutil
import xgboost as xgb
import sys
import rasterio
from rasterio.transform import from_bounds
import sqlite3

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import get_db_path

# Load the production model once
FM_MODEL = xgb.Booster()
FM_MODEL.load_model('models/fuel_moisture_model.json')

# The exact features the model expects
# Extended features list for models trained with precipitation
FEATURES = [
    'temp_c', 'rel_humidity', 'wind_speed_ms', 'hour', 'month',
    'emc_baseline', 'temp_mean_3h', 'rh_mean_3h', 'temp_mean_6h', 'rh_mean_6h',
    'precip_1h', 'precip_3h', 'precip_6h', 'precip_24h', 'hours_since_rain'
]

# Base features list (for models without precipitation)
# Use this if you haven't retrained with precipitation yet
# FEATURES = [
#     'temp_c', 'rel_humidity', 'wind_speed_ms', 'hour', 'month', 
#     'emc_baseline', 'temp_mean_3h', 'rh_mean_3h', 'temp_mean_6h', 'rh_mean_6h'
# ]


# Suppress Herbie regex warnings
warnings.filterwarnings('ignore', message='This pattern is interpreted as a regular expression')

# Set up logging to file in logs folder
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, 'forecastedfiredanger.log')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler (optional, keeps previous behavior)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)


def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[MEM] {note} RSS={mem_mb:.1f} MB")


def calculate_fire_danger(fm, rh, wind_kts):
    """
    Fire Danger Criteria based on ShowMeFire.org:
    Low: FM >= 15%
    Moderate: FM 9-14% WITH (RH < 50% AND Wind >= 10 kts)
    Elevated: FM < 9% WITH (RH < 45% OR Wind >= 10 kts)
    Critical: FM < 9% WITH (RH < 25% AND Wind >= 15 kts)
    Extreme: FM < 7% WITH (RH < 20% AND Wind >= 30 kts)
    """
    # LOW (Fuels are too wet to carry fire effectively)
    if fm >= 15: 
        return 0 
    
    # 5. EXTREME (The most restrictive)
    if fm < 7 and rh < 20 and wind_kts >= 30:
        return 4
    
    # 4. CRITICAL (High)
    if fm < 9 and rh < 25 and wind_kts >= 15:
        return 3
        
    # 3. ELEVATED
    if fm < 9 and (rh < 45 or wind_kts >= 10):
        return 2
        
    # 2. MODERATE
    # Change to AND logic: FM must be low AND weather must be active
    if (9 <= fm < 15) and (rh < 50 and wind_kts >= 10):
        return 1
        
    # 1. LOW (Default if FM is high or conditions aren't met)
    return 0


def estimate_fuel_moisture(relative_humidity, air_temp=None):
    logger.debug(f"Estimating fuel moisture from RH={relative_humidity}, Temp={air_temp}")
    """
    Estimate 10-hour fuel moisture from relative humidity.
    """
    if relative_humidity is None:
        return None
    
    fm_estimate = 3 + 0.25 * relative_humidity
    fm_estimate = np.clip(fm_estimate, 3, 30)
    
    return fm_estimate


def create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi):
    logger.info(f"Creating base map with extent={extent}, size=({pixelw}x{pixelh}), dpi={mapdpi}")
    """Create base map figure and axes."""
    figsize_width = pixelw / mapdpi
    figsize_height = pixelh / mapdpi
    
    fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=mapdpi, facecolor='#E8E8E8')
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)
    
    return fig, ax


def add_boundaries(ax, data_crs, PROJECT_DIR, county_zorder=5, state_zorder=9):
    logger.info("Adding county and state boundaries to map.")
    """Add county and state boundaries to map."""
    counties = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp')
    if counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", 
                      facecolor='none', linewidth=1, zorder=county_zorder)
    
    missouriborder = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", 
                      facecolor='none', linewidth=1.5, zorder=state_zorder)


def add_title_and_branding(fig, title, subtitle, description, RUN_DATE, SCRIPT_DIR):
    logger.info(f"Adding title and branding: {title}")
    """Add title, description, and branding to figure."""
    font_paths = [
        str(SCRIPT_DIR.parent / 'assets/Montserrat/static/Montserrat-Regular.ttf'),
        str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf'),
        str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf')
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Montserrat'
    
    fig.text(0.99, 0.97, title, fontsize=26, fontweight='bold', ha='right', va='top', fontname='Plus Jakarta Sans')
    fig.text(0.99, 0.90, subtitle, fontsize=16, ha='right', va='top', fontname='Montserrat')
    fig.text(0.99, 0.62, description, fontsize=10, ha='right', va='top', linespacing=1.6, fontname='Montserrat')
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight='bold', ha='left', va='bottom', fontname='Montserrat')
    
    # Add logo
    svg_path = str(SCRIPT_DIR.parent / 'assets/LightBackGroundLogo.svg')
    try:
        png_bytes = cairosvg.svg2png(url=svg_path)
        image = mpimg.imread(BytesIO(png_bytes), format='png')
        imagebox = OffsetImage(image, zoom=0.03)
        ab = AnnotationBbox(imagebox, (0.99, 0.01), frameon=False, xycoords='figure fraction', box_alignment=(1, 0))
        plt.gca().add_artist(ab)
    except (ImportError, FileNotFoundError):
        pass

def get_current_fuel_moisture_field(port='8000', target_date=None):
    logger.info(f"Getting current fuel moisture field from RAWS at 7 AM CT (port={port})")
    """
    Get observed fuel moisture from RAWS stations near 7 AM Central Time.
    Uses the new /api/fuel-moisture/morning endpoint for consistent timing.
    This provides the starting point for the forecast.
    
    Args:
        port: API port (default: 8000)
        target_date: Optional date string in YYYY-MM-DD format (default: today)
    """
    try:
        # Build URL for the morning fuel moisture endpoint
        url = f'http://localhost:{port}/api/fuel-moisture/morning'
        params = {}
        if target_date:
            params['date'] = target_date
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('success'):
            logger.warning(f"API returned error: {data}")
            return None
        
        stations = data.get('data', {}).get('stations', [])
        
        fuel_points = []
        for station in stations:
            obs = station.get('observations', {})
            fm_data = obs.get('fuel_moisture')
            
            if fm_data and isinstance(fm_data, dict):
                fm_value = fm_data.get('value')
            elif isinstance(fm_data, (int, float)):
                fm_value = fm_data
            else:
                fm_value = None
            
            if fm_value is not None and fm_value > 0:
                fuel_points.append((
                    station['longitude'], 
                    station['latitude'], 
                    fm_value
                ))
        
        if len(fuel_points) >= 3:
            target_time = data.get('data', {}).get('target_time_formatted', '7 AM CT')
            logger.info(f"Found {len(fuel_points)} RAWS stations with fuel moisture data at {target_time}")
            return fuel_points
        else:
            logger.warning(f"Only {len(fuel_points)} RAWS stations available with fuel moisture, using default")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching fuel moisture data from API: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching RAWS data: {e}")
        return None


def interpolate_current_fm_to_grid(fuel_points, grid_lon_mesh, grid_lat_mesh):
    logger.info(f"Interpolating {len(fuel_points) if fuel_points else 0} fuel points to grid.")
    """
    Interpolate current RAWS observations to forecast grid.
    """
    if not fuel_points:
        return None
        
    fuel_lon = [p[0] for p in fuel_points]
    fuel_lat = [p[1] for p in fuel_points]
    fuel_values = [p[2] for p in fuel_points]
    
    # Use RBF interpolation
    fuel_rbf = Rbf(fuel_lon, fuel_lat, fuel_values, function='multiquadric', smooth=0.01)
    fuel_grid = fuel_rbf(grid_lon_mesh, grid_lat_mesh)
    fuel_grid = gaussian_filter(fuel_grid, sigma=0.7)
    
    return fuel_grid


def estimate_fuel_moisture_with_lag(rh, temp_c, previous_fm, hours_elapsed=1):
    logger.debug(f"Estimating FM with lag: RH={rh}, Temp={temp_c}, PrevFM={previous_fm}, Hours={hours_elapsed}")
    """
    Estimate fuel moisture accounting for time lag (10-hour fuels).
    
    This models the physical process of fuel moisture change rather than
    just using equilibrium moisture content.
    
    Parameters:
    - rh: Relative humidity (%)
    - temp_c: Air temperature (°C)
    - previous_fm: Fuel moisture from previous timestep (%)
    - hours_elapsed: Hours since last timestep (usually 1)
    
    Returns: New fuel moisture (%)
    """
    # Calculate equilibrium moisture content (EMC) based on RH and temperature
    # This is what the fuel "wants" to be at given current conditions
    if rh <= 10:
        emc = 0.03 + 0.2626 * rh - 0.00104 * rh * temp_c
    elif rh <= 50:
        emc = 2.22 - 0.160 * rh + 0.01660 * temp_c
    else:
        emc = 21.06 - 0.4944 * rh + 0.005565 * rh**2 - 0.00063 * rh * temp_c
    
    emc = np.clip(emc, 1, 40)
    
    # Calculate response time (tau) for 10-hour fuels
    # Drying is faster than wetting
    if previous_fm > emc:  # Drying
        tau = 10.0 * np.exp(-0.05 * temp_c)
    else:  # Wetting
        tau = 10.0 * np.exp(-0.05 * temp_c) * 1.5
    
    # Exponential approach to equilibrium
    # alpha represents fraction of way to equilibrium after time elapsed
    alpha = 1 - np.exp(-hours_elapsed / tau)
    
    # Calculate new fuel moisture
    fm_new = previous_fm + alpha * (emc - previous_fm)
    
    return np.clip(fm_new, 1, 40)


def process_forecast_with_observations(ds_full, lon, lat, port='8000'):
    logger.info("Processing forecast with observations-based fuel moisture.")
    """
    Process HRRR forecast with actual fuel moisture observations as starting point.
    
    This replaces the simple RH-based estimation with a physics-based model
    that accounts for:
    1. Current actual fuel moisture (from RAWS)
    2. Time lag in fuel response
    3. Temperature effects on drying/wetting rates
    
    Returns: hourly_fm, hourly_rh, hourly_ws, hourly_temp, hourly_risks
    """
    # Get current fuel moisture observations
    fuel_points = get_current_fuel_moisture_field(port)
    
    # Create grid meshes for interpolation
    if lon.ndim == 1 and lat.ndim == 1:
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(lon, lat)
    else:
        grid_lon_mesh, grid_lat_mesh = lon, lat
    
    # Initialize fuel moisture field from observations
    if fuel_points and len(fuel_points) >= 3:
        initial_fm = interpolate_current_fm_to_grid(fuel_points, grid_lon_mesh, grid_lat_mesh)
        print(f"Initialized from RAWS observations: FM range {np.nanmin(initial_fm):.1f}-{np.nanmax(initial_fm):.1f}%")
    else:
        # Fallback: Use conservative default based on recent weather
        # You could make this more sophisticated by looking at recent RH trends
        initial_fm = np.full_like(grid_lon_mesh, 12.0)
        print("Warning: Using default FM=12% (no RAWS data available)")
    
    # Process each forecast hour
    hourly_fm = []
    hourly_rh = []
    hourly_ws = []
    hourly_temp = []
    hourly_risks = []
    
    # Buffers for rolling means
    temp_history = []
    rh_history = []
    precip_history = []
    
    # Extract precipitation if available
    has_precip = False
    try:
        if 'apcp' in ds_full or 'APCP' in ds_full or 'tp' in ds_full:
            has_precip = True
            logger.info("Precipitation data found in HRRR dataset")
    except:
        pass
    
    now = pd.Timestamp.utcnow()

    # Load Station Indices for Verification
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        indices_df = pd.read_sql("SELECT * FROM station_grid_indices", conn)
        conn.close()
        station_indices = indices_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to load station indices: {e}")
        station_indices = []
    
    for i, time_step in enumerate(ds_full.step):
        ds_hour = ds_full.sel(step=time_step)
        
        # Extract forecast variables
        rh = ds_hour['r2'].values
        temp = ds_hour['t2m'].values - 273.15  # Convert K to C
        u = ds_hour['u10'].values
        v = ds_hour['v10'].values
        
        # Extract precipitation if available
        precip_mm = np.zeros_like(temp)
        if has_precip:
            try:
                if 'apcp' in ds_hour:
                    precip_mm = ds_hour['apcp'].values
                elif 'APCP' in ds_hour:
                    precip_mm = ds_hour['APCP'].values
                elif 'tp' in ds_hour:
                    precip_mm = ds_hour['tp'].values
            except:
                pass
        
        ws_kts = np.sqrt(u**2 + v**2) * 1.94384
        ws_ms = np.sqrt(u**2 + v**2)
        
        # Get time info
        time_step_value = time_step.values
        if isinstance(time_step_value, np.timedelta64):
            hours_ahead = int(time_step_value / np.timedelta64(1, 'h'))
        else:
            hours_ahead = int(time_step_value / 3600000000000)
        forecast_time = now + pd.Timedelta(hours=hours_ahead)
        hour_val = forecast_time.hour
        month_val = forecast_time.month
        
        # Calculate fuel moisture with XGBoost
        print(f"  Predicting Fuel Moisture via XGBoost for hour {i}...")
        fm = predict_fm_grid(temp, rh, ws_ms, hour_val, month_val, temp_history, rh_history, precip_history)
        
        # Update buffers for the next hour
        temp_history.append(temp)
        rh_history.append(rh)
        precip_history.append(precip_mm)
        
        # Keep buffers at max 24 hours to allow for precip_24h calculation
        if len(temp_history) > 24:
            temp_history.pop(0)
            rh_history.pop(0)
            precip_history.pop(0)
        
        # Save hourly values
        hourly_rh.append(rh)
        hourly_temp.append(temp)
        hourly_ws.append(ws_kts)
        hourly_fm.append(fm)
        
        # Calculate fire danger # Import your existing function
        risk = np.zeros_like(rh, dtype=int)
        for ii in range(rh.shape[0]):
            for jj in range(rh.shape[1]):
                risk[ii, jj] = calculate_fire_danger(fm[ii, jj], rh[ii, jj], ws_kts[ii, jj])
        hourly_risks.append(risk)

        # Save verification data
        if station_indices:
            forecast_rows = []
            run_time_str = now.strftime('%Y-%m-%d %H:%M:%S')
            valid_time_str = forecast_time.strftime('%Y-%m-%d %H:%M:%S')
            
            for st in station_indices:
                sx, sy = st['x'], st['y']
                if 0 <= sx < fm.shape[1] and 0 <= sy < fm.shape[0]:
                    forecast_rows.append((
                        st['station_id'],
                        valid_time_str,
                        run_time_str,
                        float(temp[sy, sx]),
                        float(rh[sy, sx]),
                        float(ws_ms[sy, sx]),
                        float(precip_mm[sy, sx]),
                        float(fm[sy, sx])
                    ))
            
            if forecast_rows:
                try:
                    conn = sqlite3.connect(db_path)
                    conn.executemany('''
                        INSERT OR REPLACE INTO station_forecasts 
                        (station_id, valid_time, forecast_run_time, temp_c, rel_humidity, wind_speed_ms, precip_mm, fuel_moisture)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', forecast_rows)
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.error(f"Failed to save station forecasts for {valid_time_str}: {e}")
        
        # Progress indicator
        if i == 0:
            print(f"Hour {i}: FM range {np.nanmin(fm):.1f}-{np.nanmax(fm):.1f}%")
        elif i % 3 == 0:
            print(f"Hour {i}: FM range {np.nanmin(fm):.1f}-{np.nanmax(fm):.1f}%")
    
    return hourly_fm, hourly_rh, hourly_ws, hourly_temp, hourly_risks

def predict_fm_grid(temp_grid, rh_grid, ws_grid, hour, month, t_hist=None, rh_hist=None, precip_hist=None):
    # 1. Flatten the grids into 1D arrays
    shape = temp_grid.shape
    t_flat = temp_grid.flatten()
    rh_flat = rh_grid.flatten()
    ws_flat = ws_grid.flatten()
    
    # 2. Build the DataFrame for the model
    
    # Prepare history including current frame for rolling means
    if t_hist is None: t_hist = []
    if rh_hist is None: rh_hist = []
    
    # Combine past + current for calculation
    curr_t_stack = t_hist + [temp_grid]
    curr_rh_stack = rh_hist + [rh_grid]
    
    # Calculate rolling means
    t_mean_3h = np.mean(curr_t_stack[-3:], axis=0).flatten()
    rh_mean_3h = np.mean(curr_rh_stack[-3:], axis=0).flatten()
    t_mean_6h = np.mean(curr_t_stack[-6:], axis=0).flatten()
    rh_mean_6h = np.mean(curr_rh_stack[-6:], axis=0).flatten()
    
    # Standard Simard (1968) EMC calculation for the model baseline
    emc_baseline = 0.03229 + (0.281073 * rh_flat) - (0.000578 * rh_flat * t_flat)
    
    df = pd.DataFrame({
        'temp_c': t_flat,
        'rel_humidity': rh_flat,
        'wind_speed_ms': ws_flat,
        'hour': hour,
        'month': month,
        'emc_baseline': emc_baseline,
        'temp_mean_3h': t_mean_3h,
        'rh_mean_3h': rh_mean_3h,
        'temp_mean_6h': t_mean_6h,
        'rh_mean_6h': rh_mean_6h
    })
    
    # Add precipitation features (always, to match FEATURES list)
    if precip_hist is not None and len(precip_hist) > 0:
        # Combine past + current for precipitation calculations
        curr_precip_stack = precip_hist
        
        # Calculate rolling sums for precipitation
        precip_1h = curr_precip_stack[-1].flatten() if len(curr_precip_stack) >= 1 else np.zeros(shape).flatten()
        precip_3h = np.sum(curr_precip_stack[-3:], axis=0).flatten() if len(curr_precip_stack) >= 3 else np.zeros(shape).flatten()
        precip_6h = np.sum(curr_precip_stack[-6:], axis=0).flatten() if len(curr_precip_stack) >= 6 else np.zeros(shape).flatten()
        precip_24h = np.sum(curr_precip_stack[-24:], axis=0).flatten() if len(curr_precip_stack) >= 24 else np.zeros(shape).flatten()
        
        # Calculate hours since rain (>0.1mm threshold)
        hours_since_rain = np.zeros(shape).flatten()
        for h in range(len(curr_precip_stack)):
            mask = curr_precip_stack[-(h+1)].flatten() > 0.1
            hours_since_rain[mask & (hours_since_rain == 0)] = h
        # Cap at 24 hours
        hours_since_rain[hours_since_rain == 0] = min(24, len(curr_precip_stack))
    else:
        # No precipitation data - use zeros
        precip_1h = np.zeros(shape).flatten()
        precip_3h = np.zeros(shape).flatten()
        precip_6h = np.zeros(shape).flatten()
        precip_24h = np.zeros(shape).flatten()
        hours_since_rain = np.full(shape, 24).flatten()  # Assume no rain for 24 hours
    
    df['precip_1h'] = precip_1h
    df['precip_3h'] = precip_3h
    df['precip_6h'] = precip_6h
    df['precip_24h'] = precip_24h
    df['hours_since_rain'] = hours_since_rain
    
    # 3. Convert to DMatrix and Predict
    dmat = xgb.DMatrix(df[FEATURES])
    preds = FM_MODEL.predict(dmat)
    
    # 4. Reshape back to the original 2D map
    return preds.reshape(shape)

def generate_complete_forecast():
    logger.info("Starting complete fire weather forecast generation.")
    """
    Generate complete suite of fire weather forecast maps.
    """
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    
    load_dotenv()
    start_time = time.time()
    logger.info("Loaded environment variables.")
    
    # Configuration
    now = pd.Timestamp.utcnow()
    logger.info(f"Current UTC time: {now}")
    
    # Always use the most recent available 12z run
    # HRRR runs are available ~1-2 hours after model time
    # So 12z run is typically available by ~14z (2pm UTC)
    if now.hour < 14:
        logger.info("Using previous day's 12z HRRR run.")
        # If before 14z, use yesterday's 12z run (most recent available)
        RUN_DATE = (now - pd.Timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    else:
        logger.info("Using today's 12z HRRR run.")
        # If after 14z, today's 12z run should be available
        RUN_DATE = now.replace(hour=12, minute=0, second=0, microsecond=0)

    # Ensure RUN_DATE is tz-naive (UTC)
    if RUN_DATE.tzinfo is not None:
        RUN_DATE = RUN_DATE.tz_convert('UTC').tz_localize(None)
    
    FORECAST_HOURS = range(4, 16)
    
    pixelw = 2048
    pixelh = 1152
    mapdpi = 144
    extent = (-95.8, -89.1, 35.8, 40.8)
    
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    
    cache_dir = PROJECT_DIR / 'cache' / 'hrrr'
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")
    
    # Check for any cache file matching the date pattern, regardless of hour
    # This helps if we have 12z data but the script is looking for 13z or vice versa
    date_str = RUN_DATE.strftime('%Y%m%d')
    potential_caches = list(cache_dir.glob(f"hrrr_{date_str}_*z_f04-15.nc"))
    
    if potential_caches:
        # Use the most recent cache file found for this date
        cache_file = sorted(potential_caches)[-1]
        logger.info(f"Found existing cache file: {cache_file}")
    else:
        # Default to the specific run hour we calculated
        cache_file = cache_dir / f"hrrr_{RUN_DATE.strftime('%Y%m%d_%H')}z_f04-15.nc"
        logger.info(f"No cache found matching pattern hrrr_{date_str}_*z_f04-15.nc")
        
        # Debug: List files in directory to help diagnose
        if cache_dir.exists():
            logger.info(f"Listing files in {cache_dir}:")
            files = list(cache_dir.glob("*"))
            if not files:
                logger.info("  (Directory is empty)")
            for f in files:
                logger.info(f"  {f.name}")
        else:
            logger.info(f"  (Directory {cache_dir} does not exist)")
    
    if cache_file.exists():
        logger.info("Loading HRRR data from cache.")
        print(f"Loading cached HRRR data from {cache_file}...")
        try:
            # Remove chunks argument to avoid dask dependency
            ds_full = xr.open_dataset(cache_file, decode_cf=False)
            print("Loaded from cache successfully")
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Deleting corrupted cache and re-downloading...")
            cache_file.unlink()
            # Fall through to download section
    
    if not cache_file.exists():
        logger.info("Downloading HRRR data (not found in cache). This may take a while...")
        print(f"Downloading HRRR data for {RUN_DATE} UTC...")
        FH = FastHerbie(DATES=[RUN_DATE], fxx=list(FORECAST_HOURS), model='hrrr', product='sfc')
        
        # Remove chunks argument here
        ds_rh_temp = FH.xarray(":(TMP|RH):2 m")
        if isinstance(ds_rh_temp, list):
            ds_rh_temp = ds_rh_temp[0]
        
        ds_wind = FH.xarray(":(UGRD|VGRD):10 m")
        if isinstance(ds_wind, list):
            ds_wind = ds_wind[0]
        
        # Download precipitation data
        try:
            logger.info("Downloading precipitation data from HRRR...")
            ds_precip = FH.xarray(":APCP:")
            if isinstance(ds_precip, list):
                ds_precip = ds_precip[0]
            logger.info("Successfully downloaded precipitation data")
            ds_full = ds_rh_temp.merge(ds_wind, compat='override').merge(ds_precip, compat='override')
        except Exception as e:
            logger.warning(f"Could not download precipitation data: {e}")
            ds_full = ds_rh_temp.merge(ds_wind, compat='override')
        
        # Save to cache for future use
        try:
            print(f"Saving to cache: {cache_file}")
            ds_full.load()
            
            # Remove problematic encoding attributes
            for var in ds_full.variables:
                if 'dtype' in ds_full[var].attrs:
                    del ds_full[var].attrs['dtype']
                if 'source' in ds_full[var].attrs:
                    del ds_full[var].attrs['source']
            
            ds_full.to_netcdf(cache_file, engine='netcdf4')
            print(f"Cache saved successfully to {cache_file}")
            print(f"Cache file size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
            print("Continuing without caching...")
    
    # Extract coordinates first
    lon = ds_full['longitude'].values
    lat = ds_full['latitude'].values

    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)

    # Get port from environment
    port = os.getenv('PORT', '8000')

    # Process data with observations-based fuel moisture
    logger.info("Processing forecast data with RAWS observations...")

    # --- Save initialization observation data for ML/verification ---
    # Try to get the current fuel moisture field (RAWS obs)

    # Continue with forecast using ML model
    hourly_fm, hourly_rh, hourly_ws, hourly_temp, hourly_risks = process_forecast_with_observations(
        ds_full, lon, lat, port=port
    )
    
    # --- Extract and process precipitation data ---
    logger.info("Extracting precipitation data from HRRR...")
    try:
        # Try common HRRR precipitation variable names
        precip = None
        if 'tp' in ds_full:
            precip = ds_full['tp'].values
            logger.info("Found 'tp' (total precipitation) variable")
        elif 'apcp' in ds_full:
            precip = ds_full['apcp'].values
            logger.info("Found 'apcp' (accumulated precipitation) variable")
        elif 'APCP' in ds_full:
            precip = ds_full['APCP'].values
            logger.info("Found 'APCP' variable")
        elif 'precipitation' in ds_full:
            precip = ds_full['precipitation'].values
            logger.info("Found 'precipitation' variable")
        else:
            logger.warning(f"No precipitation variable found. Available variables: {list(ds_full.data_vars)}")
            precip = None
    except Exception as e:
        logger.error(f"Error extracting precipitation: {e}")
        precip = None
    
    # Calculate peak/min values FIRST (before cropping)
    combined_risk = np.stack(hourly_risks, axis=0)
    peak_risk = np.nanmax(combined_risk, axis=0)
    
    combined_fm = np.stack(hourly_fm, axis=0)
    min_fuel_moisture = np.nanmin(combined_fm, axis=0)
    
    combined_rh = np.stack(hourly_rh, axis=0)
    min_rh = np.nanmin(combined_rh, axis=0)
    
    combined_ws = np.stack(hourly_ws, axis=0)
    max_wind = np.nanmax(combined_ws, axis=0)
    
    combined_temp = np.stack(hourly_temp, axis=0)
    max_temp = np.nanmax(combined_temp, axis=0)
    
    # Apply smoothing
    peak_risk_smooth = gaussian_filter(peak_risk.astype(float), sigma=1.5)
    min_fuel_moisture_smooth = gaussian_filter(min_fuel_moisture, sigma=1.5)
    min_rh_smooth = gaussian_filter(min_rh, sigma=1.5)
    max_wind_smooth = gaussian_filter(max_wind, sigma=1.5)
    max_temp_smooth = gaussian_filter(max_temp, sigma=1.5)
    
    # Process precipitation if available
    if precip is not None:
        logger.info("Processing precipitation data...")
        # Sum total precipitation across all forecast hours
        if precip.ndim == 3:
            total_precip = np.sum(precip, axis=0)  # Sum over time dimension
        else:
            total_precip = precip
        
        # Convert from kg/m² to inches (1 mm = 1 kg/m², 1 inch = 25.4 mm)
        total_precip_inches = total_precip / 25.4
        
        # Apply smoothing
        total_precip_smooth = gaussian_filter(total_precip_inches, sigma=1.5)
    else:
        total_precip_smooth = None
    
    # Extract and process coordinates
    lon = ds_full['longitude'].values
    lat = ds_full['latitude'].values
    
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
    
    # Crop to Missouri region
    lon_mask = (lon >= extent[0] - 0.5) & (lon <= extent[1] + 0.5)
    lat_mask = (lat >= extent[2] - 0.5) & (lat <= extent[3] + 0.5)
    combined_mask = lon_mask & lat_mask
    
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    lon = lon[rmin:rmax+1, cmin:cmax+1]
    lat = lat[rmin:rmax+1, cmin:cmax+1]
    peak_risk_smooth = peak_risk_smooth[rmin:rmax+1, cmin:cmax+1]
    min_fuel_moisture_smooth = min_fuel_moisture_smooth[rmin:rmax+1, cmin:cmax+1]
    min_rh_smooth = min_rh_smooth[rmin:rmax+1, cmin:cmax+1]
    max_wind_smooth = max_wind_smooth[rmin:rmax+1, cmin:cmax+1]
    max_temp_smooth = max_temp_smooth[rmin:rmax+1, cmin:cmax+1]
    
    # Crop precipitation data too
    if total_precip_smooth is not None:
        total_precip_smooth = total_precip_smooth[rmin:rmax+1, cmin:cmax+1]

    # Mask to Missouri
    missouriborder = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    
    if not missouriborder.empty:
        from shapely.geometry import Point
        from shapely.prepared import prep
        
        missouri_geom = missouriborder.geometry.iloc[0]
        missouri_buffered = missouri_geom.buffer(0.01)
        prepared_geom = prep(missouri_buffered)
        
        points_flat = np.column_stack([lon.ravel(), lat.ravel()])
        mask_flat = np.array([prepared_geom.contains(Point(pt)) for pt in points_flat])
        mask = mask_flat.reshape(lon.shape)
        
        peak_risk_smooth = np.where(mask, peak_risk_smooth, np.nan)
        min_fuel_moisture_smooth = np.where(mask, min_fuel_moisture_smooth, np.nan)
        min_rh_smooth = np.where(mask, min_rh_smooth, np.nan)
        max_wind_smooth = np.where(mask, max_wind_smooth, np.nan)
        max_temp_smooth = np.where(mask, max_temp_smooth, np.nan)
        
        # Mask precipitation too
        if total_precip_smooth is not None:
            total_precip_smooth = np.where(mask, total_precip_smooth, np.nan)

    # ========== MAP 1: PEAK FIRE DANGER ==========
    logger.info("Generating peak fire danger map...")
    colors = ["#90EE90", '#FFED4E', '#FFA500', '#FF0000', '#8B0000']
    labels = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, len(colors))
    
    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
    
    cs = ax.contourf(lon, lat, peak_risk_smooth, transform=data_crs,
                     levels=bins, cmap=cmap, norm=norm, alpha=0.7, zorder=7, antialiased=True)
    ax.contour(lon, lat, peak_risk_smooth, transform=data_crs,
               levels=bins[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
    
    add_boundaries(ax, data_crs, PROJECT_DIR)
    
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Fire Danger Level')
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(labels)
    
    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    
    add_title_and_branding(
        fig, "Missouri Peak Fire Danger Forecast",
        f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
        "Peak Fire Danger Forecast (10:00–21:00 CT)\n\n"
        "Fire Danger Criteria:\n"
        "Low: FM ≥ 15% (Fuels adequately moist)\n"
        "Moderate: FM 9-14% with RH < 60% or Wind ≥ 6 kts\n"
        "Elevated: FM < 9% with RH < 45% or Wind ≥ 10 kts\n"
        "Critical: FM < 9% with RH < 25% & Wind ≥ 15 kts\n"
        "Extreme: FM < 7% with RH < 20% & Wind ≥ 30 kts\n\n"
        "Data Source: HRRR Model Forecast | ML Model | Observations\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastfiredanger.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    del fig, ax, cs, cax, cbar
    gc.collect()
    
    # ========== EXPORT GEOTIFF: PEAK FIRE DANGER ==========
    logger.info("Exporting peak fire danger as GeoTIFF...")
    try:
        geotiff_path = PROJECT_DIR / 'gis/peak_fire_danger.tif'
        geotiff_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get data dimensions
        rows, cols = peak_risk_smooth.shape
        
        # Calculate bounds
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())
        
        # Create transform (maps pixel coordinates to geographic coordinates)
        # Transform expects origin at top-left (lon_min, lat_max)
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)
        
        # Flip data vertically - numpy arrays have origin at bottom-left,
        # but GeoTIFF expects origin at top-left
        data_flipped = np.flipud(peak_risk_smooth)
        
        # Write GeoTIFF using rasterio
        with rasterio.open(
            geotiff_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=1,
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw',
            tiled=True,
            nodata=-9999
        ) as dst:
            # Write the flipped data
            dst.write(data_flipped.astype(np.float32), 1)
            
            # Set metadata
            dst.update_tags(
                DESCRIPTION='Peak Fire Danger Forecast for Missouri',
                MODEL_RUN=RUN_DATE.strftime('%Y-%m-%d %HZ'),
                VALID_TIME=(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d'),
                UNITS='Fire Danger Level (0=Low, 1=Moderate, 2=Elevated, 3=Critical, 4=Extreme)',
                SOURCE='HRRR Model + ML Model + RAWS Observations'
            )
        
        logger.info(f"GeoTIFF saved to {geotiff_path}")
    except Exception as e:
        logger.error(f"Failed to export GeoTIFF: {e}")
    
    # ========== MAP 2: MINIMUM FUEL MOISTURE ==========
    logger.info("Generating minimum fuel moisture map...")
    
    # Create an improved colormap with better contrast in critical ranges
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define colors at key fuel moisture thresholds with strong visual distinction
    # Focus on making 7-15% range very clear
    colors_and_positions = [
        (0.0, '#4D0000'),    # 0% - Very Dark Red/Brown
        (0.15, '#8B0000'),   # 4.5% - Dark Red
        (0.233, '#DC143C'),  # 7% - Crimson (EXTREME threshold)
        (0.267, '#FF4500'),  # 8% - Orange Red
        (0.30, '#FF6347'),   # 9% - Tomato (CRITICAL threshold)
        (0.35, '#FF8C00'),   # 10.5% - Dark Orange
        (0.40, '#FFA500'),   # 12% - Orange
        (0.467, '#FFB347'),  # 14% - Light Orange
        (0.50, '#FFD700'),   # 15% - Gold (ELEVATED threshold)
        (0.567, '#FFED4E'),  # 17% - Yellow
        (0.633, '#F0E68C'),  # 19% - Khaki
        (0.70, '#C8E6C9'),   # 21% - Light Green
        (0.80, '#81C784'),   # 24% - Medium Green
        (0.90, '#4CAF50'),   # 27% - Green
        (1.0, '#2E7D32')     # 30% - Dark Green
    ]
    
    # Unzip into separate lists
    positions = [x[0] for x in colors_and_positions]
    colors = [x[1] for x in colors_and_positions]
    
    # Create custom colormap
    fm_cmap = LinearSegmentedColormap.from_list('fm_enhanced', 
                                                list(zip(positions, colors)), 
                                                N=512)
    
    # Use many levels for smooth gradient
    fm_levels = np.linspace(0, 30, 512)
    
    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)

    cs = ax.contourf(lon, lat, min_fuel_moisture_smooth, transform=data_crs,
                    levels=fm_levels, cmap=fm_cmap, alpha=0.75, zorder=7, 
                    antialiased=True, extend='both')

    # Add single prominent contour line at 9% threshold
    contour_9 = ax.contour(
        lon, lat, min_fuel_moisture_smooth, levels=[9], 
        colors='black', linestyles='dotted', 
        linewidths=2, transform=data_crs, zorder=8, alpha=0.7
    )
    # Label the 9% contour
    ax.clabel(contour_9, inline=True, fontsize=9, fmt='%g%%', inline_spacing=10)

    add_boundaries(ax, data_crs, PROJECT_DIR)

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Fuel Moisture (%)', 
                       ticks=np.arange(0, 32, 3))
    
    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    
    add_title_and_branding(
        fig, "Missouri Minimum Fuel Moisture Forecast",
        f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
        "Minimum 10-Hour Fuel Moisture (10:00–21:00 CT)\n\n"
        "Critical Thresholds:\n"
        "< 7%: Extremely Dry - Extreme fire behavior possible\n"
        "7-9%: Very Dry - Critical fire behavior likely\n"
        "9-15%: Dry - Elevated fire behavior expected\n"
        "15-20%: Moderate - Fire activity possible\n"
        "> 20%: Moist - Fuels less receptive to fire\n\n"
        "Data Source: HRRR Model Forecast | ML Model | Observations\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastfuelmoisture.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    del fig, ax, cs, cax, cbar
    gc.collect()
    
    # ========== MAP 3: MINIMUM RELATIVE HUMIDITY ==========
    logger.info("Generating minimum relative humidity map...")
    rh_colors = ['#8B0000', '#FF0000', '#FFA500', '#FFED4E', '#90EE90', '#228B22']
    rh_levels = [0, 15, 25, 35, 50, 65, 100]
    rh_cmap = ListedColormap(rh_colors)
    rh_norm = BoundaryNorm(rh_levels, len(rh_colors))
    
    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
    
    cs = ax.contourf(lon, lat, min_rh_smooth, transform=data_crs,
                     levels=rh_levels, cmap=rh_cmap, norm=rh_norm, alpha=0.7, zorder=7, antialiased=True)
    ax.contour(lon, lat, min_rh_smooth, transform=data_crs,
               levels=rh_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
    
    add_boundaries(ax, data_crs, PROJECT_DIR)
    
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Relative Humidity (%)')
    cbar.set_ticks([7.5, 20, 30, 42.5, 57.5, 82.5])
    cbar.set_ticklabels(['<15%', '15-25%', '25-35%', '35-50%', '50-65%', '>65%'])
    
    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    
    add_title_and_branding(
        fig, "Missouri Minimum Relative Humidity Forecast",
        f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
        "Minimum Relative Humidity (10:00–21:00 CT)\n\n"
        "Critical Thresholds:\n"
        "< 15%: Extremely Dry - Critical fire conditions\n"
        "15-25%: Very Dry - Red Flag criteria with wind\n"
        "25-35%: Dry - Elevated fire danger possible\n"
        "35-50%: Moderate - Normal fire activity\n"
        "> 50%: Moist - Fire spread limited\n\n"
        "Data Source: HRRR Model Forecast | ML Model | Observations\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastminrh.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    del fig, ax, cs, cax, cbar
    gc.collect()
    
    # ========== MAP 4: MAXIMUM WIND SPEED ==========
    logger.info("Generating maximum wind speed map...")
    wind_colors = ['#90EE90', '#FFED4E', '#FFA500', '#FF6347', '#FF0000', '#8B0000']
    wind_levels = [0, 10, 15, 20, 25, 30, 50]
    wind_cmap = ListedColormap(wind_colors)
    wind_norm = BoundaryNorm(wind_levels, len(wind_colors))
    
    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
    
    cs = ax.contourf(lon, lat, max_wind_smooth, transform=data_crs,
                     levels=wind_levels, cmap=wind_cmap, norm=wind_norm, alpha=0.7, zorder=7, antialiased=True)
    ax.contour(lon, lat, max_wind_smooth, transform=data_crs,
               levels=wind_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
    
    add_boundaries(ax, data_crs, PROJECT_DIR)
    
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Wind Speed (knots)')
    cbar.set_ticks([5, 12.5, 17.5, 22.5, 27.5, 40])
    cbar.set_ticklabels(['<10', '10-15', '15-20', '20-25', '25-30', '>30'])
    
    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    
    add_title_and_branding(
        fig, "Missouri Maximum Wind Speed Forecast",
        f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
        "Maximum Sustained Wind Speed (10:00–21:00 CT)\n\n"
        "Critical Thresholds:\n"
        "< 10 kts: Light winds - Normal fire behavior\n"
        "10-15 kts: Moderate - Increased fire activity\n"
        "15-20 kts: Strong - Red Flag criteria with low RH\n"
        "20-25 kts: Very Strong - Critical fire weather\n"
        "> 25 kts: Extreme - Dangerous fire conditions\n\n"
        "Data Source: HRRR Model Forecast | ML Model | Observations\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastmaxwind.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    del fig, ax, cs, cax, cbar
    gc.collect()
    
    # ========== MAP 5: MAXIMUM TEMPERATURE ==========
    logger.info("Generating maximum temperature map...")
    # Define temperature levels and colors in Fahrenheit
    temp_cmap = plt.cm.turbo  # Or try: plt.cm.RdYlBu_r, plt.cm.jet, plt.cm.plasma

    # Smooth gradient from 0°F to 90°F
    temp_levels_f = np.linspace(0, 90, 50)

    # Convert max_temp_smooth from C to F for plotting
    max_temp_smooth_f = max_temp_smooth * 9/5 + 32

    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)

    cs = ax.contourf(lon, lat, max_temp_smooth_f, transform=data_crs,
                    levels=temp_levels_f, cmap=temp_cmap, alpha=0.7, zorder=7, antialiased=True)

    # Contour lines every 10°F
    contour_levels = np.arange(10, 90, 10)
    ax.contour(lon, lat, max_temp_smooth_f, transform=data_crs,
            levels=contour_levels, colors='black', linewidths=0.3, alpha=0.2, zorder=8)

    add_boundaries(ax, data_crs, PROJECT_DIR)

    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Temperature (°F)')
    cbar.set_ticks([0, 10, 20, 32, 50, 70, 90])
    
    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    
    add_title_and_branding(
        fig, "Missouri Maximum Temperature Forecast",
        f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
        "Maximum Temperature (10:00–21:00 CT)\n\n"
        "Temperature influences fire behavior:\n"
        "Higher temperatures increase fuel dryness\n"
        "and fire spread rates.\n\n"
        "Combined with low humidity and wind,\n"
        "high temperatures create dangerous\n"
        "fire weather conditions.\n\n"
        "Data Source: HRRR Model Forecast | ML Model | Observations\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastmaxtemp.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    del fig, ax, cs, cax, cbar
    gc.collect()
    
    # ========== MAP 6: RAINFALL/PRECIPITATION ==========
    if total_precip_smooth is not None:
        logger.info("Generating rainfall forecast map...")
        try:
            # Use a blue colormap for precipitation
            rain_colors = ['#FFFFFF', '#E0F0FF', '#B0D4FF', '#6AB4FF', '#0080FF', '#0050D0', '#003080']
            rain_levels = [0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
            rain_cmap = ListedColormap(rain_colors)
            rain_norm = BoundaryNorm(rain_levels, len(rain_colors))
            
            fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
            
            # Plot precipitation
            cs = ax.contourf(lon, lat, total_precip_smooth, 
                           levels=rain_levels, cmap=rain_cmap, norm=rain_norm,
                           transform=data_crs, alpha=0.8, zorder=7, antialiased=True)
            
            # Add contour lines
            ax.contour(lon, lat, total_precip_smooth, transform=data_crs,
                      levels=rain_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
            
            add_boundaries(ax, data_crs, PROJECT_DIR)
            
            cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
            cbar = plt.colorbar(cs, cax=cax, label='Total Precipitation (inches)')
            cbar.set_ticks(rain_levels)
            cbar.set_ticklabels([f'{x:.2f}"' if x < 1 else f'{x:.1f}"' for x in rain_levels])
            
            ax.set_anchor('W')
            plt.subplots_adjust(left=0.05)
            
            add_title_and_branding(
                fig, "Missouri Forecast Precipitation",
                f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
                "Total Precipitation Forecast (10:00–21:00 CT)\n\n"
                "Accumulated precipitation from\n"
                f"{RUN_DATE.strftime('%Hz %b %d')} through forecast hour 15.\n\n"
                "Precipitation reduces fire danger by:\n"
                "• Increasing fuel moisture\n"
                "• Raising relative humidity\n"
                "• Potentially preventing ignition\n\n"
                "> 0.5\": Significant fire danger reduction\n"
                "> 1.0\": Major impact on fire activity\n\n"
                "Data Source: HRRR Model Forecast\n"
                "For More Info, Visit ShowMeFire.org",
                RUN_DATE, SCRIPT_DIR
            )
            
            fig.savefig(PROJECT_DIR / 'images/mo-forecastrainfall.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
            plt.close(fig)
            logger.info(f"Saved rainfall forecast map")
            
            del fig, ax, cs, cax, cbar
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error generating rainfall map: {e}")
    else:
        logger.info("Skipping rainfall map - no precipitation data available")

    # ========== REGIONAL MAPS ==========
    logger.info("Generating Regional specific maps...")
    
    regions_dir = PROJECT_DIR / 'maps/shapefiles/SEMA Regions'
    # Find all Region*.shp files (e.g., RegionA.shp, RegionD.shp, RegionI.shp)
    region_shapes = sorted(regions_dir.glob('Region*.shp'))
    
    # List to keep track of ALL generated map files for logging/upload
    all_generated_maps = [
        'mo-forecastfiredanger.png',
        'mo-forecastfuelmoisture.png',
        'mo-forecastminrh.png',
        'mo-forecastmaxwind.png',
        'mo-forecastmaxtemp.png',
    ]
    if total_precip_smooth is not None:
        all_generated_maps.append('mo-forecastrainfall.png')
    
    region_config = {
        'regg': {'zoom_factor': 1.5, 'h_shift': 0.6, 'v_shift': 0.05},
    }
    
    # Default values if region not specified in config
    default_config = {
        'zoom_factor': 1.15,  # Zoom out by 15%
        'h_shift': 0.35,      # Horizontal shift (0-1, proportion of excess width)
        'v_shift': 0.05,      # Vertical shift (proportion of height to move up)
    }
    
    if not region_shapes:
        logger.warning(f"No region shapefiles found in {regions_dir}")
    else:
        logger.info(f"Found {len(region_shapes)} region(s) to process: {[r.stem for r in region_shapes]}")
        
    for region_path in region_shapes:
        region_filename = region_path.stem  # e.g. "RegionD" or "Region_D"
        # Convert "RegionD" -> "Region D"
        region_display_name = region_filename.replace("Region", "Region ").replace("_", " ").strip()
        # Convert "RegionD" -> "regd"
        region_code = region_filename.lower().replace("region", "reg").replace("_", "")
        
        logger.info(f"Processing {region_display_name} ({region_code})...")
        
        try:
            if not region_path.exists():
                logger.warning(f"Shapefile not found: {region_path}")
                continue
                
            reg_gdf = gpd.read_file(region_path)
            if reg_gdf.crs != data_crs.proj4_init:
                reg_gdf = reg_gdf.to_crs(data_crs.proj4_init)
            
            # --- 1. Prepare Boundaries ---
            # Pre-load and clip counties to current Region so lines only show INSIDE the region
            counties = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp')
            if counties.crs != data_crs.proj4_init:
                counties = counties.to_crs(data_crs.proj4_init)

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                counties_clipped = gpd.clip(counties, reg_gdf)

            # --- 2. Prepare Masks ---
            from shapely.prepared import prep
            from shapely.geometry import Point, box
            
            reg_geom = reg_gdf.geometry.iloc[0]
            
            # Buffer the geometry for the DATA mask (approx 3 miles)
            # This ensures data extends slightly PAST the border, eliminating white gaps
            reg_buffered = reg_geom.buffer(0.04)
            reg_prep = prep(reg_buffered)
            
            points_flat = np.column_stack([lon.ravel(), lat.ravel()])
            reg_mask_flat = np.array([reg_prep.contains(Point(pt)) for pt in points_flat])
            reg_mask = reg_mask_flat.reshape(lon.shape)
            
            # Get custom configuration for this region, or use defaults
            config = region_config.get(region_code, default_config)
            zoom_factor = config.get('zoom_factor', default_config['zoom_factor'])
            h_shift_factor = config.get('h_shift', default_config['h_shift'])
            v_shift_factor = config.get('v_shift', default_config['v_shift'])
            
            logger.info(f"{region_display_name} using: zoom={zoom_factor}, h_shift={h_shift_factor}, v_shift={v_shift_factor}")
            
            # Get extent and adjust for 16:9 aspect ratio to fill frame
            rb = reg_gdf.total_bounds
            minx, miny, maxx, maxy = rb
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            
            # Start with tight bounds plus small buffer
            buffer_deg = 0.1
            width_deg = (maxx - minx) + buffer_deg
            height_deg = (maxy - miny) + buffer_deg
            
            # ZOOM OUT: Scale up the extent dimensions to make the map appear smaller
            # Use custom zoom factor for this region
            width_deg *= zoom_factor
            height_deg *= zoom_factor
            
            # Calculate aspect ratios
            target_ar = pixelw / pixelh
            avg_lat_rad = np.radians(cy)
            lon_scale = np.cos(avg_lat_rad)
            current_ar = (width_deg * lon_scale) / height_deg
            
            if current_ar < target_ar:
                # Region is too narrow/tall - Expand width
                new_width_deg = (height_deg * target_ar) / lon_scale
                
                # SHIFT LOGIC (Horizontal): Move region to the left (by shifting view center RIGHT)
                # Use custom horizontal shift factor for this region
                excess_width = new_width_deg - width_deg
                shift_offset_x = excess_width * h_shift_factor
                new_cx = cx + shift_offset_x
                
                # SHIFT LOGIC (Vertical): Move region UP (by shifting view center DOWN)
                # This clears the branding logo at the bottom
                # Use custom vertical shift factor for this region
                shift_offset_y = height_deg * v_shift_factor
                new_cy = cy - shift_offset_y
                
                reg_extent = (new_cx - new_width_deg/2, new_cx + new_width_deg/2, 
                              new_cy - height_deg/2, new_cy + height_deg/2)
            else:
                # Region is too wide
                new_height_deg = (width_deg * lon_scale) / target_ar
                
                # Vertical Shift: Move region UP
                # Use custom vertical shift factor for this region
                shift_offset_y = new_height_deg * v_shift_factor
                new_cy = cy - shift_offset_y
                
                reg_extent = (cx - width_deg/2, cx + width_deg/2, 
                              new_cy - new_height_deg/2, new_cy + new_height_deg/2)
            
            # Apply mask to data
            reg_risk = np.where(reg_mask, peak_risk_smooth, np.nan)
            reg_fm = np.where(reg_mask, min_fuel_moisture_smooth, np.nan)
            reg_rh = np.where(reg_mask, min_rh_smooth, np.nan)
            reg_wind = np.where(reg_mask, max_wind_smooth, np.nan)
            reg_temp = np.where(reg_mask, max_temp_smooth, np.nan)
            
            # Helper function for Region borders and masking
            def add_region_overlay(ax, extent):
                # 1. Add clipped county lines (internal only)
                ax.add_geometries(counties_clipped.geometry, crs=data_crs, edgecolor="#B6B6B6", 
                                facecolor='none', linewidth=1, zorder=5)
                
                # 2. Create "Inverted Mask" to hide spilled data outside border
                # Create a box larger than the map extent
                extent_box = box(extent[0]-1, extent[2]-1, extent[1]+1, extent[3]+1)
                inverted_mask = extent_box.difference(reg_geom)
                
                # Draw mask with background color to crop data cleanly at the line
                ax.add_geometries([inverted_mask], crs=data_crs, facecolor='#E8E8E8', edgecolor='none', zorder=9)
                
                # 3. Draw heavy region boundary on top
                ax.add_geometries([reg_geom], crs=data_crs, edgecolor="black", facecolor='none', linewidth=2, zorder=10)

            # 1. Regional Peak Fire Danger
            fig, ax = create_base_map(reg_extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
            colors = ["#90EE90", '#FFED4E', '#FFA500', '#FF0000', '#8B0000']
            labels = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
            bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(bins, len(colors))
            cs = ax.contourf(lon, lat, reg_risk, transform=data_crs, levels=bins, cmap=cmap, norm=norm, alpha=0.7, zorder=7, antialiased=True)
            ax.contour(lon, lat, reg_risk, transform=data_crs, levels=bins[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
            
            add_region_overlay(ax, reg_extent)
            
            cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
            cbar = plt.colorbar(cs, cax=cax, label='Fire Danger Level')
            cbar.set_ticks([0, 1, 2, 3, 4])
            cbar.set_ticklabels(labels)
            ax.set_anchor('W')
            plt.subplots_adjust(left=0.05)

            add_title_and_branding(
                fig, f"Missouri - {region_display_name} Peak Fire Danger", 
                f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
                "Peak Fire Danger Forecast (10:00–21:00 CT)\n\n"
                "Fire Danger Criteria:\n"
                "Low: FM ≥ 15% (Fuels adequately moist)\n"
                "Moderate: FM 9-14% with RH < 60% or Wind ≥ 6 kts\n"
                "Elevated: FM < 9% with RH < 45% or Wind ≥ 10 kts\n"
                "Critical: FM < 9% with RH < 25% & Wind ≥ 15 kts\n"
                "Extreme: FM < 7% with RH < 20% & Wind ≥ 30 kts\n\n"
                "Data Source: HRRR Model Forecast | ML Model | Observations\n"
                "For More Info, Visit ShowMeFire.org", 
                RUN_DATE, SCRIPT_DIR
            )
            filename = f'{region_code}-forecastfiredanger.png'
            fig.savefig(PROJECT_DIR / 'images' / filename, dpi=mapdpi, bbox_inches=None, pad_inches=0)
            all_generated_maps.append(filename)
            plt.close(fig)
            del fig, ax, cs, cax, cbar
            gc.collect()
        
            # 2. Fuel Moisture
            fig, ax = create_base_map(reg_extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
            cs = ax.contourf(lon, lat, reg_fm, transform=data_crs, levels=fm_levels, cmap=fm_cmap, alpha=0.75, zorder=7, antialiased=True, extend='both')
            contour_9 = ax.contour(lon, lat, reg_fm, levels=[9], colors='black', linestyles='dotted', linewidths=2, transform=data_crs, zorder=8, alpha=0.7)
            ax.clabel(contour_9, inline=True, fontsize=9, fmt='%g%%', inline_spacing=10)
            
            add_region_overlay(ax, reg_extent)
            
            cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
            cbar = plt.colorbar(cs, cax=cax, label='Fuel Moisture (%)', ticks=np.arange(0, 32, 3))
            ax.set_anchor('W')
            plt.subplots_adjust(left=0.05)
            
            add_title_and_branding(
                fig, f"Missouri - {region_display_name} Fuel Moisture", 
                f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
                "Minimum 10-Hour Fuel Moisture (10:00–21:00 CT)\n\n"
                "Critical Thresholds:\n"
                "< 7%: Extremely Dry - Extreme fire behavior possible\n"
                "7-9%: Very Dry - Critical fire behavior likely\n"
                "9-15%: Dry - Elevated fire behavior expected\n"
                "15-20%: Moderate - Fire activity possible\n"
                "> 20%: Moist - Fuels less receptive to fire\n\n"
                "Data Source: HRRR Model Forecast | ML Model | Observations\n"
                "For More Info, Visit ShowMeFire.org",
                RUN_DATE, SCRIPT_DIR
            )
            filename = f'{region_code}-forecastfuelmoisture.png'
            fig.savefig(PROJECT_DIR / 'images' / filename, dpi=mapdpi, bbox_inches=None, pad_inches=0)
            all_generated_maps.append(filename)
            plt.close(fig)
            del fig, ax, cs, cax, cbar
            gc.collect()
        
            # 3. Minimum RH
            fig, ax = create_base_map(reg_extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
            rh_colors = ['#8B0000', '#FF0000', '#FFA500', '#FFED4E', '#90EE90', '#228B22']
            rh_levels = [0, 15, 25, 35, 50, 65, 100]
            rh_cmap = ListedColormap(rh_colors)
            rh_norm = BoundaryNorm(rh_levels, len(rh_colors))
            cs = ax.contourf(lon, lat, reg_rh, transform=data_crs, levels=rh_levels, cmap=rh_cmap, norm=rh_norm, alpha=0.7, zorder=7, antialiased=True)
            ax.contour(lon, lat, reg_rh, transform=data_crs, levels=rh_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
            
            add_region_overlay(ax, reg_extent)
            
            cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
            cbar = plt.colorbar(cs, cax=cax, label='Relative Humidity (%)')
            cbar.set_ticks([7.5, 20, 30, 42.5, 57.5, 82.5])
            cbar.set_ticklabels(['<15%', '15-25%', '25-35%', '35-50%', '50-65%', '>65%'])
            ax.set_anchor('W')
            plt.subplots_adjust(left=0.05)
            
            add_title_and_branding(
                fig, f"Missouri - {region_display_name} Minimum RH", 
                f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
                "Minimum Relative Humidity (10:00–21:00 CT)\n\n"
                "Critical Thresholds:\n"
                "< 15%: Extremely Dry - Critical fire conditions\n"
                "15-25%: Very Dry - Red Flag criteria with wind\n"
                "25-35%: Dry - Elevated fire danger possible\n"
                "35-50%: Moderate - Normal fire activity\n"
                "> 50%: Moist - Fire spread limited\n\n"
                "Data Source: HRRR Model Forecast | ML Model | Observations\n"
                "For More Info, Visit ShowMeFire.org",
                RUN_DATE, SCRIPT_DIR
            )
            filename = f'{region_code}-forecastminrh.png'
            fig.savefig(PROJECT_DIR / 'images' / filename, dpi=mapdpi, bbox_inches=None, pad_inches=0)
            all_generated_maps.append(filename)
            plt.close(fig)
            del fig, ax, cs, cax, cbar
            gc.collect()
        
            # 4. Maximum Wind
            fig, ax = create_base_map(reg_extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
            wind_colors = ['#90EE90', '#FFED4E', '#FFA500', '#FF6347', '#FF0000', '#8B0000']
            wind_levels = [0, 10, 15, 20, 25, 30, 50]
            wind_cmap = ListedColormap(wind_colors)
            wind_norm = BoundaryNorm(wind_levels, len(wind_colors))
            cs = ax.contourf(lon, lat, reg_wind, transform=data_crs, levels=wind_levels, cmap=wind_cmap, norm=wind_norm, alpha=0.7, zorder=7, antialiased=True)
            ax.contour(lon, lat, reg_wind, transform=data_crs, levels=wind_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
            
            add_region_overlay(ax, reg_extent)
            
            cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
            cbar = plt.colorbar(cs, cax=cax, label='Wind Speed (knots)')
            cbar.set_ticks([5, 12.5, 17.5, 22.5, 27.5, 40])
            cbar.set_ticklabels(['<10', '10-15', '15-20', '20-25', '25-30', '>30'])
            ax.set_anchor('W')
            plt.subplots_adjust(left=0.05)
            
            add_title_and_branding(
                fig, f"Missouri - {region_display_name} Maximum Wind", 
                f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
                "Maximum Sustained Wind Speed (10:00–21:00 CT)\n\n"
                "Critical Thresholds:\n"
                "< 10 kts: Light winds - Normal fire behavior\n"
                "10-15 kts: Moderate - Increased fire activity\n"
                "15-20 kts: Strong - Red Flag criteria with low RH\n"
                "20-25 kts: Very Strong - Critical fire weather\n"
                "> 25 kts: Extreme - Dangerous fire conditions\n\n"
                "Data Source: HRRR Model Forecast | ML Model | Observations\n"
                "For More Info, Visit ShowMeFire.org",
                RUN_DATE, SCRIPT_DIR
            )
            filename = f'{region_code}-forecastmaxwind.png'
            fig.savefig(PROJECT_DIR / 'images' / filename, dpi=mapdpi, bbox_inches=None, pad_inches=0)
            all_generated_maps.append(filename)
            plt.close(fig)
            del fig, ax, cs, cax, cbar
            gc.collect()
        
            # 5. Maximum Temperature
            fig, ax = create_base_map(reg_extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
            max_temp_smooth_f = reg_temp * 9/5 + 32
            temp_cmap = plt.cm.turbo
            temp_levels_f = np.linspace(0, 90, 50)
            cs = ax.contourf(lon, lat, max_temp_smooth_f, transform=data_crs, levels=temp_levels_f, cmap=temp_cmap, alpha=0.7, zorder=7, antialiased=True)
            contour_levels = np.arange(10, 90, 10)
            ax.contour(lon, lat, max_temp_smooth_f, transform=data_crs, levels=contour_levels, colors='black', linewidths=0.3, alpha=0.2, zorder=8)
            
            add_region_overlay(ax, reg_extent)
            
            cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
            cbar = plt.colorbar(cs, cax=cax, label='Temperature (°F)')
            cbar.set_ticks([0, 10, 20, 32, 50, 70, 90])
            ax.set_anchor('W')
            plt.subplots_adjust(left=0.05)
            
            add_title_and_branding(
                fig, f"Missouri - {region_display_name} Max Temp", 
                f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
                "Maximum Temperature (10:00–21:00 CT)\n\n"
                "Temperature influences fire behavior:\n"
                "Higher temperatures increase fuel dryness\n"
                "and fire spread rates.\n\n"
                "Combined with low humidity and wind,\n"
                "high temperatures create dangerous\n"
                "fire weather conditions.\n\n"
                "Data Source: HRRR Model Forecast | ML Model | Observations\n"
                "For More Info, Visit ShowMeFire.org",
                RUN_DATE, SCRIPT_DIR
            )
            filename = f'{region_code}-forecastmaxtemp.png'
            fig.savefig(PROJECT_DIR / 'images' / filename, dpi=mapdpi, bbox_inches=None, pad_inches=0)
            all_generated_maps.append(filename)
            plt.close(fig)
            del fig, ax, cs, cax, cbar
            gc.collect()
            
            # 6. Rainfall/Precipitation
            if total_precip_smooth is not None:
                reg_precip = np.where(reg_mask, total_precip_smooth, np.nan)
                fig, ax = create_base_map(reg_extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
                rain_colors = ['#FFFFFF', '#E0F0FF', '#B0D4FF', '#6AB4FF', '#0080FF', '#0050D0', '#003080']
                rain_levels = [0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
                rain_cmap = ListedColormap(rain_colors)
                rain_norm = BoundaryNorm(rain_levels, len(rain_colors))
                cs = ax.contourf(lon, lat, reg_precip, levels=rain_levels, cmap=rain_cmap, norm=rain_norm, transform=data_crs, alpha=0.8, zorder=7, antialiased=True)
                ax.contour(lon, lat, reg_precip, transform=data_crs, levels=rain_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
                
                add_region_overlay(ax, reg_extent)
                
                cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
                cbar = plt.colorbar(cs, cax=cax, label='Total Precipitation (inches)')
                cbar.set_ticks(rain_levels)
                cbar.set_ticklabels([f'{x:.2f}"' if x < 1 else f'{x:.1f}"' for x in rain_levels])
                ax.set_anchor('W')
                plt.subplots_adjust(left=0.05)
                
                add_title_and_branding(
                    fig, f"Missouri - {region_display_name} Precipitation", 
                    f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
                    "Total Precipitation Forecast (10:00–21:00 CT)\n\n"
                    "Accumulated precipitation from\n"
                    f"{RUN_DATE.strftime('%Hz %b %d')} through forecast hour 15.\n\n"
                    "Precipitation reduces fire danger by:\n"
                    "• Increasing fuel moisture\n"
                    "• Raising relative humidity\n"
                    "• Potentially preventing ignition\n\n"
                    "> 0.5\": Significant fire danger reduction\n"
                    "> 1.0\": Major impact on fire activity\n\n"
                    "Data Source: HRRR Model Forecast\n"
                    "For More Info, Visit ShowMeFire.org",
                    RUN_DATE, SCRIPT_DIR
                )
                filename = f'{region_code}-forecastrainfall.png'
                fig.savefig(PROJECT_DIR / 'images' / filename, dpi=mapdpi, bbox_inches=None, pad_inches=0)
                all_generated_maps.append(filename)
                plt.close(fig)
                del fig, ax, cs, cax, cbar
                gc.collect()

            logger.info(f"{region_display_name} maps generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating maps for {region_display_name} from {region_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    runtime_sec = time.time() - start_time

    print(f"\nAll forecast maps generated successfully!")
    for map_file in all_generated_maps:
        print(f"  images/{map_file}")
    print(f"Script runtime: {runtime_sec:.2f} seconds")

    # Log results
    logging.basicConfig(filename='logs/forecastfiredanger.log', level=logging.INFO)
    logging.info(f"All forecast maps updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
    logging.info(f"Script runtime: {runtime_sec:.2f} seconds")

    # Update status file
    status_file = PROJECT_DIR / 'status.json'
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
        except json.JSONDecodeError:
            status = {}
    else:
        status = {}

    status['ForecastFireDanger'] = {
        'last_update': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT'),
        'model_run': RUN_DATE.strftime('%Y-%m-%d %HZ'),
        'status': 'updated',
        'runtime_sec': round(runtime_sec, 2),
        'maps_generated': all_generated_maps,
        'log': [
            f"All forecast maps updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}",
            f"Script runtime: {runtime_sec:.2f} seconds"
        ]
    }

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=4)
    
    # Upload to CDN if enabled (default: true if not specified)
    upload_forecast = os.getenv('uploadForecast', 'true').lower() == 'true'
    
    if upload_forecast:
        logger.info("Uploading forecast images to CDN...")
        try:
            # Add scripts directory to path
            scripts_dir = PROJECT_DIR / 'scripts'
            
            # Check if upload_cdn.py exists
            upload_cdn_path = scripts_dir / 'upload_cdn.py'
            if not upload_cdn_path.exists():
                logger.warning(f"upload_cdn.py not found at {upload_cdn_path}. Skipping CDN upload.")
            else:
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))
                
                # Import the upload function
                import upload_cdn
                
                # Upload all generated forecast images dynamically
                forecast_files = [PROJECT_DIR / 'images' / f for f in all_generated_maps]
                
                upload_cdn.run_upload(files_to_upload=forecast_files)
                logger.info("✓ Forecast images uploaded to CDN successfully")
        except ImportError as e:
            logger.warning(f"Could not import dependencies for CDN upload (missing boto3?): {e}. Skipping CDN upload.")
        except Exception as e:
            logger.error(f"Error uploading to CDN: {e}")
    else:
        logger.info("CDN upload disabled (uploadForecast=false in .env)")

    return peak_risk_smooth

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_ml_model(model_path="models/fuel_moisture_model_latest.pkl"):
    """
    Load the trained ML model.
    
    Returns: model_dict or None if model doesn't exist
    """
    model_file = Path(model_path)
    
    if not model_file.exists():
        print(f"ML model not found at {model_path}")
        print("Falling back to physics-based estimation")
        return None
    
    try:
        with open(model_file, 'rb') as f:
            model_dict = pickle.load(f)
        print(f"✓ Loaded ML model from {model_path}")
        print(f"  Test MAE: {model_dict['test_metrics']['mae']:.3f}%")
        return model_dict
    except Exception as e:
        print(f"Error loading ML model: {e}")
        print("Falling back to physics-based estimation")
        return None


def calculate_vpd(temp_c, rh):
    """Calculate Vapor Pressure Deficit (VPD) in kPa."""
    # Saturation vapor pressure (kPa)
    svp = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    # Actual vapor pressure
    avp = svp * (rh / 100)
    # VPD
    vpd = svp - avp
    return vpd


def calculate_nelson_emc(rh, temp_c):
    """Calculate equilibrium moisture content using Nelson's equations."""
    if rh <= 10:
        emc = 0.03 + 0.2626 * rh - 0.00104 * rh * temp_c
    elif rh <= 50:
        emc = 2.22 - 0.160 * rh + 0.01660 * temp_c
    else:
        emc = 21.06 - 0.4944 * rh + 0.005565 * rh**2 - 0.00063 * rh * temp_c
    
    return np.clip(emc, 1, 40)


def prepare_ml_features(rh, temp_c, wind_kts, solar, precip, prev_fm, prev_rh, prev_temp,
                       hour, day_of_year, month, latitude, longitude, elevation):
    """
    Prepare features for ML model prediction.
    Must match the exact features used during training.
    
    Returns: DataFrame with features in correct order
    """
    # Calculate derived features
    temp_f = temp_c * 9/5 + 32
    emc_simple = 3 + 0.25 * rh
    emc_nelson = calculate_nelson_emc(rh, temp_c)
    vpd = calculate_vpd(temp_c, rh)
    wind_temp_interaction = wind_kts * temp_c
    
    # Create feature dictionary
    features = {
        'rh': rh,
        'temp': temp_f,
        'temp_c': temp_c,
        'wind': wind_kts,
        'solar': solar if solar is not None else 0,
        'precip': precip if precip is not None else 0,
        'prev_rh': prev_rh,
        'prev_temp': prev_temp,
        'prev_fm': prev_fm,
        'rh_3h_avg': prev_rh,  # Simplified - would need 3 hours of history
        'temp_3h_avg': prev_temp,  # Simplified
        'emc_simple': emc_simple,
        'emc_nelson': emc_nelson,
        'vpd': vpd,
        'wind_temp_interaction': wind_temp_interaction,
        'hour': hour,
        'day_of_year': day_of_year,
        'month': month,
        'latitude': latitude,
        'longitude': longitude,
        'elevation': elevation
    }
    
    return features


def predict_with_ml_model(model_dict, features_dict):
    """
    Use ML model to predict fuel moisture.
    
    Args:
        model_dict: Loaded model dictionary
        features_dict: Dictionary of features
    
    Returns: Predicted fuel moisture (%)
    """
    # Create DataFrame with single row
    df = pd.DataFrame([features_dict])
    
    # Ensure features are in correct order (same as training)
    feature_names = model_dict['feature_names']
    df = df[feature_names]
    
    # Scale features
    scaler = model_dict['scaler']
    features_scaled = scaler.transform(df)
    
    # Predict
    model = model_dict['model']
    prediction = model.predict(features_scaled)[0]
    
    return np.clip(prediction, 1, 40)


def process_forecast_with_ml_model(ds_full, lon, lat, port='8000', ml_model_path="models/fuel_moisture_model_latest.pkl"):
    """
    Process HRRR forecast using ML model for fuel moisture prediction.
    Falls back to physics-based model if ML model unavailable.
    
    Returns: hourly_fm, hourly_rh, hourly_ws, hourly_temp, hourly_risks
    """
    
    
    # Try to load ML model
    model_dict = load_ml_model(ml_model_path)
    use_ml = model_dict is not None
    
    if use_ml:
        print("Using ML model for fuel moisture prediction")
    else:
        print("Using physics-based model for fuel moisture prediction")
    
    # Get current fuel moisture observations
    fuel_points = get_current_fuel_moisture_field(port)
    
    # Create grid meshes for interpolation
    if lon.ndim == 1 and lat.ndim == 1:
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(lon, lat)
    else:
        grid_lon_mesh, grid_lat_mesh = lon, lat
    
    # Initialize fuel moisture field from observations
    if fuel_points and len(fuel_points) >= 3:
        initial_fm = interpolate_current_fm_to_grid(fuel_points, grid_lon_mesh, grid_lat_mesh)
        print(f"Initialized from RAWS observations: FM range {np.nanmin(initial_fm):.1f}-{np.nanmax(initial_fm):.1f}%")
    else:
        initial_fm = np.full_like(grid_lon_mesh, 12.0)
        print("Warning: Using default FM=12% (no RAWS data available)")
    
    # Get current time info for features
    now = pd.Timestamp.utcnow()
    
    # Process each forecast hour
    hourly_fm = []
    hourly_rh = []
    hourly_ws = []
    hourly_temp = []
    hourly_risks = []
    
    previous_fm = initial_fm.copy()
    previous_rh = None
    previous_temp = None
    
    print(f"\nProcessing {len(ds_full.step)} forecast hours...")
    
    for i, time_step in enumerate(ds_full.step):
        print(f"\n=== Hour {i+1}/{len(ds_full.step)} ===")
        ds_hour = ds_full.sel(step=time_step)
        
        # Extract forecast variables
        print("  Extracting forecast variables...")
        rh = ds_hour['r2'].values
        temp = ds_hour['t2m'].values - 273.15  # Convert K to C
        u = ds_hour['u10'].values
        v = ds_hour['v10'].values
        
        ws_kts = np.sqrt(u**2 + v**2) * 1.94384
        
        # Get time features for this forecast hour
        # Convert time_step to hours - handle both timedelta64 and int64 cases
        time_step_value = time_step.values
        if isinstance(time_step_value, np.timedelta64):
            hours_ahead = int(time_step_value / np.timedelta64(1, 'h'))
        else:
            # If it's an int64, it's likely nanoseconds
            hours_ahead = int(time_step_value / 3600000000000)  # nanoseconds to hours
        forecast_time = now + pd.Timedelta(hours=hours_ahead)
        hour = forecast_time.hour
        day_of_year = forecast_time.dayofyear
        month = forecast_time.month
        
        # Calculate fuel moisture
        fm = np.zeros_like(rh)
        
        if use_ml:
            # Vectorized ML prediction - much faster than loop
            print(f"  Processing {rh.size} grid points with ML model...")
            
            # Flatten arrays for batch processing
            n_points = rh.size
            rh_flat = rh.ravel()
            temp_flat = temp.ravel()
            ws_flat = ws_kts.ravel()
            prev_fm_flat = previous_fm.ravel()
            prev_rh_flat = (previous_rh.ravel() if previous_rh is not None else rh_flat)
            prev_temp_flat = (previous_temp.ravel() if previous_temp is not None else temp_flat)
            grid_lat_flat = grid_lat_mesh.ravel()
            grid_lon_flat = grid_lon_mesh.ravel();
            
            # Calculate derived features (vectorized)
            print("  Calculating features...")
            temp_f = temp_flat * 9/5 + 32
            emc_simple = 3 + 0.25 * rh_flat
            emc_nelson = np.array([calculate_nelson_emc(rh_flat[k], temp_flat[k]) for k in range(n_points)])
            vpd = calculate_vpd(temp_flat, rh_flat)
            wind_temp_interaction = ws_flat * temp_flat
            
            # Build feature DataFrame for all points at once
            print("  Building feature matrix...")
            features_df = pd.DataFrame({
                'rh': rh_flat,
                'temp': temp_f,
                'temp_c': temp_flat,
                'wind': ws_flat,
                'solar': 0,
                'precip': 0,
                'prev_rh': prev_rh_flat,
                'prev_temp': prev_temp_flat,
                'prev_fm': prev_fm_flat,
                'rh_3h_avg': prev_rh_flat,
                'temp_3h_avg': prev_temp_flat,
                'emc_simple': emc_simple,
                'emc_nelson': emc_nelson,
                'vpd': vpd,
                'wind_temp_interaction': wind_temp_interaction,
                'hour': hour,
                'day_of_year': day_of_year,
                'month': month,
                'latitude': grid_lat_flat,
                'longitude': grid_lon_flat,
                'elevation': 0
            })
            
            # Ensure correct feature order
            feature_names = model_dict['feature_names']
            features_df = features_df[feature_names]
            
            # Scale and predict all at once
            print("  Running ML predictions...")
            scaler = model_dict['scaler']
            features_scaled = scaler.transform(features_df)
            
            model = model_dict['model']
            predictions = model.predict(features_scaled)
            predictions = np.clip(predictions, 1, 40)
            
            # Reshape back to grid
            fm = predictions.reshape(rh.shape)
            
        else:
            # Fallback to physics-based lag model
            print(f"  Processing with physics-based model...")
            for ii in range(rh.shape[0]):
                if ii % 10 == 0:
                    print(f"    Row {ii}/{rh.shape[0]}")
                for jj in range(rh.shape[1]):
                    fm[ii, jj] = estimate_fuel_moisture_with_lag(
                        rh[ii, jj], 
                        temp[ii, jj], 
                        previous_fm[ii, jj],
                        hours_elapsed=1
                    )
        
        # Store for next iteration
        previous_fm = fm.copy()
        previous_rh = rh.copy()
        previous_temp = temp.copy()
        
        # Save hourly values
        hourly_rh.append(rh)
        hourly_temp.append(temp)
        hourly_ws.append(ws_kts)
        hourly_fm.append(fm)
        
        # Calculate fire danger
        print(f"  Calculating fire danger levels...")
        risk = np.zeros_like(rh, dtype=int)
        for ii in range(rh.shape[0]):
            for jj in range(rh.shape[1]):
                risk[ii, jj] = calculate_fire_danger(fm[ii, jj], rh[ii, jj], ws_kts[ii, jj])
        hourly_risks.append(risk)
        
        # Progress summary
        print(f"  ✓ Hour {i+1} complete: FM range {np.nanmin(fm):.1f}-{np.nanmax(fm):.1f}%")
    
    print("\n✓ All forecast hours processed successfully!\n")
    return hourly_fm, hourly_rh, hourly_ws, hourly_temp, hourly_risks


if __name__ == "__main__":
    logger.info("Running as main script.")
    generate_complete_forecast()
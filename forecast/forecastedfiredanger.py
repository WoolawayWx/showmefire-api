import cartopy.crs as ccrs
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
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
import os
import time
import logging
import warnings
from herbie import FastHerbie
import xarray as xr
import requests
from scipy.interpolate import Rbf
import pickle
import json

# Suppress Herbie regex warnings
warnings.filterwarnings('ignore', message='This pattern is interpreted as a regular expression')


def calculate_fire_danger(fuel_moisture, relative_humidity, wind_speed_knots):
    """
    Calculate fire danger level based on NWCG standards with extended scale.
    
    Parameters:
    - fuel_moisture: 10-hour fuel moisture percentage
    - relative_humidity: 2-meter relative humidity percentage
    - wind_speed_knots: 10-meter sustained wind speed in knots
    
    Returns: danger level (0=Low, 1=Moderate, 2=Elevated, 3=Critical, 4=Extreme)
    """
    
    if fuel_moisture >= 15:
        return 0
    
    if fuel_moisture >= 10:
        if relative_humidity < 60 or wind_speed_knots >= 6:
            return 1
        else:
            return 0
    
    if fuel_moisture < 6 and wind_speed_knots >= 25 and relative_humidity < 15:
        return 4
    
    if fuel_moisture < 7 and wind_speed_knots >= 30 and relative_humidity < 20:
        return 4
    
    if wind_speed_knots >= 15 and wind_speed_knots < 20 and relative_humidity < 20:
        return 3
    elif wind_speed_knots >= 20 and wind_speed_knots < 25 and relative_humidity < 25:
        return 3
    elif wind_speed_knots >= 25 and relative_humidity < 25:
        return 3
    elif fuel_moisture < 7 and wind_speed_knots >= 15 and relative_humidity < 30:
        return 3
    
    if wind_speed_knots >= 5 and wind_speed_knots < 10 and relative_humidity < 20:
        return 2
    elif wind_speed_knots >= 10 and wind_speed_knots < 15 and relative_humidity < 35:
        return 2
    elif wind_speed_knots >= 15 and wind_speed_knots < 20 and relative_humidity >= 20 and relative_humidity < 35:
        return 2
    elif wind_speed_knots >= 20 and wind_speed_knots < 25 and relative_humidity >= 25 and relative_humidity < 45:
        return 2
    elif wind_speed_knots >= 25 and relative_humidity >= 25 and relative_humidity < 45:
        return 2
    
    if fuel_moisture < 10:
        return 1
    
    return 0


def estimate_fuel_moisture(relative_humidity, air_temp=None):
    """
    Estimate 10-hour fuel moisture from relative humidity.
    """
    if relative_humidity is None:
        return None
    
    fm_estimate = 3 + 0.25 * relative_humidity
    fm_estimate = np.clip(fm_estimate, 3, 30)
    
    return fm_estimate


def create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi):
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

def get_current_fuel_moisture_field(port='8000'):
    """
    Get current observed fuel moisture from RAWS stations and create initial field.
    This provides the starting point for the forecast.
    """
    try:
        response = requests.get(f'http://localhost:{port}/stations/raws', timeout=5)
        raws_stations = response.json()['stations']
        
        fuel_points = []
        for s in raws_stations:
            fm = s.get('observations', {}).get('fuel_moisture', {}).get('value')
            if fm is not None and fm > 0:
                fuel_points.append((s['longitude'], s['latitude'], fm))
        
        if len(fuel_points) >= 3:
            print(f"Found {len(fuel_points)} RAWS stations with fuel moisture data")
            return fuel_points
        else:
            print(f"Warning: Only {len(fuel_points)} RAWS stations available, using default")
            return None
            
    except Exception as e:
        print(f"Error fetching RAWS data: {e}")
        return None


def interpolate_current_fm_to_grid(fuel_points, grid_lon_mesh, grid_lat_mesh):
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
    
    previous_fm = initial_fm.copy()
    
    for i, time_step in enumerate(ds_full.step):
        ds_hour = ds_full.sel(step=time_step)
        
        # Extract forecast variables
        rh = ds_hour['r2'].values
        temp = ds_hour['t2m'].values - 273.15  # Convert K to C
        u = ds_hour['u10'].values
        v = ds_hour['v10'].values
        
        ws_kts = np.sqrt(u**2 + v**2) * 1.94384
        
        # Calculate fuel moisture with lag model
        fm = np.zeros_like(rh)
        for ii in range(rh.shape[0]):
            for jj in range(rh.shape[1]):
                fm[ii, jj] = estimate_fuel_moisture_with_lag(
                    rh[ii, jj], 
                    temp[ii, jj], 
                    previous_fm[ii, jj],
                    hours_elapsed=1
                )
        
        # Store for next iteration
        previous_fm = fm.copy()
        
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
        
        # Progress indicator
        if i == 0:
            print(f"Hour {i}: FM range {np.nanmin(fm):.1f}-{np.nanmax(fm):.1f}% (initial)")
        elif i % 3 == 0:
            print(f"Hour {i}: FM range {np.nanmin(fm):.1f}-{np.nanmax(fm):.1f}%")
    
    return hourly_fm, hourly_rh, hourly_ws, hourly_temp, hourly_risks


def generate_complete_forecast():
    """
    Generate complete suite of fire weather forecast maps.
    """
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    
    load_dotenv()
    start_time = time.time()
    
    # Configuration
    now = pd.Timestamp.utcnow()
    
    # Always use the most recent available 12z run
    # HRRR runs are available ~1-2 hours after model time
    # So 12z run is typically available by ~14z (2pm UTC)
    if now.hour < 14:
        # If before 14z, use yesterday's 12z run (most recent available)
        RUN_DATE = (now - pd.Timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    else:
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
    
    cache_file = cache_dir / f"hrrr_{RUN_DATE.strftime('%Y%m%d_%H')}z_f04-15.nc"
    
    if cache_file.exists():
        print(f"Loading cached HRRR data from {cache_file}...")
        try:
            ds_full = xr.open_dataset(cache_file, decode_cf=False)
            print("Loaded from cache successfully")
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Deleting corrupted cache and re-downloading...")
            cache_file.unlink()
            # Fall through to download section
    
    if not cache_file.exists():
        print(f"Downloading HRRR data for {RUN_DATE} UTC...")
        FH = FastHerbie(DATES=[RUN_DATE], fxx=list(FORECAST_HOURS), model='hrrr', product='sfc')
        
        ds_rh_temp = FH.xarray(":(TMP|RH):2 m")
        if isinstance(ds_rh_temp, list):
            ds_rh_temp = ds_rh_temp[0]
        
        ds_wind = FH.xarray(":(UGRD|VGRD):10 m")
        if isinstance(ds_wind, list):
            ds_wind = ds_wind[0]
        
        ds_full = ds_rh_temp.merge(ds_wind, compat='override')
        
        # Save to cache for future use
        try:
            print(f"Saving to cache: {cache_file}")
            # Load data into memory and clean encoding attributes
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
    print("Processing forecast data with RAWS observations...")

    # --- Save initialization observation data for ML/verification ---
    # Try to get the current fuel moisture field (RAWS obs)

    # Continue with forecast using ML model
    hourly_fm, hourly_rh, hourly_ws, hourly_temp, hourly_risks = process_forecast_with_ml_model(
        ds_full, lon, lat, port=port, ml_model_path="models/fuel_moisture_model_latest.pkl"
    )
    
    # Calculate peak/min values FIRST
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
    
    # NOW save forecast data for verification
    print("\nSaving forecast data for verification...")
    
    # Get initial fuel moisture field for saving
    fuel_points = get_current_fuel_moisture_field(port)
    if fuel_points and len(fuel_points) >= 3:
        if lon.ndim == 1 and lat.ndim == 1:
            grid_lon_mesh, grid_lat_mesh = np.meshgrid(lon, lat)
        else:
            grid_lon_mesh, grid_lat_mesh = lon, lat
        initial_fm = interpolate_current_fm_to_grid(fuel_points, grid_lon_mesh, grid_lat_mesh)
    else:
        initial_fm = np.full_like(lon, 12.0)  # Default value if no observations
    
    forecast_archive_dir = PROJECT_DIR / 'archive' / 'forecasts'
    forecast_archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert time steps to valid times
    forecast_valid_times = []
    for ts in ds_full.step:
        time_step_value = ts.values
        if isinstance(time_step_value, np.timedelta64):
            hours_ahead = int(time_step_value / np.timedelta64(1, 'h'))
        else:
            hours_ahead = int(time_step_value / 3600000000000)
        valid_time = RUN_DATE + pd.Timedelta(hours=hours_ahead)
        forecast_valid_times.append(valid_time.isoformat())
    
    forecast_data = {
        'model_run': RUN_DATE.strftime('%Y%m%d_%H'),
        'forecast_valid_times': forecast_valid_times,
        'initial_fuel_moisture': {
            'min': float(np.nanmin(initial_fm)),
            'max': float(np.nanmax(initial_fm)),
            'mean': float(np.nanmean(initial_fm))
        },
        'hourly_forecasts': [
            {
                'hour': i,
                'fuel_moisture_min': float(np.nanmin(hourly_fm[i])),
                'fuel_moisture_max': float(np.nanmax(hourly_fm[i])),
                'fuel_moisture_mean': float(np.nanmean(hourly_fm[i])),
                'rh_min': float(np.nanmin(hourly_rh[i])),
                'temp_max': float(np.nanmax(hourly_temp[i])),
                'wind_max': float(np.nanmax(hourly_ws[i]))
            }
            for i in range(len(hourly_fm))
        ],
        'peak_values': {
            'min_fuel_moisture': float(np.nanmin(min_fuel_moisture)),
            'min_rh': float(np.nanmin(min_rh)),
            'max_wind': float(np.nanmax(max_wind)),
            'max_temp': float(np.nanmax(max_temp)),
            'peak_danger_level': int(np.nanmax(peak_risk))
        }
    }
    
    forecast_file = forecast_archive_dir / f"forecast_{RUN_DATE.strftime('%Y%m%d_%H')}.json"
    with open(forecast_file, 'w') as f:
        json.dump(forecast_data, f, indent=2)
    print(f"✓ Saved forecast data to {forecast_file}")
    
    # Apply smoothing (continue with rest of the code)
    peak_risk_smooth = gaussian_filter(peak_risk.astype(float), sigma=1.5)
    
    # Calculate peak/min values
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
    max_temp_smooth_f = max_temp_smooth * 9/5 + 32  # °C to °F
    
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

    # Convert max wind from knots to mph
    max_wind_smooth_mph = max_wind_smooth * 1.15078

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
    
    # ========== MAP 1: PEAK FIRE DANGER ==========
    print("Generating peak fire danger map...")
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
        "Moderate: FM 10-14% with RH < 60% or Wind ≥ 6 kts\n"
        "Elevated: FM < 10% with RH < 45% or Wind ≥ 10 kts\n"
        "Critical: FM < 10% with RH < 25% & Wind ≥ 15 kts\n"
        "Extreme: FM < 7% with RH < 20% & Wind ≥ 30 kts\n\n"
        "Data Source: HRRR Model Forecast\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastfiredanger.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    
    # ========== MAP 2: MINIMUM FUEL MOISTURE ==========
    print("Generating minimum fuel moisture map...")
    fm_colors = ['#8B4513', '#CD853F', '#DAA520', '#F0E68C', '#90EE90', '#228B22']
    fm_levels = [0, 5, 8, 10, 12, 15, 30]
    fm_cmap = ListedColormap(fm_colors)
    fm_norm = BoundaryNorm(fm_levels, len(fm_colors))
    
    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
    
    cs = ax.contourf(lon, lat, min_fuel_moisture_smooth, transform=data_crs,
                     levels=fm_levels, cmap=fm_cmap, norm=fm_norm, alpha=0.7, zorder=7, antialiased=True)
    ax.contour(lon, lat, min_fuel_moisture_smooth, transform=data_crs,
               levels=fm_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
    
    add_boundaries(ax, data_crs, PROJECT_DIR)
    
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Fuel Moisture (%)')
    cbar.set_ticks([2.5, 6.5, 9, 11, 13.5, 22.5])
    cbar.set_ticklabels(['<5%', '5-8%', '8-10%', '10-12%', '12-15%', '>15%'])
    
    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    
    add_title_and_branding(
        fig, "Missouri Minimum Fuel Moisture Forecast",
        f"Model Run: {RUN_DATE.strftime('%Y-%m-%d %HZ')} | Valid: {(RUN_DATE + pd.Timedelta(hours=4)).strftime('%Y-%m-%d')}",
        "Minimum 10-Hour Fuel Moisture (10:00–21:00 CT)\n\n"
        "Critical Thresholds:\n"
        "< 5%: Extremely Dry - Extreme fire behavior possible\n"
        "5-8%: Very Dry - Critical fire behavior likely\n"
        "8-10%: Dry - Elevated fire behavior expected\n"
        "10-15%: Moderate - Fire activity possible\n"
        "> 15%: Moist - Fuels less receptive to fire\n\n"
        "Data Source: HRRR Model Forecast\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastfuelmoistire.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    
    # ========== MAP 3: MINIMUM RELATIVE HUMIDITY ==========
    print("Generating minimum relative humidity map...")
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
        "Data Source: HRRR Model Forecast\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastminrh.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    
    # ========== MAP 4: MAXIMUM WIND SPEED ==========
    print("Generating maximum wind speed map...")
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
        "Data Source: HRRR Model Forecast\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastmaxwind.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    
    # ========== MAP 5: MAXIMUM TEMPERATURE ==========
    print("Generating maximum temperature map...")
    temp_colors = ['#4169E1', '#87CEEB', '#90EE90', '#FFED4E', '#FFA500', '#FF6347', '#8B0000']
    temp_levels = [-10, 0, 10, 20, 27, 32, 38, 45]
    temp_cmap = ListedColormap(temp_colors)
    temp_norm = BoundaryNorm(temp_levels, len(temp_colors))
    
    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
    
    cs = ax.contourf(lon, lat, max_temp_smooth, transform=data_crs,
                     levels=temp_levels, cmap=temp_cmap, norm=temp_norm, alpha=0.7, zorder=7, antialiased=True)
    ax.contour(lon, lat, max_temp_smooth, transform=data_crs,
               levels=temp_levels[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
    
    add_boundaries(ax, data_crs, PROJECT_DIR)
    
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Temperature (°C)')
    
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
        "Data Source: HRRR Model Forecast\n"
        "For More Info, Visit ShowMeFire.org",
        RUN_DATE, SCRIPT_DIR
    )
    
    fig.savefig(PROJECT_DIR / 'images/mo-forecastmaxtemp.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    

    runtime_sec = time.time() - start_time

    print(f"\nAll forecast maps generated successfully!")
    print(f"Peak Fire Danger: images/mo-forecastfiredanger.png")
    print(f"Minimum Fuel Moisture: images/mo-forecastfuelmoistire.png")
    print(f"Minimum RH: images/mo-forecastminrh.png")
    print(f"Maximum Wind: images/mo-forecastmaxwind.png")
    print(f"Maximum Temperature: images/mo-forecastmaxtemp.png")
    print(f"Script runtime: {runtime_sec:.2f} seconds")

    # ========== UPLOAD FORECAST MAPS TO CDN ==========
    upload_success = False
    upload_to_cdn = None
    try:
        # Try absolute import (if run as a module)
        from cdnupload import upload_to_cdn
        upload_success = True
    except ImportError:
        try:
            # Try relative import (if run as a script)
            import sys
            sys.path.append(str(PROJECT_DIR))
            from cdnupload import upload_to_cdn
            upload_success = True
        except ImportError:
            print("[WARN] cdnupload.py not found or upload_to_cdn not available. Skipping CDN upload.")

    if upload_success and upload_to_cdn:
        forecast_files = [
            PROJECT_DIR / 'images/mo-forecastfiredanger.png',
            PROJECT_DIR / 'images/mo-forecastfuelmoistire.png',
            PROJECT_DIR / 'images/mo-forecastminrh.png',
            PROJECT_DIR / 'images/mo-forecastmaxwind.png',
            PROJECT_DIR / 'images/mo-forecastmaxtemp.png'
        ]
        date_folder = RUN_DATE.strftime('%Y%m%d')
        dest_keys = []
        content_types = []
        for f in forecast_files:
            fname = f.name
            dest_keys.append(f"forecasts/latest/{fname}")
            dest_keys.append(f"forecasts/{date_folder}/{fname}")
            content_types.extend(['image/png', 'image/png'])
        files = [f for f in forecast_files for _ in (0,1)]
        print("Uploading forecast maps to CDN...")
        upload_to_cdn(files, dest_keys, content_types=content_types, cache_controls=["max-age=300"]*len(files))

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
        'maps_generated': [
            'mo-forecastfiredanger.png',
            'mo-forecastfuelmoistire.png',
            'mo-forecastminrh.png',
            'mo-forecastmaxwind.png',
            'mo-forecastmaxtemp.png'
        ],
        'log': [
            f"All forecast maps updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}",
            f"Script runtime: {runtime_sec:.2f} seconds"
        ]
    }

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=4)

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
    from scipy.interpolate import Rbf
    from scipy.ndimage import gaussian_filter
    
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
            grid_lon_flat = grid_lon_mesh.ravel()
            
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
    generate_complete_forecast()
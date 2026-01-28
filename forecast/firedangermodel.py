"""
firedangermodel.py

Fire Danger Prediction Model - Aligned with Missouri AOP Guidance
- Uses NWS Elevated Fire Weather Matrix criteria
- Predicts fire danger based on fuel moisture, RH, and wind thresholds
- Trains ML model to predict criteria-based danger scores
- Generates fire danger maps consistent with operational fire weather forecasting
"""
def safe_exists(pathlike):
    """Safely check if a path exists, converting to Path if needed."""
    if not isinstance(pathlike, Path):
        pathlike = Path(pathlike)
    return pathlike.exists()

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import pickle
import shutil
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from shapely.geometry import box, Point
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import cairosvg
from io import BytesIO

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def update_model_config(model_type, model_filename, performance_metrics=None):
    """Update the models/config.json with new model information and maintain history."""
    config_path = Path('models/config.json')
    archive_dir = Path('models/archive')
    archive_dir.mkdir(exist_ok=True)
    
    # Load existing config or create default
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "fuel_moisture": {
                "active_version": None,
                "threshold": 0.85,
                "last_updated": None,
                "history": []
            },
            "fire_danger": {
                "active_version": None,
                "history": []
            }
        }
    
    # Archive the current model if it exists
    if model_type in config and config[model_type]["active_version"]:
        current_model = config[model_type]["active_version"]
        model_path = Path('models') / current_model
        
        if model_path.exists():
            # Create archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"{Path(current_model).stem}_{timestamp}{Path(current_model).suffix}"
            archive_path = archive_dir / archive_filename
            
            # Move to archive
            shutil.move(str(model_path), str(archive_path))
            logger.info(f"Archived previous model: {current_model} → {archive_filename}")
            
            # Add to history
            history_entry = {
                "version": current_model,
                "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "archive_path": str(archive_path),
                "performance": config[model_type].get("performance", {}),
                "last_updated": config[model_type].get("last_updated")
            }
            
            if "history" not in config[model_type]:
                config[model_type]["history"] = []
            config[model_type]["history"].append(history_entry)
            
            # Keep only last 10 entries in history
            if len(config[model_type]["history"]) > 10:
                config[model_type]["history"] = config[model_type]["history"][-10:]
    
    # Update the specific model section with new model
    if model_type in config:
        config[model_type]["active_version"] = model_filename
        config[model_type]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        # Add performance metrics if provided
        if performance_metrics:
            config[model_type]["performance"] = performance_metrics
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Updated {config_path} with new {model_type} model: {model_filename}")
    logger.info(f"History maintained: {len(config[model_type].get('history', []))} previous versions")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== MAP GENERATION UTILITIES ====================

def create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi):
    """Create base map with Missouri boundaries."""
    figsize_width = pixelw / mapdpi
    figsize_height = pixelh / mapdpi
    
    fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=mapdpi, facecolor='#E8E8E8')
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)
    
    return fig, ax


def add_boundaries(ax, data_crs, project_dir):
    """Add county and state boundaries to map."""
    project_dir = Path(project_dir)
    shapefile_dir = Path(project_dir) / 'maps' / 'shapefiles'
    
    # Counties
    counties = gpd.read_file(shapefile_dir / 'MO_County_Boundaries' / 'MO_County_Boundaries.shp')
    if counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", 
                     facecolor='none', linewidth=1, zorder=5)
    
    # State border
    state = gpd.read_file(shapefile_dir / 'MO_State_Boundary' / 'MO_State_Boundary.shp')
    if state.crs != data_crs.proj4_init:
        state = state.to_crs(data_crs.proj4_init)
    ax.add_geometries(state.geometry, crs=data_crs, edgecolor="#000000", 
                     facecolor='none', linewidth=1.5, zorder=6)


def add_text_and_logo(fig, ax, title, subtitle, date_str, project_dir):
    """Add title, subtitle, legend, and logo to map."""
    project_dir = Path(project_dir)
    
    # Load fonts
    font_paths = [
        str(project_dir / 'assets/Montserrat/static/Montserrat-Regular.ttf'),
        str(project_dir / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf'),
        str(project_dir / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf')
    ]
    for font_path in font_paths:
        if safe_exists(font_path):
            font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Montserrat'
    
    # Title
    fig.text(0.99, 0.97, title, fontsize=26, fontweight='bold', 
             ha='right', va='top', fontname='Plus Jakarta Sans')
    
    # Subtitle with date
    fig.text(0.99, 0.90, f"{subtitle} | Valid: {date_str}",
             fontsize=16, ha='right', va='top', fontname='Montserrat')
    
    # Criteria text - Missouri AOP aligned
    fig.text(0.99, 0.62,
             "Fire Danger Criteria (aligns with MO AOP guidance)\n"
             "\n"
             "Low: FM ≥ 15% (Fuels adequately moist)\n"
             "Moderate: FM 10-14% with RH < 60% or Wind ≥ 6 kts\n"
             "Elevated: FM < 10% with RH < 45% or Wind ≥ 10 kts\n"
             "Critical: FM < 10% with RH < 25% & Wind ≥ 15 kts\n"
             "Extreme: FM < 7% with RH < 20% & Wind ≥ 30 kts\n\n"
             "Criteria represent potential for fire ignition and spread.\n\n"
             "Data Source: HRRR Weather Model\n"
             "For More Info, Visit ShowMeFire.org",
             fontsize=10, ha='right', va='top', linespacing=1.6, fontname='Montserrat')
    
    # Bottom left branding
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight='bold',
             ha='left', va='bottom', fontname='Montserrat')
    
    # Logo
    svg_path = str(project_dir / 'assets/LightBackGroundLogo.svg')
    try:
        png_bytes = cairosvg.svg2png(url=svg_path)
        image = mpimg.imread(BytesIO(png_bytes), format='png')
        imagebox = OffsetImage(image, zoom=0.03)
        ab = AnnotationBbox(imagebox, (0.99, 0.01), frameon=False,
                           xycoords='figure fraction', box_alignment=(1, 0))
        ax.add_artist(ab)
    except:
        pass  # Logo optional


def generate_fire_danger_map(fire_danger_grid, lon, lat, output_path, 
                             model_run_date, valid_date, project_dir):
    """
    Generate fire danger map with consistent styling.
    
    Args:
        fire_danger_grid: 2D array of fire danger scores (0-100)
        lon: 2D longitude grid
        lat: 2D latitude grid
        output_path: Where to save the map
        model_run_date: When model was run
        valid_date: Valid time for forecast
        project_dir: Project root directory
    """
    
    # Map parameters
    pixelw = 2048
    pixelh = 1152
    mapdpi = 144
    extent = (-95.8, -89.1, 35.8, 40.8)
    
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    
    # Fire danger color scheme
    colors = ["#90EE90", '#FFED4E', '#FFA500', '#FF0000', '#8B0000']
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, len(colors))
    
    # Create map
    fig, ax = create_base_map(extent, map_crs, data_crs, pixelw, pixelh, mapdpi)
    
    # Mask to Missouri
    shapefile_dir = Path(project_dir) / 'maps' / 'shapefiles'
    missouri = gpd.read_file(shapefile_dir / 'MO_State_Boundary' / 'MO_State_Boundary.shp')
    if missouri.crs != data_crs.proj4_init:
        missouri = missouri.to_crs(data_crs.proj4_init)
    
    if not missouri.empty:
        missouri_geom = missouri.geometry.iloc[0]
        grid_points = [Point(lon_val, lat_val) for lon_val, lat_val in zip(lon.ravel(), lat.ravel())]
        within_mask = gpd.GeoSeries(grid_points).within(missouri_geom).values.reshape(lon.shape)
        fire_danger_grid_masked = fire_danger_grid.copy()
        fire_danger_grid_masked[~within_mask] = np.nan
    else:
        fire_danger_grid_masked = fire_danger_grid
    
    # Plot fire danger
    cs = ax.contourf(lon, lat, fire_danger_grid_masked, transform=data_crs,
                     levels=bins, cmap=cmap, norm=norm, alpha=0.7, zorder=7, antialiased=True)
    
    # Contour lines
    ax.contour(lon, lat, fire_danger_grid_masked, transform=data_crs,
               levels=bins[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
    
    # Add boundaries
    add_boundaries(ax, data_crs, project_dir)
    
    # Colorbar
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = plt.colorbar(cs, cax=cax, label='Fire Danger Score')
    cbar.set_ticks([10, 30, 50, 70, 90])
    cbar.set_ticklabels(['Low\n(0-20)', 'Moderate\n(20-40)', 'Elevated\n(40-60)', 
                         'Critical\n(60-80)', 'Extreme\n(80-100)'])
    
    # Text and branding
    add_text_and_logo(
        fig, ax,
        title="Missouri ML Fire Danger Forecast",
        subtitle="Machine Learning Model Prediction",
        date_str=valid_date.strftime('%Y-%m-%d %H:%M CT'),
        project_dir=project_dir
    )
    
    # Save
    ax.set_anchor('W')
    plt.subplots_adjust(left=0.05)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=mapdpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    
    logger.info(f"Saved fire danger map to {output_path}")


# ==================== DATA PROCESSING ====================

def calculate_nelson_emc(rh, temp_f):
    """Calculate equilibrium moisture content using Nelson's equations."""
    if pd.isna(rh) or pd.isna(temp_f):
        return None
    
    temp_c = (temp_f - 32) * 5/9
    
    if rh <= 10:
        emc = 0.03 + 0.2626 * rh - 0.00104 * rh * temp_c
    elif rh <= 50:
        emc = 2.22 - 0.160 * rh + 0.01660 * temp_c
    else:
        emc = 21.06 - 0.4944 * rh + 0.005565 * rh**2 - 0.00063 * rh * temp_c
    
    return max(1, min(40, emc))


def calculate_vpd(temp_f, rh):
    """Calculate Vapor Pressure Deficit (VPD) in kPa."""
    temp_c = (temp_f - 32) * 5/9
    svp = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    avp = svp * (rh / 100)
    vpd = svp - avp
    return vpd


def load_archived_data(filepath: Path) -> Dict:
    """Load raw archived JSON data."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def process_timeseries_to_dataframe(api_response: Dict) -> pd.DataFrame:
    """Convert Synoptic API timeseries response to pandas DataFrame."""
    all_records = []
    
    if not api_response.get("STATION"):
        logger.warning("No stations in API response")
        return pd.DataFrame()
    
    for station in api_response["STATION"]:
        stid = station.get("STID")
        name = station.get("NAME")
        state = station.get("STATE")
        latitude = station.get("LATITUDE")
        longitude = station.get("LONGITUDE")
        elevation = station.get("ELEVATION")
        network = station.get("MNET_SHORTNAME", "Unknown")
        
        observations = station.get("OBSERVATIONS", {})
        times = observations.get("date_time", [])
        
        fm_values = observations.get("fuel_moisture_set_1", [])
        rh_values = observations.get("relative_humidity_set_1", [])
        temp_values = observations.get("air_temp_set_1", [])
        wind_values = observations.get("wind_speed_set_1", [])
        gust_values = observations.get("wind_gust_set_1", [])
        solar_values = observations.get("solar_radiation_set_1", [])
        precip_values = observations.get("precip_accum_set_1", [])
        
        for i in range(len(times)):
            timestamp = pd.to_datetime(times[i])
            
            record = {
                'stid': stid,
                'station_name': name,
                'state': state,
                'network': network,
                'timestamp': timestamp,
                'latitude': latitude,
                'longitude': longitude,
                'elevation': elevation,
                'hour': timestamp.hour,
                'day_of_year': timestamp.dayofyear,
                'month': timestamp.month,
            }
            
            record['rh'] = rh_values[i] if i < len(rh_values) and rh_values[i] is not None else None
            record['temp'] = temp_values[i] if i < len(temp_values) and temp_values[i] is not None else None
            record['wind'] = wind_values[i] if i < len(wind_values) and wind_values[i] is not None else None
            record['wind_gust'] = gust_values[i] if i < len(gust_values) and gust_values[i] is not None else None
            record['solar'] = solar_values[i] if i < len(solar_values) and solar_values[i] is not None else None
            record['precip'] = precip_values[i] if i < len(precip_values) and precip_values[i] is not None else None
            record['fuel_moisture'] = fm_values[i] if i < len(fm_values) and fm_values[i] is not None else None
            
            if i > 0:
                record['prev_rh'] = rh_values[i-1] if i-1 < len(rh_values) else None
                record['prev_temp'] = temp_values[i-1] if i-1 < len(temp_values) else None
                record['prev_fm'] = fm_values[i-1] if i-1 < len(fm_values) else None
            
            if i > 2:
                record['rh_3h_avg'] = np.mean([rh_values[j] for j in range(i-3, i) if j < len(rh_values) and rh_values[j] is not None])
                record['temp_3h_avg'] = np.mean([temp_values[j] for j in range(i-3, i) if j < len(temp_values) and temp_values[j] is not None])
            
            all_records.append(record)
    
    df = pd.DataFrame(all_records)
    
    if len(df) > 0:
        df['temp_c'] = (df['temp'] - 32) * 5/9
        df['emc_simple'] = 3 + 0.25 * df['rh']
        df['emc_nelson'] = df.apply(
            lambda row: calculate_nelson_emc(row['rh'], row['temp']) 
            if pd.notna(row['rh']) and pd.notna(row['temp']) else None,
            axis=1
        )
        df['vpd'] = df.apply(
            lambda row: calculate_vpd(row['temp'], row['rh']) 
            if pd.notna(row['temp']) and pd.notna(row['rh']) else None,
            axis=1
        )
        df['wind_temp_interaction'] = df['wind'] * df['temp_c']
        
        logger.info(f"Processed {len(df)} observations from {df['stid'].nunique()} stations")
    
    return df


def load_hrrr_archived_data(hrrr_dir: str = "cache/hrrr") -> pd.DataFrame:
    """Load archived HRRR forecast data from NetCDF files."""
    try:
        import xarray as xr
    except ImportError:
        logger.error("xarray not installed. Install with: pip install xarray netcdf4")
        return pd.DataFrame()
    
    hrrr_path = Path(hrrr_dir)
    
    if not safe_exists(hrrr_path):
        logger.warning(f"HRRR cache directory does not exist: {hrrr_dir}")
        return pd.DataFrame()
    
    all_records = []
    
    # Look for NetCDF files matching pattern: hrrr_YYYYMMDD_HHz+f**.nc
    for filepath in sorted(hrrr_path.glob("hrrr_*.nc")):
        logger.info(f"Loading HRRR NetCDF {filepath.name}...")
        try:
            # Open NetCDF file
            ds = xr.open_dataset(filepath)
            
            # Extract model run time from filename
            # Format: hrrr_20250101_12z+f04-15.nc
            filename = filepath.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                date_str = parts[1]  # YYYYMMDD
                time_str = parts[2].split('+')[0]  # HHz
                model_run_hour = int(time_str.replace('z', ''))
                model_run_date = pd.to_datetime(date_str, format='%Y%m%d') + pd.Timedelta(hours=model_run_hour)
            else:
                model_run_date = None
            
            # Get dimensions
            if 'time' in ds.dims:
                times = ds['time'].values
            elif 'valid_time' in ds.dims:
                times = ds['valid_time'].values
            else:
                logger.warning(f"No time dimension found in {filepath.name}")
                continue
            
            # Get spatial coordinates
            lats = ds['latitude'].values if 'latitude' in ds else ds['lat'].values
            lons = ds['longitude'].values if 'longitude' in ds else ds['lon'].values
            
            # Flatten spatial dimensions
            if lats.ndim == 2:
                lats_flat = lats.ravel()
                lons_flat = lons.ravel()
            else:
                lons_2d, lats_2d = np.meshgrid(lons, lats)
                lats_flat = lats_2d.ravel()
                lons_flat = lons_2d.ravel()
            
            # Extract weather variables for each time
            for time_idx, time_val in enumerate(times):
                timestamp = pd.to_datetime(time_val)
                
                # Get weather data (expecting 2D or 3D arrays)
                temp_data = ds['t2m'].isel(time=time_idx).values if 't2m' in ds else None
                if temp_data is None:
                    temp_data = ds['temperature'].isel(time=time_idx).values if 'temperature' in ds else None
                
                rh_data = ds['r2'].isel(time=time_idx).values if 'r2' in ds else None
                if rh_data is None:
                    rh_data = ds['rh'].isel(time=time_idx).values if 'rh' in ds else None
                
                wind_u = ds['u10'].isel(time=time_idx).values if 'u10' in ds else None
                wind_v = ds['v10'].isel(time=time_idx).values if 'v10' in ds else None
                
                precip_data = ds['tp'].isel(time=time_idx).values if 'tp' in ds else None
                if precip_data is None:
                    precip_data = ds['precip'].isel(time=time_idx).values if 'precip' in ds else None
                
                solar_data = ds['dswrf'].isel(time=time_idx).values if 'dswrf' in ds else None
                if solar_data is None:
                    solar_data = ds['solar'].isel(time=time_idx).values if 'solar' in ds else None
                
                # Flatten arrays
                if temp_data is not None:
                    temp_flat = temp_data.ravel()
                    # Convert from Kelvin to Fahrenheit if needed
                    if temp_flat.mean() > 200:  # Likely Kelvin
                        temp_flat = (temp_flat - 273.15) * 9/5 + 32
                else:
                    temp_flat = np.full(len(lats_flat), np.nan)
                
                rh_flat = rh_data.ravel() if rh_data is not None else np.full(len(lats_flat), np.nan)
                
                # Calculate wind speed from u and v components
                if wind_u is not None and wind_v is not None:
                    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
                    wind_flat = wind_speed.ravel() * 2.23694  # m/s to mph
                else:
                    wind_flat = np.full(len(lats_flat), np.nan)
                
                precip_flat = precip_data.ravel() if precip_data is not None else np.full(len(lats_flat), np.nan)
                solar_flat = solar_data.ravel() if solar_data is not None else np.full(len(lats_flat), np.nan)
                
                # Create records for this time step
                for i in range(len(lats_flat)):
                    record = {
                        'timestamp': timestamp,
                        'latitude': lats_flat[i],
                        'longitude': lons_flat[i],
                        'temp': temp_flat[i],
                        'rh': rh_flat[i],
                        'wind': wind_flat[i],
                        'precip': precip_flat[i],
                        'solar': solar_flat[i],
                        'hour': timestamp.hour,
                        'day_of_year': timestamp.dayofyear,
                        'month': timestamp.month,
                    }
                    all_records.append(record)
            
            ds.close()
            logger.info(f"  Loaded {len(times)} time steps with {len(lats_flat)} grid points each")
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_records:
        logger.warning("No HRRR data loaded from NetCDF files")
        return pd.DataFrame()
    
    combined_df = pd.DataFrame(all_records)
    
    # Add derived features
    if 'temp' in combined_df.columns:
        combined_df['temp_c'] = (combined_df['temp'] - 32) * 5/9
    
    if 'rh' in combined_df.columns and 'temp' in combined_df.columns:
        combined_df['emc_nelson'] = combined_df.apply(
            lambda row: calculate_nelson_emc(row['rh'], row['temp']) 
            if pd.notna(row['rh']) and pd.notna(row['temp']) else None,
            axis=1
        )
        combined_df['emc_simple'] = 3 + 0.25 * combined_df['rh']
        combined_df['vpd'] = combined_df.apply(
            lambda row: calculate_vpd(row['temp'], row['rh']) 
            if pd.notna(row['temp']) and pd.notna(row['rh']) else None,
            axis=1
        )
    
    if 'wind' in combined_df.columns and 'temp_c' in combined_df.columns:
        combined_df['wind_temp_interaction'] = combined_df['wind'] * combined_df['temp_c']
    
    # Remove NaN values
    combined_df = combined_df.dropna(subset=['temp', 'rh', 'wind'])
    
    logger.info(f"Combined HRRR dataset: {len(combined_df)} valid grid points")
    logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df


def load_all_archived_data(archive_dir: str = "archive/raw_data", 
                          hrrr_dir: str = "cache/hrrr",
                          include_hrrr: bool = True) -> pd.DataFrame:
    """Load all archived data files and combine into one DataFrame."""
    archive_path = Path(archive_dir)
    
    try:
        import xarray as xr
        import numpy as np
        import pandas as pd
        import netCDF4
    except ImportError:
        logger.error("xarray, numpy, pandas, or netCDF4 not installed.")
        return pd.DataFrame()

    hrrr_path = Path(hrrr_dir)

    if not safe_exists(hrrr_path):
        logger.warning(f"HRRR directory {hrrr_path} does not exist.")
        return pd.DataFrame()

    all_records = []

    # Look for NetCDF files matching pattern: hrrr_YYYYMMDD_HHz+f**.nc
    for filepath in sorted(hrrr_path.glob("hrrr_*.nc")):
        try:
            # Patch: Remove 'dtype' attribute from 'step' variable if present
            with netCDF4.Dataset(filepath, 'r+') as ds_nc:
                if 'step' in ds_nc.variables:
                    step_var = ds_nc.variables['step']
                    if hasattr(step_var, 'dtype') or 'dtype' in step_var.ncattrs():
                        try:
                            step_var.delncattr('dtype')
                        except Exception:
                            pass
            # Now open with xarray
            ds = xr.open_dataset(filepath)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue

        # Vectorized data extraction
        try:
            import numpy as np
            times = ds['time'].values if 'time' in ds else None
            lats = ds['latitude'].values if 'latitude' in ds else ds['lat'].values
            lons = ds['longitude'].values if 'longitude' in ds else ds['lon'].values

            # Ensure times is always an array
            if times is not None and not isinstance(times, (list, np.ndarray)):
                times = np.array([times])
            elif times is not None and np.isscalar(times):
                times = np.array([times])

            if times is None:
                logger.error(f"No time dimension found in {filepath}")
                continue

            n_lat = lats.shape[0]
            n_lon = lons.shape[0]
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()

            for t_idx, t_val in enumerate(times):
                def get_flat(var):
                    if var in ds:
                        arr = ds[var][t_idx].values
                        return arr.flatten()
                    return np.full(lat_flat.shape, np.nan)

                temp_flat = get_flat('temp')
                rh_flat = get_flat('rh')
                wind_flat = get_flat('wind')
                wind_gust_flat = get_flat('wind_gust')
                solar_flat = get_flat('solar')
                precip_flat = get_flat('precip')

                n_points = lat_flat.shape[0]
                # Append records for this time step one by one (avoid large DataFrame in memory)
                timestamp = pd.to_datetime(str(t_val))
                for i in range(n_points):
                    record = {
                        'timestamp': timestamp,
                        'latitude': lat_flat[i],
                        'longitude': lon_flat[i],
                        'temp': temp_flat[i],
                        'rh': rh_flat[i],
                        'wind': wind_flat[i],
                        'wind_gust': wind_gust_flat[i],
                        'solar': solar_flat[i],
                        'precip': precip_flat[i],
                        'data_source': 'hrrr',
                    }
                    all_records.append(record)
        except Exception as e:
            logger.error(f"Error extracting data from {filepath}: {e}")
            continue

    if not all_records:
        logger.warning("No HRRR data loaded from NetCDF files")
        return pd.DataFrame()

    combined_df = pd.DataFrame(all_records)

    # Add derived features
    if 'temp' in combined_df.columns:
        # ...existing code...
        pass

    if 'rh' in combined_df.columns and 'temp' in combined_df.columns:
        # ...existing code...
        pass

    if 'wind' in combined_df.columns and 'temp_c' in combined_df.columns:
        # ...existing code...
        pass

    # Remove NaN values
    combined_df = combined_df.dropna(subset=['temp', 'rh', 'wind'])

    logger.info(f"Combined HRRR dataset: {len(combined_df)} valid grid points")
    logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

    return combined_df
    station_count = (combined_df['data_source'] == 'station').sum()
    hrrr_count = (combined_df['data_source'] == 'hrrr').sum()
    
    logger.info(f"Combined dataset: {len(combined_df)} total observations")
    logger.info(f"  Station observations: {station_count}")
    logger.info(f"  HRRR grid points: {hrrr_count}")
    logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df


# ==================== FIRE DANGER CALCULATION (Missouri AOP Criteria) ====================

def calculate_fire_danger_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate fire danger score based on Missouri AOP guidance.
    
    Criteria (aligned with NWS Elevated Fire Weather Matrix):
    - Low: FM ≥ 15%
    - Moderate: FM 10-14% with RH < 60% or Wind ≥ 6 kts
    - Elevated: FM < 10% with RH < 45% or Wind ≥ 10 kts
    - Critical: FM < 10% with RH < 25% & Wind ≥ 15 kts
    - Extreme: FM < 7% with RH < 20% & Wind ≥ 30 kts
    
    Returns: Fire danger score (0-100 scale)
    """
    
    # Use actual fuel moisture if available, otherwise use EMC
    fm = df['fuel_moisture'].fillna(df['emc_nelson'])
    rh = df['rh']
    wind = df['wind']
    
    # Initialize score array
    score = pd.Series(0.0, index=df.index)
    
    # Convert wind from mph to knots for consistency with criteria (1 mph = 0.868976 knots)
    wind_kts = wind * 0.868976
    
    # Apply Missouri AOP criteria
    
    # LOW (0-20 points): FM ≥ 15%
    low_mask = (fm >= 15)
    score[low_mask] = 10  # Base low danger score
    
    # MODERATE (20-40 points): FM 10-14% with RH < 60% or Wind ≥ 6 kts
    moderate_mask = (
        (fm >= 10) & (fm < 15) & 
        ((rh < 60) | (wind_kts >= 6))
    )
    score[moderate_mask] = 30
    
    # ELEVATED (40-60 points): FM < 10% with RH < 45% or Wind ≥ 10 kts
    elevated_mask = (
        (fm < 10) & 
        ((rh < 45) | (wind_kts >= 10))
    )
    score[elevated_mask] = 50
    
    # CRITICAL (60-80 points): FM < 10% with RH < 25% AND Wind ≥ 15 kts
    critical_mask = (
        (fm < 10) & 
        (rh < 25) & 
        (wind_kts >= 15)
    )
    score[critical_mask] = 70
    
    # EXTREME (80-100 points): FM < 7% with RH < 20% AND Wind ≥ 30 kts
    extreme_mask = (
        (fm < 7) & 
        (rh < 20) & 
        (wind_kts >= 30)
    )
    score[extreme_mask] = 90
    
    # Add gradients within each category based on how extreme conditions are
    
    # Within Moderate: scale by RH and wind
    if moderate_mask.any():
        moderate_subset = df[moderate_mask]
        rh_factor = np.clip((60 - moderate_subset['rh']) / 60, 0, 1)
        wind_factor = np.clip(wind_kts[moderate_mask] / 15, 0, 1)
        gradient = (rh_factor + wind_factor) / 2 * 10
        score[moderate_mask] += gradient
    
    # Within Elevated: scale by RH and wind
    if elevated_mask.any():
        elevated_subset = df[elevated_mask]
        rh_factor = np.clip((45 - elevated_subset['rh']) / 45, 0, 1)
        wind_factor = np.clip(wind_kts[elevated_mask] / 20, 0, 1)
        gradient = (rh_factor + wind_factor) / 2 * 10
        score[elevated_mask] += gradient
    
    # Within Critical: scale by how close to extreme
    if critical_mask.any():
        critical_subset = df[critical_mask]
        rh_factor = np.clip((25 - critical_subset['rh']) / 25, 0, 1)
        wind_factor = np.clip(wind_kts[critical_mask] / 30, 0, 1)
        fm_factor = np.clip((10 - fm[critical_mask]) / 10, 0, 1)
        gradient = (rh_factor + wind_factor + fm_factor) / 3 * 10
        score[critical_mask] += gradient
    
    # Within Extreme: scale to 100 at most extreme conditions
    if extreme_mask.any():
        extreme_subset = df[extreme_mask]
        rh_factor = np.clip((20 - extreme_subset['rh']) / 20, 0, 1)
        wind_factor = np.clip((wind_kts[extreme_mask] - 30) / 20, 0, 1)
        fm_factor = np.clip((7 - fm[extreme_mask]) / 7, 0, 1)
        gradient = (rh_factor + wind_factor + fm_factor) / 3 * 10
        score[extreme_mask] += gradient
    
    # Recent precipitation reduction
    if 'precip' in df.columns:
        # Significant recent rain can temporarily reduce danger
        precip_mask = df['precip'] > 0.1
        if precip_mask.any():
            # Reduce score but don't eliminate it (fuels can dry quickly)
            reduction = np.clip(df['precip'][precip_mask] * 20, 0, 20)
            score[precip_mask] = np.maximum(score[precip_mask] - reduction, 10)
    
    # Ensure score stays in 0-100 range
    score = np.clip(score, 0, 100)
    
    return score


def categorize_fire_danger(score: pd.Series) -> pd.Series:
    """
    Convert continuous score to categorical levels.
    Aligned with Missouri AOP guidance.
    """
    conditions = [
        score < 20,   # Low
        score < 40,   # Moderate
        score < 60,   # Elevated
        score < 80,   # Critical
        score >= 80   # Extreme
    ]
    
    levels = [0, 1, 2, 3, 4]
    
    return pd.Series(np.select(conditions, levels), index=score.index)


# ==================== MODEL TRAINING ====================

def prepare_fire_danger_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for end-to-end fire danger prediction."""
    
    # Calculate ground truth using Missouri AOP criteria
    df['fire_danger_score'] = calculate_fire_danger_score(df)
    df['fire_danger_level'] = categorize_fire_danger(df['fire_danger_score'])
    
    # Features
    feature_cols = [
        'rh', 'temp', 'temp_c', 'wind', 'wind_gust', 'solar', 'precip',
        'prev_rh', 'prev_temp',
        'rh_3h_avg', 'temp_3h_avg',
        'emc_simple', 'emc_nelson', 'vpd', 'wind_temp_interaction',
        'hour', 'day_of_year', 'month',
        'latitude', 'longitude', 'elevation'
    ]
    
    available_cols = [col for col in feature_cols if col in df.columns]
    clean_df = df[available_cols + ['fire_danger_score', 'fire_danger_level']].dropna()
    
    logger.info(f"Training data: {len(clean_df)} complete observations")
    logger.info(f"Features: {len(available_cols)} columns")
    logger.info(f"Fire danger score range: {clean_df['fire_danger_score'].min():.1f} to {clean_df['fire_danger_score'].max():.1f}")
    
    # Distribution aligned with MO AOP criteria
    level_counts = clean_df['fire_danger_level'].value_counts().sort_index()
    logger.info("\nFire Danger Level Distribution (Missouri AOP Criteria):")
    labels = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
    for level, count in level_counts.items():
        if level < len(labels):
            logger.info(f"  {labels[int(level)]}: {count} ({count/len(clean_df)*100:.1f}%)")
    
    X = clean_df[available_cols]
    y = clean_df['fire_danger_score']
    
    return X, y


def train_fire_danger_model(X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest") -> Dict:
    """Train end-to-end fire danger prediction model."""
    logger.info(f"Training {model_type} fire danger model (Missouri AOP aligned)...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'mae': mean_absolute_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Training complete!")
    logger.info(f"  Train MAE: {train_metrics['mae']:.2f} fire danger points")
    logger.info(f"  Test MAE:  {test_metrics['mae']:.2f} fire danger points")
    logger.info(f"  Test RMSE: {test_metrics['rmse']:.2f}")
    logger.info(f"  Test R²:   {test_metrics['r2']:.3f}")
    logger.info(f"  CV MAE:    {cv_mae:.2f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_mae': cv_mae,
        'feature_importance': feature_importance,
        'feature_names': list(X.columns),
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }


def save_model(model_dict: Dict, filepath: Path):
    """Save trained model."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Path) -> Dict:
    """Load trained model."""
    if not safe_exists(filepath):
        logger.error(f"Model file not found: {filepath}")
        return None
    with open(filepath, 'rb') as f:
        model_dict = pickle.load(f)
    logger.info(f"Model loaded from {filepath}")
    return model_dict


def predict_fire_danger(model_dict: Dict, features: pd.DataFrame) -> np.ndarray:
    """Predict fire danger scores from weather features."""
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    feature_names = model_dict['feature_names']
    features = features[feature_names]
    
    features_scaled = scaler.transform(features)
    fire_danger_scores = model.predict(features_scaled)
    fire_danger_scores = np.clip(fire_danger_scores, 0, 100)
    
    return fire_danger_scores


# ==================== VISUALIZATION ====================

def generate_fire_danger_report(model_dict: Dict, output_dir: str = "reports"):
    """Generate visualizations for fire danger model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    y_test = model_dict['y_test']
    y_pred = model_dict['y_test_pred']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([0, 100], [0, 100], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Fire Danger Score')
    axes[0, 0].set_ylabel('Predicted Fire Danger Score')
    axes[0, 0].set_title('Actual vs Predicted Fire Danger (MO AOP Criteria)')
    axes[0, 0].grid(alpha=0.3)
    
    # Error distribution
    errors = y_pred - y_test
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution (MAE: {np.abs(errors).mean():.2f})')
    axes[0, 1].grid(alpha=0.3)
    
    # Error by level
    y_test_cat = categorize_fire_danger(pd.Series(y_test))
    levels = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
    level_data = []
    for i in range(5):
        mask = y_test_cat == i
        if mask.sum() > 0:
            level_data.append(errors[mask.values])
        else:
            level_data.append([])
    
    bp = axes[1, 0].boxplot(level_data, labels=levels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Error by Fire Danger Level')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Feature importance
    top_features = model_dict['feature_importance'].head(10)
    axes[1, 1].barh(range(len(top_features)), top_features['importance'].values)
    axes[1, 1].set_yticks(range(len(top_features)))
    axes[1, 1].set_yticklabels(top_features['feature'].values)
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Feature Importance')
    axes[1, 1].grid(alpha=0.3, axis='x')
    
    plt.suptitle('Fire Danger Model Performance (Missouri AOP Criteria)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / f'fire_danger_model_{timestamp}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved report to {output_file}")
    plt.close()


# ==================== MAIN WORKFLOW ====================

def train_and_save_fire_danger_model(archive_dir: str = "archive/raw_data",
                                     hrrr_dir: str = "cache/hrrr",
                                     model_dir: str = "models",
                                     include_hrrr: bool = True):
    """Main training workflow for fire danger model."""
    logger.info("="*60)
    logger.info("FIRE DANGER MODEL TRAINING - MISSOURI AOP ALIGNED")
    logger.info("="*60)
    
    # 1. Load archived data (stations + HRRR)
    df = load_all_archived_data(archive_dir, hrrr_dir, include_hrrr)
    
    if df.empty:
        logger.error("No data available!")
        return None
    
    # 2. Prepare training data
    X, y = prepare_fire_danger_training_data(df)
    
    if X.empty:
        logger.error("No training data prepared!")
        return None
    
    # 3. Train model
    model_dict = train_fire_danger_model(X, y, model_type="random_forest")
    
    # 4. Save model
    model_path = Path(model_dir) / f"fire_danger_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    save_model(model_dict, model_path)
    
    latest_path = Path(model_dir) / "fire_danger_model_latest.pkl"
    save_model(model_dict, latest_path)
    
    # 5. Update model configuration
    performance_metrics = {
        "mae": round(model_dict.get('mae', 0), 3),
        "r2_score": round(model_dict.get('r2', 0), 3),
        "training_samples": len(model_dict.get('y_train', [])),
        "test_samples": len(model_dict.get('y_test', []))
    }
    
    update_model_config(
        model_type="fire_danger",
        model_filename=latest_path.name,
        performance_metrics=performance_metrics
    )
    
    # 6. Generate report
    generate_fire_danger_report(model_dict)
    
    # 7. Print feature importance
    logger.info("\nTop 10 Most Important Features:")
    logger.info("-"*60)
    for idx, row in model_dict['feature_importance'].head(10).iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    return model_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fire Danger ML Model Training and Mapping (Missouri AOP Aligned)')
    parser.add_argument('--train', action='store_true', help='Train fire danger model')
    parser.add_argument('--generate-map', action='store_true', help='Generate fire danger map from model')
    parser.add_argument('--forecast-data', type=str, help='Path to forecast data JSON file')
    parser.add_argument('--archive-dir', default='archive/raw_data', help='Archive directory')
    parser.add_argument('--hrrr-dir', default='cache/hrrr', help='HRRR archive directory')
    parser.add_argument('--model-dir', default='models', help='Model save directory')
    parser.add_argument('--output', default='images/ml-fire-danger.png', help='Output map path')
    parser.add_argument('--with-hrrr', action='store_true', help='Include HRRR data in training (requires significant RAM)')

    args = parser.parse_args()

    if args.train:
        train_and_save_fire_danger_model(
            args.archive_dir, 
            args.hrrr_dir, 
            args.model_dir,
            include_hrrr=args.with_hrrr
        )
    
    elif args.generate_map:
        # Load the trained model
        model_path = Path(args.model_dir) / "fire_danger_model_latest.pkl"
        model_dict = load_model(model_path)
        
        if model_dict is None:
            logger.error("No trained model found! Train first with --train")
            exit(1)
        
        # Load forecast data (or use test data)
        if args.forecast_data:
            forecast_data_path = Path(args.forecast_data)
            if safe_exists(forecast_data_path):
                with open(forecast_data_path, 'r') as f:
                    forecast_data = json.load(f)
                
                # Convert forecast data to DataFrame
                if isinstance(forecast_data, dict):
                    try:
                        forecast_df = pd.DataFrame.from_dict(forecast_data)
                        logger.info(f"Loaded forecast data (dict of lists): {len(forecast_df)} grid points")
                    except Exception:
                        found = False
                        for v in forecast_data.values():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                forecast_df = pd.DataFrame(v)
                                logger.info(f"Loaded forecast data (list of dicts): {len(forecast_df)} grid points")
                                found = True
                                break
                        if not found:
                            logger.error(f"Unsupported forecast JSON structure")
                            exit(1)
                elif isinstance(forecast_data, list):
                    forecast_df = pd.DataFrame(forecast_data)
                    logger.info(f"Loaded forecast data (list of dicts): {len(forecast_df)} grid points")
                else:
                    logger.error(f"Unsupported forecast JSON structure: {type(forecast_data)}")
                    exit(1)
            else:
                logger.error(f"Forecast data file does not exist: {forecast_data_path}")
                exit(1)
        else:
            # Generate demo map using archived observations
            logger.info("No forecast data provided, generating demo map from archived observations")
            df = load_all_archived_data(args.archive_dir, args.hrrr_dir, include_hrrr=not args.no_hrrr)
            
            if df.empty:
                logger.error("No data available!")
                exit(1)
            
            # Use most recent observations
            latest_time = df['timestamp'].max()
            forecast_df = df[df['timestamp'] == latest_time].copy()
            logger.info(f"Using {len(forecast_df)} observations from {latest_time}")
        
        # Prepare features for prediction
        feature_cols = model_dict['feature_names']
        
        # Check for missing required features
        missing_cols = [col for col in feature_cols if col not in forecast_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            logger.info("Required features: " + ", ".join(feature_cols))
            exit(1)
        
        # Predict fire danger
        fire_danger_scores = predict_fire_danger(model_dict, forecast_df)
        
        # Create grid for plotting
        lon_min, lon_max = -95.8, -89.1
        lat_min, lat_max = 35.8, 40.8
        grid_lon = np.linspace(lon_min, lon_max, 400)
        grid_lat = np.linspace(lat_min, lat_max, 400)
        lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
        
        # Interpolate fire danger to grid
        from scipy.interpolate import griddata
        points = np.column_stack((forecast_df['longitude'].values, forecast_df['latitude'].values))
        fire_danger_grid = griddata(
            points, fire_danger_scores, 
            (lon_grid, lat_grid), 
            method='cubic', 
            fill_value=np.nan
        )
        
        # Smooth
        fire_danger_grid = gaussian_filter(fire_danger_grid, sigma=1.5)
        
        # Generate map
        project_dir = Path(__file__).resolve().parent.parent
        model_run = datetime.now()
        valid_time = datetime.now()
        
        generate_fire_danger_map(
            fire_danger_grid=fire_danger_grid,
            lon=lon_grid,
            lat=lat_grid,
            output_path=Path(args.output),
            model_run_date=model_run,
            valid_date=valid_time,
            project_dir=project_dir
        )
        
        logger.info(f"Map saved to {args.output}")
    
    else:
        logger.info("No command specified.")
        logger.info("Use --train to train model using Missouri AOP criteria")
        logger.info("Use --generate-map to generate fire danger map")
        logger.info("\nExamples:")
        logger.info("  python forecast/firedangermodel.py --train")
        logger.info("  python forecast/firedangermodel.py --train --no-hrrr")
        logger.info("  python forecast/firedangermodel.py --generate-map --forecast-data data.json")
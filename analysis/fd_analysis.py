import cartopy.crs as ccrs
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from shapely.ops import unary_union
from shapely.affinity import translate
from shapely.geometry import box, Point, LineString, mapping
import json
import re
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

# Reuse map helpers from DailyForecast to ensure consistent styling
try:
    from api.forecast.DailyForecast import add_boundaries, add_title_and_branding
except Exception:
    # Fallback: try alternate import path
    try:
        from forecast.DailyForecast import add_boundaries, add_title_and_branding
    except Exception:
        add_boundaries = None
        add_title_and_branding = None

if add_title_and_branding is None:
    def add_title_and_branding(fig, title, subtitle, description, RUN_DATE, SCRIPT_DIR):
        """Local fallback to add title, description and branding similar to DailyForecast.

        Keeps formatting consistent when DailyForecast helper is unavailable.
        """
        try:
            font_paths = [
                str(SCRIPT_DIR.parent / 'assets/Montserrat/static/Montserrat-Regular.ttf'),
                str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf'),
                str(SCRIPT_DIR.parent / 'assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf')
            ]
            for font_path in font_paths:
                if Path(font_path).exists():
                    font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'Montserrat'
        except Exception:
            pass

        try:
            fig.text(0.99, 0.97, title, fontsize=26, fontweight='bold', ha='right', va='top', fontname='Plus Jakarta Sans')
            fig.text(0.99, 0.90, subtitle, fontsize=16, ha='right', va='top', fontname='Montserrat')
            fig.text(0.99, 0.62, description, fontsize=10, ha='right', va='top', linespacing=1.6, fontname='Montserrat')
            fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight='bold', ha='left', va='bottom', fontname='Montserrat')
        except Exception:
            # Best-effort text placement
            try:
                fig.suptitle(title)
            except Exception:
                pass

        # Add logo if available
        try:
            svg_path = str(SCRIPT_DIR.parent / 'assets/LightBackGroundLogo.svg')
            if Path(svg_path).exists():
                png_bytes = cairosvg.svg2png(url=svg_path)
                image = mpimg.imread(BytesIO(png_bytes), format='png')
                imagebox = OffsetImage(image, zoom=0.03)
                ab = AnnotationBbox(imagebox, (0.99, 0.01), frameon=False, xycoords='figure fraction', box_alignment=(1, 0))
                plt.gca().add_artist(ab)
        except Exception:
            pass
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
    ax = plt.axes([0.05, 0.05, 0.9, 0.9], projection=map_crs)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_extent(extent, crs=data_crs)

    # hillshade_path = PROJECT_DIR / 'assets/hillshade.tif'
    # if hillshade_path.exists():
    #    add_hillshade(ax, hillshade_path, extent, data_crs, alpha=0.3, cmap='gray')
    
    map_extent = ax.get_extent(crs=data_crs)  
    bbox = box(*map_extent)
    
    counties = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp')
    if counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", 
                    facecolor='none', linewidth=1, zorder=5)

    
    missouriborder = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
    if missouriborder.crs != data_crs.proj4_init:
        missouriborder = missouriborder.to_crs(data_crs.proj4_init)
    ax.add_geometries(missouriborder.geometry, crs=data_crs, edgecolor="#000000", 
                    facecolor='none', linewidth=1, zorder=6)

    
    return fig, ax, data_crs, map_crs, mapdpi


def calculate_fire_danger_vector(fm, rh, wind_kts):
    """Vectorized fire danger calculation matching DailyForecast rules.

    Returns integer array with values 0 (Low) .. 4 (Extreme).
    """
    fm = np.array(fm)
    rh = np.array(rh)
    wind = np.array(wind_kts)

    risk = np.zeros(fm.shape, dtype=int)

    # EXTREME (4)
    mask_extreme = (fm < 7) & (rh < 20) & (wind >= 30)
    risk[mask_extreme] = 4

    # CRITICAL (3)
    mask_critical = (fm < 9) & (rh < 25) & (wind >= 15)
    risk[mask_critical & ~mask_extreme] = 3

    # ELEVATED (2)
    mask_elev = (fm < 9) & ((rh < 45) | (wind >= 10))
    risk[mask_elev & ~(mask_extreme | mask_critical)] = 2

    # MODERATE (1)
    mask_mod = (fm >= 9) & (fm < 15) & (rh < 50) & (wind >= 10)
    risk[mask_mod & ~(mask_extreme | mask_critical | mask_elev)] = 1

    # LOW (0) is default
    return risk


def calculate_fire_danger(fm, rh, wind_kts):
    """Compatibility wrapper returning a single integer danger level.

    Uses the vectorized `calculate_fire_danger_vector` under the hood so
    the same rules are applied here as in DailyForecast.
    """
    try:
        arr = calculate_fire_danger_vector([fm], [rh], [wind_kts])
        return int(arr[0])
    except Exception:
        # Fallback conservative return (Low)
        return 0



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

load_dotenv()

def find_endofday_json(preferred_date=None):
    """Search likely EndOfDay folders for endofday_raw_YYYYMMDD.json and return the chosen Path or None."""
    # quiet: locate EndOfDay JSON without verbose logging
    candidates = [
        SCRIPT_DIR / '..' / 'archive' / 'EndOfDay',      # api/../archive/EndOfDay
        SCRIPT_DIR.parent.parent / 'archive' / 'EndOfDay',
        SCRIPT_DIR / 'archive' / 'EndOfDay',             # api/analysis/archive/EndOfDay
        Path('/app') / 'archive' / 'EndOfDay',           # in-container absolute
        Path.cwd() / 'archive' / 'EndOfDay',             # cwd/archive/EndOfDay
    ]
    tried = []
    for c in candidates:
        try:
            d = c.resolve()
        except Exception:
            tried.append(str(c))
            continue
        if not d.exists():
            tried.append(str(d))
            continue
        files = sorted(d.glob('endofday_raw_*.json'))
        # found files in candidate directory
        if not files:
            tried.append(str(d))
            continue
        if preferred_date:
            key = preferred_date.strftime('%Y%m%d')
            matched = [f for f in files if key in f.name]
            if matched:
                return matched[-1]
        return files[-1]
    return None


def load_stations_from_endofday(preferred_date=None):
    p = find_endofday_json(preferred_date)
    if p is None:
        return []
    try:
        data = json.loads(p.read_text())
    except Exception:
        return []
    # expected shapes: NOAA/RAWS-style top-level 'STATION' list, or {'stations': [...] }, or a list
    if isinstance(data, dict) and 'STATION' in data:
        return data['STATION']
    if isinstance(data, dict) and 'stations' in data:
        return data['stations']
    if isinstance(data, list):
        return data
    # maybe mapping of id->station
    if isinstance(data, dict):
        vals = [v for v in data.values() if isinstance(v, dict) and 'observations' in v]
        if vals:
            return vals
    return []


def _collect_and_diagnose(preferred_date=None):
    stations = load_stations_from_endofday(preferred_date)
    # EndOfDay STATION records use 'STATE' and uppercase keys; filter by state instead
    filtered_stations = [s for s in stations if (s.get('STATE') == 'MO' or s.get('state') == 'MO')]

    # Diagnostics
    total_stations = len(stations)
    total_filtered = len(filtered_stations)
    with_rh = 0
    with_latlon = 0
    rh_missing_examples = []
    sample = []
    for s in stations[:80]:
        sid = s.get('station_id') or s.get('STID') or s.get('id')
        sample.append({
            'id': sid,
            'network': s.get('network'),
            'lat': s.get('latitude') or s.get('LATITUDE') or s.get('lat'),
            'lon': s.get('longitude') or s.get('LONGITUDE') or s.get('lon'),
            'observations_keys': list(s.get('observations', {}).keys()) if isinstance(s.get('observations'), dict) else None
        })
        # use helper (may get added later) via local logic for now
        rh_val = None
        obs = s.get('observations') or s.get('OBSERVATIONS') or {}
        if isinstance(obs, dict):
            for k, val in obs.items():
                if 'fuel_moisture' in k.lower():
                    # handle dict-wrapped series
                    if isinstance(val, dict):
                        arr = val.get('value') or val.get('values') or None
                        if isinstance(arr, list) and arr:
                            cleaned = [x for x in arr if x is not None]
                            if cleaned:
                                try:
                                    rh_val = float(min(cleaned))
                                    break
                                except Exception:
                                    pass
                        for nested in val.values():
                            if isinstance(nested, list):
                                cleaned = [x for x in nested if x is not None]
                                if cleaned:
                                    try:
                                        rh_val = float(min(cleaned))
                                        break
                                    except Exception:
                                        pass
                    elif isinstance(val, list):
                        cleaned = [x for x in val if x is not None]
                        if cleaned:
                            try:
                                rh_val = float(min(cleaned))
                                break
                            except Exception:
                                pass
                    else:
                        try:
                            rh_val = float(val)
                            break
                        except Exception:
                            pass
        # also check top-level station keys like 'relative_humidity_set_1'
        if rh_val is None:
            for k, val in s.items():
                if isinstance(k, str) and 'fuel_moisture' in k.lower():
                    if val is None:
                        continue
                    if isinstance(val, dict):
                        arr = val.get('value') or val.get('values') or None
                        if isinstance(arr, list) and arr:
                            cleaned = [x for x in arr if x is not None]
                            if cleaned:
                                try:
                                    rh_val = float(min(cleaned))
                                    break
                                except Exception:
                                    pass
                        for nested in val.values():
                            if isinstance(nested, list):
                                cleaned = [x for x in nested if x is not None]
                                if cleaned:
                                    try:
                                        rh_val = float(min(cleaned))
                                        break
                                    except Exception:
                                        pass
                    elif isinstance(val, list):
                        cleaned = [x for x in val if x is not None]
                        if cleaned:
                            try:
                                rh_val = float(min(cleaned))
                                break
                            except Exception:
                                pass
                    else:
                        try:
                            rh_val = float(val)
                            break
                        except Exception:
                            pass
        if rh_val is not None:
            with_rh += 1
        lat = s.get('latitude') or s.get('LATITUDE') or s.get('lat')
        lon = s.get('longitude') or s.get('LONGITUDE') or s.get('lon')
        if lat is not None and lon is not None:
            with_latlon += 1
        if rh_val is None:
            rh_missing_examples.append(sid)

    # Diagnostics collected in `sample` and `rh_missing_examples` during development.
    # Suppress verbose debug output in production runs.

    return stations, filtered_stations


def main(preferred_date=None, out_png='analysis/images/fm_analysis.png'):
    stations, filtered_stations = _collect_and_diagnose(preferred_date)

    # Build points list for interpolation
    points = []
    for s in filtered_stations:
        obs = s.get('observations') or s.get('OBSERVATIONS') or {}
        # extract min relative humidity from available keys (including relative_humidity_set_1)
        rh = None
        if isinstance(obs, dict):
            for k, val in obs.items():
                if 'fuel_moisture' in k.lower():
                    if val is None:
                        continue
                    if isinstance(val, dict):
                        arr = val.get('value') or val.get('values') or None
                        if isinstance(arr, list) and arr:
                            cleaned = [x for x in arr if x is not None]
                            if cleaned:
                                try:
                                    rh = float(min(cleaned))
                                    break
                                except Exception:
                                    pass
                        for nested in val.values():
                            if isinstance(nested, list):
                                cleaned = [x for x in nested if x is not None]
                                if cleaned:
                                    try:
                                        rh = float(min(cleaned))
                                        break
                                    except Exception:
                                        pass
                    elif isinstance(val, list):
                        cleaned = [x for x in val if x is not None]
                        if cleaned:
                            try:
                                rh = float(min(cleaned))
                                break
                            except Exception:
                                pass
                    else:
                        try:
                            rh = float(val)
                            break
                        except Exception:
                            pass
        # also check top-level keys
        if rh is None:
            for k, val in s.items():
                if isinstance(k, str) and 'fuel_moisture' in k.lower():
                    if val is None:
                        continue
                    if isinstance(val, dict):
                        arr = val.get('value') or val.get('values') or None
                        if isinstance(arr, list) and arr:
                            cleaned = [x for x in arr if x is not None]
                            if cleaned:
                                try:
                                    rh = float(min(cleaned))
                                    break
                                except Exception:
                                    pass
                        for nested in val.values():
                            if isinstance(nested, list):
                                cleaned = [x for x in nested if x is not None]
                                if cleaned:
                                    try:
                                        rh = float(min(cleaned))
                                        break
                                    except Exception:
                                        pass
                    elif isinstance(val, list):
                        cleaned = [x for x in val if x is not None]
                        if cleaned:
                            try:
                                rh = float(min(cleaned))
                                break
                            except Exception:
                                pass
                    else:
                        try:
                            rh = float(val)
                            break
                        except Exception:
                            pass
        if rh is not None:
            try:
                lon = float(s.get('longitude') or s.get('LONGITUDE') or s.get('lon'))
                lat = float(s.get('latitude') or s.get('LATITUDE') or s.get('lat'))
            except Exception:
                continue
            capped_fm = min(float(rh), 30.0)
            points.append((lon, lat, capped_fm))

    fig, ax, data_crs, map_crs, mapdpi = generate_basemap()

    lon_min, lon_max, lat_min, lat_max = -95.8, -89.1, 35.8, 40.8
    grid_lon = np.linspace(lon_min, lon_max, 400)
    grid_lat = np.linspace(lat_min, lat_max, 400)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    if points:
        points_lon = [p[0] for p in points]
        points_lat = [p[1] for p in points]
        values = [p[2] for p in points]

        # Interpolation and smoothing
        rbf = Rbf(points_lon, points_lat, values, function='multiquadric', smooth=0.005)
        grid_values = rbf(grid_lon_mesh, grid_lat_mesh)
        grid_values = np.minimum(grid_values, 100)
        grid_values = gaussian_filter(grid_values, sigma=0)

        # Masking to Missouri border
        missouriborder = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
        if missouriborder.crs != data_crs.proj4_init:
            missouriborder = missouriborder.to_crs(data_crs.proj4_init)
        if not missouriborder.empty:
            missouri_geom = missouriborder.geometry.iloc[0]
            grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
            within_mask = gpd.GeoSeries(grid_points).within(missouri_geom).values.reshape(grid_lon_mesh.shape)
            grid_values[~within_mask] = np.nan

        # Use same FM colormap and bounds as DailyForecast
        colors_and_positions = [
            (0.0, '#4D0000'),
            (0.1, '#8B0000'),
            (0.2, '#DC143C'),
            (0.233, '#FF4500'),
            (0.267, '#FF6347'),
            (0.30, '#FF8C00'),
            (0.333, '#FFA500'),
            (0.367, '#FFB347'),
            (0.40, '#FFD700'),
            (0.50, '#FFED4E'),
            (0.60, '#F0E68C'),
            (0.70, '#C8E6C9'),
            (0.80, '#81C784'),
            (0.90, '#4CAF50'),
            (1.0, '#2E7D32')
        ]
        positions = [x[0] for x in colors_and_positions]
        fm_colors = [x[1] for x in colors_and_positions]
        fm_cmap = LinearSegmentedColormap.from_list('fm_enhanced', list(zip(positions, fm_colors)), N=512)
        bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,18,21,24,27,30]
        norm = mcolors.BoundaryNorm(bounds, fm_cmap.N)

        cs = ax.contourf(
            grid_lon_mesh, grid_lat_mesh, grid_values, transform=data_crs,
            levels=bounds, cmap=fm_cmap, norm=norm, alpha=0.75, zorder=7, antialiased=True, extend='both'
        )
        # Use standard left-side colorbar placement to match other maps
        cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
        cbar = plt.colorbar(cs, cax=cax, label='Fuel Moisture (%)', ticks=[0,3,6,9,12,15,18,21,24,27,30])

    # station markers + labels
    if filtered_stations:
        label_positions = []
        min_dist = 0.12
        for s in filtered_stations:
            obs = s.get('observations') or {}
            rh = None
            if isinstance(obs, dict):
                r = obs.get('relative_humidity')
                if isinstance(r, dict):
                    rh = r.get('value')
                else:
                    rh = r
            if rh is None:
                continue
            try:
                lon = float(s.get('longitude') or s.get('LONGITUDE') or s.get('lon'))
                lat = float(s.get('latitude') or s.get('LATITUDE') or s.get('lat'))
            except Exception:
                continue
            pos = np.array([lon, lat])
            if all(np.linalg.norm(pos - np.array(lp)) > min_dist for lp in label_positions):
                ax.scatter(lon, lat, transform=data_crs, color='black', s=20, zorder=10, edgecolor='white', linewidth=1)
                ax.text(lon, lat + 0.03, f"{int(round(float(rh)))}%", transform=data_crs, fontsize=9,
                        color='black', zorder=11, ha='center', va='bottom', path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
                label_positions.append(pos)

    # Determine title_date for maps (from EndOfDay filename or preferred_date)
    try:
        p = find_endofday_json(preferred_date)
        title_date = None
        if p is not None:
            m = re.search(r'endofday_raw_(\d{8})', p.name)
            if m:
                try:
                    title_date = pd.to_datetime(m.group(1), format='%Y%m%d').strftime('%Y-%m-%d')
                except Exception:
                    title_date = None
        if title_date is None and preferred_date is not None:
            try:
                title_date = pd.to_datetime(preferred_date).strftime('%Y-%m-%d')
            except Exception:
                title_date = None
        if title_date is None:
            title_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    except Exception:
        title_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    # Close the FM figure — this analysis focuses on peak fire danger only
    plt.close(fig)
    runtime_sec = time.time() - start_time
    print(f"Script runtime: {runtime_sec:.2f} seconds")

    # ------------------ Peak Fire Danger from EndOfDay station peaks ------------------
    try:
        # Build per-station peak fire danger by scanning observation series
        station_peaks = []
        # Do not print per-station diagnostics during normal execution
        for s in filtered_stations:
            try:
                lon = float(s.get('longitude') or s.get('LONGITUDE') or s.get('lon'))
                lat = float(s.get('latitude') or s.get('LATITUDE') or s.get('lat'))
            except Exception:
                continue

            obs = s.get('observations') or s.get('OBSERVATIONS') or {}

            def extract_series(d, candidates):
                # Return the first found list-like series or single value wrapped in list
                if not isinstance(d, dict):
                    return None
                for k in candidates:
                    if k in d:
                        v = d.get(k)
                        if v is None:
                            continue
                        if isinstance(v, dict):
                            arr = v.get('value') or v.get('values') or v.get('values_list') or None
                            if isinstance(arr, list) and arr:
                                return [x for x in arr if x is not None]
                            # sometimes nested lists
                            for nested in v.values():
                                if isinstance(nested, list) and nested:
                                    return [x for x in nested if x is not None]
                        if isinstance(v, list):
                            cleaned = [x for x in v if x is not None]
                            if cleaned:
                                return cleaned
                        # scalar
                        try:
                            return [float(v)]
                        except Exception:
                            continue
                return None

            fm_series = extract_series(obs, ['fuel_moisture', 'fuel_moisture_set_1', 'fuel_moisture_set_2'])
            rh_series = extract_series(obs, ['relative_humidity', 'relative_humidity_set_1', 'rh'])
            wind_series = extract_series(obs, ['wind_speed', 'wind_speed_set_1', 'wind_gust', 'wind_speed_mean'])

            def to_float_list(series):
                if series is None:
                    return []
                out = []
                for v in series:
                    try:
                        out.append(float(v))
                    except Exception:
                        out.append(None)
                return out

            fm_list = to_float_list(fm_series)
            rh_list = to_float_list(rh_series)
            wind_list = to_float_list(wind_series)

            # quiet: skip per-station debug printing

            peak = None

            # If FM missing but RH present, estimate FM from RH (simple heuristic)
            if (not fm_list) and rh_list:
                try:
                    fm_list = [min(30.0, max(3.0, 3.0 + 0.25 * float(r))) if r is not None else None for r in rh_list]
                except Exception:
                    pass

            # Helper to convert wind to knots
            def wind_to_kts(vals):
                cleaned = [v for v in vals if v is not None]
                if not cleaned:
                    return []
                vmax = max(cleaned)
                if vmax <= 25:  # likely m/s
                    return [v * 1.94384 if v is not None else None for v in vals]
                if vmax <= 80:  # likely knots
                    return [v if v is not None else None for v in vals]
                # assume mph
                return [v * 0.868976 if v is not None else None for v in vals]

            wind_kts_list = wind_to_kts(wind_list)

            # If we have time-series alignment, iterate by index
            max_len = max(len(fm_list), len(rh_list), len(wind_kts_list))
            if max_len > 0:
                for i in range(max_len):
                    try:
                        fm = fm_list[i] if i < len(fm_list) else None
                        rh = rh_list[i] if i < len(rh_list) else None
                        w = wind_kts_list[i] if i < len(wind_kts_list) else None
                    except Exception:
                        fm = rh = w = None
                    if fm is None or rh is None or w is None:
                        continue
                    try:
                        danger = int(calculate_fire_danger(float(fm), float(rh), float(w)))
                    except Exception:
                        continue
                    if peak is None or danger > peak:
                        peak = danger

            # Fallback: use single-value heuristics
            if peak is None:
                try:
                    fm = (min([v for v in fm_list if v is not None]) if fm_list else None)
                    rh = (min([v for v in rh_list if v is not None]) if rh_list else None)
                    w = (max([v for v in wind_kts_list if v is not None]) if wind_kts_list else None)
                    if fm is not None and rh is not None and w is not None:
                        peak = int(calculate_fire_danger(float(fm), float(rh), float(w)))
                except Exception:
                    peak = None

            if peak is not None:
                station_peaks.append((lon, lat, peak))

        # Interpolate station peak danger to grid and plot like DailyForecast
        # If no station peaks were found from time-series, attempt a single-value fallback
        if not station_peaks:
            for s in filtered_stations:
                try:
                    lon = float(s.get('longitude') or s.get('LONGITUDE') or s.get('lon'))
                    lat = float(s.get('latitude') or s.get('LATITUDE') or s.get('lat'))
                except Exception:
                    continue
                obs = s.get('observations') or s.get('OBSERVATIONS') or {}
                # try to get a scalar RH or FM or wind
                rh = None
                fm = None
                w = None
                if isinstance(obs, dict):
                    # RH
                    for key in ('relative_humidity', 'relative_humidity_set_1', 'rh'):
                        v = obs.get(key)
                        if isinstance(v, dict):
                            vv = v.get('value') or v.get('values')
                            if isinstance(vv, list) and vv:
                                rh = vv[0]
                                break
                        elif isinstance(v, list) and v:
                            rh = v[0]
                            break
                        elif v is not None:
                            try:
                                rh = float(v); break
                            except Exception:
                                pass
                    # FM
                    for key in ('fuel_moisture', 'fuel_moisture_set_1'):
                        v = obs.get(key)
                        if isinstance(v, dict):
                            vv = v.get('value') or v.get('values')
                            if isinstance(vv, list) and vv:
                                fm = vv[0]; break
                        elif isinstance(v, list) and v:
                            fm = v[0]; break
                        elif v is not None:
                            try:
                                fm = float(v); break
                            except Exception:
                                pass
                    # Wind
                    for key in ('wind_speed', 'wind_speed_set_1', 'wind_gust', 'wind_speed_mean'):
                        v = obs.get(key)
                        if isinstance(v, dict):
                            vv = v.get('value') or v.get('values')
                            if isinstance(vv, list) and vv:
                                w = vv[0]; break
                        elif isinstance(v, list) and v:
                            w = v[0]; break
                        elif v is not None:
                            try:
                                w = float(v); break
                            except Exception:
                                pass
                # Estimate FM from RH if necessary
                if fm is None and rh is not None:
                    try:
                        fm = min(30.0, max(3.0, 3.0 + 0.25 * float(rh)))
                    except Exception:
                        fm = None
                # If still missing, check top-level station keys for common names
                if rh is None or fm is None or w is None:
                    for k, v in s.items():
                        if rh is None and isinstance(k, str) and ('relative_humidity' in k.lower() or k.lower().startswith('rh')):
                            try:
                                if isinstance(v, dict):
                                    vv = v.get('value') or v.get('values')
                                    if isinstance(vv, list) and vv:
                                        rh = vv[0]
                                        continue
                                if isinstance(v, list) and v:
                                    rh = v[0]; continue
                                rh = float(v); continue
                            except Exception:
                                pass
                        if fm is None and isinstance(k, str) and 'fuel_moisture' in k.lower():
                            try:
                                if isinstance(v, dict):
                                    vv = v.get('value') or v.get('values')
                                    if isinstance(vv, list) and vv:
                                        fm = vv[0]; continue
                                if isinstance(v, list) and v:
                                    fm = v[0]; continue
                                fm = float(v); continue
                            except Exception:
                                pass
                        if w is None and isinstance(k, str) and ('wind' in k.lower() or 'gust' in k.lower()):
                            try:
                                if isinstance(v, dict):
                                    vv = v.get('value') or v.get('values')
                                    if isinstance(vv, list) and vv:
                                        w = vv[0]; continue
                                if isinstance(v, list) and v:
                                    w = v[0]; continue
                                w = float(v); continue
                            except Exception:
                                pass
                # Convert wind to kts if plausible (assume m/s if small)
                if w is not None:
                    try:
                        wv = float(w)
                        if wv <= 25:
                            w_kts = wv * 1.94384
                        elif wv <= 80:
                            w_kts = wv
                        else:
                            w_kts = wv * 0.868976
                    except Exception:
                        w_kts = None
                else:
                    w_kts = None
                # Require at least RH; estimate FM from RH if missing; default missing wind to 0 kts
                if rh is not None:
                    if fm is None:
                        try:
                            fm = min(30.0, max(3.0, 3.0 + 0.25 * float(rh)))
                        except Exception:
                            fm = None
                    if fm is None:
                        continue
                    if w_kts is None:
                        w_kts = 0.0
                    try:
                        danger = int(calculate_fire_danger(float(fm), float(rh), float(w_kts)))
                        station_peaks.append((lon, lat, danger))
                    except Exception:
                        pass

        if station_peaks:
            plon = [p[0] for p in station_peaks]
            plat = [p[1] for p in station_peaks]
            pval = [p[2] for p in station_peaks]

            # Interpolate with RBF and smooth
            rbf_p = Rbf(plon, plat, pval, function='multiquadric', smooth=0.01)
            peak_grid = rbf_p(grid_lon_mesh, grid_lat_mesh)
            peak_grid = gaussian_filter(peak_grid, sigma=0.6)

            # Mask to Missouri
            missouriborder = gpd.read_file(PROJECT_DIR / 'maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp')
            if missouriborder.crs != data_crs.proj4_init:
                missouriborder = missouriborder.to_crs(data_crs.proj4_init)
            if not missouriborder.empty:
                missouri_geom = missouriborder.geometry.iloc[0]
                grid_points = [Point(lon, lat) for lon, lat in zip(grid_lon_mesh.ravel(), grid_lat_mesh.ravel())]
                within_mask = gpd.GeoSeries(grid_points).within(missouri_geom).values.reshape(grid_lon_mesh.shape)
                peak_grid[~within_mask] = np.nan

            # Plot using same colors/bins as DailyForecast peak map
            colors = ["#90EE90", '#FFED4E', '#FFA500', '#FF0000', '#8B0000']
            labels = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
            bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(bins, len(colors))

            fig2, ax2, data_crs2, map_crs2, mapdpi2 = generate_basemap()
            cs2 = ax2.contourf(grid_lon_mesh, grid_lat_mesh, peak_grid, transform=data_crs,
                               levels=bins, cmap=cmap, norm=norm, alpha=0.7, zorder=7, antialiased=True)
            ax2.contour(grid_lon_mesh, grid_lat_mesh, peak_grid, transform=data_crs,
                        levels=bins[1:-1], colors='black', linewidths=0.3, alpha=0.2, zorder=8)
            if add_boundaries:
                add_boundaries(ax2, data_crs, PROJECT_DIR)
            cax2 = fig2.add_axes([0.2, 0.08, 0.02, 0.6])
            cbar2 = plt.colorbar(cs2, cax=cax2, label='Fire Danger Level')
            cbar2.set_ticks([0,1,2,3,4])
            cbar2.set_ticklabels(labels)
            if add_title_and_branding:
                add_title_and_branding(
                    fig2,
                    "Missouri Peak Fire Danger (Observed)",
                    f"Analysis date: {title_date}",
                    "Based on fuel moisture, wind, and\n relative humidity observations. No snowfall or water is \n factored into these.",
                    pd.Timestamp.now(),
                    SCRIPT_DIR
                )
            # Save peak fire danger to the primary output path (args.out)
            peak_out = out_png
            fig2.savefig(peak_out, dpi=mapdpi, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig2)
            print(f"Peak fire danger map written to: {peak_out}")
        else:
            print("No station peak fire danger values found; peak map not generated.")

    except Exception as e:
        print(f"Failed to generate peak fire danger from EndOfDay: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Missouri RH filtered map from EndOfDay JSON')
    parser.add_argument('--date', '-d', help='Target date YYYY-MM-DD to load specific EndOfDay file', required=False)
    parser.add_argument('--out', help='Output PNG path', default='analysis/images/firedanger_analysis.png')
    args = parser.parse_args()
    preferred = None
    if args.date:
        try:
            preferred = pd.to_datetime(args.date).date()
        except Exception:
            preferred = None
    main(preferred_date=preferred, out_png=args.out)

import cartopy.crs as ccrs
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
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



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

load_dotenv()

def find_endofday_json(preferred_date=None):
    """Search likely EndOfDay folders for endofday_raw_YYYYMMDD.json and return the chosen Path or None."""
    print(f"[find_endofday_json] start; preferred_date={preferred_date}")
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
        except Exception as e:
            print(f"[find_endofday_json] resolve failed for {c}: {e}")
            tried.append(str(c))
            continue
        print(f"[find_endofday_json] checking directory: {d}")
        if not d.exists():
            print(f"[find_endofday_json] directory does not exist: {d}")
            tried.append(str(d))
            continue
        files = sorted(d.glob('endofday_raw_*.json'))
        print(f"[find_endofday_json] found {len(files)} files in {d}")
        if not files:
            tried.append(str(d))
            continue
        if preferred_date:
            key = preferred_date.strftime('%Y%m%d')
            print(f"[find_endofday_json] looking for key '{key}' in filenames")
            matched = [f for f in files if key in f.name]
            print(f"[find_endofday_json] matched {len(matched)} files for key '{key}'")
            if matched:
                chosen = matched[-1]
                print(f"[find_endofday_json] returning matched file: {chosen}")
                return chosen
        chosen = files[-1]
        print(f"[find_endofday_json] returning latest file: {chosen}")
        return chosen
    print(f"[find_endofday_json] no file found; tried directories: {tried}")
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
    with_wind = 0
    with_latlon = 0
    wind_missing_examples = []
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
        wind_val = None
        obs = s.get('observations') or s.get('OBSERVATIONS') or {}
        if isinstance(obs, dict):
            for k, val in obs.items():
                    if 'wind_speed' in k.lower():
                    # handle dict-wrapped series
                        if isinstance(val, dict):
                            arr = val.get('value') or val.get('values') or None
                            if isinstance(arr, list) and arr:
                                cleaned = [x for x in arr if x is not None]
                                if cleaned:
                                    try:
                                        wind_val = float(max(cleaned))
                                        break
                                    except Exception:
                                        pass
                            for nested in val.values():
                                if isinstance(nested, list):
                                    cleaned = [x for x in nested if x is not None]
                                    if cleaned:
                                        try:
                                            wind_val = float(max(cleaned))
                                            break
                                        except Exception:
                                            pass
                    elif isinstance(val, list):
                        cleaned = [x for x in val if x is not None]
                        if cleaned:
                            try:
                                wind_val = float(max(cleaned))
                                break
                            except Exception:
                                pass
                    else:
                        try:
                            wind_val = float(val)
                            break
                        except Exception:
                            pass
        # also check top-level station keys for wind values
        if wind_val is None:
            for k, val in s.items():
                if isinstance(k, str) and ('wind_speed' in k.lower() or k.lower().startswith('wind')):
                    if val is None:
                        continue
                    if isinstance(val, dict):
                        arr = val.get('value') or val.get('values') or None
                        if isinstance(arr, list) and arr:
                            cleaned = [x for x in arr if x is not None]
                            if cleaned:
                                try:
                                    wind_val = float(max(cleaned))
                                    break
                                except Exception:
                                    pass
                        for nested in val.values():
                            if isinstance(nested, list):
                                cleaned = [x for x in nested if x is not None]
                                if cleaned:
                                    try:
                                        wind_val = float(max(cleaned))
                                        break
                                    except Exception:
                                        pass
                    elif isinstance(val, list):
                        cleaned = [x for x in val if x is not None]
                        if cleaned:
                            try:
                                wind_val = float(max(cleaned))
                                break
                            except Exception:
                                pass
                    else:
                        try:
                            wind_val = float(val)
                            break
                        except Exception:
                            pass
        if wind_val is not None:
            with_wind += 1
        lat = s.get('latitude') or s.get('LATITUDE') or s.get('lat')
        lon = s.get('longitude') or s.get('LONGITUDE') or s.get('lon')
        if lat is not None and lon is not None:
            with_latlon += 1
        if wind_val is None:
            wind_missing_examples.append(sid)

    print(f"Loaded stations={total_stations}, filtered={total_filtered}, with_wind={with_wind}, with_latlon={with_latlon}")
    os.makedirs('logs', exist_ok=True)
    try:
        with open('logs/endofday_sample.json', 'w') as fh:
            json.dump({'sample': sample, 'wind_missing_examples': wind_missing_examples[:80]}, fh, indent=2)
    except Exception:
        pass

    return stations, filtered_stations


def main(preferred_date=None, out_png='analysis/images/mo-wind.png'):
    stations, filtered_stations = _collect_and_diagnose(preferred_date)

    # Build points list for interpolation
    points = []
    for s in filtered_stations:
        obs = s.get('observations') or s.get('OBSERVATIONS') or {}
        # extract max wind speed from available keys (including wind_speed_set_1)
        wind = None
        if isinstance(obs, dict):
            for k, val in obs.items():
                    if 'wind_speed' in k.lower():
                        if val is None:
                            continue
                        if isinstance(val, dict):
                            arr = val.get('value') or val.get('values') or None
                            if isinstance(arr, list) and arr:
                                cleaned = [x for x in arr if x is not None]
                                if cleaned:
                                    try:
                                        wind = float(max(cleaned))
                                        break
                                    except Exception:
                                        pass
                            for nested in val.values():
                                if isinstance(nested, list):
                                    cleaned = [x for x in nested if x is not None]
                                    if cleaned:
                                        try:
                                            wind = float(max(cleaned))
                                            break
                                        except Exception:
                                            pass
                    elif isinstance(val, list):
                        cleaned = [x for x in val if x is not None]
                        if cleaned:
                            try:
                                wind = float(max(cleaned))
                                break
                            except Exception:
                                pass
                    else:
                        try:
                            wind = float(val)
                            break
                        except Exception:
                            pass
        # also check top-level keys for wind values
            if wind is None:
                for k, val in s.items():
                    if isinstance(k, str) and ('wind_speed' in k.lower() or k.lower().startswith('wind')):
                        if val is None:
                            continue
                        if isinstance(val, dict):
                            arr = val.get('value') or val.get('values') or None
                            if isinstance(arr, list) and arr:
                                cleaned = [x for x in arr if x is not None]
                                if cleaned:
                                    try:
                                        wind = float(max(cleaned))
                                        break
                                    except Exception:
                                        pass
                            for nested in val.values():
                                if isinstance(nested, list):
                                    cleaned = [x for x in nested if x is not None]
                                    if cleaned:
                                        try:
                                            wind = float(max(cleaned))
                                            break
                                        except Exception:
                                            pass
                        elif isinstance(val, list):
                            cleaned = [x for x in val if x is not None]
                            if cleaned:
                                try:
                                    wind = float(max(cleaned))
                                    break
                                except Exception:
                                    pass
                        else:
                            try:
                                wind = float(val)
                                break
                            except Exception:
                                pass
        if wind is not None:
            try:
                lon = float(s.get('longitude') or s.get('LONGITUDE') or s.get('lon'))
                lat = float(s.get('latitude') or s.get('LATITUDE') or s.get('lat'))
            except Exception:
                continue
            points.append((lon, lat, float(wind)))

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

        cs = ax.contourf(
            grid_lon_mesh, grid_lat_mesh, grid_values, transform=data_crs,
            levels=np.linspace(0, 101, 256), cmap='RdYlGn', alpha=0.75, zorder=7, antialiased=True
        )
        cax = fig.add_axes([0.2, 0.08, 0.02, 0.6])
        plt.colorbar(cs, cax=cax, label='Wind Speed (mph)', ticks=np.arange(0, 102, 5))

    # station markers + labels (show wind speed)
    if filtered_stations:
        label_positions = []
        min_dist = 0.12
        for s in filtered_stations:
            obs = s.get('observations') or {}
            wind_label = None
            if isinstance(obs, dict):
                r = obs.get('wind_speed') or obs.get('wind')
                if isinstance(r, dict):
                    wind_label = r.get('value') or r.get('values')
                else:
                    wind_label = r
            if wind_label is None:
                # try top-level keys
                for k, v in s.items():
                    if isinstance(k, str) and ('wind_speed' in k.lower() or k.lower().startswith('wind')):
                        if isinstance(v, dict):
                            wind_label = v.get('value') or v.get('values')
                        else:
                            wind_label = v
                        break
            # if wind_label is a list or dict, try to extract a sensible value
            try:
                if isinstance(wind_label, list):
                    cleaned = [x for x in wind_label if x is not None]
                    if cleaned:
                        wind_val_for_label = float(max(cleaned))
                    else:
                        continue
                elif isinstance(wind_label, dict):
                    arr = wind_label.get('value') or wind_label.get('values')
                    if isinstance(arr, list) and arr:
                        wind_val_for_label = float(max([x for x in arr if x is not None]))
                    else:
                        continue
                else:
                    wind_val_for_label = float(wind_label)
            except Exception:
                continue
            try:
                lon = float(s.get('longitude') or s.get('LONGITUDE') or s.get('lon'))
                lat = float(s.get('latitude') or s.get('LATITUDE') or s.get('lat'))
            except Exception:
                continue
            pos = np.array([lon, lat])
            if all(np.linalg.norm(pos - np.array(lp)) > min_dist for lp in label_positions):
                ax.scatter(lon, lat, transform=data_crs, color='black', s=20, zorder=10, edgecolor='white', linewidth=1)
                ax.text(lon, lat + 0.03, f"{int(round(float(wind_val_for_label)))} mph", transform=data_crs, fontsize=9,
                        color='black', zorder=11, ha='center', va='bottom', path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
                label_positions.append(pos)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    # Add a title with the observation date (try to infer from EndOfDay filename, fallback to preferred_date or today)
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
        ax.set_title(f"Observed Maximum Wind Speed — {title_date}", fontsize=18, fontweight='bold', pad=12)
    except Exception:
        pass
    fig.savefig(out_png, dpi=mapdpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    runtime_sec = time.time() - start_time
    print(f"Wind Speed Filtered Map updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M CT')}")
    print(f"Script runtime: {runtime_sec:.2f} seconds")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate Missouri wind-speed filtered map from EndOfDay JSON')
    parser.add_argument('--date', '-d', help='Target date YYYY-MM-DD to load specific EndOfDay file', required=False)
    parser.add_argument('--out', help='Output PNG path', default='analysis/images/wind_analysis.png')
    args = parser.parse_args()
    preferred = None
    if args.date:
        try:
            preferred = pd.to_datetime(args.date).date()
        except Exception:
            preferred = None
    main(preferred_date=preferred, out_png=args.out)

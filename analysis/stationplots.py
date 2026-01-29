import os
import glob
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import importlib.util
from scipy.interpolate import griddata
from rasterio.transform import from_bounds
import rasterio

from pathlib import Path

# Robustly find project root so paths work both on host and inside container.
FILE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = None
for p in [FILE_DIR] + list(FILE_DIR.parents)[:5]:
    # Heuristic: project root contains either 'archive' or 'api' or 'maps'
    if (p / 'archive').is_dir() or (p / 'api').is_dir() or (p / 'maps').is_dir():
        PROJECT_DIR = str(p)
        break
if PROJECT_DIR is None:
    PROJECT_DIR = str(FILE_DIR.parent)

# Ensure project root is on sys.path so `api.*` imports work when running
# the script directly (e.g., `python analysis/stationplots.py`).
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

def generate_basemap():
    """Create a Cartopy basemap matching fuelmoisturemap's style.

    Returns: fig, ax, data_crs, map_crs, mapdpi
    """
    pixelw, pixelh, mapdpi = 2048, 1152, 144
    center_lon, center_lat = -92.5, 38.5
    extent = (-95.8, -89.1, 35.8, 40.8)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    fig = plt.figure(figsize=(pixelw / mapdpi, pixelh / mapdpi), dpi=mapdpi, facecolor='#E8E8E8')
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)
    return fig, ax, data_crs, map_crs, mapdpi


def load_archive_records(archive_path_or_dir, target_date=None):
    """Load station observation records from JSON files in archive_dir.
    Expects each file to contain a list or dict of observations with fields
    like station_id, time, lat, lon, rh, fuel_moisture, wind_speed, temp.
    If target_date is provided (YYYY-MM-DD), only include that date.
    """
    # archive_path_or_dir may be a single file or a directory containing many jsons
    if os.path.isfile(archive_path_or_dir):
        files = [archive_path_or_dir]
    else:
        files = sorted(glob.glob(os.path.join(archive_path_or_dir, '*.json')))
    records = []
    for f in files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)
            # Handle NOAA/RAWS-style archive with top-level 'STATION' list
            if isinstance(data, dict) and 'STATION' in data:
                for st in data.get('STATION', []):
                    stid = st.get('STID') or st.get('ID')
                    lat = st.get('LATITUDE')
                    lon = st.get('LONGITUDE')
                    obs = st.get('OBSERVATIONS', {})
                    # date_time array
                    times = obs.get('date_time') or []

                    # helper to find obs key by substring
                    def find_key(containing):
                        for k in obs.keys():
                            if containing in k:
                                return k
                        return None

                    rh_key = find_key('relative_humidity')
                    fm_key = find_key('fuel_moisture')
                    ws_key = find_key('wind_speed')
                    t_key = find_key('air_temp')

                    for i, t in enumerate(times):
                        rec = {}
                        rec['station_id'] = stid
                        try:
                            rec['time'] = pd.to_datetime(t)
                        except Exception:
                            rec['time'] = None
                        try:
                            rec['lat'] = float(lat) if lat is not None else None
                        except Exception:
                            rec['lat'] = None
                        try:
                            rec['lon'] = float(lon) if lon is not None else None
                        except Exception:
                            rec['lon'] = None

                        def get_obs_value(key):
                            if not key:
                                return None
                            arr = obs.get(key)
                            if arr is None:
                                return None
                            # some archives wrap arrays inside dict with set keys
                            if isinstance(arr, dict):
                                # try to extract first nested array
                                for v in arr.values():
                                    if isinstance(v, list) and len(v) > i:
                                        return v[i]
                                return None
                            if isinstance(arr, list):
                                return arr[i] if i < len(arr) else None
                            return None

                        rec['rh'] = get_obs_value(rh_key)
                        rec['fuel_moisture'] = get_obs_value(fm_key)
                        rec['wind_speed'] = get_obs_value(ws_key)
                        rec['temp'] = get_obs_value(t_key)
                        records.append(rec)
                # finished processing STATION-style file
            else:
                # data can be list or dict
                if isinstance(data, dict):
                    items = data.get('observations') or data.get('data') or [data]
                else:
                    items = data
                for it in items:
                    # normalize minimal fields
                    rec = {}
                    rec['station_id'] = it.get('station_id') or it.get('id') or it.get('station')
                    time_s = it.get('time') or it.get('timestamp') or it.get('date')
                    try:
                        rec['time'] = pd.to_datetime(time_s)
                    except Exception:
                        rec['time'] = None
                    rec['lat'] = it.get('lat') or it.get('latitude')
                    rec['lon'] = it.get('lon') or it.get('longitude')
                    # possible field names
                    rec['rh'] = it.get('rh') or it.get('relative_humidity') or it.get('humidity')
                    rec['fuel_moisture'] = it.get('fuel_moisture') or it.get('fm') or it.get('fm10')
                    rec['wind_speed'] = it.get('wind_speed') or it.get('wind_kts') or it.get('wind')
                    rec['temp'] = it.get('temp') or it.get('temperature')
                    records.append(rec)
        except Exception:
            continue

    if target_date:
        t0 = pd.to_datetime(target_date).normalize()
        t1 = t0 + pd.Timedelta(days=1)
        records = [r for r in records if r['time'] is not None and t0 <= r['time'] < t1]

    return pd.DataFrame(records)


def compute_station_daily_stats(df):
    """Compute per-station min/max/mean for rh and fuel_moisture."""
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby('station_id')
    out = grp.agg({
        'lat': 'first', 'lon': 'first',
        'rh': ['min', 'max', 'mean'],
        'fuel_moisture': ['min', 'max', 'mean'],
        'wind_speed': ['min', 'max', 'mean'],
        'temp': ['min', 'max', 'mean']
    })
    # flatten columns
    out.columns = ['_'.join(c).strip() if isinstance(c, tuple) else c for c in out.columns.values]
    out = out.reset_index()
    return out


def interpolate_to_grid(points_lon, points_lat, values, bbox, nx=300, ny=300):
    """Interpolate scattered point values to a regular lon/lat grid covering bbox.
    bbox = (minx, miny, maxx, maxy)
    Returns lon_grid, lat_grid, value_grid
    """
    minx, miny, maxx, maxy = bbox
    xi = np.linspace(minx, maxx, nx)
    yi = np.linspace(miny, maxy, ny)
    lon_grid, lat_grid = np.meshgrid(xi, yi)

    # mask invalid
    mask_valid = np.isfinite(points_lon) & np.isfinite(points_lat) & np.isfinite(values)
    if mask_valid.sum() < 3:
        return lon_grid, lat_grid, np.full(lon_grid.shape, np.nan)

    pts = np.column_stack((points_lon[mask_valid], points_lat[mask_valid]))
    vals = values[mask_valid]
    grid_z = griddata(pts, vals, (lon_grid, lat_grid), method='linear')
    # fill remaining with nearest
    nanmask = ~np.isfinite(grid_z)
    if nanmask.any():
        grid_z[nanmask] = griddata(pts, vals, (lon_grid[nanmask], lat_grid[nanmask]), method='nearest')

    return lon_grid, lat_grid, grid_z


def save_geotiff(path, data, bbox):
    minx, miny, maxx, maxy = bbox
    height, width = data.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=data.dtype,
        crs='EPSG:4326', transform=transform, compress='lzw', nodata=np.nan
    ) as dst:
        dst.write(np.flipud(data), 1)


def plot_map(lon_grid, lat_grid, value_grid, counties_shp, out_png, title, cmap='RdYlGn'):
    # Use shared basemap generator for consistent styling
    fig, ax, data_crs, map_crs, mapdpi = generate_basemap()

    # contourf expects lon/lat grid in PlateCarree (data_crs)
    cs = ax.contourf(
        lon_grid, lat_grid, value_grid, transform=data_crs,
        levels=256, cmap=cmap, alpha=0.8, zorder=7
    )

    # overlay counties
    try:
        counties = gpd.read_file(counties_shp)
        if counties.crs is None:
            counties = counties.set_crs('EPSG:4326')
        counties = counties.to_crs(data_crs.proj4_init)
        ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor='none', linewidth=1, zorder=8)
    except Exception:
        pass

    # station points (if available in grid extent) -- optional, left out here
    fig.colorbar(cs, ax=ax, shrink=0.6)
    ax.set_title(title)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=mapdpi, bbox_inches='tight')
    plt.close(fig)


def main(target_date=None, df=None):
    # Paths (archive and outputs live under the repo root)
    archive_dir = os.path.join(PROJECT_DIR, 'archive', 'raw_data')
    out_dir = os.path.join(PROJECT_DIR, 'data', 'station_maps')
    counties_shp = os.path.join(PROJECT_DIR, 'maps', 'shapefiles', 'MO_County_Boundaries', 'MO_County_Boundaries.shp')

    endofday_dir = os.path.join(PROJECT_DIR, 'archive', 'EndOfDay')
    # don't assume raw_data exists; we'll search EndOfDay first for dated archives
    # If a DataFrame was provided (caller preloaded a file), use it.
    if df is not None:
        # provided df should already be filtered by date if desired
        pass
    else:
        # If no date provided, pick the newest archive JSON file and use it
        if target_date is None:
            files = sorted(glob.glob(os.path.join(archive_dir, '*.json')), key=os.path.getmtime)
            if not files:
                print('No archive files found in', archive_dir)
                return
            latest_file = files[-1]
            print('Using latest archive file:', latest_file)
            df = load_archive_records(latest_file, target_date=None)
        else:
            # Try to find an archive file matching the date in its filename (YYYY-MM-DD or YYYYMMDD)
            date_str = target_date
            date_str_compact = None
            try:
                date_obj = pd.to_datetime(target_date)
                date_str_compact = date_obj.strftime('%Y%m%d')
                date_str_iso = date_obj.strftime('%Y-%m-%d')
            except Exception:
                date_str_iso = target_date

            matched_file = None
            search_dirs = [endofday_dir, archive_dir]
            for d in search_dirs:
                if not os.path.isdir(d):
                    continue
                for pattern in [f'*{date_str}*.json', f'*{date_str_compact}*.json', f'*{date_str_iso}*.json']:
                    if pattern is None:
                        continue
                    files = sorted(glob.glob(os.path.join(d, pattern)), key=os.path.getmtime)
                    if files:
                        matched_file = files[-1]
                        print('Found archive file matching date in', d + ':', matched_file)
                        break
                if matched_file:
                    break

            if matched_file:
                df = load_archive_records(matched_file, target_date=None)
            else:
                # debugging info for why matching may have failed
                print('No archive filename matched for patterns:', [f'*{date_str}*.json', f'*{date_str_compact}*.json', f'*{date_str_iso}*.json'])
                print('Searched in:', archive_dir)
                # fallback to loading all and filtering by record timestamps
                df = load_archive_records(archive_dir, target_date)

    # If no records found for the requested date, fall back to the latest file
    if df is None or df.empty:
        print('No records found for date', target_date)
        files = sorted(glob.glob(os.path.join(archive_dir, '*.json')), key=os.path.getmtime)
        if files:
            latest_file = files[-1]
            print('Falling back to latest archive file:', latest_file)
            df = load_archive_records(latest_file, target_date=None)
        else:
            return

    stats = compute_station_daily_stats(df)
    # Save station summary json
    os.makedirs(out_dir, exist_ok=True)
    summary_json = os.path.join(out_dir, f'station_daily_summary_{target_date or "latest"}.json')
    stats.to_json(summary_json, orient='records', date_format='iso')

    # Interpolate min RH and min fuel_moisture
    bbox = (-95.8, 35.8, -89.1, 40.8)

    for field in ['rh_min', 'fuel_moisture_min']:
        if field not in stats.columns:
            continue
        # find sensible lon/lat columns produced by aggregation
        if 'lon_first' in stats.columns and 'lat_first' in stats.columns:
            pts_lon = stats['lon_first'].values
            pts_lat = stats['lat_first'].values
        elif 'lon' in stats.columns and 'lat' in stats.columns:
            pts_lon = stats['lon'].values
            pts_lat = stats['lat'].values
        else:
            pts_lon = np.array([])
            pts_lat = np.array([])
        values = stats[field].values

        lon_grid, lat_grid, grid_z = interpolate_to_grid(pts_lon, pts_lat, values, bbox, nx=400, ny=400)
        # Save outputs
        png_path = os.path.join(out_dir, f'{field}_{target_date or "latest"}.png')
        title = f'{field} for {target_date or "latest"}'
        plot_map(lon_grid, lat_grid, grid_z, counties_shp, png_path, title)

    print('Saved station summary and maps to', out_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate station daily min/max maps from archive JSONs')
    parser.add_argument('--date', '-d', help='Target date in YYYY-MM-DD (optional). If omitted, latest archive file is used or yesterday by default')
    parser.add_argument('--file', '-f', help='Process a specific archive JSON file instead of the directory')
    args = parser.parse_args()

    if args.file:
        # process single file (ignores --date)
        main_target = None
        # call main by providing the file path directly to loader
        # reuse main by temporarily setting archive directory variable
        archive_dir = os.path.join(PROJECT_DIR, 'archive', 'raw_data')
        # call load directly
        df = load_archive_records(args.file, target_date=None)
        if df.empty:
            print('No records found in file', args.file)
        else:
            # pass the loaded DataFrame into main so it processes these records
            main(args.date, df=df)
    else:
        main(args.date)

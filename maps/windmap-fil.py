"""Missouri statewide sustained wind speed map (images/mo-windfilmap.png).

Station wind speed from the four map networks, anchored at the border by
neighbor-state stations, RBF-interpolated across the state, with a wind barb
at every station. Also exports gis/realtime/wind_speed.tif and reports into
the root status.json under 'WindFiltered'.

Basemap, boundaries, branding and the station feed come from
maps/station_plot.py. The interpolation is this map's own (smooth=0.005, no
blur) and is unchanged.
"""
import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

import station_plot as sp
import status_lock
from core.logging_setup import file_logger
from realtime_geotiff import export_continuous_rgba_geotiff
from station_rbf import buffer_points

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GRID_SIZE = 400


def now_ct():
    return pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')


# The raw /stations payload (via station_plot's short-lived shared cache, so
# this map doesn't re-fetch what another script in the same maps.sh cycle
# just fetched). Deliberately not sp.fetch_stations(): that also applies the
# neighbor-outlier QC, which would change this map's inputs.
all_stations = sp._read_station_cache(sp.STATION_CACHE_MAX_AGE_SECONDS)
if all_stations is None:
    all_stations = sp._fetch_and_cache_stations()
filtered_stations = [s for s in all_stations if s.get('network') in sp.VALID_NETWORKS]

points = []
for s in filtered_stations:
    ws = s.get('observations', {}).get('wind_speed', {}).get('value')
    if ws is not None:
        points.append((s['longitude'], s['latitude'], ws))

# Neighbor-state stations anchor the interpolation at the border (barbs stay Missouri-only).
points += buffer_points('wind_speed')

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
grid_lon_mesh, grid_lat_mesh, within_mask, grid_x_mesh, grid_y_mesh = sp._fill_grid(GRID_SIZE, map_crs, data_crs)

start_time = time.time()

if points:
    points_lon, points_lat, values = zip(*points)
    rbf = Rbf(points_lon, points_lat, values, function='multiquadric', smooth=0.005)
    # Cells outside Missouri are masked to NaN anyway, so only evaluate the
    # RBF inside the state (identical values, ~45% fewer cells).
    if within_mask is not None:
        grid_values = np.full(grid_lon_mesh.shape, np.nan)
        grid_values[within_mask] = rbf(grid_lon_mesh[within_mask], grid_lat_mesh[within_mask])
    else:
        grid_values = rbf(grid_lon_mesh, grid_lat_mesh)
    grid_values = np.clip(grid_values, 0, 100)  # Cap wind speed at 100 mph

    geotiff_ok = export_continuous_rgba_geotiff(
        grid_values=grid_values,
        lon_mesh=grid_lon_mesh,
        lat_mesh=grid_lat_mesh,
        out_path=PROJECT_DIR / 'gis/realtime/wind_speed.tif',
        cmap_name='RdYlGn_r',
        vmin=0,
        vmax=51,
        description='Missouri realtime sustained wind speed (RGBA)',
        source='Synoptic observations + ShowMeFire interpolation',
        legend='Sustained wind speed (mph) mapped with RdYlGn_r, transparent outside Missouri',
    )
    if not geotiff_ok:
        print("Warning: Failed to export wind speed GeoTIFF")

    cs = ax.contourf(
        grid_x_mesh, grid_y_mesh, grid_values,
        levels=np.linspace(0, 51, 256),
        cmap=sp.opaque_blend_cmap('RdYlGn_r', 0.75), zorder=7, antialiased=False
    )
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    fig.colorbar(cs, cax=cax, label='Wind Speed (mph)', ticks=np.arange(0, 52, 5))

# A barb at every station reporting speed and direction. Not
# sp.plot_wind_barbs: that drops a station sitting at exactly the same
# coordinates as one already plotted, which this map never did.
lons, lats, us, vs = [], [], [], []
for s in filtered_stations:
    ws = s.get('observations', {}).get('wind_speed', {}).get('value')
    wd = s.get('observations', {}).get('wind_direction', {}).get('value')
    if ws is not None and wd is not None:
        lons.append(s['longitude'])
        lats.append(s['latitude'])
        # Wind blows from wd, so the vector points to wd + 180
        us.append(-ws * np.sin(np.radians(wd)))
        vs.append(-ws * np.cos(np.radians(wd)))
if lons:
    ax.barbs(np.array(lons), np.array(lats), np.array(us), np.array(vs), transform=data_crs, length=6, zorder=10, color='black')

# Title/subtitle/wordmark/logo from the shared layout; this map's description
# keeps its own (default) line spacing, so it is drawn here instead.
sp.add_branding(
    fig, ax,
    title="Missouri Statewide Sustained Wind Speed Map",
    subtitle=f"Valid Time: {now_ct()}",
    description="",
)
fig.text(
    0.99, 0.6,
    "Data Source: Official Sites Only from Synoptic Weather\n"
    "AWOS, RAWS, Missouri Mesonet\n\n"
    "Wind speed is a critical factor in fire behavior and spread.\n"
    "Values interpolated from available stations; local variations may occur.",
    fontsize=12, ha='right', va='top', fontname='Montserrat'
)

fig.savefig('images/mo-windfilmap.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

runtime_sec = time.time() - start_time
updated_msg = f"Wind Map updated at {now_ct()}"
runtime_msg = f"Script runtime: {runtime_sec:.2f} seconds"
print(updated_msg)
print(runtime_msg)

# Set up rotating log handler (max 5MB per file, keep 5 backup files)
logger = file_logger('windmap-fil', 'windmap-fil.log')
logger.info(updated_msg)
logger.info(runtime_msg)

plt.close(fig)

status_file = PROJECT_DIR / 'status.json'
status_lock.acquire(status_file)
try:
    with open(status_file, 'r') as f:
        status = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    status = {}

status['WindFiltered'] = {
    'last_update': now_ct(),
    'status': 'updated',
    'runtime_sec': round(runtime_sec, 2),
    'log': [updated_msg, runtime_msg],
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
status_lock.release()

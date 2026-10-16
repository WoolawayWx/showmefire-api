"""Missouri statewide relative humidity map (images/mo-rh.png).

Station RH (capped at 100%) from the four map networks, anchored at the
border by neighbor-state stations, RBF-interpolated across the state. Also
exports gis/realtime/relative_humidity.tif and reports into the root
status.json under 'HumidityFiltered'.

Basemap, boundaries, branding, station labels and the station feed come from
maps/station_plot.py. The interpolation is this map's own (sharper than
station_plot.interpolate_to_grid: smooth=0.005, no blur) and is unchanged.
"""
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf

import station_plot as sp
import status_lock
from realtime_geotiff import export_continuous_rgba_geotiff
from station_rbf import buffer_points

start_time = time.time()

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
    rh = s.get('observations', {}).get('relative_humidity', {}).get('value')
    if rh is not None:
        points.append((s['longitude'], s['latitude'], min(rh, 100)))  # Cap at 100%

# Neighbor-state stations anchor the interpolation at the border (markers/labels stay Missouri-only).
points += buffer_points('relative_humidity', cap=100)

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
grid_lon_mesh, grid_lat_mesh, within_mask, grid_x_mesh, grid_y_mesh = sp._fill_grid(GRID_SIZE, map_crs, data_crs)

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
    grid_values = np.minimum(grid_values, 100)  # Cap all values at 100

    geotiff_ok = export_continuous_rgba_geotiff(
        grid_values=grid_values,
        lon_mesh=grid_lon_mesh,
        lat_mesh=grid_lat_mesh,
        out_path=PROJECT_DIR / 'gis/realtime/relative_humidity.tif',
        cmap_name='RdYlGn',
        vmin=0,
        vmax=100,
        description='Missouri realtime relative humidity (RGBA)',
        source='Synoptic observations + ShowMeFire interpolation',
        legend='Relative Humidity (%) mapped with RdYlGn, transparent outside Missouri',
    )
    if not geotiff_ok:
        print("Warning: Failed to export relative humidity GeoTIFF")

    cs = ax.contourf(
        grid_x_mesh, grid_y_mesh, grid_values,
        levels=np.linspace(0, 101, 256),
        cmap=sp.opaque_blend_cmap('RdYlGn', 0.75), zorder=7, antialiased=False
    )
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    fig.colorbar(cs, cax=cax, label='Relative Humidity (%)', ticks=np.arange(0, 102, 5))

sp.plot_station_values(
    ax, data_crs, filtered_stations,
    field_path='observations.relative_humidity.value',
    formatter=lambda rh: f"{int(round(rh))}%",
)

# Title/subtitle/wordmark/logo from the shared layout; this map's description
# keeps its own (default) line spacing, so it is drawn here instead.
sp.add_branding(
    fig, ax,
    title="Missouri Statewide Relative Humidity Map",
    subtitle=f"Valid Time: {now_ct()}",
    description="",
)
fig.text(
    0.99, 0.6,
    "Data Source: Offical Sites Only from Synoptic Weather\n"
    "AWOS, RAWS, Missouri Mesonet\n\n"
    "Relative humidity is a key factor in fire weather and fuel drying.\n"
    "Values interpolated from available stations; local variations may occur.",
    fontsize=12, ha='right', va='top', fontname='Montserrat'
)

fig.savefig('images/mo-rh.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

runtime_sec = time.time() - start_time
updated_msg = f"RH% Filtered Map updated at {now_ct()}"
runtime_msg = f"Script runtime: {runtime_sec:.2f} seconds"
print(updated_msg)
print(runtime_msg)

# Set up rotating log handler (max 5MB per file, keep 5 backup files)
log_file = PROJECT_DIR / 'logs/rhmap-fil.log'
log_file.parent.mkdir(exist_ok=True)
logger = logging.getLogger('rhmap-fil')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
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

status['HumidityFiltered'] = {
    'last_update': now_ct(),
    'status': 'updated',
    'runtime_sec': round(runtime_sec, 2),
    'log': [updated_msg, runtime_msg],
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
status_lock.release()

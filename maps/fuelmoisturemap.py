"""Missouri statewide 10-hour fuel moisture map (images/mo-fuelmoisture.png).

RAWS fuel moisture (/stations/raws, including neighbor-state RAWS so the fill
reaches the border), RBF-interpolated across the state, with the 9% critical
threshold drawn as a dotted contour. Also exports
gis/realtime/fuel_moisture.tif and reports into the root status.json under
'RealTimeFuelMoisture'.

Basemap, boundaries and branding come from maps/station_plot.py. The station
feed (/stations/raws, not /stations), interpolation, and station dots/labels
(every MO RAWS dot, one-decimal labels, no decluttering) are this map's own
and unchanged.
"""
import json
import logging
import os
import time
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.interpolate import Rbf

import station_plot as sp
import status_lock
from core.logging_setup import file_logger
from realtime_geotiff import export_continuous_rgba_geotiff

start_time = time.time()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GRID_SIZE = 400


def now_ct():
    return pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')


load_dotenv()
port = os.getenv('PORT', '8000')
response = requests.get(f'http://localhost:{port}/stations/raws')
stations = response.json()['stations']

points = []
for s in stations:
    fm = s.get('observations', {}).get('fuel_moisture', {}).get('value')
    if fm is not None:
        points.append((s['longitude'], s['latitude'], fm))

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

    geotiff_ok = export_continuous_rgba_geotiff(
        grid_values=grid_values,
        lon_mesh=grid_lon_mesh,
        lat_mesh=grid_lat_mesh,
        out_path=PROJECT_DIR / 'gis/realtime/fuel_moisture.tif',
        cmap_name='RdYlGn',
        vmin=0,
        vmax=30,
        description='Missouri realtime fuel moisture (RGBA)',
        source='Synoptic RAWS observations + ShowMeFire interpolation',
        legend='Fuel Moisture (%) mapped with RdYlGn, transparent outside Missouri',
    )
    if not geotiff_ok:
        print("Warning: Failed to export fuel moisture GeoTIFF")

    cs = ax.contourf(
        grid_x_mesh, grid_y_mesh, grid_values,
        levels=np.linspace(0, 30, 256),
        cmap=sp.opaque_blend_cmap('RdYlGn', 0.75), zorder=7, antialiased=False
    )
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = fig.colorbar(cs, cax=cax, label='Fuel Moisture (%)', ticks=np.arange(0, 32, 2))
    # 9% critical threshold. zorder 7.5 (above the fill, below the county
    # lines): sp.generate_basemap() adds the county lines (zorder 8) first, so
    # a zorder-8 contour would now draw on top of them instead of under them.
    ax.contour(
        grid_x_mesh, grid_y_mesh, grid_values, levels=[9], colors='black',
        linestyles='dotted', linewidths=2, zorder=7.5
    )
    cbar.ax.axhline(y=9, color='black', linewidth=2.5, linestyle='--', zorder=10)

mo_stations = [s for s in stations if s.get('state') == 'MO']
ax.scatter(
    [s['longitude'] for s in mo_stations], [s['latitude'] for s in mo_stations],
    transform=data_crs, color='black', s=20, zorder=10, edgecolor='white', linewidth=1
)
for s in mo_stations:
    fm = s.get('observations', {}).get('fuel_moisture', {}).get('value')
    if fm is not None:
        ax.text(
            s['longitude'], s['latitude']+0.05, f"{fm:.1f}%",
            transform=data_crs, fontsize=9, color='black', zorder=11,
            ha='center', va='bottom',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='white')]
        )

# Title/subtitle/wordmark/logo from the shared layout; this map's description
# keeps its own (default) line spacing, so it is drawn here instead.
sp.add_branding(
    fig, ax,
    title="Missouri Statewide Fuel Moisture Map",
    subtitle=f"Valid Time: {now_ct()}",
    description="",
)
fig.text(
    0.99, 0.6,
    "Data Source: USFS RAWS stations from Synoptic Weather\n\n"
    "The 9% fuel moisture (black dotted line) is the critical threshold\n"
    "in place across much of Missouri, as defined in the Fire Weather AOP\n"
    "by NWS LSX, EAX, SGF, and PAH.\n\n"
    "Due to the low number of stations reporting fuel moisture,\n"
    "data may not be accurate, and local variations may occur.\n"
    "RAWS stations outside of Missouri are used in order\n to create a proper plot.",
    fontsize=12, ha='right', va='top', fontname='Montserrat'
)

fig.savefig('images/mo-fuelmoisture.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

runtime_sec = time.time() - start_time
updated_msg = f"Fuel Moisture updated at {now_ct()}"
runtime_msg = f"Script runtime: {runtime_sec:.2f} seconds"
print(updated_msg)
print(runtime_msg)

# Set up rotating log handler (max 5MB per file, keep 5 backup files)
logger = file_logger('fuelmoisturemap', 'fuelmoisturemap.log')
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

status['RealTimeFuelMoisture'] = {
    'last_update': now_ct(),
    'status': 'updated',
    'runtime_sec': round(runtime_sec, 2),
    'log': [updated_msg, runtime_msg],
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
status_lock.release()

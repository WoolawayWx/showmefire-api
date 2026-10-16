"""Missouri realtime fire danger (images/mo-realtimefiredanger.png) and
today's observed peak (images/mo-observedpeakfiredanger.png).

RH and wind from every /stations station, fuel moisture from RAWS
(/stations/raws), each RBF-interpolated, then classified cell by cell with the
canonical core.fire_danger rule. Also exports
gis/realtime/realtime_fire_danger.tif, folds the grid into the observed-peak
running max (maps/observed_peak_history.py), reports into the root
status.json under 'RealtimeFireDanger', and appends the station-count history
(maps/station_danger_history.py).

Basemap, boundaries and branding come from maps/station_plot.py. The
interpolation and classification math are unchanged; they are only evaluated
where the result is used (inside Missouri).
"""
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.interpolate import Rbf
from scipy.ndimage import binary_dilation, gaussian_filter

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.fire_danger import calculate_fire_danger
from core.fire_danger import CATEGORY_BINS, CATEGORY_HEX_COLORS, CATEGORY_LABELS, CATEGORY_RGBA_COLORS

import station_plot as sp
import status_lock
from realtime_geotiff import export_discrete_rgba_geotiff
from station_danger_history import (
    count_station_fire_danger_categories_from_grid,
    empty_station_fire_danger_summary,
    append_fire_danger_snapshot,
    export_fire_danger_daily_csv,
)
from observed_peak_history import update_observed_peak_grid, load_today_running_grid

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GRID_SIZE = 400
BLUR_SIGMA = 0.7

CRITERIA_TEXT = (
    "Fire Danger Criteria:\n"
    "\n"
    "Low: FM ≥ 15% (fuels too wet to spread significantly)\n\n"
    "Moderate: FM < 15% AND (RH < 45% OR Wind ≥ 10 kts)\n\n"
    "Elevated: FM < 9% AND\n"
    "  (RH < 35% & Wind ≥ 12 kts) OR (RH < 25% & Wind ≥ 5 kts)\n\n"
    "Critical: FM < 9% AND (RH < 25% & Wind ≥ 15 kts)\n\n"
    "Extreme: FM < 7% AND (RH < 20% & Wind ≥ 25 kts)\n\n"
    "Data Source: AWOS, RAWS, Missouri Mesonet, CWOP Stations\n"
    "For More Info, Visit ShowMeFire.org"
)

bins = list(CATEGORY_BINS)
labels = list(CATEGORY_LABELS)
cmap = ListedColormap(list(CATEGORY_HEX_COLORS))
norm = BoundaryNorm(bins, cmap.N)


def now_ct():
    return pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')


def draw_danger(fig, ax, grid_x_mesh, grid_y_mesh, grid):
    """Fire danger classes as a filled contour + categorical colorbar."""
    cs = ax.contourf(
        grid_x_mesh, grid_y_mesh, grid,
        levels=bins, cmap=sp.opaque_blend_cmap(cmap, 0.7), norm=norm, zorder=7, antialiased=True
    )
    cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
    cbar = fig.colorbar(cs, cax=cax, label='Fire Danger Level')
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(labels)


def brand(fig, ax, title, subtitle):
    # Title/subtitle/wordmark/logo from the shared layout; the criteria block
    # keeps its own position, size and line spacing, so it is drawn here.
    sp.add_branding(fig, ax, title=title, subtitle=subtitle, description="")
    fig.text(0.99, 0.62, CRITERIA_TEXT, fontsize=10, ha='right', va='top', linespacing=1.6, fontname='Montserrat')


def rbf_in(points, smooth, lon_mesh, lat_mesh, where):
    """The station RBF (multiquadric) evaluated only at the `where` cells;
    every other cell is 0 (never read: see the callers)."""
    rbf = Rbf([p[0] for p in points], [p[1] for p in points], [p[2] for p in points],
              function='multiquadric', smooth=smooth)
    grid = np.zeros(lon_mesh.shape)
    grid[where] = rbf(lon_mesh[where], lat_mesh[where])
    return grid


load_dotenv()
port = os.getenv('PORT', '8000')

# --- Use RAWS endpoint for fuel moisture ---
response = requests.get(f'http://localhost:{port}/stations/raws')
raws_stations = response.json()['stations']
station_danger_summary = empty_station_fire_danger_summary(raws_stations)
peak_running_grid = None

# All /stations stations (every network) for RH and wind - the raw payload via
# station_plot's short-lived shared cache. Deliberately not
# sp.fetch_stations(): that filters networks and applies the neighbor-outlier
# QC, which would change this map's inputs.
all_stations = sp._read_station_cache(sp.STATION_CACHE_MAX_AGE_SECONDS)
if all_stations is None:
    all_stations = sp._fetch_and_cache_stations()

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
grid_lon_mesh, grid_lat_mesh, within_mask, grid_x_mesh, grid_y_mesh = sp._fill_grid(GRID_SIZE, map_crs, data_crs)

# Collect data for interpolation
rh_points = []
wind_points = []
fuel_points = []  # RAWS stations with actual fuel moisture

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
        fuel_points.append((s['longitude'], s['latitude'], fm))

print(f"Data summary: {len(rh_points)} RH, {len(wind_points)} wind, "
      f"{len(fuel_points)} measured FM (RAWS only)")

start_time = time.time()

if rh_points and wind_points and fuel_points:
    # Only cells inside Missouri are classified (the rest are masked to NaN).
    # The blurred fields need their exact values within the blur radius
    # (gaussian_filter's default truncate=4 -> 3 cells at sigma 0.7) of those
    # cells, so the RBFs are evaluated on the state mask grown by that radius;
    # in-state results are identical to evaluating and blurring the full grid.
    inside = within_mask if within_mask is not None else np.ones(grid_lon_mesh.shape, dtype=bool)
    radius = int(4.0 * BLUR_SIGMA + 0.5)
    near = binary_dilation(inside, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool))

    rh_grid = gaussian_filter(rbf_in(rh_points, 0.01, grid_lon_mesh, grid_lat_mesh, near), sigma=BLUR_SIGMA)[inside]
    wind_grid = gaussian_filter(rbf_in(wind_points, 0.01, grid_lon_mesh, grid_lat_mesh, near), sigma=BLUR_SIGMA)[inside]
    fuel_grid = gaussian_filter(rbf_in(fuel_points, 0.01, grid_lon_mesh, grid_lat_mesh, near), sigma=BLUR_SIGMA)[inside]

    # Blend toward the (sharper, unblurred) measured fuel moisture near each RAWS station.
    measured_grid = rbf_in(fuel_points, 0.005, grid_lon_mesh, grid_lat_mesh, inside)[inside]
    lon_in, lat_in = grid_lon_mesh[inside], grid_lat_mesh[inside]
    influence_grid = np.zeros_like(lon_in)
    for mlon, mlat, _ in fuel_points:
        dist = np.sqrt((lon_in - mlon)**2 + (lat_in - mlat)**2)
        influence_grid = np.maximum(influence_grid, np.exp(-dist / 0.5))
    fuel_grid = influence_grid * measured_grid + (1 - influence_grid) * fuel_grid

    # Classify with the canonical rule (0=Low .. 4=Extreme); .tolist() hands it
    # plain floats, which is faster than numpy scalars and gives the same result.
    grid_values = np.full(grid_lon_mesh.shape, np.nan)
    grid_values[inside] = np.array([
        calculate_fire_danger(fm, rh, wind)
        for fm, rh, wind in zip(fuel_grid.tolist(), rh_grid.tolist(), wind_grid.tolist())
    ], dtype=float)

    station_danger_summary = count_station_fire_danger_categories_from_grid(
        stations=raws_stations,
        grid_values=grid_values,
        lon_mesh=grid_lon_mesh,
        lat_mesh=grid_lat_mesh,
        state_filter='MO',
    )

    geotiff_ok = export_discrete_rgba_geotiff(
        grid_values=grid_values,
        lon_mesh=grid_lon_mesh,
        lat_mesh=grid_lat_mesh,
        out_path=PROJECT_DIR / 'gis/realtime/realtime_fire_danger.tif',
        class_colors=CATEGORY_RGBA_COLORS,
        description='Missouri realtime fire danger classes (RGBA)',
        source='Synoptic observations + ShowMeFire fire danger model',
        legend='Low=green, Moderate=yellow, Elevated=orange, Critical=red, Extreme=dark red',
    )
    if not geotiff_ok:
        print("Warning: Failed to export realtime fire danger GeoTIFF")

    try:
        peak_status = update_observed_peak_grid(
            grid_values=grid_values,
            lon_mesh=grid_lon_mesh,
            lat_mesh=grid_lat_mesh,
            run_time_utc=datetime.now(timezone.utc),
        )
        print(f"Observed peak grid updated: {peak_status}")
        peak_running_grid = load_today_running_grid()

        # Day rollover: the PNG still on disk from the previous run is that
        # finished day's final rendered peak - archive it before this run's
        # savefig overwrites it with today's (fresh, just-reset) grid.
        if peak_status.get('archived_path'):
            try:
                prev_png = Path('images/mo-observedpeakfiredanger.png')
                if prev_png.exists():
                    archive_date = Path(peak_status['archived_path']).stem
                    png_archive_dir = Path('images/observed_peak/archive')
                    png_archive_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(prev_png, png_archive_dir / f'{archive_date}.png')
            except Exception as e:
                print(f"Warning: Failed to archive observed peak PNG: {e}")
    except Exception as e:
        print(f"Warning: Failed to update observed peak fire danger grid: {e}")

    draw_danger(fig, ax, grid_x_mesh, grid_y_mesh, grid_values)
else:
    print("Insufficient data for fire danger calculation.")

brand(fig, ax, "Missouri Fire Danger Assessment", f"Observations Analysis | Valid Time: {now_ct()}")
fig.savefig('images/mo-realtimefiredanger.png', dpi=mapdpi, bbox_inches=None, pad_inches=0)

if peak_running_grid is not None:
    try:
        peak_fig, peak_ax, _, _, peak_mapdpi = sp.generate_basemap()
        draw_danger(peak_fig, peak_ax, grid_x_mesh, grid_y_mesh, peak_running_grid)
        brand(
            peak_fig, peak_ax, "Missouri Peak Fire Danger (Today)",
            "Worst fire danger class reached at each location so far today\n"
            f"As of: {now_ct()}",
        )
        peak_fig.savefig('images/mo-observedpeakfiredanger.png', dpi=peak_mapdpi, bbox_inches=None, pad_inches=0)
        plt.close(peak_fig)
        print("Peak fire danger (today) map updated")
    except Exception as e:
        print(f"Warning: Failed to render peak fire danger PNG: {e}")

runtime_sec = time.time() - start_time
updated_msg = f"Fire Danger Map updated at {now_ct()}"
runtime_msg = f"Script runtime: {runtime_sec:.2f} seconds"
print(updated_msg)
print(runtime_msg)

# Set up rotating log handler (max 5MB per file, keep 5 backup files)
log_file = PROJECT_DIR / 'logs/realtimefiredanger.log'
log_file.parent.mkdir(exist_ok=True)
logger = logging.getLogger('realtimefiredanger')
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

counts = station_danger_summary['counts']
status['RealtimeFireDanger'] = {
    'last_update': now_ct(),
    'status': 'updated',
    'runtime_sec': round(runtime_sec, 2),
    'station_fire_danger_counts': counts,
    'total_mo_stations': station_danger_summary['total_mo_stations'],
    'classified_mo_stations': station_danger_summary['classified_mo_stations'],
    'unclassified_mo_stations': station_danger_summary['unclassified_mo_stations'],
    'log': [
        updated_msg,
        runtime_msg,
        (
            "Station fire danger counts: "
            f"Low={counts['Low']}, "
            f"Moderate={counts['Moderate']}, "
            f"Elevated={counts['Elevated']}, "
            f"Critical={counts['Critical']}, "
            f"Extreme={counts['Extreme']}"
        )
    ]
}

with open(status_file, 'w') as f:
    json.dump(status, f, indent=4)
status_lock.release()

try:
    history = append_fire_danger_snapshot(
        summary=station_danger_summary,
        run_time_utc=datetime.now(timezone.utc),
        keep_hours=48,
    )
    csv_path = export_fire_danger_daily_csv(history)
    logger.info(f"Saved fire danger 48h history snapshots: {len(history)}")
    logger.info(f"Updated daily fire danger CSV: {csv_path}")
except Exception as e:
    logger.error(f"Failed to persist fire danger history artifacts: {e}")

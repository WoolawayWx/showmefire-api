"""Dewpoint Temperature and Winds: dewpoint value + wind barb at each
station."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "dewpoint_winds_combo.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-dewpoint-winds-combo.png"

start_time = time.time()
stations = sp.fetch_stations()

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
sp.plot_filled_field(
    fig, ax, data_crs, stations,
    field_path="observations.dew_point_temperature.value",
    cmap="BrBG", label="Dewpoint Temperature (F)",
)
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.dew_point_temperature.value",
    formatter=lambda v: f"{v:.0f}",
)
sp.plot_wind_barbs(ax, data_crs, stations)

sp.add_branding(
    fig, ax,
    title="Dewpoint Temperature and Winds",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Dewpoint temperature (F) with wind barbs (speed/direction) at\n"
        "each reporting station."
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "dewpoint_winds_combo",
        "status": "updated",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Dewpoint + winds combo map updated ({plotted} stations, {runtime_sec:.2f}s)")

"""Current Dewpoint Temperature: station plot."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "dewpointmap.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-dewpointmap.png"

start_time = time.time()
stations = sp.fetch_stations()

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.dew_point_temperature.value",
    formatter=lambda v: f"{v:.0f}",
)

sp.add_branding(
    fig, ax,
    title="Missouri Dewpoint Temperature",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Dewpoint temperature (F) at each reporting station - a better\n"
        "measure of moisture content than relative humidity alone."
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "dewpoint_current",
        "status": "updated",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Dewpoint map updated ({plotted} stations, {runtime_sec:.2f}s)")

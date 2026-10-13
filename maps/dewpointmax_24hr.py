"""24-hour Maximum Dewpoint Temperature: a true trailing 24-hour rolling max
per station (station_history.get_rolling_extreme), distinct from the Air
Temperature category's "Today's Max" which resets at local midnight - this
product is always "the last 24 hours," regardless of calendar day."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps.station_history import get_rolling_extremes_batch

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "dewpointmax_24hr.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-dewpointmax-24hr.png"

start_time = time.time()
stations = sp.fetch_stations()

extremes = get_rolling_extremes_batch("dew_point_temperature_f", 24)
have_history = bool(extremes)
for station in stations:
    extreme = extremes.get(station.get("stid"))
    station.setdefault("observations", {})["dewpoint_max_24hr"] = {
        "value": extreme["max"] if extreme else None
    }

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
sp.plot_filled_field(
    fig, ax, data_crs, stations,
    field_path="observations.dewpoint_max_24hr.value",
    cmap="BrBG", label="Max Dewpoint Temperature (F)", vmin=-10, vmax=80,
)
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.dewpoint_max_24hr.value",
    formatter=lambda v: f"{v:.0f}",
    color="#8B0000",
)

status_note = "" if have_history else " - accumulating history, full coverage after 24h"
sp.add_branding(
    fig, ax,
    title="24-Hour Maximum Dewpoint Temperature",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Highest dewpoint temperature (F) recorded at each station over\n"
        "the trailing 24 hours (rolling window, not reset at midnight)." + status_note
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "dewpoint_max_24hr",
        "status": "updated" if have_history else "insufficient_history",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"24-hour max dewpoint map updated ({plotted} stations, {runtime_sec:.2f}s)")

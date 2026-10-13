"""Today's Minimum Air Temperature: each station's running min-so-far
(America/Chicago local day), from maps/station_history.py."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps.station_history import get_today_extremes_batch

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "airtempmin_today.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-airtempmin-today.png"

start_time = time.time()
stations = sp.fetch_stations()

extremes = get_today_extremes_batch("air_temp_f")
have_history = bool(extremes)
for station in stations:
    extreme = extremes.get(station.get("stid"))
    station.setdefault("observations", {})["air_temp_min_today"] = {
        "value": extreme["min"] if extreme else None
    }

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
sp.plot_filled_field(
    fig, ax, data_crs, stations,
    field_path="observations.air_temp_min_today.value",
    cmap="RdYlBu_r", label="Min Air Temperature (F)", vmin=-20, vmax=110,
)
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.air_temp_min_today.value",
    formatter=lambda v: f"{v:.0f}",
    color="#1D4ED8",
)

status_note = "" if have_history else " - accumulating history today"
sp.add_branding(
    fig, ax,
    title="Today's Minimum Air Temperature",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Lowest air temperature (F) recorded at each station so far today\n"
        "(resets at local midnight)." + status_note
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "air_temperature_min_today",
        "status": "updated" if have_history else "insufficient_history",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Today's min air temp map updated ({plotted} stations, {runtime_sec:.2f}s)")

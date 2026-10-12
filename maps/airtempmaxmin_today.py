"""Today's Maximum & Minimum Air Temperature: both values stacked at each
station dot, from maps/station_history.py."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps.station_history import get_today_extremes_batch

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "airtempmaxmin_today.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-airtempmaxmin-today.png"

start_time = time.time()
stations = sp.fetch_stations()

extremes = get_today_extremes_batch("air_temp_f")
have_history = bool(extremes)
for station in stations:
    extreme = extremes.get(station.get("stid"))
    obs = station.setdefault("observations", {})
    obs["air_temp_max_today"] = {"value": extreme["max"] if extreme else None}
    obs["air_temp_min_today"] = {"value": extreme["min"] if extreme else None}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
plotted = sp.plot_station_dual_values(
    ax, data_crs, stations,
    top_field_path="observations.air_temp_max_today.value",
    bottom_field_path="observations.air_temp_min_today.value",
    top_formatter=lambda v: f"{v:.0f}",
    bottom_formatter=lambda v: f"{v:.0f}",
)

status_note = "" if have_history else " - accumulating history today"
sp.add_branding(
    fig, ax,
    title="Today's Max & Min Air Temperature",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Highest (red, top) and lowest (blue, bottom) air temperature (F)\n"
        "recorded at each station so far today (resets at local midnight)." + status_note
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "air_temperature_maxmin_today",
        "status": "updated" if have_history else "insufficient_history",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Today's max/min air temp map updated ({plotted} stations, {runtime_sec:.2f}s)")

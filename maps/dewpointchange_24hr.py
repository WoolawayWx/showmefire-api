"""24-hour Dewpoint Temperature Change: current minus the value ~24 hours ago
per station, from maps/station_history.py."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps.station_history import get_changes_batch

LOOKBACK_HOURS = 24
STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "dewpointchange_24hr.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-dewpointchange-24hr.png"

start_time = time.time()
stations = sp.fetch_stations()

current_values = {
    station["stid"]: station["observations"]["dew_point_temperature"]["value"]
    for station in stations
    if station.get("stid") and station.get("observations", {}).get("dew_point_temperature", {}).get("value") is not None
}
changes = get_changes_batch("dew_point_temperature_f", LOOKBACK_HOURS, current_values)
have_history = bool(changes)
for station in stations:
    station.setdefault("observations", {})["dewpoint_change_24hr"] = {"value": changes.get(station.get("stid"))}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.dewpoint_change_24hr.value",
    formatter=lambda v: f"{v:+.0f}",
)

status_note = "" if have_history else " - accumulating history, full coverage after 24h"
sp.add_branding(
    fig, ax,
    title="24-Hour Dewpoint Temperature Change",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Change in dewpoint temperature (F) over the last 24 hours at\n"
        "each reporting station." + status_note
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "dewpoint_change_24hr",
        "status": "updated" if have_history else "insufficient_history",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"24-hour dewpoint change map updated ({plotted} stations, {runtime_sec:.2f}s)")

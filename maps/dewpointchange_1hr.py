"""1-hour Dewpoint Temperature Change: current minus the value ~1 hour ago
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

LOOKBACK_HOURS = 1
STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "dewpointchange_1hr.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-dewpointchange-1hr.png"

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
    station.setdefault("observations", {})["dewpoint_change_1hr"] = {"value": changes.get(station.get("stid"))}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
sp.plot_filled_field(
    fig, ax, data_crs, stations,
    field_path="observations.dewpoint_change_1hr.value",
    cmap="BrBG", diverging=True, label="1-Hr Dewpoint Change (F)", vmax=8,
)
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.dewpoint_change_1hr.value",
    formatter=lambda v: f"{v:+.0f}",
)

status_note = "" if have_history else " - accumulating history, full coverage after 1h"
sp.add_branding(
    fig, ax,
    title="1-Hour Dewpoint Temperature Change",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Change in dewpoint temperature (F) over the last hour at each\n"
        "reporting station. A sharp drop can signal a dry air intrusion." + status_note
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "dewpoint_change_1hr",
        "status": "updated" if have_history else "insufficient_history",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"1-hour dewpoint change map updated ({plotted} stations, {runtime_sec:.2f}s)")

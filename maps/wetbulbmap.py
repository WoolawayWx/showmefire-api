"""Wet-Bulb Temperature: current value, computed live per station via
services/derived_met.py's Stull (2011) approximation."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from services import derived_met

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "wetbulbmap.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-wetbulbmap.png"

start_time = time.time()
stations = sp.fetch_stations()

for station in stations:
    obs = station.get("observations", {})
    temp = obs.get("air_temp", {}).get("value")
    rh = obs.get("relative_humidity", {}).get("value")
    wet_bulb = derived_met.wet_bulb_f(temp, rh) if None not in (temp, rh) else None
    obs["wet_bulb"] = {"value": wet_bulb}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.wet_bulb.value",
    formatter=lambda v: f"{v:.0f}",
)

sp.add_branding(
    fig, ax,
    title="Missouri Wet-Bulb Temperature",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Wet-bulb temperature (F), Stull (2011) approximation from air\n"
        "temperature and relative humidity - accuracy degrades outside\n"
        "roughly -4 to 122F / 5-99% RH."
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "wet_bulb_current",
        "status": "updated",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Wet-bulb map updated ({plotted} stations, {runtime_sec:.2f}s)")

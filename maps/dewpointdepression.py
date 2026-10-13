"""Dewpoint Depression: air temperature minus dewpoint at each station - a
dryness/fuel-drying proxy. Computed live via services/derived_met.py."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from services import derived_met

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "dewpointdepression.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-dewpointdepression.png"

start_time = time.time()
stations = sp.fetch_stations()

for station in stations:
    obs = station.get("observations", {})
    temp = obs.get("air_temp", {}).get("value")
    dewpoint = obs.get("dew_point_temperature", {}).get("value")
    depression = derived_met.dewpoint_depression_f(temp, dewpoint) if None not in (temp, dewpoint) else None
    obs["dewpoint_depression"] = {"value": depression}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
sp.plot_filled_field(
    fig, ax, data_crs, stations,
    field_path="observations.dewpoint_depression.value",
    cmap="BrBG_r", label="Dewpoint Depression (F)", vmin=0, vmax=50,
)
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.dewpoint_depression.value",
    formatter=lambda v: f"{v:.0f}",
)

sp.add_branding(
    fig, ax,
    title="Missouri Dewpoint Depression",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "Air temperature minus dewpoint (F) at each station. Larger\n"
        "values indicate drier air - a useful fuel-drying proxy."
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "dewpoint_depression_current",
        "status": "updated",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Dewpoint depression map updated ({plotted} stations, {runtime_sec:.2f}s)")

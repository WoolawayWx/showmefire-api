"""Estimated Wet Bulb Globe Temperature (WBGT) Risk: current value, computed
live per station via services/derived_met.py.

Missouri's station network has no wet-globe thermometer - this is an
outdoor-estimate approximation from temperature/humidity/wind, NOT a true
WBGT measurement. Must stay labeled "Estimated" everywhere, matching
services/spread_rate.py's "Experimental" convention for its own model-only
product.
"""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from services import derived_met

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "wbgtmap.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-wbgtmap.png"

start_time = time.time()
stations = sp.fetch_stations()

for station in stations:
    obs = station.get("observations", {})
    temp = obs.get("air_temp", {}).get("value")
    rh = obs.get("relative_humidity", {}).get("value")
    wind = obs.get("wind_speed", {}).get("value")
    wbgt = derived_met.estimate_wbgt_f(temp, rh, wind) if None not in (temp, rh, wind) else None
    obs["wbgt_estimated"] = {"value": wbgt}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.wbgt_estimated.value",
    formatter=lambda v: f"{v:.0f}",
)

sp.add_branding(
    fig, ax,
    title="Estimated WBGT Risk",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "ESTIMATED Wet Bulb Globe Temperature (F) from air temperature,\n"
        "humidity, and wind - no station has a true wet-globe sensor.\n"
        "Treat as an approximation, not a measured WBGT reading."
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "wbgt_estimated_current",
        "status": "updated",
        "estimated": True,
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Estimated WBGT map updated ({plotted} stations, {runtime_sec:.2f}s)")

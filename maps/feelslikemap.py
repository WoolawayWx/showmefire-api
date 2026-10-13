"""Current Wind Chill / Heat Index: NWS-convention composite "feels like"
temperature (heat index when hot, wind chill when cold, plain temp between),
computed live per station via services/derived_met.py."""
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from services import derived_met

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "feelslikemap.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-feelslikemap.png"

start_time = time.time()
stations = sp.fetch_stations()

for station in stations:
    obs = station.get("observations", {})
    temp = obs.get("air_temp", {}).get("value")
    rh = obs.get("relative_humidity", {}).get("value")
    wind = obs.get("wind_speed", {}).get("value")
    feels_like = derived_met.feels_like_f(temp, rh, wind) if None not in (temp, rh, wind) else None
    obs["feels_like"] = {"value": feels_like}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
sp.plot_filled_field(
    fig, ax, data_crs, stations,
    field_path="observations.feels_like.value",
    cmap="RdYlBu_r", label="Feels Like (F)",
)
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.feels_like.value",
    formatter=lambda v: f"{v:.0f}",
)

sp.add_branding(
    fig, ax,
    title="Missouri Wind Chill / Heat Index",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "NWS \"feels like\" temperature (F): heat index at/above 80F,\n"
        "wind chill at/below 50F with wind 3mph+, plain air temperature\n"
        "in between."
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "feels_like_current",
        "status": "updated",
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Feels-like map updated ({plotted} stations, {runtime_sec:.2f}s)")

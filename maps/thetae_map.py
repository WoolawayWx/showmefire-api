"""Theta-E (equivalent potential temperature): current value, computed live
per station via services/derived_met.py.

Missouri stations don't reliably report their own station pressure (it's not
in services/synoptic.py's validated OBSERVATION_LIMITS), so pressure is
estimated from station elevation via derived_met.standard_atmosphere_pressure_hpa
- a climatological estimate that ignores the actual synoptic pressure
pattern. This product is ESTIMATED and must stay labeled as such, same
convention as wbgtmap.py.
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

STATUS_PATH = PROJECT_DIR / "data" / "current_conditions" / "status" / "thetae_map.json"
PNG_PATH = PROJECT_DIR / "images" / "mo-thetaemap.png"
FEET_TO_METERS = 0.3048

start_time = time.time()
stations = sp.fetch_stations()

for station in stations:
    obs = station.get("observations", {})
    temp_f = obs.get("air_temp", {}).get("value")
    dewpoint_f = obs.get("dew_point_temperature", {}).get("value")
    elevation_ft = station.get("elevation")
    theta_e = None
    if None not in (temp_f, dewpoint_f, elevation_ft):
        pressure_hpa = derived_met.standard_atmosphere_pressure_hpa(elevation_ft * FEET_TO_METERS)
        theta_e_k = derived_met.theta_e_k(
            derived_met.f_to_c(temp_f), derived_met.f_to_c(dewpoint_f), pressure_hpa,
        )
        theta_e = theta_e_k - 273.15  # report in Celsius, standard meteorological convention
    obs["theta_e"] = {"value": theta_e}

fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
sp.plot_filled_field(
    fig, ax, data_crs, stations,
    field_path="observations.theta_e.value",
    cmap="RdYlBu_r", label="Theta-E (C)", vmin=10, vmax=80,
)
plotted = sp.plot_station_values(
    ax, data_crs, stations,
    field_path="observations.theta_e.value",
    formatter=lambda v: f"{v:.0f}",
)

sp.add_branding(
    fig, ax,
    title="Estimated Theta-E (Equivalent Potential Temp.)",
    subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
    description=(
        "Data Source: Official Sites Only from Synoptic Weather\n"
        "AWOS, RAWS, Missouri Mesonet\n\n"
        "ESTIMATED theta-E (C), Bolton (1980). Station pressure isn't\n"
        "reliably available - pressure is estimated from elevation\n"
        "(standard atmosphere), so this is an approximation, not a true\n"
        "sounding-based theta-E."
    ),
)

runtime_sec = time.time() - start_time
sp.save_and_record_status(
    fig, PNG_PATH, STATUS_PATH,
    {
        "product": "theta_e_estimated_current",
        "status": "updated",
        "estimated": True,
        "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(runtime_sec, 2),
        "stations_plotted": plotted,
    },
)
print(f"Theta-E map updated ({plotted} stations, {runtime_sec:.2f}s)")

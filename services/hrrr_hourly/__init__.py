"""Hour-by-hour HRRR forecast product built from the 12z run (f00-f18).

Separate from the operational daily forecast: its own directories, status key
and file names, so a run here can never overwrite today's forecast outputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PRODUCT = "hrrr_hourly"
STATUS_KEY = "HrrrHourly"
CYCLE_HOUR = 12
FORECAST_HOURS = tuple(range(0, 19))
SCHEMA_VERSION = 1


def root_dir() -> Path:
    return Path(os.environ.get("SMF_HRRR_HOURLY_DIR", "data/hrrr_hourly"))


def images_dir() -> Path:
    return Path(os.environ.get("SMF_HRRR_HOURLY_IMAGES", "images/hrrr_hourly"))


def keep_days() -> int:
    return int(os.environ.get("SMF_HRRR_HOURLY_KEEP_DAYS", "14"))


def cycle_id(cycle_time) -> str:
    """'20261016_12z' for a run initialised at 2026-10-16 12:00 UTC."""
    return f"{cycle_time:%Y%m%d}_{cycle_time:%H}z"


@dataclass(frozen=True)
class Element:
    name: str
    long_name: str
    units: str
    scale: float            # quantisation step: values are rounded to this before storage
    source: str             # HRRR field(s) or formula
    derived: bool = False   # computed here rather than read from HRRR
    # rendering
    cmap: str = "viridis"
    vmin: float = 0.0
    vmax: float = 1.0
    ticks: tuple = ()
    image: bool = True
    categorical: bool = False


ELEMENTS: tuple[Element, ...] = (
    Element("mixing_height_ft", "Mixing height (boundary-layer depth)", "ft AGL", 10, "HRRR HPBL",
            cmap="YlGnBu", vmax=8000, ticks=tuple(range(0, 8001, 1000))),
    Element("holzworth_ft", "Mixing height, parcel method", "ft AGL", 10, "Holzworth from profile", True, image=False),
    Element("transport_speed_mph", "Transport wind speed (mean through mixed layer)", "mph", 0.1,
            "HRRR winds averaged 10 m to mixing height", True, cmap="YlOrRd", vmax=40, ticks=tuple(range(0, 41, 5))),
    Element("transport_dir_deg", "Transport wind direction (from)", "deg", 1, "HRRR winds", True, image=False),
    Element("ventilation_kt_ft", "Ventilation rate", "kt ft", 100, "transport wind x mixing height", True, image=False),
    Element("ventilation_category", "Ventilation category (0 Poor .. 4 Excellent)", "category", 1,
            "NWS Springfield categories", True, categorical=True),
    Element("inversion_strength_f", "Surface inversion strength", "degF", 0.1, "HRRR profile", True,
            cmap="PuBu", vmax=10, ticks=tuple(range(0, 11, 2))),
    Element("inversion_top_ft", "Surface inversion top", "ft AGL", 10, "HRRR profile", True, image=False),
    Element("lapse_rate_c_km", "0-1 km lapse rate", "degC/km", 0.05, "HRRR profile", True,
            cmap="RdYlBu_r", vmin=-6, vmax=12, ticks=(-6, -3, 0, 3, 6, 9.8, 12)),
    Element("haines", "Haines index (low elevation)", "index 2-6", 1, "HRRR 950/850 mb", True, categorical=True),
    Element("hdw", "Hot-Dry-Windy index", "hPa m/s", 0.5, "HRRR VPD x wind, lowest 50 hPa", True,
            cmap="YlOrRd", vmax=300, ticks=tuple(range(0, 301, 50))),
    Element("temp_f", "2 m temperature", "degF", 0.1, "HRRR TMP 2 m", cmap="RdYlBu_r", vmin=-10, vmax=110,
            ticks=tuple(range(-10, 111, 20))),
    Element("dewpoint_f", "2 m dewpoint", "degF", 0.1, "HRRR DPT 2 m", cmap="YlGnBu", vmin=0, vmax=80,
            ticks=tuple(range(0, 81, 10))),
    Element("rh_pct", "2 m relative humidity", "%", 0.5, "HRRR RH 2 m", cmap="YlGnBu", vmax=100,
            ticks=tuple(range(0, 101, 20))),
    Element("vpd_hpa", "2 m vapor pressure deficit", "hPa", 0.05, "Bolton from T and Td", True,
            cmap="YlOrRd", vmax=40, ticks=tuple(range(0, 41, 10))),
    Element("wind_speed_mph", "10 m wind speed", "mph", 0.1, "HRRR UGRD/VGRD 10 m", cmap="YlOrRd", vmax=40,
            ticks=tuple(range(0, 41, 5))),
    Element("wind_dir_deg", "10 m wind direction (from)", "deg", 1, "HRRR UGRD/VGRD 10 m", image=False),
    Element("wind_gust_mph", "Surface wind gust", "mph", 0.1, "HRRR GUST", cmap="YlOrRd", vmax=60,
            ticks=tuple(range(0, 61, 10))),
    Element("fuel_moisture_pct", "Estimated 1-hr fuel moisture", "%", 0.1, "3.0 + 0.25 x RH (uncalibrated)", True,
            cmap="RdYlGn", vmax=30, ticks=tuple(range(0, 31, 5))),
    Element("lcl_m", "Lifted condensation level", "m AGL", 10, "125 x (T - Td) from 2 m", True,
            cmap="YlGnBu", vmax=4000, ticks=tuple(range(0, 4001, 1000))),
    Element("cape_jkg", "Surface CAPE", "J/kg", 5, "HRRR CAPE surface", cmap="YlOrRd", vmax=3000,
            ticks=tuple(range(0, 3001, 500))),
    Element("cin_jkg", "Surface CIN", "J/kg", 5, "HRRR CIN surface", cmap="PuBu", vmin=-300, vmax=0,
            ticks=tuple(range(-300, 1, 100))),
    Element("pwat_mm", "Precipitable water", "mm", 0.1, "HRRR PWAT", cmap="YlGnBu", vmax=60,
            ticks=tuple(range(0, 61, 10))),
    Element("cloud_cover_pct", "Total cloud cover", "%", 1, "HRRR TCDC", cmap="Greys", vmax=100,
            ticks=tuple(range(0, 101, 20))),
    Element("precip_mm_1h", "Precipitation, previous hour", "mm", 0.01, "HRRR APCP", cmap="GnBu", vmax=10,
            ticks=(0, 1, 2, 4, 6, 8, 10)),
    Element("smoke_surface_ugm3", "Near-surface smoke (8 m)", "ug/m3", 0.1, "HRRR MASSDEN", cmap="Greys",
            vmax=100, ticks=tuple(range(0, 101, 20))),
    Element("smoke_column_mgm2", "Vertically integrated smoke", "mg/m2", 0.1, "HRRR COLMD", cmap="Greys",
            vmax=50, ticks=tuple(range(0, 51, 10))),
)
BY_NAME = {element.name: element for element in ELEMENTS}

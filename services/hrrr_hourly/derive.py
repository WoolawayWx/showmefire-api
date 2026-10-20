"""Every element of the hourly product for one forecast hour, on the cropped HRRR grid."""
from __future__ import annotations

import numpy as np

from services.fire_weather import hrrr_profiles as hp
from services.fire_weather import indices as ix
from services.hrrr_hourly import danger

MS_TO_MPH = ix.MS_TO_MPH
ESTIMATED_FM_INTERCEPT = 3.0
ESTIMATED_FM_RH_SLOPE = 0.25
KG_M3_TO_UG_M3 = 1e9
KG_M2_TO_MG_M2 = 1e6


def _f(temp_c):
    return temp_c * 1.8 + 32.0


def derive_hour(profiles: hp.Profiles, extras: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Element name -> 2-D float array. Elements whose source is missing are omitted."""
    base = hp.derive_fields(profiles)
    p = profiles
    nan = np.full(p.t2_c.shape, np.nan)
    out = {name: base[name] for name in (
        "mixing_height_ft", "holzworth_ft", "transport_speed_mph", "transport_dir_deg", "ventilation_kt_ft",
        "ventilation_category", "inversion_strength_f", "inversion_top_ft", "lapse_rate_c_km", "haines", "hdw")}

    east, north = ix.earth_relative_winds(p.u10, p.v10, p.lon, p.central_longitude, p.standard_parallel)
    rh = extras.get("rh2")
    if rh is None:
        rh = 100.0 * ix.saturation_vapor_pressure_hpa(p.td2_c) / ix.saturation_vapor_pressure_hpa(p.t2_c)
    rh = np.clip(rh, 0.0, 100.0)
    out.update({
        "temp_f": _f(p.t2_c),
        "dewpoint_f": _f(p.td2_c),
        "rh_pct": rh,
        "vpd_hpa": ix.vapor_pressure_deficit_hpa(p.t2_c, p.td2_c),
        "wind_speed_mph": ix.wind_speed(east, north) * MS_TO_MPH,
        "wind_dir_deg": ix.wind_direction_from(east, north),
        "fuel_moisture_pct": ESTIMATED_FM_INTERCEPT + ESTIMATED_FM_RH_SLOPE * rh,
        # Espy approximation; adequate for a 3 km display field.
        "lcl_m": np.maximum(125.0 * (p.t2_c - p.td2_c), 0.0),
    })
    out["fire_danger_est"] = danger.fire_danger_category(
        out["fuel_moisture_pct"], rh, out["wind_speed_mph"] * danger.MPH_TO_KT)
    if "gust" in extras:
        out["wind_gust_mph"] = extras["gust"] * MS_TO_MPH
    for key, name in (("cape", "cape_jkg"), ("cin", "cin_jkg"), ("pwat", "pwat_mm"), ("cloud", "cloud_cover_pct"),
                      ("precip", "precip_mm_1h")):
        if key in extras:
            out[name] = extras[key]
    if "smoke_surface" in extras:
        out["smoke_surface_ugm3"] = extras["smoke_surface"] * KG_M3_TO_UG_M3
    if "smoke_column" in extras:
        out["smoke_column_mgm2"] = extras["smoke_column"] * KG_M2_TO_MG_M2
    return {name: np.asarray(values, dtype="float32") for name, values in out.items() if values is not None}

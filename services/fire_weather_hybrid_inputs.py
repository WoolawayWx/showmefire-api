"""Reuse forecast-v1 normalized native cubes for the experimental 1 km grid.

Acquisition ownership stays with the existing framework. This adapter never
changes its blend, registry, files, scheduler, or public output.
"""
from __future__ import annotations
import numpy as np
import pyproj
from rasterio.transform import from_origin
from services.fire_weather_hybrid.contracts import GRID
from services.fire_weather_hybrid.regrid import remap
from services.fire_weather_hybrid.sources import blend, validate_availability

ALIASES = {"temperature_2m":"temp_c", "dewpoint_2m":"dewpoint_c", "relative_humidity_2m":"rh_pct",
           "wind_u_10m":"wind_u_ms", "wind_v_10m":"wind_v_ms", "wind_gust_10m":"gust_ms",
           "precipitation_increment":"precip_mm", "shortwave_down":"solar_wm2"}


def normalized_hour(cube, valid_time, *, wind_reference):
    """One normalized SourceCube hour; preserves accumulation semantics.

    wind_reference is required (earth/grid), since silently guessing this
    metadata can rotate the forecast incorrectly. Unavailable hours return {}.
    """
    if wind_reference not in {"earth", "grid"}:
        raise ValueError("declare earth-relative or grid-relative source winds")
    ds = cube.dataset
    instant = np.datetime64(valid_time.replace(tzinfo=None), "ns")
    if instant not in ds.time.values.astype("datetime64[ns]"):
        return {}
    hour = ds.sel(time=instant)
    if "member" in hour.dims:
        hour = hour.mean("member", skipna=True, keep_attrs=True)
    crs = pyproj.CRS.from_user_input(ds.attrs["crs"])
    x,y = np.asarray(ds.x.values),np.asarray(ds.y.values)
    if x.ndim != 1 or y.ndim != 1 or min(len(x),len(y)) < 2:
        raise ValueError("source requires regular projected axes")
    dx,dy=abs(float(x[-1]-x[0]))/(len(x)-1),abs(float(y[-1]-y[0]))/(len(y)-1)
    def regular(axis,step):
        # WeatherNext stores 0.1-degree coordinates as float32, including
        # longitudes near 270E. Subtraction/longitude wrapping introduces a few
        # ulps of spacing jitter; this is quantization, not an irregular grid.
        eps=np.finfo(axis.dtype).eps if np.issubdtype(axis.dtype,np.floating) else 0.
        tolerance=4*eps*max(1,float(np.max(np.abs(axis))))
        differences=np.diff(axis)
        return (np.all(differences>0) or np.all(differences<0)) and np.allclose(abs(differences),step,rtol=1e-5,atol=tolerance)
    if not regular(x,dx) or not regular(y,dy):
        raise ValueError("irregular projected source axes")
    transform = from_origin(min(x)-dx/2,max(y)+dy/2,dx,dy)
    native = {target:np.asarray(hour[source].transpose("y","x").values,"float32")
              for source,target in ALIASES.items() if source in hour}
    if wind_reference == "grid" and {"wind_u_ms","wind_v_ms"} <= native.keys():
        xx,yy = np.meshgrid(x,y)
        lon,lat = pyproj.Transformer.from_crs(crs,"EPSG:4326",always_xy=True).transform(xx,yy)
        angle = np.radians(pyproj.Proj(crs).get_factors(lon,lat).meridian_convergence)
        u,v = native["wind_u_ms"],native["wind_v_ms"]
        native["wind_u_ms"],native["wind_v_ms"] = u*np.cos(angle)+v*np.sin(angle),v*np.cos(angle)-u*np.sin(angle)
    output = {}
    for name,values in native.items():
        if y[0] < y[-1]: values = values[::-1]
        if x[0] > x[-1]: values = values[:,::-1]
        output[name] = remap(values,transform,crs,kind="accumulation" if name == "precip_mm" else "continuous")
    if {"wind_u_ms","wind_v_ms"} <= output.keys():
        output["wind_speed_ms"] = np.hypot(output["wind_u_ms"],output["wind_v_ms"])
    return output


def forecast_hour(cubes, valid_time, issued_at, source_metadata, weights):
    fields, used = {}, []
    for source,cube in cubes.items():
        meta = source_metadata[source]
        validate_availability(source,cube.cycle_time,meta["available_at"],issued_at)
        hour = normalized_hour(cube,valid_time,wind_reference=meta["wind_reference"])
        if hour:
            fields[source] = hour
            used.append(source)
    if not fields:
        raise ValueError("no forecast source covers this valid time")
    output,coverage = blend(fields,weights)
    return output,{"sources":used,"available_weight":coverage}


def analysis_hour(rtma, mrms, *, valid_time, issued_at):
    """Attach already-normalized historical analyses without leaking future data."""
    if valid_time >= issued_at:
        raise ValueError("analysis inputs must precede issuance")
    fields = dict(rtma)
    # Missing precipitation is unknown, even when RTMA omits its rain field.
    fields["precip_mm"] = np.asarray(mrms["precip_mm"],"float32") if mrms is not None else np.full((GRID.height,GRID.width),np.nan,"float32")
    return fields


"""Does adding neighbor-state stations improve the in-state interpolation?

Leave-one-out cross-validation: each Missouri station is predicted from the *other* stations
with the same RBF the station maps use (multiquadric, smooth=0.005), once from Missouri
stations alone and once with the neighbor-state buffer added. Reports RMSE/MAE/bias overall
and for stations within BORDER_KM of the state border, where extrapolation hurts most.

Usage (from api/):  python scripts/evaluate_buffer.py [variable ...]
Variables: relative_humidity air_temp wind_speed fuel_moisture (default: all four).
"""
import asyncio
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf
from shapely.geometry import Point

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from services.synoptic import (fetch_buffer_stations, fetch_synoptic_data,  # noqa: E402
                               get_buffer_station_data, get_station_data)

BORDER_KM = 50
VARIABLES = ["relative_humidity", "air_temp", "wind_speed", "fuel_moisture"]
NETWORKS = {"RAWS", "ASOS/AWOS", "MOCOMAGNET", "Missouri Mesonet"}  # same filter as the maps


def points_for(stations, variable):
    rows = []
    for s in stations:
        value = (s.get("observations") or {}).get(variable, {}).get("value")
        if value is None or s.get("network") not in NETWORKS:
            continue
        if variable == "relative_humidity":
            value = min(value, 100)
        rows.append((s["longitude"], s["latitude"], float(value), s.get("state")))
    return rows


def border_distances_km(lon_lat):
    boundary = gpd.read_file(API_DIR / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp")
    boundary = boundary.to_crs(32615).geometry.union_all().boundary
    projected = gpd.GeoSeries([Point(x, y) for x, y in lon_lat], crs=4326).to_crs(32615)
    return np.array([boundary.distance(p) / 1000 for p in projected])


def loo_predict(mo, extra, index):
    """Predict Missouri station ``index`` from the other Missouri stations plus ``extra``."""
    training = [p for i, p in enumerate(mo) if i != index] + extra
    lon, lat, value = zip(*[(p[0], p[1], p[2]) for p in training])
    model = Rbf(lon, lat, value, function="multiquadric", smooth=0.005)
    return float(model(mo[index][0], mo[index][1]))


def summarize(errors):
    errors = np.asarray(errors)
    if errors.size == 0:
        return "n=0"
    return f"n={errors.size:3d}  RMSE={math.sqrt(np.mean(errors ** 2)):6.2f}  MAE={np.mean(np.abs(errors)):6.2f}  bias={np.mean(errors):+6.2f}"


async def main(variables):
    await fetch_synoptic_data()
    await fetch_buffer_stations()
    mo_stations = [s for s in (get_station_data()["stations"] or []) if s.get("state") == "MO"]
    buffer_stations = get_buffer_station_data()["stations"] or []
    print(f"Missouri stations: {len(mo_stations)}, buffer stations: {len(buffer_stations)}")

    for variable in variables:
        mo = points_for(mo_stations, variable)
        extra = [p for p in points_for(buffer_stations, variable)]
        if len(mo) < 10:
            print(f"\n{variable}: only {len(mo)} Missouri stations, skipped")
            continue
        border = border_distances_km([(p[0], p[1]) for p in mo]) <= BORDER_KM
        without = [loo_predict(mo, [], i) - mo[i][2] for i in range(len(mo))]
        with_buffer = [loo_predict(mo, extra, i) - mo[i][2] for i in range(len(mo))]
        print(f"\n{variable}  (error = predicted - observed; buffer points: {len(extra)})")
        for label, mask in (("all MO       ", np.ones(len(mo), bool)), (f"<= {BORDER_KM} km border", border), (f"> {BORDER_KM} km border", ~border)):
            print(f"  {label:16s} no buffer: {summarize(np.array(without)[mask])}")
            print(f"  {'':16s} buffer:    {summarize(np.array(with_buffer)[mask])}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or VARIABLES))

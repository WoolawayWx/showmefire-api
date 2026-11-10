"""The part of the feature vector HRRR cannot give: the rain of the last 30 days and the slow fuel moisture.

Both come from gridMET (Abatzoglou 2013), fetched as Missouri subsets from the THREDDS NetCDF Subset Service,
the same source the models were trained on. gridMET runs about a day behind, so the latest day it has is
used for fuel moisture and any day between it and today counts as no rain; `lag_days` says how far behind it
was, and the job refuses a state more than MAX_LAG_DAYS old rather than quietly forecast from stale fuel.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

NCSS = "http://thredds.northwestknowledge.net:8080/thredds/ncss/MET/{var}/{var}_{year}.nc"
BBOX = {"north": 40.9, "south": 35.7, "west": -96.1, "east": -88.8}
HEADERS = {"User-Agent": "Mozilla/5.0 (ShowMeFire)"}
RAIN_DAYS = 30
MAX_LAG_DAYS = 4
STATE_VARS = {"fm100_prev": "fm100", "fm1000_prev": "fm1000", "erc_prev": "erc"}


def _fetch_year(var: str, year: int, first: date, last: date, target: Path, timeout: int = 180) -> None:
    import requests

    params = {"var": "all", **BBOX, "horizStride": 1, "accept": "netcdf",
              "time_start": f"{first.isoformat()}T00:00:00Z", "time_end": f"{last.isoformat()}T23:59:59Z"}
    part = target.with_suffix(".part")
    with requests.get(NCSS.format(var=var, year=year), params=params, headers=HEADERS, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(part, "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    part.replace(target)


def _read(path: Path):
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        values = dataset[next(iter(dataset.data_vars))].to_numpy().astype("float32")
        return (dataset["day"].to_numpy().astype("datetime64[D]"), dataset["lat"].to_numpy().astype(float),
                dataset["lon"].to_numpy().astype(float), values)


def fetch(var: str, first: date, last: date, cache: Path):
    """(days, lat, lon, values (days, lat, lon)) of `var` for first..last, whatever part of it gridMET has. One request
    per calendar year touched; a request is cached per (var, first, last), so a rerun the same day costs nothing."""
    cache.mkdir(parents=True, exist_ok=True)
    days, values, lat, lon = [], [], None, None
    for year in range(first.year, last.year + 1):
        start, end = max(first, date(year, 1, 1)), min(last, date(year, 12, 31))
        target = cache / f"{var}_{start:%Y%m%d}_{end:%Y%m%d}.nc"
        if not target.is_file():
            _fetch_year(var, year, start, end, target)
        part_days, lat, lon, part_values = _read(target)
        days.append(part_days)
        values.append(part_values)
    return np.concatenate(days), lat, lon, np.concatenate(values)


def state_from_arrays(day: date, rain_days, rain, state_days: dict, state: dict, lat, lon) -> Optional[dict]:
    """The feature inputs for `day` from fetched arrays (split out so it can be tested without the network).

    Returns None when gridMET has nothing recent enough. Otherwise: `rain` (RAIN_DAYS, lat, lon), the daily rain
    of day-30 .. day-1 with days gridMET does not have yet as 0; fm100_prev / fm1000_prev / erc_prev from the
    latest day at or before day-1; `lag_days` (1 = yesterday's data, as in training); lat and lon vectors."""
    yesterday = np.datetime64(day - timedelta(days=1))
    newest = []
    for days in state_days.values():
        past = days[days <= yesterday]
        if not past.size:
            return None
        newest.append(past.max())
    latest = min(newest)                       # the newest day every variable has
    lag = int((np.datetime64(day) - latest) / np.timedelta64(1, "D"))
    if lag > MAX_LAG_DAYS:
        return None
    window = np.zeros((RAIN_DAYS,) + rain.shape[1:], dtype="float32")
    known = np.zeros(RAIN_DAYS, dtype=bool)
    for i in range(RAIN_DAYS):
        wanted = np.datetime64(day - timedelta(days=RAIN_DAYS - i))
        hit = np.flatnonzero(rain_days == wanted)
        if hit.size:
            window[i], known[i] = rain[hit[0]], True
    out = {"rain": window, "rain_days_known": int(known.sum()), "lag_days": lag, "lat": np.asarray(lat), "lon": np.asarray(lon),
           "latest_day": str(latest)}
    for name in STATE_VARS:
        hit = np.flatnonzero(state_days[name] == latest)
        if not hit.size:
            return None
        out[name] = state[name][hit[0]]
    return out


def latest_state(day: date, cache: Path) -> Optional[dict]:
    """Fetch and assemble the gridMET inputs for `day` (see `state_from_arrays`)."""
    first, last = day - timedelta(days=RAIN_DAYS + 1), day - timedelta(days=1)
    rain_days, lat, lon, rain = fetch("pr", first, last, cache)
    state_days, state = {}, {}
    for name, var in STATE_VARS.items():
        state_days[name], _lat, _lon, state[name] = fetch(var, day - timedelta(days=MAX_LAG_DAYS + 1), last, cache)
    return state_from_arrays(day, rain_days, rain, state_days, state, lat, lon)


def to_cells(field, grid_lat, grid_lon, lat, lon) -> np.ndarray:
    """Nearest gridMET value at each (lat, lon) cell. `field` is (..., grid_lat, grid_lon) on regular coordinate
    vectors (either order); the result is (..., *lat.shape). Cells outside the subset are NaN."""
    def index(vector, wanted):
        step = vector[1] - vector[0]
        position = np.rint((np.asarray(wanted) - vector[0]) / step).astype(int)
        return position, (position >= 0) & (position < vector.size)
    row, row_ok = index(np.asarray(grid_lat), lat)
    col, col_ok = index(np.asarray(grid_lon), lon)
    inside = row_ok & col_ok
    values = np.asarray(field, dtype=float)[..., np.where(inside, row, 0), np.where(inside, col, 0)]
    return np.where(inside, values, np.nan)

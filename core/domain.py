"""Shared geographic domain helpers without data-download dependencies.

Byte-identical mirror of model-training/spatial/domain.py - the same
crop is proven correct there against real HRRR grids (produces a
(273, 267)-shaped file). Kept as a separate module rather than duplicated
inline so this repo's own crop call sites (DailyForecast.py,
DailyForecast_ModelFD.py) share one definition instead of drifting.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


MO_BUFFERED_BBOX = (-96.8, -88.1, 34.8, 41.8)


def crop(ds: xr.Dataset) -> xr.Dataset:
    """Crop a latitude/longitude grid to the buffered Missouri domain."""
    lon = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    west, east, south, north = MO_BUFFERED_BBOX
    mask = (lon >= west) & (lon <= east) & (ds.latitude >= south) & (ds.latitude <= north)
    rows, cols = np.where(mask.values)
    if not len(rows):
        raise ValueError("weather grid does not intersect the configured domain")
    ydim, xdim = mask.dims
    return ds.isel({ydim: slice(rows.min(), rows.max() + 1), xdim: slice(cols.min(), cols.max() + 1)})

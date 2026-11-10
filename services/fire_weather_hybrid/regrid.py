"""Explicit reprojection semantics, including conservative rainfall depths."""
from __future__ import annotations

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject
from .contracts import GRID, Grid


def remap(values, source_transform, source_crs, *, kind="continuous", grid: Grid = GRID):
    if kind not in {"continuous", "accumulation", "category", "fraction"}:
        raise ValueError(f"unknown remapping kind: {kind}")
    data = np.asarray(values, dtype="float32")
    if data.ndim != 2:
        raise ValueError("remap expects one two-dimensional field")
    out = np.full((grid.height, grid.width), np.nan, dtype="float32")
    # GDAL average computes area-overlap-weighted cell means, preserving depth
    # integrals over matching extents. A coarse cell is not divided by cell count.
    method = {"continuous": Resampling.bilinear, "accumulation": Resampling.average,
              "category": Resampling.nearest, "fraction": Resampling.average}[kind]
    reproject(data, out, src_transform=source_transform, src_crs=source_crs, src_nodata=np.nan,
              dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan, resampling=method,
              num_threads=1, init_dest_nodata=True)
    return out


def increments(totals, interval_start_hours, interval_end_hours):
    """Disaggregate already decoded accumulation intervals, rejecting overlaps.

    Source adapters must difference cumulative/resetting buckets first. Missing
    intervals stay unknown. An accumulation ending at h covers (start, h].
    """
    totals = np.asarray(totals, dtype="float32")
    starts, ends = np.asarray(interval_start_hours, int), np.asarray(interval_end_hours, int)
    if len(totals) != len(starts) or len(starts) != len(ends) or not len(ends):
        raise ValueError("interval lengths do not match")
    if np.any(starts < 0) or np.any(ends <= starts):
        raise ValueError("invalid accumulation interval")
    out = np.full((int(ends.max()), *totals.shape[1:]), np.nan, dtype="float32")
    occupied = np.zeros(len(out), bool)
    for amount, start, end in zip(totals, starts, ends):
        if occupied[start:end].any() or np.any(amount[np.isfinite(amount)] < 0):
            raise ValueError("overlapping intervals or negative precipitation")
        out[start:end] = amount / (end-start)
        occupied[start:end] = True
    return out


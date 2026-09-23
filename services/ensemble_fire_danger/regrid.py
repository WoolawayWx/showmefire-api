"""
Cached barycentric (linear) regridding of member grids onto the one
target grid every ensemble product is computed on: the DailyForecast
HRRR "mo_bounds" subset (196 x 205), the same grid the public map, the
forecast-state file, and core/risk_fusion_reference/county_cells.json use.

Contract-mirror discipline (api/core/contract_mirrors.json, pair
"ensemble_fire_danger_regrid"): byte-identical to
model-training/ensemble_fire_danger/regrid.py; numpy/scipy/stdlib only.

HRRR, NAM nest, RRFS, HiResW and the ensprod files are all different
curvilinear grids (the reason model-training/fire_weather_index/
grid_score.py::regrid_to_grid uses scattered-point interpolation too).
That function re-triangulates on every call (~35 s); here the Delaunay
simplices + barycentric weights are computed once per (source grid,
target grid) pair and cached to .npz, keyed by a hash of both coordinate
arrays - so a model's grid change invalidates its cache automatically.
Target cells outside a source's convex hull come back NaN (never
extrapolated), exactly like regrid_to_grid.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _grid_hash(lat: np.ndarray, lon: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.round(lat, 5), dtype="float64").tobytes())
    digest.update(np.ascontiguousarray(np.round(lon, 5), dtype="float64").tobytes())
    digest.update(str(lat.shape).encode())
    return digest.hexdigest()[:20]


class Regridder:
    def __init__(self, src_lat: np.ndarray, src_lon: np.ndarray, dst_lat: np.ndarray, dst_lon: np.ndarray,
                 cache_dir: Optional[Path] = None):
        self.src_shape = src_lat.shape
        self.dst_shape = dst_lat.shape
        key = f"{_grid_hash(src_lat, src_lon)}_{_grid_hash(dst_lat, dst_lon)}"
        path = Path(cache_dir) / f"regrid_{key}.npz" if cache_dir else None
        if path is not None and path.exists():
            with np.load(path) as cached:
                self.vertices, self.weights, self.inside = cached["vertices"], cached["weights"], cached["inside"]
            return
        self.vertices, self.weights, self.inside = self._build(src_lat, src_lon, dst_lat, dst_lon)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, vertices=self.vertices, weights=self.weights, inside=self.inside)
            tmp.replace(path)

    @staticmethod
    def _build(src_lat, src_lon, dst_lat, dst_lon) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.spatial import Delaunay

        src = np.column_stack([np.ravel(src_lon), np.ravel(src_lat)])
        dst = np.column_stack([np.ravel(dst_lon), np.ravel(dst_lat)])
        tri = Delaunay(src)
        simplex = tri.find_simplex(dst)
        inside = simplex >= 0
        safe = np.where(inside, simplex, 0)
        vertices = tri.simplices[safe]
        transform = tri.transform[safe]
        delta = dst - transform[:, 2]
        bary = np.einsum("nij,nj->ni", transform[:, :2], delta)
        weights = np.column_stack([bary, 1.0 - bary.sum(axis=1)])
        return vertices.astype("int64"), weights.astype("float64"), inside

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """(..., src_y, src_x) -> (..., dst_y, dst_x)."""
        field = np.asarray(field, dtype="float64")
        lead_shape = field.shape[:-2]
        flat = field.reshape(lead_shape + (-1,))
        values = np.take(flat, self.vertices, axis=-1)            # (..., n_dst, 3)
        out = np.sum(values * self.weights, axis=-1)               # NaN propagates from any vertex
        out = np.where(self.inside, out, np.nan)
        return out.reshape(lead_shape + self.dst_shape)


def cell_size_km(lat: np.ndarray, lon: np.ndarray) -> float:
    """Median grid spacing (km) of a curvilinear lat/lon grid."""
    coslat = np.cos(np.radians(lat))

    def step(axis: int) -> np.ndarray:
        dlat = np.diff(lat, axis=axis) * 111.2
        dlon = np.diff(lon, axis=axis) * 111.2 * (coslat[1:, :] if axis == 0 else coslat[:, 1:])
        return np.hypot(dlat, dlon)

    return float(np.median(np.concatenate([step(0).ravel(), step(1).ravel()])))

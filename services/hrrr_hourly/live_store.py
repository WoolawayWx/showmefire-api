"""A Zarr store that is written one step at a time, so a forecast hour can be shown as soon as it is processed.

The store is created once at its full length (every forecast hour, or every day for the peak product) with every array
empty (NaN fill value, no chunk files). A step is then written by saving its one chunk, which is a single new file. So
the array metadata (`.zmetadata`, `.zarray`) never changes after creation and can be cached for good, and a step's chunk
file only changes if that step is rewritten (the peak product does that while a day is still filling in).

Same layout the one-shot writer in store.py produces, so the viewer reads either: arrays `(step, y, x)` chunked one step
each, 2-D `latitude`/`longitude`, zlib + byte-shuffle (see store.write_zarr for why not Blosc), consolidated metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

STEP_DIM = "lead_hour"


def _codec():
    import numcodecs
    return numcodecs.Zlib(level=5), numcodecs


def quantize(values: np.ndarray, scale: float) -> np.ndarray:
    """Round to the element's step, as the one-shot writer does, keeping NaN."""
    values = np.asarray(values, dtype="float64")
    return np.where(np.isfinite(values), np.round(values / scale) * scale, np.nan).astype("float32")


def exists(path: Path) -> bool:
    return (Path(path) / ".zmetadata").is_file()


def create(path: Path, elements: Iterable, steps: int, lat: np.ndarray, lon: np.ndarray, step_values: np.ndarray,
           valid_hours_since_epoch: np.ndarray, step_dim: str = STEP_DIM) -> None:
    """Create the store with every array empty. `elements` carry .name/.long_name/.units/.source/.derived/.scale."""
    import zarr

    codec, numcodecs = _codec()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    ny, nx = lat.shape

    def make(name: str, data=None, *, shape, chunks, dtype, dims, attrs=None, fill=None):
        array = root.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, compressor=codec,
                                    filters=[numcodecs.Shuffle(elementsize=np.dtype(dtype).itemsize)],
                                    fill_value=fill, overwrite=True)
        array.attrs["_ARRAY_DIMENSIONS"] = dims
        for key, value in (attrs or {}).items():
            array.attrs[key] = value
        if data is not None:
            array[:] = data
        return array

    make("latitude", lat.astype("float32"), shape=(ny, nx), chunks=(ny, nx), dtype="float32", dims=["y", "x"])
    make("longitude", lon.astype("float32"), shape=(ny, nx), chunks=(ny, nx), dtype="float32", dims=["y", "x"])
    make(step_dim, np.asarray(step_values, dtype="int16"), shape=(steps,), chunks=(steps,), dtype="int16",
         dims=[step_dim], attrs={"long_name": "forecast hour" if step_dim == STEP_DIM else "step"})
    make("valid_time", np.asarray(valid_hours_since_epoch, dtype="int64"), shape=(steps,), chunks=(steps,), dtype="int64",
         dims=[step_dim], attrs={"units": "hours since 1970-01-01 00:00:00", "calendar": "proleptic_gregorian"})
    for element in elements:
        make(element.name, shape=(steps, ny, nx), chunks=(1, ny, nx), dtype="float32", dims=[step_dim, "y", "x"], fill=np.nan,
             attrs={"long_name": element.long_name, "units": element.units, "source": element.source,
                    "derived": int(element.derived), "scale": element.scale})
    zarr.consolidate_metadata(str(path))


def write_step(path: Path, name: str, index: int, values: np.ndarray, scale: float) -> str:
    """Write one step of one array; returns the chunk's path relative to the store (the file to upload)."""
    import zarr

    root = zarr.open_group(str(path), mode="r+")
    root[name][index] = quantize(values, scale)
    return f"{name}/{index}.0.0"


def chunk_path(name: str, index: int) -> str:
    return f"{name}/{index}.0.0"


def read_step(path: Path, name: str, index: int) -> Optional[np.ndarray]:
    import zarr

    array = zarr.open_group(str(path), mode="r")[name]
    if not (Path(path) / name / f"{index}.0.0").is_file():
        return None
    return np.asarray(array[index])


def metadata_files(path: Path, element_names: Iterable[str]) -> list[str]:
    """Everything in the store except the element chunks, relative to it: the array metadata and the coordinate chunks.
    A first publish uploads these once; they never change."""
    path = Path(path)
    elements = set(element_names)
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*")
                  if p.is_file() and (p.name.startswith(".z") or p.parent.name not in elements))


def point_series(path: Path, manifest: dict, lat: float, lon: float, max_distance_km: float,
                 names: Optional[Iterable[str]] = None) -> Optional[dict]:
    """Every element's series at the grid cell nearest (lat, lon), read from the store, for the steps the manifest lists.
    The same answer store.point_series gives from the NetCDF copy; used while a run is still being filled in."""
    import zarr

    root = zarr.open_group(str(path), mode="r")
    glat, glon = np.asarray(root["latitude"][:]), np.asarray(root["longitude"][:])
    dy = np.radians(glat - lat)
    dx = np.radians(glon - lon) * np.cos(np.radians(lat))
    distance = 6371.0 * np.hypot(dy, dx)
    y, x = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
    if distance[y, x] > max_distance_km:
        return {"error": "outside_grid", "distance_km": round(float(distance[y, x]), 1)}
    steps = list(manifest.get("zarr_leads") or manifest["forecast_hours"])
    present = list(manifest["forecast_hours"])
    series = {}
    wanted = set(names) if names is not None else None
    for name, info in manifest["variables"].items():
        if wanted is not None and name not in wanted:
            continue
        have = set(info.get("hours_present", present))
        values = []
        for step in present:
            value = float(root[name][steps.index(step), y, x]) if step in have else float("nan")
            values.append(round(value, 3) if np.isfinite(value) else None)
        series[name] = values
    return {"cycle": manifest["cycle"], "grid_lat": float(glat[y, x]), "grid_lon": float(glon[y, x]),
            "distance_km": round(float(distance[y, x]), 2), "lead_hours": present,
            "valid_times_utc": list(manifest["valid_times_utc"]), "series": series}

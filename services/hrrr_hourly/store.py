"""Write and read the hourly HRRR product: Zarr store, NetCDF copy, manifest.

Layout (root = services.hrrr_hourly.root_dir()):
    <cycle>/forecast.zarr      chunked (lead, y, x) arrays, consolidated metadata, for JS/Python readers
    <cycle>/manifest.json      labels, variables, grid, per-hour statewide series
    nc/hrrr_hourly_<cycle>.nc  flat int16-packed NetCDF copy (what the R2 archive picks up)
    latest.json                pointer to the newest complete cycle
A cycle is built in a temporary directory and swapped in only when complete.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from services.hrrr_hourly import (BY_NAME, ELEMENTS, PRODUCT, SCHEMA_VERSION, cycle_id, live_store, root_dir)

logger = logging.getLogger(__name__)

FILL = -32768
MAX_POINT_DISTANCE_KM = 10.0


def nc_dir(root: Optional[Path] = None) -> Path:
    return (root or root_dir()) / "nc"


def nc_path(cycle: str, root: Optional[Path] = None) -> Path:
    return nc_dir(root) / f"{PRODUCT}_{cycle}.nc"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(temporary, path)


def _stack(hours: dict[int, dict[str, np.ndarray]], leads: list[int], name: str, shape) -> np.ndarray | None:
    present = [lead for lead in leads if name in hours.get(lead, {})]
    if not present:
        return None
    scale = BY_NAME[name].scale
    cube = np.full((len(leads), *shape), np.nan, dtype="float32")
    for index, lead in enumerate(leads):
        if lead in present:
            values = np.asarray(hours[lead][name], dtype="float64")
            cube[index] = np.where(np.isfinite(values), np.round(values / scale) * scale, np.nan)
    return cube


def build_dataset(cycle_time: datetime, hours: dict[int, dict[str, np.ndarray]], lat: np.ndarray, lon: np.ndarray,
                  issued_at: datetime):
    import xarray as xr

    leads = sorted(hours)
    shape = lat.shape
    data_vars = {}
    for element in ELEMENTS:
        cube = _stack(hours, leads, element.name, shape)
        if cube is None:
            continue
        data_vars[element.name] = (("lead_hour", "y", "x"), cube,
                                   {"long_name": element.long_name, "units": element.units, "source": element.source,
                                    "derived": int(element.derived), "scale": element.scale})
    base = np.datetime64(cycle_time.replace(tzinfo=None), "ns")
    valid = base + np.array([np.timedelta64(lead, "h") for lead in leads])
    return xr.Dataset(
        data_vars,
        coords={"lead_hour": ("lead_hour", np.array(leads, dtype="int16"), {"long_name": "forecast hour"}),
                "valid_time": ("lead_hour", valid), "latitude": (("y", "x"), lat.astype("float32")),
                "longitude": (("y", "x"), lon.astype("float32"))},
        attrs={"title": f"HRRR hourly fire-weather forecast, {cycle_time:%H}z run", "model": "HRRR",
               "cycle": f"{cycle_time:%H}z", "cycle_time_utc": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "issued_at_utc": issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "note": f"Forecast issued from the {cycle_time:%H}z HRRR run; valid_time = cycle time + lead_hour",
               "schema_version": SCHEMA_VERSION})


def write_zarr(dataset, path: Path) -> bool:
    """Chunked one-hour-per-chunk Zarr (v2 format, widest JS support). False if zarr isn't installed."""
    try:
        import numcodecs
        import zarr  # noqa: F401
    except ImportError:
        logger.warning("zarr is not installed; skipping the Zarr store (NetCDF copy is still written)")
        return False
    # Byte-shuffle + zlib rather than Blosc: browsers decode both natively (zlib via DecompressionStream, shuffle in
    # plain JS). Blosc needs a WebAssembly decoder that calls Function(), which a strict Content-Security-Policy
    # (no 'unsafe-eval') blocks, so the website could not read Blosc chunks.
    codec = numcodecs.Zlib(level=5)

    def settings(name: str, chunks: tuple) -> dict:
        return {"chunks": chunks, "compressor": codec,
                "filters": [numcodecs.Shuffle(elementsize=dataset[name].dtype.itemsize)]}

    ny, nx = dataset["latitude"].shape
    encoding = {name: settings(name, (1, ny, nx)) for name in dataset.data_vars}
    encoding["latitude"] = settings("latitude", (ny, nx))
    encoding["longitude"] = settings("longitude", (ny, nx))
    dataset.to_zarr(path, mode="w", consolidated=True, encoding=encoding)
    return True


def write_netcdf(dataset, path: Path) -> None:
    """int16-packed, zlib-compressed. Values were already rounded to each element's scale.

    Packed by hand so NaN becomes the fill value (xarray's own cast would turn it into 0)."""
    packed = {}
    for name in dataset.data_vars:
        scale = BY_NAME[name].scale
        values = dataset[name].values
        ints = np.where(np.isfinite(values), np.round(values / scale), FILL).clip(FILL, 32767).astype("int16")
        attrs = {k: v for k, v in dataset[name].attrs.items() if k != "scale"}
        packed[name] = (dataset[name].dims, ints, {**attrs, "scale_factor": float(scale)})
    encoding = {name: {"_FillValue": FILL, "zlib": True, "complevel": 4} for name in packed}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    dataset.assign(packed).to_netcdf(temporary, encoding=encoding)
    os.replace(temporary, path)


def build_manifest(cycle_time: datetime, issued_at: datetime, dataset, expected_hours, grid: dict,
                   series: dict, files: dict) -> dict:
    leads = [int(v) for v in dataset["lead_hour"].values]
    return {
        "product": PRODUCT, "schema_version": SCHEMA_VERSION, "model": "HRRR",
        "label": f"Forecast from the {cycle_time:%H}z HRRR run",
        "cycle": cycle_id(cycle_time), "cycle_time_utc": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issued_at_utc": issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_hours": leads, "missing_hours": [h for h in expected_hours if h not in leads],
        "valid_times_utc": [(cycle_time + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ") for h in leads],
        "variables": {name: {"long_name": ds.attrs["long_name"], "units": ds.attrs["units"],
                             "source": ds.attrs["source"], "derived": bool(ds.attrs["derived"]),
                             "scale": ds.attrs["scale"],
                             "hours_present": [int(leads[i]) for i in range(len(leads))
                                               if np.isfinite(ds.values[i]).any()]}
                      for name, ds in dataset.data_vars.items()},
        "grid": grid, "series": series, "files": files,
    }


def write_cycle(cycle_time: datetime, hours: dict[int, dict[str, np.ndarray]], lat: np.ndarray, lon: np.ndarray,
                grid: dict, series: dict, expected_hours, issued_at: Optional[datetime] = None,
                root: Optional[Path] = None) -> dict:
    """Write zarr + NetCDF + manifest for a cycle and swap them in. Returns the manifest."""
    root = Path(root or root_dir())
    issued_at = issued_at or datetime.now(timezone.utc)
    cycle = cycle_id(cycle_time)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    dataset = build_dataset(cycle_time, hours, lat, lon, issued_at)
    timings["dataset_sec"] = round(time.perf_counter() - started, 2)
    if not dataset.data_vars:
        raise ValueError("no elements to write")

    staging = root / f".{cycle}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    started = time.perf_counter()
    zarr_written = write_zarr(dataset, staging / "forecast.zarr")
    timings["zarr_sec"] = round(time.perf_counter() - started, 2)
    started = time.perf_counter()
    write_netcdf(dataset, nc_path(cycle, root))
    timings["netcdf_sec"] = round(time.perf_counter() - started, 2)
    files = {"netcdf": f"nc/{PRODUCT}_{cycle}.nc", "netcdf_bytes": nc_path(cycle, root).stat().st_size}
    if zarr_written:
        files["zarr"] = f"{cycle}/forecast.zarr"
    manifest = build_manifest(cycle_time, issued_at, dataset, expected_hours, grid, series, files)
    _atomic_json(staging / "manifest.json", manifest)

    final = root / cycle
    retired = root / f".{cycle}.old"
    shutil.rmtree(retired, ignore_errors=True)
    if final.exists():
        os.replace(final, retired)
    os.replace(staging, final)
    shutil.rmtree(retired, ignore_errors=True)
    # Cycles are built newest first but not always in order (a late older one can finish after a newer one), and
    # the pointer must never go back to an older run than the one being served.
    current = _latest_cycle_id(root)
    if current is None or cycle >= current:
        _atomic_json(root / "latest.json", {"cycle": cycle, "cycle_time_utc": manifest["cycle_time_utc"],
                                            "issued_at_utc": manifest["issued_at_utc"]})
    manifest["write_timings"] = timings  # returned for the build's timings report; not part of manifest.json
    return manifest


def _latest_cycle_id(root: Path) -> Optional[str]:
    try:
        return json.loads((root / "latest.json").read_text())["cycle"]
    except (OSError, ValueError, KeyError):
        return None


def list_cycles(root: Optional[Path] = None) -> list[str]:
    root = Path(root or root_dir())
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if p.is_dir() and (p / "manifest.json").is_file()), reverse=True)


def timings_path(cycle: str, root: Optional[Path] = None) -> Path:
    return Path(root or root_dir()) / cycle / "timings.json"


def read_timings(cycle: str, root: Optional[Path] = None) -> dict | None:
    try:
        return json.loads(timings_path(cycle, root).read_text())
    except (OSError, ValueError):
        return None


def read_manifest(cycle: str, root: Optional[Path] = None) -> dict | None:
    path = Path(root or root_dir()) / cycle / "manifest.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def prune(now: Optional[datetime] = None, days: Optional[int] = None, root: Optional[Path] = None) -> list[str]:
    """Remove cycles (zarr, manifest, NetCDF) older than `days`. Images are pruned by the caller."""
    from services.hrrr_hourly import keep_days

    root = Path(root or root_dir())
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days if days is not None else keep_days())
    removed = []
    for cycle in list_cycles(root):
        try:
            when = datetime.strptime(cycle, f"%Y%m%d_%Hz").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if when < cutoff:
            shutil.rmtree(root / cycle, ignore_errors=True)
            nc_path(cycle, root).unlink(missing_ok=True)
            removed.append(cycle)
    # Work folders (.work/<cycle>) are normally removed when a build ends; a failed or abandoned build leaves one.
    work_cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=2)
    for folder in (root / ".work").glob("*") if (root / ".work").is_dir() else ():
        try:
            when = datetime.strptime(folder.name, "%Y%m%d_%Hz").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if when < work_cutoff:
            shutil.rmtree(folder, ignore_errors=True)
    return removed


def point_series(cycle: str, lat: float, lon: float, root: Optional[Path] = None,
                 names: Optional[Iterable[str]] = None) -> dict | None:
    """Every element's hourly series at the grid cell nearest (lat, lon), read from the NetCDF copy.

    `names` limits it to those elements (unknown names are left out): each element read decompresses its whole grid for
    every lead, so a caller that needs a few (the burn check needs 5 of ~28) should say which."""
    import xarray as xr

    path = nc_path(cycle, root)
    if not path.is_file():
        # A cycle still being filled in has no NetCDF copy yet (it is written when the cycle finishes): read the Zarr store.
        manifest = read_manifest(cycle, root)
        zarr_dir = Path(root or root_dir()) / cycle / "forecast.zarr"
        if manifest is None or manifest.get("complete", True) or not zarr_dir.is_dir():
            return None
        return live_store.point_series(zarr_dir, manifest, lat, lon, MAX_POINT_DISTANCE_KM, names=names)
    with xr.open_dataset(path) as ds:
        glat, glon = ds["latitude"].values, ds["longitude"].values
        dy = np.radians(glat - lat)
        dx = np.radians(glon - lon) * np.cos(np.radians(lat))
        distance = 6371.0 * np.hypot(dy, dx)
        y, x = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
        if distance[y, x] > MAX_POINT_DISTANCE_KM:
            return {"error": "outside_grid", "distance_km": round(float(distance[y, x]), 1)}
        series = {}
        wanted = set(names) if names is not None else None
        for name in ds.data_vars:
            if wanted is not None and name not in wanted:
                continue
            # Index before .values: `.values[:, y, x]` would load the whole grid for every lead to keep one cell.
            values = ds[name][:, y, x].values
            series[name] = [None if not np.isfinite(v) else round(float(v), 3) for v in values]
        return {"cycle": cycle, "grid_lat": float(glat[y, x]), "grid_lon": float(glon[y, x]),
                "distance_km": round(float(distance[y, x]), 2),
                "lead_hours": [int(v) for v in ds["lead_hour"].values],
                "valid_times_utc": [str(v)[:19] + "Z" for v in ds["valid_time"].values], "series": series}

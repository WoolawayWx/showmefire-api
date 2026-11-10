"""Private, immutable run storage; one-hour streaming writes."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import zlib
import numpy as np
from services.fire_weather_hybrid.contracts import GRID, Grid, LABELS, COLORS, UNITS, LOCAL_TZ, iso
from services.fire_weather_hybrid.trees import sha256

RUN_ID = re.compile(r"^\d{8}_\d{2}z$")


def root_dir():
    return Path(os.getenv("SMF_FIRE_WEATHER_HYBRID_DIR", "data/fire-weather-hybrid"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def resolve_run(root, run_id):
    if run_id == "latest":
        run_id = (read_json(Path(root)/"latest.json") or {}).get("cycle", "")
    if not RUN_ID.fullmatch(run_id):
        raise FileNotFoundError("invalid or missing run")
    directory = (Path(root)/"runs"/run_id).resolve()
    if Path(root).resolve() not in directory.parents or not (directory/"manifest.json").is_file():
        raise FileNotFoundError("run not found")
    return directory


def safe_asset(directory, relative):
    target = (Path(directory)/relative).resolve()
    if Path(directory).resolve() not in target.parents or not target.is_file():
        raise FileNotFoundError("asset not found")
    return target


class ZarrWriter:
    """Minimal standard Zarr v2 writer using browser-supported zlib compression.

    Edge chunks are padded to the declared shape. No pickle, custom codecs,
    WebAssembly, or optional Zarr Python installation is needed for writing.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        self.metadata = {".zgroup": {"zarr_format": 2}, ".zattrs": {}}

    def create(self, name, shape, dims, units=""):
        directory = self.root/name
        directory.mkdir()
        chunks = (1, min(shape[-2], 256), min(shape[-1], 256)) if len(shape) == 3 else (min(shape[-2], 256), min(shape[-1], 256))
        meta = {"zarr_format": 2, "shape": list(shape), "chunks": list(chunks), "dtype": "<f4",
                "compressor": {"id": "zlib", "level": 5}, "fill_value": "NaN", "order": "C", "filters": None}
        attrs = {"_ARRAY_DIMENSIONS": list(dims), "units": units}
        self.metadata[f"{name}/.zarray"], self.metadata[f"{name}/.zattrs"] = meta, attrs

    def write(self, name, values, index=None):
        data = np.asarray(values, "<f4")
        chunks = self.metadata[f"{name}/.zarray"]["chunks"][-2:]
        for row in range(0, data.shape[0], chunks[0]):
            for col in range(0, data.shape[1], chunks[1]):
                block = np.full(chunks, np.nan, "<f4")
                subset = data[row:row+chunks[0], col:col+chunks[1]]
                block[:subset.shape[0], :subset.shape[1]] = subset
                key = f"{row//chunks[0]}.{col//chunks[1]}"
                if index is not None:
                    key = f"{index}.{key}"
                (self.root/name/key).write_bytes(zlib.compress(block.tobytes(), 5))

    def finish(self):
        for name, value in self.metadata.items():
            atomic_json(self.root/name, value)
        atomic_json(self.root/".zmetadata", {"zarr_consolidated_format": 1, "metadata": self.metadata})


def write_run(root, issued_at, times, hours, metadata, *, grid: Grid = GRID, render_maps=False, cycle_time=None):
    """Consume (fields) iterable in valid-time order without holding a cube."""
    import netCDF4
    from pyproj import Transformer
    import rasterio
    from rasterio.shutil import copy as raster_copy
    root = Path(root)
    cycle = (cycle_time or issued_at).astimezone(timezone.utc).strftime("%Y%m%d_%Hz")
    final = root/"runs"/cycle
    if final.exists():
        raise ValueError("run is immutable and already exists")
    staging = root/"runs"/f".{cycle}.building"
    if staging.exists():
        raise ValueError("unfinished run requires inspection before retry")
    staging.mkdir(parents=True)
    zz = ZarrWriter(staging/"forecast.zarr")
    xx, yy = np.meshgrid(grid.x, grid.y)
    lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    for name, values in (("latitude", lat), ("longitude", lon)):
        zz.create(name, values.shape, ("y", "x"))
        zz.write(name, values)
    nc = netCDF4.Dataset(staging/"forecast.nc", "w")
    writers, variables, peak, count = {}, {}, None, 0
    try:
        nc.createDimension("lead_hour", len(times))
        nc.createDimension("y", grid.height)
        nc.createDimension("x", grid.width)
        nc.setncattr("crs", grid.crs)
        nc.setncattr("grid_fingerprint", grid.fingerprint)
        nc.setncattr("issued_at", iso(issued_at))
        for name, values in (("latitude", lat), ("longitude", lon)):
            nc.createVariable(name, "f4", ("y", "x"))[:] = values
        time_var = nc.createVariable("valid_time", "f8", ("lead_hour",))
        time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        time_var[:] = [time.timestamp() for time in times]
        for index, fields in enumerate(hours):
            if index >= len(times):
                raise ValueError("too many hours")
            if index == 0:
                names = sorted(set(fields) & set(UNITS))
                if "danger_category" not in names:
                    raise ValueError("run has no danger category")
                for name in names:
                    zz.create(name, (len(times), grid.height, grid.width), ("lead_hour", "y", "x"), UNITS[name])
                    value = nc.createVariable(name, "f4", ("lead_hour", "y", "x"), zlib=True, complevel=4, fill_value=np.nan,
                                              chunksizes=(1, min(256, grid.height), min(256, grid.width)))
                    value.units = UNITS[name]
                    variables[name] = {"long_name": name.replace("_", " "), "units": UNITS[name], "source": "experimental hybrid", "derived": True}
                    (staging/"rasters").mkdir(exist_ok=True)
                    writers[name] = rasterio.open(staging/"rasters"/f"{name}.working.tif", "w", driver="GTiff", width=grid.width,
                        height=grid.height, count=len(times), dtype="float32", crs=grid.crs, transform=grid.transform,
                        nodata=-9999., tiled=True, blockxsize=256, blockysize=256, compress="DEFLATE")
            for name in variables:
                data = np.asarray(fields.get(name, np.full((grid.height,grid.width), np.nan)), "float32")
                if data.shape != (grid.height, grid.width):
                    raise ValueError(f"wrong grid shape for {name}")
                nc[name][index] = data
                zz.write(name, data, index)
                writers[name].write(np.where(np.isfinite(data), data, -9999.), index+1)
                writers[name].set_band_description(index+1, iso(times[index]))
            hazard = np.asarray(fields["danger_category"])
            peak = hazard.copy() if peak is None else np.fmax(peak, hazard)
            count += 1
        if count != len(times):
            raise ValueError("incomplete valid-time sequence")
    finally:
        nc.close()
        for writer in writers.values():
            writer.close()
    zz.finish()
    for name in variables:
        temporary = staging/"rasters"/f"{name}.working.tif"
        raster_copy(temporary, staging/"rasters"/f"{name}.tif", driver="COG", compress="DEFLATE", blocksize=256,
                    overview_resampling="nearest" if name in {"danger_category", "red_flag_criteria"} else "average")
        temporary.unlink()
    steps = metadata.get("step_kinds") or ["analysis" if time < issued_at else "forecast" for time in times]
    if len(steps)!=len(times) or not set(steps)<={"analysis","forecast"}:
        raise ValueError("invalid analysis/forecast step metadata")
    manifest = {**metadata, "schema": "fire-weather-hybrid-v1", "cycle": cycle, "label": "Experimental Missouri fire danger",
                "issued_at_utc": iso(issued_at), "forecast_hours": list(range(len(times))), "missing_hours": [],
                "valid_times_utc": [iso(time) for time in times], "step_kinds": steps,
                "step_labels": [time.astimezone(LOCAL_TZ).strftime('%I %p %Z').lstrip('0') + f" · {kind}"
                                for time,kind in zip(times,steps)],
                "forecast_lead_hours": [(time-issued_at).total_seconds()/3600 for time in times],
                "variables": variables, "complete": True, "kind": "hybrid-day", "grid": {**asdict(grid), "fingerprint": grid.fingerprint},
                "categories": list(LABELS), "colors": list(COLORS), "advisory_only": True, "public": False,
                "note": "1 km output spacing; source weather resolution and verified skill differ. Experimental, staff only."}
    atomic_json(staging/"manifest.json", manifest)
    if render_maps:
        from services.fire_weather_hybrid_maps import render
        manifest["images"] = render(staging)
    manifest["checksums"] = {str(path.relative_to(staging)).replace("\\", "/"): sha256(path)
                             for path in staging.rglob("*") if path.is_file()}
    manifest["checksums"].pop("manifest.json", None)
    atomic_json(staging/"manifest.json", manifest)
    os.replace(staging, final)
    previous = read_json(root/"latest.json") or {}
    if previous.get("cycle", "") < cycle:
        atomic_json(root/"latest.json", {"cycle": cycle})
    return manifest


def point(directory, lat, lon):
    import netCDF4
    manifest = read_json(Path(directory)/"manifest.json")
    grid_data = {key: value for key,value in manifest["grid"].items() if key != "fingerprint"}
    grid = Grid(**grid_data)
    row, col = grid.cell(lat, lon)
    with netCDF4.Dataset(Path(directory)/"forecast.nc") as ds:
        if "missouri_mask" in ds.variables and not ds["missouri_mask"][row, col]:
            raise ValueError("point outside Missouri")
        series = {}
        for name in manifest["variables"]:
            values = np.ma.filled(ds[name][:, row, col], np.nan)
            series[name] = [float(value) if np.isfinite(value) else None for value in values]
        if not any(value is not None for value in series.get("danger_category", [])):
            raise ValueError("point has no burnable forecast coverage")
        center_lat, center_lon = float(ds["latitude"][row,col]), float(ds["longitude"][row,col])
    return {"cycle": manifest["cycle"], "cell_id": f"{grid.id}:{row}:{col}", "row": row, "column": col,
            "grid_lat": center_lat, "grid_lon": center_lon,
            "distance_km": float(np.hypot((lat-center_lat)*111.2, (lon-center_lon)*111.2*np.cos(np.radians(lat)))),
            "lead_hours": manifest["forecast_hours"], "valid_times_utc": manifest["valid_times_utc"],
            "step_kinds": manifest["step_kinds"], "series": series, "units": {key:value["units"] for key,value in manifest["variables"].items()},
            "quality": manifest.get("quality"), "model_version": manifest.get("model_version")}


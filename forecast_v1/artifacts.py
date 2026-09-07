from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from rasterio.shutil import copy as raster_copy
from rasterio.transform import from_origin
import rasterio

from .contracts import ARCHIVE_ENCODINGS, PUBLIC_GRID, VARIABLE_UNITS, utc_rfc3339


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _netcdf_encoding(dataset: xr.Dataset) -> dict:
    encoding = {}
    for name in dataset.data_vars:
        spec = ARCHIVE_ENCODINGS.get(name)
        if not spec:
            encoding[name] = {"zlib": True, "complevel": 6, "shuffle": True}
            continue
        item = {"dtype": spec.dtype, "_FillValue": spec.fill_value, "zlib": True, "complevel": 6, "shuffle": True}
        if spec.scale_factor is not None:
            item["scale_factor"] = spec.scale_factor
        encoding[name] = item
    return encoding


def write_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    dataset.to_netcdf(temp, engine="netcdf4", encoding=_netcdf_encoding(dataset))
    with xr.open_dataset(temp) as reopened:
        if dict(reopened.sizes) != dict(dataset.sizes):
            raise ValueError(f"NetCDF round-trip dimension mismatch for {target}")
    os.replace(temp, target)
    return target


def write_cog(data: xr.DataArray, path: str | Path, *, variable: str, band_times: Iterable[str], categorical: bool = False, byte_data: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(data.values)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.shape[-2:] != (PUBLIC_GRID.height, PUBLIC_GRID.width):
        raise ValueError(f"public COG must be {PUBLIC_GRID.height}x{PUBLIC_GRID.width}, got {values.shape[-2:]}")
    dtype = "uint8" if categorical or byte_data else "float32"
    nodata = 255 if categorical or byte_data else -9999.0
    encoded = np.where(np.isfinite(values), values, nodata).astype(dtype)
    temp = target.with_suffix(".working.tif")
    with rasterio.open(
        temp, "w", driver="GTiff", width=PUBLIC_GRID.width, height=PUBLIC_GRID.height,
        count=encoded.shape[0], dtype=dtype, crs=PUBLIC_GRID.crs,
        transform=from_origin(PUBLIC_GRID.west, PUBLIC_GRID.north, PUBLIC_GRID.resolution_m, PUBLIC_GRID.resolution_m),
        tiled=True, blockxsize=256, blockysize=256, interleave="band", compress="DEFLATE",
        predictor=1 if dtype == "uint8" else 3, nodata=nodata,
    ) as dst:
        dst.write(encoded)
        timestamps = list(band_times)
        for band in range(1, encoded.shape[0] + 1):
            valid_time = timestamps[band - 1] if band - 1 < len(timestamps) else ""
            dst.set_band_description(band, f"{variable};lead={band - 1};valid={valid_time}")
            dst.update_tags(band, lead_hour=str(band - 1), valid_time=valid_time, units=VARIABLE_UNITS.get(variable, str(data.attrs.get("units", ""))))
        factors = [factor for factor in (2, 4, 8, 16) if min(PUBLIC_GRID.width, PUBLIC_GRID.height) // factor >= 16]
        if factors:
            dst.build_overviews(factors, Resampling.nearest if categorical else Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="nearest" if categorical else "average")
    raster_copy(temp, target, driver="COG", compress="DEFLATE", blocksize=256, overview_resampling="nearest" if categorical else "average")
    temp.unlink(missing_ok=True)
    with rasterio.open(target) as check:
        if check.count != encoded.shape[0] or not check.is_tiled:
            raise ValueError(f"COG validation failed for {target}")
    return target


def write_points_parquet(rows: list[dict], path: str | Path) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet publication requires the pinned pyarrow dependency") from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    sort_keys = [(name, "ascending") for name in ("station_id", "model", "member", "valid_time_utc") if name in table.column_names]
    if sort_keys:
        table = table.sort_by(sort_keys)
    pq.write_table(table, target, compression="zstd", use_dictionary=True, row_group_size=10_000)
    return target


def write_manifest(payload: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, target)
    return target


def write_static_graphic(
    data: xr.DataArray,
    png_path: str | Path,
    webp_path: str | Path,
    *,
    title: str,
    valid_period: str,
    run_time: str,
    status: str,
    categorical: bool = False,
    boundary_geojson: str | Path | None = None,
) -> tuple[Path, Path]:
    """Render a consistent 16:9 archival PNG and optimized WebP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from PIL import Image as PILImage

    png = Path(png_path)
    webp = Path(webp_path)
    png.parent.mkdir(parents=True, exist_ok=True)
    webp.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(16, 9), dpi=128)
    values = np.asarray(data.values, dtype=float)
    if categorical:
        values[values == 255] = np.nan
        colors = ["#90EE90", "#FFED4E", "#FFA500", "#FF0000", "#8B0000"]
        image = axis.imshow(values, cmap=ListedColormap(colors), norm=BoundaryNorm([-0.5, .5, 1.5, 2.5, 3.5, 4.5], 5), interpolation="nearest")
        colorbar = figure.colorbar(image, ax=axis, ticks=range(5), fraction=.035, pad=.025)
        colorbar.ax.set_yticklabels(["Low", "Moderate", "Elevated", "Critical", "Extreme"])
    else:
        image = axis.imshow(values, cmap="viridis", interpolation="bilinear")
        figure.colorbar(image, ax=axis, fraction=.035, pad=.025, label=str(data.attrs.get("units", "")))
    if boundary_geojson and Path(boundary_geojson).is_file():
        from pyproj import Transformer
        boundary = json.loads(Path(boundary_geojson).read_text(encoding="utf-8"))
        transformer = Transformer.from_crs("EPSG:4326", PUBLIC_GRID.crs, always_xy=True)

        def rings(geometry: dict):
            if geometry.get("type") == "Polygon":
                yield from geometry.get("coordinates", [])[:1]
            elif geometry.get("type") == "MultiPolygon":
                for polygon in geometry.get("coordinates", []):
                    yield from polygon[:1]

        features = boundary.get("features", []) if boundary.get("type") == "FeatureCollection" else [boundary]
        for feature in features:
            geometry = feature.get("geometry", feature)
            for ring in rings(geometry):
                projected = [transformer.transform(float(lon), float(lat)) for lon, lat, *_ in ring]
                columns = [(x - PUBLIC_GRID.west) / PUBLIC_GRID.resolution_m - .5 for x, _ in projected]
                rows = [(PUBLIC_GRID.north - y) / PUBLIC_GRID.resolution_m - .5 for _, y in projected]
                axis.plot(columns, rows, color="#111827", linewidth=1.5, alpha=.9)
    axis.set_title(title, fontsize=22, weight="bold", pad=15)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.text(0, -0.035, f"Valid: {valid_period}", transform=axis.transAxes, ha="left", va="top", fontsize=11)
    axis.text(1, -0.035, f"12Z run: {run_time}  •  {status}", transform=axis.transAxes, ha="right", va="top", fontsize=11)
    figure.tight_layout(rect=(0.02, .05, .98, .96))
    figure.savefig(png, format="png", dpi=128, metadata={"Title": title, "Description": f"{valid_period}; {status}"})
    plt.close(figure)
    with PILImage.open(png) as source:
        source.save(webp, "WEBP", quality=86, method=6)
    return png, webp


def artifact_record(path: Path, *, run_id: str, kind: str, variable: str = "", aggregation: str = "", object_key: str | None = None, dtype: str = "binary", unit: str | None = None, valid_start: str | None = None, valid_end: str | None = None) -> dict:
    return {
        "run_id": run_id, "kind": kind, "variable": variable, "aggregation": aggregation,
        "valid_start_utc": valid_start, "valid_end_utc": valid_end, "unit": unit,
        "storage_data_type": dtype, "crs": PUBLIC_GRID.crs if kind == "raster" else None,
        "grid_id": PUBLIC_GRID.id if kind in {"raster", "cube"} else None,
        "object_key": object_key, "local_path": str(path.resolve()), "checksum": sha256_file(path),
        "byte_size": path.stat().st_size, "status": "ready",
    }


def promote_directory(staging: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"immutable run directory already exists: {final}")
    os.replace(staging, final)

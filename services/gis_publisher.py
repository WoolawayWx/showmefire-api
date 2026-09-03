"""Atomic publication of Show Me Fire GIS products for MapServer.

Operational rasters use one Missouri-wide EPSG:32615 grid.  GeoJSON remains
RFC 7946 longitude/latitude while GeoPackages use the operational CRS.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy.interpolate import griddata
from shapely.geometry import box

from core.config import GIS_DIR

CRS = "EPSG:32615"
GEOJSON_CRS = "EPSG:4326"
MISSOURI_BOUNDS_WGS84 = (-95.8, 35.8, -89.1, 40.8)
RESOLUTION_METERS = int(os.getenv("SMF_GIS_RESOLUTION_METERS", "3000"))
RETENTION_DAYS = int(os.getenv("SMF_GIS_RETENTION_DAYS", "30"))
SCHEMA_VERSION = "1.0.0"
PUBLISH_ROOT = Path(os.getenv("SMF_GIS_PUBLISH_DIR", str(GIS_DIR)))


def _utc(value: datetime | str | None = None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_grid(resolution: int = RESOLUTION_METERS) -> dict[str, Any]:
    west, south, east, north = transform_bounds(
        "EPSG:4326", CRS, *MISSOURI_BOUNDS_WGS84, densify_pts=41
    )
    west = math.floor(west / resolution) * resolution
    south = math.floor(south / resolution) * resolution
    east = math.ceil(east / resolution) * resolution
    north = math.ceil(north / resolution) * resolution
    return {
        "crs": CRS,
        "bounds": [west, south, east, north],
        "resolution": resolution,
        "width": int(round((east - west) / resolution)),
        "height": int(round((north - south) / resolution)),
        "transform": from_origin(west, north, resolution, resolution),
    }


def regrid_lonlat(
    values: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    categorical: bool = False,
    accumulated: bool = False,
) -> np.ndarray:
    """Interpolate a curvilinear lon/lat grid directly onto EPSG:32615."""
    values = np.asarray(values)
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)
    if longitude.ndim == 1 and latitude.ndim == 1:
        longitude, latitude = np.meshgrid(longitude, latitude)
    if values.shape != longitude.shape or values.shape != latitude.shape:
        raise ValueError("values, longitude, and latitude must have identical shapes")
    longitude = np.where(longitude > 180, longitude - 360, longitude)
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    source_x, source_y = transformer.transform(longitude, latitude)
    grid = canonical_grid()
    west, _, _, north = grid["bounds"]
    target_x = west + (np.arange(grid["width"]) + 0.5) * grid["resolution"]
    target_y = north - (np.arange(grid["height"]) + 0.5) * grid["resolution"]
    xx, yy = np.meshgrid(target_x, target_y)
    valid = np.isfinite(values) & np.isfinite(source_x) & np.isfinite(source_y)
    if not valid.any():
        return np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
    result = griddata(
        np.column_stack((source_x[valid], source_y[valid])),
        values[valid],
        (xx, yy),
        method="nearest" if categorical else "linear",
        fill_value=np.nan,
    )
    # Curvilinear accumulated fields are interpolated in projected space and
    # then mean-corrected. This preserves the domain-integrated magnitude to a
    # tight tolerance without blending categorical fields.
    if accumulated:
        source_mean = float(np.nanmean(values[valid]))
        target_mean = float(np.nanmean(result))
        if np.isfinite(source_mean) and np.isfinite(target_mean) and target_mean != 0:
            result = result * (source_mean / target_mean)
        if np.nanmin(values[valid]) >= 0:
            result = np.maximum(result, 0)
    return result.astype(np.float32)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_catalog(root: Path) -> dict[str, Any]:
    try:
        return json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        grid = canonical_grid()
        return {
            "schema_version": SCHEMA_VERSION,
            "native_crs": CRS,
            "supported_crs": [CRS, "EPSG:4326", "EPSG:3857"],
            "grid": {k: v for k, v in grid.items() if k != "transform"},
            "products": {},
        }


def create_staging_root(*, root: Path | None = None) -> Path:
    """Create a same-filesystem staging tree seeded with the active catalog."""
    root = Path(root or PUBLISH_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=root))
    catalog = root / "catalog.json"
    if catalog.is_file():
        shutil.copy2(catalog, staging / "catalog.json")
    return staging


def discard_staging_root(staging: Path | None) -> None:
    if staging and staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def commit_staging_root(staging: Path, *, root: Path | None = None) -> None:
    """Promote all staged files, making the new catalog visible last."""
    root = Path(root or PUBLISH_ROOT)
    changed_products = {
        path.parent.name for path in (staging / "rasters").glob("*/*.tif")
    } if (staging / "rasters").exists() else set()
    for directory in ("rasters", "latest"):
        source_root = staging / directory
        if not source_root.exists():
            continue
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            destination = root / source.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
    staged_catalog_path = staging / "catalog.json"
    if not staged_catalog_path.is_file():
        raise ValueError("staged publication has no catalog")
    staged_catalog = _load_catalog(staging)
    active_catalog = _load_catalog(root)
    for product in changed_products:
        active_catalog.setdefault("products", {})[product] = staged_catalog["products"][product]
    active_catalog["updated_at"] = staged_catalog.get("updated_at", _utc())
    _atomic_json(root / "catalog.json", active_catalog)
    _rebuild_raster_index(root, _load_catalog(root))
    discard_staging_root(staging)


def _write_raster(path: Path, values: np.ndarray, *, categorical: bool, tags: Mapping[str, Any]) -> None:
    grid = canonical_grid()
    if values.shape != (grid["height"], grid["width"]):
        raise ValueError(f"raster shape {values.shape} does not match canonical grid")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tif", dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    nodata = 255 if categorical else -9999.0
    dtype = "uint8" if categorical else "float32"
    encoded = np.where(np.isfinite(values), np.rint(values) if categorical else values, nodata).astype(dtype)
    try:
        with rasterio.open(
            temp_path, "w", driver="GTiff", height=grid["height"], width=grid["width"],
            count=1, dtype=dtype, crs=CRS, transform=grid["transform"], nodata=nodata,
            compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        ) as dst:
            dst.write(encoded, 1)
            dst.set_band_description(1, str(tags.get("product", "Show Me Fire product")))
            dst.update_tags(**{str(k).upper(): str(v) for k, v in tags.items() if v is not None})
            factors = [factor for factor in (2, 4, 8, 16) if min(grid["width"], grid["height"]) // factor >= 1]
            if factors:
                dst.build_overviews(factors, Resampling.nearest if categorical else Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="nearest" if categorical else "average")
        with rasterio.open(temp_path) as check:
            if check.crs.to_string() != CRS or check.count != 1:
                raise ValueError("published raster failed CRS/band validation")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _rebuild_raster_index(root: Path, catalog: Mapping[str, Any]) -> None:
    rows = []
    grid = canonical_grid()
    footprint = box(*grid["bounds"])
    for product, record in catalog.get("products", {}).items():
        for item in record.get("items", []):
            rows.append({
                "location": f"/data/{item['path'].replace(os.sep, '/')}",
                "product": product,
                "run_time": item.get("run_time"),
                "valid_time": item.get("valid_time") or item.get("observation_time"),
                "latest": int(bool(item.get("latest"))),
                "geometry": footprint,
            })
    if not rows:
        return
    index = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    fd, temporary = tempfile.mkstemp(prefix=".raster_catalog.", suffix=".gpkg", dir=root)
    os.close(fd)
    temp_path = Path(temporary)
    temp_path.unlink(missing_ok=True)
    try:
        index.to_file(temp_path, layer="rasters", driver="GPKG")
        os.replace(temp_path, root / "raster_catalog.gpkg")
    finally:
        temp_path.unlink(missing_ok=True)


def publish_raster(
    product: str,
    values: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    units: str,
    run_time: datetime | str | None = None,
    valid_time: datetime | str | None = None,
    observation_time: datetime | str | None = None,
    categorical: bool = False,
    accumulated: bool = False,
    style: str = "continuous",
    source: str = "Show Me Fire",
    root: Path | None = None,
    rebuild_index: bool = True,
) -> Path:
    root = Path(root or PUBLISH_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    generated_at = _utc()
    timestamp = _utc(valid_time or observation_time or run_time).replace("-", "").replace(":", "")
    timestamp = timestamp.replace("T", "_").replace("Z", "Z")
    relative = Path("rasters") / product / f"{timestamp}.tif"
    projected = regrid_lonlat(
        values, longitude, latitude, categorical=categorical, accumulated=accumulated,
    )
    metadata = {
        "product": product, "units": units, "source": source, "run_time": _utc(run_time) if run_time else None,
        "valid_time": _utc(valid_time) if valid_time else None,
        "observation_time": _utc(observation_time) if observation_time else None,
        "generated_at": generated_at, "processing_version": SCHEMA_VERSION, "crs": CRS,
    }
    _write_raster(root / relative, projected, categorical=categorical, tags=metadata)
    latest_relative = Path("latest") / f"{product}.tif"
    latest_path = root / latest_relative
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{product}.", suffix=".tif", dir=latest_path.parent)
    os.close(fd)
    shutil.copy2(root / relative, temporary)
    os.replace(temporary, latest_path)

    catalog = _load_catalog(root)
    record = catalog.setdefault("products", {}).setdefault(product, {
        "kind": "raster", "units": units, "style": style, "wms_layer": product, "items": [],
    })
    for item in record["items"]:
        item["latest"] = False
    item = {**metadata, "path": relative.as_posix(), "latest": True}
    record["items"] = [old for old in record["items"] if old.get("path") != item["path"]] + [item]
    record["latest"] = latest_relative.as_posix()
    record["status"] = "ready"
    record["updated_at"] = generated_at
    catalog["updated_at"] = generated_at
    _atomic_json(root / "catalog.json", catalog)
    if rebuild_index:
        _rebuild_raster_index(root, catalog)
    return root / relative


def rebuild_raster_index(*, root: Path | None = None) -> None:
    root = Path(root or PUBLISH_ROOT)
    _rebuild_raster_index(root, _load_catalog(root))


def mark_stale(product: str, error: str, *, root: Path | None = None) -> None:
    root = Path(root or PUBLISH_ROOT)
    catalog = _load_catalog(root)
    record = catalog.setdefault("products", {}).setdefault(product, {"items": []})
    record.update({"status": "stale", "error": str(error), "checked_at": _utc()})
    catalog["updated_at"] = _utc()
    _atomic_json(root / "catalog.json", catalog)


def cleanup_retention(*, root: Path | None = None, now: datetime | None = None) -> list[str]:
    root = Path(root or PUBLISH_ROOT)
    catalog = _load_catalog(root)
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=RETENTION_DAYS)
    removed: list[str] = []
    for record in catalog.get("products", {}).values():
        kept = []
        for item in record.get("items", []):
            stamp = item.get("valid_time") or item.get("observation_time") or item.get("run_time")
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else cutoff
            if not item.get("latest") and parsed < cutoff:
                target = root / item["path"]
                target.unlink(missing_ok=True)
                removed.append(item["path"])
            else:
                kept.append(item)
        record["items"] = kept
    if removed:
        catalog["updated_at"] = _utc()
        _atomic_json(root / "catalog.json", catalog)
        _rebuild_raster_index(root, catalog)
    return removed


def publish_vectors(
    product: str,
    features: Iterable[Mapping[str, Any]],
    *,
    generated_at: datetime | str | None = None,
    root: Path | None = None,
) -> dict[str, Path]:
    root = Path(root or PUBLISH_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    stamp = _utc(generated_at)
    collection = {"type": "FeatureCollection", "metadata": {"product": product, "generated_at": stamp}, "features": list(features)}
    geojson_path = root / "vectors" / f"{product}.geojson"
    _atomic_json(geojson_path, collection)
    if collection["features"]:
        frame = gpd.GeoDataFrame.from_features(collection["features"], crs=GEOJSON_CRS)
    else:
        frame = gpd.GeoDataFrame(
            {"product": []}, geometry=gpd.GeoSeries([], crs=GEOJSON_CRS), crs=GEOJSON_CRS,
        )
    frame = frame.to_crs(CRS)
    gpkg_path = root / "vectors" / f"{product}.gpkg"
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{product}.", suffix=".gpkg", dir=gpkg_path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    temp_path.unlink(missing_ok=True)
    try:
        frame.to_file(temp_path, layer=product, driver="GPKG")
        os.replace(temp_path, gpkg_path)
    finally:
        temp_path.unlink(missing_ok=True)
    catalog = _load_catalog(root)
    catalog.setdefault("products", {})[product] = {
        "kind": "vector", "wms_layer": product, "wfs_layer": product, "status": "ready",
        "updated_at": stamp, "geojson": geojson_path.relative_to(root).as_posix(),
        "geopackage": gpkg_path.relative_to(root).as_posix(), "feature_count": len(frame),
    }
    catalog["updated_at"] = stamp
    _atomic_json(root / "catalog.json", catalog)
    return {"geojson": geojson_path, "geopackage": gpkg_path}

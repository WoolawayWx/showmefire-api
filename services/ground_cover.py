"""Ground cover and fuel mix for an incident from the published `ground_cover_static` product (480 m cells).

The product (built in ShowMeFire-Models, see docs/ground_cover_product_plan.md) stores for each 480 m EPSG:5070 cell the
share of each NLCD land-cover class and each LANDFIRE FBFM40 fuel group, plus canopy. This module samples it over the
area an incident actually covers, so the answer is a real mix rather than the single value the 3 km bundle gives:

  * NGFS incidents       the union of the members' detection-footprint pixel polygons
  * incidents with an outline   the stored outline polygon
  * point-only sources   the 3 x 3 block of cells around the point (about 1.4 km square)

Cells are weighted by how much of each lies inside the polygon. The module never raises to callers of `sample()`;
it returns None when the bundle is unavailable or the incident is off the grid, and the caller falls back.

Production never builds or downloads geography: the bundle arrives only through the model registry
(`ground_cover_static`, stable channel), or from GROUND_COVER_STATIC_BUNDLE for development and tests.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

CELL_M = 480
RESOLUTION_KM = 0.48
SOURCE = "NLCD land cover and LANDFIRE fuel models (Show Me Fire 480 m ground cover product)"

NLCD_VARS = (
    ("nlcd_water_pct", "Water"), ("nlcd_developed_pct", "Developed"), ("nlcd_barren_pct", "Barren"),
    ("nlcd_forest_pct", "Forest"), ("nlcd_shrub_pct", "Shrub"), ("nlcd_grass_pct", "Grassland"),
    ("nlcd_pasture_pct", "Pasture/hay"), ("nlcd_crops_pct", "Cultivated crops"),
    ("nlcd_wetlands_pct", "Wetlands"), ("nlcd_other_pct", "Other"),
)
# Same labels as services/incident_fuel.py `_RANGES` so the panel can use either source unchanged.
FBFM_VARS = (
    ("fbfm_nb_developed_pct", "Developed"), ("fbfm_nb_snow_pct", "Snow and ice"),
    ("fbfm_nb_cropland_pct", "Cropland"), ("fbfm_nb_water_pct", "Water"), ("fbfm_nb_bare_pct", "Bare ground"),
    ("fbfm_grass_pct", "Grass"), ("fbfm_grass_shrub_pct", "Grass and shrub"), ("fbfm_shrub_pct", "Shrub"),
    ("fbfm_timber_understory_pct", "Timber with understory"), ("fbfm_timber_litter_pct", "Timber litter"),
    ("fbfm_slash_pct", "Slash and blowdown"), ("fbfm_other_pct", "Other"),
)
# NOAA NGFS class names built from NLCD groups. Not a guess: tested against NOAA's own LAND_COVER strings on 24 live
# incidents (median error 0.8-4 points per class); counting pasture/hay as Cropland instead gave ~33 points of error.
NOAA_CLASSES = (
    ("Trees", ("nlcd_forest_pct", "nlcd_wetlands_pct")),
    ("Grass/Herbs", ("nlcd_grass_pct", "nlcd_pasture_pct")),
    ("Shrubs", ("nlcd_shrub_pct",)),
    ("Cropland", ("nlcd_crops_pct",)),
    ("Developed", ("nlcd_developed_pct",)),
    ("Water", ("nlcd_water_pct",)),
    ("Barren", ("nlcd_barren_pct",)),
)
MODE_FILL = -9999

_RUNTIME: Optional[dict] = None
_DIAGNOSTICS = {"available": False, "fallback_reason": "not initialized", "bundle_version": None}
_TRANSFORMER = None


def diagnostics() -> dict:
    return dict(_DIAGNOSTICS)


def _env_bundle_path() -> Optional[Path]:
    override = os.getenv("GROUND_COVER_STATIC_BUNDLE")
    if not override:
        return None
    path = Path(override)
    return path if path.is_file() else None


def _transformer():
    global _TRANSFORMER
    if _TRANSFORMER is None:
        import pyproj

        _TRANSFORMER = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    return _TRANSFORMER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locate_bundle() -> tuple[Path, Path, str]:
    env_path = _env_bundle_path()
    if env_path is not None:
        return env_path, env_path.with_suffix(".json"), "env-override"
    from models.versioning import load_active_assets

    assets = load_active_assets("ground_cover_static", "stable")
    bundle = assets["static_bundle"]["path"]
    manifest = assets.get("static_manifest", {}).get("path") or bundle.with_suffix(".json")
    version = None
    try:
        version = json.loads(Path(manifest).read_text()).get("bundle_version")
    except (OSError, ValueError):
        pass
    return Path(bundle), Path(manifest), version or "unknown"


def _load() -> dict:
    """Open the bundle into memory (about 90 MB), re-reading only when the file changes. Raises if unavailable."""
    global _RUNTIME
    import xarray as xr

    bundle, manifest, version = _locate_bundle()
    signature = (str(bundle), bundle.stat().st_mtime)
    if _RUNTIME and _RUNTIME["signature"] == signature:
        return _RUNTIME
    if manifest.is_file():
        expected = json.loads(manifest.read_text()).get("sha256")
        if expected and _sha256(bundle) != expected:
            raise ValueError("ground cover static bundle checksum mismatch")
    with xr.open_dataset(bundle) as ds:
        ds = ds.load()
    x, y = np.asarray(ds["x"].values, dtype=float), np.asarray(ds["y"].values, dtype=float)
    if len(x) < 2 or len(y) < 2 or abs((x[1] - x[0]) - CELL_M) > 1e-6:
        raise ValueError(f"ground cover bundle is not on a {CELL_M} m grid")
    runtime = {
        "signature": signature, "bundle_version": version, "bundle_path": str(bundle),
        "x0": float(x[0]) - CELL_M / 2, "y1": float(y[0]) + CELL_M / 2, "ncols": len(x), "nrows": len(y),
        "nlcd": np.stack([ds[name].values for name, _ in NLCD_VARS]).astype(np.uint8),
        "fbfm": np.stack([ds[name].values for name, _ in FBFM_VARS]).astype(np.uint8),
        # xarray decodes the -9999 fill value to NaN; casting NaN straight to int16 gives an arbitrary "code".
        "mode": np.nan_to_num(np.asarray(ds["fbfm40_mode"].values, dtype=float), nan=MODE_FILL).astype(np.int16),
        "cover": np.asarray(ds["canopy_cover_mean_pct"].values, dtype=np.float32),
        "height": np.asarray(ds["canopy_height_mean_m"].values, dtype=np.float32),
        "valid": np.asarray(ds["valid_pct"].values, dtype=np.uint8),
    }
    _RUNTIME = runtime
    _DIAGNOSTICS.update({"available": True, "fallback_reason": None, "bundle_version": version,
                         "bundle_path": str(bundle)})
    return runtime


def _runtime() -> Optional[dict]:
    try:
        return _load()
    except Exception as exc:  # noqa: BLE001 - the product is optional; callers fall back
        logger.info("ground cover bundle unavailable: %s", exc)
        _DIAGNOSTICS.update({"available": False, "fallback_reason": str(exc)})
        return None


# ------------------------------------------------------------------------------------------ geometry
def _geometries(value: Any) -> list[dict]:
    """GeoJSON geometries from a JSON string / geometry / Feature / FeatureCollection; bad input yields []."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, dict):
        return []
    kind = value.get("type")
    if kind == "FeatureCollection":
        return [g for f in value.get("features") or [] for g in _geometries(f)]
    if kind == "Feature":
        return _geometries(value.get("geometry"))
    return [value] if kind in ("Polygon", "MultiPolygon") and value.get("coordinates") else []


def _projected_union(texts: tuple[str, ...]):
    from shapely.geometry import shape
    from shapely.ops import transform, unary_union

    polygons = []
    for text in texts:
        for geometry in _geometries(text):
            try:
                polygon = shape(geometry)
            except Exception:  # noqa: BLE001
                continue
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                polygons.append(transform(_transformer().transform, polygon))
    return unary_union(polygons) if polygons else None


def _polygon_cells(rt: dict, geometry) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import shapely

    minx, miny, maxx, maxy = geometry.bounds
    c0, c1 = int((minx - rt["x0"]) // CELL_M), int((maxx - rt["x0"]) // CELL_M)
    r0, r1 = int((rt["y1"] - maxy) // CELL_M), int((rt["y1"] - miny) // CELL_M)
    c0, c1 = max(c0, 0), min(c1, rt["ncols"] - 1)
    r0, r1 = max(r0, 0), min(r1, rt["nrows"] - 1)
    if c0 > c1 or r0 > r1:
        return None
    rows, cols = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    left = rt["x0"] + cols * CELL_M
    top = rt["y1"] - rows * CELL_M
    boxes = shapely.box(left, top - CELL_M, left + CELL_M, top)
    weights = shapely.area(shapely.intersection(boxes, geometry))
    keep = weights > 0
    return (rows[keep], cols[keep], weights[keep]) if keep.any() else None


def _point_cells(rt: dict, latitude: float, longitude: float) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x, y = _transformer().transform(longitude, latitude)
    col, row = int((x - rt["x0"]) // CELL_M), int((rt["y1"] - y) // CELL_M)
    if not (0 <= col < rt["ncols"] and 0 <= row < rt["nrows"]):
        return None
    rows, cols = np.meshgrid(np.arange(max(row - 1, 0), min(row + 2, rt["nrows"])),
                             np.arange(max(col - 1, 0), min(col + 2, rt["ncols"])), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    return rows, cols, np.ones(len(rows))


# ---------------------------------------------------------------------------------------- aggregation
def _percent(values: np.ndarray) -> list[int]:
    """Round non-negative shares to integers that sum to exactly 100 (largest remainder)."""
    total = float(values.sum())
    if total <= 0:
        return [0] * len(values)
    exact = values * 100.0 / total
    base = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - base), kind="stable")[: 100 - int(base.sum())]:
        base[index] += 1
    return base.tolist()


def _groups(labels, pct: list[int]) -> list[dict]:
    rows = [{"group": label, "pct": p} for (_, label), p in zip(labels, pct) if p > 0]
    return sorted(rows, key=lambda row: -row["pct"])


def _aggregate(rt: dict, rows, cols, weights, basis: str) -> Optional[dict[str, Any]]:
    usable = rt["valid"][rows, cols] > 0
    if not usable.any():
        return None
    rows, cols, weights = rows[usable], cols[usable], weights[usable].astype(float)
    total = float(weights.sum())
    nlcd = (rt["nlcd"][:, rows, cols] * weights).sum(axis=1) / total
    fbfm = (rt["fbfm"][:, rows, cols] * weights).sum(axis=1) / total
    nlcd_pct = dict(zip([n for n, _ in NLCD_VARS], _percent(nlcd)))
    ground = [{"group": name, "pct": sum(nlcd_pct[v] for v in members)} for name, members in NOAA_CLASSES]
    ground.append({"group": "Other", "pct": nlcd_pct["nlcd_other_pct"]})
    ground = sorted((g for g in ground if g["pct"] > 0), key=lambda g: -g["pct"])

    mode_codes = rt["mode"][rows, cols]
    mode_weight: dict[int, float] = {}
    for code, weight in zip(mode_codes.tolist(), weights.tolist()):
        if code != MODE_FILL:
            mode_weight[code] = mode_weight.get(code, 0.0) + weight

    def weighted_mean(array):
        values = array[rows, cols].astype(float)
        ok = np.isfinite(values)
        return round(float((values[ok] * weights[ok]).sum() / weights[ok].sum()), 1) if ok.any() else None

    return {
        "groups": _groups(FBFM_VARS, _percent(fbfm)),
        "nlcd_groups": _groups(NLCD_VARS, list(nlcd_pct.values())),
        "ground_cover": ground,
        "fuel_model_fbfm40": max(mode_weight, key=mode_weight.get) if mode_weight else None,
        "canopy_cover_pct": weighted_mean(rt["cover"]),
        "canopy_height_m": weighted_mean(rt["height"]),
        "cells": int(len(rows)),
        "resolution_km": RESOLUTION_KM,
        "basis": basis,
        "source": SOURCE,
        "bundle_version": rt["bundle_version"],
    }


def _compute(footprints: tuple[str, ...], outline: Optional[str], latitude: float, longitude: float):
    rt = _runtime()
    if rt is None:
        return None
    for basis, texts in (("footprint", footprints), ("outline", (outline,) if outline else ())):
        if not texts:
            continue
        geometry = _projected_union(texts)
        cells = _polygon_cells(rt, geometry) if geometry is not None else None
        result = _aggregate(rt, *cells, basis) if cells else None
        if result:
            return result
    cells = _point_cells(rt, latitude, longitude)
    return _aggregate(rt, *cells, "point") if cells else None


@lru_cache(maxsize=1024)
def _cached(footprints, outline, latitude, longitude, version):
    return _compute(footprints, outline, latitude, longitude)


def sample(latitude: Any, longitude: Any, footprints: Any = None, outline: Any = None) -> Optional[dict[str, Any]]:
    """Ground cover and fuel mix for an incident, or None. `footprints` is a list of footprint GeoJSON (strings or
    dicts, duplicates welcome), `outline` the incident outline. Never raises."""
    try:
        lat, lon = float(latitude), float(longitude)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        texts = tuple(sorted({t if isinstance(t, str) else json.dumps(t, sort_keys=True)
                              for t in (footprints or []) if t}))
        outline_text = outline if isinstance(outline, str) or outline is None else json.dumps(outline, sort_keys=True)
        rt = _runtime()
        if rt is None:
            return None
        # Rounded to ~100 m so repeated reads of one incident hit the cache; the bundle version is in the key so a
        # newly promoted bundle is never answered from the old one.
        result = _cached(texts, outline_text or None, round(lat, 3), round(lon, 3), rt["bundle_version"])
        return json.loads(json.dumps(result)) if result else None
    except Exception:  # noqa: BLE001 - enrichment only; the incident still renders without it
        logger.info("ground cover sample failed", exc_info=True)
        return None

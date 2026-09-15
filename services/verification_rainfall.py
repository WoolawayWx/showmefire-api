"""Rainfall-conditioned fire-danger verification adjustments.

This module deliberately contains policy, not data acquisition.  Callers pass
already-normalized precipitation in millimetres and the service returns a
bounded category adjustment plus provenance suitable for verification reports.
The contract is versioned so historical reports remain interpretable when the
policy changes.
"""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "verification-rainfall-v1"
MM_PER_INCH = 25.4


@dataclass(frozen=True)
class FuelRegime:
    name: str
    threshold_mm: float
    relief_hours: float
    nlcd_classes: tuple[int, ...]


FUEL_REGIMES: tuple[FuelRegime, ...] = (
    FuelRegime("grass_pasture", 2.5, 18.0, (71, 72, 73, 74, 81)),
    FuelRegime("agriculture", 5.0, 48.0, (21, 22, 23, 24, 31, 82)),
    FuelRegime("shrubland", 6.3, 72.0, (51, 52)),
    FuelRegime("open_woodland", 12.7, 120.0, (90,)),
    FuelRegime("dense_forest", 38.1, 336.0, (41, 42, 43)),
)

# The tuples above intentionally document the broad NLCD classes, but a class
# can only have one operational regime.  This ordered map is the authoritative
# lookup and avoids ambiguity for forest classes.
NLCD_TO_REGIME: dict[int, str] = {
    21: "agriculture", 22: "agriculture", 23: "agriculture", 24: "agriculture",
    31: "agriculture",
    41: "dense_forest", 42: "dense_forest", 43: "dense_forest",
    51: "shrubland", 52: "shrubland",
    71: "grass_pasture", 72: "grass_pasture", 73: "grass_pasture", 74: "grass_pasture",
    81: "grass_pasture", 82: "agriculture",
    90: "open_woodland",
}

REGIME_BY_NAME = {item.name: item for item in FUEL_REGIMES}


def default_nlcd_raster_path() -> Path:
    """Return the standard NLCD path used by the acquisition script."""
    return (
        Path("/app/data/static/nlcd_class.tif")
        if Path("/app").exists()
        else Path(__file__).resolve().parents[1] / "data" / "static" / "nlcd_class.tif"
    )


def contract() -> dict[str, Any]:
    """Return the serializable policy contract used by generated reports."""
    return {
        "version": CONTRACT_VERSION,
        "mm_per_inch": MM_PER_INCH,
        "nlcd_to_regime": dict(NLCD_TO_REGIME),
        "regimes": {item.name: asdict(item) for item in FUEL_REGIMES},
        "formula": {
            "rainfall_fraction": "clamp(accumulation_mm / threshold_mm, 0, 1)",
            "time_decay": "exp(-hours_since_rain / relief_hours)",
            "weather_factor": "clamp(0.65 + 0.35 * RH/60 - 0.25 * wind_kts/25, 0.35, 1.0)",
            "category_reduction": "round(2 * rainfall_fraction * time_decay * weather_factor)",
        },
    }


def regime_for_nlcd(value: Any) -> str | None:
    """Map a scalar NLCD class to a documented fuel regime."""
    try:
        if value is None or not np.isfinite(float(value)):
            return None
        return NLCD_TO_REGIME.get(int(round(float(value))))
    except (TypeError, ValueError):
        return None


def _weather_factor(relative_humidity: float | None, wind_kts: float | None) -> float:
    rh = 45.0 if relative_humidity is None or not np.isfinite(relative_humidity) else float(relative_humidity)
    wind = 0.0 if wind_kts is None or not np.isfinite(wind_kts) else max(0.0, float(wind_kts))
    return float(np.clip(0.65 + 0.35 * np.clip(rh, 0, 100) / 60.0 - 0.25 * wind / 25.0, 0.35, 1.0))


def category_reduction(
    rainfall_mm: float | None,
    regime: str | None,
    *,
    hours_since_rain: float = 0.0,
    relative_humidity: float | None = None,
    wind_kts: float | None = None,
) -> dict[str, Any]:
    """Return the bounded ordinal reduction caused by realized rainfall.

    A complete, recent threshold event can lower danger by at most two
    categories.  It never increases danger and never independently proves
    that fuels are Low.  Missing rainfall or land use returns zero reduction
    with an explicit reason.
    """
    result = {
        "contract_version": CONTRACT_VERSION,
        "regime": regime,
        "rainfall_mm": None if rainfall_mm is None else float(rainfall_mm),
        "reduction": 0,
        "rainfall_fraction": 0.0,
        "time_decay": 0.0,
        "weather_factor": None,
        "reason": None,
    }
    if regime not in REGIME_BY_NAME:
        result["reason"] = "land_use_unavailable"
        return result
    if rainfall_mm is None or not np.isfinite(rainfall_mm) or float(rainfall_mm) <= 0:
        result["reason"] = "rainfall_unavailable"
        return result
    spec = REGIME_BY_NAME[regime]
    rain_fraction = float(np.clip(float(rainfall_mm) / spec.threshold_mm, 0.0, 1.0))
    decay = float(np.exp(-max(0.0, float(hours_since_rain)) / spec.relief_hours))
    weather = _weather_factor(relative_humidity, wind_kts)
    reduction = int(np.clip(np.rint(2.0 * rain_fraction * decay * weather), 0, 2))
    result.update(
        rainfall_fraction=round(rain_fraction, 4),
        time_decay=round(decay, 4),
        weather_factor=round(weather, 4),
        reduction=reduction,
        reason="applied" if reduction else "insufficient_or_expired_relief",
        threshold_mm=spec.threshold_mm,
        relief_hours=spec.relief_hours,
    )
    return result


def adjust_category(raw_category: Any, suppression: Mapping[str, Any]) -> int | None:
    """Apply a suppression result to a 0..4 danger category."""
    try:
        if raw_category is None or not np.isfinite(float(raw_category)):
            return None
        raw = int(np.clip(np.rint(float(raw_category)), 0, 4))
        reduction = int(np.clip(suppression.get("reduction", 0), 0, 2))
        return max(0, raw - reduction)
    except (TypeError, ValueError):
        return None


def adjust_grid(
    raw_grid: np.ndarray,
    rainfall_mm: np.ndarray | float | None,
    nlcd_grid: np.ndarray | None,
    *,
    hours_since_rain: np.ndarray | float = 0.0,
    relative_humidity: np.ndarray | float | None = None,
    wind_kts: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust a danger grid and return ``(adjusted, reduction)`` arrays."""
    raw = np.asarray(raw_grid, dtype=float)
    shape = raw.shape
    rain = np.broadcast_to(np.nan if rainfall_mm is None else rainfall_mm, shape)
    nlcd = np.broadcast_to(np.nan, shape) if nlcd_grid is None else np.broadcast_to(nlcd_grid, shape)
    age = np.broadcast_to(hours_since_rain, shape)
    rh = np.broadcast_to(np.nan if relative_humidity is None else relative_humidity, shape)
    wind = np.broadcast_to(np.nan if wind_kts is None else wind_kts, shape)
    adjusted = np.full(shape, np.nan, dtype=float)
    reductions = np.zeros(shape, dtype=np.uint8)
    for index in np.ndindex(shape):
        suppression = category_reduction(
            rain[index],
            regime_for_nlcd(nlcd[index]),
            hours_since_rain=float(age[index]) if np.isfinite(age[index]) else 0.0,
            relative_humidity=float(rh[index]) if np.isfinite(rh[index]) else None,
            wind_kts=float(wind[index]) if np.isfinite(wind[index]) else None,
        )
        value = adjust_category(raw[index], suppression)
        if value is not None:
            adjusted[index] = value
            reductions[index] = suppression["reduction"]
    return adjusted, reductions


def combine_category_grids(*grids: np.ndarray | None) -> np.ndarray:
    """Combine available categorical grids using a rounded mean."""
    available = [
        np.asarray(grid, dtype=float)
        for grid in grids
        if grid is not None
    ]
    if not available:
        return np.array([], dtype=float)
    shape = available[0].shape
    if any(grid.shape != shape for grid in available):
        raise ValueError("verification grids must have identical shapes")
    stack = np.stack([np.where(np.isfinite(grid), grid, np.nan) for grid in available])
    return np.nanmean(stack, axis=0).round()


def provider_precedence(
    *,
    mrms_mm: float | None = None,
    rtma_mm: float | None = None,
    station_mm: float | None = None,
) -> dict[str, Any]:
    """Select the best realized rainfall value and retain all provenance."""
    candidates = (
        ("mrms", mrms_mm),
        ("rtma", rtma_mm),
        ("station", station_mm),
    )
    for provider, value in candidates:
        if value is not None:
            try:
                if np.isfinite(float(value)) and float(value) >= 0:
                    return {
                        "provider": provider,
                        "rainfall_mm": float(value),
                        "candidates": {name: value for name, value in candidates},
                    }
            except (TypeError, ValueError):
                continue
    return {
        "provider": None,
        "rainfall_mm": None,
        "candidates": {name: value for name, value in candidates},
    }


def load_nlcd_raster() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a configured NLCD GeoTIFF or static NetCDF bundle.

    The API never downloads geography.  ``VERIFICATION_NLCD_RASTER`` may point
    to a GeoTIFF; ``VERIFICATION_STATIC_BUNDLE`` may point to a NetCDF bundle
    containing ``nlcd_class``, ``latitude`` and ``longitude``.
    """
    configured_raster = os.getenv("VERIFICATION_NLCD_RASTER", "").strip()
    default_raster = default_nlcd_raster_path()
    raster_path = configured_raster or (str(default_raster) if default_raster.is_file() else "")
    bundle_path = os.getenv("VERIFICATION_STATIC_BUNDLE", "").strip()
    if raster_path:
        import rasterio
        from rasterio.warp import transform

        with rasterio.open(raster_path) as src:
            values = src.read(1)
            rows, cols = np.indices(values.shape)
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            lon, lat = transform(src.crs, "EPSG:4326", np.asarray(xs).ravel(), np.asarray(ys).ravel())
        return values, np.asarray(lon).reshape(values.shape), np.asarray(lat).reshape(values.shape), {
            "source": raster_path, "format": "geotiff"
        }
    if bundle_path:
        import xarray as xr

        with xr.open_dataset(bundle_path) as ds:
            required = {"nlcd_class", "latitude", "longitude"}
            missing = required - set(ds.variables)
            if missing:
                raise ValueError(f"NLCD bundle missing {sorted(missing)}")
            return (
                np.asarray(ds["nlcd_class"].values),
                np.asarray(ds["longitude"].values),
                np.asarray(ds["latitude"].values),
                {"source": bundle_path, "format": "netcdf"},
            )
    raise FileNotFoundError("VERIFICATION_NLCD_RASTER or VERIFICATION_STATIC_BUNDLE is not configured")


def load_mrms_grid(valid_time: datetime) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load an administrator-provided MRMS accumulation grid.

    Files must be under ``VERIFICATION_MRMS_ROOT`` and contain either a
    NetCDF variable named ``precipitation``, ``precip_mm``, or ``apcp`` plus
    latitude/longitude coordinates, or be a GeoTIFF whose band is millimetres.
    This intentionally does not fetch or infer MRMS data.
    """
    root = os.getenv("VERIFICATION_MRMS_ROOT", "").strip()
    if not root:
        raise FileNotFoundError("VERIFICATION_MRMS_ROOT is not configured")
    date_token = valid_time.strftime("%Y%m%d")
    hour_token = valid_time.strftime("%Y%m%d_%H")
    candidates = sorted(Path(root).glob(f"**/mrms_{hour_token}z.nc"))
    if not candidates:
        candidates = sorted(Path(root).glob(f"**/*{date_token}*"))
    candidates = [path for path in candidates if path.suffix.lower() in {".nc", ".nc4", ".tif", ".tiff"}]
    if not candidates:
        raise FileNotFoundError(f"no MRMS file found for {date_token}")
    path = candidates[-1]
    if path.suffix.lower() in {".nc", ".nc4"}:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            variable = next(
                (name for name in ("precipitation", "precip_mm", "apcp") if name in ds),
                None,
            )
            if variable is None:
                raise ValueError(f"MRMS file missing precipitation variable: {path}")
            lat_name = next((name for name in ("latitude", "lat") if name in ds), None)
            lon_name = next((name for name in ("longitude", "lon") if name in ds), None)
            if not lat_name or not lon_name:
                raise ValueError(f"MRMS file missing latitude/longitude: {path}")
            return (
                np.asarray(ds[variable].values).squeeze(),
                np.asarray(ds[lon_name].values),
                np.asarray(ds[lat_name].values),
                {"source": str(path), "format": "netcdf", "provider": "mrms"},
            )
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(path) as src:
        values = src.read(1)
        rows, cols = np.indices(values.shape)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lon, lat = transform(src.crs, "EPSG:4326", np.asarray(xs).ravel(), np.asarray(ys).ravel())
    return values, np.asarray(lon).reshape(values.shape), np.asarray(lat).reshape(values.shape), {
        "source": str(path), "format": "geotiff", "provider": "mrms",
    }


def diagnostics() -> dict[str, Any]:
    """Return configuration state without raising during API health checks."""
    default_raster = default_nlcd_raster_path()
    configured = bool(
        os.getenv("VERIFICATION_NLCD_RASTER", "").strip()
        or os.getenv("VERIFICATION_STATIC_BUNDLE", "").strip()
        or default_raster.is_file()
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "nlcd_configured": configured,
        "nlcd_default_path": str(default_raster),
        "mrms_configured": bool(os.getenv("VERIFICATION_MRMS_ROOT", "").strip()),
        "provider_precedence": ["mrms", "rtma", "station"],
    }

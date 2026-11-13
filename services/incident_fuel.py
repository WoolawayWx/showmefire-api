"""Fuel and cover around an incident, computed from our own static geography.

For any point (an incident's centre) this reads the published fire-behavior static bundle that the spread-rate maps
already use (services/fire_behavior_static.py: LANDFIRE FBFM40 fuel models plus canopy cover on a statewide grid) and
summarises the cells within a small radius: the share of each fuel group, the most common fuel model, and mean canopy
cover. It is computed when an incident is read, not stored, so it covers every incident (old ones too) and does not
depend on a detection feed supplying land cover or on raster files being present at ingest time.

Fuel groups are the Scott & Burgan 40 fuel-model families; the 9x codes are the non-burnable classes (developed,
cropland, water, bare ground), which double as a coarse land-cover split. Returns None when the bundle is unavailable
or the point is off the grid; callers then fall back to whatever the stored detections carried.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_KM = 1.5
SOURCE = "LANDFIRE fuel models (Show Me Fire static bundle)"

# (first code, last code, label) per Scott & Burgan FBFM40 family.
_RANGES = (
    (91, 91, "Developed"), (92, 92, "Snow and ice"), (93, 93, "Cropland"), (98, 98, "Water"), (99, 99, "Bare ground"),
    (101, 109, "Grass"), (121, 124, "Grass and shrub"), (141, 149, "Shrub"),
    (161, 165, "Timber with understory"), (181, 189, "Timber litter"), (201, 204, "Slash and blowdown"),
)
OTHER = "Other"


def fuel_group(code: int) -> str:
    for low, high, label in _RANGES:
        if low <= code <= high:
            return label
    return OTHER


def _nearest_cell(lat: np.ndarray, lon: np.ndarray, latitude: float, longitude: float) -> Optional[tuple[int, int]]:
    scale = math.cos(math.radians(latitude))
    distance = (lat - latitude) ** 2 + ((lon - longitude) * scale) ** 2
    row, col = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
    # Farther than ~0.1 degrees from the nearest cell means the point is off the grid, not just between cells.
    if float(distance[row, col]) > 0.1 ** 2:
        return None
    return int(row), int(col)


def _compute(latitude: float, longitude: float, radius_km: float) -> Optional[dict[str, Any]]:
    from services.fire_behavior_static import load_static_fields

    fields = load_static_fields()
    lat, lon = fields["lat"], fields["lon"]
    cell = _nearest_cell(lat, lon, latitude, longitude)
    if cell is None:
        return None
    row, col = cell
    # Cell size from the neighbouring cells, so the window works whatever resolution the bundle was built at.
    spacing_km = max(0.05, 111.0 * float(np.nanmean([
        abs(lat[min(row + 1, lat.shape[0] - 1), col] - lat[max(row - 1, 0), col]) / 2.0 or 0.0,
        abs(lon[row, min(col + 1, lon.shape[1] - 1)] - lon[row, max(col - 1, 0)]) / 2.0 * math.cos(math.radians(latitude)) or 0.0,
    ])))
    reach = int(min(60, max(1, math.ceil(radius_km / spacing_km))))
    rows = slice(max(0, row - reach), row + reach + 1)
    cols = slice(max(0, col - reach), col + reach + 1)
    km_y = (lat[rows, cols] - latitude) * 111.0
    km_x = (lon[rows, cols] - longitude) * 111.0 * math.cos(math.radians(latitude))
    inside = (km_x ** 2 + km_y ** 2 <= radius_km ** 2) & fields["valid_mask"][rows, cols]
    if not inside.any():
        inside = np.zeros_like(inside)
        inside[min(reach, row), min(reach, col)] = bool(fields["valid_mask"][row, col])
    codes = fields["fuel_model_code"][rows, cols][inside]
    if codes.size == 0:
        return None

    counts: dict[str, int] = {}
    for code in codes:
        label = fuel_group(int(code))
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    groups = [{"group": label, "pct": round(100.0 * n / total)} for label, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    groups = [g for g in groups if g["pct"] > 0]
    values, frequency = np.unique(codes, return_counts=True)
    canopy = fields["canopy_cover_pct"][rows, cols][inside]
    canopy = canopy[np.isfinite(canopy)]
    return {
        "fuel_model_fbfm40": int(values[int(np.argmax(frequency))]),
        "canopy_cover_pct": round(float(canopy.mean()), 1) if canopy.size else None,
        "groups": groups,
        "radius_km": radius_km,
        "cells": int(total),
        "source": SOURCE,
        "bundle_version": fields.get("bundle_version"),
    }


@lru_cache(maxsize=2048)
def _cached(latitude: float, longitude: float, radius_km: float, bundle_version: Optional[str]) -> Optional[dict[str, Any]]:
    return _compute(latitude, longitude, radius_km)


def fuel_context(latitude: Any, longitude: Any, radius_km: float = DEFAULT_RADIUS_KM) -> Optional[dict[str, Any]]:
    """Fuel mix, dominant fuel model and canopy cover within `radius_km` of a point, or None. Never raises."""
    try:
        lat, lon = float(latitude), float(longitude)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        from services.fire_behavior_static import diagnostics

        # Rounded to ~100 m so every request for the same incident hits the cache; the bundle version is part of the
        # key so a newly promoted bundle is never answered from the old one.
        result = _cached(round(lat, 3), round(lon, 3), radius_km, diagnostics().get("bundle_version"))
        return dict(result) if result else None
    except Exception:  # noqa: BLE001 - enrichment only; the incident still renders without it
        logger.info("incident fuel context unavailable", exc_info=True)
        return None

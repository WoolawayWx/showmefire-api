"""Load versioned fire-behavior static geography for Rothermel spread-rate products."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

_RUNTIME = None
_DIAGNOSTICS = {"available": False, "fallback_reason": "not initialized", "bundle_version": None}


def diagnostics() -> dict:
    return dict(_DIAGNOSTICS)


def _root() -> Path:
    return Path("/app") if Path("/app").exists() else Path(__file__).resolve().parent.parent


def _env_bundle_path() -> Path | None:
    override = os.getenv("FIRE_BEHAVIOR_STATIC_BUNDLE")
    if not override:
        return None
    path = Path(override)
    return path if path.is_file() else None


def _load_from_registry():
    from models.versioning import load_active_assets

    assets = load_active_assets("fire_behavior_static", "stable")
    bundle_path = assets["static_bundle"]["path"]
    manifest_path = assets.get("static_manifest", {}).get("path") or bundle_path.with_suffix(".json")
    return bundle_path, manifest_path, assets


@lru_cache(maxsize=1)
def _open_bundle(path_str: str, mtime: float):
    with xr.open_dataset(path_str) as ds:
        return ds.load()


def load_static_fields():
    """Return dict with numpy arrays and metadata needed for spread-rate inference."""
    global _RUNTIME
    bundle_path = manifest_path = None
    bundle_version = None
    try:
        env_path = _env_bundle_path()
        if env_path is not None:
            bundle_path = env_path
            manifest_path = env_path.with_suffix(".json")
            bundle_version = "env-override"
        else:
            bundle_path, manifest_path, assets = _load_from_registry()
            bundle_version = json.loads(manifest_path.read_text()).get("bundle_version")
        mtime = bundle_path.stat().st_mtime
        signature = (str(bundle_path), mtime)
        if _RUNTIME and _RUNTIME.get("signature") == signature:
            return _RUNTIME
        ds = _open_bundle(str(bundle_path), mtime)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("sha256"):
                from hashlib import sha256

                digest = sha256()
                with open(bundle_path, "rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != manifest["sha256"]:
                    raise ValueError("fire behavior static bundle checksum mismatch")
        runtime = {
            "signature": signature,
            "bundle_path": str(bundle_path),
            "bundle_version": bundle_version or ds.attrs.get("bundle_version"),
            "x": np.asarray(ds.x.values, dtype=float),
            "y": np.asarray(ds.y.values, dtype=float),
            "lat": np.asarray(ds.latitude.values, dtype=float),
            "lon": np.asarray(ds.longitude.values, dtype=float),
            "slope_deg": np.asarray(ds.slope_degrees.values, dtype=float),
            "aspect_sin": np.asarray(ds.aspect_sin.values, dtype=float),
            "aspect_cos": np.asarray(ds.aspect_cos.values, dtype=float),
            "canopy_cover_pct": np.asarray(ds.canopy_cover_pct.values, dtype=float),
            "canopy_height_m": np.asarray(ds.canopy_height_m.values, dtype=float),
            "fuel_model_code": np.rint(np.asarray(ds.fuel_model_fbfm40.values, dtype=float)).astype(np.int32),
            "valid_mask": np.asarray(ds.static_valid_mask.values, dtype=float) > 0.5,
        }
        _RUNTIME = runtime
        _DIAGNOSTICS.update(
            {
                "available": True,
                "fallback_reason": None,
                "bundle_version": runtime["bundle_version"],
                "bundle_path": runtime["bundle_path"],
            }
        )
        return runtime
    except Exception as exc:
        logger.warning("Fire behavior static bundle unavailable: %s", exc)
        _DIAGNOSTICS.update({"available": False, "fallback_reason": str(exc), "bundle_version": bundle_version})
        raise

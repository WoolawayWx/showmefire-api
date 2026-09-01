"""Create a minimal synthetic fire-behavior static bundle for local dev and tests."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

GRID_SIZE = 256
SCHEMA_VERSION = 1


def _fingerprint(x, y, crs_wkt: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.asarray(x, dtype="float64").tobytes())
    digest.update(np.asarray(y, dtype="float64").tobytes())
    digest.update(crs_wkt.encode())
    digest.update(f"schema={SCHEMA_VERSION};size={GRID_SIZE}".encode())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output_dir: Path, version: str = "synthetic-v1") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-1.0e6, -8.0e5, GRID_SIZE)
    y = np.linspace(3.8e6, 4.0e6, GRID_SIZE)
    xx, yy = np.meshgrid(x, y)
    lat = np.linspace(35.8, 40.8, GRID_SIZE)[:, None] * np.ones(GRID_SIZE)
    lon = np.linspace(-95.8, -89.1, GRID_SIZE)[None, :] * np.ones((GRID_SIZE, 1))
    slope = np.clip(5.0 + 10.0 * np.sin(xx / 2.0e5), 0.0, 45.0)
    aspect = np.arctan2(np.sin(yy / 2.0e5), np.cos(xx / 2.0e5))
    fuel = np.full((GRID_SIZE, GRID_SIZE), 101, dtype=np.int32)
    fuel[0:20, :] = 91
    valid = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    crs = 'LOCAL_CS["synthetic"]'
    fingerprint = _fingerprint(x, y, crs)
    ds = xr.Dataset(
        {
            "elevation_m": (("y", "x"), np.full((GRID_SIZE, GRID_SIZE), 250.0, dtype="float32")),
            "slope_degrees": (("y", "x"), slope.astype("float32")),
            "aspect_sin": (("y", "x"), np.sin(aspect).astype("float32")),
            "aspect_cos": (("y", "x"), np.cos(aspect).astype("float32")),
            "canopy_cover_pct": (("y", "x"), np.full((GRID_SIZE, GRID_SIZE), 20.0, dtype="float32")),
            "canopy_height_m": (("y", "x"), np.full((GRID_SIZE, GRID_SIZE), 8.0, dtype="float32")),
            "latitude": (("y", "x"), lat.astype("float32")),
            "longitude": (("y", "x"), lon.astype("float32")),
            "static_valid_mask": (("y", "x"), valid.astype("float32")),
            "fuel_model_fbfm40": (("y", "x"), fuel.astype("float32")),
        },
        coords={"x": x, "y": y},
        attrs={
            "schema_version": SCHEMA_VERSION,
            "bundle_version": version,
            "grid_fingerprint": fingerprint,
            "crs_wkt": crs,
            "transform": tuple(np.eye(6).ravel()),
            "bbox": (-96.8, 34.8, -88.1, 41.8),
        },
    )
    bundle = output_dir / f"fire_behavior_static_{version}.nc"
    ds.to_netcdf(bundle, engine="netcdf4")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_version": version,
        "sha256": _sha256(bundle),
        "grid_fingerprint": fingerprint,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True,
    }
    bundle.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/fire_behavior_static"))
    parser.add_argument("--version", default="synthetic-v1")
    args = parser.parse_args()
    path = build(args.output_dir, args.version)
    print(path)

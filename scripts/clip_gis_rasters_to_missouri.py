"""One-off: clip already-published GIS rasters to the Missouri boundary.

Rasters published before ``regrid_lonlat`` learned to trim to the state fill
the whole bounding rectangle (nearest-neighbour smear).  Values inside
Missouri are correct, so this masks everything outside the state boundary to
nodata in place, using the publisher's own raster writer.

    python -m scripts.clip_gis_rasters_to_missouri            # dry run
    python -m scripts.clip_gis_rasters_to_missouri --apply    # rewrite latest/*.tif

Originals are copied to ``<publish dir>/latest_preclip_backup/`` first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

from services import gis_publisher


def clip_raster(path: Path, *, apply: bool, backup_dir: Path) -> str:
    grid = gis_publisher.canonical_grid()
    inside = gis_publisher._state_mask(
        tuple(grid["bounds"]), grid["resolution"], grid["width"], grid["height"]
    )
    if inside is None:
        raise RuntimeError(f"Missouri boundary unavailable: {gis_publisher.STATE_BOUNDARY_SHP}")
    with rasterio.open(path) as src:
        if src.crs.to_string() != grid["crs"] or (src.height, src.width) != inside.shape:
            return "skipped (not on the canonical grid)"
        categorical = src.dtypes[0] == "uint8"
        data = src.read(1).astype("float64")
        valid = data != src.nodata
        tags = {key.lower(): value for key, value in src.tags().items()}
    clipped = np.where(valid & inside, data, np.nan)
    removed = int((valid & ~inside).sum())
    if not apply:
        return f"would blank {removed} of {int(valid.sum())} valid pixels"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)
    gis_publisher._write_raster(path, clipped, categorical=categorical, tags=tags)
    return f"blanked {removed} of {int(valid.sum())} valid pixels"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="rewrite the files (default is a dry run)")
    parser.add_argument("--root", type=Path, default=gis_publisher.PUBLISH_ROOT, help="GIS publish directory")
    args = parser.parse_args()
    latest = args.root / "latest"
    rasters = sorted(latest.glob("*.tif"))
    if not rasters:
        print(f"No GeoTIFFs in {latest}")
        return 1
    for path in rasters:
        print(f"{path.name}: {clip_raster(path, apply=args.apply, backup_dir=args.root / 'latest_preclip_backup')}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to rewrite the files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

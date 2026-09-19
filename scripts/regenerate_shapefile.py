"""
Regenerate today's peak fire danger shapefile bundle on demand.

Rebuilds the styled shapefile zip by directly vectorizing the already-
published, operational forecast GeoTIFF (gis/latest/forecast_peak_fire_danger.tif
by default - the same file the site's map and "Day 1 fire danger raster"
download use) instead of re-running the forecast pipeline. This guarantees
the shapefile matches the production 12Z forecast maps pixel-for-pixel.

Useful when the shapefile export was added/changed after the day's forecast
already ran, or to repair a corrupted zip.

Usage:
    python scripts/regenerate_shapefile.py
    python scripts/regenerate_shapefile.py --tif gis/peak_fire_danger_09z.tif --out gis/peak_fire_danger_shapefile_09z.zip
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "forecast"))

from core.config import GIS_DIR
from export_fire_danger_gis import export_shapefile_from_raster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tif", type=Path, default=GIS_DIR / "latest" / "forecast_peak_fire_danger.tif",
        help="Source danger-level GeoTIFF (default: the live operational Day-1 forecast raster)",
    )
    parser.add_argument(
        "--out", type=Path, default=GIS_DIR / "peak_fire_danger_shapefile.zip",
        help="Output shapefile zip path",
    )
    args = parser.parse_args()

    if not args.tif.exists():
        print(f"Source GeoTIFF not found: {args.tif}", file=sys.stderr)
        sys.exit(1)

    ok = export_shapefile_from_raster(args.tif, args.out)
    if not ok:
        print("Shapefile regeneration failed -- see logs above.", file=sys.stderr)
        sys.exit(1)

    print(f"Shapefile regenerated -> {args.out}")


if __name__ == "__main__":
    main()

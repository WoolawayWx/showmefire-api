"""
Regenerate today's per-county peak fire danger shapefile bundle on demand.

One polygon per Missouri county, colored by PerCounty.py's area-threshold
classification (the same logic behind dangerbycounty.json and the public
county map) - not the finer-grained dissolved-raster regions produced by
scripts/regenerate_shapefile.py. Sources from the operational HRRR raster
(gis/peak_fire_danger.tif by default, the same file PerCounty.py itself
reads) instead of re-running the forecast pipeline.

Usage:
    python scripts/regenerate_county_shapefile.py
    python scripts/regenerate_county_shapefile.py --tif gis/peak_fire_danger_09z.tif --out gis/peak_fire_danger_by_county_09z.zip
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "forecast"))

from core.config import GIS_DIR
from PerCounty import COUNTY_SHAPEFILE, FIRE_DANGER_TIF, export_county_shapefile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tif", type=Path, default=Path(FIRE_DANGER_TIF),
        help="Source danger-level GeoTIFF (default: the operational HRRR raster PerCounty.py reads)",
    )
    parser.add_argument(
        "--out", type=Path, default=GIS_DIR / "peak_fire_danger_by_county.zip",
        help="Output shapefile zip path",
    )
    args = parser.parse_args()

    if not args.tif.exists():
        print(f"Source GeoTIFF not found: {args.tif}", file=sys.stderr)
        sys.exit(1)

    ok = export_county_shapefile(args.out, tif_path=str(args.tif), county_shapefile=COUNTY_SHAPEFILE)
    if not ok:
        print("County shapefile regeneration failed -- see logs above.", file=sys.stderr)
        sys.exit(1)

    print(f"County shapefile regenerated -> {args.out}")


if __name__ == "__main__":
    main()

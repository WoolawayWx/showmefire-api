"""
Regenerate today's peak fire danger shapefile bundle on demand.

Rebuilds the styled shapefile zip from the already-published
peak_fire_danger_polygons.geojson instead of re-running the forecast
pipeline. Useful when the shapefile export was added/changed after the
day's forecast already ran, or to repair a corrupted zip.

Usage:
    python scripts/regenerate_shapefile.py
    python scripts/regenerate_shapefile.py --geojson gis/peak_fire_danger_polygons_09z.geojson --out gis/peak_fire_danger_shapefile_09z.zip
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "forecast"))

from core.config import GIS_DIR
from export_fire_danger_gis import export_shapefile_from_geojson


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geojson", type=Path, default=GIS_DIR / "peak_fire_danger_polygons.geojson",
        help="Source polygons GeoJSON (default: today's published polygons)",
    )
    parser.add_argument(
        "--out", type=Path, default=GIS_DIR / "peak_fire_danger_shapefile.zip",
        help="Output shapefile zip path",
    )
    args = parser.parse_args()

    if not args.geojson.exists():
        print(f"Source GeoJSON not found: {args.geojson}", file=sys.stderr)
        sys.exit(1)

    ok = export_shapefile_from_geojson(args.geojson, args.out)
    if not ok:
        print("Shapefile regeneration failed — see logs above.", file=sys.stderr)
        sys.exit(1)

    print(f"Shapefile regenerated -> {args.out}")


if __name__ == "__main__":
    main()

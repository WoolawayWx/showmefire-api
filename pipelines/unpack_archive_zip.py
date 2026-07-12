"""Unpack per-day archive_zips bundles into the locations the training
pipeline already expects.

The server bundles each day's HRRR grids, raw station observations, and
station forecasts into a single zip under data/archive_zips/{date}.zip.
This routes each entry by filename pattern into:

    *.nc                     -> cache/hrrr/
    raw_data_*.json          -> archive/raw_data/
    station_forecasts_*.json -> archive/forecasts/

Anything that doesn't match a known pattern is logged, never silently
dropped. Safe to re-run - files already present at the destination (same
name and size) are skipped.

Usage:
    python pipelines/unpack_archive_zip.py                # all zips in data/archive_zips
    python pipelines/unpack_archive_zip.py --zip 20260129.zip
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

API_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCHIVE_ZIPS_DIR = API_DIR / "data" / "archive_zips"
DESTINATIONS = [
    (lambda name: name.endswith(".nc"), API_DIR / "cache" / "hrrr"),
    (lambda name: name.startswith("raw_data_") and name.endswith(".json"), API_DIR / "archive" / "raw_data"),
    (lambda name: name.startswith("station_forecasts_") and name.endswith(".json"), API_DIR / "archive" / "forecasts"),
]


def _destination_for(entry_name):
    basename = Path(entry_name).name
    for matches, dest_dir in DESTINATIONS:
        if matches(basename):
            return dest_dir, basename
    return None, basename


def unpack_zip(zip_path):
    print(f"Unpacking {zip_path.name}...")
    unrecognized = []
    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            dest_dir, basename = _destination_for(info.filename)
            if dest_dir is None:
                unrecognized.append(info.filename)
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / basename

            if dest_path.exists() and dest_path.stat().st_size == info.file_size:
                skipped += 1
                continue

            with zf.open(info) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

    print(f"  extracted {extracted}, skipped {skipped} (already present)")
    if unrecognized:
        print(f"  WARNING: {len(unrecognized)} unrecognized entries not extracted:")
        for name in unrecognized:
            print(f"    - {name}")

    return extracted, skipped, unrecognized


def main():
    parser = argparse.ArgumentParser(description="Unpack archive_zips bundles into the pipeline's expected data locations")
    parser.add_argument("--zip", default=None, help="Unpack a single zip by filename instead of all of data/archive_zips")
    args = parser.parse_args()

    if args.zip:
        zip_paths = [ARCHIVE_ZIPS_DIR / args.zip]
    else:
        zip_paths = sorted(ARCHIVE_ZIPS_DIR.glob("*.zip"))

    if not zip_paths:
        print(f"No zip files found in {ARCHIVE_ZIPS_DIR}")
        return

    total_unrecognized = []
    for zip_path in zip_paths:
        if not zip_path.exists():
            print(f"SKIP: {zip_path} does not exist")
            continue
        try:
            _, _, unrecognized = unpack_zip(zip_path)
            total_unrecognized.extend(unrecognized)
        except zipfile.BadZipFile as e:
            print(f"SKIP: {zip_path.name} is not a valid zip yet ({e}) - server may still be writing it")

    if total_unrecognized:
        print(f"\n{len(total_unrecognized)} total unrecognized entries across all zips - review DESTINATIONS in this script.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Download the daily map-archive zips to a local folder (run it on your own machine).

The server zips each day's 15-minute CDN map copies (services/map_archive.py)
and keeps each zip on the CDN for a limited time; Discord says when new ones
are ready. This pulls every zip that is not already in the folder, checking its
size, and skips ones you already have. Needs only Python 3 and `requests`.

    python pull_map_archives.py D:/ShowMeFire/map-archive
    python pull_map_archives.py ~/map-archive --api https://api.example.org
"""
import argparse
import sys
from pathlib import Path

import requests

DEFAULT_API = "https://api.showmefire.org"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="where to keep the zips")
    parser.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default {DEFAULT_API})")
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)

    listing = requests.get(f"{args.api.rstrip('/')}/api/map-archives", timeout=60)
    listing.raise_for_status()
    archives = listing.json()["archives"]
    fetched = skipped = failed = 0
    for archive in sorted(archives, key=lambda item: item["day"]):
        target = args.folder / f"{archive['day']}.zip"
        if target.is_file() and target.stat().st_size == archive["zip_bytes"]:
            skipped += 1
            continue
        partial = target.with_suffix(".zip.part")
        print(f"{archive['day']}: downloading {archive['zip_bytes'] / 1e9:.2f} GB ...", flush=True)
        try:
            with requests.get(archive["url"], stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(partial, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        handle.write(chunk)
            if partial.stat().st_size != archive["zip_bytes"]:
                raise RuntimeError(f"size {partial.stat().st_size} != expected {archive['zip_bytes']}")
            partial.replace(target)
            fetched += 1
        except Exception as exc:  # noqa: BLE001 - keep going with the other days
            print(f"{archive['day']}: FAILED ({exc})", file=sys.stderr)
            partial.unlink(missing_ok=True)
            failed += 1
    print(f"Done: {fetched} downloaded, {skipped} already here, {failed} failed ({len(archives)} available).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Refresh the live station feeds right before maps.sh plots.

The API's scheduler polls Synoptic every 5 minutes, so at plot time the in-memory station list can be
up to 5 minutes old (plus the 2-minute disk cache the maps read through). This asks the API for fresh
Missouri and neighbor-state (buffer) observations, then drops the disk cache so the first map script
re-reads the new list. Best effort: any failure is logged and the maps just plot the data they already
have, so it always exits 0.
"""
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
# Keep in step with STATION_CACHE_PATH in maps/station_plot.py (not imported: that pulls in cartopy).
STATION_CACHE_PATH = PROJECT_DIR / "data" / "current_conditions" / "station_cache.json"
TIMEOUT_SECONDS = 60


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    base = f"http://localhost:{os.getenv('PORT', '8000')}"
    refreshed = False
    for label, path, params in (("stations", "/stations/refresh", None),
                                ("buffer stations", "/stations/buffer", {"refresh": "true"})):
        try:
            response = requests.get(base + path, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            print(f"Refreshed {label}")
            refreshed = refreshed or label == "stations"
        except requests.RequestException as error:
            print(f"Could not refresh {label}: {error}")
    if refreshed:
        try:
            STATION_CACHE_PATH.unlink(missing_ok=True)
        except OSError as error:
            print(f"Could not clear station cache: {error}")


if __name__ == "__main__":
    main()
    sys.exit(0)

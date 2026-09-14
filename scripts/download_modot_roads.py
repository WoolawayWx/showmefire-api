"""Download the statewide MoDOT road archive outside of Git.

The source is the MSDIS ArcGIS Hub item. The archive is cached and only
downloaded when the required shapefile is missing or ROAD_DATA_REFRESH=1.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ITEM_ID = "c309725bacdf47079b2a45612e0a3e55"
DEFAULT_URL = f"https://hub.arcgis.com/api/download/v1/items/{ITEM_ID}/shapefile?layers=0"
TARGET_ROOT = Path(os.getenv("SMF_ROADS_DIR", "/app/data/roads"))
TARGET_DIR = TARGET_ROOT / "MO_MoDOT_Roads_Arcs"
REQUIRED = TARGET_DIR / "MO_MoDOT_Roads_Arcs.shp"


def main() -> int:
    if REQUIRED.is_file() and os.getenv("ROAD_DATA_REFRESH") != "1":
        print(f"MoDOT roads already present: {REQUIRED}")
        return 0

    url = os.getenv("SMF_MODOT_ROADS_URL", DEFAULT_URL)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="modot-roads-") as temp_dir:
        archive = Path(temp_dir) / "modot-roads.zip"
        print(f"Downloading MoDOT roads from {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "ShowMeFire/1.0"})
        with urllib.request.urlopen(request, timeout=300) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)

        expected_sha = os.getenv("SMF_MODOT_ROADS_SHA256", "").strip().lower()
        if expected_sha:
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != expected_sha:
                raise RuntimeError(f"MoDOT archive checksum mismatch: {digest}")

        staging = Path(temp_dir) / "extract"
        staging.mkdir()
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                destination = (staging / member.filename).resolve()
                if not str(destination).startswith(str(staging.resolve()) + os.sep):
                    raise RuntimeError(f"Unsafe archive member: {member.filename}")
            zipped.extractall(staging)

        shapefile = next(staging.rglob("MO_MoDOT_Roads_Arcs.shp"), None)
        if shapefile is None:
            raise RuntimeError("Downloaded archive does not contain MO_MoDOT_Roads_Arcs.shp")
        extracted_dir = shapefile.parent
        replacement = TARGET_ROOT / "MO_MoDOT_Roads_Arcs.new"
        if replacement.exists():
            shutil.rmtree(replacement)
        shutil.copytree(extracted_dir, replacement)
        if TARGET_DIR.exists():
            shutil.rmtree(TARGET_DIR)
        replacement.rename(TARGET_DIR)

    print(f"MoDOT roads installed at {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

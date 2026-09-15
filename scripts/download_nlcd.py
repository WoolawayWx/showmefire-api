"""Infrequently acquire and clip an official NLCD land-cover raster.

This is an explicit operator command, not an API startup task:

    python scripts/download_nlcd.py --year 2023 --output data/static/nlcd_class.tif

Use ``--url`` when USGS changes the collection/version path.  The default is
the documented USGS Annual NLCD Collection 1.0 CONUS mosaic URL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import rasterio
import requests
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

BBOX = (-96.8, 34.8, -88.1, 41.8)  # west, south, east, north
DEFAULT_URL = (
    "https://usgs-landcover.s3.us-west-2.amazonaws.com/"
    "annual-nlcd/c1/v0/cu/mosaic/Annual_NLCD_Land_Cover_{year}_CU_C1V0.tif"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=300) as response:
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with partial.open(mode) as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    partial.replace(target)


def materialize(path: Path, directory: Path) -> Path:
    if path.suffix.lower() not in {".zip", ".gz"}:
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError("gzip NLCD input is unsupported; provide a GeoTIFF or ZIP")
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if Path(name).suffix.lower() in {".tif", ".tiff"}]
        if len(names) != 1:
            raise ValueError(f"expected one GeoTIFF in NLCD ZIP, found {len(names)}")
        output = directory / Path(names[0]).name
        with archive.open(names[0]) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)
        return output


def clip_nlcd(source: Path, output: Path) -> dict:
    with rasterio.open(source) as src:
        if src.count != 1:
            raise ValueError("NLCD source must contain exactly one band")
        if src.crs is None:
            raise ValueError("NLCD source has no CRS")
        bounds = transform_bounds("EPSG:4326", src.crs, BBOX[0], BBOX[1], BBOX[2], BBOX[3])
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window)
        if data.size == 0:
            raise ValueError("NLCD source does not intersect the configured Missouri region")
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            count=1,
            dtype=data.dtype,
            transform=transform,
            compress="lzw",
        )
        if data.shape[1] >= 16 and data.shape[0] >= 16:
            profile.update(
                tiled=True,
                blockxsize=min(256, max(16, (data.shape[1] // 16) * 16)),
                blockysize=min(256, max(16, (data.shape[0] // 16) * 16)),
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, "NLCD land-cover class")
            dst.update_tags(SOURCE="USGS Annual NLCD / MRLC", BBOX=",".join(map(str, BBOX)))
        return {
            "crs": src.crs.to_string(),
            "width": int(data.shape[1]),
            "height": int(data.shape[0]),
            "nodata": src.nodata,
            "bounds": list(rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and clip official Annual NLCD land cover for Missouri.")
    parser.add_argument("--year", default=os.getenv("VERIFICATION_NLCD_YEAR", "2023"))
    parser.add_argument("--url", default=os.getenv("VERIFICATION_NLCD_URL"))
    parser.add_argument("--output", type=Path, default=Path("data/static/nlcd_class.tif"))
    parser.add_argument("--release", default="Annual NLCD Collection 1.0")
    args = parser.parse_args()
    url = args.url or DEFAULT_URL.format(year=args.year)

    with tempfile.TemporaryDirectory(prefix="nlcd-") as temp:
        source_name = Path(url.split("?")[0]).name or "nlcd_source.tif"
        downloaded = Path(temp) / source_name
        download(url, downloaded)
        source = materialize(downloaded, Path(temp))
        raster_metadata = clip_nlcd(source, args.output)

    manifest = {
        "product": "nlcd_class",
        "source_url": url,
        "source_release": args.release,
        "year": str(args.year),
        "bbox": BBOX,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "path": str(args.output.resolve()),
        "sha256": sha256(args.output),
        "size": args.output.stat().st_size,
        "raster": raster_metadata,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

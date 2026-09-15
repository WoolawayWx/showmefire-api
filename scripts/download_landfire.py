"""Infrequently acquire an official LANDFIRE FBFM40 fuel model raster.

This is an explicit operator command, not an API startup task:

    python scripts/download_landfire.py --output data/static/fbfm40_class.tif

USGS ScienceBase's bulk NLCD download flow is captcha-gated and currently
returns a server-side 500 for this region's archives. LFPS (the LANDFIRE
Product Service) instead submits an async clip-to-AOI job over plain HTTP
and hands back a direct, unauthenticated download link once it finishes —
no browser, captcha, or AWS credentials required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

BBOX = (-96.8, 34.8, -88.1, 41.8)  # west, south, east, north
LFPS_SUBMIT_URL = "https://lfps.usgs.gov/api/job/submit"
LFPS_STATUS_URL = "https://lfps.usgs.gov/api/job/status"
DEFAULT_LAYER = "LF2023_FBFM40"
DEFAULT_EMAIL = "showmefire-api@example.invalid"  # RFC 2606 reserved TLD; LFPS only format-validates this field


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def submit_job(layer: str, bbox: tuple[float, float, float, float], email: str) -> str:
    aoi = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
    response = requests.post(
        LFPS_SUBMIT_URL,
        json={"Layer_List": layer, "Area_of_Interest": aoi, "Email": email},
        timeout=60,
    )
    response.raise_for_status()
    job_id = response.json().get("jobId")
    if not job_id:
        raise RuntimeError(f"LFPS did not return a jobId: {response.text}")
    return job_id


def await_job(job_id: str, *, poll_seconds: float, timeout_minutes: float) -> str:
    """Poll an LFPS job until it succeeds and return the output file URL."""
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        response = requests.get(LFPS_STATUS_URL, params={"JobId": job_id}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status == "Succeeded":
            output = payload.get("outputFile")
            if not output:
                raise RuntimeError(f"LFPS job {job_id} succeeded without an outputFile: {payload}")
            return output
        if status == "Failed":
            raise RuntimeError(f"LFPS job {job_id} failed: {payload}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"LFPS job {job_id} did not finish within {timeout_minutes} minutes")
        time.sleep(poll_seconds)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)


def materialize(path: Path, directory: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if Path(name).suffix.lower() in {".tif", ".tiff"}]
        if len(names) != 1:
            raise ValueError(f"expected one GeoTIFF in LANDFIRE ZIP, found {len(names)}")
        output = directory / Path(names[0]).name
        with archive.open(names[0]) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)
        return output


def clip_fuel_model(source: Path, output: Path) -> dict:
    with rasterio.open(source) as src:
        if src.count != 1:
            raise ValueError("FBFM40 source must contain exactly one band")
        if src.crs is None:
            raise ValueError("FBFM40 source has no CRS")
        bounds = transform_bounds("EPSG:4326", src.crs, BBOX[0], BBOX[1], BBOX[2], BBOX[3])
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window)
        if data.size == 0:
            raise ValueError("FBFM40 source does not intersect the configured Missouri region")
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
            dst.set_band_description(1, "LANDFIRE FBFM40 fuel model")
            dst.update_tags(SOURCE="USGS/USFS LANDFIRE FBFM40", BBOX=",".join(map(str, BBOX)))
        return {
            "crs": src.crs.to_string(),
            "width": int(data.shape[1]),
            "height": int(data.shape[0]),
            "nodata": src.nodata,
            "bounds": list(rasterio.transform.array_bounds(data.shape[0], data.shape[1], transform)),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and clip the LANDFIRE FBFM40 fuel model raster for Missouri via LFPS."
    )
    parser.add_argument("--layer", default=os.getenv("VERIFICATION_LANDFIRE_LAYER", DEFAULT_LAYER))
    parser.add_argument("--email", default=os.getenv("VERIFICATION_LANDFIRE_EMAIL", DEFAULT_EMAIL))
    parser.add_argument("--output", type=Path, default=Path("data/static/fbfm40_class.tif"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    args = parser.parse_args()

    job_id = submit_job(args.layer, BBOX, args.email)
    output_url = await_job(job_id, poll_seconds=args.poll_seconds, timeout_minutes=args.timeout_minutes)

    with tempfile.TemporaryDirectory(prefix="landfire-") as temp:
        downloaded = Path(temp) / (Path(output_url.split("?")[0]).name or "landfire_source.zip")
        download(output_url, downloaded)
        source = materialize(downloaded, Path(temp))
        raster_metadata = clip_fuel_model(source, args.output)

    manifest = {
        "product": "fbfm40_class",
        "source_url": output_url,
        "source_release": args.layer,
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

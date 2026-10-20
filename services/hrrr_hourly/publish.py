"""Publish a built HRRR hourly cycle to the CDN (services/cdn_datasets.py).

Only the Zarr store and the manifest go up. The NetCDF copy stays off the CDN: it is large, and the R2 archive
already picks it up (services/archive_store.py, source 'hrrr_hourly'). The point time series is computed from
that NetCDF by the API, so /hrrr-hourly/point stays on the API too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from services import cdn_datasets
from services.hrrr_hourly import SCHEMA_VERSION, root_dir
from services.hrrr_hourly import store

DATASET = "hrrr-hourly"


def run_summary(manifest: dict) -> dict:
    """The cycle's row in runs.json: the same fields as the API's /hrrr-hourly/runs, plus id and time_utc."""
    return {"id": manifest["cycle"], "cycle": manifest["cycle"], "label": manifest.get("label"),
            "cycle_time_utc": manifest.get("cycle_time_utc"), "issued_at_utc": manifest.get("issued_at_utc"),
            "forecast_hours": manifest.get("forecast_hours"), "missing_hours": manifest.get("missing_hours"),
            "variables": len(manifest.get("variables", {})), "time_utc": manifest["cycle_time_utc"],
            "complete": manifest.get("complete", True), "pending_hours": manifest.get("pending_hours", []),
            "hours_expected": manifest.get("hours_expected")}


def try_publish_cycle(cycle: str, root: Optional[Path] = None, client=None) -> Optional[dict]:
    """Upload `cycle` and make it the dataset's newest. Never raises; None if nothing was published."""
    root = Path(root or root_dir())
    manifest = store.read_manifest(cycle, root)
    if manifest is None or not (root / cycle / "forecast.zarr").is_dir():
        return None
    return cdn_datasets.try_publish_run(
        DATASET, cycle, root / cycle, run_summary(manifest), title="Hourly HRRR forecast",
        kind="gridded-forecast", schema_version=SCHEMA_VERSION, include=["forecast.zarr", "manifest.json"],
        client=client)

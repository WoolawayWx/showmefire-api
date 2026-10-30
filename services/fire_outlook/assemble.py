"""
Assemble one outlook run on disk: compressed grids + an immutable manifest, published atomically.

Layout (run_dir/, loadable by services/cdn_datasets.publish_run, which needs manifest.json):
    manifest.json
    agreement_elevated.npz / agreement_critical.npz   uint8 member COUNTS (0..n_members, 255 = NoData)
    fm10_min.npz       p10/p50/p90 of window-minimum MODELED 10-h fuel moisture (float32 %)
    weather.npz        mean_hours_*, mean_precip_6h_mm, temp/rh/wind envelopes (float32)
    nodata.npz         bool mask (28, y, x)

Counts, not percentages, are stored for agreement so the 31-member denominator is exact and
percent = 100 * count / n_members is derived, never stored twice. The run directory is built under a
temporary name and renamed into place, so a reader never sees a half-written run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from services.fire_outlook import windows

SCHEMA_VERSION = 1
DATASET_ID = "fire-outlook"
LABEL = "scenario agreement"
LABEL_NOTE = ("Percentages are the share of weather scenarios (GEFS members) reaching the threshold. "
              "They reflect weather-scenario variation only, not every source of fuel or model uncertainty, "
              "and 3 km display spacing does not establish 3 km weather skill, especially at longer leads.")
NODATA_COUNT = 255

MAX_AGE_HOURS_STALE = 24


def run_id_for(cycle_time: datetime) -> str:
    return cycle_time.astimezone(timezone.utc).strftime("%Y%m%dT%HZ")


def window_table(cycle_time: datetime) -> list[dict]:
    start = cycle_time.astimezone(timezone.utc)
    rows = []
    for w, (a, b) in enumerate(windows.window_bounds()):
        rows.append({"index": w, "start_hour": a, "end_hour": b,
                     "start_utc": (start + timedelta(hours=a)).isoformat().replace("+00:00", "Z"),
                     "end_utc": (start + timedelta(hours=b)).isoformat().replace("+00:00", "Z")})
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _counts(fraction: np.ndarray, n_members: int) -> np.ndarray:
    counts = np.rint(np.nan_to_num(fraction, nan=0.0) * n_members).astype("uint8")
    return np.where(np.isnan(fraction), NODATA_COUNT, counts).astype("uint8")


def validate(result: Dict[str, np.ndarray]) -> None:
    e, c = result["agreement_elevated"], result["agreement_critical"]
    if e.shape[0] != windows.N_WINDOWS:
        raise ValueError(f"expected {windows.N_WINDOWS} windows, got {e.shape[0]}")
    if np.any(c > e + 1e-6):
        raise ValueError("Critical agreement exceeds Elevated agreement")
    if np.nanmax(e) > 1.0 + 1e-6 or np.nanmin(e) < -1e-6:
        raise ValueError("agreement outside [0, 1]")
    if not np.array_equal(np.isnan(e), np.isnan(c)):
        raise ValueError("Elevated and Critical NoData masks differ")


def write_run(out_root: Path, cycle_time: datetime, result: Dict[str, np.ndarray], *, grid: dict,
              source_cycles: dict, rule_info: dict, conditioning_quality: dict,
              coverage: Optional[dict] = None, driver_params: Optional[dict] = None,
              runtime: Optional[dict] = None, now: Optional[datetime] = None,
              extra_files: Optional[Dict[str, bytes]] = None, layers: Optional[list] = None) -> Path:
    """Write run_dir atomically and return it. Raises (and writes nothing) on an invalid result.

    extra_files: {relative/path: bytes} (e.g. contour GeoJSON); checksummed like every other file.
    layers: the map layers the contour files provide (services/fire_outlook/layers.manifest_layers()), for the viewer."""
    validate(result)
    n = int(result["n_members"])
    run_id = run_id_for(cycle_time)
    out_root = Path(out_root)
    final = out_root / run_id
    tmp = out_root / f".{run_id}.tmp"
    if final.exists():
        raise FileExistsError(f"run {run_id} already published (runs are immutable): {final}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        files = {
            "agreement_elevated.npz": {"counts": _counts(result["agreement_elevated"], n)},
            "agreement_critical.npz": {"counts": _counts(result["agreement_critical"], n)},
            "fm10_min.npz": {k: result[f"fm10_min_{k}"] for k in ("p10", "p50", "p90")},
            "weather.npz": {k: result[k] for k in (
                "mean_hours_elevated", "mean_hours_critical", "mean_precip_6h_mm",
                "temp_max_c_envelope", "rh_min_pct_envelope", "wind_max_kts_envelope",
                "temp_max_c_mean", "rh_min_pct_mean", "wind_max_kts_mean") if k in result},
            "nodata.npz": {"mask": result["nodata"]},
        }
        for name, arrays in files.items():
            np.savez_compressed(tmp / name, **arrays)
        for name, payload in (extra_files or {}).items():
            target = tmp / name
            if tmp.resolve() not in target.resolve().parents:
                raise ValueError(f"extra file escapes the run directory: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        all_names = list(files) + list(extra_files or {})
        issued = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        manifest = {
            "schema_version": SCHEMA_VERSION, "dataset": DATASET_ID, "run_id": run_id,
            "cycle_time_utc": cycle_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "published_utc": issued.isoformat().replace("+00:00", "Z"),
            "stale_after_hours": MAX_AGE_HOURS_STALE,
            "label": LABEL, "label_note": LABEL_NOTE,
            "member_count": n, "agreement_encoding": {"type": "uint8_count", "nodata": NODATA_COUNT,
                                                       "percent": "100 * count / member_count"},
            "windows": window_table(cycle_time), "window_hours": windows.WINDOW_HOURS,
            "grid": grid,
            "units": {"fm10_min": "percent (modeled)", "precip_6h": "mm", "temperature": "degC",
                      "rh": "percent", "wind": "knots", "hours": "hours per member-mean"},
            "fuel_moisture": "modeled; RAWS only anchors the initial 10-h state",
            "layers": layers or [],
            "source_cycles": source_cycles, "rules": rule_info, "driver": driver_params or {},
            "conditioning": conditioning_quality, "coverage": coverage or {},
            "nodata_cells_any_window": int(result["nodata"].any(axis=0).sum()),
            "runtime": runtime or {},
            "files": {name: {"sha256": _sha256(tmp / name), "bytes": (tmp / name).stat().st_size} for name in all_names},
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def read_agreement_percent(run_dir: Path, threshold: str = "elevated") -> np.ndarray:
    """(28, y, x) float32 percent with NaN for NoData, derived from the stored counts."""
    run_dir = Path(run_dir)
    n = json.loads((run_dir / "manifest.json").read_text())["member_count"]
    counts = np.load(run_dir / f"agreement_{threshold}.npz")["counts"]
    return np.where(counts == NODATA_COUNT, np.nan, 100.0 * counts / n).astype("float32")


def verify_checksums(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return all(_sha256(run_dir / name) == meta["sha256"] for name, meta in manifest["files"].items())


def is_stale(manifest: dict, now: Optional[datetime] = None) -> bool:
    cycle = datetime.fromisoformat(manifest["cycle_time_utc"].replace("Z", "+00:00"))
    return ((now or datetime.now(timezone.utc)) - cycle) > timedelta(hours=manifest.get("stale_after_hours", MAX_AGE_HOURS_STALE))

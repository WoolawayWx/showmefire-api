"""Capture-only collection of every weather model we may train on: RRFS, GEFS, NBM (GRIB byte-range fetches) and MRMS.

Why this exists: the server only kept what a live product used. RRFS's own script had been dead since February, and GEFS,
NBM and MRMS were fetched only by products that are switched off - so there was no history to train on. This job keeps a
Missouri-cropped copy of each model, whether or not any product reads it yet.

Rules:
  * Files land in the SAME layout the training PC uses: `cache/<model>/<prefix>_YYYYMMDD_HHz_fNN-NN.nc`, so the nightly
    day bundle restores straight into training with no renaming.
  * A run is captured only when its LAST lead is published (no partial runs), and never twice (the file is the marker).
  * Disk guard: nothing is captured while free space under the data root is below SMF_CAPTURE_MIN_FREE_GB (default 5).
  * Files older than `keep_days` are pruned; the nightly bundle (built daily for the last 7 days) has them by then.
  * Each tick does a bounded amount of work (SMF_CAPTURE_MAX_RUNS per tick) so it can never hog the server.
Reuses services/ensemble_fire_danger/grib_idx.py (the byte-range GRIB fetcher, a contract mirror - not modified here) and
services/mrms_capture.py.

    python -m services.data_capture.capture --tick             # what the scheduler runs
    python -m services.data_capture.capture --tick --dry-run   # list what would be captured
    python -m services.data_capture.capture --model rrfs --cycle 2026111812
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).with_name("capture_config.json")
_NAME_DATE = re.compile(r"_(\d{8})_(\d{2})z")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def data_root() -> Path:
    """Same root rule as services/day_archive.root()."""
    return Path(os.getenv("SMF_ARCHIVE_ROOT") or ("/app" if Path("/app").exists() else API_ROOT))


def min_free_bytes() -> int:
    return int(float(os.getenv("SMF_CAPTURE_MIN_FREE_GB", "5")) * 1e9)


def enough_disk(root: Path) -> bool:
    try:
        return shutil.disk_usage(root).free >= min_free_bytes()
    except OSError:
        return False


def leads_for(model: dict) -> List[int]:
    spec = model["leads"]
    return list(range(int(spec["start"]), int(spec["end"]) + 1, int(spec.get("step", 1))))


def due_cycles(model: dict, now: datetime, lookback_hours: float) -> List[datetime]:
    """Cycles (newest first) that should be fully published by now: cycle + latency <= now, within the lookback window."""
    newest = now - timedelta(hours=float(model.get("latency_hours", 0)))
    oldest = now - timedelta(hours=lookback_hours)
    hours = set(int(h) for h in model["cycles"])
    cycle = newest.replace(minute=0, second=0, microsecond=0)
    out = []
    while cycle >= oldest:
        if cycle.hour in hours:
            out.append(cycle)
        cycle -= timedelta(hours=1)
    return out


def output_path(root: Path, model: dict, cycle: datetime) -> Path:
    leads = leads_for(model)
    return root / model["dir"] / f"{model['prefix']}_{cycle:%Y%m%d_%H}z_f{leads[0]:02d}-{leads[-1]:02d}.nc"


def unscan_alternate_rows(values: np.ndarray) -> np.ndarray:
    """Undo GRIB 'alternativeRowScanning' (boustrophedon): every other row is stored right-to-left, and the decoder returns
    the values in storage order while latitudes/longitudes come back in geographic order - so values land on the wrong
    cells (NBM showed -2 C at Springfield IL on a 17 C afternoon). Flipping the odd rows restores them.
    Needed only for grids with that flag; set `alternate_row_scan` on the model in capture_config.json."""
    fixed = np.array(values, copy=True)
    fixed[1::2] = fixed[1::2, ::-1]
    return fixed


def capture_run(model: dict, cycle: datetime, crop: Callable, tmp_dir: Path, session=None):
    """One model run -> cropped xarray Dataset(step, y, x): every configured field, plus `precip_rate_mmh` when APCP is
    configured (mm per hour over the latest accumulation window ending at each lead). Raises SourceUnavailable when the
    run is not (fully) published, KeyError when a required field is missing."""
    import requests
    import xarray as xr
    from services.ensemble_fire_danger import grib_idx

    searches: Dict[str, str] = model["searches"]
    leads = leads_for(model)
    own_session = session is None
    session = session or requests.Session()
    try:
        # Only whole runs: if the last lead is not out yet, the run is not ready (fetch_idx raises SourceUnavailable).
        grib_idx.fetch_idx(session, grib_idx.render_url(model["url"], cycle, leads[-1]))
        fields = {key: [] for key in searches if key != "apcp"}
        apcp_by_lead = {}
        lat = lon = None
        for lead in leads:
            url = grib_idx.render_url(model["url"], cycle, lead)
            found, lead_lat, lead_lon = grib_idx.fetch_fields(session, url, searches, tmp_dir)
            if lat is None and lead_lat is not None and lead_lat.size:
                lat, lon = lead_lat, lead_lon
            fix = unscan_alternate_rows if model.get("alternate_row_scan") else (lambda a: a)
            for key in fields:
                fields[key].append(fix(found[key][0][1]) if found.get(key) else None)
            if "apcp" in searches:
                apcp_by_lead[lead] = {window: fix(values) for window, values in grib_idx.apcp_windows(found.get("apcp", [])).items()}
    finally:
        if own_session:
            session.close()
    if lat is None:
        raise grib_idx.SourceUnavailable(f"{model['prefix']} {cycle:%Y%m%d%H}: none of the configured fields were found")
    missing = [key for key in model.get("required", []) if all(value is None for value in fields.get(key, [None]))]
    if missing:
        raise KeyError(f"{model['prefix']} {cycle:%Y%m%d%H}: required fields missing everywhere: {missing}")

    shape = lat.shape
    data_vars = {key: (("step", "y", "x"), np.stack([v if v is not None else np.full(shape, np.nan) for v in values]).astype("float32"))
                 for key, values in fields.items()}
    if "apcp" in searches:
        precip = grib_idx.hourly_precip(apcp_by_lead, leads, shape)
        data_vars["precip_rate_mmh"] = (("step", "y", "x"), np.stack([precip[f] for f in leads]).astype("float32"))
    ds = xr.Dataset(data_vars, coords={"step": ("step", np.asarray(leads, dtype="int32"), {"units": "forecast lead hours"}),
                                       "latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)})
    ds = crop(ds)
    ds.attrs.update({
        "model": model["prefix"], "label": model.get("label", ""), "source_url_template": model["url"],
        "cycle_utc": f"{cycle:%Y-%m-%dT%H:%M:%SZ}", "leads": ",".join(str(f) for f in leads),
        "captured_at_utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}", "capture_version": 1,
        "missing_fields": ",".join(sorted(k for k, v in fields.items() if all(x is None for x in v))),
    })
    return ds


def write_atomic(ds, path: Path) -> Path:
    """NetCDF with lossless zlib compression (the old caches were uncompressed float32, ~4x larger)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4, "shuffle": True} for name, var in ds.data_vars.items() if var.dtype.kind == "f"}
    temporary = path.with_suffix(".nc.part")
    ds.to_netcdf(temporary, encoding=encoding)
    os.replace(temporary, path)
    return path


def prune(folder: Path, prefix: str, keep_days: int, now: datetime) -> int:
    """Delete captured files whose cycle date is older than keep_days. Only our own prefix, only dated files."""
    if not folder.is_dir():
        return 0
    cutoff = (now - timedelta(days=keep_days)).strftime("%Y%m%d")
    removed = 0
    for path in folder.glob(f"{prefix}_*.nc"):
        if not path.name.startswith(f"{prefix}_"):     # exact, case-sensitive: Windows globbing would also match RRFS_...
            continue
        match = _NAME_DATE.search(path.name)
        if match and match.group(1) < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    for stale in folder.glob(f"{prefix}_*.nc.part"):     # an interrupted write never becomes a "captured" file
        if time.time() - stale.stat().st_mtime > 3600:
            stale.unlink(missing_ok=True)
    return removed


def _mrms_hours(now: datetime, lookback_hours: float) -> List[datetime]:
    newest = (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return [newest - timedelta(hours=h) for h in range(int(lookback_hours))]


def tick(now: Optional[datetime] = None, *, config: Optional[dict] = None, root: Optional[Path] = None,
         max_runs: Optional[int] = None, dry_run: bool = False, crop: Optional[Callable] = None,
         capture: Callable = capture_run, writer: Callable = write_atomic, mrms_fetch: Optional[Callable] = None) -> dict:
    """One scheduler tick: capture what is due and missing (bounded), prune old files. Never raises for one model's failure."""
    now = now or datetime.now(timezone.utc)
    config = config or load_config()
    root = Path(root or data_root())
    max_runs = int(max_runs if max_runs is not None else os.getenv("SMF_CAPTURE_MAX_RUNS", "3"))
    keep_days = int(config.get("keep_days", 14))
    lookback = float(config.get("lookback_hours", 30))
    summary = {"captured": [], "waiting": [], "failed": {}, "pruned": {}, "skipped": None, "would_capture": []}
    if not dry_run and not enough_disk(root):
        summary["skipped"] = f"free disk below {min_free_bytes() / 1e9:.0f} GB under {root}; nothing captured"
        logger.warning("data capture: %s", summary["skipped"])
        return summary
    if crop is None and not dry_run:
        from core.domain import crop as crop_function
        crop = crop_function

    budget = max_runs
    with tempfile.TemporaryDirectory(prefix="smf-capture-") as tmp:
        for name, model in config["models"].items():
            if model.get("enabled", True) is False:
                continue
            if model["kind"] == "mrms":
                for hour in _mrms_hours(now, lookback):
                    target = root / model["dir"] / f"mrms_{hour:%Y%m%d_%H}z.nc"
                    if target.exists():
                        continue
                    if dry_run:
                        summary["would_capture"].append(target.name)
                        continue
                    if budget <= 0:
                        break
                    try:
                        if mrms_fetch is None:
                            from services.mrms_capture import fetch_mrms
                            fetch_mrms(hour, cache_dir=target.parent)
                        else:
                            mrms_fetch(hour, target.parent)
                        summary["captured"].append(target.name)
                    except Exception as error:  # noqa: BLE001 - MRMS keeps only ~1 day; a missed hour is logged, not fatal
                        summary["failed"][target.name] = str(error)[:200]
                    budget -= 1
                if not dry_run:
                    summary["pruned"][name] = prune(root / model["dir"], "mrms", keep_days, now)
                continue
            for cycle in due_cycles(model, now, lookback):
                target = output_path(root, model, cycle)
                if target.exists():
                    continue
                if dry_run:
                    summary["would_capture"].append(target.name)
                    continue
                if budget <= 0:
                    break
                try:
                    ds = capture(model, cycle, crop, Path(tmp))
                    writer(ds, target)
                    summary["captured"].append(target.name)
                except Exception as error:  # noqa: BLE001 - one model's outage must not stop the others
                    kind = type(error).__name__
                    if kind == "SourceUnavailable":
                        summary["waiting"].append(target.name)          # not published yet, or already rolled off
                    else:
                        summary["failed"][target.name] = f"{kind}: {str(error)[:200]}"
                        logger.warning("data capture failed for %s: %s", target.name, error)
                budget -= 1
            if not dry_run:
                summary["pruned"][name] = prune(root / model["dir"], model["prefix"], keep_days, now)
    return summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tick", action="store_true", help="capture whatever is due and missing (what the scheduler runs)")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be captured")
    parser.add_argument("--model", help="capture one model run by hand (with --cycle)")
    parser.add_argument("--cycle", help="YYYYMMDDHH for --model")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO)
    if args.model:
        config = load_config()
        model = config["models"][args.model]
        cycle = datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        from core.domain import crop
        with tempfile.TemporaryDirectory(prefix="smf-capture-") as tmp:
            path = write_atomic(capture_run(model, cycle, crop, Path(tmp)), output_path(data_root(), model, cycle))
        print(json.dumps({"captured": str(path), "bytes": path.stat().st_size}))
        return 0
    summary = tick(max_runs=args.max_runs, dry_run=args.dry_run)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

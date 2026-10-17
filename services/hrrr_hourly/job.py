"""Build one 12z cycle of the hourly HRRR product: fetch each hour, derive, store (Zarr + NetCDF).

Map images are not rendered by the build; an admin asks for one on demand (see render.render_from_store).
Every build records where its time went (timings.json next to the manifest, a summary in status.json).

Runs as a subprocess (scripts/hrrr_hourly_forecast.py) so its memory is returned to the
OS afterwards. Per-hour results are cached in a work dir, so an interrupted run resumes.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from services.hrrr_hourly import CYCLE_HOUR, FORECAST_HOURS, cycle_id, images_dir, root_dir
from services.hrrr_hourly import fetch, store

logger = logging.getLogger(__name__)

FORECAST_LOCK = Path(".locks/showmefire-forecast.lock")
MAPS_LOCK = Path(".locks/showmefire-maps.lock")
# Priority order: 1 daily forecast + notices, 2 the 15-minute maps, 3 this product. Between hours (and between
# images) the build pauses while a higher-priority job holds its lock, up to these caps so it can't stall forever.
MAX_PAUSE_FORECAST_SECONDS = 45 * 60
MAX_PAUSE_MAPS_SECONDS = 10 * 60
POLL_SECONDS = 10
MIN_AVAILABLE_MEMORY_MB = int(os.environ.get("SMF_HRRR_HOURLY_MIN_MEM_MB", "3000"))
MIN_FREE_DISK_MB = int(os.environ.get("SMF_HRRR_HOURLY_MIN_DISK_MB", "3000"))
SERIES_ELEMENTS = ("mixing_height_ft", "transport_speed_mph", "ventilation_kt_ft", "temp_f", "rh_pct",
                   "wind_speed_mph", "wind_gust_mph", "hdw", "cape_jkg", "smoke_surface_ugm3")


def forecast_lock_free(path: Path = FORECAST_LOCK) -> bool:
    """False while the lock at `path` is held exclusively (daily forecast, or a maps.sh cycle).
    Never holds the lock itself."""
    if not path.exists():
        return True
    try:
        with open(path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
        return True
    except OSError:
        return False


def available_memory_mb() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def free_disk_mb(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free // (1024 * 1024)


def wait_for_priority_work(root: Path, on_wait: Optional[Callable] = None, sleep: Callable = time.sleep,
                           clock: Callable = time.monotonic, forecast_lock: Path = FORECAST_LOCK,
                           maps_lock: Path = MAPS_LOCK, memory_mb: Callable = available_memory_mb) -> None:
    """Block while a higher-priority job runs: the daily forecast first, then the 15-minute maps.

    Also waits for free memory. Each wait is capped (see MAX_PAUSE_*), after which the build carries on.
    Raises RuntimeError if the disk is nearly full, since a half-written cycle helps nobody."""
    if free_disk_mb(root) < MIN_FREE_DISK_MB:
        raise RuntimeError(f"less than {MIN_FREE_DISK_MB} MB free disk; not building the hourly product")
    checks = (("daily forecast", lambda: forecast_lock_free(forecast_lock), MAX_PAUSE_FORECAST_SECONDS),
              ("15-minute maps", lambda: forecast_lock_free(maps_lock), MAX_PAUSE_MAPS_SECONDS),
              ("free memory", lambda: (memory_mb() or MIN_AVAILABLE_MEMORY_MB) >= MIN_AVAILABLE_MEMORY_MB,
               MAX_PAUSE_MAPS_SECONDS))
    for reason, free, cap in checks:
        started = clock()
        while not free() and clock() - started < cap:
            if on_wait:
                on_wait(reason)
            sleep(POLL_SECONDS)


STATES_UPDATING = ("building", "rendering")
FORECAST_STATUS = Path("status.json")
FORECAST_STATUS_KEY = "ForecastFireDanger"
FORECAST_GRACE = timedelta(hours=6)  # proceed anyway if the daily forecast never reports done


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_status(root: Optional[Path] = None) -> dict:
    try:
        return json.loads((Path(root or root_dir()) / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def set_state(root: Path, state: str, cycle: str, **extra) -> dict:
    """Write data/hrrr_hourly/status.json, the file the front end polls (also GET /hrrr-hourly/status).

    `updating` is true only while a rebuild is in progress; the previously built cycle keeps being served
    (`serving_cycle`) until the new one is swapped in."""
    previous = read_status(root)
    payload = {"product": "hrrr_hourly", "state": state, "cycle": cycle,
               "updating": state in STATES_UPDATING, "pending": state.startswith("waiting"),
               "serving_cycle": _serving_cycle(root), "started_at": previous.get("started_at") if
               previous.get("cycle") == cycle and state in STATES_UPDATING else None,
               "updated_at": _now_iso(), **extra}
    if state == "building" and not payload["started_at"]:
        payload["started_at"] = payload["updated_at"]
    path = root / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("status.json.tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    os.replace(temporary, path)
    return payload


def _serving_cycle(root: Path) -> Optional[str]:
    try:
        return json.loads((root / "latest.json").read_text())["cycle"]
    except (OSError, ValueError, KeyError):
        return None


def forecast_complete(cycle_time: datetime, status_path: Path = FORECAST_STATUS) -> bool:
    """True once the daily forecast has published this cycle (its status.json entry is 'updated')."""
    try:
        entry = json.loads(Path(status_path).read_text())[FORECAST_STATUS_KEY]
    except (OSError, ValueError, KeyError):
        return False
    return entry.get("status") == "updated" and entry.get("model_run") == f"{cycle_time:%Y-%m-%d} {cycle_time:%H}Z"


def _work_file(work: Path, lead: int) -> Path:
    return work / f"f{lead:02d}.npz"


def _series(hours: dict[int, dict[str, np.ndarray]], mask: np.ndarray) -> dict:
    out = {}
    for name in SERIES_ELEMENTS:
        rows = []
        for lead in sorted(hours):
            values = hours[lead].get(name)
            if values is None or not np.isfinite(values[mask]).any():
                rows.append(None)
                continue
            inside = values[mask][np.isfinite(values[mask])]
            rows.append({"lead_hour": lead, "min": round(float(inside.min()), 2),
                         "median": round(float(np.median(inside)), 2), "max": round(float(inside.max()), 2)})
        if any(rows):
            out[name] = rows
    return out


def _grid_meta(profiles, lat) -> dict:
    return {"ny": int(lat.shape[0]), "nx": int(lat.shape[1]), "projection": "lambert_conformal_conic",
            "central_longitude": profiles.central_longitude, "standard_parallel": profiles.standard_parallel,
            "lat_min": round(float(np.nanmin(lat)), 3), "lat_max": round(float(np.nanmax(lat)), 3),
            "resolution_km": 3.0}


def _timed_step(totals: dict, key: str):
    """Context manager adding the elapsed seconds of its block to totals[key]."""
    from contextlib import contextmanager

    @contextmanager
    def timer():
        started = time.perf_counter()
        try:
            yield
        finally:
            totals[key] = totals.get(key, 0.0) + time.perf_counter() - started
    return timer()


def _timings_report(totals: dict, per_hour: list, started: float, grid: Optional[dict], hours_requested: int) -> dict:
    """Round the totals and add the derived numbers that point at the slowest phase."""
    report = {key: round(value, 2) for key, value in totals.items()}
    report["total_sec"] = round(time.time() - started, 1)
    report["active_sec"] = round(report["total_sec"] - report.get("paused_sec", 0.0), 1)
    report["hours_requested"] = hours_requested
    report["prefetch"] = "prefetch_download_sec" in report
    report["hours_built"] = sum(1 for row in per_hour if not row.get("cached") and not row.get("failed"))
    report["hours_cached"] = sum(1 for row in per_hour if row.get("cached"))
    report["hours_failed"] = [row["lead"] for row in per_hour if row.get("failed")]
    built = [row for row in per_hour if not row.get("cached") and not row.get("failed")]
    if built:
        report["avg_hour_sec"] = round(sum(row["wall_sec"] for row in built) / len(built), 2)
        slowest = max(built, key=lambda row: row["wall_sec"])
        report["slowest_hour"] = {"lead": slowest["lead"], "sec": slowest["wall_sec"]}
        fields: dict[str, float] = {}
        for row in built:
            for name, seconds in (row.get("extras_fields") or {}).items():
                fields[name] = fields.get(name, 0.0) + seconds
        if fields:
            report["extras_fields_sec"] = {k: round(v, 2) for k, v in sorted(fields.items(), key=lambda kv: -kv[1])}
    # prefetch_download_sec runs alongside the other phases, so it is not a phase of its own here: the build only
    # loses time to it as prefetch_wait_sec (decoding finished before the next hour's download did).
    phases = {k: report[k] for k in ("prefetch_wait_sec", "fetch_profiles_sec", "fetch_extras_sec", "fetch_other_sec",
                                     "derive_sec",
                                     "cache_io_sec", "series_sec", "dataset_sec", "zarr_sec", "netcdf_sec",
                                     "render_sec") if k in report}
    if phases:
        report["slowest_phase"] = max(phases, key=phases.get)
    report["grid"] = {k: grid[k] for k in ("ny", "nx") if grid and k in grid}
    report["per_hour"] = per_hour
    return report


def build_cycle(cycle_time: datetime, hours=FORECAST_HOURS, render: bool = False, fetch_hour: Callable = fetch.fetch_hour,
                derive_hour: Optional[Callable] = None, root: Optional[Path] = None,
                images: Optional[Path] = None, on_progress: Optional[Callable] = None,
                wait: Optional[Callable] = None, prefetch: bool = True) -> dict:
    """Build the Zarr + NetCDF store for a cycle. `render=True` also draws every map image (slow; off by default,
    images are rendered on demand instead). The returned manifest carries a "timings" report, also saved as
    <cycle>/timings.json.

    `prefetch` downloads the next hour's GRIB files on a background thread while this hour is decoded (only with
    the real fetch_hour; a custom one is called as fetch_hour(cycle_time, lead))."""
    from services.fire_weather import summary as fs
    from services.hrrr_hourly import derive, render as renderer

    derive_hour = derive_hour or derive.derive_hour
    root = Path(root or root_dir())
    cycle = cycle_id(cycle_time)
    work = root / ".work" / cycle
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict[int, dict[str, np.ndarray]] = {}
    lat = lon = meta = None
    failed = []
    hours = tuple(hours)
    totals: dict[str, float] = {}
    per_hour: list[dict] = []

    def progress(phase: str, done: int):
        # Time spent blocked on higher-priority jobs is reported apart, so it never looks like slow code.
        if wait:
            with _timed_step(totals, "paused_sec"):
                wait(root)
        if on_progress:
            on_progress(phase, done, len(hours))

    uncached = [lead for lead in hours if not _work_file(work, lead).is_file()]
    prefetcher = None
    if prefetch and fetch_hour is fetch.fetch_hour and uncached:
        prefetcher = fetch.Prefetcher(cycle_time, uncached, work / "grib")
    try:
        for done, lead in enumerate(hours):
            progress("fetching", done)
            cached = _work_file(work, lead)
            row: dict = {"lead": lead}
            hour_started = time.perf_counter()
            hour_dir = None
            try:
                if cached.is_file():
                    with _timed_step(totals, "cache_io_sec"):
                        with np.load(cached) as data:
                            results[lead] = {k: data[k] for k in data.files if k not in ("lat", "lon")}
                            lat, lon = data["lat"], data["lon"]
                    row.update(cached=True, fetch_total_sec=0.0)
                    continue
                if prefetcher:
                    with _timed_step(totals, "prefetch_wait_sec"):
                        waited = time.perf_counter()
                        hour_dir = prefetcher.wait_for(lead)
                        row["prefetch_wait_sec"] = round(time.perf_counter() - waited, 2)
                    row.update({f"prefetch_{k}": v for k, v in prefetcher.results.get(lead, {}).items()
                                if k != "failed"})
                    if prefetcher.results.get(lead, {}).get("failed"):
                        row["prefetch_failed"] = prefetcher.results[lead]["failed"]
                    totals["prefetch_download_sec"] = totals.get("prefetch_download_sec", 0.0) + row.get(
                        "prefetch_download_sec", 0.0)
                fetch.take_timings()  # drop anything left from an earlier call
                fetch_started = time.perf_counter()
                profiles, extras = (fetch_hour(cycle_time, lead, workdir=hour_dir) if prefetcher
                                    else fetch_hour(cycle_time, lead))
                fetch_total = time.perf_counter() - fetch_started
                detail = fetch.take_timings()
                derive_started = time.perf_counter()
                fields = derive_hour(profiles, extras)
                derive_sec = time.perf_counter() - derive_started
                lat, lon = profiles.lat, profiles.lon
                meta = meta or _grid_meta(profiles, lat)
                save_started = time.perf_counter()
                np.savez(_work_file(work, lead).with_suffix(".tmp.npz"), lat=lat, lon=lon, **fields)
                os.replace(_work_file(work, lead).with_suffix(".tmp.npz"), cached)
                totals["cache_io_sec"] = totals.get("cache_io_sec", 0.0) + time.perf_counter() - save_started
                results[lead] = fields
                profiles_sec = detail.get("profiles_sec", 0.0)
                extras_sec = detail.get("extras_sec", 0.0)
                totals["fetch_profiles_sec"] = totals.get("fetch_profiles_sec", 0.0) + profiles_sec
                totals["fetch_extras_sec"] = totals.get("fetch_extras_sec", 0.0) + extras_sec
                totals["fetch_other_sec"] = totals.get("fetch_other_sec", 0.0) + max(
                    fetch_total - profiles_sec - extras_sec, 0)
                totals["derive_sec"] = totals.get("derive_sec", 0.0) + derive_sec
                row.update(fetch_total_sec=round(fetch_total, 2), fetch_profiles_sec=profiles_sec,
                           fetch_extras_sec=extras_sec, extras_fields=detail.get("extras_fields"),
                           extras_missing=detail.get("extras_missing"), derive_sec=round(derive_sec, 2),
                           elements=len(fields))
            except Exception:
                logger.exception("hrrr_hourly %s f%02d failed", cycle, lead)
                failed.append(lead)
                row.update(failed=True, fetch_total_sec=round(time.perf_counter() - hour_started, 2))
            finally:
                if prefetcher and hour_dir is not None:
                    prefetcher.release(lead)
                row["wall_sec"] = round(time.perf_counter() - hour_started, 2)
                per_hour.append(row)
    finally:
        if prefetcher:
            prefetcher.close()
    if not results:
        raise RuntimeError(f"no forecast hours could be built for {cycle}")
    meta = meta or {"ny": int(lat.shape[0]), "nx": int(lat.shape[1])}
    with _timed_step(totals, "series_sec"):
        mask = fs.missouri_mask(lat, lon)
        series = _series(results, mask)
    manifest = store.write_cycle(cycle_time, results, lat, lon, meta, series, hours, root=root)
    for key, seconds in manifest.pop("write_timings", {}).items():
        totals[key] = totals.get(key, 0.0) + seconds
    manifest["failed_hours"] = failed
    if render:
        counts = {}
        for done, lead in enumerate(sorted(results)):
            progress("rendering", done)
            with _timed_step(totals, "render_sec"):
                counts[lead] = len(renderer.render_hour(results[lead], lat, lon, cycle_time, lead, root=images))
        manifest["images"] = {"dir": f"{images or images_dir()}/{cycle}", "per_hour": counts}
    shutil.rmtree(work, ignore_errors=True)
    manifest["runtime_sec"] = round(time.time() - started, 1)
    timings = _timings_report(totals, per_hour, started, meta, len(hours))
    manifest["timings"] = timings
    try:
        store._atomic_json(store.timings_path(cycle, root), timings)
    except OSError:
        logger.exception("hrrr_hourly %s: could not write timings.json", cycle)
    logger.info("hrrr_hourly %s timings: total %.0fs (paused %.0fs), slowest phase %s", cycle, timings["total_sec"],
                timings.get("paused_sec", 0.0), timings.get("slowest_phase"))
    return manifest


def run_auto(now: Optional[datetime] = None, force: bool = False, cycle_time: Optional[datetime] = None,
             hours=FORECAST_HOURS, build: Callable = build_cycle, ready: Callable = fetch.cycle_ready,
             lock_free: Callable = forecast_lock_free, forecast_done: Callable = forecast_complete,
             root: Optional[Path] = None) -> dict:
    """One scheduler tick: build the newest 12z cycle only after the daily forecast has finished.

    Order is fixed: the original forecast (maps, Discord/mobile notices, beta run) runs first and holds the
    forecast lock until its script exits; this product starts after that, never alongside it."""
    root = Path(root or root_dir())
    now = now or datetime.now(timezone.utc)
    cycle_time = cycle_time or fetch.latest_cycle(now, CYCLE_HOUR)
    cycle = cycle_id(cycle_time)
    existing = store.read_manifest(cycle, root)
    if existing and not existing.get("missing_hours") and not force:
        return {"action": "skip", "reason": "already built", "cycle": cycle}
    if not force and not ready(cycle_time, max(hours)):
        set_state(root, "waiting_for_hrrr", cycle, detail="HRRR 12z run not fully published yet")
        return {"action": "skip", "reason": "cycle not fully published", "cycle": cycle}
    if not force and not forecast_done(cycle_time) and now < cycle_time + FORECAST_GRACE:
        set_state(root, "waiting_for_forecast", cycle, detail="daily forecast has not finished yet")
        return {"action": "skip", "reason": "daily forecast not finished", "cycle": cycle}
    if not lock_free():
        set_state(root, "waiting_for_forecast", cycle, detail="daily forecast is running")
        return {"action": "skip", "reason": "forecast running", "cycle": cycle}
    root.mkdir(parents=True, exist_ok=True)
    guard = open(root / ".build.lock", "a+")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        guard.close()
        return {"action": "skip", "reason": "another build is running", "cycle": cycle}
    try:
        return _build_locked(root, cycle, cycle_time, now, hours, build)
    finally:
        guard.close()


def _build_locked(root: Path, cycle: str, cycle_time: datetime, now: datetime, hours, build: Callable) -> dict:
    hours = tuple(hours)
    set_state(root, "building", cycle, phase="fetching", hours_total=len(hours), hours_done=0)

    def on_progress(phase: str, done: int, total: int):
        paused["done"] = done
        set_state(root, "building" if phase == "fetching" else "rendering", cycle, phase=phase,
                  hours_total=total, hours_done=done)

    paused = {}

    def wait(root_dir_: Path):
        def note(reason: str):
            if paused.get("reason") != reason:
                paused["reason"] = reason
                set_state(root, "building", cycle, phase="paused", paused_for=reason, hours_total=len(hours),
                          hours_done=paused.get("done", 0))
        wait_for_priority_work(root_dir_, on_wait=note)
        paused.pop("reason", None)

    try:
        manifest = build(cycle_time, hours=hours, root=root, on_progress=on_progress, wait=wait)
    except Exception as exc:
        set_state(root, "failed", cycle, error=str(exc)[:300], finished_at=_now_iso())
        raise
    removed = store.prune(now, root=root)
    if removed:
        from services.hrrr_hourly import render as renderer
        renderer.prune_images(store.list_cycles(root))
    set_state(root, "complete" if not manifest["missing_hours"] else "partial", cycle, label=manifest["label"],
              issued_at_utc=manifest["issued_at_utc"], hours=manifest["forecast_hours"],
              missing_hours=manifest["missing_hours"], runtime_sec=manifest.get("runtime_sec"),
              variables=len(manifest["variables"]), finished_at=_now_iso(),
              timings={k: v for k, v in manifest.get("timings", {}).items() if k != "per_hour"})
    return {"action": "built", "cycle": cycle, "hours": manifest["forecast_hours"],
            "missing_hours": manifest["missing_hours"]}

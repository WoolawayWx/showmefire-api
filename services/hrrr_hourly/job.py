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
from services.hrrr_hourly import controls, fetch, publish, store

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
                           maps_lock: Path = MAPS_LOCK, memory_mb: Callable = available_memory_mb,
                           include_forecast: bool = True) -> None:
    """Block while a higher-priority job runs: the daily forecast first, then the 15-minute maps.

    Also waits for free memory. Each wait is capped (see MAX_PAUSE_*), after which the build carries on.
    Raises RuntimeError if the disk is nearly full, since a half-written cycle helps nobody.

    `include_forecast=False` skips the daily-forecast wait: the early prefetch is meant to run alongside the
    forecast (it only downloads and decodes single hours, at the lowest priority), so waiting for the forecast's
    lock would stall it behind the very job it is there to get ahead of."""
    if free_disk_mb(root) < MIN_FREE_DISK_MB:
        raise RuntimeError(f"less than {MIN_FREE_DISK_MB} MB free disk; not building the hourly product")
    checks = (("daily forecast", lambda: forecast_lock_free(forecast_lock), MAX_PAUSE_FORECAST_SECONDS),
              ("15-minute maps", lambda: forecast_lock_free(maps_lock), MAX_PAUSE_MAPS_SECONDS),
              ("free memory", lambda: (memory_mb() or MIN_AVAILABLE_MEMORY_MB) >= MIN_AVAILABLE_MEMORY_MB,
               MAX_PAUSE_MAPS_SECONDS))
    if not include_forecast:
        checks = checks[1:]
    for reason, free, cap in checks:
        started = clock()
        while not free() and clock() - started < cap:
            if on_wait:
                on_wait(reason)
            sleep(POLL_SECONDS)


STATES_UPDATING = ("building", "rendering", "publishing")
FORECAST_STATUS = Path("status.json")
FORECAST_STATUS_KEY = "ForecastFireDanger"
# Build the 12z cycle anyway if the daily forecast has not reported done this long after the cycle. Kept under the time a
# cycle stays among the candidates (newest + BACKFILL_CYCLES), or a day the forecast fails would skip the 12z run.
FORECAST_GRACE = timedelta(hours=4)


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


# Keys saved in a work file beside the derived elements. They are never part of the product: the loader drops
# them, so they cannot reach the Zarr store, the series or the manifest.
#   raw_*   unprocessed HRRR surface fields (K, %, m/s) that another product reading the same hour can reuse
#   grid_*  projection details, so a build resumed entirely from work files still writes full grid metadata
RAW_PREFIX = "raw_"
GRID_PREFIX = "grid_"
CACHE_ONLY_PREFIXES = (RAW_PREFIX, GRID_PREFIX)


def _raw_fields(profiles, extras: dict) -> dict[str, np.ndarray]:
    """The surface fields the daily forecast also reads (2 m T and RH, 10 m wind, gust), unprocessed.

    Temperature is returned to Kelvin from the profile's degC; the rest are the GRIB values. Fields a run
    lacks are left out."""
    raw: dict[str, np.ndarray] = {}
    t2_c = getattr(profiles, "t2_c", None)
    if t2_c is not None:
        raw[f"{RAW_PREFIX}t2m_k"] = (np.asarray(t2_c, dtype="float64") + 273.15).astype("float32")
    for name in ("u10", "v10"):
        value = getattr(profiles, name, None)
        if value is not None:
            raw[f"{RAW_PREFIX}{name}"] = np.asarray(value, dtype="float32")
    for name in ("rh2", "gust"):
        if extras.get(name) is not None:
            raw[f"{RAW_PREFIX}{name}"] = np.asarray(extras[name], dtype="float32")
    return raw


def _save_work_file(path: Path, lat, lon, fields: dict, profiles=None, extras: Optional[dict] = None) -> None:
    """Write one hour's derived elements (plus raw and grid details when `profiles` is given), atomically."""
    extra: dict[str, np.ndarray] = {}
    if profiles is not None:
        extra.update(_raw_fields(profiles, extras or {}))
        for name in ("central_longitude", "standard_parallel"):
            value = getattr(profiles, name, None)
            if value is not None:
                extra[f"{GRID_PREFIX}{name}"] = np.float64(value)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, lat=lat, lon=lon, **fields, **extra)
    os.replace(temporary, path)


def _load_work_file(path: Path) -> tuple[dict, np.ndarray, np.ndarray, Optional[dict]]:
    """(derived elements, lat, lon, grid metadata or None) from a work file, ignoring raw_/grid_ keys."""
    with np.load(path) as data:
        fields = {k: data[k] for k in data.files
                  if k not in ("lat", "lon") and not k.startswith(CACHE_ONLY_PREFIXES)}
        lat, lon = data["lat"], data["lon"]
        meta = None
        if f"{GRID_PREFIX}central_longitude" in data.files and f"{GRID_PREFIX}standard_parallel" in data.files:
            meta = _grid_meta_from(float(data[f"{GRID_PREFIX}central_longitude"]),
                                   float(data[f"{GRID_PREFIX}standard_parallel"]), lat)
    return fields, lat, lon, meta


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


def _grid_meta_from(central_longitude: float, standard_parallel: float, lat) -> dict:
    return {"ny": int(lat.shape[0]), "nx": int(lat.shape[1]), "projection": "lambert_conformal_conic",
            "central_longitude": central_longitude, "standard_parallel": standard_parallel,
            "lat_min": round(float(np.nanmin(lat)), 3), "lat_max": round(float(np.nanmax(lat)), 3),
            "resolution_km": 3.0}


def _grid_meta(profiles, lat) -> dict:
    return _grid_meta_from(profiles.central_longitude, profiles.standard_parallel, lat)


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
                        results[lead], lat, lon, cached_meta = _load_work_file(cached)
                        meta = meta or cached_meta
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
                _save_work_file(cached, lat, lon, fields, profiles, extras)
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


PREFETCH_TIMEOUT_SECONDS = int(os.environ.get("SMF_HRRR_HOURLY_PREFETCH_TIMEOUT_SECONDS", str(2 * 3600)))
PREFETCH_POLL_SECONDS = int(os.environ.get("SMF_HRRR_HOURLY_PREFETCH_POLL_SECONDS", "60"))


def _wait_for_prefetch_slot(root: Path) -> None:
    wait_for_priority_work(root, include_forecast=False)


def prefetch_cycle(cycle_time: Optional[datetime] = None, hours=FORECAST_HOURS, root: Optional[Path] = None,
                   fetch_hour: Callable = fetch.fetch_hour, derive_hour: Optional[Callable] = None,
                   hour_ready: Callable = fetch.cycle_ready, wait: Optional[Callable] = None,
                   sleep: Callable = time.sleep, clock: Callable = time.monotonic,
                   poll_seconds: Optional[int] = None, timeout_seconds: Optional[int] = None,
                   now: Optional[datetime] = None, on_hour_done: Optional[Callable] = None) -> dict:
    """Process each forecast hour of the 12z run as HRRR publishes it, ahead of the real build.

    Does exactly what `build_cycle` does per hour (fetch, derive, save `.work/<cycle>/f{lead}.npz`) and nothing
    after it: no Zarr, NetCDF, manifest, publish or status.json change. The later build then finds every hour
    already done and only writes the store. Spreading the work across the ~14 minutes HRRR takes to post f00-f18
    also keeps the peak load and memory low, instead of decoding everything in one go after the daily forecast.

    Waits (polling `hour_ready`, the idx files of both products) for an hour that is not published yet, up to
    `timeout_seconds` overall. An hour that fails is left for the build to retry. Holds `.build.lock`, so it never
    runs alongside a build. Safe to run again: hours that already have a work file are skipped.

    `on_hour_done(lead, done_hours)`, if given, is called after each hour is saved (or found already saved); an
    exception in it is logged and ignored. The caller uses it to start work as soon as the hours it needs exist.
    Hours are processed in the order given, so a caller that needs some of them first can list those first."""
    from services.hrrr_hourly import derive

    derive_hour = derive_hour or derive.derive_hour
    wait = wait or _wait_for_prefetch_slot
    poll = PREFETCH_POLL_SECONDS if poll_seconds is None else poll_seconds
    timeout = PREFETCH_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    root = Path(root or root_dir())
    cycle_time = cycle_time or fetch.latest_cycle(now, CYCLE_HOUR)
    cycle = cycle_id(cycle_time)
    existing = store.read_manifest(cycle, root)
    if existing and not existing.get("missing_hours"):
        return {"action": "skip", "reason": "already built", "cycle": cycle}
    root.mkdir(parents=True, exist_ok=True)
    guard = open(root / ".build.lock", "a+")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        guard.close()
        return {"action": "skip", "reason": "another build is running", "cycle": cycle}
    try:
        return _prefetch_locked(root, cycle, cycle_time, tuple(hours), fetch_hour, derive_hour, hour_ready, wait,
                                sleep, clock, poll, timeout, on_hour_done)
    finally:
        guard.close()


def _prefetch_locked(root: Path, cycle: str, cycle_time: datetime, hours: tuple, fetch_hour: Callable,
                     derive_hour: Callable, hour_ready: Callable, wait: Callable, sleep: Callable, clock: Callable,
                     poll: int, timeout: int, on_hour_done: Optional[Callable] = None) -> dict:
    work = root / ".work" / cycle
    work.mkdir(parents=True, exist_ok=True)
    started = clock()
    done: list[int] = []
    failed: list[int] = []

    def hour_done(lead: int) -> None:
        done.append(lead)
        if on_hour_done:
            try:
                on_hour_done(lead, list(done))
            except Exception:
                logger.exception("hrrr_hourly prefetch %s: the after-hour callback failed for f%02d", cycle, lead)

    def result(reason: str) -> dict:
        left = [lead for lead in hours if lead not in done and lead not in failed]
        return {"action": "prefetched", "cycle": cycle, "reason": reason, "hours_done": done,
                "hours_failed": failed, "hours_pending": left}

    for lead in hours:
        target = _work_file(work, lead)
        if target.is_file():
            hour_done(lead)
            continue
        while not hour_ready(cycle_time, lead):
            if clock() - started >= timeout:
                logger.warning("hrrr_hourly prefetch %s: f%02d still not published after %ds; stopping", cycle, lead,
                               timeout)
                return result("timed out waiting for HRRR")
            sleep(poll)
        try:
            wait(root)
        except RuntimeError as exc:  # nearly full disk: stop rather than half-write hours
            logger.error("hrrr_hourly prefetch %s stopped: %s", cycle, exc)
            return result(str(exc))
        try:
            fetch.take_timings()  # drop anything left from an earlier call
            profiles, extras = fetch_hour(cycle_time, lead)
            fields = derive_hour(profiles, extras)
            _save_work_file(target, profiles.lat, profiles.lon, fields, profiles, extras)
            logger.info("hrrr_hourly prefetch %s f%02d saved (%d of %d)", cycle, lead, len(done) + 1, len(hours))
            hour_done(lead)
        except Exception:
            logger.exception("hrrr_hourly prefetch %s f%02d failed; the build will retry it", cycle, lead)
            failed.append(lead)
    return result("complete" if not failed else "some hours failed")


BACKFILL_CYCLES = int(os.environ.get("SMF_HRRR_HOURLY_BACKFILL_CYCLES", "3"))


def run_auto(now: Optional[datetime] = None, force: bool = False, cycle_time: Optional[datetime] = None,
             hours=FORECAST_HOURS, build: Callable = build_cycle, ready: Callable = fetch.cycle_ready,
             lock_free: Callable = forecast_lock_free, forecast_done: Callable = forecast_complete,
             root: Optional[Path] = None) -> dict:
    """One scheduler tick: build the newest hourly cycle that is fully published and not built yet.

    With no `cycle_time` it looks at the newest cycle old enough to be ready and BACKFILL_CYCLES earlier ones, newest
    first, so a builder that falls behind skips old cycles instead of building them long after they were current. An
    explicit `cycle_time` builds just that cycle.

    The cycle the daily forecast also uses (CYCLE_HOUR, 12z) goes after that forecast: the original forecast (maps,
    Discord/mobile notices, beta run) runs first and holds the forecast lock until its script exits. Every cycle, 12z
    or not, also waits while that lock is held; nothing here runs alongside the forecast."""
    root = Path(root or root_dir())
    now = now or datetime.now(timezone.utc)
    # The admin pause switch (controls.py) stops new builds from starting; one already running finishes. It does not
    # apply to an explicit cycle or --force, which are manual.
    if cycle_time is None and not force and controls.is_paused(root):
        serving = _serving_cycle(root) or ""
        if read_status(root).get("state") != "paused":
            set_state(root, "paused", serving, detail="paused by an admin")
        return {"action": "skip", "reason": "paused", "cycle": serving}
    if read_status(root).get("state") == "paused":      # resumed: do not leave the status saying paused
        set_state(root, "complete", _serving_cycle(root) or "", detail="resumed")
    candidates = [cycle_time] if cycle_time is not None else fetch.cycle_candidates(now, BACKFILL_CYCLES)
    # Why nothing could be built: a ready cycle held up by the forecast takes precedence over a newer cycle that is
    # simply not published yet. Each is (reason, state, cycle, detail).
    blocked: Optional[tuple] = None
    unpublished: Optional[tuple] = None
    built: Optional[str] = None
    for candidate in candidates:
        cycle = cycle_id(candidate)
        existing = store.read_manifest(cycle, root)
        if existing and not existing.get("missing_hours") and not force:
            built = built or cycle
            continue
        reason = None
        if not force and not ready(candidate, max(hours)):
            reason = ("cycle not fully published", "waiting_for_hrrr",
                      f"HRRR {candidate:%H}z run not fully published yet")
        elif (candidate.hour == CYCLE_HOUR and not force and not forecast_done(candidate)
              and now < candidate + FORECAST_GRACE):
            reason = ("daily forecast not finished", "waiting_for_forecast", "daily forecast has not finished yet")
        elif not lock_free():
            reason = ("forecast running", "waiting_for_forecast", "daily forecast is running")
        if reason:
            entry = (*reason[:2], cycle, reason[2])
            if reason[0] == "cycle not fully published":
                unpublished = unpublished or entry
                continue
            blocked = blocked or entry
            if reason[0] == "forecast running":
                break   # the lock is global: no other cycle can run either
            continue
        root.mkdir(parents=True, exist_ok=True)
        guard = open(root / ".build.lock", "a+")
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            guard.close()
            return {"action": "skip", "reason": "another build is running", "cycle": cycle}
        try:
            return _build_locked(root, cycle, candidate, now, hours, build)
        finally:
            guard.close()
    if blocked or unpublished:
        reason, state, cycle, detail = blocked or unpublished
        set_state(root, state, cycle, detail=detail)
        return {"action": "skip", "reason": reason, "cycle": cycle}
    return {"action": "skip", "reason": "already built", "cycle": built or cycle_id(candidates[0])}


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
        controls.record_event({"cycle": cycle, "outcome": "failed", "error": str(exc)[:300]}, root)
        raise
    # Before "complete", so a browser that follows the new serving_cycle finds the run on the CDN. Best effort:
    # on failure the API still serves it and the viewer falls back to the API.
    set_state(root, "publishing", cycle, label=manifest["label"], hours_total=len(hours), hours_done=len(hours))
    publish.try_publish_cycle(cycle, root)
    removed = store.prune(now, root=root)
    if removed:
        from services.hrrr_hourly import render as renderer
        renderer.prune_images(store.list_cycles(root))
    set_state(root, "complete" if not manifest["missing_hours"] else "partial", cycle, label=manifest["label"],
              issued_at_utc=manifest["issued_at_utc"], hours=manifest["forecast_hours"],
              missing_hours=manifest["missing_hours"], runtime_sec=manifest.get("runtime_sec"),
              variables=len(manifest["variables"]), finished_at=_now_iso(),
              timings={k: v for k, v in manifest.get("timings", {}).items() if k != "per_hour"})
    controls.record_event({"cycle": cycle, "outcome": "partial" if manifest["missing_hours"] else "complete",
                           "runtime_sec": manifest.get("runtime_sec"), "hours": len(manifest["forecast_hours"]),
                           "missing_hours": manifest["missing_hours"]}, root)
    return {"action": "built", "cycle": cycle, "hours": manifest["forecast_hours"],
            "missing_hours": manifest["missing_hours"]}

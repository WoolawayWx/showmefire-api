"""Build one 12z cycle of the hourly HRRR product: fetch each hour, derive, store, render.

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


def build_cycle(cycle_time: datetime, hours=FORECAST_HOURS, render: bool = True, fetch_hour: Callable = fetch.fetch_hour,
                derive_hour: Optional[Callable] = None, root: Optional[Path] = None,
                images: Optional[Path] = None, on_progress: Optional[Callable] = None,
                wait: Optional[Callable] = None) -> dict:
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

    def progress(phase: str, done: int):
        if wait:
            wait(root)
        if on_progress:
            on_progress(phase, done, len(hours))

    for done, lead in enumerate(hours):
        progress("fetching", done)
        cached = _work_file(work, lead)
        try:
            if cached.is_file():
                with np.load(cached) as data:
                    results[lead] = {k: data[k] for k in data.files if k not in ("lat", "lon")}
                    lat, lon = data["lat"], data["lon"]
                continue
            profiles, extras = fetch_hour(cycle_time, lead)
            fields = derive_hour(profiles, extras)
            lat, lon = profiles.lat, profiles.lon
            meta = meta or _grid_meta(profiles, lat)
            np.savez(_work_file(work, lead).with_suffix(".tmp.npz"), lat=lat, lon=lon, **fields)
            os.replace(_work_file(work, lead).with_suffix(".tmp.npz"), cached)
            results[lead] = fields
        except Exception:
            logger.exception("hrrr_hourly %s f%02d failed", cycle, lead)
            failed.append(lead)
    if not results:
        raise RuntimeError(f"no forecast hours could be built for {cycle}")
    meta = meta or {"ny": int(lat.shape[0]), "nx": int(lat.shape[1])}
    mask = fs.missouri_mask(lat, lon)
    manifest = store.write_cycle(cycle_time, results, lat, lon, meta, _series(results, mask), hours, root=root)
    manifest["failed_hours"] = failed
    if render:
        counts = {}
        for done, lead in enumerate(sorted(results)):
            progress("rendering", done)
            counts[lead] = len(renderer.render_hour(results[lead], lat, lon, cycle_time, lead, root=images))
        manifest["images"] = {"dir": f"{images or images_dir()}/{cycle}", "per_hour": counts}
    shutil.rmtree(work, ignore_errors=True)
    manifest["runtime_sec"] = round(time.time() - started, 1)
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
              variables=len(manifest["variables"]), finished_at=_now_iso())
    return {"action": "built", "cycle": cycle, "hours": manifest["forecast_hours"],
            "missing_hours": manifest["missing_hours"]}

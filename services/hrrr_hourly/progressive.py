"""Build an HRRR cycle one forecast hour at a time, and publish each hour as soon as it is processed.

The one-shot build (job.build_cycle / job.run_auto) waits for the whole cycle (f18) before it writes anything. This does the
same per-hour work (fetch, derive, save a work file; see job.prefetch_cycle) but, after every hour:

  * writes that hour into the cycle's Zarr store, which was created at full length and starts empty (live_store.py),
  * rewrites the manifest (`complete: false`, the hours so far, the hours still `pending_hours`),
  * updates the peak fire danger product for the local day the hour belongs to (peak.py),
  * uploads what changed to the CDN (cdn_datasets.publish_update) and, once enough of the cycle is in, makes it the latest.

So the viewer can show f00 about an hour after the cycle starts instead of waiting for f18 (and, for 12z, the daily forecast).
When every hour is in (or the missing ones are given up on), the cycle is finished: the NetCDF copy is written, the manifests
are marked complete and cached for good, and the work files go.

Each scheduler tick (scripts/hrrr_hourly_forecast.py --auto) calls `advance`: it processes whatever hours are published and not
done yet, for the newest cycle that has work, then exits. All state is on disk (`<cycle>/live.json`, the work files, the
store), so a crashed or interrupted tick just carries on at the next one. SMF_HRRR_HOURLY_PROGRESSIVE=0 goes back to the
one-shot build.
"""
from __future__ import annotations

import fcntl
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from services import cdn_datasets
from services.hrrr_hourly import CYCLE_HOUR, ELEMENTS, FORECAST_HOURS, PRODUCT, SCHEMA_VERSION, cycle_id, root_dir
from services.hrrr_hourly import controls, fetch, job, live_store, peak, publish, store

logger = logging.getLogger(__name__)

# The cycle becomes the dataset's "latest" (the one the viewer opens by default) only once this many hours are in, so the
# newest run is never a nearly empty map. Until then it is listed, and reachable by its id, as a run that is still filling in.
LATEST_AFTER_HOURS = int(os.environ.get("SMF_HRRR_HOURLY_LATEST_AFTER_HOURS", "6"))
# Hours HRRR has not published this long after the cycle are recorded as missing and the cycle is finished without them.
GIVE_UP_AFTER = timedelta(hours=int(os.environ.get("SMF_HRRR_HOURLY_GIVE_UP_HOURS", "3")))
MAX_HOUR_ATTEMPTS = 3
TICK_BUDGET_SECONDS = 20 * 60          # one tick stops after this long; the next carries on
BACKFILL_CYCLES = job.BACKFILL_CYCLES


def enabled() -> bool:
    return os.environ.get("SMF_HRRR_HOURLY_PROGRESSIVE", "1").strip().lower() not in ("0", "false", "no", "off")


# --- per-cycle bookkeeping ---------------------------------------------------------------------

def _live_path(cycle_dir: Path) -> Path:
    return cycle_dir / "live.json"


def _read_live(cycle_dir: Path) -> Optional[dict]:
    import json

    try:
        return json.loads(_live_path(cycle_dir).read_text())
    except (OSError, ValueError):
        return None


def _write_live(cycle_dir: Path, live: dict) -> None:
    store._atomic_json(_live_path(cycle_dir), live)


def _new_live(hours, now: datetime) -> dict:
    return {"hours": [int(h) for h in hours], "written": [], "failed": {}, "elements": {}, "series": {},
            "peak_days": {}, "pending_upload": {"hourly": {"stable": [], "changing": []},
                                                "peak": {"stable": [], "changing": []}},
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "revision": 0, "first_hour_at": None,
            "work_seconds": 0.0}


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_fields(path: Path, names=None) -> dict[str, np.ndarray]:
    """A work file's derived elements, or just `names` of them (cheaper when only a few are needed)."""
    with np.load(path) as data:
        wanted = names if names is not None else [k for k in data.files if k not in ("lat", "lon")
                                                   and not k.startswith(job.CACHE_ONLY_PREFIXES)]
        return {k: data[k] for k in wanted if k in data.files}


class Cycle:
    """Everything one advance of one cycle needs."""

    def __init__(self, root: Path, peak_root: Path, cycle_time: datetime, now: datetime, live: dict, client=None,
                 publishing: bool = True):
        self.root, self.peak_root, self.cycle_time, self.now = root, peak_root, cycle_time, now
        self.cycle = cycle_id(cycle_time)
        self.dir = root / self.cycle
        self.peak_dir = peak_root / self.cycle
        self.work = root / ".work" / self.cycle
        self.live = live
        self.hours = tuple(live["hours"])
        self.client = client
        self.publishing = publishing
        self.lat = self.lon = None
        self.meta: Optional[dict] = None
        self.days = peak.plan(cycle_time, self.hours)
        self._mask = None

    def work_file(self, lead: int) -> Path:
        return job._work_file(self.work, lead)

    def mask(self):
        if self._mask is None:
            from services.fire_weather import summary as fs
            self._mask = fs.missouri_mask(self.lat, self.lon)
        return self._mask

    @property
    def present(self) -> list[int]:
        return sorted(self.live["written"])

    @property
    def latest_eligible(self) -> bool:
        return len(self.present) >= min(LATEST_AFTER_HOURS, len(self.hours))

    def save(self) -> None:
        _write_live(self.dir, self.live)

    def ensure_grid(self) -> None:
        """Load the grid from a saved hour when this tick has not processed one itself (a resumed or finishing cycle)."""
        if self.lat is not None:
            return
        for lead in self.present:
            if self.work_file(lead).is_file():
                _fields, self.lat, self.lon, meta = job._load_work_file(self.work_file(lead))
                self.meta = self.meta or meta or {"ny": int(self.lat.shape[0]), "nx": int(self.lat.shape[1])}
                return


# --- one hour ------------------------------------------------------------------------------------

def _process_hour(ctx: Cycle, lead: int, fetch_hour: Callable, derive_hour: Callable) -> None:
    """Fetch, derive and save one hour's work file (the same steps as build_cycle and prefetch_cycle)."""
    fetch.take_timings()                      # drop anything left from an earlier call
    profiles, extras = fetch_hour(ctx.cycle_time, lead)
    fields = derive_hour(profiles, extras)
    ctx.work.mkdir(parents=True, exist_ok=True)
    job._save_work_file(ctx.work_file(lead), profiles.lat, profiles.lon, fields, profiles, extras)


def _write_hourly_manifest(ctx: Cycle, finished: bool, files: Optional[dict] = None) -> dict:
    live, present = ctx.live, ctx.present
    series = {}
    for name in job.SERIES_ELEMENTS:
        rows = [live["series"].get(name, {}).get(str(lead)) for lead in present]
        if any(rows):
            series[name] = rows
    variables = {}
    for element in ELEMENTS:
        have = [lead for lead in present if lead in live["elements"].get(element.name, [])]
        if have:
            variables[element.name] = {"long_name": element.long_name, "units": element.units, "source": element.source,
                                       "derived": bool(element.derived), "scale": element.scale, "hours_present": have}
    pending = [lead for lead in ctx.hours if lead not in present]
    manifest = {
        "product": PRODUCT, "schema_version": SCHEMA_VERSION, "model": "HRRR",
        "label": f"Forecast from the {ctx.cycle_time:%H}z HRRR run",
        "cycle": ctx.cycle, "cycle_time_utc": _iso(ctx.cycle_time),
        "issued_at_utc": live["first_hour_at"] or live["created_at"], "updated_at_utc": _iso(ctx.now),
        "forecast_hours": present, "missing_hours": pending if finished else [],
        "pending_hours": [] if finished else pending, "hours_expected": len(ctx.hours),
        "zarr_leads": list(ctx.hours), "complete": finished, "revision": live["revision"], "volatile_steps": [],
        "valid_times_utc": [_iso(ctx.cycle_time + timedelta(hours=lead)) for lead in present],
        "variables": variables, "grid": ctx.meta or {}, "series": series,
        "files": {"zarr": f"{ctx.cycle}/forecast.zarr", **(files or {})},
    }
    store._atomic_json(ctx.dir / "manifest.json", manifest)
    return manifest


def _set_local_latest(root: Path, manifest: dict) -> None:
    """latest.json (what the API serves as 'latest') moves to this cycle, never back to an older one."""
    current = store._latest_cycle_id(root)
    if current is None or manifest["cycle"] >= current:
        store._atomic_json(root / "latest.json", {"cycle": manifest["cycle"], "cycle_time_utc": manifest["cycle_time_utc"],
                                                  "issued_at_utc": manifest["issued_at_utc"]})


def _publish(ctx: Cycle, which: str, manifest: dict, finished: bool) -> None:
    """Upload what is waiting for this product (and its manifest). What fails to go up stays queued for the next hour."""
    queue = ctx.live["pending_upload"][which]
    if not ctx.publishing or (ctx.client is None and not cdn_datasets.configured()):
        queue["stable"], queue["changing"] = [], []        # nothing to upload to: do not let the list grow
        return
    if which == "hourly":
        dataset, title, kind, version, run_dir = publish.DATASET, "Hourly HRRR forecast", "gridded-forecast", SCHEMA_VERSION, ctx.dir
        summary = publish.run_summary(manifest)
    else:
        dataset, title, kind, version, run_dir = peak.DATASET, "Peak fire danger (estimate)", "daily-peak", peak.SCHEMA_VERSION, ctx.peak_dir
        summary = publish.run_summary(manifest)
    summary["ready"] = ctx.latest_eligible or finished
    summary["complete"] = finished
    stable, changing = list(dict.fromkeys(queue["stable"])), list(dict.fromkeys(queue["changing"]))
    result = cdn_datasets.try_publish_update(
        dataset, ctx.cycle, run_dir, summary, title=title, kind=kind, schema_version=version, stable=stable,
        changing=changing, complete=finished, client=ctx.client)
    if result is not None:
        # A changing file (a day still filling in) is queued again, as a stable one, when its day completes.
        queue["stable"], queue["changing"] = [], []


def _commit_hour(ctx: Cycle, lead: int, finished: bool = False) -> None:
    """Make an hour that has a work file part of the product: Zarr, manifest, peak product, CDN."""
    live = ctx.live
    fields, lat, lon, meta = job._load_work_file(ctx.work_file(lead))
    ctx.lat, ctx.lon = lat, lon
    ctx.meta = ctx.meta or meta or {"ny": int(lat.shape[0]), "nx": int(lat.shape[1])}
    zarr_dir = ctx.dir / "forecast.zarr"
    queue = live["pending_upload"]["hourly"]
    if not live_store.exists(zarr_dir):
        valid = [int((ctx.cycle_time + timedelta(hours=h)).timestamp() // 3600) for h in ctx.hours]
        live_store.create(zarr_dir, ELEMENTS, len(ctx.hours), lat, lon, np.array(ctx.hours), np.array(valid))
        queue["stable"] += [f"forecast.zarr/{name}" for name in live_store.metadata_files(zarr_dir, [e.name for e in ELEMENTS])]
    index = ctx.hours.index(lead)
    for element in ELEMENTS:
        if element.name in fields:
            queue["stable"].append(f"forecast.zarr/{live_store.write_step(zarr_dir, element.name, index, fields[element.name], element.scale)}")
            if lead not in live["elements"].setdefault(element.name, []):
                live["elements"][element.name].append(lead)
    rows = job._series({lead: fields}, ctx.mask())
    for name, row in rows.items():
        if row[0] is not None:
            live["series"].setdefault(name, {})[str(lead)] = row[0]
    live["written"] = sorted(set(live["written"]) | {lead})
    live["first_hour_at"] = live["first_hour_at"] or _iso(ctx.now)
    live["revision"] += 1
    manifest = _write_hourly_manifest(ctx, finished)
    if ctx.latest_eligible or finished:
        _set_local_latest(ctx.root, manifest)
    _update_peak(ctx, lead, finished)
    _publish(ctx, "hourly", manifest, finished)
    ctx.save()
    job.set_state(ctx.root, "building", ctx.cycle, phase="fetching", hours_total=len(ctx.hours),
                  hours_done=len(ctx.present), hours=ctx.present, label=manifest["label"])


# --- peak fire danger product ----------------------------------------------------------------------

def _update_peak(ctx: Cycle, lead: Optional[int], finished: bool) -> None:
    """Recompute the day(s) the new hour belongs to (every day when finishing) and rewrite the peak manifest."""
    if not ctx.days:
        return
    live = ctx.live
    zarr_dir = ctx.peak_dir / "forecast.zarr"
    queue = live["pending_upload"]["peak"]
    if not live_store.exists(zarr_dir):
        valid = [int(day.window_end_utc.timestamp() // 3600) for day in ctx.days]
        live_store.create(zarr_dir, peak.PEAK_ELEMENTS, len(ctx.days), ctx.lat, ctx.lon, np.arange(len(ctx.days)),
                          np.array(valid), step_dim="day")
        queue["stable"] += [f"forecast.zarr/{name}"
                            for name in live_store.metadata_files(zarr_dir, [e.name for e in peak.PEAK_ELEMENTS])]
    present = ctx.present
    for day in ctx.days:
        if not (finished or lead in day.window_leads):
            continue
        have = [h for h in day.window_leads if h in present]
        if not have:
            continue
        agg = peak.aggregate(day, {h: _load_fields(ctx.work_file(h), peak.SOURCE_FIELDS) for h in have
                                   if ctx.work_file(h).is_file()})
        if not agg:
            continue
        final = peak.day_complete(day, present, finished)
        for name, values in agg.items():
            chunk = live_store.write_step(zarr_dir, name, day.index, values, peak.BY_NAME[name].scale)
            queue["stable" if final else "changing"].append(f"forecast.zarr/{chunk}")
        live["peak_days"][str(day.index)] = sorted(agg)
    manifest = peak.build_manifest(
        ctx.cycle_time, ctx.days, present, {int(k): v for k, v in live["peak_days"].items()}, ctx.meta or {},
        datetime.strptime(live["first_hour_at"] or live["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc),
        ctx.now, finished, live["revision"])
    store._atomic_json(ctx.peak_dir / "manifest.json", manifest)
    if ctx.latest_eligible or finished:
        _set_local_latest(ctx.peak_root, manifest)
    _publish(ctx, "peak", manifest, finished)


# --- finishing a cycle ---------------------------------------------------------------------------------

def _finish(ctx: Cycle) -> dict:
    """Every hour is in (or given up on): NetCDF copy, final manifests, then tidy up."""
    live = ctx.live
    present = ctx.present
    started = time.perf_counter()
    ctx.ensure_grid()
    hours_data: dict[int, dict] = {}
    for lead in present:
        hours_data[lead] = _load_fields(ctx.work_file(lead))
    files = {}
    if hours_data:
        dataset = store.build_dataset(ctx.cycle_time, hours_data, ctx.lat, ctx.lon,
                                      datetime.strptime(live["first_hour_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc))
        store.write_netcdf(dataset, store.nc_path(ctx.cycle, ctx.root))
        files = {"netcdf": f"nc/{PRODUCT}_{ctx.cycle}.nc", "netcdf_bytes": store.nc_path(ctx.cycle, ctx.root).stat().st_size}
    live["revision"] += 1
    manifest = _write_hourly_manifest(ctx, True, files)
    manifest["runtime_sec"] = round((ctx.now - datetime.strptime(live["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds(), 1)
    store._atomic_json(ctx.dir / "manifest.json", manifest)
    _set_local_latest(ctx.root, manifest)
    job.set_state(ctx.root, "publishing", ctx.cycle, label=manifest["label"], hours_total=len(ctx.hours),
                  hours_done=len(present))
    _update_peak(ctx, None, True)
    _publish(ctx, "hourly", manifest, True)
    live["finished_at"] = _iso(ctx.now)
    live["finish_seconds"] = round(time.perf_counter() - started, 2)
    ctx.save()
    shutil.rmtree(ctx.work, ignore_errors=True)
    removed = store.prune(ctx.now, root=ctx.root)
    store.prune(ctx.now, root=ctx.peak_root)
    if removed:
        from services.hrrr_hourly import render as renderer
        renderer.prune_images(store.list_cycles(ctx.root))
    missing = manifest["missing_hours"]
    job.set_state(ctx.root, "partial" if missing else "complete", ctx.cycle, label=manifest["label"],
                  issued_at_utc=manifest["issued_at_utc"], hours=manifest["forecast_hours"], missing_hours=missing,
                  runtime_sec=manifest["runtime_sec"], variables=len(manifest["variables"]), finished_at=job._now_iso())
    controls.record_event({"cycle": ctx.cycle, "outcome": "partial" if missing else "complete",
                           "runtime_sec": manifest["runtime_sec"], "hours": len(manifest["forecast_hours"]),
                           "missing_hours": missing, "mode": "progressive"}, ctx.root)
    return {"action": "built", "cycle": ctx.cycle, "hours": manifest["forecast_hours"], "missing_hours": missing}


# --- the daily forecast's shared HRRR cache --------------------------------------------------------------

def _build_shared_cache(ctx: Cycle) -> None:
    """Once the hours the daily forecast reads (f04-f15 of the 12z run) are in, assemble its HRRR cache file from the same
    work files (services/hrrr_shared_cache.py: a shadow copy unless FORECAST_USE_SHARED_HRRR). Once per cycle, never fatal."""
    if ctx.cycle_time.hour != CYCLE_HOUR or ctx.live.get("shared_cache"):
        return
    from services import hrrr_shared_cache

    if not set(hrrr_shared_cache.required_hours()) <= set(ctx.live["written"]):
        return
    ctx.live["shared_cache"] = True
    ctx.save()
    try:
        logger.info("shared HRRR cache: %s", hrrr_shared_cache.build_shared_cache(ctx.cycle_time, ctx.root))
    except hrrr_shared_cache.SharedCacheUnavailable as error:
        logger.warning("shared HRRR cache not written: %s", error)
    except Exception:
        logger.exception("shared HRRR cache failed")


# --- the scheduler tick ---------------------------------------------------------------------------------

def _advance_locked(ctx: Cycle, fetch_hour: Callable, derive_hour: Callable, hour_ready: Callable, wait: Callable,
                    clock: Callable) -> dict:
    live = ctx.live
    started = clock()
    progressed: list[int] = []
    blocked_by_hrrr = False
    for lead in ctx.hours:
        if lead in live["written"]:
            continue
        if int(live["failed"].get(str(lead), 0)) >= MAX_HOUR_ATTEMPTS:
            continue
        if not ctx.work_file(lead).is_file():
            if not hour_ready(ctx.cycle_time, lead):
                blocked_by_hrrr = True
                break                                  # HRRR publishes hours in order: the later ones are not out either
            try:
                wait(ctx.root)
            except RuntimeError as exc:                # nearly full disk: stop rather than half-write hours
                logger.error("hrrr_hourly %s stopped: %s", ctx.cycle, exc)
                job.set_state(ctx.root, "failed", ctx.cycle, error=str(exc)[:300], finished_at=job._now_iso())
                return {"action": "skip", "reason": str(exc), "cycle": ctx.cycle}
            hour_started = time.perf_counter()
            try:
                _process_hour(ctx, lead, fetch_hour, derive_hour)
            except Exception:
                logger.exception("hrrr_hourly %s f%02d failed (attempt %d)", ctx.cycle, lead,
                                 int(live["failed"].get(str(lead), 0)) + 1)
                live["failed"][str(lead)] = int(live["failed"].get(str(lead), 0)) + 1
                ctx.save()
                continue
            live["work_seconds"] = round(live["work_seconds"] + time.perf_counter() - hour_started, 1)
        try:
            _commit_hour(ctx, lead)
        except Exception:
            logger.exception("hrrr_hourly %s f%02d could not be added to the store", ctx.cycle, lead)
            live["failed"][str(lead)] = int(live["failed"].get(str(lead), 0)) + 1
            ctx.save()
            continue
        progressed.append(lead)
        logger.info("hrrr_hourly %s f%02d is in (%d of %d)", ctx.cycle, lead, len(ctx.present), len(ctx.hours))
        if clock() - started > TICK_BUDGET_SECONDS:
            break
    _build_shared_cache(ctx)
    remaining = [lead for lead in ctx.hours if lead not in live["written"]
                 and int(live["failed"].get(str(lead), 0)) < MAX_HOUR_ATTEMPTS]
    gave_up = bool(remaining) and blocked_by_hrrr and ctx.now - ctx.cycle_time > GIVE_UP_AFTER
    if (not remaining or gave_up) and ctx.present:
        if gave_up:
            logger.warning("hrrr_hourly %s: giving up on hours %s", ctx.cycle, remaining)
        return {**_finish(ctx), "advanced": progressed}
    if not ctx.present:
        job.set_state(ctx.root, "waiting_for_hrrr", ctx.cycle, detail=f"HRRR {ctx.cycle_time:%H}z run: first hour not out yet")
        return {"action": "skip", "reason": "no hour published yet", "cycle": ctx.cycle}
    if progressed:
        ctx.save()
    job.set_state(ctx.root, "building", ctx.cycle, phase="waiting", hours_total=len(ctx.hours), hours_done=len(ctx.present),
                  hours=ctx.present, detail="waiting for the next HRRR hour")
    return {"action": "advanced" if progressed else "skip", "cycle": ctx.cycle, "advanced": progressed,
            "reason": None if progressed else "waiting for the next hour"}


def advance(now: Optional[datetime] = None, hours=FORECAST_HOURS, root: Optional[Path] = None,
            peak_root: Optional[Path] = None, fetch_hour: Callable = fetch.fetch_hour, derive_hour: Optional[Callable] = None,
            hour_ready: Callable = lambda run, lead: fetch.cycle_ready(run, lead), wait: Optional[Callable] = None,
            clock: Callable = time.monotonic, client=None, publishing: bool = True,
            candidates: Optional[list[datetime]] = None) -> dict:
    """One scheduler tick: bring the newest cycle that has work up to date, publishing each hour it adds."""
    from services.hrrr_hourly import derive

    derive_hour = derive_hour or derive.derive_hour
    wait = wait or (lambda root_: job.wait_for_priority_work(root_, include_forecast=False))
    root = Path(root or root_dir())
    peak_root = Path(peak_root or peak.root_dir())
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if controls.is_paused(root):
        serving = job._serving_cycle(root) or ""
        if job.read_status(root).get("state") != "paused":
            job.set_state(root, "paused", serving, detail="paused by an admin")
        return {"action": "skip", "reason": "paused", "cycle": serving}
    if job.read_status(root).get("state") == "paused":
        job.set_state(root, "complete", job._serving_cycle(root) or "", detail="resumed")
    candidates = candidates or fetch.cycle_candidates(now, BACKFILL_CYCLES, earliest=fetch.FIRST_HOUR_LAG)
    first_skip: Optional[dict] = None
    built: Optional[str] = None
    for candidate in candidates:
        cycle = cycle_id(candidate)
        existing = store.read_manifest(cycle, root)
        if existing and existing.get("complete", True):
            built = built or cycle
            continue
        root.mkdir(parents=True, exist_ok=True)
        guard = open(root / ".build.lock", "a+")
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            guard.close()
            return {"action": "skip", "reason": "another build is running", "cycle": cycle}
        try:
            live = _read_live(root / cycle)
            if live is None:
                if existing:        # a cycle with a manifest but no live state: built another way; leave it alone
                    continue
                live = _new_live(hours, now)
            ctx = Cycle(root, peak_root, candidate, now, live, client=client, publishing=publishing)
            result = _advance_locked(ctx, fetch_hour, derive_hour, hour_ready, wait, clock)
        finally:
            guard.close()
        if result["action"] in ("built", "advanced"):
            return result
        first_skip = first_skip or result
    return first_skip or {"action": "skip", "reason": "already built", "cycle": built or cycle_id(candidates[0])}

"""
The fire-outlook worker: one isolated run of the seven-day outlook, with the plan's operating rules.

Not part of the API scheduler (nothing here is registered in core/scheduler.py). Stages are injected via
`Adapters` so the pipeline is testable and acquisition can be swapped in when it exists.

Rules enforced here:
  * Targets 00Z and 12Z cycles; `run_once` is cheap when there is nothing to do (cron every 30 min) and
    idempotent: a cycle already completed and checksum-verified on disk is not recomputed (restart/resume).
  * A main run is only written when ALL members cover ALL windows (driver raises IncompleteMember otherwise).
  * On any failure the previous complete run stays current: runs are immutable, the local and CDN `latest`
    pointers only move after a run is written, verified and (when configured) uploaded.
  * Upload order is files -> manifest -> latest (services/cdn_datasets.publish_run). An upload failure is
    recorded as a failure of the `publish` stage; the verified local run is kept and retried next time.
  * Status (stage, cycle age, errors, runtime, peak memory, source coverage, 4-hour budget) is written
    atomically to status.json, even on failure.
  * Defaults: 1 thread, 8 GiB address-space ceiling, 7 days of public runs, 30 days of caches (env-configurable).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from services.fire_outlook import assemble, contours, driver, layers, verification

logger = logging.getLogger(__name__)

CYCLE_HOURS = (0, 12)
BUDGET_HOURS = 4.0
STAGES = ("checking_inputs", "conditioning", "calculating", "contours", "writing", "publishing", "pruning")


@dataclass
class Config:
    root: Path
    memory_limit_gb: float = 8.0
    threads: int = 1
    public_keep_days: int = 7
    cache_keep_days: int = 30
    publish: bool = True

    @classmethod
    def from_env(cls, root: Optional[Path] = None) -> "Config":
        """publish defaults to FALSE from the environment: the CDN dataset is public, so the staff-only pilot must not upload to
        it. Set SMF_FIRE_OUTLOOK_PUBLISH=true when the outlook is meant to be public (together with SMF_FIRE_OUTLOOK_PUBLIC)."""
        env = os.getenv
        return cls(
            root=Path(root or env("SMF_FIRE_OUTLOOK_DIR", "data/fire-outlook")),
            memory_limit_gb=float(env("SMF_FIRE_OUTLOOK_MEMORY_LIMIT_GB", "8")),
            threads=int(env("SMF_FIRE_OUTLOOK_THREADS", "1")),
            public_keep_days=int(env("SMF_FIRE_OUTLOOK_PUBLIC_KEEP_DAYS", "7")),
            cache_keep_days=int(env("SMF_FIRE_OUTLOOK_CACHE_KEEP_DAYS", "30")),
            publish=env("SMF_FIRE_OUTLOOK_PUBLISH", "false").strip().lower() in {"1", "true", "yes", "on"},
        )

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"


@dataclass
class Adapters:
    """Everything the worker needs from the outside world."""
    inputs_ready: Callable[[datetime], tuple[bool, dict]]          # -> (ready, {coverage..., 'available_utc'?})
    members: Callable[[datetime], Iterable]                        # normalized hourly datasets, one at a time
    n_members: int
    conditioning: Callable[[datetime], tuple[dict, dict]]          # -> (initial state, quality)
    rules: Callable[[], tuple[Callable, Optional[Callable], dict]] # -> (categorize, dampen, rule_info)
    grid: dict
    source_cycles: Callable[[datetime], dict]
    driver_params: dict = field(default_factory=dict)
    # () -> [{'station_id', 'lat', 'lon'}]: stations to verify against (RAWS with 10-h fuel moisture). Optional:
    # without it no verification records are exported, but the outlook itself is unaffected.
    stations: Optional[Callable[[], list]] = None
    # (cycle, stations) -> {var: (28, S)} NBM window statistics, or None. Baseline for verification only; optional and
    # never fatal (the outlook does not depend on it).
    reference: Optional[Callable[[datetime, list], Optional[dict]]] = None


def latest_cycle(now: datetime) -> datetime:
    now = now.astimezone(timezone.utc)
    hour = max(h for h in CYCLE_HOURS if h <= now.hour)
    return now.replace(hour=hour, minute=0, second=0, microsecond=0)


def peak_rss_mb() -> Optional[float]:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / 1024.0, 1) if os.uname().sysname != "Darwin" else round(peak / 1048576.0, 1)
    except Exception:  # noqa: BLE001 - Windows has no resource module
        try:
            import psutil
            info = psutil.Process().memory_info()
            return round(getattr(info, "peak_wset", info.rss) / 1048576.0, 1)  # Windows exposes the true peak
        except Exception:  # noqa: BLE001
            return None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _iso(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class Status:
    def __init__(self, config: Config, now: Callable[[], datetime]):
        self.config, self._now = config, now
        prior = {}
        if config.status_path.exists():
            try:
                prior = json.loads(config.status_path.read_text(encoding="utf-8"))
            except ValueError:
                prior = {}
        self.data: dict = {"last_complete": prior.get("last_complete"), "errors": []}

    def update(self, **fields) -> None:
        self.data.update(fields, updated_utc=_iso(self._now()))
        cycle = self.data.get("cycle")
        if cycle:
            self.data["cycle_age_hours"] = round(
                (self._now() - datetime.fromisoformat(cycle.replace("Z", "+00:00"))).total_seconds() / 3600, 2)
        _atomic_json(self.config.status_path, self.data)

    def error(self, stage: str, error: BaseException) -> None:
        self.data["errors"].append({"stage": stage, "type": type(error).__name__, "message": str(error), "at": _iso(self._now())})


def _memory_limit(limit_gb: float, func: Callable, *args):
    """Reversible address-space ceiling (same approach as services/forecast_v1_job.py)."""
    try:
        import resource
    except ImportError:  # Windows dev machines
        return func(*args)
    if limit_gb <= 0:
        raise ValueError("memory limit must be greater than zero")
    previous = resource.getrlimit(resource.RLIMIT_AS)
    hard = previous[1]
    requested = int(limit_gb * 1024 ** 3)
    effective = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (effective, hard))
    except (OSError, ValueError):
        logger.warning("memory ceiling unsupported on this platform")
        return func(*args)
    try:
        return func(*args)
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)


def completed_run(config: Config, cycle: datetime) -> Optional[Path]:
    run = config.runs_dir / assemble.run_id_for(cycle)
    if (run / "manifest.json").is_file() and assemble.verify_checksums(run):
        return run
    return None


def _published_ids(config: Config) -> set:
    path = config.root / "published.json"
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("run_ids", []))
    except (OSError, ValueError):
        return set()


def _record_published(config: Config, run_id: str) -> None:
    ids = sorted(_published_ids(config) | {run_id})[-200:]
    _atomic_json(config.root / "published.json", {"run_ids": ids})


def prune(config: Config, now: datetime) -> dict:
    """Remove public runs older than public_keep_days (never the current latest) and old cache entries."""
    removed = {"runs": [], "cache": 0}
    latest = None
    if config.latest_path.exists():
        latest = json.loads(config.latest_path.read_text(encoding="utf-8")).get("run_id")
    if config.runs_dir.exists():
        for run in config.runs_dir.iterdir():
            if not run.is_dir() or run.name.startswith(".") or run.name == latest:
                continue
            try:
                stamp = datetime.strptime(run.name, "%Y%m%dT%HZ").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if now - stamp > timedelta(days=config.public_keep_days):
                shutil.rmtree(run, ignore_errors=True)
                removed["runs"].append(run.name)
    if config.cache_dir.exists():
        cutoff = now.timestamp() - config.cache_keep_days * 86400
        for entry in config.cache_dir.iterdir():
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
                removed["cache"] += 1
    return removed


def _summary(manifest: dict) -> dict:
    return {"id": manifest["run_id"], "time_utc": manifest["cycle_time_utc"], "cycle_time_utc": manifest["cycle_time_utc"],
            "label": manifest["label"], "member_count": manifest["member_count"], "windows": len(manifest["windows"]),
            "stale_after_hours": manifest["stale_after_hours"], "conditioning_level": manifest["conditioning"].get("level")}


def _publish(run: Path, manifest: dict, config: Config, publish_fn: Optional[Callable]) -> Optional[dict]:
    from services import cdn_datasets
    if publish_fn is None:
        if not cdn_datasets.configured():
            return None
        publish_fn = cdn_datasets.publish_run
    return publish_fn(assemble.DATASET_ID, manifest["run_id"], run, _summary(manifest), title="Seven-day fire outlook",
                      kind="gridded-forecast", schema_version=assemble.SCHEMA_VERSION,
                      keep_days=config.public_keep_days)


def run_cycle(cycle: datetime, adapters: Adapters, config: Config, *, publish_fn: Optional[Callable] = None,
              now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    """Run one cycle. Returns {'ok', 'state', 'run_id', ...}. Never raises for pipeline failures: they are recorded
    in status.json and returned, and the previous complete run stays current."""
    started = now_fn()
    t0 = time.monotonic()
    status = Status(config, now_fn)
    status.update(state="running", stage="checking_inputs", cycle=_iso(cycle), started_utc=_iso(started),
                  threads=config.threads, memory_limit_gb=config.memory_limit_gb, errors=[])
    stage = "checking_inputs"

    def at(name: str) -> None:
        nonlocal stage
        stage = name
        status.update(stage=name, runtime_s=round(time.monotonic() - t0, 1), peak_rss_mb=peak_rss_mb())

    try:
        existing = completed_run(config, cycle)
        if existing is not None and (not config.publish or existing.name in _published_ids(config)):
            status.update(state="complete", stage="done", run_id=existing.name, note="already complete on disk")
            return {"ok": True, "state": "already_complete", "run_id": existing.name}

        if existing is None:
            ready, coverage = adapters.inputs_ready(cycle)
            status.update(source_coverage=coverage)
            if not ready:
                status.update(state="waiting", stage="checking_inputs", note="inputs for this cycle are not complete yet")
                return {"ok": True, "state": "waiting_for_inputs", "run_id": None}
            available = coverage.get("available_utc")

            at("conditioning")
            initial, quality = adapters.conditioning(cycle)
            categorize, dampen, rule_info = adapters.rules()
            if quality.get("level") == "emc_seed":
                logger.warning("fire-outlook cycle %s is using an EMC seed (no conditioned history)", cycle)

            at("calculating")
            params = {"precip_interval_hours": 3, "wind_factor": 1.0, "snow_swe_mm": 1.0, **adapters.driver_params}
            grid_shape = (adapters.grid["height"], adapters.grid["width"])
            stations, cells = [], []
            station_error = None
            if adapters.stations is not None:
                try:
                    stations, cells = verification.station_cells(adapters.stations(), adapters.grid)
                except Exception as error:  # noqa: BLE001 - no station list must never block the outlook itself
                    logger.exception("fire-outlook station list failed; no verification rows for this run")
                    station_error = f"{type(error).__name__}: {error}"
            result = _memory_limit(
                config.memory_limit_gb,
                lambda: driver.run_ensemble(adapters.members(cycle), cycle, adapters.n_members, grid_shape,
                                            lambda ds: initial, categorize, dampen, station_cells=cells or None,
                                            **params))

            at("contours")
            n = int(result["n_members"])
            files = contours.contour_files(assemble._counts(result["agreement_elevated"], n),
                                           assemble._counts(result["agreement_critical"], n), n, adapters.grid)
            files.update(layers.layer_files(result, adapters.grid, wind_factor=params.get("wind_factor", 1.0)))

            at("writing")
            runtime = {"threads": config.threads, "memory_limit_gb": config.memory_limit_gb,
                       "peak_rss_mb": peak_rss_mb(), "calculation_runtime_s": round(time.monotonic() - t0, 1),
                       "input_available_utc": None if available is None else str(available)}
            run = assemble.write_run(
                config.runs_dir, cycle, result, grid=adapters.grid, source_cycles=adapters.source_cycles(cycle),
                rule_info=rule_info, conditioning_quality=quality, coverage=coverage,
                driver_params=params, runtime=runtime, now=now_fn(), extra_files=files,
                layers=layers.manifest_layers())
            if not assemble.verify_checksums(run):
                shutil.rmtree(run, ignore_errors=True)
                raise RuntimeError("written run failed checksum verification")
            _atomic_json(config.latest_path, {"run_id": run.name, "cycle_time_utc": _iso(cycle), "written_utc": _iso(now_fn())})
            # Forecast-side verification rows. Never fatal: a problem here is recorded, not allowed to block the outlook.
            try:
                manifest_now = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
                reference, reference_error = None, None
                if adapters.reference is not None and stations:
                    try:
                        reference = adapters.reference(cycle, stations)
                        if reference is None:
                            reference_error = "NBM baseline unavailable"
                    except Exception as error:  # noqa: BLE001 - baseline only
                        logger.exception("NBM baseline failed for %s", run.name)
                        reference_error = f"{type(error).__name__}: {error}"
                path = verification.write_forecast_side(config.root, manifest_now, result, stations, cells, reference)
                status.update(verification_export={"stations": len(stations), "file": None if path is None else path.name,
                                                   "nbm": reference is not None,
                                                   **({"nbm_error": reference_error} if reference_error else {}),
                                                   **({"error": station_error} if station_error else {})})
            except Exception as error:  # noqa: BLE001
                logger.exception("verification export failed for %s", run.name)
                status.update(verification_export={"stations": len(stations), "error": f"{type(error).__name__}: {error}"})
        else:
            run, available = existing, None

        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        published = None
        if config.publish:
            at("publishing")
            published = _publish(run, manifest, config, publish_fn)
            if published is not None:
                _record_published(config, run.name)

        at("pruning")
        pruned = prune(config, now_fn())
        finished = now_fn()
        within = None
        if available is not None:
            av = available if isinstance(available, datetime) else datetime.fromisoformat(str(available).replace("Z", "+00:00"))
            within = (finished - av) <= timedelta(hours=BUDGET_HOURS)
        status.update(state="complete", stage="done", run_id=run.name, completed_utc=_iso(finished),
                      runtime_s=round(time.monotonic() - t0, 1), peak_rss_mb=peak_rss_mb(), published=bool(published),
                      within_4h_budget=within, pruned=pruned,
                      last_complete={"run_id": run.name, "cycle_time_utc": _iso(cycle)})
        return {"ok": True, "state": "complete", "run_id": run.name, "published": bool(published), "pruned": pruned}
    except Exception as error:  # noqa: BLE001 - recorded; the previous complete run is untouched
        logger.exception("fire-outlook %s failed in stage %s", cycle, stage)
        status.error(stage, error)
        status.update(state="failed", stage=stage, runtime_s=round(time.monotonic() - t0, 1), peak_rss_mb=peak_rss_mb())
        return {"ok": False, "state": "failed", "stage": stage, "error": str(error), "run_id": None}


def run_once(adapters: Adapters, config: Config, *, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
             **kwargs) -> dict:
    """Cron entry (every 30 minutes): process the latest 00Z/12Z cycle if it is not already done."""
    return run_cycle(latest_cycle(now_fn()), adapters, config, now_fn=now_fn, **kwargs)

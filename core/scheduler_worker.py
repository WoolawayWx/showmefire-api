"""The heavy scheduler worker: APScheduler without the web server.

    python -m core.scheduler_worker --groups heavy        # run the long jobs (forecast_v1, HRRR, RTMA, archives...)
    python -m core.scheduler_worker --groups heavy --list # print the jobs that would run, then exit

An API deploy restarts only the API container, so these jobs keep running. See core/job_groups.py for which jobs run
where and docs/scheduler_worker.md for the compose service and the deploy order.

It does what `main.py`'s lifespan does for the scheduler and nothing else: logging + Discord error alerts, the idempotent
database setup (under a file lock, so it cannot race the API's), the model-pointer env vars, then the scheduler, a heartbeat
file every minute, and a graceful stop: on SIGTERM no new job starts and running jobs get SMF_SCHEDULER_STOP_WAIT_SECONDS
(default 25 min, keep the container's stop_grace_period above it) to finish before the process exits.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from apscheduler.events import (
    EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MAX_INSTANCES, EVENT_JOB_MISSED, EVENT_JOB_SUBMITTED,
)

from core.job_groups import ALL_GROUPS, HEAVY, groups_from_env

logger = logging.getLogger("scheduler_worker")

HEARTBEAT_SECONDS = 60
DEFAULT_STOP_WAIT_SECONDS = 25 * 60


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return os.getenv("SMF_GIT_SHA", "unknown")


class RunningJobs:
    """The set of jobs executing right now, from APScheduler's events (used for the heartbeat and the graceful stop)."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def attach(self, scheduler) -> None:
        scheduler.add_listener(self._submitted, EVENT_JOB_SUBMITTED)
        scheduler.add_listener(self._finished, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES)

    def _submitted(self, event) -> None:
        self._counts[event.job_id] = self._counts.get(event.job_id, 0) + 1

    def _finished(self, event) -> None:
        if event.code == EVENT_JOB_MAX_INSTANCES or event.code == EVENT_JOB_MISSED:
            return   # these never started a run, so there is nothing to decrement
        count = self._counts.get(event.job_id, 0) - 1
        if count > 0:
            self._counts[event.job_id] = count
        else:
            self._counts.pop(event.job_id, None)

    @property
    def ids(self) -> list[str]:
        return sorted(self._counts)


def describe(scheduler, groups, running: RunningJobs, started_at: str) -> dict:
    jobs = []
    for job in scheduler.get_jobs():
        next_run = getattr(job, "next_run_time", None)
        jobs.append({"id": job.id, "next_run_at": next_run.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if next_run else None})
    return {
        "groups": sorted(groups), "host": socket.gethostname(), "pid": os.getpid(), "git_sha": _git_sha(),
        "started_at": started_at, "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "running_jobs": running.ids, "jobs": jobs,
        "forecast_v1_enabled": os.getenv("SMF_FORECAST_V1_ENABLED", "false").lower() == "true",
    }


def bootstrap() -> None:
    """Logging + Discord error alerts, then the same database setup the API does (idempotent, under the startup lock)."""
    from core.logging_setup import configure_logging

    configure_logging(log_file="api.log")   # the admin Logs page reads scheduler output from api.log
    from services.system_alerts import install_discord_log_handler

    install_discord_log_handler()
    from core.database import init_database
    from core.field_store import ensure_field_schema
    from core.startup_lock import startup_lock
    from forecast_v1.repository import ensure_schema as ensure_forecast_v1_schema

    with startup_lock():
        init_database()
        ensure_forecast_v1_schema()
        ensure_field_schema()
    try:
        from models.shadow_bundles import reapply_active_env_vars

        reapply_active_env_vars()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to reapply guarded-shadow active bundle pointers (non-fatal): %s", exc)


async def run(groups: set[str]) -> None:
    from core import scheduler as scheduler_module
    from core.executors import shutdown_process_pool

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    scheduler = scheduler_module.create_scheduler()
    running = RunningJobs()
    running.attach(scheduler)
    if HEAVY in groups:
        from services.forecast_v1_job import reconcile_orphaned_run

        logger.info("Start-up reconciliation of an interrupted forecast_v1 run: %s", reconcile_orphaned_run())
    scheduler_module.start_scheduler_jobs(scheduler, groups)

    from services.scheduler_heartbeat import write_heartbeat

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))

    async def heartbeat_loop() -> None:
        while True:
            try:
                write_heartbeat(describe(scheduler, groups, running, started_at))
            except Exception:  # noqa: BLE001
                logger.warning("Could not write the scheduler heartbeat", exc_info=True)
            await asyncio.sleep(HEARTBEAT_SECONDS)

    beat = asyncio.create_task(heartbeat_loop())
    logger.info("Scheduler worker running groups %s", ",".join(sorted(groups)))
    await stop.wait()

    scheduler.pause()
    await drain(running, int(os.getenv("SMF_SCHEDULER_STOP_WAIT_SECONDS", str(DEFAULT_STOP_WAIT_SECONDS))))
    beat.cancel()
    scheduler.shutdown(wait=False)
    shutdown_process_pool()


async def drain(running: RunningJobs, wait_seconds: float, poll_seconds: float = 2.0) -> bool:
    """After a stop request: wait for the running jobs to finish, up to `wait_seconds`. True when none are left."""
    logger.info("Stop requested: no new jobs; waiting for %s", running.ids or "no running jobs")
    deadline = time.monotonic() + wait_seconds
    while running.ids and time.monotonic() < deadline:
        await asyncio.sleep(poll_seconds)
    if running.ids:
        logger.warning("Stopping with jobs still running: %s", running.ids)
    return not running.ids


def list_jobs(groups: set[str]) -> list[str]:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from core.scheduler import register_jobs
    from pytz import timezone as tz

    registered = register_jobs(AsyncIOScheduler(timezone=tz("America/Chicago")), groups)
    return sorted(registered.registered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", default=None, help=f"comma list of {','.join(ALL_GROUPS)} (default SMF_SCHEDULER_GROUPS, else heavy)")
    parser.add_argument("--list", action="store_true", help="print the job ids this worker would run and exit")
    args = parser.parse_args(argv)
    groups = groups_from_env(args.groups if args.groups is not None else os.getenv("SMF_SCHEDULER_GROUPS", HEAVY))
    if args.list:
        for job_id in list_jobs(groups):
            print(job_id)
        return 0
    os.environ.setdefault("SMF_METRICS_ENABLED", "0")   # the API owns the metrics snapshots
    bootstrap()
    asyncio.run(run(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())

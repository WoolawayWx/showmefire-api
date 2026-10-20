"""A record of every scheduled job and cron script run, for the admin Logs page.

Before this, the only way to tell whether a job had run, or failed, was to
search the container's stdout. Each run now leaves one row in `job_runs`:
when it started, how long it took, whether it worked, and the first error
it logged.

Two writers:
  - track_scheduler_jobs() wraps every APScheduler job. Most jobs catch
    their own exceptions and log an error instead of raising, so a run is
    marked "error" when it raised *or* logged at ERROR, and "warning" when
    it logged a warning. Records are counted per run through a contextvar
    (asyncio.to_thread copies it, so work a job hands to a thread counts).
  - `python -m services.job_runs record ...`, which scripts/run_job.sh calls
    after each cron script.

Rows older than RETENTION_DAYS are pruned by the daily log-maintenance job.
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import functools
import inspect
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from core.database import get_db_path

logger = logging.getLogger(__name__)

RETENTION_DAYS = 14
MESSAGE_LIMIT = 500
STATUSES = ("ok", "warning", "error")

_table_ready = False


@dataclass
class _RunCounter:
    warnings: int = 0
    errors: int = 0
    first_error: Optional[str] = None
    first_warning: Optional[str] = None


_current_run: contextvars.ContextVar[Optional[_RunCounter]] = contextvars.ContextVar("smf_job_run", default=None)


class _RunCountingHandler(logging.Handler):
    """Counts warnings and errors logged while a tracked job is running."""

    def __init__(self):
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        counter = _current_run.get()
        if counter is None or record.name == __name__:
            return
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        if record.levelno >= logging.ERROR:
            counter.errors += 1
            counter.first_error = counter.first_error or message
        else:
            counter.warnings += 1
            counter.first_warning = counter.first_warning or message


_counting_handler: Optional[_RunCountingHandler] = None


def _connect() -> sqlite3.Connection:
    global _table_ready
    conn = sqlite3.connect(get_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    if not _table_ready:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'scheduler',
                started_at REAL NOT NULL,
                duration_s REAL,
                status TEXT NOT NULL,
                warnings INTEGER NOT NULL DEFAULT 0,
                errors INTEGER NOT NULL DEFAULT 0,
                message TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_runs_job_started ON job_runs(job, started_at)")
        conn.commit()
        _table_ready = True
    return conn


def record_run(
    job: str,
    status: str,
    started_at: float,
    duration_s: Optional[float],
    *,
    source: str = "scheduler",
    warnings: int = 0,
    errors: int = 0,
    message: Optional[str] = None,
) -> None:
    """Store one run. Never raises: losing a status row must not break the job itself."""
    if status not in STATUSES:
        status = "error"
    if message:
        message = " ".join(str(message).split())[:MESSAGE_LIMIT]
    try:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO job_runs (job, source, started_at, duration_s, status, warnings, errors, message)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job, source, started_at, duration_s, status, warnings, errors, message),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.debug("Could not record run of %s: %s", job, exc)


def _finish(job: str, started: float, t0: float, counter: _RunCounter, exc: Optional[BaseException]) -> Dict[str, Any]:
    duration = round(time.perf_counter() - t0, 3)
    if exc is not None:
        status, message = "error", f"{type(exc).__name__}: {exc}"
        counter.errors += 1
    elif counter.errors:
        status, message = "error", counter.first_error
    elif counter.warnings:
        status, message = "warning", counter.first_warning
    else:
        status, message = "ok", None
    return dict(job=job, status=status, started_at=started, duration_s=duration, source="scheduler",
                warnings=counter.warnings, errors=counter.errors, message=message)


def tracked(job: str, func: Callable) -> Callable:
    """Wrap a scheduler job function so each run is recorded. Keeps it sync or async as it was."""
    if getattr(func, "_smf_tracked", False):
        return func

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def run_async(*args, **kwargs):
            counter = _RunCounter()
            token = _current_run.set(counter)
            started, t0, error = time.time(), time.perf_counter(), None
            try:
                return await func(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                _current_run.reset(token)
                row = _finish(job, started, t0, counter, error)
                await asyncio.to_thread(lambda: record_run(**row))
        wrapper = run_async
    else:
        @functools.wraps(func)
        def run_sync(*args, **kwargs):
            counter = _RunCounter()
            token = _current_run.set(counter)
            started, t0, error = time.time(), time.perf_counter(), None
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                _current_run.reset(token)
                record_run(**_finish(job, started, t0, counter, error))
        wrapper = run_sync

    wrapper._smf_tracked = True  # type: ignore[attr-defined]
    return wrapper


def track_scheduler_jobs(scheduler) -> int:
    """Wrap every job already added to `scheduler` (call before scheduler.start()). Returns the count."""
    global _counting_handler
    if _counting_handler is None:
        _counting_handler = _RunCountingHandler()
        logging.getLogger().addHandler(_counting_handler)
    count = 0
    for job in scheduler.get_jobs():
        if getattr(job.func, "_smf_tracked", False):
            continue
        scheduler.modify_job(job.id, func=tracked(job.id, job.func))
        count += 1
    return count


def _iso(ts: Optional[float]) -> Optional[str]:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z") if ts else None


def _row(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "started_at": _iso(row["started_at"]),
        "duration_s": row["duration_s"],
        "status": row["status"],
        "warnings": row["warnings"],
        "errors": row["errors"],
        "message": row["message"],
    }


def job_summaries(window_hours: float = 24) -> Dict[str, Dict[str, Any]]:
    """Per job: the latest run, the last failure, and run/failure counts over the window."""
    since = time.time() - window_hours * 3600
    try:
        conn = _connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Job run history unavailable: %s", exc)
        return {}
    try:
        latest = conn.execute(
            """
            SELECT r.* FROM job_runs r
            JOIN (SELECT job, MAX(started_at) AS started_at FROM job_runs GROUP BY job) m
              ON m.job = r.job AND m.started_at = r.started_at
            """
        ).fetchall()
        failures = conn.execute(
            """
            SELECT r.* FROM job_runs r
            JOIN (SELECT job, MAX(started_at) AS started_at FROM job_runs WHERE status = 'error' GROUP BY job) m
              ON m.job = r.job AND m.started_at = r.started_at
            """
        ).fetchall()
        counts = conn.execute(
            """
            SELECT job, source, COUNT(*) AS runs,
                   SUM(status = 'error') AS failed,
                   SUM(status = 'warning') AS warned,
                   AVG(duration_s) AS avg_duration_s,
                   MAX(duration_s) AS max_duration_s
            FROM job_runs WHERE started_at >= ? GROUP BY job, source
            """,
            (since,),
        ).fetchall()
    finally:
        conn.close()

    out: Dict[str, Dict[str, Any]] = {}
    for row in latest:
        out[row["job"]] = {"job": row["job"], "source": row["source"], "last": _row(row),
                           "last_failure": None, "window": None}
    for row in failures:
        if row["job"] in out:
            out[row["job"]]["last_failure"] = _row(row)
    for row in counts:
        if row["job"] in out:
            out[row["job"]]["window"] = {
                "hours": window_hours,
                "runs": row["runs"],
                "failed": row["failed"] or 0,
                "warned": row["warned"] or 0,
                "avg_duration_s": round(row["avg_duration_s"], 3) if row["avg_duration_s"] is not None else None,
                "max_duration_s": row["max_duration_s"],
            }
    return out


def recent_runs(job: str, limit: int = 50) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM job_runs WHERE job = ? ORDER BY started_at DESC LIMIT ?", (job, max(1, min(limit, 500)))
        ).fetchall()
    finally:
        conn.close()
    return [_row(row) for row in rows]


def prune_runs(retention_days: int = RETENTION_DAYS) -> int:
    conn = _connect()
    try:
        cursor = conn.execute("DELETE FROM job_runs WHERE started_at < ?", (time.time() - retention_days * 86400,))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Record one cron script run for the admin Logs page.")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("job")
    rec.add_argument("--started", type=float, required=True, help="start time, Unix seconds")
    rec.add_argument("--exit-code", type=int, required=True)
    rec.add_argument("--message", default="")
    rec.add_argument("--warnings", type=int, default=0)
    args = parser.parse_args()

    duration = max(0.0, time.time() - args.started)
    status = "error" if args.exit_code else ("warning" if args.warnings else "ok")
    message = args.message.strip() or (f"exit code {args.exit_code}" if args.exit_code else None)
    record_run(args.job, status, args.started, round(duration, 1), source="cron",
               warnings=args.warnings, errors=1 if args.exit_code else 0, message=message)


if __name__ == "__main__":
    _cli()

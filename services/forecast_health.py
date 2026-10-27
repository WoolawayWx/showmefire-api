"""Health ledger for the scheduled forecast runs (forecasts_12z, forecasts_09z).

The forecasts run as cron shell scripts in their own processes, so the Discord
log handler installed in main.py never sees them. Everything they (and the
Python they call) want staff to know goes through here instead:

  - ``record(job, stage, status, message)`` writes one stage result to
    ``data/forecast_health.json`` and, for ``degraded``/``failed``, sends a
    staff Discord alert (services.system_alerts, the "system_issue" type).
  - ``degraded`` means the run carried on with a substitute (older fuel
    observations, HRRR not confirmed complete, a skipped optional step);
    ``failed`` means a step stopped it.
  - ``check_overdue`` is the watchdog, run by the scheduler: a run that never
    started or never finished by its deadline is alerted once per day.

Shell scripts use the CLI: ``python -m services.forecast_health begin|stage|finish``.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
STATE_PATH = Path(os.getenv("FORECAST_HEALTH_PATH") or PROJECT_DIR / "data" / "forecast_health.json")

logger = logging.getLogger(__name__)

SEVERITY = {"ok": 0, "degraded": 1, "failed": 2}
HISTORY_RUNS = 14

# job id (matches the run_job.sh id) -> UTC hour:minute by which a run should have finished.
# 12z starts ~13:10Z (waits for HRRR up to 14:30Z) and takes ~1 h; 09z starts 12:05Z.
EXPECTED_BY_UTC = {
    "forecasts_09z": (15, 30),
    "forecasts_12z": (17, 30),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def current_job() -> str:
    return os.getenv("SMF_FORECAST_JOB") or "forecast"


@contextmanager
def _locked_state() -> Iterator[dict]:
    """Read-modify-write the state file under an exclusive lock (cron, scheduler and API all touch it)."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {}
        state.setdefault("jobs", {})
        yield state
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, STATE_PATH)


def _new_run(now: datetime) -> dict:
    return {"started": _iso(now), "finished": None, "status": "running", "stages": {}}


def _run_status(run: dict) -> str:
    worst = max((SEVERITY[s["status"]] for s in run["stages"].values()), default=0)
    return {0: "ok", 1: "degraded", 2: "failed"}[worst]


def begin_run(job: Optional[str] = None, now: Optional[datetime] = None) -> None:
    job, now = job or current_job(), now or _now()
    with _locked_state() as state:
        entry = state["jobs"].setdefault(job, {"run": None, "history": []})
        _archive(entry)
        entry["run"] = _new_run(now)


def _archive(entry: dict) -> None:
    if entry.get("run"):
        entry["history"] = (entry.get("history", []) + [entry["run"]])[-HISTORY_RUNS:]


def record(stage: str, status: str, message: str, *, job: Optional[str] = None, detail: str = "",
           alert: bool = True, now: Optional[datetime] = None) -> None:
    """Record one stage result. Never raises: health reporting must not break a forecast."""
    if status not in SEVERITY:
        raise ValueError(f"status must be one of {sorted(SEVERITY)}")
    job, now = job or current_job(), now or _now()
    try:
        with _locked_state() as state:
            entry = state["jobs"].setdefault(job, {"run": None, "history": []})
            if entry.get("run") is None or entry["run"].get("finished"):
                _archive(entry)
                entry["run"] = _new_run(now)
            run = entry["run"]
            run["stages"][stage] = {"status": status, "message": message[:500], "detail": detail[:1500],
                                    "at": _iso(now)}
            run["status"] = "running" if not run.get("finished") else _run_status(run)
    except Exception:  # noqa: BLE001
        logger.warning("Could not write forecast health state", exc_info=True)
    if status != "ok" and alert:
        _alert(job, stage, status, message, detail)


def finish_run(job: Optional[str] = None, now: Optional[datetime] = None) -> str:
    """Close the current run and return its overall status. Each degraded/failed stage was already alerted
    when it was recorded, so this sends nothing; it is what lets the watchdog tell "finished" from "hung"."""
    job, now = job or current_job(), now or _now()
    with _locked_state() as state:
        entry = state["jobs"].setdefault(job, {"run": None, "history": []})
        run = entry.get("run") or _new_run(now)
        entry["run"] = run
        run["finished"] = _iso(now)
        run["status"] = _run_status(run)
        return run["status"]


def _alert(job: str, stage: str, status: str, message: str, detail: str = "") -> None:
    try:
        from services.system_alerts import notify_system_issue

        verb = "FAILED" if status == "failed" else "degraded"
        notify_system_issue(f"forecast:{job}", f"{stage} {verb}: {message}", detail=detail or message)
    except Exception:  # noqa: BLE001
        logger.warning("Could not send forecast health alert", exc_info=True)


def summary(now: Optional[datetime] = None) -> dict:
    """What GET /health/forecast returns: the latest run of each job plus whether it is overdue."""
    now = now or _now()
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    jobs = {}
    for job, deadline in EXPECTED_BY_UTC.items():
        entry = (state.get("jobs") or {}).get(job) or {}
        jobs[job] = {"latest": entry.get("run"), "recent": [r["status"] for r in entry.get("history", [])][-7:],
                     "overdue": overdue_reason(job, entry, now) is not None, "expected_by_utc": "%02d:%02d" % deadline}
    worst = max((SEVERITY.get((j["latest"] or {}).get("status"), 0) for j in jobs.values()), default=0)
    overdue = any(j["overdue"] for j in jobs.values())
    return {"generated": _iso(now), "status": "failed" if worst == 2 or overdue else "degraded" if worst else "ok",
            "jobs": jobs}


def overdue_reason(job: str, entry: dict, now: datetime) -> Optional[str]:
    """Why today's run of ``job`` is overdue, or None. Only judged after the deadline, on the UTC day."""
    hour, minute = EXPECTED_BY_UTC[job]
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now < deadline:
        return None
    run = entry.get("run")
    started = run and datetime.strptime(run["started"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if not run or started < deadline - timedelta(hours=12):
        return f"no {job} run has started today (expected done by {hour:02d}:{minute:02d}Z)"
    if not run.get("finished"):
        return f"{job} started {_iso(started)} and has not finished (expected done by {hour:02d}:{minute:02d}Z)"
    return None


def check_overdue(now: Optional[datetime] = None) -> list[str]:
    """Watchdog: alert (once per job per UTC day) for runs that never started or are still unfinished."""
    now = now or _now()
    reasons = []
    with _locked_state() as state:
        for job in EXPECTED_BY_UTC:
            entry = state["jobs"].setdefault(job, {"run": None, "history": []})
            reason = overdue_reason(job, entry, now)
            if reason and entry.get("overdue_alerted") != now.strftime("%Y-%m-%d"):
                entry["overdue_alerted"] = now.strftime("%Y-%m-%d")
                reasons.append((job, reason))
    for job, reason in reasons:
        _alert(job, "schedule", "failed", reason)
    return [reason for _, reason in reasons]


def _tail(path: Optional[str], lines: int = 15) -> str:
    try:
        return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except (OSError, TypeError):
        return ""


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("begin", "finish", "show"):
        sub.add_parser(name).add_argument("--job")
    stage = sub.add_parser("stage")
    stage.add_argument("name")
    stage.add_argument("status", choices=sorted(SEVERITY))
    stage.add_argument("message")
    stage.add_argument("--job")
    stage.add_argument("--log", help="log file whose last lines go in the alert")
    stage.add_argument("--no-alert", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "begin":
        begin_run(args.job)
    elif args.command == "finish":
        print(finish_run(args.job))
    elif args.command == "stage":
        record(args.name, args.status, args.message, job=args.job, detail=_tail(args.log), alert=not args.no_alert)
    else:
        print(json.dumps(summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_DIR))
    raise SystemExit(main())

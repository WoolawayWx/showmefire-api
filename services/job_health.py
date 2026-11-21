"""Tell staff when a job keeps failing or stops running, and say what probably went wrong and how to fix it.

The Logs page already records every run (services.job_runs). What was missing:
  - a job that fails every time just sat on the page (the FIRMS key was rejected for 24 days);
  - nothing turned a raw error ("exit code 127") into a cause and a next step.

check_jobs() runs every 15 minutes from the API scheduler. For each job it looks at the newest runs:
  - alert once when the last N runs all failed (N depends on how often the job runs), then remind every
    REMINDER_HOURS while it still fails, and send a recovery notice when it works again;
  - alert when a cron script has not started at all for twice its interval (the worker may be down).
Alerting state lives in a small JSON file so it survives restarts. diagnose() maps an error message to a likely
cause and fix from the HINTS table below; add a row there whenever a new failure mode turns up.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_DIR / "data" / "current_conditions" / "job_health_state.json"

REMINDER_HOURS = 12        # a still-failing job is mentioned again this often
LOOKBACK_DAYS = 3          # jobs with no run in this long are not watched for failures (removed or disabled)
OVERDUE_MAX_INTERVALS = 10  # stop calling a cron script overdue after this many missed intervals (disabled or retired)

# (pattern, likely cause, what to do). First match wins, so keep the specific rows above the general ones.
HINTS: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"MAP_KEY is invalid|Invalid MAP_KEY", re.I),
     "NASA FIRMS rejected the MODIS_KEY (expired, revoked, mistyped, or over its transaction limit).",
     "Request a new key at https://firms.modaps.eosdis.nasa.gov/api/map_key/, set MODIS_KEY in showmefire-api/.env "
     "(one line only), then recreate the API container: docker compose up -d --no-deps --force-recreate backend"),
    (re.compile(r"exit code 127|command not found", re.I),
     "A command the script calls is not on cron's PATH (cron runs with almost no environment).",
     "Call it by full path (for example /opt/venv/bin/python) in the script, then redeploy."),
    (re.compile(r"exit code (137|139)\b|\bOut of memory\b|\bKilled\b|MemoryError", re.I),
     "The process was killed, usually for running out of memory.",
     "Check memory use (docker stats, dmesg | grep -i oom) and whether other heavy jobs ran at the same time."),
    (re.compile(r"exit code 143|\bTerminated\b|SIGTERM", re.I),
     "The run was stopped from outside (SIGTERM), usually a deploy or container restart in the middle of the run.",
     "Nothing to fix if it succeeds on the next run. If it repeats, look for deploys at the same time or a "
     "container restart loop (docker ps, docker logs)."),
    (re.compile(r"No space left on device|disk (is )?full|only [\d.]+ GB free", re.I),
     "The disk is full.",
     "Check df -h and docker system df, then run scripts/host_disk_cleanup.sh or prune old archives."),
    (re.compile(r"Missing data files|No archived data found|Missing (beta |09z )?validation summary", re.I),
     "Verification could not find a forecast file and an observation file for the same day.",
     "Check archive/forecasts and archive/raw_data for that date, and that the forecast job that writes them ran."),
    (re.compile(r"InternalError|SlowDown|ServiceUnavailable|reached max retries|\b5\d\d Server Error", re.I),
     "The storage or data service (Cloudflare R2, AWS, or an upstream API) returned a server error.",
     "Usually temporary: the job retries on its next run. If it persists for hours, check the provider's status page."),
    (re.compile(r"Read timed out|ReadTimeout|ConnectTimeout|Connection (reset|aborted|refused)|Max retries exceeded|"
                r"Temporary failure in name resolution", re.I),
     "An upstream server (NOAA, AWS, a data API) was slow or unreachable.",
     "Usually temporary. If it persists, check that the host can reach the internet (curl -I the URL) and the source's status."),
    (re.compile(r"\b(401|403)\b|Unauthorized|Forbidden|AccessDenied|InvalidAccessKey|SignatureDoesNotMatch", re.I),
     "Credentials were rejected (an API token or storage key is wrong, expired or lacks permission).",
     "Check the matching key in showmefire-api/.env, then recreate the container that runs the job."),
    (re.compile(r"ModuleNotFoundError|ImportError", re.I),
     "A Python package or module is missing in this container's image.",
     "Rebuild the image after adding the package to requirements, or redeploy if a file was only just added."),
    (re.compile(r"FileNotFoundError|No such file or directory", re.I),
     "A file the job needs does not exist (not generated yet, or the path is wrong in this container).",
     "Check the path inside the container (docker exec <container> ls <path>) and the job that creates it."),
]


def diagnose(message: Optional[str]) -> Optional[Dict[str, str]]:
    """The likely cause and fix for an error message, or None when it matches nothing known."""
    if not message:
        return None
    for pattern, cause, fix in HINTS:
        if pattern.search(message):
            return {"cause": cause, "fix": fix}
    return None


def failures_needed(median_gap_s: Optional[float]) -> int:
    """How many failed runs in a row make an alert: more for jobs that run often, so one bad poll stays quiet."""
    if median_gap_s is None or median_gap_s >= 6 * 3600:
        return 2
    if median_gap_s >= 3600:
        return 3
    return 6


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def failing_streak(runs: List[Dict[str, Any]]) -> Tuple[int, Optional[Dict[str, Any]], Optional[float]]:
    """(failed runs in a row at the head of `runs`, newest first; the newest failed run; median gap between runs)."""
    streak = 0
    for run in runs:
        if run["status"] != "error":
            break
        streak += 1
    gaps = [_ts(a["started_at"]) - _ts(b["started_at"]) for a, b in zip(runs, runs[1:])]
    return streak, (runs[0] if streak else None), (statistics.median(gaps) if gaps else None)


def _label(job: str) -> str:
    try:
        from services.log_viewer import CRON_JOBS, _humanize

        return CRON_JOBS[job]["label"] if job in CRON_JOBS else _humanize(job)
    except Exception:  # noqa: BLE001
        return job


def _describe(cause: Optional[Dict[str, str]]) -> str:
    if not cause:
        return "No known cause for this error yet. Read the job's log on the admin Logs page."
    return f"Likely cause: {cause['cause']}\nWhat to do: {cause['fix']}"


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _default_notify(source: str, message: str, detail: str = "") -> Any:
    from services.system_alerts import notify_system_issue

    return notify_system_issue(source, message, detail=detail)


def plan(summaries: Dict[str, Dict[str, Any]], runs_by_job: Dict[str, List[Dict[str, Any]]], cron_jobs: Dict[str, Dict[str, Any]],
         state: dict, now: float) -> Tuple[List[Tuple[str, str]], dict]:
    """Pure decision step: (alerts to send as (message, detail), new state)."""
    alerts: List[Tuple[str, str]] = []
    new_state: dict = {}
    remind_s = REMINDER_HOURS * 3600

    def decide(key: str, bad: bool, message: str, detail: str, recovered: str) -> None:
        was = state.get(key)
        if bad:
            if was is None or now - was["alerted_at"] >= remind_s:
                alerts.append((message if was is None else f"{message} (still)", detail))
                new_state[key] = {"alerted_at": now, "since": (was or {}).get("since", now)}
            else:
                new_state[key] = was
        elif was is not None:
            alerts.append((f"{recovered}", "It is running normally again."))

    for job, runs in runs_by_job.items():
        if not runs:
            continue
        label = _label(job)
        streak, newest_failure, gap = failing_streak(runs)
        need = failures_needed(gap)
        bad = streak >= need
        detail = ""
        if bad:
            first = runs[streak - 1]
            error = newest_failure.get("message") or "(no message was recorded)"
            detail = (f"{streak} runs in a row have failed, since {first['started_at']}.\nLatest error: {error}\n"
                      + _describe(diagnose(error)))
        decide(f"{job}:failing", bad, f"{label} keeps failing", detail, f"{label} recovered")

    for job, meta in cron_jobs.items():
        summary = summaries.get(job)
        if not summary or not summary.get("last"):
            continue
        every_s = meta.get("every_s") or 0
        age = now - _ts(summary["last"]["started_at"])
        overdue = every_s and 2 * every_s + 1800 < age < OVERDUE_MAX_INTERVALS * every_s
        detail = (f"Last started {age / 3600:.1f} h ago, but it should run {meta.get('schedule', 'regularly').lower()}.\n"
                  "Likely cause: the worker container is stopped, restarting, or cron inside it is not running.\n"
                  "What to do: docker ps and docker logs show-me-fire-worker (or -long for the long jobs).")
        decide(f"{job}:overdue", bool(overdue), f"{_label(job)} has stopped running", detail, f"{_label(job)} is running again")

    return alerts, new_state


def check_jobs(notify: Optional[Callable[..., Any]] = None, state_path: Optional[Path] = None, now: Optional[float] = None) -> List[str]:
    """Run the check and send the alerts. Returns the alert messages. Never raises."""
    from services import job_runs
    from services.log_viewer import CRON_JOBS

    path = state_path or STATE_PATH
    now = now or time.time()
    try:
        summaries = job_runs.job_summaries(window_hours=LOOKBACK_DAYS * 24)
        runs_by_job = {}
        for job, summary in summaries.items():
            last = summary.get("last")
            if last and now - _ts(last["started_at"]) <= LOOKBACK_DAYS * 86400:
                runs_by_job[job] = job_runs.recent_runs(job, limit=12)
        alerts, new_state = plan(summaries, runs_by_job, CRON_JOBS, _load_state(path), now)
    except Exception:  # noqa: BLE001 - a broken watcher must not take the scheduler down
        logger.warning("Job health check failed", exc_info=True)
        return []
    send = notify or _default_notify
    for message, detail in alerts:
        try:
            send("job-health", message, detail=detail)
        except Exception:  # noqa: BLE001
            logger.warning("Job health alert failed: %s", message, exc_info=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Could not save job health state", exc_info=True)
    return [message for message, _ in alerts]

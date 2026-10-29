"""Read side of the admin Logs page (website/app/pages/admin/logs.vue).

- overview(): every scheduled job and cron script with its last run and
  state, plus every file in logs/ with what writes it and how many
  warnings/errors are near its end.
- read_log(): the end of one log file (or a rotated .gz copy) as entries,
  optionally only warnings/errors or lines matching a search. Reads from
  the end of the file instead of loading all of it.

Entries: a line that starts with a timestamp, a "===" banner, or a bare
"Traceback" starts a new entry; indented lines and the final exception line
of a traceback belong to the entry above, so a traceback is one entry.
"""
from __future__ import annotations

import gzip
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Tuple

from fastapi import HTTPException

from core.config import LOGS_DIR
from core.logging_setup import redact
from services import job_runs

# What writes each log file. First match wins; patterns are fnmatch globs on the path inside logs/.
LOG_CATALOG: List[Tuple[str, str, str]] = [
    ("api.log", "API server", "The API process: requests that failed, and every scheduled job (station and alert polling, fire detection ingest, archiving, purges)."),
    ("maps.log", "Map refresh", "scripts/maps.sh, every 15 minutes: each maps/*.py script's output, its run time, then the CDN upload, RSS feed and product health check."),
    ("cron.log", "12z forecast (cron)", "Anything forecasts.sh printed outside its own dated log. The forecast's own output is in forecast_YYYYMMDD.log."),
    ("forecast_09z_*.log", "09z forecast", "One file per day: the 09z fire-danger forecast run (scripts/forecasts_09z.sh)."),
    ("forecast_2*.log", "12z forecast", "One file per day: every step of the 12z forecast (scripts/forecasts.sh): HRRR wait, DailyForecast, AI text, county maps, beta forecast."),
    ("cron09z.log", "09z forecast (cron)", "Anything forecasts_09z.sh printed outside its own dated log."),
    ("ensemble_12z_*.log", "Ensemble fire danger (beta)", "One file per day: the 15:00Z ensemble fire-danger beta run (scripts/ensemble_12z.sh)."),
    ("cron_ensemble12z.log", "Ensemble fire danger (cron)", "Anything ensemble_12z.sh printed outside its own dated log."),
    ("valForecast.log", "Nightly verification", "scripts/validateForecast.sh at 04:30Z: end-of-day verification reports, RTMA peak, 09z comparison."),
    ("create_empty_outlook_maps.log", "Blank outlook maps", "Nightly at 02:00 Central: resets the day 2/3 outlook graphics until a forecaster publishes."),
    ("firedetections.log", "FIRMS detections", "Satellite fire detections from NASA FIRMS (tools/firedetections.py), one summary line per poll."),
    ("ngfs_ogc_firedetect.log", "NGFS detections", "GOES NGFS fire detections from NESDIS (tools/ngfs_ogc_firedetect.py)."),
    ("nfgs_firedetect.log", "NGFS detections (old)", "Written by the retired tools/nfgs_firedetect.py. Nothing writes it now; it ages out after 30 days."),
    ("forecast_v1.log", "forecast_v1 pipeline", "The NWP-blend forecast pipeline (services/forecast_v1_job.py), including its worker processes."),
    ("forecastedfiredanger.log", "DailyForecast (old)", "Detail log DailyForecast.py used to keep alongside the dated forecast log. No longer written."),
    ("forecastedfiredanger_modelfd.log", "Beta DailyForecast (old)", "Detail log DailyForecast_ModelFD.py used to keep. No longer written."),
    ("forecastfiredanger.log", "Forecast summary (old)", "Two summary lines per forecast run. No longer written; the same lines are in forecast_YYYYMMDD.log."),
    ("realtimefiredanger.log", "Real-time fire danger map", "maps/realtimefiredanger.py: one summary per map refresh."),
    ("rhmap-fil.log", "Relative humidity map", "maps/rhmap-fil.py: one summary per map refresh."),
    ("windmap-fil.log", "Wind map", "maps/windmap-fil.py: one summary per map refresh."),
    ("fuelmoisturemap.log", "Fuel moisture map", "maps/fuelmoisturemap.py: one summary per map refresh."),
    ("outlookgraphic*.log", "Outlook graphics", "maps/outlookgraphic.py: day 1-3 outlook map renders."),
    ("training_*.log", "Model training", "scripts/forecast_ml.sh, one file per day. Training now normally runs in ShowMeFire-Models."),
    ("validation_*.log", "Model validation", "scripts/validate_model.sh, one file per run."),
    ("verification.log", "Observation fetch for verification", "forecast/analysis.py."),
    ("worker_forward.log", "Worker port forward", "docker-compose worker only: forwards localhost:8000 to the API container."),
]

# The cron.d entries in api/Dockerfile, run through scripts/run_job.sh. Keep in step with that file.
CRON_JOBS: Dict[str, Dict[str, Any]] = {
    "maps": {"label": "Map refresh", "schedule": "Every 15 minutes", "every_s": 15 * 60, "log": "maps.log",
             "description": "Redraws the real-time maps, uploads them to the CDN, rebuilds the RSS feed and checks product health."},
    "forecasts_12z": {"label": "12z forecast", "schedule": "Daily, starts 13:00 UTC", "every_s": 86400, "log": "forecast_YYYYMMDD.log",
                      "description": "Waits for the 12z HRRR, then builds the daily fire-danger forecast, AI text, county maps and the beta forecast."},
    "forecasts_09z": {"label": "09z forecast", "schedule": "Daily, starts 12:05 UTC", "every_s": 86400, "log": "forecast_09z_YYYYMMDD.log",
                      "description": "The early 09z HRRR forecast used for comparison with the 12z run."},
    "ensemble_12z": {"label": "Ensemble fire danger (beta)", "schedule": "Daily, 15:00 UTC", "every_s": 86400, "log": "ensemble_12z_YYYYMMDD.log",
                     "description": "REFS/HREF ensemble fire danger, after those models publish."},
    "verification": {"label": "Nightly verification", "schedule": "Daily, 04:30 UTC", "every_s": 86400, "log": "valForecast.log",
                     "description": "Scores the day's forecasts against observations and builds the verification reports."},
    "empty_outlook_maps": {"label": "Blank outlook maps", "schedule": "Daily, 02:00 Central", "every_s": 86400, "log": "create_empty_outlook_maps.log",
                           "description": "Resets the day 2/3 outlook graphics for the new day."},
}

READ_LINES_MAX = 5000
SCAN_BYTES_MAX = 64 * 1024 * 1024  # a filtered search reads at most this much from the end
OVERVIEW_TAIL_BYTES = 128 * 1024

_TIMESTAMP = re.compile(r"^\[?(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})(?:[,.]\d+)?(Z|[+-]\d{2}:?\d{2})?\]?")
_LEVEL = re.compile(r"\b(CRITICAL|FATAL|ERROR|WARNING|WARN|INFO|DEBUG)\b")
_ERROR_HINT = re.compile(r"\b(FAILED|Traceback \(most recent call last\)|ERROR:|\[ERROR\])|^ERROR\b")
_WARN_HINT = re.compile(r"^(WARNING|WARN)\b|\bWARNING:|\bWARN:")
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning|Failure)\b(:|$)")
_TRACEBACK_GLUE = ("During handling of the above exception", "The above exception was the direct cause")
_LEVEL_NAMES = {"CRITICAL": "error", "FATAL": "error", "ERROR": "error", "WARNING": "warning", "WARN": "warning",
                "INFO": "info", "DEBUG": "debug"}
_RANK = {"error": 3, "warning": 2, "info": 1, "debug": 0, None: 1}


def _describe(rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    name = rel_path.rsplit("/", 1)[-1]
    for pattern, label, description in LOG_CATALOG:
        if fnmatch(rel_path, pattern) or fnmatch(name, pattern):
            return label, description
    return None, None


def _resolve(rel_path: str) -> Path:
    base = LOGS_DIR.resolve()
    target = (LOGS_DIR / rel_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden: path is outside the logs folder")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"Log not found: {rel_path}")
    return target


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# --- parsing -----------------------------------------------------------------

def _line_level(line: str, timestamped: bool) -> Optional[str]:
    if timestamped:
        head = _TIMESTAMP.sub("", line, count=1)[:60]
        match = _LEVEL.search(head)
        if match:
            return _LEVEL_NAMES[match.group(1)]
    else:
        match = re.match(r"^(CRITICAL|ERROR|WARNING|INFO|DEBUG):", line)  # logging's default "LEVEL:name:msg"
        if match:
            return _LEVEL_NAMES[match.group(1)]
    if _ERROR_HINT.search(line):
        return "error"
    if _WARN_HINT.search(line):
        return "warning"
    return None


def _max_level(a: Optional[str], b: Optional[str]) -> Optional[str]:
    return b if _RANK[b] > _RANK[a] else a


def _timestamp(line: str) -> Optional[str]:
    match = _TIMESTAMP.match(line)
    if not match:
        return None
    date, clock, zone = match.groups()
    return f"{date}T{clock}{'Z' if zone == 'Z' else (zone or '')}"


def iter_entries(lines: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Group lines into entries (see the module docstring), yielding each one once it is complete."""
    current: Optional[Dict[str, Any]] = None
    for raw in lines:
        line = redact(raw.rstrip("\r\n"))
        stripped = line.strip()
        timestamp = _timestamp(line)
        is_banner = stripped.startswith("===")
        if current is not None and timestamp is None and not is_banner and (
            (line[:1].isspace() and stripped)
            or _EXCEPTION_LINE.match(line)
            or line.startswith(_TRACEBACK_GLUE)
            or (line.startswith("Traceback (most recent call last)") and current["timestamp"])
        ):
            current["text"] += "\n" + line
            current["lines"] += 1
            # An indented "WARNING"/"Traceback" lifts a plain block (a maps.sh script's output, say),
            # but a logger.warning(..., exc_info=True) entry stays a warning.
            if current["level"] not in ("warning", "error"):
                inner = stripped if line[:1].isspace() else line
                current["level"] = _max_level(current["level"], _line_level(inner, _timestamp(inner) is not None))
            continue
        if current is not None:
            yield current
        current = {
            "timestamp": timestamp,
            "level": _line_level(line, timestamp is not None),
            "banner": is_banner,
            "text": line,
            "lines": 1,
        }
    if current is not None:
        yield current


def parse_entries(lines: Iterable[str]) -> List[Dict[str, Any]]:
    return list(iter_entries(lines))


# --- reading -----------------------------------------------------------------

def _tail_lines(path: Path, count: int) -> Tuple[List[str], bool]:
    """The last `count` lines of a plain file, reading backwards in blocks. Also says whether it reached the start."""
    block = 64 * 1024
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        pos, data = end, b""
        while pos > 0 and data.count(b"\n") <= count and end - pos < SCAN_BYTES_MAX:
            step = min(block, pos)
            pos -= step
            handle.seek(pos)
            data = handle.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines()
    if pos > 0 and lines:
        lines = lines[1:]  # the first line was cut in half
    return lines[-count:], pos == 0 and len(lines) <= count


def _iter_lines(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
        return
    size = path.stat().st_size
    with open(path, "rb") as handle:
        if size > SCAN_BYTES_MAX:
            handle.seek(size - SCAN_BYTES_MAX)
            handle.readline()
        for raw in handle:
            yield raw.decode("utf-8", errors="replace")


def _matches(entry: Dict[str, Any], level: str, needle: str) -> bool:
    if level == "error" and entry["level"] != "error":
        return False
    if level == "warning" and entry["level"] not in ("warning", "error"):
        return False
    return not needle or needle in entry["text"].lower()


def read_log(rel_path: str, lines: int = 500, level: str = "all", query: str = "") -> Dict[str, Any]:
    """The last `lines` lines of a log as entries; with a level or search, the last matching entries in the whole file."""
    path = _resolve(rel_path)
    lines = max(20, min(int(lines), READ_LINES_MAX))
    level = level if level in ("all", "warning", "error") else "all"
    needle = query.strip().lower()
    filtered = level != "all" or bool(needle)
    stat = path.stat()
    started = time.perf_counter()

    if not filtered and path.suffix != ".gz":
        tail, from_start = _tail_lines(path, lines)
        entries = parse_entries(tail)
    elif not filtered:
        from_start = False
        window: Deque[str] = deque(maxlen=lines)
        window.extend(_iter_lines(path))
        entries = parse_entries(window)
    else:
        # Keep only the newest matches, within the line budget, so memory stays flat however big the file is.
        matched: Deque[Dict[str, Any]] = deque()
        budget = 0
        for entry in iter_entries(_iter_lines(path)):
            if not _matches(entry, level, needle):
                continue
            matched.append(entry)
            budget += entry["lines"]
            while budget > lines and len(matched) > 1:
                budget -= matched.popleft()["lines"]
        entries = list(matched)
        from_start = stat.st_size <= SCAN_BYTES_MAX or path.suffix == ".gz"

    counts = {"error": 0, "warning": 0}
    for entry in entries:
        if entry["level"] in counts:
            counts[entry["level"]] += 1
    return {
        "path": rel_path,
        "size": stat.st_size,
        "modified": _iso(stat.st_mtime),
        "from_start": from_start,
        "filtered": filtered,
        "entries": entries,
        "counts": counts,
        "read_ms": round((time.perf_counter() - started) * 1000, 1),
    }


# --- overview ----------------------------------------------------------------

_scan_cache: Dict[str, Tuple[Tuple[int, float], Dict[str, Any]]] = {}
_scan_lock = threading.Lock()


def _recent_problems(path: Path, stat: os.stat_result) -> Dict[str, Any]:
    key = str(path)
    signature = (stat.st_size, stat.st_mtime)
    with _scan_lock:
        cached = _scan_cache.get(key)
        if cached and cached[0] == signature:
            return cached[1]
    with open(path, "rb") as handle:
        start = max(0, stat.st_size - OVERVIEW_TAIL_BYTES)
        handle.seek(start)
        if start:
            handle.readline()
        text = handle.read().decode("utf-8", errors="replace")
    entries = parse_entries(text.splitlines())
    errors = [e for e in entries if e["level"] == "error"]
    warnings = [e for e in entries if e["level"] == "warning"]
    last_problem = (errors or warnings or [None])[-1]
    summary = {
        "errors": len(errors),
        "warnings": len(warnings),
        "scanned_bytes": stat.st_size - start,
        "last_problem": None if last_problem is None else {
            "level": last_problem["level"],
            "timestamp": last_problem["timestamp"],
            "text": last_problem["text"].split("\n")[-1 if last_problem["text"].startswith("Traceback") else 0][:300],
        },
        "last_timestamp": next((e["timestamp"] for e in reversed(entries) if e["timestamp"]), None),
    }
    with _scan_lock:
        _scan_cache[key] = (signature, summary)
    return summary


_ROTATED = re.compile(r"^(?P<base>.+\.log)\.(?P<n>\d+)(\.gz)?$")


def list_log_files() -> Dict[str, Any]:
    files: Dict[str, Dict[str, Any]] = {}
    rotated: List[Tuple[str, Dict[str, Any]]] = []
    total = rotated_total = 0
    if LOGS_DIR.exists():
        for path in LOGS_DIR.rglob("*"):
            if not path.is_file() or any(part.startswith(".") for part in path.relative_to(LOGS_DIR).parts):
                continue
            rel = path.relative_to(LOGS_DIR).as_posix()
            stat = path.stat()
            total += stat.st_size
            match = _ROTATED.match(rel)
            if match:
                rotated_total += stat.st_size
                rotated.append((match.group("base"), {"path": rel, "size": stat.st_size, "modified": _iso(stat.st_mtime),
                                                      "index": int(match.group("n"))}))
                continue
            label, description = _describe(rel)
            files[rel] = {
                "path": rel,
                "folder": rel.rsplit("/", 1)[0] if "/" in rel else "",
                "label": label,
                "description": description,
                "size": stat.st_size,
                "modified": _iso(stat.st_mtime),
                "age_s": max(0, round(time.time() - stat.st_mtime)),
                "viewable": path.suffix in (".log", ".txt", ".json", ".csv", ".out"),
                "rotated": [],
                "recent": _recent_problems(path, stat) if path.suffix in (".log", ".txt", ".out") else None,
            }
    for base, item in rotated:
        if base in files:
            files[base]["rotated"].append(item)
        else:
            label, description = _describe(base)
            files.setdefault(item["path"], {
                "path": item["path"], "folder": "", "label": label, "description": description,
                "size": item["size"], "modified": item["modified"], "age_s": None, "viewable": True,
                "rotated": [], "recent": None, "orphan_rotation": True,
            })
    for entry in files.values():
        entry["rotated"].sort(key=lambda r: r["index"])
    ordered = sorted(files.values(), key=lambda f: f["modified"], reverse=True)
    return {"files": ordered, "total_bytes": total, "rotated_bytes": rotated_total}


def _humanize(job_id: str) -> str:
    text = job_id.replace("_job", "").replace("_", " ").strip()
    return text[:1].upper() + text[1:]


# For jobs whose function has no docstring, or one that only says it is a scheduler wrapper.
JOB_DESCRIPTIONS: Dict[str, str] = {
    "archive_stale_ngfs_detections": "Zips and uploads raw NGFS detection captures that have aged out of the 7-day window.",
    "check_mobile_push_receipts": "Checks Expo push receipts and removes tokens for uninstalled apps.",
    "check_staff_push_receipts": "Checks Expo receipts for staff-app alerts and removes phones that no longer have the app.",
    "purge_staff_push_records": "Deletes staff-app push send records older than a week.",
    "purge_admin_auth_records": "Deletes ended admin sign-in sessions after 90 days and sign-in events after a year (they hold IP addresses).",
    "reset_idle_reviewer_sandbox": "Resets the app-store reviewer sandbox to its demo data once nobody has used it for 30 minutes.",
    "daily_day_archive": "Builds and uploads the previous day's archive bundle.",
    "daily_map_archive": "Archives the previous day's published maps.",
    "end_of_day_archive": "Groups the day's HRRR cache, forecasts and raw data into dated archive zips.",
    "end_of_day_rtma_peak": "Builds the day's peak RTMA fire-weather analysis for verification.",
    "fetch_advanced_fire_detections": "Polls NASA FIRMS for VIIRS/MODIS satellite fire detections (log: firedetections.log).",
    "fetch_fire_detections": "Polls NOAA NESDIS for GOES NGFS fire detections (log: ngfs_ogc_firedetect.log).",
    "fetch_timeseries": "Fetches recent station time series from Synoptic for station charts.",
    "full_observation_capture": "Captures yesterday's complete station observations for training data.",
    "hrrr_hourly_forecast": "Processes each new HRRR forecast hour as it posts and builds the shared HRRR cache.",
    "refresh_spc_department_graphics": "Watches SPC outlooks and re-renders department graphics when one changes.",
}

_WRAPPER_DOC = re.compile(r"^(async )?(entry point|apscheduler entry point|main execution function|scheduler wrapper)", re.I)


def _first_doc_line(func) -> Optional[str]:
    doc = (getattr(func, "__doc__", None) or "").strip()
    if not doc:
        return None
    paragraph = " ".join(doc.split("\n\n", 1)[0].split())
    return None if _WRAPPER_DOC.match(paragraph) else paragraph[:300]


def _interval_seconds(trigger) -> Optional[float]:
    interval = getattr(trigger, "interval", None)
    return interval.total_seconds() if interval is not None else None


def _minute_step(minute: str) -> Optional[int]:
    """"*/3" -> 3, "0,10,20,30,40,50" -> 10, else None."""
    if minute.startswith("*/") and minute[2:].isdigit():
        return int(minute[2:])
    parts = minute.split(",")
    if len(parts) >= 2 and all(p.isdigit() for p in parts):
        values = [int(p) for p in parts]
        steps = {b - a for a, b in zip(values, values[1:])}
        if len(steps) == 1 and values[0] < next(iter(steps)) and (values[-1] + next(iter(steps))) == 60 + values[0]:
            return steps.pop()
    return None


def _schedule_text(trigger) -> str:
    seconds = _interval_seconds(trigger)
    if seconds:
        if seconds % 3600 == 0:
            hours = int(seconds // 3600)
            return "Every hour" if hours == 1 else f"Every {hours} hours"
        if seconds >= 60:
            minutes = round(seconds / 60, 1)
            return "Every minute" if minutes == 1 else f"Every {minutes:g} minutes"
        return f"Every {seconds:g} seconds"
    fields = {f.name: str(f) for f in getattr(trigger, "fields", [])}
    if not fields:
        return str(trigger)
    hour, minute, dow = fields.get("hour", "*"), fields.get("minute", "*"), fields.get("day_of_week", "*")
    day_names = {"mon": "Mondays", "tue": "Tuesdays", "wed": "Wednesdays", "thu": "Thursdays", "fri": "Fridays",
                 "sat": "Saturdays", "sun": "Sundays"}
    days = "Daily" if dow == "*" else day_names.get(dow, dow)
    step = _minute_step(minute)
    if hour.isdigit() and minute.isdigit():
        return f"{days} at {hour}:{minute.zfill(2)} Central"
    if step:
        hours = "" if hour == "*" else f", {hour.replace('-', ':00-')}:00 Central"
        return f"Every {step} minutes{hours}" if days == "Daily" else f"{days}, every {step} minutes{hours}"
    if minute.isdigit() and hour == "*":
        return f"Hourly at :{minute.zfill(2)}" if days == "Daily" else f"{days}, hourly at :{minute.zfill(2)}"
    if minute.isdigit() and all(h.isdigit() for h in hour.split(",")):
        return f"{days} at " + ", ".join(f"{h}:{minute.zfill(2)}" for h in hour.split(",")) + " Central"
    return f"{days}, hour {hour}, minute {minute} (Central)"


def _state(last: Optional[Dict[str, Any]], every_s: Optional[float], now: float) -> str:
    if not last:
        return "never"
    started = datetime.fromisoformat(last["started_at"].replace("Z", "+00:00")).timestamp()
    # Three missed runs: one skipped because the previous was still going (max_instances=1) is normal.
    if every_s and now - started > 3 * every_s + 120 + (last.get("duration_s") or 0):
        return "overdue"
    return last["status"]


def overview(scheduler=None) -> Dict[str, Any]:
    now = time.time()
    summaries = job_runs.job_summaries()
    jobs: List[Dict[str, Any]] = []
    seen = set()

    if scheduler is not None:
        from core.scheduler import MODEL_RELEVANT_JOBS

        running = bool(getattr(scheduler, "running", False))
        for job in scheduler.get_jobs():
            seen.add(job.id)
            next_run = getattr(job, "next_run_time", None)  # unset until the scheduler starts
            summary = summaries.get(job.id, {})
            every = _interval_seconds(job.trigger)
            func = getattr(job.func, "__wrapped__", job.func)
            meta = MODEL_RELEVANT_JOBS.get(job.id, {})
            jobs.append({
                "id": job.id,
                "label": _humanize(job.id),
                "description": meta.get("description") or JOB_DESCRIPTIONS.get(job.id) or _first_doc_line(func),
                "source": "scheduler",
                "schedule": _schedule_text(job.trigger),
                "next_run_at": next_run.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if next_run else None,
                "paused": running and next_run is None,
                "log": "api.log",
                "state": _state(summary.get("last"), every, now),
                "last": summary.get("last"),
                "last_failure": summary.get("last_failure"),
                "window": summary.get("window"),
            })

    for job_id, meta in CRON_JOBS.items():
        seen.add(job_id)
        summary = summaries.get(job_id, {})
        jobs.append({
            "id": job_id, "label": meta["label"], "description": meta["description"], "source": "cron",
            "schedule": meta["schedule"], "next_run_at": None, "paused": False, "log": meta["log"],
            "state": _state(summary.get("last"), meta["every_s"], now),
            "last": summary.get("last"), "last_failure": summary.get("last_failure"), "window": summary.get("window"),
        })

    # Jobs with history that are no longer scheduled (removed, or the scheduler is off in this process).
    for job_id, summary in summaries.items():
        if job_id in seen:
            continue
        jobs.append({
            "id": job_id, "label": _humanize(job_id), "description": None, "source": summary.get("source", "scheduler"),
            "schedule": None, "next_run_at": None, "paused": False, "log": "api.log",
            "state": summary["last"]["status"] if summary.get("last") else "never",
            "last": summary.get("last"), "last_failure": summary.get("last_failure"), "window": summary.get("window"),
            "unscheduled": True,
        })

    order = {"error": 0, "overdue": 1, "warning": 2, "ok": 3, "never": 4}
    jobs.sort(key=lambda j: (order.get(j["state"], 5), j["label"].lower()))
    states: Dict[str, int] = {}
    for job in jobs:
        states[job["state"]] = states.get(job["state"], 0) + 1

    files = list_log_files()
    return {
        "generated_at": _iso(now),
        "scheduler_running": bool(scheduler is not None and getattr(scheduler, "running", False)),
        "jobs": jobs,
        "job_states": states,
        **files,
    }

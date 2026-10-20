"""Job-run tracking, log rotation and the log reader behind the admin Logs page."""
import asyncio
import gzip
import logging
import os
import time

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from core.logging_setup import UtcFormatter, redact
from services import job_runs, log_maintenance, log_viewer


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(job_runs, "_table_ready", False)
    return tmp_path


@pytest.fixture
def logs(tmp_path, monkeypatch):
    folder = tmp_path / "logs"
    folder.mkdir()
    monkeypatch.setattr(log_maintenance, "LOGS_DIR", folder)
    monkeypatch.setattr(log_viewer, "LOGS_DIR", folder)
    log_viewer._scan_cache.clear()
    return folder


# --- job runs ----------------------------------------------------------------

def test_a_job_that_logs_an_error_is_recorded_as_failed_even_though_it_did_not_raise(db):
    log = logging.getLogger("tests.jobs")

    async def quiet_failure():
        await asyncio.to_thread(lambda: log.error("upstream returned 500"))

    async def fine():
        return 1

    def warns():
        log.warning("slow response")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(quiet_failure, "interval", minutes=5, id="quiet_failure")
    scheduler.add_job(fine, "interval", minutes=5, id="fine")
    scheduler.add_job(warns, "interval", minutes=5, id="warns")
    assert job_runs.track_scheduler_jobs(scheduler) == 3
    assert job_runs.track_scheduler_jobs(scheduler) == 0  # never wraps twice

    jobs = {job.id: job.func for job in scheduler.get_jobs()}
    assert asyncio.iscoroutinefunction(jobs["fine"]) and not asyncio.iscoroutinefunction(jobs["warns"])
    asyncio.run(jobs["quiet_failure"]())
    asyncio.run(jobs["fine"]())
    jobs["warns"]()

    summaries = job_runs.job_summaries()
    assert summaries["quiet_failure"]["last"]["status"] == "error"
    assert summaries["quiet_failure"]["last"]["message"] == "upstream returned 500"
    assert summaries["quiet_failure"]["last_failure"] is not None
    assert summaries["fine"]["last"]["status"] == "ok"
    assert summaries["warns"]["last"]["status"] == "warning"
    assert summaries["warns"]["window"]["runs"] == 1


def test_a_raised_exception_is_recorded_and_still_raised(db):
    def boom():
        raise RuntimeError("disk full")

    wrapped = job_runs.tracked("boom", boom)
    with pytest.raises(RuntimeError):
        wrapped()
    run = job_runs.recent_runs("boom")[0]
    assert run["status"] == "error" and run["message"] == "RuntimeError: disk full"


def test_warnings_outside_a_job_are_not_counted(db):
    job_runs.track_scheduler_jobs(AsyncIOScheduler())
    logging.getLogger("tests.jobs").warning("not inside any job")
    job_runs.tracked("clean", lambda: None)()
    assert job_runs.recent_runs("clean")[0]["status"] == "ok"


def test_overview_marks_an_interval_job_overdue_when_it_stops_running(db, logs):
    async def poll():
        return None

    scheduler = AsyncIOScheduler()
    scheduler.add_job(poll, "interval", minutes=5, id="poll")
    job_runs.record_run("poll", "ok", time.time() - 3600, 1.0)
    job_runs.record_run("maps", "error", time.time() - 60, 90.0, source="cron", message="1 failed")

    result = log_viewer.overview(scheduler)
    jobs = {job["id"]: job for job in result["jobs"]}
    assert jobs["poll"]["state"] == "overdue"
    assert jobs["poll"]["schedule"] == "Every 5 minutes"
    assert jobs["maps"]["state"] == "error" and jobs["maps"]["source"] == "cron"
    assert jobs["forecasts_12z"]["state"] == "never"
    assert [job["state"] for job in result["jobs"]][:2] == ["error", "overdue"]


def test_old_runs_are_pruned(db):
    job_runs.record_run("old", "ok", time.time() - 30 * 86400, 1.0)
    job_runs.record_run("new", "ok", time.time(), 1.0)
    assert job_runs.prune_runs() == 1
    assert set(job_runs.job_summaries()) == {"new"}


# --- rotation ----------------------------------------------------------------

def test_an_oversized_log_is_rotated_to_gzip_not_wiped(logs):
    live = logs / "maps.log"
    live.write_text("old line\n" * 1000)
    for index in range(1, 6):
        (logs / f"maps.log.{index}.gz").write_bytes(gzip.compress(f"copy {index}\n".encode()))

    result = log_maintenance.purge_old_logs(max_size_bytes=100)

    assert result["rotated"] == ["maps.log"]
    assert live.read_text() == ""
    assert gzip.decompress((logs / "maps.log.1.gz").read_bytes()).decode().count("old line") == 1000
    assert gzip.decompress((logs / "maps.log.2.gz").read_bytes()) == b"copy 1\n"
    assert not (logs / "maps.log.6.gz").exists()


def test_stale_files_are_deleted_and_fresh_small_ones_left_alone(logs):
    stale = logs / "forecast_20260101.log"
    stale.write_text("x")
    old = time.time() - 40 * 86400
    os.utime(stale, (old, old))
    (logs / "api.log").write_text("fresh")

    result = log_maintenance.purge_old_logs()
    assert result == {"removed": ["forecast_20260101.log"], "rotated": []}
    assert (logs / "api.log").read_text() == "fresh"


# --- reading -----------------------------------------------------------------

def test_tracebacks_and_indented_output_are_grouped_into_one_entry():
    entries = log_viewer.parse_entries([
        "2026-10-20 13:00:01Z INFO core.scheduler: Scheduler started",
        "2026-10-20 13:00:02Z WARNING forecast_v1.acquisition: Optional rrfs acquisition failed",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "ValueError: bad grid",
        "--- rhmap-fil.py: ok in 12s",
        "    WARNING: 3 stations missing",
        "--- fuelmoisturemap.py: FAILED with exit code 1 after 4s",
        "=== [maps] finished OK in 95s ===",
    ])
    assert [e["level"] for e in entries] == ["info", "warning", "warning", "error", None]
    assert entries[1]["lines"] == 4 and entries[1]["timestamp"] == "2026-10-20T13:00:02Z"
    assert entries[4]["banner"]


def test_reading_returns_the_tail_and_a_filter_searches_the_whole_file(logs):
    lines = [f"2026-10-20 13:{i // 60:02d}:{i % 60:02d}Z INFO app: line {i}" for i in range(3000)]
    lines[10] = "2026-10-20 13:00:10Z ERROR app: early failure"
    (logs / "api.log").write_text("\n".join(lines) + "\n")

    tail = log_viewer.read_log("api.log", lines=100)
    assert len(tail["entries"]) == 100 and tail["entries"][-1]["text"].endswith("line 2999")
    assert not tail["from_start"]

    errors = log_viewer.read_log("api.log", lines=100, level="error")
    assert [e["text"] for e in errors["entries"]] == ["2026-10-20 13:00:10Z ERROR app: early failure"]

    search = log_viewer.read_log("api.log", lines=100, query="LINE 299")
    assert search["entries"][-1]["text"].endswith("line 2999") and len(search["entries"]) == 11


def test_rotated_copies_are_readable_and_listed_under_their_log(logs):
    (logs / "maps.log").write_text("--- a.py: ok in 1s\n")
    (logs / "maps.log.1.gz").write_bytes(gzip.compress(b"--- a.py: FAILED with exit code 1 after 2s\n"))

    listing = log_viewer.list_log_files()
    assert [f["path"] for f in listing["files"]] == ["maps.log"]
    assert listing["files"][0]["label"] == "Map refresh"
    assert listing["files"][0]["rotated"][0]["path"] == "maps.log.1.gz"
    assert log_viewer.read_log("maps.log.1.gz")["entries"][0]["level"] == "error"


def test_paths_outside_the_logs_folder_are_refused(logs):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        log_viewer.read_log("../secrets.env")
    assert caught.value.status_code == 403


def test_secrets_are_redacted_in_new_lines_and_when_reading_old_ones(logs):
    leaked = "2026-03-25 16:50:06,221 - ERROR - 500 for url: https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/abc123/VIIRS/x/1/"
    (logs / "firedetections.log").write_text(leaked + "\n")
    assert "abc123" not in log_viewer.read_log("firedetections.log")["entries"][0]["text"]
    assert redact("https://x/api?token=SECRET&state=MO") == "https://x/api?token=[redacted]&state=MO"

    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "GET %s failed", ("https://h/?api_key=k1",), None)
    assert "k1" not in UtcFormatter().format(record)

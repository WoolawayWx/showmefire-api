"""Tests for the fire outlook cron scripts (scripts/fire_outlook.sh, scripts/fire_outlook_verification.sh) and their
registration (Dockerfile cron entries <-> services/log_viewer.py). flock/python are stubbed so the shell logic runs anywhere
with bash; the real flock is util-linux in the production image."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]


def _find_bash():
    """A real bash. On Windows `bash` on PATH is often the WSL launcher (system32), which cannot run these paths: prefer Git Bash."""
    for candidate in (os.getenv("SMF_TEST_BASH"), "C:/Program Files/Git/bin/bash.exe", "C:/Program Files (x86)/Git/bin/bash.exe"):
        if candidate and Path(candidate).is_file():
            return candidate
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


BASH = _find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="a real bash is not available")


@pytest.fixture
def proj(tmp_path):
    """A throw-away project dir laid out like /app, with stub worker/export scripts and a flock shim."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("fire_outlook.sh", "fire_outlook_verification.sh"):
        shutil.copy(API_ROOT / "scripts" / name, scripts / name)
    stub = ("import os, sys, time, pathlib\n"
            "root = pathlib.Path(__file__).resolve().parents[1]\n"
            "with open(root / 'marker.txt', 'a') as f:\n"
            "    f.write(f\"{os.path.basename(__file__)} omp={os.getenv('OMP_NUM_THREADS')} blas={os.getenv('OPENBLAS_NUM_THREADS')}\\n\")\n"
            "if os.getenv('STUB_SLEEP'): time.sleep(float(os.getenv('STUB_SLEEP')))\n"
            "sys.exit(int(os.getenv('STUB_EXIT', '0')))\n")
    (scripts / "run_fire_outlook.py").write_text(stub)
    (scripts / "export_fire_outlook_verification.py").write_text(stub)
    shim = tmp_path / "shim"
    shim.mkdir()
    flock = shim / "flock"
    flock.write_text('#!/bin/bash\n'
                     'case "$*" in\n'
                     '  *" -n "*|"-n "*) [ "$FAKE_BUSY" = "own" ] && exit 1 ;;\n'
                     '  *-w*) [ "$FAKE_BUSY" = "forecast" ] && exit 1 ;;\n'
                     'esac\nexit 0\n')
    flock.chmod(0o755)
    return tmp_path


def run(proj, script="fire_outlook.sh", **env):
    summary = proj / "summary.txt"
    full = {**os.environ, "PATH": f"{(proj / 'shim').as_posix()}{os.pathsep}{os.environ['PATH']}", "JOB_SUMMARY_FILE": str(summary),
            "SMF_FIRE_OUTLOOK_DIR": "data/fire-outlook", "PYTHONUTF8": "1", **env}
    for key in ("SMF_FIRE_OUTLOOK_ENABLED", "FAKE_BUSY", "STUB_EXIT", "STUB_SLEEP"):
        if key not in env:
            full.pop(key, None)
    result = subprocess.run([BASH, (proj / "scripts" / script).as_posix()], cwd=proj, env=full, capture_output=True, text=True, timeout=60)
    marker = proj / "marker.txt"
    return result, (marker.read_text() if marker.exists() else ""), (summary.read_text().strip() if summary.exists() else "")


def write_status(proj, **fields):
    path = proj / "data" / "fire-outlook"
    path.mkdir(parents=True)
    (path / "status.json").write_text(json.dumps(fields))


def test_disabled_by_default_does_nothing(proj):
    result, marker, _ = run(proj)
    assert result.returncode == 0 and "disabled" in result.stdout and marker == ""
    assert not (proj / ".locks").exists()                                   # not even a lock file is touched


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_enabled_runs_once_with_one_thread_and_logs(proj, value):
    write_status(proj, state="complete", stage="done", run_id="20261029T00Z", cycle_age_hours=7.5)
    result, marker, summary = run(proj, SMF_FIRE_OUTLOOK_ENABLED=value)
    assert result.returncode == 0, result.stderr
    assert marker.count("run_fire_outlook.py") == 1 and "omp=1" in marker and "blas=1" in marker
    assert "run=20261029T00Z" in summary and summary.startswith("complete")
    assert list((proj / "logs").glob("fire_outlook_2*.log"))
    assert (proj / ".locks" / "fire-outlook.lock").exists() and (proj / ".locks" / "showmefire-forecast.lock").exists()


def test_thread_setting_is_honoured(proj):
    _, marker, _ = run(proj, SMF_FIRE_OUTLOOK_ENABLED="true", SMF_FIRE_OUTLOOK_THREADS="2")
    assert "omp=2" in marker


def test_failure_is_reported_with_the_exit_code_and_status(proj):
    write_status(proj, state="failed", stage="calculating", run_id=None, cycle_age_hours=6.0)
    result, marker, summary = run(proj, SMF_FIRE_OUTLOOK_ENABLED="true", STUB_EXIT="1")
    assert result.returncode == 1 and "fire outlook failed (exit 1)" in summary and "failed calculating" in summary


def test_missing_status_file_still_gives_a_summary(proj):
    _, _, summary = run(proj, SMF_FIRE_OUTLOOK_ENABLED="true")
    assert summary == "no status yet"


def test_skips_when_a_previous_run_is_still_active(proj):
    result, marker, _ = run(proj, SMF_FIRE_OUTLOOK_ENABLED="true", FAKE_BUSY="own")
    assert result.returncode == 0 and "still active" in result.stdout and marker == ""


def test_skips_while_the_daily_forecast_holds_the_forecast_lock(proj):
    result, marker, _ = run(proj, SMF_FIRE_OUTLOOK_ENABLED="true", FAKE_BUSY="forecast")
    assert result.returncode == 0 and "forecast lock" in result.stdout and marker == ""


@pytest.mark.skipif(sys.platform == "win32", reason="timeout(1) cannot reliably kill a native Windows python from Git Bash")
def test_hard_time_limit_is_reported(proj):
    result, _, summary = run(proj, SMF_FIRE_OUTLOOK_ENABLED="true", STUB_SLEEP="30", SMF_FIRE_OUTLOOK_TIMEOUT_SECONDS="1")
    assert result.returncode in (124, 137) and "killed after the time limit" in summary


def test_verification_export_script_gated_and_reports_failure(proj):
    off, marker, _ = run(proj, "fire_outlook_verification.sh")
    assert off.returncode == 0 and "disabled" in off.stdout and marker == ""
    ok, marker, _ = run(proj, "fire_outlook_verification.sh", SMF_FIRE_OUTLOOK_ENABLED="true")
    assert ok.returncode == 0 and "export_fire_outlook_verification.py" in marker
    bad, _, summary = run(proj, "fire_outlook_verification.sh", SMF_FIRE_OUTLOOK_ENABLED="true", STUB_EXIT="1")
    assert bad.returncode == 1 and "verification export failed (exit 1)" in summary


# --- registration ----------------------------------------------------------------------------------------------

def _dockerfile_cron():
    text = (API_ROOT / "Dockerfile").read_text(encoding="utf-8")
    return re.findall(r'echo "([^"]+?) root /bin/bash /app/scripts/run_job\.sh (\w+) (/app/scripts/\S+) >> (/app/logs/\S+) 2>&1"', text)


def test_dockerfile_cron_entries_exist_and_match_the_log_viewer_registry():
    from services.log_viewer import CRON_JOBS, LOG_CATALOG
    entries = {job: (schedule, script, log) for schedule, job, script, log in _dockerfile_cron()}
    assert {"fire_outlook", "fire_outlook_verification"} <= set(entries)
    for job, (schedule, script, log) in entries.items():
        assert job in CRON_JOBS, f"cron job {job} is not registered in services/log_viewer.py"
        assert (API_ROOT / script.replace("/app/", "")).is_file(), script
    assert entries["fire_outlook"][0] == "7,37 * * * *" and entries["fire_outlook_verification"][0] == "40 06 * * *"
    assert entries["fire_outlook"][2] == "/app/logs/cron_fire_outlook.log"
    patterns = [p for p, _, _ in LOG_CATALOG]
    from fnmatch import fnmatch
    for name in ("fire_outlook_20261029.log", "fire_outlook_verification.log", "cron_fire_outlook.log"):
        assert any(fnmatch(name, p) for p in patterns), f"{name} has no LOG_CATALOG entry"
    assert CRON_JOBS["fire_outlook"]["every_s"] == 1800 and CRON_JOBS["fire_outlook_verification"]["every_s"] == 86400


def test_outlook_ticks_never_land_on_the_map_refresh_minutes():
    minutes = {int(m) for m in _dockerfile_cron_schedule("fire_outlook").split()[0].split(",")}
    assert not minutes & {0, 15, 30, 45}                                    # maps.sh runs */15


def _dockerfile_cron_schedule(job):
    return next(schedule for schedule, j, _, _ in _dockerfile_cron() if j == job)


def test_dev_compose_never_enables_the_worker():
    text = (API_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    assert re.search(r'SMF_FIRE_OUTLOOK_ENABLED:\s*"false"', text)

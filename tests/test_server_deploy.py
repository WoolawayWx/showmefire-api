"""Replay the server deploy flow with real Git and simulated Docker services."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from core.job_groups import guard_forecast_job

fcntl = pytest.importorskip("fcntl")
pty = pytest.importorskip("pty")
termios = pytest.importorskip("termios")

API = Path(__file__).resolve().parents[1]
ADVISORY_MAP = "maps/advisory_category_maps.py"
NOW = 1795592400  # 09:00 UTC / 03:00 Central, outside the restart guard windows

DOCKER_STUB = """
def docker(args):
    state_file = Path(os.environ["FAKE_STATE"])
    state = json.loads(state_file.read_text())
    containers = state["containers"]
    images = state["images"]
    if args[:2] == ["compose", "config"]:
        print(json.dumps(state["config"]))
    elif args[:2] in (["compose", "restart"], ["compose", "up"]):
        services = [value for value in args[2:] if not value.startswith("-")]
        log(args[:2] + services)
        if "forecast" in services:
            if state.get("fail_start"):
                return 1
            for name in ("forecast", "worker-long"):
                containers[name] = images["showmefire-api:long"]
            if not state.get("stale_heartbeat"):
                heartbeat = Path("showmefire-api/data/current_conditions/status/scheduler_forecast.json")
                heartbeat.parent.mkdir(parents=True, exist_ok=True)
                heartbeat.write_text(json.dumps({
                    "host": "forecast-host",
                    "groups": ["forecast"],
                    "git_sha": images["showmefire-api:long"],
                    "written_at": datetime.fromtimestamp(NOW + 0.5, timezone.utc).isoformat().replace("+00:00", "Z"),
                    "running_jobs": [],
                }))
    elif args[0] == "inspect":
        fmt, name = args[2:]
        service = name.removeprefix("show-me-fire-").replace("api", "backend")
        if service not in containers:
            return 1
        revision = containers[service]
        if ".State.Running" in fmt:
            print("true")
        elif ".State.StartedAt" in fmt:
            print(NOW - 1)
        elif ".Config.Hostname" in fmt:
            print("forecast-host")
        elif "smf.long_job_lock" in fmt:
            print("1")
        elif ".Config.Env" in fmt:
            print("SMF_LONG_JOB_LOCK=/app/.locks/long-deploy.lock")
        elif "smf.git_sha" in fmt:
            print(revision)
        elif ".Image" in fmt:
            print("image-" + revision)
    elif args[:2] == ["image", "inspect"]:
        fmt, tag = args[3:]
        if tag not in images:
            return 1
        print("image-" + images[tag] if fmt == "{{.Id}}" else images[tag])
    elif args[0] == "build":
        revision = args[args.index("--build-arg") + 1].removeprefix("GIT_SHA=")
        snapshot = Path(args[-1])
        assert (snapshot / "Dockerfile").exists()
        assert not (snapshot / "host-only.txt").exists()
        images[args[args.index("--tag") + 1]] = revision
        log(["build", revision])
        if state.get("job_starts_during_build"):
            state["busy_cron"] = True
    elif args[0] == "tag":
        source, tag = args[1:]
        images[tag] = source.removeprefix("image-") if source.startswith("image-") else images[source]
        log(args)
    elif args[0] == "top":
        print("COMMAND")
        if state.get("busy_cron"):
            print("/bin/bash /app/scripts/run_job.sh forecasts_09z")
    elif args[0] == "exec":
        print("cron")
    elif args[0] == "logs":
        pass
    else:
        raise AssertionError(args)
    state_file.write_text(json.dumps(state))
    return 0
"""

TOOL_STUB = """
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys

def log(args):
    with open(os.environ["FAKE_LOG"], "a") as stream:
        print(json.dumps(args), file=stream)

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
if tool == "date":
    if "-d" in args:
        value = args[args.index("-d") + 1]
        if value.isdecimal():
            print(value)
        elif value.endswith("Z"):
            print(int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()))
        else:
            print(NOW - 3600)
    else:
        print(NOW)
elif tool == "stat":
    print(NOW)
elif tool == "flock":
    mode = fcntl.LOCK_SH if "-s" in args else fcntl.LOCK_EX
    try:
        fcntl.flock(int(args[-1]), mode | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(1)
elif tool == "docker":
    sys.exit(docker(args))
elif tool == "curl":
    sys.exit(int(json.loads(Path(os.environ["FAKE_STATE"]).read_text()).get("fail_api_health", False)))
else:
    raise AssertionError(tool)
"""


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


@pytest.fixture
def deploy(tmp_path, monkeypatch):
    if not shutil.which("jq"):
        pytest.skip("requires jq (also required on the server)")
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Deploy test")
    git(origin, "config", "user.email", "deploy@example.test")
    (origin / "Dockerfile").write_text("FROM scratch\n")
    (origin / ADVISORY_MAP).parent.mkdir()
    (origin / ADVISORY_MAP).write_text("severity = 1\n")
    git(origin, "add", ".")
    git(origin, "commit", "-m", "Deployed base")
    revision = git(origin, "rev-parse", "HEAD")
    repo = tmp_path / "showmefire-api"
    git(tmp_path, "clone", str(origin), str(repo))
    (repo / ".env").touch()
    (repo / "host-only.txt").touch()
    (tmp_path / "docker-compose.yml").touch()
    (tmp_path / ".deploy-config.sha256").write_text(hashlib.sha256(b"").hexdigest() + "\n")

    config = {
        "backend": {"environment": {"SMF_SCHEDULER_GROUPS": "api"}},
        "scheduler": {"command": ["python", "-m", "core.scheduler_worker", "--groups", "heavy"]},
        "worker": {"environment": {"SMF_CRON_SET": "short"}},
        "forecast": {"command": ["python", "-m", "core.scheduler_worker", "--groups", "forecast"]},
        "worker-long": {"environment": {"SMF_CRON_SET": "long"}},
    }
    for name in ("forecast", "worker-long"):
        config[name].setdefault("environment", {})["SMF_LONG_JOB_LOCK"] = "/app/.locks/long-deploy.lock"
        config[name]["volumes"] = [{"type": "bind", "source": str(repo / ".locks"), "target": "/app/.locks"}]
    state_file = tmp_path / "docker.json"
    state_file.write_text(json.dumps({
        "config": {"services": config},
        "containers": dict.fromkeys(config, revision),
        "images": {"showmefire-api:long": revision},
    }))
    event_log = tmp_path / "docker.log"
    event_log.touch()
    commands = tmp_path / "bin"
    commands.mkdir()
    interpreter = commands / "test-python"
    interpreter.symlink_to(sys.executable)
    for name in ("docker", "date", "flock", "curl", "stat"):
        stub = commands / name
        stub.write_text(f"#!{interpreter}\nNOW = {NOW}\n" + DOCKER_STUB + TOOL_STUB)
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{commands}:{os.environ['PATH']}")
    monkeypatch.setenv("SMF_DEPLOY_ROOT", str(tmp_path))
    monkeypatch.setenv("FAKE_STATE", str(state_file))
    monkeypatch.setenv("FAKE_LOG", str(event_log))
    monkeypatch.setenv("DEPLOY_HEALTH_WAIT_SECONDS", "0")
    return tmp_path, origin, repo, state_file, event_log


def change(origin, path, content):
    target = origin / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    git(origin, "add", path)
    git(origin, "commit", "-m", f"Update {path}")
    return git(origin, "rev-parse", "HEAD")


def run_deploy(*args):
    return subprocess.run(
        ["bash", str(API / "docs/server/deploy.sh"), *args],
        capture_output=True, text=True, timeout=20,
    )


def events(event_log):
    return [json.loads(line) for line in event_log.read_text().splitlines()]


def test_advisory_diff_restarts_only_backend(deploy):
    _, origin, repo, _, event_log = deploy
    target = change(origin, ADVISORY_MAP, "severity = 2\n")
    result = run_deploy("--yes")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "image: no build" in result.stdout
    assert events(event_log) == [["compose", "restart", "backend"]]
    assert git(repo, "rev-parse", "HEAD") == target


def test_cancel_leaves_live_checkout_and_state_alone(deploy):
    root, origin, repo, _, event_log = deploy
    before = git(repo, "rev-parse", "HEAD")
    change(origin, ADVISORY_MAP, "updated")
    master, terminal = pty.openpty()
    # Keep a tty on stdin and stdout so the real interactive branch is exercised.
    try:
        attrs = termios.tcgetattr(terminal)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(terminal, termios.TCSANOW, attrs)
        with subprocess.Popen(
            ["bash", str(API / "docs/server/deploy.sh")],
            stdin=terminal, stdout=terminal, stderr=terminal,
        ) as process:
            os.write(master, b"q\n")
            assert process.wait(timeout=20) == 0
    finally:
        os.close(terminal)
        os.close(master)
    assert git(repo, "rev-parse", "HEAD") == before
    assert events(event_log) == []
    assert not (root / ".deploy-state").exists()


def test_api_only_leaves_mixed_change_pending_for_other_services(deploy):
    _, origin, _, _, event_log = deploy
    change(origin, ADVISORY_MAP, "new map")
    change(origin, "core/shared.py", "new shared code")
    first = run_deploy("--api-only")
    assert first.returncode == 0, first.stderr or first.stdout
    assert events(event_log) == [["compose", "restart", "backend"]]
    second = run_deploy("--yes")
    assert second.returncode == 0, second.stderr or second.stdout
    assert ["compose", "restart", "scheduler"] in events(event_log)
    assert ["compose", "up", "forecast", "worker-long"] in events(event_log)


def test_failed_api_probe_is_retried_after_checkout_advances(deploy):
    root, origin, _, state_file, event_log = deploy
    change(origin, ADVISORY_MAP, "new map")
    state = json.loads(state_file.read_text())
    state["fail_api_health"] = True
    state_file.write_text(json.dumps(state))
    first = run_deploy("--yes")
    assert first.returncode == 1, first.stdout
    assert "backend: failed verification" in first.stderr
    assert (root / ".deploy-state" / "pending-backend").exists()

    state = json.loads(state_file.read_text())
    state["fail_api_health"] = False
    state_file.write_text(json.dumps(state))
    second = run_deploy("--yes")
    assert second.returncode == 0, second.stderr or second.stdout
    assert events(event_log) == [["compose", "restart", "backend"]] * 2
    assert not (root / ".deploy-state" / "pending-backend").exists()


def test_single_build_reused_after_job_starts_during_build(deploy):
    root, origin, _, state_file, event_log = deploy
    target = change(origin, "requirements.lock.txt", "new package")
    state = json.loads(state_file.read_text())
    state["job_starts_during_build"] = True
    state_file.write_text(json.dumps(state))
    first = run_deploy("--yes")
    assert first.returncode == 0, first.stderr or first.stdout
    assert "Long workers: deferred" in first.stdout
    assert (root / ".long-pending").read_text().strip() == target
    assert ["compose", "up", "forecast", "worker-long"] not in events(event_log)

    state = json.loads(state_file.read_text())
    state.update(job_starts_during_build=False, busy_cron=False)
    state_file.write_text(json.dumps(state))
    second = run_deploy("--long")
    assert second.returncode == 0, second.stderr or second.stdout
    assert events(event_log).count(["build", target]) == 1
    assert not (root / ".long-pending").exists()


def test_shared_job_lock_defers_long_rollout(deploy):
    _, origin, repo, _, event_log = deploy
    change(origin, "core/shared.py", "new shared code")
    lock_dir = repo / ".locks"
    lock_dir.mkdir()
    with (lock_dir / "long-deploy.lock").open("a") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_SH)
        result = run_deploy("--yes")
    assert result.returncode == 0, result.stderr
    assert "a guarded job is active" in result.stdout
    assert ["compose", "up", "forecast", "worker-long"] not in events(event_log)


def test_cron_wrapper_does_not_start_during_worker_switch(deploy, monkeypatch):
    root, _, _, _, _ = deploy
    path = root / "long-deploy.lock"
    monkeypatch.setenv("SMF_LONG_JOB_LOCK", str(path))
    with path.open("a") as deploy_lock:
        fcntl.flock(deploy_lock, fcntl.LOCK_EX)
        result = subprocess.run(
            ["bash", str(API / "scripts/run_job.sh"), "forecasts_09z", "must-not-run"],
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode == 0, result.stderr
    assert "skipped while long workers are being deployed" in result.stdout


@pytest.mark.parametrize("failure", ["fail_start", "stale_heartbeat"])
def test_failed_long_rollout_preserves_pending_and_previous_image(deploy, failure):
    root, origin, repo, state_file, event_log = deploy
    previous = git(repo, "rev-parse", "HEAD")
    target = change(origin, "core/shared.py", "new shared code")
    state = json.loads(state_file.read_text())
    state[failure] = True
    state_file.write_text(json.dumps(state))
    if failure == "stale_heartbeat":
        from datetime import datetime, timezone

        heartbeat = repo / "data/current_conditions/status/scheduler_forecast.json"
        heartbeat.parent.mkdir(parents=True)
        heartbeat.write_text(json.dumps({
            "host": "forecast-host",
            "groups": ["forecast"],
            "git_sha": target,
            "written_at": datetime.fromtimestamp(NOW - 300, timezone.utc).isoformat().replace("+00:00", "Z"),
            "running_jobs": [],
        }))
    result = run_deploy("--yes")
    assert result.returncode == 1, result.stdout
    assert "failed verification or startup" in result.stderr
    assert (root / ".long-pending").read_text().strip() == target
    assert (root / ".deploy-state" / "long").read_text().startswith(previous)
    assert json.loads(state_file.read_text())["images"]["showmefire-api:long-previous"] == previous
    assert events(event_log).count(["build", target]) == 1


def test_forecast_job_defers_before_claiming_work(tmp_path):
    import asyncio

    job = AsyncMock(return_value="claimed")
    job.__name__ = "forecast_job"
    path = tmp_path / "long-deploy.lock"
    guarded = guard_forecast_job(job, path)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert asyncio.run(guarded()) is None
        job.assert_not_awaited()
    assert asyncio.run(guarded()) == "claimed"
    job.assert_awaited_once()

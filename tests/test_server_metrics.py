import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import server_metrics as sm
from core.security import require_admin


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def raw(cpu_total=1000.0, cpu_idle=700.0, iowait=0.0, steal=0.0, proc_cpu=10.0, disk=(0, 0), net=(0, 0),
        mem_used=4 * 2**30, mem_total=16 * 2**30, container=None, load=(0.5, 0.4, 0.3), started=1_799_990_000.0):
    return {
        "cores": 4, "cpu": {"total": cpu_total, "busy": cpu_total - cpu_idle - iowait, "iowait": iowait, "steal": steal},
        "load": load, "memory": {"total": mem_total, "available": mem_total - mem_used, "used": mem_used},
        "swap": {"total": 2 * 2**30, "used": 0}, "disk_io": {"read": disk[0], "write": disk[1]},
        "net": {"rx": net[0], "tx": net[1]},
        "process": {"cpu": proc_cpu, "rss": 300 * 2**20, "threads": 12, "fds": 40, "started": started},
        "container": container, "host_uptime": 86_400.0,
    }


class FakeProbe:
    """Plays back scripted readings; one `read()` per sample."""

    def __init__(self, readings, processes=None, volumes=None):
        self.readings = list(readings)
        self._processes = processes or []
        self._volumes = volumes if volumes is not None else [
            {"path": "/app", "total": 100, "used": 40, "free": 60, "used_pct": 40.0}]

    def read(self):
        return self.readings.pop(0)

    def processes(self):
        return self._processes.pop(0) if self._processes and isinstance(self._processes[0], list) else self._processes

    def volumes(self, paths):
        return self._volumes


def sampler(readings, clock, **kwargs):
    return sm.ServerSampler(FakeProbe(readings, **kwargs), clock=clock, dir_paths=[])


# --- rates -----------------------------------------------------------------------------------------------------------

def test_first_sample_has_no_rates_then_the_second_does(tmp_path):
    clock = Clock()
    s = sampler([raw(), raw(cpu_total=1400, cpu_idle=900, iowait=40, steal=8, proc_cpu=14, disk=(10_000_000, 4_000_000),
                            net=(2_000_000, 500_000))], clock)
    first = s.sample()
    assert first["cpu_pct"] is None and first["disk_read_bps"] is None and first["process"]["cpu_pct"] is None
    clock.now += 10
    second = s.sample()
    # 400 ticks elapsed across all cores: idle +200, iowait +40 -> busy +160 = 40%
    assert second["cpu_pct"] == 40.0 and second["iowait_pct"] == 10.0 and second["steal_pct"] == 2.0
    assert second["process"]["cpu_pct"] == 40.0                     # 4 CPU-seconds in 10 s = 40% of one core
    assert second["disk_read_bps"] == 1_000_000 and second["disk_write_bps"] == 400_000
    assert second["net_rx_bps"] == 200_000 and second["net_tx_bps"] == 50_000
    assert second["mem"]["used_pct"] == 25.0 and second["load"] == [0.5, 0.4, 0.3]
    assert second["process"]["rss_bytes"] == 300 * 2**20 and second["process"]["threads"] == 12


def test_a_counter_that_goes_backwards_is_not_a_negative_rate():
    clock = Clock()
    s = sampler([raw(disk=(5_000, 5_000), net=(9_000, 9_000)), raw(cpu_total=1100, disk=(100, 100), net=(50, 50))], clock)
    s.sample()
    clock.now += 10
    again = s.sample()                                              # e.g. an interface or the counters were reset
    assert again["disk_read_bps"] is None and again["net_rx_bps"] is None


def test_missing_counters_are_left_out_not_zero():
    clock = Clock()
    no_io = raw()
    no_io["disk_io"] = None
    no_io["net"] = None
    other = raw(cpu_total=1100)
    other["disk_io"], other["net"] = None, None
    s = sampler([no_io, other], clock)
    s.sample()
    clock.now += 10
    sample = s.sample()
    assert sample["disk_read_bps"] is None and sample["net_tx_bps"] is None and sample["cpu_pct"] is not None


def test_container_figures_come_from_the_cgroup():
    clock = Clock()
    box1 = {"memory": 2 * 2**30, "memory_limit": 4 * 2**30, "cpu_seconds": 100.0, "cpu_limit_cores": 2.0}
    box2 = {"memory": 3 * 2**30, "memory_limit": 4 * 2**30, "cpu_seconds": 105.0, "cpu_limit_cores": 2.0}
    s = sampler([raw(container=box1), raw(cpu_total=1100, container=box2)], clock)
    s.sample()
    clock.now += 10
    container = s.sample()["container"]
    assert container["memory_bytes"] == 3 * 2**30 and container["memory_limit_bytes"] == 4 * 2**30
    assert container["cpu_pct"] == 50.0 and container["cpu_limit_cores"] == 2.0
    assert sampler([raw()], clock).sample()["container"] is None


# --- the per-minute store and the snapshot ----------------------------------------------------------------------------

def run_minutes(minutes, per_minute=6, cpu_idle_step=70):
    """Sample every 10 s for `minutes` minutes at a steady busy share; returns the sampler and clock."""
    clock = Clock()
    readings, total, idle = [], 1000.0, 700.0
    for _ in range(minutes * per_minute + 1):
        readings.append(raw(cpu_total=total, cpu_idle=idle))
        total, idle = total + 100, idle + cpu_idle_step
    s = sampler(readings, clock)
    for index in range(len(readings)):
        s.sample()
        clock.now += 10
    return s, clock


def test_minutes_hold_the_mean_and_peak_of_their_samples():
    s, clock = run_minutes(3)
    snapshot = s.snapshot(15, now=clock.now)
    series = snapshot["series"]
    assert series["step_minutes"] == 1
    measured = [p for p in series["points"] if p["cpu_pct"] is not None]
    assert measured and all(p["cpu_pct"] == 30.0 and p["cpu_pct_max"] == 30.0 for p in measured)   # 30 busy of every 100
    assert snapshot["current"]["cpu_pct"] == 30.0 and snapshot["sample_seconds"] == sm.SAMPLE_SECONDS
    assert measured[0]["mem_used_pct"] == 25.0 and measured[0]["load1"] == 0.5


def test_windows_choose_the_step_and_clamp():
    s, clock = run_minutes(2)
    assert s.snapshot(30, now=clock.now)["series"]["step_minutes"] == 1
    assert s.snapshot(120, now=clock.now)["series"]["step_minutes"] == 5
    assert s.snapshot(1440, now=clock.now)["series"]["step_minutes"] == 15
    assert s.snapshot(1, now=clock.now)["window_minutes"] == 5
    assert s.snapshot(99999, now=clock.now)["window_minutes"] == 1440


def test_only_24_hours_are_kept():
    clock = Clock()
    s = sampler([raw(), raw(cpu_total=1100)], clock)
    s.sample()
    clock.now += 25 * 3600
    s.sample()
    assert len(s.to_dict()["minutes"]) == 1


def test_persistence_round_trip_and_stale_or_damaged_files(tmp_path):
    s, clock = run_minutes(2)
    path = tmp_path / "server.json"
    s.save(path)
    again = sm.ServerSampler(FakeProbe([]), clock=clock, dir_paths=[])
    assert again.load(path) >= 2
    assert [p["cpu_pct"] for p in again.snapshot(15, now=clock.now)["series"]["points"] if p["cpu_pct"] is not None]
    clock.now += 30 * 3600
    assert sm.ServerSampler(FakeProbe([]), clock=clock, dir_paths=[]).load(path) == 0
    (tmp_path / "bad.json").write_text("{nope")
    assert again.load(tmp_path / "bad.json") == 0
    (tmp_path / "odd.json").write_text(json.dumps({"minutes": {"1": {"cpu_pct": "x", "evil": [1, 1, 1]}, "z": 5}}))
    assert again.load(tmp_path / "odd.json") in (0, 1)              # never raises, junk keys are ignored
    assert "evil" not in again.to_dict()["minutes"].get("1", {})


# --- processes, folders, alerts --------------------------------------------------------------------------------------

def test_busiest_processes_are_listed_by_cpu_then_memory():
    clock = Clock()
    first = [{"pid": 1, "name": "uvicorn", "cpu": 100.0, "rss": 400 * 2**20},
             {"pid": 2, "name": "python hrrr_hourly_forecast.py", "cpu": 50.0, "rss": 900 * 2**20},
             {"pid": 3, "name": "sleep", "cpu": 0.0, "rss": 1 * 2**20}]
    second = [{"pid": 1, "name": "uvicorn", "cpu": 102.0, "rss": 400 * 2**20},
              {"pid": 2, "name": "python hrrr_hourly_forecast.py", "cpu": 59.0, "rss": 950 * 2**20},
              {"pid": 3, "name": "sleep", "cpu": 0.0, "rss": 1 * 2**20},
              {"pid": 4, "name": "new", "cpu": 1.0, "rss": 5 * 2**20}]
    s = sampler([raw(), raw(cpu_total=1100)], clock, processes=[first, second])
    s.sample()
    clock.now += 10
    s.sample()
    names = [(p["name"], p["cpu_pct"]) for p in s.snapshot(15, now=clock.now)["processes"]]
    assert names[0] == ("python hrrr_hourly_forecast.py", 90.0) and names[1] == ("uvicorn", 20.0)
    assert any(name == "sleep" for name, _ in names)                  # small, but in the memory ranking
    assert {"pid", "name", "cpu_pct", "rss_bytes"} == set(s.snapshot(15, now=clock.now)["processes"][0])


def test_folder_sizes_count_bytes_and_files_and_respect_the_budget(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.bin").write_bytes(b"x" * 1000)
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "b" / "two.bin").write_bytes(b"y" * 250)
    (tmp_path / "top.txt").write_bytes(b"z" * 50)
    result = sm.dir_size(tmp_path)
    assert result == {"bytes": 1300, "files": 3, "partial": False}
    ticks = iter([0, 100, 200, 300])
    assert sm.dir_size(tmp_path, budget_seconds=5, clock=lambda: next(ticks))["partial"] is True
    assert sm.dir_size(tmp_path / "missing") == {"bytes": 0, "files": 0, "partial": False}


def test_refresh_dirs_lists_only_folders_that_exist(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "f").write_bytes(b"1" * 10)
    clock = Clock()
    s = sm.ServerSampler(FakeProbe([]), clock=clock, dir_paths=[tmp_path / "data", tmp_path / "nope"])
    s.refresh_dirs()
    assert [(d["bytes"], d["files"]) for d in s.snapshot(15, now=clock.now)["dirs"]] == [(10, 1)]


def level(alerts):
    return {a["message"].split()[0]: a["level"] for a in alerts}


def test_alerts_for_a_full_disk_low_memory_swap_and_load():
    base = raw()
    current = {"mem": {"total": 100, "available": 4, "used": 96, "used_pct": 96.0}, "swap": {"total": 10, "used": 8, "used_pct": 80.0},
               "cores": 4, "load": [9.0, 5.0, 3.0],
               "container": {"memory_bytes": 95, "memory_limit_bytes": 100, "cpu_pct": 1, "cpu_limit_cores": None}}
    volumes = [{"path": "/app", "total": 100, "used": 96, "free": 4, "used_pct": 96.0}]
    messages = " | ".join(a["message"] for a in sm.compute_alerts(current, volumes, [{"cpu_pct": 95}] * 5))
    for expected in ("is 96.0% full", "of memory is available", "Swap is 80.0%", "Load average 9.0", "90% of its memory limit",
                     "CPU has averaged over 90%"):
        assert expected in messages
    assert [a["level"] for a in sm.compute_alerts(current, volumes, [])][0] == "bad"
    assert sm.compute_alerts(None, [], []) == []
    healthy = {"mem": {"total": 100, "available": 60, "used": 40, "used_pct": 40.0}, "swap": {"total": 10, "used": 0, "used_pct": 0.0},
               "cores": 4, "load": [0.5, 0.5, 0.5], "container": None}
    assert sm.compute_alerts(healthy, [{"path": "/", "total": 100, "used": 40, "free": 60, "used_pct": 40.0}],
                             [{"cpu_pct": 10}] * 5) == []


# --- the real probe --------------------------------------------------------------------------------------------------

def test_psutil_probe_reads_this_machine_and_the_sampler_runs_on_it(tmp_path):
    pytest.importorskip("psutil")
    probe = sm.PsutilProbe(read=lambda path: None)                  # no cgroup files here
    reading = probe.read()
    assert reading["cores"] >= 1 and reading["memory"]["total"] > 0 and reading["process"]["rss"] > 0
    assert reading["cpu"]["total"] > 0 and reading["container"] is None
    assert any(p["pid"] for p in probe.processes())
    assert probe.volumes([tmp_path, tmp_path])[0]["total"] > 0 and len(probe.volumes([tmp_path, tmp_path])) == 1
    real = sm.ServerSampler(probe, dir_paths=[])
    real.sample()
    time.sleep(1.1)
    sample = real.sample()
    assert sample["mem"]["used_pct"] is not None and sample["process"]["rss_bytes"] > 0
    assert 0 <= (sample["cpu_pct"] or 0) <= 100


def cgroup_reader(files):
    return lambda path: files.get(path)


def test_cgroup_v2_as_on_current_ubuntu_with_and_without_limits():
    pytest.importorskip("psutil")
    limited = {"/sys/fs/cgroup/memory.current": "2147483648", "/sys/fs/cgroup/memory.max": "4294967296",
               "/sys/fs/cgroup/cpu.max": "200000 100000", "/sys/fs/cgroup/cpu.stat": "usage_usec 5500000\nuser_usec 4000000\n"}
    box = sm.PsutilProbe(read=cgroup_reader(limited))._cgroup()
    assert box == {"memory": 2147483648, "memory_limit": 4294967296, "cpu_seconds": 5.5, "cpu_limit_cores": 2.0}
    unlimited = {**limited, "/sys/fs/cgroup/memory.max": "max", "/sys/fs/cgroup/cpu.max": "max 100000"}
    box = sm.PsutilProbe(read=cgroup_reader(unlimited))._cgroup()
    assert box["memory_limit"] is None and box["cpu_limit_cores"] is None


def test_cgroup_v1_and_no_cgroup():
    pytest.importorskip("psutil")
    v1 = {"/sys/fs/cgroup/memory/memory.usage_in_bytes": "1000", "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712",
          "/sys/fs/cgroup/cpuacct/cpuacct.usage": "2500000000", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1",
          "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}
    box = sm.PsutilProbe(read=cgroup_reader(v1))._cgroup()
    assert box == {"memory": 1000, "memory_limit": None, "cpu_seconds": 2.5, "cpu_limit_cores": None}
    v1["/sys/fs/cgroup/memory/memory.limit_in_bytes"] = "2147483648"
    v1["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"] = "150000"
    box = sm.PsutilProbe(read=cgroup_reader(v1))._cgroup()
    assert box["memory_limit"] == 2147483648 and box["cpu_limit_cores"] == 1.5
    assert sm.PsutilProbe(read=cgroup_reader({}))._cgroup() is None


def test_a_process_is_named_by_its_script_never_its_full_command_line():
    pytest.importorskip("psutil")

    class FakePsutil:
        class Error(Exception):
            pass

        @staticmethod
        def Process():
            return SimpleNamespace()

        @staticmethod
        def process_iter(attrs):
            def proc(pid, name, cmdline):
                return SimpleNamespace(info={"pid": pid, "name": name, "cpu_times": SimpleNamespace(user=1.0, system=0.5),
                                             "memory_info": SimpleNamespace(rss=2048), "cmdline": cmdline})
            return [proc(1, "python", ["/opt/venv/bin/python", "/app/scripts/hrrr_hourly_forecast.py", "--token=SECRET"]),
                    proc(2, "uvicorn", ["uvicorn", "main:app", "--password", "hunter2"])]

    names = [p["name"] for p in sm.PsutilProbe(psutil_module=FakePsutil).processes()]
    assert names == ["python hrrr_hourly_forecast.py", "uvicorn"]
    assert not any("SECRET" in n or "hunter2" in n for n in names)


# --- start, stop, and the admin endpoint -----------------------------------------------------------------------------

def test_the_thread_samples_saves_and_stops_cleanly(tmp_path, monkeypatch):
    path = tmp_path / "server.json"
    readings = [raw(cpu_total=1000 + 100 * i) for i in range(50)]
    s = sm.ServerSampler(FakeProbe(readings), dir_paths=[])
    s.start(path, interval_seconds=0.01)
    deadline = time.time() + 5
    while s.current is None and time.time() < deadline:
        time.sleep(0.01)
    s.stop(path)
    assert s.current is not None and path.is_file()
    assert json.loads(path.read_text())["minutes"]


def test_a_missing_psutil_stops_start_with_import_error_not_a_thread_that_fails_forever(tmp_path, monkeypatch):
    broken = sm.ServerSampler(dir_paths=[])
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)      # makes `import psutil` raise ImportError
    with pytest.raises(ImportError):
        broken.start(tmp_path / "x.json")


def admin_client(monkeypatch, collector, admin=True):
    from routers import admin_metrics

    monkeypatch.setattr(admin_metrics, "server_metrics", collector)
    app = FastAPI()
    app.include_router(admin_metrics.router)
    if admin:
        app.dependency_overrides[require_admin] = lambda: "admin@example.org"
    return TestClient(app)


def test_server_endpoint_returns_the_snapshot_and_requires_admin(monkeypatch):
    s, clock = run_minutes(2)
    body = admin_client(monkeypatch, s).get("/admin/metrics/server?window=30").json()
    assert body["current"]["cpu_pct"] == 30.0 and body["window_minutes"] == 30
    assert {"series", "volumes", "dirs", "processes", "alerts", "system"} <= set(body)
    assert body["system"]["kernel"] and body["system"]["machine"]
    assert admin_client(monkeypatch, s).get("/admin/metrics/server?window=1").status_code == 422
    assert admin_client(monkeypatch, s, admin=False).get("/admin/metrics/server").status_code == 401


def test_a_container_can_switch_metrics_off(tmp_path, monkeypatch):
    from core import metrics as api_metrics

    monkeypatch.setenv("SMF_METRICS_ENABLED", "0")
    monkeypatch.setenv("SMF_SERVER_METRICS_PATH", str(tmp_path / "server.json"))
    monkeypatch.setenv("SMF_API_METRICS_PATH", str(tmp_path / "api.json"))
    assert sm.start_server_metrics() == 0 and api_metrics.start_persistence() == 0
    sm.stop_server_metrics()
    api_metrics.stop_persistence()
    assert not (tmp_path / "server.json").exists() and not (tmp_path / "api.json").exists()   # nothing started, nothing saved
    for value in ("1", "true", "", "yes"):
        monkeypatch.setenv("SMF_METRICS_ENABLED", value)
        assert api_metrics.metrics_enabled() is True
    for value in ("0", "false", "No", "OFF"):
        monkeypatch.setenv("SMF_METRICS_ENABLED", value)
        assert api_metrics.metrics_enabled() is False

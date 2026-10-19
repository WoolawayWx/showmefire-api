"""Server resource metrics (CPU, memory, load, disk, network, the API process, big folders) for the admin metrics page.

A background thread samples the machine every `SAMPLE_SECONDS` and keeps a rolling 24 hours, averaged per minute, saved
to disk once a minute and reloaded at start-up (so a restart does not blank the charts). Rates (CPU %, bytes per second)
come from the difference between two samples of a running counter. The readings come from psutil (already a pinned
dependency) through a small `PsutilProbe`, kept apart from the sampling and averaging so those can be tested with a fake.

What this measures depends on where the API runs. In Docker, CPU, memory, disk and load are the HOST's (psutil reads the
host's /proc), network is the container's own interface, and the "container" figures (memory against its limit, CPU) come
from the cgroup files when they exist. The process list is only what the API's container can see, which includes the HRRR
build it starts. Cron jobs run in a different container and are not listed. A process is shown by name and the basename of
a .py script, never its full command line.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

SAMPLE_SECONDS = 10
KEEP_SECONDS = 24 * 3600
MINUTE = 60
SAVE_SECONDS = 60
DIR_SCAN_SECONDS = 600
DIR_SCAN_FIRST_DELAY = 20
DIR_SCAN_BUDGET_SECONDS = 15
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = ("data", "data/hrrr_hourly", "images", "cache", "archive", "logs")

# Per-minute series (each stored as [sum, count, max] so a minute holds the mean of its samples).
SERIES_KEYS = (
    "cpu_pct", "iowait_pct", "steal_pct", "load1", "mem_used_pct", "mem_used_bytes", "swap_used_pct",
    "proc_cpu_pct", "proc_rss_bytes", "container_mem_bytes", "container_cpu_pct",
    "disk_read_bps", "disk_write_bps", "net_rx_bps", "net_tx_bps",
)


def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


# --- the probe: every call into the operating system ---------------------------------------------------------------

class PsutilProbe:
    """Reads the machine with psutil. Nothing here does arithmetic beyond summing; the sampler does the rest."""

    def __init__(self, psutil_module: Any = None, read: Callable[[str], Optional[str]] = _read_text) -> None:
        if psutil_module is None:
            import psutil as psutil_module
        self.psutil = psutil_module
        self._read = read
        self._me = psutil_module.Process()

    def read(self) -> dict[str, Any]:
        ps = self.psutil
        times = ps.cpu_times()
        fields = {name: float(getattr(times, name, 0.0) or 0.0) for name in times._fields}
        total = sum(fields.values()) - fields.get("guest", 0.0) - fields.get("guest_nice", 0.0)   # guest is inside user
        idle, iowait = fields.get("idle", 0.0), fields.get("iowait", 0.0)
        memory, swap = ps.virtual_memory(), ps.swap_memory()
        try:
            load = tuple(round(v, 2) for v in ps.getloadavg())
        except (AttributeError, OSError):
            load = None
        try:
            io = ps.disk_io_counters()
            disk_io = {"read": int(io.read_bytes), "write": int(io.write_bytes)} if io else None
        except (OSError, RuntimeError):
            disk_io = None
        try:
            nics = {n: c for n, c in ps.net_io_counters(pernic=True).items() if not n.lower().startswith("lo")}
            net = {"rx": sum(c.bytes_recv for c in nics.values()), "tx": sum(c.bytes_sent for c in nics.values())} if nics else None
        except (OSError, RuntimeError):
            net = None
        me = self._me
        cpu = me.cpu_times()
        try:
            fds = me.num_fds()
        except (AttributeError, ps.Error):
            fds = getattr(me, "num_handles", lambda: None)()
        return {
            "cores": ps.cpu_count() or 1,
            "cpu": {"total": total, "busy": total - idle - iowait, "iowait": iowait, "steal": fields.get("steal", 0.0)},
            "load": load,
            "memory": {"total": int(memory.total), "available": int(memory.available), "used": int(memory.total - memory.available)},
            "swap": {"total": int(swap.total), "used": int(swap.used)},
            "disk_io": disk_io, "net": net,
            "process": {"cpu": float(cpu.user + cpu.system), "rss": int(me.memory_info().rss), "threads": me.num_threads(),
                        "fds": fds, "started": float(me.create_time())},
            "container": self._cgroup(),
            "host_uptime": max(0.0, time.time() - float(ps.boot_time())),
        }

    def processes(self) -> list[dict[str, Any]]:
        found = []
        for proc in self.psutil.process_iter(["pid", "name", "cpu_times", "memory_info", "cmdline"]):
            try:
                info = proc.info
                cpu, mem = info.get("cpu_times"), info.get("memory_info")
                if cpu is None or mem is None:
                    continue
                script = next((os.path.basename(a) for a in (info.get("cmdline") or []) if str(a).endswith(".py")), None)
                name = info.get("name") or "?"
                found.append({"pid": info["pid"], "name": f"{name} {script}" if script else name,
                              "cpu": float(cpu.user + cpu.system), "rss": int(mem.rss)})
            except (self.psutil.Error, KeyError, AttributeError):
                continue
        return found

    def volumes(self, paths: list[Path]) -> list[dict[str, Any]]:
        """Disk usage of the mounts holding `paths`, one entry per distinct mount."""
        seen, out = set(), []
        for path in paths:
            try:
                device = os.stat(path).st_dev
                if device in seen:
                    continue
                seen.add(device)
                usage = self.psutil.disk_usage(str(path))
                out.append({"path": str(path), "total": int(usage.total), "used": int(usage.used), "free": int(usage.free),
                            "used_pct": round(usage.percent, 1)})
            except OSError:
                continue
        return out

    def _cgroup(self) -> Optional[dict[str, Any]]:
        """This container's memory (and limit) and CPU seconds, from the cgroup files; None outside a container."""
        read = self._read

        def number(text: Optional[str]) -> Optional[int]:
            return int(text) if text and text.lstrip("-").isdigit() else None

        current = number(read("/sys/fs/cgroup/memory.current"))
        if current is not None:                                                   # cgroup v2
            limit = number(read("/sys/fs/cgroup/memory.max"))
            quota = (read("/sys/fs/cgroup/cpu.max") or "max 100000").split()
            usec = next((int(line.split()[1]) for line in (read("/sys/fs/cgroup/cpu.stat") or "").splitlines()
                         if line.startswith("usage_usec ")), None)
            cores = (int(quota[0]) / int(quota[1])) if quota[0].isdigit() and len(quota) > 1 and int(quota[1]) else None
            return {"memory": current, "memory_limit": limit, "cpu_seconds": usec / 1e6 if usec is not None else None,
                    "cpu_limit_cores": cores}
        current = number(read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        if current is None:
            return None
        limit = number(read("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        nanos = number(read("/sys/fs/cgroup/cpuacct/cpuacct.usage"))
        quota, period = number(read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")), number(read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        return {"memory": current, "memory_limit": limit if limit and limit < (1 << 60) else None,
                "cpu_seconds": nanos / 1e9 if nanos is not None else None,
                "cpu_limit_cores": quota / period if quota and quota > 0 and period else None}


# --- folder sizes --------------------------------------------------------------------------------------------------

def dir_size(path: Path, budget_seconds: float = DIR_SCAN_BUDGET_SECONDS, clock: Callable[[], float] = time.monotonic) -> dict:
    """Total bytes and file count under `path`, giving up (partial=True) after `budget_seconds`. Symlinks are not followed."""
    deadline = clock() + budget_seconds
    total = files = 0
    pending = [str(path)]
    partial = False
    while pending:
        if clock() > deadline:
            partial = True
            break
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return {"bytes": total, "files": files, "partial": partial}


# --- the sampler ---------------------------------------------------------------------------------------------------

def _rate(now: float, before: Optional[float], after: Optional[float], then: float) -> Optional[float]:
    """Per-second change of a running counter, or None if either reading is missing or the counter went backwards."""
    if before is None or after is None or now - then < 1 or after < before:
        return None
    return (after - before) / (now - then)


class ServerSampler:
    def __init__(self, probe: Any = None, clock: Callable[[], float] = time.time,
                 dir_paths: Optional[list[Path]] = None) -> None:
        self._probe = probe
        self._clock = clock
        self._lock = threading.Lock()
        self._minutes: dict[int, dict[str, list[float]]] = {}
        self._prev: Optional[dict[str, Any]] = None
        self._prev_pids: dict[int, tuple[float, float]] = {}
        self.current: Optional[dict[str, Any]] = None
        self.processes: list[dict[str, Any]] = []
        self.dirs: list[dict[str, Any]] = []
        self._dir_paths = dir_paths if dir_paths is not None else self._default_dirs()
        self._dirs_due = clock() + DIR_SCAN_FIRST_DELAY
        self._last_prune = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @staticmethod
    def _default_dirs() -> list[Path]:
        listed = os.environ.get("SMF_METRICS_DIRS", "")
        names = [n.strip() for n in listed.split(",") if n.strip()] or list(DEFAULT_DIRS)
        return [PROJECT_ROOT / name for name in names]

    @property
    def probe(self) -> Any:
        if self._probe is None:
            self._probe = PsutilProbe()
        return self._probe

    # -- sampling ---------------------------------------------------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        """Take one reading, work out the rates since the last one, add it to the current minute and return it."""
        now = self._clock()
        raw = self.probe.read()
        prev = self._prev
        then = prev["time"] if prev else now
        elapsed = now - then if prev else 0

        cpu, pcpu = raw["cpu"], prev["raw"]["cpu"] if prev else None
        cpu_pct = iowait_pct = steal_pct = None
        if pcpu and cpu["total"] > pcpu["total"] and elapsed >= 1:
            span = cpu["total"] - pcpu["total"]
            cpu_pct = round(max(0.0, min(100.0, 100 * (cpu["busy"] - pcpu["busy"]) / span)), 1)
            iowait_pct = round(max(0.0, 100 * (cpu["iowait"] - pcpu["iowait"]) / span), 1)
            steal_pct = round(max(0.0, 100 * (cpu["steal"] - pcpu["steal"]) / span), 1)

        def delta(path: tuple[str, ...]) -> tuple[Optional[float], Optional[float]]:
            def dig(source: Optional[dict]) -> Optional[float]:
                for key in path:
                    source = source.get(key) if isinstance(source, dict) else None
                return source
            return (dig(prev["raw"]) if prev else None), dig(raw)

        def rate(path: tuple[str, ...]) -> Optional[float]:
            before, after = delta(path)
            return _rate(now, before, after, then) if prev else None

        proc_cpu = rate(("process", "cpu"))
        container_cpu = rate(("container", "cpu_seconds"))
        memory, swap, proc, box = raw["memory"], raw["swap"], raw["process"], raw.get("container")
        mem_pct = round(100 * memory["used"] / memory["total"], 1) if memory["total"] else None
        sample = {
            "time": now, "cores": raw["cores"], "cpu_pct": cpu_pct, "iowait_pct": iowait_pct, "steal_pct": steal_pct,
            "load": list(raw["load"]) if raw.get("load") else None,
            "mem": {"total": memory["total"], "used": memory["used"], "available": memory["available"], "used_pct": mem_pct},
            "swap": {"total": swap["total"], "used": swap["used"],
                     "used_pct": round(100 * swap["used"] / swap["total"], 1) if swap["total"] else 0.0},
            "disk_read_bps": rate(("disk_io", "read")), "disk_write_bps": rate(("disk_io", "write")),
            "net_rx_bps": rate(("net", "rx")), "net_tx_bps": rate(("net", "tx")),
            "process": {"cpu_pct": round(100 * proc_cpu, 1) if proc_cpu is not None else None, "rss_bytes": proc["rss"],
                        "threads": proc["threads"], "fds": proc["fds"], "uptime_seconds": int(max(0, now - proc["started"]))},
            "container": ({"memory_bytes": box["memory"], "memory_limit_bytes": box["memory_limit"],
                           "cpu_pct": round(100 * container_cpu, 1) if container_cpu is not None else None,
                           "cpu_limit_cores": box["cpu_limit_cores"]} if box else None),
            "host_uptime_seconds": int(raw["host_uptime"]),
        }
        self._add_to_minute(now, sample)
        with self._lock:
            self.current = sample
            self._prev = {"time": now, "raw": raw}
        self._scan_processes(now)
        return sample

    def _add_to_minute(self, now: float, sample: dict[str, Any]) -> None:
        values = {
            "cpu_pct": sample["cpu_pct"], "iowait_pct": sample["iowait_pct"], "steal_pct": sample["steal_pct"],
            "load1": (sample["load"] or [None])[0], "mem_used_pct": sample["mem"]["used_pct"],
            "mem_used_bytes": sample["mem"]["used"], "swap_used_pct": sample["swap"]["used_pct"],
            "proc_cpu_pct": sample["process"]["cpu_pct"], "proc_rss_bytes": sample["process"]["rss_bytes"],
            "container_mem_bytes": (sample["container"] or {}).get("memory_bytes"),
            "container_cpu_pct": (sample["container"] or {}).get("cpu_pct"),
            "disk_read_bps": sample["disk_read_bps"], "disk_write_bps": sample["disk_write_bps"],
            "net_rx_bps": sample["net_rx_bps"], "net_tx_bps": sample["net_tx_bps"],
        }
        minute = int(now // MINUTE)
        with self._lock:
            row = self._minutes.setdefault(minute, {})
            for key, value in values.items():
                if value is None:
                    continue
                cell = row.setdefault(key, [0.0, 0, float("-inf")])
                cell[0] += value
                cell[1] += 1
                cell[2] = max(cell[2], value)
            if minute != self._last_prune:
                self._last_prune = minute
                cutoff = int((now - KEEP_SECONDS) // MINUTE)
                for old in [m for m in self._minutes if m < cutoff]:
                    del self._minutes[old]

    def _scan_processes(self, now: float, top_cpu: int = 6, top_memory: int = 4) -> None:
        try:
            found = self.probe.processes()
        except Exception:  # noqa: BLE001 - the list is a nicety; never let it break sampling
            logger.debug("process scan failed", exc_info=True)
            return
        previous, current_pids, rows = self._prev_pids, {}, []
        for proc in found:
            current_pids[proc["pid"]] = (now, proc["cpu"])
            before = previous.get(proc["pid"])
            pct = None
            if before and now - before[0] >= 1 and proc["cpu"] >= before[1]:
                pct = round(100 * (proc["cpu"] - before[1]) / (now - before[0]), 1)
            rows.append({"pid": proc["pid"], "name": proc["name"], "cpu_pct": pct, "rss_bytes": proc["rss"]})
        self._prev_pids = current_pids
        by_cpu = sorted((r for r in rows if r["cpu_pct"]), key=lambda r: r["cpu_pct"], reverse=True)[:top_cpu]
        by_memory = sorted(rows, key=lambda r: r["rss_bytes"], reverse=True)[:top_memory]
        merged = {r["pid"]: r for r in by_cpu + by_memory}
        with self._lock:
            self.processes = sorted(merged.values(), key=lambda r: (r["cpu_pct"] or 0, r["rss_bytes"]), reverse=True)

    # -- folders ----------------------------------------------------------------------------------------------------

    def refresh_dirs(self, now: Optional[float] = None) -> None:
        """Size every watched folder (each within its time budget). Slow on big trees, so it runs every 10 minutes."""
        stamp = self._clock() if now is None else now
        rows = []
        for path in self._dir_paths:
            if not path.is_dir():
                continue
            rows.append({"path": str(path.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in path.parents else str(path),
                         **dir_size(path), "scanned_at": int(stamp)})
        with self._lock:
            self.dirs = rows
        self._dirs_due = stamp + DIR_SCAN_SECONDS

    # -- reading ----------------------------------------------------------------------------------------------------

    def snapshot(self, window_minutes: int = 60, now: Optional[float] = None) -> dict[str, Any]:
        window = max(5, min(1440, int(window_minutes)))
        now = self._clock() if now is None else now
        step = 1 if window <= 60 else 5 if window <= 360 else 15
        last_minute = int(now // MINUTE)
        first_minute = last_minute - window + 1
        with self._lock:
            minutes = {m: {k: list(v) for k, v in row.items()} for m, row in self._minutes.items()
                       if first_minute <= m <= last_minute}
            current, processes, dirs = self.current, list(self.processes), list(self.dirs)
        points = []
        for start in range(first_minute - (first_minute % step), last_minute + 1, step):
            point: dict[str, Any] = {"time": start * MINUTE}
            sums: dict[str, list[float]] = {}
            for minute in range(start, start + step):
                for key, (total, count, peak) in minutes.get(minute, {}).items():
                    cell = sums.setdefault(key, [0.0, 0, float("-inf")])
                    cell[0] += total
                    cell[1] += count
                    cell[2] = max(cell[2], peak)
            for key in SERIES_KEYS:
                cell = sums.get(key)
                point[key] = round(cell[0] / cell[1], 2) if cell and cell[1] else None
            point["cpu_pct_max"] = round(sums["cpu_pct"][2], 1) if "cpu_pct" in sums else None
            points.append(point)
        volumes = []
        try:
            volumes = self.probe.volumes([PROJECT_ROOT, *self._dir_paths[:1]])
        except Exception:  # noqa: BLE001
            logger.debug("volume check failed", exc_info=True)
        # The kernel is the host's even inside Docker (containers share it); the OS release is not, so it is not shown.
        system = {"platform": platform.system(), "kernel": platform.release(), "machine": platform.machine()}
        return {"current": current, "series": {"step_minutes": step, "points": points}, "volumes": volumes,
                "dirs": dirs, "processes": processes, "alerts": compute_alerts(current, volumes, points),
                "system": system, "window_minutes": window, "sample_seconds": SAMPLE_SECONDS}

    # -- persistence ------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"version": 1, "saved_at": int(self._clock()),
                    "minutes": {str(m): row for m, row in self._minutes.items()}}

    def load_dict(self, data: dict[str, Any]) -> int:
        oldest = int((self._clock() - KEEP_SECONDS) // MINUTE)
        loaded = 0
        with self._lock:
            for text, row in (data.get("minutes") or {}).items():
                if not (text.isdigit() and int(text) >= oldest and isinstance(row, dict)):
                    continue
                target = self._minutes.setdefault(int(text), {})
                for key, cell in row.items():
                    if key in SERIES_KEYS and isinstance(cell, list) and len(cell) == 3 and key not in target:
                        target[key] = [float(cell[0]), int(cell[1]), float(cell[2])]
                loaded += 1
        return loaded

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        os.replace(temporary, path)

    def load(self, path: Path) -> int:
        try:
            return self.load_dict(json.loads(Path(path).read_text()))
        except (OSError, ValueError, AttributeError):
            return 0

    # -- the background thread --------------------------------------------------------------------------------------

    def start(self, path: Optional[Path] = None, interval_seconds: float = SAMPLE_SECONDS) -> int:
        """Reload the saved history and start sampling (and saving every minute). Returns the minutes reloaded."""
        _ = self.probe                      # raises ImportError here, not inside the thread, if psutil is missing
        loaded = self.load(path) if path else 0
        self._stop.clear()

        def run() -> None:
            last_save = time.monotonic()
            while True:
                try:
                    self.sample()
                    if self._clock() >= self._dirs_due:
                        self.refresh_dirs()
                except Exception:  # noqa: BLE001 - keep sampling whatever one reading did
                    logger.warning("server metrics sample failed", exc_info=True)
                if path and time.monotonic() - last_save >= SAVE_SECONDS:
                    last_save = time.monotonic()
                    try:
                        self.save(path)
                    except OSError:
                        logger.warning("could not save server metrics", exc_info=True)
                if self._stop.wait(interval_seconds):
                    return

        self._thread = threading.Thread(target=run, name="server-metrics", daemon=True)
        self._thread.start()
        return loaded

    def stop(self, path: Optional[Path] = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if path:
            try:
                self.save(path)
            except OSError:
                logger.warning("could not save server metrics at shutdown", exc_info=True)


# --- warnings the page shows at the top ----------------------------------------------------------------------------

def compute_alerts(current: Optional[dict], volumes: list[dict], points: list[dict]) -> list[dict[str, str]]:
    alerts: list[dict[str, str]] = []
    for volume in volumes:
        free_pct = 100 - volume["used_pct"]
        if free_pct < 5:
            alerts.append({"level": "bad", "message": f"Disk {volume['path']} is {volume['used_pct']}% full ({free_pct:.1f}% free)."})
        elif free_pct < 10:
            alerts.append({"level": "warn", "message": f"Disk {volume['path']} is {volume['used_pct']}% full."})
    if current:
        mem = current["mem"]
        if mem["total"] and mem["available"] / mem["total"] < 0.05:
            alerts.append({"level": "bad", "message": f"Only {100 * mem['available'] / mem['total']:.1f}% of memory is available."})
        elif mem["total"] and mem["available"] / mem["total"] < 0.10:
            alerts.append({"level": "warn", "message": f"Only {100 * mem['available'] / mem['total']:.1f}% of memory is available."})
        if current["swap"]["total"] and current["swap"]["used_pct"] > 50:
            alerts.append({"level": "warn", "message": f"Swap is {current['swap']['used_pct']}% used."})
        box = current.get("container")
        if box and box.get("memory_limit_bytes") and box["memory_bytes"] / box["memory_limit_bytes"] > 0.9:
            alerts.append({"level": "bad", "message": "The API container is using over 90% of its memory limit."})
        if current["load"] and current["load"][0] > 2 * current["cores"]:
            alerts.append({"level": "warn", "message": f"Load average {current['load'][0]} is over twice the {current['cores']} cores."})
    recent = [p["cpu_pct"] for p in points[-5:] if p.get("cpu_pct") is not None]
    if len(recent) >= 3 and sum(recent) / len(recent) > 90:
        alerts.append({"level": "warn", "message": "CPU has averaged over 90% for the last few minutes."})
    return alerts


server_metrics = ServerSampler()


def server_metrics_path() -> Path:
    return Path(os.environ.get("SMF_SERVER_METRICS_PATH", "data/server_metrics.json"))


def start_server_metrics() -> int:
    """Start the sampler (reloading the saved history). Returns the minutes reloaded; 0 if psutil is unavailable."""
    from core.metrics import metrics_enabled

    if not metrics_enabled():
        return 0
    try:
        return server_metrics.start(server_metrics_path())
    except ImportError:
        logger.warning("psutil is not installed; server metrics are off")
        return 0


def stop_server_metrics() -> None:
    from core.metrics import metrics_enabled

    if metrics_enabled():
        server_metrics.stop(server_metrics_path())

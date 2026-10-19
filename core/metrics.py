"""In-process API request metrics for the admin metrics page.

A pure ASGI middleware (`MetricsMiddleware`) times every HTTP request and records it, by route template (e.g.
`GET /hrrr-hourly/runs/{cycle}/zarr/{path}`, never the raw URL, so the number of routes stays bounded), into two
rolling stores that cover the last 24 hours:

* one row per minute for the whole API (the charts), and
* one row per route per 10 minutes (the table of routes).

A row is: request count, 2xx/3xx/4xx/5xx counts, total and maximum latency, response bytes, and a latency histogram,
from which percentiles are estimated. Latency is the time until the response starts (the server's own work), not the
time a slow client takes to download a large body.

The API runs as one uvicorn process, so these counters are the whole picture; cron jobs and the HRRR builder
subprocesses are separate processes and are not counted. A snapshot is saved to disk every minute and reloaded at
start-up, so a restart does not blank the page. Nothing here is a security log: no client addresses are kept.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

LATENCY_EDGES_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)   # buckets: <=5, <=10 ... <=10000, >10000
N_BUCKETS = len(LATENCY_EDGES_MS) + 1
# Row layout: [count, 2xx, 3xx, 4xx, 5xx, latency_sum_ms, latency_max_ms, bytes, *histogram]
COUNT, C2XX, C3XX, C4XX, C5XX, SUM_MS, MAX_MS, BYTES, HIST = range(9)
ROW_LEN = HIST + N_BUCKETS

KEEP_SECONDS = 24 * 3600
MINUTE = 60
ROUTE_BUCKET = 600
MAX_ROUTE_KEYS = 400
OTHER = "(other)"
UNMATCHED = "(static/unmatched)"
SKIP_PREFIXES = ("/admin/metrics",)   # the metrics page must not measure itself


def _new_row() -> list[float]:
    return [0] * ROW_LEN


def _add(row: list[float], status: int, latency_ms: float, nbytes: int) -> None:
    row[COUNT] += 1
    klass = status // 100
    if klass in (2, 3, 4, 5):
        row[{2: C2XX, 3: C3XX, 4: C4XX, 5: C5XX}[klass]] += 1
    row[SUM_MS] += latency_ms
    row[MAX_MS] = max(row[MAX_MS], latency_ms)
    row[BYTES] += nbytes
    row[HIST + bisect.bisect_left(LATENCY_EDGES_MS, latency_ms)] += 1


def _merge(into: list[float], row: list[float]) -> None:
    for index in range(ROW_LEN):
        if index == MAX_MS:
            into[index] = max(into[index], row[index])
        else:
            into[index] += row[index]


def percentile(row: list[float], quantile: float) -> Optional[float]:
    """Latency (ms) at `quantile` (0-1), estimated from the histogram by interpolating inside the bucket."""
    total = sum(row[HIST:])
    if not total:
        return None
    target = quantile * total
    seen = 0.0
    for index in range(N_BUCKETS):
        in_bucket = row[HIST + index]
        if in_bucket and seen + in_bucket >= target:
            lower = LATENCY_EDGES_MS[index - 1] if index else 0
            upper = LATENCY_EDGES_MS[index] if index < len(LATENCY_EDGES_MS) else max(row[MAX_MS], lower)
            upper = min(upper, row[MAX_MS]) if row[MAX_MS] >= lower else upper
            return round(lower + (upper - lower) * (target - seen) / in_bucket, 1)
        seen += in_bucket
    return round(row[MAX_MS], 1)


def _summary(row: list[float]) -> dict[str, Any]:
    count = int(row[COUNT])
    return {
        "requests": count, "ok": int(row[C2XX] + row[C3XX]), "client_errors": int(row[C4XX]),
        "server_errors": int(row[C5XX]),
        "avg_ms": round(row[SUM_MS] / count, 1) if count else None,
        "p50_ms": percentile(row, 0.50), "p95_ms": percentile(row, 0.95), "p99_ms": percentile(row, 0.99),
        "max_ms": round(row[MAX_MS], 1) if count else None, "bytes": int(row[BYTES]),
    }


class ApiMetrics:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._minutes: dict[int, list[float]] = {}
        self._routes: dict[int, dict[str, list[float]]] = {}
        self._route_keys: set[str] = set()
        self._last_prune = 0
        self.started_at = clock()
        self.inflight = 0

    # -- recording --------------------------------------------------------------------------------------------------

    def begin(self) -> None:
        with self._lock:
            self.inflight += 1

    def end(self) -> None:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)

    def record(self, method: str, route: str, status: int, latency_ms: float, nbytes: int = 0) -> None:
        now = self._clock()
        minute, bucket = int(now // MINUTE), int(now // ROUTE_BUCKET)
        key = f"{method} {route}"
        with self._lock:
            _add(self._minutes.setdefault(minute, _new_row()), status, latency_ms, nbytes)
            if key not in self._route_keys:
                if len(self._route_keys) >= MAX_ROUTE_KEYS:
                    key = OTHER
                self._route_keys.add(key)
            _add(self._routes.setdefault(bucket, {}).setdefault(key, _new_row()), status, latency_ms, nbytes)
            if minute != self._last_prune:
                self._last_prune = minute
                self._prune(now)

    def _prune(self, now: float) -> None:
        oldest_minute = int((now - KEEP_SECONDS) // MINUTE)
        for minute in [m for m in self._minutes if m < oldest_minute]:
            del self._minutes[minute]
        oldest_bucket = int((now - KEEP_SECONDS) // ROUTE_BUCKET)
        for bucket in [b for b in self._routes if b < oldest_bucket]:
            del self._routes[bucket]
        self._route_keys = {key for rows in self._routes.values() for key in rows}

    # -- reading ----------------------------------------------------------------------------------------------------

    def snapshot(self, window_minutes: int = 60, now: Optional[float] = None, max_routes: int = 100) -> dict[str, Any]:
        """Everything the metrics page shows for the last `window_minutes` (5 to 1440)."""
        window = max(5, min(1440, int(window_minutes)))
        now = self._clock() if now is None else now
        step = 1 if window <= 60 else 5 if window <= 360 else 15
        last_minute = int(now // MINUTE)
        first_minute = last_minute - window + 1
        with self._lock:
            minutes = {m: list(row) for m, row in self._minutes.items() if first_minute <= m <= last_minute}
            first_bucket = int((first_minute * MINUTE) // ROUTE_BUCKET)
            routes: dict[str, list[float]] = {}
            for bucket, rows in self._routes.items():
                if first_bucket <= bucket <= int(now // ROUTE_BUCKET):
                    for key, row in rows.items():
                        _merge(routes.setdefault(key, _new_row()), row)
            inflight, started = self.inflight, self.started_at

        total = _new_row()
        for row in minutes.values():
            _merge(total, row)
        overview = _summary(total)
        overview.update({
            "window_minutes": window, "requests_per_second": round(total[COUNT] / (window * 60), 3),
            "server_error_rate": round(total[C5XX] / total[COUNT], 4) if total[COUNT] else 0.0,
            "client_error_rate": round(total[C4XX] / total[COUNT], 4) if total[COUNT] else 0.0,
            "inflight": inflight, "uptime_seconds": int(now - started), "generated_at": int(now),
        })

        points = []
        start = first_minute - (first_minute % step)
        for point_start in range(start, last_minute + 1, step):
            row = _new_row()
            for minute in range(point_start, point_start + step):
                if minute in minutes:
                    _merge(row, minutes[minute])
            point = _summary(row)
            point["time"] = point_start * MINUTE
            points.append(point)

        route_rows = []
        for key, row in routes.items():
            summary = _summary(row)
            method, _, path = key.partition(" ")
            summary.update({"route": path or key, "method": method if path else ""})
            route_rows.append(summary)
        route_rows.sort(key=lambda r: r["requests"], reverse=True)
        return {"overview": overview, "series": {"step_minutes": step, "points": points},
                "routes": route_rows[:max_routes], "route_count": len(route_rows)}

    # -- persistence ------------------------------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"version": 1, "saved_at": int(self._clock()),
                    "minutes": {str(m): row for m, row in self._minutes.items()},
                    "routes": {str(b): rows for b, rows in self._routes.items()}}

    def load_dict(self, data: dict[str, Any]) -> int:
        """Merge a saved snapshot, dropping anything older than 24 hours. Returns the number of minute rows loaded."""
        now = self._clock()
        oldest_minute, oldest_bucket = int((now - KEEP_SECONDS) // MINUTE), int((now - KEEP_SECONDS) // ROUTE_BUCKET)
        loaded = 0
        with self._lock:
            for text, row in (data.get("minutes") or {}).items():
                if text.isdigit() and int(text) >= oldest_minute and isinstance(row, list) and len(row) == ROW_LEN:
                    _merge(self._minutes.setdefault(int(text), _new_row()), row)
                    loaded += 1
            for text, rows in (data.get("routes") or {}).items():
                if text.isdigit() and int(text) >= oldest_bucket and isinstance(rows, dict):
                    for key, row in rows.items():
                        if isinstance(row, list) and len(row) == ROW_LEN:
                            _merge(self._routes.setdefault(int(text), {}).setdefault(key, _new_row()), row)
            self._route_keys = {key for rows in self._routes.values() for key in rows}
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


metrics = ApiMetrics()


def metrics_path() -> Path:
    return Path(os.environ.get("SMF_API_METRICS_PATH", "data/api_metrics.json"))


def metrics_enabled() -> bool:
    """SMF_METRICS_ENABLED=0 turns off the saving and the server sampler in a container (for example a worker that shares
    the data volume with the API and must not overwrite its snapshot). The middleware still counts, harmlessly."""
    return os.environ.get("SMF_METRICS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")


# --- the middleware ------------------------------------------------------------------------------------------------

def route_label(scope: dict) -> str:
    """The matched route's template, or a single bucket for static files and unknown paths."""
    return getattr(scope.get("route"), "path", None) or UNMATCHED


class MetricsMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware), so streaming and file responses are untouched."""

    def __init__(self, app, collector: Optional[ApiMetrics] = None, skip_prefixes: tuple = SKIP_PREFIXES) -> None:
        self.app = app
        self.collector = collector or metrics
        self.skip_prefixes = skip_prefixes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path", "").startswith(self.skip_prefixes):
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status, latency_ms, sent = 500, None, 0   # 500 stands if the app raises before it starts a response

        async def send_wrapper(message):
            nonlocal status, latency_ms, sent
            if message["type"] == "http.response.start":
                status = message["status"]
                latency_ms = (time.perf_counter() - started) * 1000
            elif message["type"] == "http.response.body":
                sent += len(message.get("body") or b"")
            await send(message)

        self.collector.begin()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.collector.end()
            if latency_ms is None:
                latency_ms = (time.perf_counter() - started) * 1000
            try:
                self.collector.record(scope.get("method", "GET"), route_label(scope), status, latency_ms, sent)
            except Exception:  # noqa: BLE001 - metrics must never break a request
                logger.exception("could not record API metrics")


# --- saving to disk ------------------------------------------------------------------------------------------------

_saver: Optional[threading.Thread] = None
_stop = threading.Event()


def start_persistence(interval_seconds: int = 60, collector: Optional[ApiMetrics] = None,
                      path: Optional[Path] = None) -> int:
    """Reload the saved snapshot and start saving a fresh one every `interval_seconds`. Returns minutes reloaded."""
    global _saver
    if not metrics_enabled():
        return 0
    collector, path = collector or metrics, path or metrics_path()
    loaded = collector.load(path)
    _stop.clear()

    def run() -> None:
        while not _stop.wait(interval_seconds):
            try:
                collector.save(path)
            except OSError:
                logger.exception("could not save API metrics")

    _saver = threading.Thread(target=run, name="api-metrics-saver", daemon=True)
    _saver.start()
    return loaded


def stop_persistence(collector: Optional[ApiMetrics] = None, path: Optional[Path] = None) -> None:
    """Stop the saver and write one last snapshot."""
    global _saver
    if not metrics_enabled():
        return
    _stop.set()
    if _saver is not None:
        _saver.join(timeout=5)
        _saver = None
    try:
        (collector or metrics).save(path or metrics_path())
    except OSError:
        logger.exception("could not save API metrics at shutdown")

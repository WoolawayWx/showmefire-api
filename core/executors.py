"""Shared process pool for CPU-heavy scheduled jobs (RTMA/spread-rate grid math,
raster generation, the forecast-v1 NWP pipeline) so concurrent job bursts spread
across cores instead of serializing on the GIL inside `asyncio.to_thread` worker
threads - and so a job that blows up its memory only takes down its own worker
process, not the API server itself.

Created lazily but expected to be warmed up once at scheduler startup
(`core.scheduler.create_scheduler`) so the fork happens before the process
accumulates many threads.

A worker that's killed out-of-band (OOM, segfault) leaves the executor
permanently broken - every future submission raises BrokenProcessPool until
the whole container restarts. `run_in_process_pool`/`run_in_process_pool_async`
detect that and transparently recreate the pool so scheduled jobs keep
running on their next tick instead of failing forever.

A worker can also hang instead of crashing - e.g. a stale SQLite WAL/SHM
handle left over from a prior crash can deadlock a later query forever.
That looks perfectly healthy to ProcessPoolExecutor (the process is alive,
just idle), so BrokenProcessPool never fires and `.result()` blocks
indefinitely - which also permanently occupies the worker's pool slot and
(for jobs with max_instances=1) blocks every future scheduled tick.
`run_in_process_pool`/`run_in_process_pool_async` take a `timeout` so a
hang gets converted into a clean failure: the hung worker is force-killed
and the pool recreated instead of being abandoned to leak forever.
"""
import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool

logger = logging.getLogger(__name__)

_process_pool: ProcessPoolExecutor | None = None

# Applies whenever a caller doesn't pass its own `timeout` - a safety net so a
# new call site can't reintroduce an unbounded hang just by omitting it.
# RTMA/spread-rate grid jobs finish in well under a minute; this is generous
# for those while still catching a stuck job long before it blocks hours of
# scheduled ticks. The forecast-v1 pipeline explicitly overrides this with
# its own (much longer) SMF_FORECAST_V1_JOB_TIMEOUT_SECONDS ceiling.
DEFAULT_JOB_TIMEOUT_SECONDS = int(os.getenv("SMF_CPU_POOL_JOB_TIMEOUT_SECONDS", "1800"))


def get_process_pool() -> ProcessPoolExecutor:
    global _process_pool
    if _process_pool is None:
        # Grid jobs are memory-heavy. Keep the default bounded so a host with
        # many CPUs does not start enough simultaneous workers to breach the
        # container memory limit. Deployments can still opt in to more.
        default_workers = min(2, max(1, (os.cpu_count() or 2) - 2))
        workers = max(1, int(os.getenv("SMF_CPU_POOL_WORKERS", str(default_workers))))
        logger.info("Starting CPU job process pool with %d workers", workers)
        _process_pool = ProcessPoolExecutor(max_workers=workers)
    return _process_pool


def _kill_hung_workers(pool: ProcessPoolExecutor) -> None:
    """Force-kill a pool's worker processes.

    A hung (not crashed) worker never responds to cancel_futures or a normal
    shutdown - it's still alive, just stuck. Left alone it leaks as an
    orphaned process holding whatever stale file/DB handles caused the hang.
    `_processes` is a private ProcessPoolExecutor attribute; there is no
    public API for "kill the worker running this future".
    """
    for process in list((getattr(pool, "_processes", None) or {}).values()):
        try:
            process.kill()
        except Exception:
            logger.exception("Failed to kill hung process pool worker pid=%s", getattr(process, "pid", "?"))


def shutdown_process_pool():
    """Tear down the shared pool and make sure its OS processes actually die.

    `ProcessPoolExecutor.shutdown(wait=False)` only stops the pool from
    accepting new work - it does not signal workers that are still running
    (or hung on) a task. Once `_process_pool` is reassigned, nothing in the
    app still references those processes, so a still-running or hung worker
    is silently orphaned and keeps holding whatever memory it had at fork
    time forever. Killing every worker here, before dropping the reference,
    is what actually reclaims that memory.
    """
    global _process_pool
    if _process_pool is not None:
        _kill_hung_workers(_process_pool)
        _process_pool.shutdown(wait=False, cancel_futures=True)
        _process_pool = None


def run_in_process_pool(func, *args, timeout: float | None = DEFAULT_JOB_TIMEOUT_SECONDS, **kwargs):
    """Blocking submit+result against the shared pool, for callers already off
    the event loop (a background thread). Recreates the pool once if a prior
    job left it broken. `timeout` (seconds) bounds how long a single job may
    run - on expiry the hung worker is killed and the pool recreated so the
    pool's slot and any future callers aren't blocked forever. Pass
    `timeout=None` to opt out entirely."""
    try:
        return get_process_pool().submit(func, *args, **kwargs).result(timeout=timeout)
    except BrokenProcessPool:
        logger.error("Process pool worker died unexpectedly; recreating pool and retrying once")
        shutdown_process_pool()
        return get_process_pool().submit(func, *args, **kwargs).result(timeout=timeout)
    except FuturesTimeoutError:
        logger.error("Process pool job exceeded %ss timeout; killing hung worker(s)", timeout)
        shutdown_process_pool()
        raise


async def run_in_process_pool_async(func, *args, timeout: float | None = DEFAULT_JOB_TIMEOUT_SECONDS):
    """Async equivalent of `run_in_process_pool` for callers on the event loop."""
    loop = asyncio.get_running_loop()
    pool = get_process_pool()
    try:
        return await asyncio.wait_for(loop.run_in_executor(pool, func, *args), timeout=timeout)
    except BrokenProcessPool:
        logger.error("Process pool worker died unexpectedly; recreating pool and retrying once")
        shutdown_process_pool()
        return await loop.run_in_executor(get_process_pool(), func, *args)
    except asyncio.TimeoutError:
        logger.error("Process pool job exceeded %ss timeout; killing hung worker(s)", timeout)
        shutdown_process_pool()
        raise

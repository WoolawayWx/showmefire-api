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
"""
import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

logger = logging.getLogger(__name__)

_process_pool: ProcessPoolExecutor | None = None


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


def shutdown_process_pool():
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=False, cancel_futures=True)
        _process_pool = None


def run_in_process_pool(func, *args, **kwargs):
    """Blocking submit+result against the shared pool, for callers already off
    the event loop (a background thread). Recreates the pool once if a prior
    job left it broken."""
    try:
        return get_process_pool().submit(func, *args, **kwargs).result()
    except BrokenProcessPool:
        logger.error("Process pool worker died unexpectedly; recreating pool and retrying once")
        shutdown_process_pool()
        return get_process_pool().submit(func, *args, **kwargs).result()


async def run_in_process_pool_async(func, *args):
    """Async equivalent of `run_in_process_pool` for callers on the event loop."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(get_process_pool(), func, *args)
    except BrokenProcessPool:
        logger.error("Process pool worker died unexpectedly; recreating pool and retrying once")
        shutdown_process_pool()
        return await loop.run_in_executor(get_process_pool(), func, *args)

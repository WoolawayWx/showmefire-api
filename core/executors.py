"""Shared process pool for CPU-heavy scheduled jobs (RTMA/spread-rate grid math,
raster generation) so concurrent job bursts spread across cores instead of
serializing on the GIL inside `asyncio.to_thread` worker threads.

Created lazily but expected to be warmed up once at scheduler startup
(`core.scheduler.create_scheduler`) so the fork happens before the process
accumulates many threads.
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

_process_pool: ProcessPoolExecutor | None = None


def get_process_pool() -> ProcessPoolExecutor:
    global _process_pool
    if _process_pool is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
        logger.info("Starting CPU job process pool with %d workers", workers)
        _process_pool = ProcessPoolExecutor(max_workers=workers)
    return _process_pool


def shutdown_process_pool():
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=False, cancel_futures=True)
        _process_pool = None

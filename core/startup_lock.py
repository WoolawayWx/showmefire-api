"""A host-wide lock for database setup.

The API (lifespan) and the scheduler worker both call `init_database()` when they start. Most of it is idempotent, but the
forced `forecasts` table rebuild is a multi-step drop/rename that must not run twice at once, so both take this lock around
the whole setup. A killed process drops an flock automatically, so it can never stay stuck. No-op where fcntl is missing
(Windows development).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None


def _path() -> Path:
    return Path(os.getenv("DATA_DIR", "/app/data" if Path("/app").exists() else "data")) / ".startup.lock"


@contextmanager
def startup_lock(timeout_seconds: float = 120.0):
    """Hold the setup lock; wait up to `timeout_seconds`, then proceed anyway (setup is idempotent, so a lock we cannot take
    must not stop the service from starting)."""
    import time

    fd = None
    if fcntl is not None:
        try:
            _path().parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(_path(), os.O_CREAT | os.O_RDWR)
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        fd = None
                        break
                    time.sleep(0.5)
        except OSError:
            fd = None
    try:
        yield
    finally:
        if fd is not None and fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

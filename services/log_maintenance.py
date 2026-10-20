"""Keep the app's logs/ folder bounded. The one rotation policy for every log file.

Two growth patterns, two rules:
- A file still being appended to (the cron logs such as maps.log and
  cron.log, api.log, the per-script logs) is rotated once it passes
  `max_size_bytes`: copied to <name>.1.gz (older copies shift up to
  <name>.<keep>.gz, the oldest is dropped), then truncated in place.
  Copy-then-truncate keeps any process that has the file open (cron's
  `>>` redirect, a FileHandler) writing to the same file.
- A file whose mtime hasn't moved in `retention_days` (a one-off dated log
  such as forecast_20260118.log, or an old rotated copy) is deleted.

Run daily by core/scheduler.py's purge_old_logs_job and every map cycle by
scripts/maps.sh (`python -m services.log_maintenance`). A lock file keeps
the two from rotating the same file at once.
"""
from __future__ import annotations

import fcntl
import gzip
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from core.config import LOGS_DIR

logger = logging.getLogger(__name__)

RETENTION_DAYS = 30
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
KEEP_ROTATED = 5
LOCK_NAME = ".log_maintenance.lock"


def _rotated_name(path: Path, index: int) -> Path:
    return path.with_name(f"{path.name}.{index}.gz")


def rotate_log(path: Path, keep: int = KEEP_ROTATED) -> None:
    """Gzip a copy of `path` to <name>.1.gz (shifting older copies) and truncate it."""
    _rotated_name(path, keep).unlink(missing_ok=True)
    for index in range(keep - 1, 0, -1):
        older = _rotated_name(path, index)
        if older.exists():
            older.replace(_rotated_name(path, index + 1))
    target = _rotated_name(path, 1)
    tmp = target.with_name(target.name + ".tmp")
    with open(path, "rb") as source, gzip.open(tmp, "wb", compresslevel=6) as dest:
        shutil.copyfileobj(source, dest, 1024 * 1024)
    tmp.replace(target)
    # Lines written between the copy and here are lost; at the sizes involved that is a few lines at most.
    with open(path, "r+b") as live:
        live.truncate(0)


def purge_old_logs(retention_days: int = RETENTION_DAYS, max_size_bytes: int = MAX_SIZE_BYTES,
                   keep: int = KEEP_ROTATED) -> dict:
    """Rotate oversized logs and delete stale ones. Never raises."""
    result = {"removed": [], "rotated": []}
    if not LOGS_DIR.exists():
        return result

    lock_path = LOGS_DIR / LOCK_NAME
    try:
        lock = open(lock_path, "a")
    except OSError as exc:
        logger.warning("Log maintenance skipped, cannot open %s: %s", lock_path, exc)
        return result
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return result  # another copy is already doing this

        cutoff = datetime.now(timezone.utc).timestamp() - retention_days * 86400
        for path in sorted(LOGS_DIR.rglob("*")):
            if not path.is_file() or path.name == LOCK_NAME:
                continue
            try:
                stat = path.stat()
                if stat.st_mtime < cutoff:
                    path.unlink()
                    result["removed"].append(str(path.relative_to(LOGS_DIR)))
                elif stat.st_size > max_size_bytes and path.suffix == ".log":
                    rotate_log(path, keep)
                    result["rotated"].append(str(path.relative_to(LOGS_DIR)))
            except OSError as exc:
                logger.warning("Could not process log file %s: %s", path, exc)
    finally:
        lock.close()
    return result


if __name__ == "__main__":
    outcome = purge_old_logs()
    for name in outcome["rotated"]:
        print(f"Rotated {name}")
    for name in outcome["removed"]:
        print(f"Removed {name} (unchanged for {RETENTION_DAYS} days)")

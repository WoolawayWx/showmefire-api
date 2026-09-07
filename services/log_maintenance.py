"""
Prune the app's logs/ directory, which nothing previously rotated or
deleted from: the production image's cron jobs (see Dockerfile) each
append into a static-named log file (cron.log, maps.log, ...) forever,
and the scripts under scripts/ each write their own one-file-per-day
dated log (forecast_20260118.log, training_20260103.log, ...) that
nothing ever removes. (The handful of loggers already using
logging.handlers.RotatingFileHandler, e.g. maps/*.py and tools/*.py,
already cap themselves and are left alone here - deleting one of their
size-capped backups is harmless, but nothing here needs to touch them.)

Two independent rules, since the two growth patterns are different:
- A file whose mtime hasn't moved in `retention_days` is a stale
  one-off dated log - just delete it.
- A file still being actively appended to (the static cron logs) never
  ages out by mtime, so anything over `max_size_bytes` gets truncated
  in place instead of deleted.
"""
import logging
from datetime import datetime, timezone

from core.config import LOGS_DIR

logger = logging.getLogger(__name__)

RETENTION_DAYS = 30
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def purge_old_logs(retention_days: int = RETENTION_DAYS, max_size_bytes: int = MAX_SIZE_BYTES) -> dict:
    """Delete stale dated logs and truncate oversized ever-growing ones. Never raises."""
    if not LOGS_DIR.exists():
        return {"removed": [], "truncated": []}

    cutoff = datetime.now(timezone.utc).timestamp() - retention_days * 86400
    removed = []
    truncated = []
    for path in LOGS_DIR.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            if stat.st_mtime < cutoff:
                path.unlink()
                removed.append(str(path.relative_to(LOGS_DIR)))
            elif stat.st_size > max_size_bytes:
                with open(path, "w"):
                    pass
                truncated.append(str(path.relative_to(LOGS_DIR)))
        except OSError as exc:
            logger.warning("Could not process log file %s: %s", path, exc)
    return {"removed": removed, "truncated": truncated}

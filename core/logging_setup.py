"""One log format and setup for the API process and the scripts it runs.

Every log line written by this app looks like

    2026-10-20 13:00:01Z INFO core.scheduler: Log maintenance: rotated=1 removed=0

(UTC, with the Z, so the admin Logs page can show it in local time without
guessing the server's timezone). services/log_viewer.py parses this shape;
change both together.

Files are opened with plain append handlers, never RotatingFileHandler:
the API forks a process pool and cron scripts write the same folder, and
in-process rotation across several writers loses lines. Rotation happens
in one place instead, services/log_maintenance.py (copy, gzip, truncate).
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

from core.config import LOGS_DIR

# core.config.LOGS_DIR is relative to the working directory (the API and cron run from api/);
# anchor it here so a script started from somewhere else still writes the same folder.
LOGS_ROOT = LOGS_DIR if LOGS_DIR.is_absolute() else Path(__file__).resolve().parents[1] / LOGS_DIR

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%SZ"

# Libraries that log every request, tile or file at INFO. Their warnings
# still come through.
QUIET_LOGGERS = (
    "apscheduler.executors.default",  # "Running job ..." / "executed successfully" for every run; services/job_runs.py records runs instead
    "apscheduler.scheduler",
    "asyncio",
    "boto3",
    "botocore",
    "s3transfer",
    "urllib3",
    "requests",
    "httpx",
    "httpcore",
    "matplotlib",
    "PIL",
    "fiona",
    "rasterio",
    "pyproj",
    "shapely",
    "numexpr",
    "cfgrib",
    "herbie",
    "multipart",
    "python_multipart",
    "watchfiles",
)


# Secrets that end up in exception messages via request URLs (requests puts the full URL in
# HTTPError), e.g. the FIRMS map key is a path segment: /api/area/csv/<key>/...
_SECRETS = (
    (re.compile(r"(?i)\b((?:api[_-]?)?key|token|access_token|secret|password|signature)=([^&\s'\"]+)"), r"\1=[redacted]"),
    (re.compile(r"(/api/area/csv/)[^/\s]+"), r"\1[redacted]"),
)


def redact(text: str) -> str:
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


class UtcFormatter(logging.Formatter):
    """The shared format, in UTC, with secrets in messages and tracebacks redacted."""

    converter = time.gmtime

    def __init__(self, fmt: str = LOG_FORMAT, datefmt: str = DATE_FORMAT):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def running_tests() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


def file_handler(filename: str, level: int = logging.INFO) -> logging.Handler:
    """An append-only handler for logs/<filename>; a NullHandler if the folder isn't writable."""
    try:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(LOGS_ROOT / filename, mode="a", encoding="utf-8")
    except OSError as exc:
        logging.getLogger(__name__).warning("Log file %s unavailable: %s", LOGS_ROOT / filename, exc)
        return logging.NullHandler()
    handler.setLevel(level)
    handler.setFormatter(UtcFormatter())
    return handler


def configure_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Root logger to stderr (and optionally logs/<log_file>), libraries quieted. Safe to call twice."""
    root = logging.getLogger()
    root.setLevel(level)
    if not getattr(root, "_smf_configured", False):
        # A plain basicConfig() stream handler some imported module set up first would print every line twice.
        for handler in list(root.handlers):
            if type(handler) is logging.StreamHandler:
                root.removeHandler(handler)
        stream = logging.StreamHandler()
        stream.setFormatter(UtcFormatter())
        root.addHandler(stream)
        if log_file and not running_tests():
            root.addHandler(file_handler(log_file, level))
        root._smf_configured = True  # type: ignore[attr-defined]
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def file_logger(name: str, filename: str, level: int = logging.INFO) -> logging.Logger:
    """A named logger that also writes to logs/<filename> (once, however often a module is imported).

    It still propagates, so inside the API process its errors reach the
    root handlers (stderr, api.log, the Discord staff alert) too.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(getattr(h, "_smf_file", None) == filename for h in logger.handlers):
        handler = file_handler(filename, level)
        handler._smf_file = filename  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger

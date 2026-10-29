"""The reviewer sandbox: a separate, synthetic SQLite database that app-store reviewers' staff sessions run against.

`core.database.get_db_path()` returns the override set here while a `sandbox_scope()` is open, and the real path
otherwise, so every existing query function works on the sandbox without being touched. Three rules keep that safe:

  * The scope is entered in exactly one place (the staff API's `staff_context` dependency) and only for sandbox roles,
    and sandbox roles can reach nothing else (core/admin_auth/lockdown.py), so there is no path to real data.
  * Accounts, sessions, throttles and staff push devices stay on the REAL database (`real_db_path()`), because they
    describe real people; the sandbox holds only the demo content.
  * Anything that would reach the outside world (Discord, push, CDN uploads, AI calls) must check `is_sandbox()` in the
    calling thread before it starts: a new thread does not inherit the override, and silently falling back to the real
    database from one would be the worst outcome. If the sandbox file is missing the request fails (503); it never
    falls back to the real database.

This module deliberately imports nothing from core.database (which imports it).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, Optional

_override: ContextVar[Optional[Path]] = ContextVar("sandbox_db_path", default=None)


def active_override() -> Optional[Path]:
    return _override.get()


def is_sandbox() -> bool:
    return _override.get() is not None


@contextmanager
def sandbox_scope(path: Path) -> Iterator[Path]:
    """Run the enclosed code against the sandbox database at `path`."""
    token = _override.set(Path(path))
    try:
        yield Path(path)
    finally:
        _override.reset(token)


def sandbox_root() -> Path:
    """Folder for everything sandbox-only (database, template, photos), beside the real database."""
    from core.database import real_db_path

    return real_db_path().parent / "sandbox"


def sandbox_db_path() -> Path:
    return sandbox_root() / "sandbox.db"


def sandbox_template_path() -> Path:
    return sandbox_root() / "sandbox.template.db"


def sandbox_media_dir() -> Path:
    return sandbox_root() / "media"

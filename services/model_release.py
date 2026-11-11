"""Shared archive verification and installation for model families."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

MANIFEST_NAME = "release_manifest.json"
INSTALLED_NAME = "installed.json"
MAX_BYTES = 500 * 1024 * 1024       # a release is about 1 MB; anything near this is not one


class InstallError(Exception):
    """The release failed a check. Nothing was installed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(name: str) -> bool:
    """A plain relative file path: no drive, no leading slash, no '..', no backslashes."""
    path = PurePosixPath(name)
    return (bool(path.parts) and path.as_posix() == name and not path.is_absolute()
            and all(part != ".." and not part.endswith((".", " ")) for part in path.parts)
            and "\\" not in name and ":" not in name)


def extract_verified(archive: Path, destination: Path, *, schema: str, required: tuple[str, ...]) -> dict:
    """Extract `archive` into the empty folder `destination` and return its manifest, or raise InstallError."""
    try:
        bundle = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise InstallError(f"{archive} is not a readable archive: {exc}") from exc
    with bundle:
        names = [info.filename for info in bundle.infolist() if not info.is_dir()]
        if len(names) != len({name.casefold() for name in names}):
            raise InstallError("duplicate archive entries")
        if sum(info.file_size for info in bundle.infolist()) > MAX_BYTES:
            raise InstallError("archive is far larger than a release")
        if MANIFEST_NAME not in names:
            raise InstallError(f"no {MANIFEST_NAME} in the archive")
        try:
            manifest = json.loads(bundle.read(MANIFEST_NAME))
        except (ValueError, RuntimeError) as exc:
            raise InstallError("invalid release manifest") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise InstallError("invalid release manifest")
        if manifest.get("schema") != schema:
            raise InstallError(f"unknown release schema {manifest.get('schema')!r}")
        listed = manifest.get("files") or {}
        missing = [name for name in required if name not in listed]
        if missing:
            raise InstallError(f"release is missing {missing}")
        unsafe = [name for name in names if not _safe(name)]
        if unsafe:
            raise InstallError(f"unsafe path in archive: {unsafe[0]!r}")
        if {MANIFEST_NAME, INSTALLED_NAME} & listed.keys():
            raise InstallError("manifest lists a reserved file")
        unlisted = sorted(set(names) - set(listed) - {MANIFEST_NAME})
        absent = sorted(set(listed) - set(names))
        if unlisted or absent:
            raise InstallError(f"archive and manifest disagree (unlisted {unlisted}, absent {absent})")
        destination.mkdir(parents=True, exist_ok=False)
        for name in listed:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
            spec = listed[name]
            if not isinstance(spec, dict) or _sha256(target) != spec.get("sha256") or target.stat().st_size != spec.get("bytes"):
                raise InstallError(f"checksum mismatch: {name}")
    return manifest


def install_archive(archive: Path, target: Path, *, schema: str, required: tuple[str, ...],
                    source: str, validate, smoke_test=None, expected_tag: str | None = None) -> dict:
    """Stage a verified candidate and keep the replaced directory for rollback."""
    incoming, previous = target.with_name(target.name + ".incoming"), target.with_name(target.name + ".previous")
    shutil.rmtree(incoming, ignore_errors=True)
    try:
        manifest = extract_verified(Path(archive), incoming, schema=schema, required=required)
        if expected_tag is not None and manifest.get("tag") != expected_tag:
            raise InstallError("archive tag does not match the requested release")
        validate(manifest)
        if smoke_test:
            smoke_test(incoming)
        manifest = {**manifest, "load_tested": smoke_test is not None, "installed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source or str(archive)}
        (incoming / INSTALLED_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    except Exception:
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(previous, ignore_errors=True)
        target.rename(previous)
    try:
        incoming.rename(target)
    except OSError:
        if not target.exists() and previous.exists():
            previous.rename(target)
        raise
    return manifest


def installed(target: Path) -> Optional[dict]:
    """The manifest of the release in place, or None (nothing installed, or a folder copied in by hand)."""
    path = target / INSTALLED_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def rollback(target: Path) -> dict:
    """Swap the installed release with the one it replaced. Running it twice undoes it."""
    previous, aside = target.with_name(target.name + ".previous"), target.with_name(target.name + ".swap")
    if not previous.is_dir():
        raise InstallError("there is no previous release to go back to")
    shutil.rmtree(aside, ignore_errors=True)
    if target.exists():
        target.rename(aside)
    try:
        previous.rename(target)
    except OSError:
        if aside.exists() and not target.exists():
            aside.rename(target)
        raise
    if aside.exists():
        aside.rename(previous)
    return installed(target) or {"tag": "unknown (installed by hand)"}


@contextmanager
def release_lock(target: Path):
    """Serialize CLI and web install/rollback in the shared model directory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise InstallError("another release operation is in progress") from exc
        yield

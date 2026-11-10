"""Install a fire_escape release on this server: fetch, verify, test-load, swap in. Keeps the last one for rollback.

A release is one archive published by model-training (`python -m fire_escape.publish`): the two model bundles,
the HRRR adapter and the fuel grid, with a manifest giving the SHA-256 of every file. It installs into
`fire_escape.bundle_dir()` (DATA_DIR/models/fire_escape_release unless SMF_FIRE_ESCAPE_BUNDLE says otherwise),
which is where the daily job looks, so no path has to be configured and no training data has to exist here.

Nothing is replaced until the new release has passed every check: file names are plain relative paths, every
file matches its checksum, nothing unlisted is in the archive, and the models load and score a test row. The
release it replaces is kept next to it as `<name>.previous`.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

ARCHIVE_NAME = "fire_escape_release.zip"
MANIFEST_NAME = "release_manifest.json"
INSTALLED_NAME = "installed.json"
SCHEMA = "fire-escape-release-1"
TAG_PREFIX = "fire_escape-v"
REQUIRED = ("county/bundle.json", "fire/bundle.json", "hrrr_adapter.json", "fuel_grid.npz")
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
    return bool(name) and "\\" not in name and ":" not in name and not path.is_absolute() and ".." not in path.parts


def extract_verified(archive: Path, destination: Path) -> dict:
    """Extract `archive` into the empty folder `destination` and return its manifest, or raise InstallError."""
    try:
        bundle = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise InstallError(f"{archive} is not a readable archive: {exc}") from exc
    with bundle:
        names = [info.filename for info in bundle.infolist() if not info.is_dir()]
        if MANIFEST_NAME not in names:
            raise InstallError(f"no {MANIFEST_NAME} in the archive")
        manifest = json.loads(bundle.read(MANIFEST_NAME))
        if manifest.get("schema") != SCHEMA:
            raise InstallError(f"unknown release schema {manifest.get('schema')!r}")
        listed = manifest.get("files") or {}
        missing = [name for name in REQUIRED if name not in listed]
        if missing:
            raise InstallError(f"release is missing {missing}")
        unsafe = [name for name in names if not _safe(name)]
        if unsafe:
            raise InstallError(f"unsafe path in archive: {unsafe[0]!r}")
        unlisted = sorted(set(names) - set(listed) - {MANIFEST_NAME})
        absent = sorted(set(listed) - set(names))
        if unlisted or absent:
            raise InstallError(f"archive and manifest disagree (unlisted {unlisted}, absent {absent})")
        if sum(info.file_size for info in bundle.infolist()) > MAX_BYTES:
            raise InstallError("archive is far larger than a release")
        destination.mkdir(parents=True, exist_ok=False)
        for name in listed:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
            if _sha256(target) != listed[name]["sha256"]:
                raise InstallError(f"checksum mismatch: {name}")
    return manifest


def smoke_test(directory: Path) -> None:
    """The release loads and both models give a probability for an ordinary dry spring day."""
    import numpy as np
    import pandas as pd

    from services.fire_escape import job, predict

    release = job.load_release(directory)
    if release is None:
        raise InstallError("release does not load")
    typical = {"tmax_c": 20.0, "rh_min": 25.0, "rh_max": 70.0, "wind_ms": 5.0, "vpd_kpa": 1.2, "srad_wm2": 200.0, "precip_day_mm": 0.0,
               "precip_1d_mm": 0.0, "precip_3d_mm": 0.0, "precip_7d_mm": 2.0, "precip_30d_mm": 40.0, "days_since_rain": 6.0,
               "fm100_prev": 11.0, "fm1000_prev": 15.0, "erc_prev": 40.0, "pfdi_flame_ft": 5.0, "pfdi_fm1": 4.0,
               "stn_wind_peak_ms": 8.0, "stn_gust_peak_ms": 12.0, "stn_rh_min": 25.0, "pfdi_peak_flame_ft": 7.0,
               "cured_fraction": 1.0, "doy_sin": 0.96, "doy_cos": 0.28, "frac_grass": 0.3, "frac_timber": 0.5,
               "frac_nonburnable": 0.1, "canopy_cover_pct": 40.0}
    for name in ("county", "fire"):
        unknown = [feature for feature in release[name]["features"] if feature not in typical]
        if unknown:
            raise InstallError(f"{name} model needs features this server's code does not compute: {unknown}")
        for target, probability in predict.predict(release[name], pd.DataFrame([typical])).items():
            if not (np.isfinite(probability).all() and 0.0 <= float(probability[0]) <= 1.0):
                raise InstallError(f"{name}:{target} did not give a probability")
    missing = [name for name in release["adapter"]["features"] if name not in typical]
    if missing or not {"lat", "lon", "frac_grass", "frac_timber", "frac_nonburnable", "canopy_cover_pct"} <= set(release["fuel_grid"]):
        raise InstallError("adapter or fuel grid is not what this server's code expects")


def install_archive(archive: Path, target: Optional[Path] = None, source: str = "") -> dict:
    """Verify `archive` and make it the installed release. Returns the manifest (with `installed_at`, `source`)."""
    from services import fire_escape

    target = Path(target or fire_escape.bundle_dir())
    incoming, previous = target.with_name(target.name + ".incoming"), target.with_name(target.name + ".previous")
    shutil.rmtree(incoming, ignore_errors=True)
    try:
        manifest = extract_verified(Path(archive), incoming)
        smoke_test(incoming)
        manifest = {**manifest, "installed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source or str(archive)}
        (incoming / INSTALLED_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    except Exception:
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(previous, ignore_errors=True)
        target.rename(previous)
    incoming.rename(target)
    return manifest


def installed(target: Optional[Path] = None) -> Optional[dict]:
    """The manifest of the release in place, or None (nothing installed, or a folder copied in by hand)."""
    from services import fire_escape

    path = Path(target or fire_escape.bundle_dir()) / INSTALLED_NAME
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def rollback(target: Optional[Path] = None) -> dict:
    """Swap the installed release with the one it replaced. Running it twice undoes it."""
    from services import fire_escape

    target = Path(target or fire_escape.bundle_dir())
    previous, aside = target.with_name(target.name + ".previous"), target.with_name(target.name + ".swap")
    if not previous.is_dir():
        raise InstallError("there is no previous release to go back to")
    shutil.rmtree(aside, ignore_errors=True)
    if target.exists():
        target.rename(aside)
    previous.rename(target)
    if aside.exists():
        aside.rename(previous)
    return installed(target) or {"tag": "unknown (installed by hand)"}


def latest_tag(repo: str, timeout: int = 60) -> str:
    """The newest fire_escape release tag on `repo` (gh CLI)."""
    out = subprocess.run(["gh", "release", "list", "--repo", repo, "--limit", "200", "--json", "tagName,createdAt"],
                         check=True, capture_output=True, text=True, timeout=timeout).stdout
    tags = sorted((row["createdAt"], row["tagName"]) for row in json.loads(out) if row["tagName"].startswith(TAG_PREFIX))
    if not tags:
        raise InstallError(f"no {TAG_PREFIX}* release on {repo}")
    return tags[-1][1]


def download(tag: str, repo: str, directory: Path, timeout: int = 300) -> Path:
    """Fetch the release archive for `tag` into `directory` (gh CLI, as pipelines/import_model.py does)."""
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", tag, "--repo", repo, "--pattern", ARCHIVE_NAME, "--dir", str(directory), "--clobber"],
                   check=True, timeout=timeout)
    archive = directory / ARCHIVE_NAME
    if not archive.is_file():
        raise InstallError(f"release {tag} has no {ARCHIVE_NAME}")
    return archive


def install_release(tag: str, repo: str, target: Optional[Path] = None) -> dict:
    """Download `tag` from `repo` and install it; the manifest's own tag must be the one asked for."""
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        archive = download(tag, repo, Path(folder))
        with zipfile.ZipFile(archive) as bundle:
            named = json.loads(bundle.read(MANIFEST_NAME)).get("tag") if MANIFEST_NAME in bundle.namelist() else None
        if named != tag:
            raise InstallError(f"release {tag} carries an archive for {named!r}")
        return install_archive(archive, target, source=f"{repo}@{tag}")

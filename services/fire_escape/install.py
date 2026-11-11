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
import subprocess
from pathlib import Path
from typing import Optional

from services import fire_escape, model_release
from services.model_release import InstallError

ARCHIVE_NAME = "fire_escape_release.zip"
MANIFEST_NAME = "release_manifest.json"
INSTALLED_NAME = "installed.json"
SCHEMA = "fire-escape-release-1"
TAG_PREFIX = "fire_escape-v"
REQUIRED = ("county/bundle.json", "fire/bundle.json", "hrrr_adapter.json", "fuel_grid.npz")


def extract_verified(archive: Path, destination: Path) -> dict:
    return model_release.extract_verified(archive, destination, schema=SCHEMA, required=REQUIRED)


def validate_contract(manifest: dict, *, require_contract: bool = False):
    supplied = manifest.get("feature_hashes")
    if supplied is None and not require_contract:
        return  # Legacy releases predate the feature contract.
    api = Path(__file__).resolve().parents[2]
    pairs = json.loads((api / "core" / "contract_mirrors.json").read_text(encoding="utf-8"))["pairs"]
    actual = {
        pair["name"]: hashlib.sha256((api / pair["api_path"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for pair in pairs if "fire_escape" in pair["invalidates"]
    }
    if supplied != actual:
        raise InstallError("release feature code differs from the API checkout")
    contents = "".join(f"{name}:{spec['sha256']}\n" for name, spec in sorted(manifest["files"].items()))
    contents += json.dumps(actual, sort_keys=True)
    content = hashlib.sha256(contents.encode()).hexdigest()
    version = manifest.get("models", {}).get("county", {}).get("version", "")
    if manifest.get("content_sha256") != content or manifest.get("tag") != (
        f"{TAG_PREFIX}{version.removeprefix('fire-escape-')}-{content[:8]}"
    ):
        raise InstallError("release content does not match its tag")


def install_archive(archive: Path, target: Optional[Path] = None, source: str = "", load_test: bool = True,
                    *, expected_tag: str | None = None, require_contract: bool = False) -> dict:
    target = Path(target or fire_escape.bundle_dir())
    with model_release.release_lock(target):
        return model_release.install_archive(
            archive, target, schema=SCHEMA, required=REQUIRED, source=source,
            validate=lambda manifest: validate_contract(manifest, require_contract=require_contract),
            smoke_test=smoke_test if load_test else None, expected_tag=expected_tag,
        )


def installed(target: Optional[Path] = None) -> Optional[dict]:
    return model_release.installed(Path(target or fire_escape.bundle_dir()))


def rollback(target: Optional[Path] = None) -> dict:
    target = Path(target or fire_escape.bundle_dir())
    with model_release.release_lock(target):
        return model_release.rollback(target)


def smoke_test(directory: Path) -> None:
    """The release loads and both models give a probability for an ordinary dry spring day."""
    try:
        import numpy as np
        import pandas as pd

        from services.fire_escape import job, predict
    except ImportError as exc:
        raise InstallError(
            f"this Python cannot load the models ({exc}). Run the installer where the API runs, e.g. inside the "
            "container with /opt/venv/bin/python, or pass --no-load-test to install on checksums alone") from exc

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


def install_release(tag: str, repo: str, target: Optional[Path] = None, load_test: bool = True,
                    *, require_contract: bool = False) -> dict:
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        archive = download(tag, repo, Path(folder))
        return install_archive(archive, target, source=f"{repo}@{tag}", load_test=load_test,
                               expected_tag=tag, require_contract=require_contract)

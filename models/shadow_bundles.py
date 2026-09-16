"""
Server-side version history + live-active pointer for the guarded shadow
model families that sit outside models/versioning.py's stable/beta/history
registry: v4, v5, risk_fusion_glm, fire_weather_ml, fire_weather_index.
(risk_fusion Phase A has no bundle at all - the rule Monte Carlo needs no
trained artifact - so it is intentionally not one of these families.)

Each family's live bundle directory today is a single fixed path pointed at
by an SMF_<X>_BUNDLE env var, silently (or, for v4/v5, manually-then-)
overwritten in place on every new registration - the server has never
actually retained more than "whatever is currently there." This module is
what gives it real version history going forward, without changing how any
*_shadow.py module resolves its bundle: every one of them already reads its
BUNDLE_ENV fresh via os.getenv() on each call (confirmed for all five), so
set_active() below just mutates the running process's os.environ - no
restart needed for a live switch.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
BUNDLES_ROOT = DATA_DIR / "model-bundles"

SHADOW_FAMILIES = ["v4", "v5", "risk_fusion_glm", "fire_weather_ml", "fire_weather_index"]


def _validators():
    """Deferred import - these services pull in xgboost/torch/etc., no need
    to pay that cost for callers that only list versions or read the active
    pointer."""
    from services import v4_shadow, v5_shadow, risk_fusion_glm_shadow, fire_weather_ml_shadow, fire_weather_index_shadow
    return {
        "v4": (v4_shadow.BUNDLE_ENV, v4_shadow.validate_bundle),
        "v5": (v5_shadow.BUNDLE_ENV, v5_shadow.validate_bundle),
        "risk_fusion_glm": (risk_fusion_glm_shadow.BUNDLE_ENV, risk_fusion_glm_shadow.load_bundle),
        "fire_weather_ml": (fire_weather_ml_shadow.BUNDLE_ENV, fire_weather_ml_shadow.load_bundle),
        "fire_weather_index": (fire_weather_index_shadow.BUNDLE_ENV, fire_weather_index_shadow.load_bundle),
    }


def _require_family(family: str) -> None:
    if family not in SHADOW_FAMILIES:
        raise ValueError(f"unknown shadow family: {family!r} (expected one of {SHADOW_FAMILIES})")


def _family_root(family: str) -> Path:
    _require_family(family)
    return BUNDLES_ROOT / family


def _active_path(family: str) -> Path:
    return _family_root(family) / "active.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(directory: Path) -> str:
    """Order-independent checksum over every file's (name, content) pair -
    stable regardless of filesystem iteration order."""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _safe_extract_target(root: Path, name: str) -> Path:
    """Same path-traversal guard as scripts/install_v5_shadow_bundle.py's
    _safe_target - refuses any archive member that would escape `root`."""
    target = (root / name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"unsafe archive member: {name}")
    return target


def extract_bundle_archive(archive: Path, destination: Path) -> None:
    """Extracts a zip/tar bundle archive into `destination` - same
    conventions as scripts/install_v5_shadow_bundle.py's extract(): rejects
    path-traversal members and symlinks/hardlinks in tar archives."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                _safe_extract_target(destination, member.filename)
            source.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                _safe_extract_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise ValueError("bundle archive links are not allowed")
            source.extractall(destination, filter="data")
    else:
        raise ValueError("bundle must be a zip or tar archive")


def list_versions(family: str) -> List[Dict]:
    """Every installed version for `family`, oldest first, each with its
    install manifest merged in. Empty list if nothing has ever been
    installed through this module yet (e.g. only the original
    SMF_<X>_BUNDLE-pointed directory exists, never uploaded here)."""
    root = _family_root(family)
    if not root.is_dir():
        return []
    versions = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        versions.append({"version": entry.name, "path": str(entry), **manifest})
    versions.sort(key=lambda v: v.get("installed_at") or "")
    return versions


def get_active(family: str) -> Optional[Dict]:
    active_path = _active_path(family)
    if not active_path.exists():
        return None
    try:
        return json.loads(active_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_bundle_root(extracted: Path, validate_fn) -> Path:
    """An uploaded archive may have its asset files at the archive root, or
    one level down inside a wrapper folder (the common case when someone
    just zips the candidate directory itself) - try both, using the
    family's own validator as the judge of "is this actually a bundle
    here," rather than guessing from filenames."""
    candidates = [extracted] + [item for item in extracted.iterdir() if item.is_dir()]
    for candidate in candidates:
        try:
            validate_fn(candidate)
            return candidate
        except Exception:
            continue
    raise ValueError("uploaded archive does not contain a valid bundle at its root or one level down")


def install_bundle(family: str, archive_path: Path, uploaded_by: str, version: Optional[str] = None) -> Dict:
    """Extracts `archive_path` (a zip/tar upload), validates it with the
    family's OWN real shadow-module validator/loader (so a malformed or
    contract-mismatched upload can never be installed, let alone
    activated), then commits it into a new versioned directory under this
    family's root. Never activates - that is a separate, deliberate
    set_active() call, never implicit."""
    import tempfile

    _, validate_fn = _validators()[family]
    with tempfile.TemporaryDirectory(prefix=f".{family}-upload-") as scratch:
        scratch_path = Path(scratch)
        extract_bundle_archive(archive_path, scratch_path)
        bundle_root = _resolve_bundle_root(scratch_path, validate_fn)
        bundle = validate_fn(bundle_root)
        resolved_version = version or (bundle.get("version") if isinstance(bundle, dict) else None) \
            or (bundle.get("registered_version") if isinstance(bundle, dict) else None)
        checksum = _bundle_sha256(bundle_root)
        directory_name = resolved_version or f"unversioned-{checksum[:12]}"

        root = _family_root(family)
        root.mkdir(parents=True, exist_ok=True)
        final = root / directory_name
        if final.exists():
            raise FileExistsError(f"{family} version {directory_name!r} is already installed at {final}")
        shutil.copytree(bundle_root, final)

    manifest = {
        "installed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "uploaded_by": uploaded_by,
        "bundle_sha256": checksum,
        "version": resolved_version,
    }
    (final / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"version": directory_name, "path": str(final), **manifest}


def set_active(family: str, version: str) -> Dict:
    """Flips the persisted active pointer AND this process's own
    os.environ for the family's BUNDLE_ENV - every *_shadow.py module reads
    that env var fresh via os.getenv() on each call, so this takes effect
    immediately, with no restart. The persisted active.json is what lets a
    FUTURE restart (a fresh process with no explicit env var override still
    set) resolve to the same choice - see reapply_active_env_vars(), called
    once at API startup."""
    bundle_env, validate_fn = _validators()[family]
    directory = _family_root(family) / version
    if not directory.is_dir():
        raise FileNotFoundError(f"{family} version {version!r} is not installed")
    validate_fn(directory)  # re-validate before flipping live traffic - never trust a stale install blindly

    active_path = _active_path(family)
    active_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"version": version, "path": str(directory),
              "activated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    temporary = active_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(active_path)

    os.environ[bundle_env] = str(directory)
    return record


def reapply_active_env_vars() -> None:
    """Call once at API startup. For any family whose BUNDLE_ENV is not
    already set explicitly (an operator's own env var always wins - same
    precedence as before this module existed), apply its persisted active
    pointer if one exists. This is what makes a live-switched bundle
    survive a container restart without the operator ever having to
    hand-edit an env var."""
    for family in SHADOW_FAMILIES:
        bundle_env, _ = _validators()[family]
        if os.getenv(bundle_env, "").strip():
            continue
        active = get_active(family)
        if active and Path(active.get("path", "")).is_dir():
            os.environ[bundle_env] = active["path"]

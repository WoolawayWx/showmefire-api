"""
Loads the trained ensemble_fire_danger bundle (calibration knots,
categorical thresholds, member config, evaluation report) - or reports
"uncalibrated" when none exists, which is a normal, supported state: the
product then shows raw (smoothed) member fractions and says so on the
graphic.

Resolution order (first hit wins):
  1. SMF_ENSEMBLE_FD_BUNDLE - a raw bundle directory (operator override,
     same pattern as the other shadow families' *_BUNDLE env vars);
  2. the server registry's `stable` entry for model_type
     "ensemble_fire_danger";
  3. the registry's `beta` entry (this whole product is beta-labelled and
     advisory-only, so a registered beta calibration is still better than
     none - the evidence file records which channel was used).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

MODEL_TYPE = "ensemble_fire_danger"
BUNDLE_ENV = "SMF_ENSEMBLE_FD_BUNDLE"
ASSET_FILENAMES = {
    "member_config": "member_config.json",
    "calibration": "calibration.json",
    "categorical_thresholds": "categorical_thresholds.json",
    "evaluation": "evaluation.json",
}


CORE_MODULE_PATH = Path(__file__).with_name("core.py")


def core_module_sha256() -> str:
    """Checksum of the numeric core a calibration must have been fitted
    against (normalized to LF so a CRLF checkout doesn't change it)."""
    return hashlib.sha256(CORE_MODULE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _checksum(paths) -> str:
    digest = hashlib.sha256()
    for path in sorted(str(p) for p in paths):
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _from_paths(paths: Dict[str, Path], source: str, version: Optional[str]) -> dict:
    bundle = {"source": source, "version": version, "calibrated": False,
              "calibration": None, "categorical_thresholds": None, "member_config": None, "evaluation": None}
    for role in ASSET_FILENAMES:
        if role in paths and Path(paths[role]).exists():
            bundle[role] = _read(paths[role])
    bundle["calibrated"] = bool(bundle["calibration"])
    bundle["checksum"] = _checksum([p for p in paths.values() if Path(p).exists()])
    if version is None and bundle.get("calibration"):
        bundle["version"] = bundle["calibration"].get("version")
    return bundle


def load_bundle() -> dict:
    raw = os.getenv(BUNDLE_ENV, "").strip()
    if raw:
        root = Path(raw)
        return _from_paths({role: root / name for role, name in ASSET_FILENAMES.items()}, f"env:{root}", None)
    try:
        from models.versioning import get_model_entry, load_active_assets
        entry = get_model_entry(MODEL_TYPE)
        for channel in ("stable", "beta"):
            if not entry.get(channel):
                continue
            assets = load_active_assets(MODEL_TYPE, channel=channel)
            return _from_paths({role: asset["path"] for role, asset in assets.items()},
                               f"registry:{channel}", entry[channel].get("version"))
    except Exception:
        pass
    return {"source": "none", "version": None, "calibrated": False, "calibration": None,
            "categorical_thresholds": None, "member_config": None, "evaluation": None, "checksum": None}


def track_calibration(bundle: dict, track: str) -> Optional[dict]:
    calibration = bundle.get("calibration") or {}
    return (calibration.get("tracks") or {}).get(track)


def track_thresholds(bundle: dict, track: str) -> Optional[Dict[int, float]]:
    thresholds = ((bundle.get("categorical_thresholds") or {}).get("tracks") or {}).get(track)
    return {int(k): float(v) for k, v in thresholds.items()} if thresholds else None

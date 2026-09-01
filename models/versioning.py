"""Shared model version registry backed by models/config.json.

Introduces a real semantic-version scheme with separate `stable` and `beta`
channels per model type, replacing the old convention where "version" was
just the static output filename and every retrain silently overwrote
whatever was being served.

    stable  -> what serving code should load today
    beta    -> the latest trained candidate, evaluated in isolation
    history -> capped trail of past stable/beta entries

Nothing lands in `stable` except through promote(); freshly trained or imported
artifacts must enter the beta channel.
"""
import json
import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
API_DIR = MODELS_DIR.parent
CONFIG_PATH = MODELS_DIR / "config.json"
VERSIONS_DIR = MODELS_DIR / "versions"

MAX_HISTORY = 20
REQUIRED_BETA_METADATA = {
    "feature_schema_version", "rule_spec_version", "training_window",
    "data_match_policy", "validation_folds", "class_support", "feature_columns",
}

# fire_risk_fusion is a separate model family from the fuel_moisture lineage
# (V2-V5): different target (independent fire-occurrence labels, not target_fm),
# different unit of analysis (county-day), and a v1 that is advisory-only by
# design - see model-training/risk_fusion/risk_fusion_contract.py. This
# metadata set is checked IN ADDITION TO REQUIRED_BETA_METADATA.
REQUIRED_RISK_FUSION_METADATA = {
    "label_manifest_sha256", "label_min_tier", "label_rows_by_tier",
    "cause_filter", "count_family", "model_family",
    "offset_definition_sha256", "feature_module_sha256",
    "policy_version", "policy_sha256", "guard_active_row_fraction",
    "advisory_only",
}

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-beta\.(\d+))?$")

# Static filenames older/ad-hoc scripts still hardcode. promote() keeps these
# in sync as a compatibility shim so anything not yet wired to the registry
# doesn't silently start serving a stale file.
_LEGACY_STATIC_FILENAMES = {
    "fuel_moisture": "fuel_moisture_model.json",
    "fire_danger": "fire_danger_model.json",
}


def _load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = CONFIG_PATH.with_suffix(CONFIG_PATH.suffix + ".tmp")
    with open(temp, "w") as f:
        json.dump(config, f, indent=2)
    temp.replace(CONFIG_PATH)


def _entry(model_type, config):
    return config.setdefault(model_type, {"stable": None, "beta": None, "history": []})


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_version(version):
    """Split a version string into (major, minor, patch, beta_n | None)."""
    m = _VERSION_RE.match(version)
    if not m:
        raise ValueError(f"Not a recognized semantic version: {version!r}")
    major, minor, patch, beta = m.groups()
    return int(major), int(minor), int(patch), (int(beta) if beta else None)


def next_version(model_type, bump="patch", beta=False):
    """Compute the next semantic version for a model type.

    `bump` advances major/minor/patch off the current stable version (or
    0.0.0 if none exists yet). When `beta` is True the result gets a
    `-beta.N` suffix; if a beta candidate already exists for the same base
    version, its counter is incremented instead of bumping the base again.
    """
    config = _load_config()
    entry = _entry(model_type, config)

    stable = entry.get("stable")
    base_version = stable["version"] if stable else "0.0.0"
    major, minor, patch, _ = parse_version(base_version)

    if bump == "major":
        major, minor, patch = major + 1, 0, 0
    elif bump == "minor":
        minor, patch = minor + 1, 0
    elif bump == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump type: {bump!r}")

    base = f"{major}.{minor}.{patch}"
    if not beta:
        return base

    existing_beta = entry.get("beta")
    if existing_beta and existing_beta["version"].startswith(f"{base}-beta."):
        _, _, _, beta_n = parse_version(existing_beta["version"])
        return f"{base}-beta.{beta_n + 1}"
    return f"{base}-beta.1"


def register_trained_model(model_type, source_path=None, performance=None, bump="patch", channel="beta", assets=None,
                           metadata=None):
    """Register a freshly trained model artifact under the given channel.

    Copies `source_path` into models/versions/ under an immutable, versioned
    filename, updates config.json, and returns the assigned version string.
    Defaults to the `beta` channel so a retrain never silently replaces what
    is currently being served.
    """
    if channel != "beta":
        raise ValueError("Fresh artifacts must enter the beta channel and pass promotion gates")

    version = next_version(model_type, bump=bump, beta=(channel == "beta"))

    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    asset_records = {}
    if assets:
        for role, value in assets.items():
            specification = value if isinstance(value, dict) else {"path": value}
            source = Path(specification["path"]); destination = VERSIONS_DIR / f"{model_type}_{version}_{role}{source.suffix}"
            shutil.copy2(source, destination)
            asset_records[role] = {"file": str(destination.relative_to(API_DIR)), "sha256": _sha256(destination),
                                   **{key: val for key, val in specification.items() if key != "path"}}
        primary = asset_records.get("model") or asset_records.get("checkpoint") or asset_records.get("static_bundle")
        versioned_path = API_DIR / primary["file"] if primary else None
    else:
        source_path = Path(source_path); versioned_path = VERSIONS_DIR / f"{model_type}_{version}{source_path.suffix}"; shutil.copy2(source_path, versioned_path)

    config = _load_config()
    entry = _entry(model_type, config)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "version": version,
        "file": str(versioned_path.relative_to(API_DIR)) if versioned_path else None,
        "performance": performance or {},
        ("trained_at" if channel == "beta" else "promoted_at"): now,
        "metadata": metadata or {},
    }
    if versioned_path:
        record["sha256"] = _sha256(versioned_path)
    if asset_records:
        record["assets"] = asset_records

    entry[channel] = record
    entry.setdefault("history", []).append({**record, "channel": channel, "recorded_at": now})
    entry["history"] = entry["history"][-MAX_HISTORY:]

    _save_config(config)
    return version


def validate_promotion_candidate(model_type, candidate):
    """Return promotion blockers without modifying registry state."""
    blockers = []
    metadata = candidate.get("metadata") or {}
    if model_type == "fire_risk_fusion":
        from core.fire_danger import RULE_SPEC_SHA256, RULE_SPEC_VERSION
    if model_type in {"fuel_moisture", "fire_danger"}:
        missing = sorted(REQUIRED_BETA_METADATA.difference(metadata))
        if missing:
            blockers.append(f"missing metadata: {', '.join(missing)}")
    if model_type == "fire_behavior_static":
        if not (candidate.get("assets") or {}).get("static_bundle"):
            blockers.append("static_bundle asset is required")
        manifest_asset = (candidate.get("assets") or {}).get("static_manifest")
        if not manifest_asset:
            blockers.append("static_manifest asset is required")
    if model_type == "fire_risk_fusion":
        missing = sorted(REQUIRED_BETA_METADATA.union(REQUIRED_RISK_FUSION_METADATA).difference(metadata))
        if missing:
            blockers.append(f"missing metadata: {', '.join(missing)}")
        # v1 authorizes advisory publication only (see risk-fusion-promotion-policy-v1).
        # A candidate must be structurally unpromotable to a serving path -
        # this is not a gate that can be satisfied later, it is a hard v1 boundary.
        if metadata.get("advisory_only") is not True:
            blockers.append("fire_risk_fusion candidates must have advisory_only=True in v1")
        if metadata.get("rule_spec_version") != RULE_SPEC_VERSION:
            blockers.append("rule spec version mismatch")
        if metadata.get("rule_spec_sha256") != RULE_SPEC_SHA256:
            blockers.append("rule spec checksum mismatch")
        weight = metadata.get("guard_active_row_fraction")
        if metadata.get("model_family") != "glm" and (weight is None or float(weight) < 0.10):
            blockers.append("guard_active_row_fraction must be >= 0.10 unless model_family == 'glm'")
    precipitation_features = [name for name in metadata.get("feature_columns", [])
                              if name.startswith("precip_") or name == "hours_since_rain"]
    if model_type == "fuel_moisture" and precipitation_features:
        from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
        if metadata.get("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION:
            blockers.append("precipitation contract version mismatch")
        if metadata.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256:
            blockers.append("precipitation contract checksum mismatch")
    artifact = API_DIR / candidate.get("file", "")
    if not artifact.is_file():
        blockers.append(f"artifact is missing: {artifact}")
    elif candidate.get("sha256") and _sha256(artifact) != candidate["sha256"]:
        blockers.append("artifact checksum mismatch")
    gates = metadata.get("promotion_gates") or {}
    failed = sorted(name for name, value in gates.items() if value is False)
    if failed:
        blockers.append(f"failed promotion gates: {', '.join(failed)}")
    if model_type in {"fuel_moisture", "fire_danger"} and metadata.get("shadow_required", True):
        shadow = metadata.get("shadow") or {}
        if not shadow.get("passed"):
            blockers.append("shadow validation has not passed")
    if model_type == "fuel_moisture" and artifact.is_file() and metadata.get("feature_columns"):
        try:
            import pandas as pd
            import xgboost as xgb
            ranges = metadata.get("feature_ranges") or {}
            row = {name: (float(ranges[name]["min"]) + float(ranges[name]["max"])) / 2
                   if name in ranges else 0.0 for name in metadata["feature_columns"]}
            booster = xgb.Booster(); booster.load_model(str(artifact))
            if precipitation_features:
                if booster.attr("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION:
                    blockers.append("artifact precipitation contract mismatch")
            prediction = booster.predict(xgb.DMatrix(pd.DataFrame([row]), feature_names=metadata["feature_columns"]))
            if len(prediction) != 1 or not float(prediction[0]) == float(prediction[0]):
                blockers.append("candidate smoke inference returned an invalid prediction")
        except Exception as exc:
            blockers.append(f"candidate smoke inference failed: {exc}")
    return blockers


def promote(model_type, version=None):
    """Promote the beta candidate (or a specific matching version) to stable.

    The previous stable entry is archived into history. Raises if there is
    no beta candidate, or if `version` doesn't match the current beta.
    """
    config = _load_config()
    entry = _entry(model_type, config)

    beta = entry.get("beta")
    if not beta:
        raise ValueError(f"No beta candidate registered for {model_type!r}")
    if version and beta["version"] != version:
        raise ValueError(
            f"Requested version {version!r} is not the current beta "
            f"({beta['version']!r}) for {model_type!r}"
        )
    blockers = validate_promotion_candidate(model_type, beta)
    if blockers:
        raise ValueError("Candidate is not promotable: " + "; ".join(blockers))

    previous_stable = entry.get("stable")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if previous_stable:
        entry.setdefault("history", []).append({**previous_stable, "channel": "stable", "recorded_at": now})
        entry["history"] = entry["history"][-MAX_HISTORY:]

    # A promoted candidate becomes a clean release - drop the -beta.N suffix
    # from both the version string and the on-disk filename.
    major, minor, patch, _ = parse_version(beta["version"])
    release_version = f"{major}.{minor}.{patch}"

    old_path = API_DIR / beta["file"]
    new_path = old_path
    if not beta.get("assets"):
        new_path = old_path.with_name(f"{model_type}_{release_version}{old_path.suffix}")
        if old_path != new_path: old_path.rename(new_path)

    promoted = {k: v for k, v in beta.items() if k != "trained_at"}
    promoted["version"] = release_version
    promoted["file"] = str(new_path.relative_to(API_DIR))
    promoted["promoted_at"] = now

    entry["stable"] = promoted
    entry["beta"] = None

    _save_config(config)

    legacy_filename = _LEGACY_STATIC_FILENAMES.get(model_type)
    if legacy_filename:
        shutil.copy2(API_DIR / promoted["file"], MODELS_DIR / legacy_filename)

    return promoted["version"]


def rollback(model_type, version=None):
    """Reactivate a prior stable artifact and synchronize legacy consumers."""
    config = _load_config()
    entry = _entry(model_type, config)
    current = entry.get("stable")
    candidates = [record for record in reversed(entry.get("history", []))
                  if record.get("channel") == "stable" and record.get("file")]
    if version:
        candidates = [record for record in candidates if record.get("version") == version]
    elif current:
        candidates = [record for record in candidates if record.get("version") != current.get("version")]
    if not candidates:
        raise ValueError(f"No rollback target found for {model_type!r}")
    target = {key: value for key, value in candidates[0].items()
              if key not in {"channel", "recorded_at"}}
    path = API_DIR / target["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Rollback artifact missing: {path}")
    if target.get("sha256") and _sha256(path) != target["sha256"]:
        raise ValueError("Rollback artifact checksum mismatch")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if current:
        entry.setdefault("history", []).append({**current, "channel": "stable", "recorded_at": now})
    target["promoted_at"] = now
    target["rollback_from"] = current.get("version") if current else None
    entry["stable"] = target
    entry["history"] = entry.get("history", [])[-MAX_HISTORY:]
    _save_config(config)
    legacy = _LEGACY_STATIC_FILENAMES.get(model_type)
    if legacy:
        shutil.copy2(path, MODELS_DIR / legacy)
    return target["version"]


def get_model_entry(model_type):
    """Return the full registry entry (stable/beta/history) for a model type."""
    return _load_config().get(model_type) or {}


def update_beta_metadata(model_type, updates):
    """Merge validation/shadow evidence into the current beta candidate."""
    config = _load_config()
    entry = _entry(model_type, config)
    beta = entry.get("beta")
    if not beta:
        raise ValueError(f"No beta candidate registered for {model_type!r}")
    metadata = beta.setdefault("metadata", {})
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(metadata.get(key), dict):
            metadata[key].update(value)
        else:
            metadata[key] = value
    _save_config(config)
    return beta["version"]


def load_active_model_path(model_type, channel="stable", auto_rollback=False):
    """Resolve the filesystem path serving code should load for `model_type`."""
    config = _load_config()
    entry = config.get(model_type) or {}
    active = entry.get(channel)
    if not active:
        raise FileNotFoundError(f"No {channel!r} model registered for {model_type!r}")

    path = API_DIR / active["file"]
    invalid = not path.exists()
    if not invalid and active.get("sha256"):
        invalid = _sha256(path) != active["sha256"]
    if invalid and auto_rollback and channel == "stable":
        rollback(model_type)
        return load_active_model_path(model_type, channel, auto_rollback=False)
    if invalid:
        raise FileNotFoundError(f"Registered {channel} model file missing or invalid: {path}")
    return path


def load_active_assets(model_type, channel="stable"):
    entry = (_load_config().get(model_type) or {}).get(channel)
    if not entry or not entry.get("assets"):
        raise FileNotFoundError(f"No asset contract for {model_type!r} channel {channel!r}")
    resolved = {}
    for role, asset in entry["assets"].items():
        path = API_DIR / asset["file"]
        if not path.exists() or _sha256(path) != asset["sha256"]: raise FileNotFoundError(f"Missing or invalid {role} asset: {path}")
        resolved[role] = {**asset, "path": path}
    return resolved

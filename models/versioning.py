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

# fire_weather_ml is a third, independent model family (alongside
# fuel_moisture and fire_risk_fusion): trained against the same Rothermel
# calculation services/spread_rate.py runs live, not against fire-
# occurrence reports - see model-training/docs/fire_weather_ml_plan.md.
# Like fire_risk_fusion, v1 is advisory-only by design and is not currently
# promoted through this registry at all - it's scored in shadow only via a
# raw bundle directory (see services/fire_weather_ml_shadow.py). This
# metadata set mirrors model-training/fire_weather_ml/register_beta.py's
# own REQUIRED_METADATA_FIELDS exactly, so a future real promotion pipeline
# for this model family has a matching gate ready rather than needing one
# invented from scratch once it exists.
REQUIRED_FIRE_WEATHER_ML_METADATA = {
    "feature_module_sha256", "label_module_sha256", "label_column",
    "model_family", "training_row_count", "split_manifest_sha256", "advisory_only",
}

# fire_weather_index is a fourth, independent model family: a continuous,
# weighted fire-WEATHER-conditions score (not fit to any label - see
# model-training/fire_weather_index/__init__.py), scored in shadow only via
# a raw bundle directory (services/fire_weather_index_shadow.py), not
# currently promoted through this registry at all - same v1 boundary as
# fire_weather_ml, ready for a future real promotion pipeline.
REQUIRED_FIRE_WEATHER_INDEX_METADATA = {
    "model_family", "advisory_only",
}

# risk_fusion_glm is the first of the formerly shadow_bundles.py-only
# families to be migrated onto this registry (see
# services/risk_fusion_glm_shadow.py's own load_bundle(), whose contract
# checks this mirrors: advisory_only, model_family=="glm", and the
# feature-module checksum against core/risk_fusion_features.py). Chosen as
# the pilot because it's advisory-only with no live-serving fallback risk.
REQUIRED_RISK_FUSION_GLM_METADATA = {
    "model_family", "advisory_only", "feature_module_sha256",
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
    try:
        temp.replace(CONFIG_PATH)
    except PermissionError:
        # Some Windows/network volumes permit writes but deny replace while a
        # reader has the registry open. Keep the already-complete temp file as
        # the source and flush the destination before removing it.
        with open(temp, "r", encoding="utf-8") as source, open(CONFIG_PATH, "w", encoding="utf-8") as destination:
            destination.write(source.read())
            destination.flush()
        temp.unlink()


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
                           metadata=None, live_pointer_env=None):
    """Register a freshly trained model artifact under the given channel.

    Copies `source_path` into models/versions/ under an immutable, versioned
    filename, updates config.json, and returns the assigned version string.
    Defaults to the `beta` channel so a retrain never silently replaces what
    is currently being served.

    `live_pointer_env`: for model types being migrated off the legacy
    shadow_bundles.py fixed-directory/env-var mechanism (see
    shadow_bundles.SHADOW_FAMILIES) - the env var name (e.g.
    "SMF_V4_SHADOW_BUNDLE") that family's *_shadow.py service still reads
    directly. Recorded on the entry so promote()/rollback() can keep that
    env var pointed at the active bundle during the migration, so the
    service's existing read path needs no code change until it's fully
    cut over to load_active_assets(). Not yet acted on by promote()/
    rollback() - that lands alongside each family's actual migration.
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
    if live_pointer_env:
        record["live_pointer_env"] = live_pointer_env
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
    if model_type == "fire_weather_ml":
        missing = sorted(REQUIRED_FIRE_WEATHER_ML_METADATA.difference(metadata))
        if missing:
            blockers.append(f"missing metadata: {', '.join(missing)}")
        # Same hard v1 boundary as fire_risk_fusion - not a gate that can be
        # satisfied later, a structural refusal to promote this model
        # family to anything serving-facing in v1.
        if metadata.get("advisory_only") is not True:
            blockers.append("fire_weather_ml candidates must have advisory_only=True in v1")
    if model_type == "fire_weather_index":
        missing = sorted(REQUIRED_FIRE_WEATHER_INDEX_METADATA.difference(metadata))
        if missing:
            blockers.append(f"missing metadata: {', '.join(missing)}")
        # Same hard v1 boundary as fire_risk_fusion/fire_weather_ml.
        if metadata.get("advisory_only") is not True:
            blockers.append("fire_weather_index candidates must have advisory_only=True in v1")
    if model_type == "risk_fusion_glm":
        missing = sorted(REQUIRED_RISK_FUSION_GLM_METADATA.difference(metadata))
        if missing:
            blockers.append(f"missing metadata: {', '.join(missing)}")
        # Mirrors services/risk_fusion_glm_shadow.py::load_bundle()'s own
        # checks exactly - "stable" here means "the version currently
        # scored in shadow," never a public-facing path, so this isn't the
        # same kind of permanent structural boundary as fire_risk_fusion's
        # (nothing downstream of this family's output reaches the public
        # forecast), but the bundle must still be internally consistent.
        if metadata.get("advisory_only") is not True:
            blockers.append("risk_fusion_glm candidates must have advisory_only=True")
        if metadata.get("model_family") != "glm":
            blockers.append("risk_fusion_glm shadow only knows how to score model_family='glm'")
        from services.risk_fusion_glm_shadow import FEATURES_MODULE_PATH, _sha256_file
        if metadata.get("feature_module_sha256") != _sha256_file(FEATURES_MODULE_PATH):
            blockers.append("feature_module_sha256 does not match core/risk_fusion_features.py - retrain or re-mirror")
    if model_type in ("v4", "v5"):
        # Mirrors services/v4_shadow.py / v5_shadow.py::validate_bundle()'s
        # own checks - "stable" here means "the version currently scored in
        # shadow," never a public-facing path (same reasoning as
        # risk_fusion_glm above).
        from core.fire_danger import RULE_SPEC_SHA256 as LIVE_RULE_SPEC_SHA256
        from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
        if metadata.get("advisory_only") is not True:
            blockers.append(f"{model_type} candidates must have advisory_only=True")
        if metadata.get("rule_spec_sha256") != LIVE_RULE_SPEC_SHA256:
            blockers.append("rule spec checksum mismatch")
        if metadata.get("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION:
            blockers.append("precipitation contract version mismatch")
        if metadata.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256:
            blockers.append("precipitation contract checksum mismatch")
    precipitation_features = [name for name in metadata.get("feature_columns", [])
                              if name.startswith("precip_") or name == "hours_since_rain"]
    if model_type == "fuel_moisture" and precipitation_features:
        from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
        if metadata.get("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION:
            blockers.append("precipitation contract version mismatch")
        if metadata.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256:
            blockers.append("precipitation contract checksum mismatch")
    # candidate["file"] is only set for single-file registrations, or
    # multi-asset ones whose roles happen to include model/checkpoint/
    # static_bundle (see register_trained_model) - it's None for bundles
    # like fire_risk_fusion/fire_weather_index/fire_weather_ml whose roles
    # don't match any of those. Each individual asset is still checksum-
    # verified via its own `assets` entry, so there's nothing extra to
    # check here for those candidates.
    artifact = None
    if candidate.get("file"):
        artifact = API_DIR / candidate["file"]
        if not artifact.is_file():
            blockers.append(f"artifact is missing: {artifact}")
        elif candidate.get("sha256") and _sha256(artifact) != candidate["sha256"]:
            blockers.append("artifact checksum mismatch")
    elif not candidate.get("assets"):
        blockers.append("candidate has neither a file nor assets to promote")
    gates = metadata.get("promotion_gates") or {}
    failed = sorted(name for name, value in gates.items() if value is False)
    if failed:
        blockers.append(f"failed promotion gates: {', '.join(failed)}")
    if model_type in {"fuel_moisture", "fire_danger"} and metadata.get("shadow_required", True):
        shadow = metadata.get("shadow") or {}
        if not shadow.get("passed"):
            blockers.append("shadow validation has not passed")
        # Operational-stability shadow evidence (above) only checks that beta
        # didn't crash and stayed roughly close to stable - it never checks
        # which one was actually closer to reality. This second, independent
        # check requires real observed-outcome evidence (see
        # services/shadow_ground_truth.py) before a fuel_moisture candidate can
        # be promoted. Not extended to fire_danger: no live forecast generator
        # loads a registry fire_danger model today, so there's nothing to shadow.
        if model_type == "fuel_moisture" and metadata.get("ground_truth_shadow_required", True):
            ground_truth = shadow.get("ground_truth") or {}
            if not ground_truth.get("passed"):
                blockers.append("ground-truth shadow accuracy has not passed")
    if model_type == "fuel_moisture" and artifact is not None and artifact.is_file() and metadata.get("feature_columns"):
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
    # from the version string only. The on-disk filename stays as-is: it's
    # already immutable/content-addressed under models/versions/, and trying
    # to rename it here used to crash for any multi-asset bundle whose roles
    # don't include model/checkpoint/static_bundle (beta["file"] is None for
    # those - see register_trained_model) - fire_risk_fusion, fire_weather_ml,
    # and fire_weather_index all hit this. Single-file models previously got
    # renamed to drop the suffix too; that rename bought nothing real (the
    # registry, not the filename, is what serving code and operators consult)
    # and is dropped here rather than kept as a special case just for them.
    major, minor, patch, _ = parse_version(beta["version"])
    release_version = f"{major}.{minor}.{patch}"

    promoted = {k: v for k, v in beta.items() if k != "trained_at"}
    promoted["version"] = release_version
    promoted["promoted_at"] = now

    entry["stable"] = promoted
    entry["beta"] = None

    _save_config(config)

    legacy_filename = _LEGACY_STATIC_FILENAMES.get(model_type)
    if legacy_filename:
        shutil.copy2(API_DIR / promoted["file"], MODELS_DIR / legacy_filename)

    return promoted["version"]


def _verify_record_artifacts(record) -> None:
    """Raise if `record`'s artifact(s) are missing or checksum-mismatched.
    Handles both single-file records (`file`) and multi-asset bundles
    (`assets`) - a record can have `file=None` and still be entirely valid
    if it's a multi-asset bundle whose roles don't include model/checkpoint/
    static_bundle (see register_trained_model); checking only `file` here
    used to silently exclude those bundles from rollback candidacy
    entirely, and would have crashed on `API_DIR / None` if it hadn't."""
    if record.get("assets"):
        for role, asset in record["assets"].items():
            path = API_DIR / asset["file"]
            if not path.is_file():
                raise FileNotFoundError(f"Rollback asset missing: {path}")
            if asset.get("sha256") and _sha256(path) != asset["sha256"]:
                raise ValueError(f"Rollback asset checksum mismatch: {role}")
    elif record.get("file"):
        path = API_DIR / record["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Rollback artifact missing: {path}")
        if record.get("sha256") and _sha256(path) != record["sha256"]:
            raise ValueError("Rollback artifact checksum mismatch")
    else:
        raise ValueError("rollback candidate has neither a file nor assets to restore")


def rollback(model_type, version=None):
    """Reactivate a prior stable artifact and synchronize legacy consumers."""
    config = _load_config()
    entry = _entry(model_type, config)
    current = entry.get("stable")
    candidates = [record for record in reversed(entry.get("history", []))
                  if record.get("channel") == "stable" and (record.get("file") or record.get("assets"))]
    if version:
        candidates = [record for record in candidates if record.get("version") == version]
    elif current:
        candidates = [record for record in candidates if record.get("version") != current.get("version")]
    if not candidates:
        raise ValueError(f"No rollback target found for {model_type!r}")
    target = {key: value for key, value in candidates[0].items()
              if key not in {"channel", "recorded_at"}}
    _verify_record_artifacts(target)
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
        shutil.copy2(API_DIR / target["file"], MODELS_DIR / legacy)
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

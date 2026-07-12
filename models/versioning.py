"""Shared model version registry backed by models/config.json.

Introduces a real semantic-version scheme with separate `stable` and `beta`
channels per model type, replacing the old convention where "version" was
just the static output filename and every retrain silently overwrote
whatever was being served.

    stable  -> what serving code should load today
    beta    -> the latest trained candidate, evaluated in isolation
    history -> capped trail of past stable/beta entries

Nothing lands in `stable` except through promote() - register_trained_model()
always writes to `beta` unless a caller explicitly asks otherwise.
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


def register_trained_model(model_type, source_path=None, performance=None, bump="patch", channel="beta", assets=None):
    """Register a freshly trained model artifact under the given channel.

    Copies `source_path` into models/versions/ under an immutable, versioned
    filename, updates config.json, and returns the assigned version string.
    Defaults to the `beta` channel so a retrain never silently replaces what
    is currently being served.
    """
    if channel not in ("beta", "stable"):
        raise ValueError(f"Unknown channel: {channel!r}")

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
        primary = asset_records.get("model") or asset_records.get("checkpoint"); versioned_path = API_DIR / primary["file"] if primary else None
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
    }
    if asset_records:
        record["assets"] = asset_records

    entry[channel] = record
    entry.setdefault("history", []).append({**record, "channel": channel, "recorded_at": now})
    entry["history"] = entry["history"][-MAX_HISTORY:]

    _save_config(config)
    return version


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


def get_model_entry(model_type):
    """Return the full registry entry (stable/beta/history) for a model type."""
    return _load_config().get(model_type) or {}


def load_active_model_path(model_type, channel="stable"):
    """Resolve the filesystem path serving code should load for `model_type`."""
    config = _load_config()
    entry = config.get(model_type) or {}
    active = entry.get(channel)
    if not active:
        raise FileNotFoundError(f"No {channel!r} model registered for {model_type!r}")

    path = API_DIR / active["file"]
    if not path.exists():
        raise FileNotFoundError(f"Registered {channel} model file missing: {path}")
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

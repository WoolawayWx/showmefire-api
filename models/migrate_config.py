"""One-time migration of models/config.json to the stable/beta/history schema.

The old schema tracked a single `active_version` (just a filename) per model
type with no real version number. This baselines whatever is currently being
served as version 1.0.0 under the `stable` channel so versioning.py has a
real starting point. Safe to re-run - already-migrated model types are
left untouched.
"""
import json
import shutil
from datetime import datetime

from versioning import MODELS_DIR, API_DIR, CONFIG_PATH, VERSIONS_DIR

BASELINE_VERSION = "1.0.0"


def migrate():
    if not CONFIG_PATH.exists():
        print(f"[migrate] no config at {CONFIG_PATH}, nothing to do")
        return

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    changed = False

    for model_type, entry in list(config.items()):
        if not isinstance(entry, dict):
            continue
        if "stable" in entry or "beta" in entry:
            print(f"[migrate] {model_type}: already migrated, skipping")
            continue

        old_active_filename = entry.get("active_version")
        source_path = None
        if old_active_filename:
            candidate = MODELS_DIR / old_active_filename
            if candidate.exists():
                source_path = candidate
        if source_path is None:
            candidate = MODELS_DIR / f"{model_type}_model.json"
            if candidate.exists():
                source_path = candidate

        new_entry = {"stable": None, "beta": None, "history": []}

        # Preserve business-logic fields that aren't versioning state.
        if "threshold" in entry:
            new_entry["threshold"] = entry["threshold"]

        if source_path is not None:
            VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
            versioned_path = VERSIONS_DIR / f"{model_type}_{BASELINE_VERSION}{source_path.suffix}"
            if not versioned_path.exists():
                shutil.copy2(source_path, versioned_path)

            new_entry["stable"] = {
                "version": BASELINE_VERSION,
                "file": str(versioned_path.relative_to(API_DIR)),
                "performance": entry.get("performance", {}),
                "promoted_at": entry.get("last_updated") or datetime.now().strftime("%Y-%m-%d"),
            }
            print(f"[migrate] {model_type}: baselined {source_path.name} -> {BASELINE_VERSION}")
        else:
            print(f"[migrate] {model_type}: no existing model file found "
                  f"(active_version={old_active_filename!r}); leaving stable empty")

        for old_hist in entry.get("history", []):
            new_entry["history"].append({
                "version": f"legacy:{old_hist.get('version')}",
                "channel": "stable",
                "file": old_hist.get("archive_path"),
                "performance": old_hist.get("performance", {}),
                "recorded_at": old_hist.get("archived_at") or old_hist.get("last_updated"),
            })

        config[model_type] = new_entry
        changed = True

    if changed:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[migrate] wrote {CONFIG_PATH}")
    else:
        print("[migrate] nothing to do, config already in the new schema")


if __name__ == "__main__":
    migrate()

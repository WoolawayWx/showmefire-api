"""Register a fire-behavior static bundle as a beta asset-only registry entry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.versioning import register_trained_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch")
    args = parser.parse_args()
    manifest = args.manifest or args.bundle.with_suffix(".json")
    metadata = json.loads(manifest.read_text())
    version = register_trained_model(
        model_type="fire_behavior_static",
        performance={"bundle_version": metadata.get("bundle_version")},
        bump=args.bump,
        channel="beta",
        assets={"static_bundle": args.bundle, "static_manifest": manifest},
        metadata={
            "bundle_schema_version": metadata.get("schema_version"),
            "grid_fingerprint": metadata.get("grid_fingerprint"),
            "promotion_gates": {"static_bundle_validated": True},
            "shadow_required": False,
        },
    )
    print(f"Registered fire_behavior_static beta {version}")


if __name__ == "__main__":
    main()

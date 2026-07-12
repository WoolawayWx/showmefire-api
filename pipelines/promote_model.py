"""Promote a beta model candidate to stable (the version serving code loads).

Usage:
    python pipelines/promote_model.py --model fuel_moisture
    python pipelines/promote_model.py --model fuel_moisture --version 1.5.0-beta.1
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.versioning import promote, get_model_entry


def main():
    parser = argparse.ArgumentParser(description="Promote a beta model to stable")
    parser.add_argument("--model", required=True, choices=["fuel_moisture", "fire_danger", "fuel_moisture_spatial"],
                         help="Which model type to promote")
    parser.add_argument("--version", default=None,
                         help="Beta version to promote (defaults to whatever is currently in beta)")
    args = parser.parse_args()

    entry = get_model_entry(args.model)
    beta = entry.get("beta")
    stable = entry.get("stable")

    if not beta:
        print(f"No beta candidate registered for {args.model!r}. Nothing to promote.")
        sys.exit(1)

    print(f"Current stable: {stable['version'] if stable else '(none)'} "
          f"{stable.get('performance') if stable else ''}")
    print(f"Beta candidate: {beta['version']} {beta.get('performance')}")

    version = promote(args.model, version=args.version)
    print(f"Promoted {args.model} {version} to stable.")


if __name__ == "__main__":
    main()

"""Explicitly reactivate a previous stable registry entry."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.versioning import rollback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_type"); parser.add_argument("--version")
    args = parser.parse_args()
    try: version = rollback(args.model_type, args.version)
    except Exception as error: raise SystemExit(str(error)) from error
    print(json.dumps({"model_type": args.model_type, "active_version": version, "rolled_back": True}, indent=2))


if __name__ == "__main__": main()

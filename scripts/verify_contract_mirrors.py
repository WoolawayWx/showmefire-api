"""
Verify that every artifact pair listed in core/contract_mirrors.json is
byte-identical between the api and model-training repos, and matches its
recorded sha256.

This is a WORKSPACE tool, not a production preflight check: it requires
both repos checked out side by side (api/ and ../model-training/), which
is true for local development but never true inside the deployed API
container (api/Dockerfile only COPYs the api/ build context). Run it
manually before cutting a training-side release, or from CI against a
full workspace checkout - do not add it to scripts/predeploy_check.py.

A sha256 mismatch means one side changed without the other. Read the
`invalidates` list before touching either file - it names every model
bundle whose contract now needs re-validation.

Usage:
    python scripts/verify_contract_mirrors.py
    python scripts/verify_contract_mirrors.py --training-root ../model-training
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAINING_ROOT = API_ROOT.parent / "model-training"
MIRRORS_PATH = API_ROOT / "core" / "contract_mirrors.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(training_root: Path) -> list:
    """Returns a list of failure dicts; empty means every pair is in sync."""
    mirrors = json.loads(MIRRORS_PATH.read_text(encoding="utf-8"))
    failures = []

    for pair in mirrors["pairs"]:
        api_path = API_ROOT / pair["api_path"]
        training_path = training_root / pair["training_path"]

        if not api_path.exists():
            failures.append({"name": pair["name"], "reason": f"missing api file: {api_path}"})
            continue
        if not training_path.exists():
            failures.append({"name": pair["name"], "reason": f"missing training file: {training_path}"})
            continue

        api_sha = _sha256_file(api_path)
        training_sha = _sha256_file(training_path)

        if api_sha != training_sha:
            failures.append({
                "name": pair["name"],
                "reason": "api and training copies differ",
                "api_sha256": api_sha,
                "training_sha256": training_sha,
                "invalidates": pair.get("invalidates", []),
            })
        elif api_sha != pair["sha256"]:
            failures.append({
                "name": pair["name"],
                "reason": "both copies match each other but not the recorded sha256 - "
                          "update contract_mirrors.json intentionally, don't silently accept drift",
                "recorded_sha256": pair["sha256"],
                "actual_sha256": api_sha,
                "invalidates": pair.get("invalidates", []),
            })

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT,
                         help="Path to the model-training repo checkout.")
    args = parser.parse_args()

    failures = verify(args.training_root)
    if failures:
        print("CONTRACT MIRROR DRIFT DETECTED:")
        for failure in failures:
            print(json.dumps(failure, indent=2))
        return 1

    print("All contract mirrors are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

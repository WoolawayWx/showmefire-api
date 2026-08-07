"""Grant V5 production eligibility after prospective evidence and three canaries."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.v5_shadow import EVIDENCE_ROOT, validate_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("prospective_report", type=Path)
    parser.add_argument("--bundle", type=Path, required=True); parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    args = parser.parse_args(); report = json.loads(args.prospective_report.read_text()); contract = validate_bundle(args.bundle)
    if report.get("status") != "prospective_checkpoint" or not report.get("authorization_allowed"):
        raise SystemExit("V5 authorization refused: passing predeclared prospective checkpoint required")
    if report.get("bundle_manifest_sha256") != contract.get("bundle_sha256"):
        raise SystemExit("V5 authorization refused: evidence bundle checksum mismatch")
    canaries = [json.loads(path.read_text()) for path in sorted((args.evidence_root / "canaries").glob("*.canary.json"))]
    matching = [item for item in canaries if item.get("bundle_manifest_sha256") == contract.get("bundle_sha256")]
    if len({item["run_id"] for item in matching if item.get("pass")}) < 3:
        raise SystemExit("V5 authorization refused: three passing, distinct canary forecasts required")
    target = args.evidence_root / "promotion-authorization.json"
    if target.exists(): raise SystemExit("Refusing to overwrite V5 promotion authorization")
    authorization = {"production_eligible": True, "serving_changed": False,
                     "explicit_activation_required": True, "prospective_report": str(args.prospective_report),
                     "bundle_manifest_sha256": contract["bundle_sha256"],
                     "canary_runs": sorted({item["run_id"] for item in matching if item.get("pass")})[:3],
                     "authorized_at": datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(authorization, indent=2)); print(json.dumps(authorization, indent=2))

if __name__ == "__main__": main()

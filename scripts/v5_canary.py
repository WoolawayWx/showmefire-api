"""Record a non-public V5 canary comparison from one completed shadow forecast."""
from __future__ import annotations
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.v5_shadow import EVIDENCE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_id")
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT); args = parser.parse_args()
    source = args.evidence_root / f"{args.run_id}.prediction.json"
    if not source.exists(): raise SystemExit("completed V5 shadow prediction is required")
    output_dir = args.evidence_root / "canaries"; output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.run_id}.canary.json"
    if output.exists(): raise SystemExit(f"Refusing to overwrite canary {output}")
    record = json.loads(source.read_text()); stable, candidate = record["stable_fm"], record["v5_fm"]
    valid = [math.isfinite(float(a)) and math.isfinite(float(b)) for a, b in zip(stable, candidate)]
    report = {"run_id": args.run_id, "pass": len(stable) == len(candidate) == len(record["row_keys"]) and all(valid)
              and record.get("unavailable", 0) == 0,
              "rows": len(stable), "missing_rows": valid.count(False),
              "category_disagreements": record.get("category_disagreements"), "latency_ms": record.get("latency_ms"),
              "fallback_fraction": sum(record.get("fallback", [])) / max(len(stable), 1),
              "bundle_manifest_sha256": record.get("bundle_manifest_sha256"),
              "recorded_at": datetime.now(timezone.utc).isoformat()}
    output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)

if __name__ == "__main__": main()

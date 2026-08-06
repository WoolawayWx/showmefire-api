"""Evaluate accumulated shadow evidence and attach it to the beta candidate."""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from services.model_shadow import record_shadow_gate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None)
    parser.add_argument("--minimum-days", type=int, default=30)
    parser.add_argument("--minimum-elevated-samples", type=int, default=1)
    args = parser.parse_args()
    kwargs = {"minimum_days": args.minimum_days,
              "minimum_elevated_samples": args.minimum_elevated_samples}
    if args.log: kwargs["path"] = Path(args.log)
    print(json.dumps(record_shadow_gate(**kwargs), indent=2))

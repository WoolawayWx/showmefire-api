"""Evaluate accumulated shadow evidence and attach it to the beta candidate.

Writes two independent pieces of evidence into the fuel_moisture beta
candidate's metadata: operational-stability evidence (did beta run without
crashing, `record_shadow_gate`) and ground-truth accuracy evidence (was beta
actually closer to real observed outcomes than stable,
`evaluate_ground_truth_accuracy`). Both must pass before `promote_model.py`
will allow promotion - see `models/versioning.py::validate_promotion_candidate`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.versioning import update_beta_metadata
from services.model_shadow import record_shadow_gate
from services.shadow_ground_truth import evaluate_ground_truth_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None)
    parser.add_argument("--minimum-days", type=int, default=30)
    parser.add_argument("--minimum-elevated-samples", type=int, default=1)
    parser.add_argument("--ground-truth-minimum-days", type=int, default=14)
    parser.add_argument("--ground-truth-minimum-daily-samples", type=int, default=50)
    parser.add_argument("--skip-ground-truth", action="store_true",
                         help="Only evaluate operational-stability shadow evidence, not ground-truth accuracy")
    args = parser.parse_args()
    kwargs = {"minimum_days": args.minimum_days,
              "minimum_elevated_samples": args.minimum_elevated_samples}
    if args.log: kwargs["path"] = Path(args.log)
    stability_evidence = record_shadow_gate(**kwargs)
    result = {"stability": stability_evidence}

    if not args.skip_ground_truth:
        ground_truth_evidence = evaluate_ground_truth_accuracy(
            minimum_days=args.ground_truth_minimum_days,
            minimum_daily_samples=args.ground_truth_minimum_daily_samples,
        )
        update_beta_metadata("fuel_moisture", {"shadow": {"ground_truth": ground_truth_evidence}})
        result["ground_truth"] = ground_truth_evidence

    print(json.dumps(result, indent=2))

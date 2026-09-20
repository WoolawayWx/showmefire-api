"""Promote a beta model candidate to stable (the version serving code loads).

Usage:
    python pipelines/promote_model.py --model fuel_moisture
    python pipelines/promote_model.py --model fuel_moisture --version 1.5.0-beta.1

Note: `fire_danger` remains a registry choice for backward compatibility,
but no live forecast generator calls load_active_model_path("fire_danger")
anywhere in this codebase - public forecasts use the rule-based
core/fire_danger.py instead. Promoting a fire_danger candidate here updates
the registry only; it has no effect on anything currently served. Flagged
for possible removal from the registry entirely, pending explicit
confirmation - not removed unilaterally since some other operator workflow
might still depend on its presence here.
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.versioning import promote, rollback, get_model_entry, validate_promotion_candidate


def main():
    parser = argparse.ArgumentParser(description="Promote a beta model to stable")
    parser.add_argument("--model", required=True,
                         choices=["fuel_moisture", "fire_danger", "fuel_moisture_spatial", "fire_behavior_static",
                                  "fire_risk_fusion", "risk_fusion_glm", "fire_weather_ml", "fire_weather_index"],
                         help="Which model type to promote. Note: fire_risk_fusion has a hard v1 "
                              "advisory_only boundary in validate_promotion_candidate() - this choice "
                              "makes attempting the promotion possible (it used to crash), not "
                              "guaranteed to succeed; lifting that boundary is a separate policy decision. "
                              "risk_fusion_glm/fire_weather_ml/fire_weather_index are migrated shadow "
                              "families (see models/shadow_bundles.py) - promoting here is equivalent to "
                              "the website admin's Activate button, but does not sync "
                              "shadow_bundles.py's own active.json/env-var pointer the way the website's "
                              "/activate endpoint does; prefer the website for those three unless you "
                              "also intend to update the env var by hand.")
    parser.add_argument("--version", default=None,
                         help="Beta version to promote (defaults to whatever is currently in beta)")
    parser.add_argument("--rollback", action="store_true", help="Reactivate a prior stable version")
    args = parser.parse_args()

    if args.rollback:
        version = rollback(args.model, version=args.version)
        print(f"Rolled back {args.model} to stable version {version}.")
        return

    entry = get_model_entry(args.model)
    beta = entry.get("beta")
    stable = entry.get("stable")

    if not beta:
        print(f"No beta candidate registered for {args.model!r}. Nothing to promote.")
        sys.exit(1)

    print(f"Current stable: {stable['version'] if stable else '(none)'} "
          f"{stable.get('performance') if stable else ''}")
    print(f"Beta candidate: {beta['version']} {beta.get('performance')}")
    ground_truth = ((beta.get('metadata') or {}).get('shadow') or {}).get('ground_truth')
    if args.model == "fuel_moisture" and ground_truth is not None:
        print(f"Ground-truth shadow evidence: {ground_truth}")
    blockers = validate_promotion_candidate(args.model, beta)
    if blockers:
        print("Promotion blocked:")
        for blocker in blockers:
            print(f"  - {blocker}")
        sys.exit(1)

    version = promote(args.model, version=args.version)
    print(f"Promoted {args.model} {version} to stable.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run standalone fire-danger-model workflow end-to-end.

This script intentionally remains standalone and does not integrate with
production forecast processes.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"


def run_step(step_name, cmd):
    print(f"\n==> {step_name}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(BASE_DIR))


def latest_evaluation_report():
    candidates = sorted(REPORTS_DIR.glob("evaluation_*.json"))
    if not candidates:
        return None
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Run fire-danger-model data prep, training, and evaluation in order."
    )
    parser.add_argument(
        "--source-csv",
        default=None,
        help="Optional source csv for data prep (defaults to api/data/final_training_data.csv).",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Run only data prep + train.",
    )
    args = parser.parse_args()

    py = sys.executable

    prep_cmd = [py, "data_prep.py"]
    if args.source_csv:
        prep_cmd.extend(["--source-csv", args.source_csv])
    run_step("Data Prep", prep_cmd)

    train_cmd = [
        py,
        "train.py",
        "--n-estimators",
        str(args.n_estimators),
        "--learning-rate",
        str(args.learning_rate),
        "--max-depth",
        str(args.max_depth),
    ]
    run_step("Train", train_cmd)

    if not args.skip_evaluate:
        eval_cmd = [py, "evaluate.py"]
        run_step("Evaluate", eval_cmd)

        latest_report = latest_evaluation_report()
        if latest_report is not None:
            with latest_report.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            gates = payload.get("gates", {})
            overall = gates.get("overall", {}).get("passed")
            macro_f1 = payload.get("category", {}).get("macro_f1")
            baseline_f1 = payload.get("baseline", {}).get("macro_f1")
            print("\nEvaluation Gate Summary")
            print(f"  report: {latest_report}")
            print(f"  macro_f1: {macro_f1}")
            print(f"  baseline_macro_f1: {baseline_f1}")
            print(f"  overall_passed: {overall}")

    print("\nStandalone fire-danger-model pipeline completed.")


if __name__ == "__main__":
    main()

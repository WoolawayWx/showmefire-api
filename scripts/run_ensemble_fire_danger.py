"""
Run the Day-1 ensemble fire danger product (BETA) once.

Called from scripts/forecasts.sh right after DailyForecast.py, as a
non-blocking step (a failure here never blocks or changes the public
forecast). See services/ensemble_fire_danger/runtime.py for what one run
does, and model-training/docs/ensemble_fire_danger.md for the design.

Usage:
    python scripts/run_ensemble_fire_danger.py                 # today's 12z-anchored run
    python scripts/run_ensemble_fire_danger.py --date 2026-09-22
    python scripts/run_ensemble_fire_danger.py --no-upload --primary ensprod

Environment (all optional):
    SMF_ENSEMBLE_FD_ENABLED   false = skip entirely (default true)
    SMF_ENSEMBLE_FD_PRIMARY   members | ensprod - which track feeds the public images (default members)
    SMF_ENSEMBLE_FD_UPLOAD    false = never upload to R2 (default true; uploadForecast=false also disables)
    SMF_ENSEMBLE_FD_BUNDLE    raw calibration bundle dir (overrides the model registry)
    SMF_ENSEMBLE_FM_MODEL     XGBoost fuel-moisture model path (default: registry stable fuel_moisture)
    SMF_ENSEMBLE_FD_CACHE     member-run cache dir (default /app/cache/ensemble or data/cache/ensemble)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(API_ROOT / ".env")
except ImportError:
    pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", help="forecast day (YYYY-MM-DD); anchors on that day's 12z cycle")
    parser.add_argument("--no-upload", action="store_true", help="never upload to R2 for this run")
    parser.add_argument("--no-render", action="store_true", help="compute + evidence only, no graphics")
    parser.add_argument("--primary", choices=("members", "ensprod"), help="override SMF_ENSEMBLE_FD_PRIMARY")
    parser.add_argument("--workers", type=int, default=6, help="parallel member fetches (default 6)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    if os.getenv("SMF_ENSEMBLE_FD_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        print("ensemble fire danger disabled (SMF_ENSEMBLE_FD_ENABLED=false)")
        return 0
    if args.primary:
        os.environ["SMF_ENSEMBLE_FD_PRIMARY"] = args.primary

    from services.ensemble_fire_danger import runtime

    anchor = None
    now = None
    if args.date:
        day = datetime.strptime(args.date, "%Y-%m-%d")
        anchor = day.replace(hour=12)
        # For a past date, "now" is the end of that forecast day, so the plan
        # uses the cycles that existed then (not ones published later).
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        now = utc_now if day.date() >= utc_now.date() else day.replace(hour=23, minute=59)
    try:
        summary = runtime.run(anchor, now=now, render=not args.no_render,
                              upload=False if args.no_upload else None, workers=args.workers)
    except Exception as error:
        runtime.record_failure(error)
        logging.exception("ensemble fire danger run failed")
        return 1
    compact = {
        "run_id": summary["run_id"],
        "primary_track": summary["primary_track"],
        "bundle": summary["bundle"],
        "fm_anchor": summary["fm_anchor"],
        "members": {p["member_id"]: f"{p['status']} {p['cycle_utc'] or ''}".strip() for p in summary["plan"]},
        "tracks": {n: {"members": t["member_count"], "degraded": t["degraded"],
                       "max_prob": t["statewide_max_neighborhood_probability"]} for n, t in summary["tracks"].items()},
        "public_images": [Path(p).name for p in summary["images"]["public"]],
        "runtime_sec": summary["runtime_sec"],
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Build seven-day fire outlook verification records from matured windows (run daily; idempotent).

Joins each recent run's forecast-side rows with the observations table once a window ended more than
SMF_FIRE_OUTLOOK_VERIFY_LATENCY_HOURS (default 6) ago, and writes <worker root>/verification/records/<run_id>.(parquet|csv).
Copy that folder to the training machine ($SMF_DATA_ROOT/fire_outlook/verification) and run
`python -m fire_outlook.retrain` in model-training. See services/fire_outlook/verification.py for the column definitions.

    python scripts/export_fire_outlook_verification.py [--db PATH] [--max-age-days 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
    parser.add_argument("--db", help="observations database (default: the app's showmefire.db)")
    parser.add_argument("--max-age-days", type=float, default=10.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    from core.database import get_db_path
    from services.fire_outlook import verification, worker
    root = worker.Config.from_env().root
    latency = float(os.getenv("SMF_FIRE_OUTLOOK_VERIFY_LATENCY_HOURS", verification.LATENCY_HOURS))
    obs = verification.SqliteObservations(Path(args.db) if args.db else get_db_path())
    result = verification.build_all(root, obs, max_age_days=args.max_age_days, latency_hours=latency)
    print(json.dumps(result, indent=2))
    return 1 if any(str(v).startswith("error") for v in result.values()) else 0


if __name__ == "__main__":
    sys.exit(main())

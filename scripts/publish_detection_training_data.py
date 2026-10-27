"""
Export detections, incidents and feedback and publish them to Cloudflare R2 for the training machine.

Needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (R2_TRAINING_BUCKET / R2_TRAINING_PREFIX optional).
The training side downloads it with `python -m detection_dataset.build --pull`.

Usage:
    python scripts/publish_detection_training_data.py
    python scripts/publish_detection_training_data.py --since 2025-12-01T00:00:00Z --keep 30
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.cdn_datasets import configured
from services.detection_r2 import DEFAULT_KEEP, publish


def main():
    parser = argparse.ArgumentParser(description="Publish the detection training export to R2.")
    parser.add_argument("--db", default=None, help="SQLite DB to read (default: the API's own DB)")
    parser.add_argument("--since", default=None)
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help="snapshots to keep in R2")
    args = parser.parse_args()
    if not configured():
        sys.exit("R2 credentials are not configured (R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY)")
    print(json.dumps(publish(Path(args.db) if args.db else None, args.since, args.keep), indent=2))


if __name__ == "__main__":
    main()

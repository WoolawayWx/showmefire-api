"""
Run the seven-day fire outlook worker (isolated; never registered in the API scheduler).

    python scripts/run_fire_outlook.py                  # process the latest 00Z/12Z cycle if not already done (cron, every 30 min)
    python scripts/run_fire_outlook.py --watch          # same, looping every 30 minutes
    python scripts/run_fire_outlook.py --cycle 2026-10-29T12
    python scripts/run_fire_outlook.py --no-publish     # compute + verify locally, no CDN upload
    python scripts/run_fire_outlook.py --status         # print status.json and exit

Environment (all optional):
    SMF_FIRE_OUTLOOK_ENABLED            true to run at all (default FALSE until the staff-only pilot is approved)
    SMF_FIRE_OUTLOOK_PUBLISH            true to upload runs to the (public) CDN dataset (default FALSE: the pilot is staff-only)
    SMF_FIRE_OUTLOOK_PUBLIC             true to open the public /fire-outlook API routes (default FALSE; staff use /api/admin/fire-outlook)
    SMF_FIRE_OUTLOOK_DIR                worker root (default data/fire-outlook)
    SMF_FIRE_OUTLOOK_MEMORY_LIMIT_GB    address-space ceiling (default 8)
    SMF_FIRE_OUTLOOK_THREADS            processing threads (default 1)
    SMF_FIRE_OUTLOOK_PUBLIC_KEEP_DAYS   public runs kept (default 7); SMF_FIRE_OUTLOOK_CACHE_KEEP_DAYS (default 30)

Exit codes: 0 ok/waiting/disabled, 1 run failed (recorded in status.json), 3 acquisition not implemented.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(API_ROOT / ".env")
except ImportError:
    pass

CHECK_SECONDS = 30 * 60


def build_adapters(config):
    """Wire the real stages. GEFS 168 h + NBM acquisition (services/fire_outlook/acquisition.py) does not exist
    yet, so this refuses rather than fabricating inputs."""
    try:
        from services.fire_outlook import acquisition
    except ImportError as error:
        raise NotImplementedError(
            "GEFS 168-hour / NBM acquisition (services/fire_outlook/acquisition.py) is not implemented yet") from error
    return acquisition.build_adapters(config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--cycle", help="UTC cycle, e.g. 2026-10-29T12 (default: latest 00Z/12Z)")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    from services.fire_outlook import worker
    config = worker.Config.from_env()
    config.publish = config.publish and not args.no_publish   # SMF_FIRE_OUTLOOK_PUBLISH (default false) AND not --no-publish
    if args.status:
        print(config.status_path.read_text(encoding="utf-8") if config.status_path.exists() else json.dumps({"state": "never_run"}))
        return 0
    if os.getenv("SMF_FIRE_OUTLOOK_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        print("fire outlook worker disabled (set SMF_FIRE_OUTLOOK_ENABLED=true)")
        return 0
    try:
        adapters = build_adapters(config)
    except NotImplementedError as error:
        print(error)
        return 3

    while True:
        if args.cycle:
            cycle = datetime.strptime(args.cycle, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            result = worker.run_cycle(cycle, adapters, config)
        else:
            result = worker.run_once(adapters, config)
        print(json.dumps(result, default=str))
        if not args.watch or args.cycle:
            return 0 if result["ok"] else 1
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    sys.exit(main())

"""Alert staff (Discord, via services.system_alerts) when an automated product
goes stale, and again when it recovers.

Run at the end of every scripts/maps.sh cycle. Each run is a fresh process,
so the alerting state (which products were already reported) lives in a small
JSON file: an alert goes out when a product turns stale, a reminder every
REMINDER_HOURS while it stays stale, and a recovery notice when it is fine again.
Pass --dry-run to print what would be sent without sending it.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from services.product_health import REMINDER_HOURS, check_all, plan_alerts  # noqa: E402,F401

STATE_PATH = PROJECT_DIR / "data" / "current_conditions" / "product_health_state.json"


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    report = check_all(now)
    alerts, new_state = plan_alerts(report, _load_state(), now)
    for message, detail in alerts:
        if args.dry_run:
            print(f"[dry-run] {message}: {detail}")
            continue
        from services.system_alerts import notify_system_issue

        sent = notify_system_issue("product-health", message, detail=detail)
        print(f"{'Sent' if sent else 'Not sent'}: {message}")
    if not args.dry_run:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    print(f"Product health: {report['status']} ({report['stale_count']} stale)")


if __name__ == "__main__":
    main()

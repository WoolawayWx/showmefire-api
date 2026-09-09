"""Manually regenerate fire-incident graphics.

Examples:
    python scripts/render_fire_incident_graphics.py --all
    python scripts/render_fire_incident_graphics.py --incident-id 42
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.fire_incident_graphics import refresh_incident_graphics


def main() -> None:
    parser = argparse.ArgumentParser(description="Render shareable fire incident PNG graphics.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="Render every incident, including small clusters.")
    target.add_argument("--incident-id", type=int, help="Render one incident by database ID.")
    args = parser.parse_args()
    result = refresh_incident_graphics(incident_id=args.incident_id, force=True)
    print(f"Rendered {result['rendered']} incident graphic(s).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""List the admin routes that still only check "is someone signed in" instead of a named permission.

  python scripts/list_unguarded_admin_routes.py

Imports the real app, so it needs the full production dependencies. Prints the gap grouped by the file each route lives in, so
it can be worked down: give a route a permission (`require_permission(...)` or `@requires(...)` + `authorize(...)`, see
core/admin_auth/permissions.py) and it disappears from this list. Sign-in routes appear here by nature and are not a gap.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.admin_auth import permissions  # noqa: E402
from main import app  # noqa: E402


def main() -> None:
    by_file = defaultdict(list)
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith("/api/admin") or not getattr(route, "methods", None) or permissions.route_permissions(route):
            continue
        module = getattr(getattr(route, "endpoint", None), "__module__", "?")
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            by_file[module].append(f"{method:6} {path}")
    total = sum(len(rows) for rows in by_file.values())
    for module in sorted(by_file):
        print(f"\n{module}  ({len(by_file[module])})")
        for row in sorted(by_file[module]):
            print(f"  {row}")
    print(f"\n{total} admin route(s) without a named permission")


if __name__ == "__main__":
    main()

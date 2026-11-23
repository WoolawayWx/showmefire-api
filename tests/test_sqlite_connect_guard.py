"""Application code opens the database through core.database.connect (15 s busy timeout, synchronous=NORMAL).

A bare sqlite3.connect() waits only 5 s for a write lock, so contention with a scheduler job surfaces as "database is
locked". This keeps new call sites from slipping back in. Connections that already pass an explicit timeout, in-memory
databases and read-only URIs are fine.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("core", "services", "routers", "forecast", "forecast_v1", "maps", "alerts", "analysis")
CALL = re.compile(r"sqlite3\.connect\(")


class SqliteConnectGuard(unittest.TestCase):
    def test_no_bare_sqlite_connect(self):
        offenders = []
        for folder in FOLDERS:
            for path in (ROOT / folder).rglob("*.py"):
                if "venv" in path.parts or "__pycache__" in path.parts:
                    continue
                for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if CALL.search(line) and not any(token in line for token in ("timeout", ":memory:", "mode=ro")):
                        offenders.append(f"{path.relative_to(ROOT)}:{number}")
        # core/database.py's own helper is the only place that may call it bare (it passes timeout= on another line).
        offenders = [item for item in offenders if not item.startswith("core/database.py:")]
        self.assertEqual(offenders, [], "use core.database.connect instead of sqlite3.connect")


if __name__ == "__main__":
    unittest.main()

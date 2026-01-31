#!/bin/sh
set -e

DATA_DIR="${DATA_DIR:-/app/data}"

# Ensure data dir exists and is writable
mkdir -p "$DATA_DIR"
chmod 755 "$DATA_DIR" || true

# Initialize the sqlite DB (idempotent)
python3 - <<'PY'
try:
    from core.database import init_database
    init_database()
except Exception as e:
    import sys, traceback
    print("DB init error:", e, file=sys.stderr)
    traceback.print_exc()
PY

# Exec the main process
exec "$@"

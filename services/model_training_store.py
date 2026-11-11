"""SQLite persistence for the outbound training runner."""
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager

from core.database import get_db_path


@contextmanager
def connect():
    conn = sqlite3.connect(str(get_db_path()), timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS mt_runners (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, digest TEXT NOT NULL UNIQUE,
                actor TEXT NOT NULL, revoked INTEGER DEFAULT 0, seen REAL, snapshot TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS mt_control (
                id INTEGER PRIMARY KEY CHECK (id=1), paused INTEGER NOT NULL, actor TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mt_requests (
                id TEXT PRIMARY KEY, runner TEXT NOT NULL, model TEXT NOT NULL, stage TEXT,
                actor TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                status TEXT DEFAULT 'queued', cancel INTEGER DEFAULT 0
            );
        """)
        with conn:
            yield conn
    finally:
        conn.close()


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def create(conn, name, actor):
    runner_id = secrets.token_hex(12)
    token = "smfrun_" + secrets.token_urlsafe(32)
    conn.execute("INSERT INTO mt_runners(id,name,digest,actor) VALUES (?,?,?,?)",
                 (runner_id, name, digest(token), actor))
    return {"id": runner_id, "token": token}


def paused(conn):
    row = conn.execute("SELECT paused FROM mt_control WHERE id=1").fetchone()
    return bool(row and row["paused"])


def expire(conn, now):
    conn.execute("UPDATE mt_requests SET status='expired' WHERE status='queued' AND created < ?", (now - 900,))
    conn.execute("UPDATE mt_requests SET status='lost' WHERE status IN ('claimed','running') AND updated < ?", (now - 900,))


def sync(conn, runner_id, snapshot):
    now = time.time()
    conn.execute("BEGIN IMMEDIATE")
    updated = conn.execute("UPDATE mt_runners SET seen=?,snapshot=? WHERE id=? AND revoked=0",
                           (now, json.dumps(snapshot), runner_id))
    if not updated.rowcount:
        return None
    for job in snapshot["jobs"]:
        conn.execute(
            "UPDATE mt_requests SET status=?,updated=? WHERE runner=? AND id=? AND status IN ('claimed','running','lost')",
            (job["status"], now, runner_id, job["id"]),
        )
    expire(conn, now)
    cancels = [row["id"] for row in conn.execute(
        "SELECT id FROM mt_requests WHERE runner=? AND cancel=1 AND status IN ('claimed','running')", (runner_id,),
    )]
    assignment = None
    if snapshot["remote"] and not paused(conn) and not conn.execute(
        "SELECT 1 FROM mt_requests WHERE runner=? AND status IN ('claimed','running')", (runner_id,),
    ).fetchone():
        row = conn.execute(
            "SELECT id,model,stage FROM mt_requests WHERE runner=? AND status='queued' ORDER BY created LIMIT 1",
            (runner_id,),
        ).fetchone()
        if row:
            conn.execute("UPDATE mt_requests SET status='claimed',updated=? WHERE id=?", (now, row["id"]))
            assignment = dict(row)
    return {"assignment": assignment, "cancel": cancels}


def overview(conn):
    now = time.time()
    expire(conn, now)
    runners = []
    for row in conn.execute("SELECT id,name,seen,snapshot FROM mt_runners WHERE revoked=0 ORDER BY name"):
        runner = dict(row)
        runner["online"] = row["seen"] is not None and now - row["seen"] < 60
        runner["snapshot"] = json.loads(row["snapshot"])
        runners.append(runner)
    requests = [dict(row) for row in conn.execute("SELECT * FROM mt_requests ORDER BY created DESC LIMIT 50")]
    return {"runners": runners, "requests": requests}

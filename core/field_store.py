# Copyright (c) 2026 Show Me Fire. PolyForm Internal Use License 1.0.0; see LICENSE-FIELD.md.
"""
Storage for ShowMeFire Field team collaboration: teams, members, shared maps with an
ordered operation log, and last-known member positions.

Design notes
------------
* Tokens are random secrets; only their SHA-256 is stored, so a leaked database does not
  leak working credentials. Join codes are stored the same way.
* A shared map is a set of features keyed by a client-chosen ``uid``. Every change is an
  operation (``upsert`` or ``delete``) that receives the next integer ``rev`` for that map,
  so all clients apply changes in the same order and converge (last writer wins per feature).
* Deleted features keep a tombstone row so a late ``upsert`` of an older edit cannot
  resurrect them silently; the operation log is pruned to ``OPS_KEEP`` entries per map and
  clients that fall behind the pruned range are told to reload the whole map.
* Positions keep only the newest report per member, ordered by a per-team sequence number
  so clients can ask "what changed since N".
"""
import hashlib
import json
import secrets
import sqlite3
import time
from typing import Any, Dict, List, Optional

from core import field_accounts as accounts
from core.database import get_db_path

TOKEN_PREFIX = "fld_"
# No 0/O/1/I/L so a code read aloud over a radio is not ambiguous.
JOIN_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
JOIN_CODE_LEN = 10

MAX_MEMBERS_PER_TEAM = 50
MAX_MAPS_PER_TEAM = 50
MAX_FEATURES_PER_MAP = 20000
MAX_FEATURE_BYTES = 64 * 1024
MAX_OPS_PER_BATCH = 500
OPS_KEEP = 5000
POSITION_MAX_AGE_SECONDS = 24 * 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS field_teams (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    join_code_hash TEXT NOT NULL,
    pos_seq INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_field_teams_join ON field_teams(join_code_hash);

CREATE TABLE IF NOT EXISTS field_accounts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    contact TEXT NOT NULL DEFAULT '',
    plan TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    expires_at INTEGER,
    token_hash TEXT NOT NULL UNIQUE,
    limits_json TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS field_usage (
    team_id TEXT NOT NULL REFERENCES field_teams(id) ON DELETE CASCADE,
    month TEXT NOT NULL,
    ops INTEGER NOT NULL DEFAULT 0,
    positions INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (team_id, month)
);

CREATE TABLE IF NOT EXISTS field_members (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES field_teams(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    is_owner INTEGER NOT NULL DEFAULT 0,
    revoked INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    last_seen INTEGER
);
CREATE INDEX IF NOT EXISTS idx_field_members_team ON field_members(team_id);

CREATE TABLE IF NOT EXISTS field_maps (
    id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL REFERENCES field_teams(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    rev INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_field_maps_team ON field_maps(team_id);

CREATE TABLE IF NOT EXISTS field_features (
    map_id TEXT NOT NULL REFERENCES field_maps(id) ON DELETE CASCADE,
    uid TEXT NOT NULL,
    feature_json TEXT,
    deleted INTEGER NOT NULL DEFAULT 0,
    rev INTEGER NOT NULL,
    PRIMARY KEY (map_id, uid)
);

CREATE TABLE IF NOT EXISTS field_ops (
    map_id TEXT NOT NULL REFERENCES field_maps(id) ON DELETE CASCADE,
    rev INTEGER NOT NULL,
    uid TEXT NOT NULL,
    op TEXT NOT NULL,
    feature_json TEXT,
    member_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    PRIMARY KEY (map_id, rev)
);

CREATE TABLE IF NOT EXISTS field_positions (
    member_id TEXT PRIMARY KEY REFERENCES field_members(id) ON DELETE CASCADE,
    team_id TEXT NOT NULL REFERENCES field_teams(id) ON DELETE CASCADE,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    accuracy REAL,
    speed REAL,
    course REAL,
    alt REAL,
    cleared INTEGER NOT NULL DEFAULT 0,
    seq INTEGER NOT NULL,
    ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_field_positions_team ON field_positions(team_id, seq);
"""


class FieldError(accounts.AccountError):
    """A request the store refuses; ``status`` is the HTTP status the router should use.

    It subclasses ``AccountError`` so the router can handle store and account refusals alike.
    """


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def ensure_field_schema() -> None:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        # Teams created before accounts existed have no account (operator-created).
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(field_teams)")}
        if "account_id" not in cols:
            conn.execute("ALTER TABLE field_teams ADD COLUMN account_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_field_teams_account ON field_teams(account_id)")
        conn.commit()
    finally:
        conn.close()


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _new_id() -> str:
    return secrets.token_hex(8)


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def new_join_code() -> str:
    return "".join(secrets.choice(JOIN_ALPHABET) for _ in range(JOIN_CODE_LEN))


def normalize_join_code(code: str) -> str:
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


def _now() -> int:
    return int(time.time())


# ------------------------------------------------------------------ teams and members

def create_team(
    team_name: str, owner_display_name: str, account_token: Optional[str] = None
) -> Dict[str, Any]:
    """Create a team. With an account token the account's plan limits apply; without one the
    caller has already proven they hold the operator key (checked by the router)."""
    ensure_field_schema()
    team_id, member_id = _new_id(), _new_id()
    code, token = new_join_code(), new_token()
    conn = _connect()
    try:
        account_id = None
        if account_token is not None:
            acct = accounts.account_for_token(conn, account_token)
            accounts.require_active(acct)
            limits = accounts.resolve_limits(acct["plan"], acct["limits_json"])
            have = conn.execute(
                "SELECT COUNT(*) FROM field_teams WHERE account_id=?", (acct["id"],)
            ).fetchone()[0]
            if have >= limits["max_teams"]:
                raise FieldError(
                    f"Your {acct['plan']} plan allows {limits['max_teams']} team(s). "
                    "Contact Show Me Fire to add more.",
                    403,
                )
            account_id = acct["id"]
        conn.execute(
            "INSERT INTO field_teams (id, name, join_code_hash, created_at, account_id) VALUES (?,?,?,?,?)",
            (team_id, team_name, _hash(code), _now(), account_id),
        )
        conn.execute(
            "INSERT INTO field_members (id, team_id, display_name, token_hash, is_owner, created_at)"
            " VALUES (?,?,?,?,1,?)",
            (member_id, team_id, owner_display_name, _hash(token), _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "team_id": team_id,
        "team_name": team_name,
        "join_code": code,
        "member_id": member_id,
        "token": token,
    }


def join_team(join_code: str, display_name: str) -> Dict[str, Any]:
    ensure_field_schema()
    code = normalize_join_code(join_code)
    conn = _connect()
    try:
        team = conn.execute(
            "SELECT id, name FROM field_teams WHERE join_code_hash=?", (_hash(code),)
        ).fetchone()
        if not team:
            raise FieldError("That join code is not valid.", 404)
        ent = accounts.team_entitlement(conn, team["id"])
        accounts.require_writable(ent)
        count = conn.execute(
            "SELECT COUNT(*) FROM field_members WHERE team_id=? AND revoked=0", (team["id"],)
        ).fetchone()[0]
        if count >= min(MAX_MEMBERS_PER_TEAM, ent["limits"]["max_members"]):
            raise FieldError("That team is full.", 409)
        member_id, token = _new_id(), new_token()
        conn.execute(
            "INSERT INTO field_members (id, team_id, display_name, token_hash, created_at)"
            " VALUES (?,?,?,?,?)",
            (member_id, team["id"], display_name, _hash(token), _now()),
        )
        conn.commit()
        return {
            "team_id": team["id"],
            "team_name": team["name"],
            "member_id": member_id,
            "token": token,
        }
    finally:
        conn.close()


def authenticate(token: Optional[str]) -> Dict[str, Any]:
    """Return the active member for a bearer token, or raise 401."""
    if not token or not token.startswith(TOKEN_PREFIX) or len(token) > 128:
        raise FieldError("Missing or invalid token.", 401)
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT m.id, m.team_id, m.display_name, m.is_owner, t.name AS team_name"
            " FROM field_members m JOIN field_teams t ON t.id = m.team_id"
            " WHERE m.token_hash=? AND m.revoked=0",
            (_hash(token),),
        ).fetchone()
        if not row:
            raise FieldError("Missing or invalid token.", 401)
        conn.execute("UPDATE field_members SET last_seen=? WHERE id=?", (_now(), row["id"]))
        conn.commit()
        return dict(row)
    finally:
        conn.close()


def list_members(team_id: str) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, display_name, is_owner, last_seen FROM field_members"
            " WHERE team_id=? AND revoked=0 ORDER BY created_at",
            (team_id,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "display_name": r["display_name"],
                "is_owner": bool(r["is_owner"]),
                "last_seen": r["last_seen"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def rotate_join_code(member: Dict[str, Any]) -> str:
    _require_owner(member)
    code = new_join_code()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE field_teams SET join_code_hash=? WHERE id=?", (_hash(code), member["team_id"])
        )
        conn.commit()
    finally:
        conn.close()
    return code


def revoke_member(owner: Dict[str, Any], member_id: str) -> None:
    _require_owner(owner)
    if member_id == owner["id"]:
        raise FieldError("Owners cannot remove themselves.", 400)
    conn = _connect()
    try:
        cur = conn.execute(
            "UPDATE field_members SET revoked=1 WHERE id=? AND team_id=?",
            (member_id, owner["team_id"]),
        )
        if cur.rowcount == 0:
            raise FieldError("No such member.", 404)
        # Tell teammates to drop the removed member's marker.
        seq = _bump_seq(conn, owner["team_id"])
        conn.execute(
            "UPDATE field_positions SET cleared=1, seq=?, ts=? WHERE member_id=?",
            (seq, _now(), member_id),
        )
        conn.commit()
    finally:
        conn.close()


def _require_owner(member: Dict[str, Any]) -> None:
    if not member.get("is_owner"):
        raise FieldError("Only the team owner can do that.", 403)


# ------------------------------------------------------------------ maps

def _map_row(conn: sqlite3.Connection, member: Dict[str, Any], map_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, team_id, name, rev, updated_at FROM field_maps WHERE id=?", (map_id,)
    ).fetchone()
    # A map in another team looks exactly like a missing one.
    if not row or row["team_id"] != member["team_id"]:
        raise FieldError("No such map.", 404)
    return row


def _clean_feature(feature: Any) -> str:
    if not isinstance(feature, dict):
        raise FieldError("A feature must be a JSON object.", 422)
    text = json.dumps(feature, separators=(",", ":"), ensure_ascii=False)
    if len(text.encode("utf-8")) > MAX_FEATURE_BYTES:
        raise FieldError("A feature is too large.", 413)
    return text


def create_map(member: Dict[str, Any], name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    if len(features) > MAX_FEATURES_PER_MAP:
        raise FieldError("Too many features.", 413)
    cleaned = {uid: _clean_feature(f) for uid, f in features.items()}
    for uid in cleaned:
        _check_uid(uid)
    map_id = _new_id()
    now = _now()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        ent = accounts.team_entitlement(conn, member["team_id"])
        accounts.require_writable(ent)
        count = conn.execute(
            "SELECT COUNT(*) FROM field_maps WHERE team_id=?", (member["team_id"],)
        ).fetchone()[0]
        if count >= min(MAX_MAPS_PER_TEAM, ent["limits"]["max_maps"]):
            raise FieldError("This team has too many maps for its plan.", 409)
        if len(cleaned) > min(MAX_FEATURES_PER_MAP, ent["limits"]["max_features"]):
            raise FieldError("Too many features for this plan.", 413)
        conn.execute(
            "INSERT INTO field_maps (id, team_id, name, rev, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (map_id, member["team_id"], name, 0, now, now),
        )
        rev = 0
        for uid, text in cleaned.items():
            rev += 1
            conn.execute(
                "INSERT INTO field_features (map_id, uid, feature_json, deleted, rev) VALUES (?,?,?,0,?)",
                (map_id, uid, text, rev),
            )
            conn.execute(
                "INSERT INTO field_ops (map_id, rev, uid, op, feature_json, member_id, ts) VALUES (?,?,?,?,?,?,?)",
                (map_id, rev, uid, "upsert", text, member["id"], now),
            )
        conn.execute("UPDATE field_maps SET rev=? WHERE id=?", (rev, map_id))
        accounts.bump_usage(conn, member["team_id"], ops=rev)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"id": map_id, "name": name, "rev": rev}


def list_maps(member: Dict[str, Any]) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT id, name, rev, updated_at FROM field_maps WHERE team_id=? ORDER BY updated_at DESC",
            (member["team_id"],),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_map(member: Dict[str, Any], map_id: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        m = _map_row(conn, member, map_id)
        rows = conn.execute(
            "SELECT uid, feature_json FROM field_features WHERE map_id=? AND deleted=0", (map_id,)
        ).fetchall()
        return {
            "id": m["id"],
            "name": m["name"],
            "rev": m["rev"],
            "features": {r["uid"]: json.loads(r["feature_json"]) for r in rows},
        }
    finally:
        conn.close()


def delete_map(owner: Dict[str, Any], map_id: str) -> None:
    _require_owner(owner)
    conn = _connect()
    try:
        _map_row(conn, owner, map_id)
        conn.execute("DELETE FROM field_maps WHERE id=?", (map_id,))
        conn.commit()
    finally:
        conn.close()


def _check_uid(uid: Any) -> str:
    if not isinstance(uid, str) or not (8 <= len(uid) <= 64):
        raise FieldError("Feature ids must be 8 to 64 characters.", 422)
    if not all(ch.isalnum() or ch in "-_" for ch in uid):
        raise FieldError("Feature ids may only contain letters, digits, '-' and '_'.", 422)
    return uid


def apply_ops(member: Dict[str, Any], map_id: str, ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply a batch atomically; each operation takes the next revision."""
    if not ops:
        raise FieldError("No operations were sent.", 422)
    if len(ops) > MAX_OPS_PER_BATCH:
        raise FieldError("Too many operations in one request.", 413)
    prepared = []
    for o in ops:
        kind = o.get("op")
        uid = _check_uid(o.get("uid"))
        if kind == "upsert":
            prepared.append((kind, uid, _clean_feature(o.get("feature"))))
        elif kind == "delete":
            prepared.append((kind, uid, None))
        else:
            raise FieldError("Unknown operation.", 422)
    now = _now()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        m = _map_row(conn, member, map_id)
        ent = accounts.team_entitlement(conn, member["team_id"])
        accounts.require_writable(ent)
        feature_cap = min(MAX_FEATURES_PER_MAP, ent["limits"]["max_features"])
        rev = m["rev"]
        live = conn.execute(
            "SELECT COUNT(*) FROM field_features WHERE map_id=? AND deleted=0", (map_id,)
        ).fetchone()[0]
        uids = list({uid for _, uid, _ in prepared})
        placeholders = ",".join("?" * len(uids))
        existing_by_uid = {
            row["uid"]: bool(row["deleted"])
            for row in conn.execute(
                f"SELECT uid, deleted FROM field_features WHERE map_id=? AND uid IN ({placeholders})",
                (map_id, *uids),
            ).fetchall()
        }
        for kind, uid, text in prepared:
            existing_deleted = existing_by_uid.get(uid)
            rev += 1
            if kind == "upsert":
                if (existing_deleted is None or existing_deleted) and live >= feature_cap:
                    raise FieldError("This map has too many features for its plan.", 413)
                if existing_deleted is None or existing_deleted:
                    live += 1
                conn.execute(
                    "INSERT INTO field_features (map_id, uid, feature_json, deleted, rev) VALUES (?,?,?,0,?)"
                    " ON CONFLICT(map_id, uid) DO UPDATE SET feature_json=excluded.feature_json,"
                    " deleted=0, rev=excluded.rev",
                    (map_id, uid, text, rev),
                )
                existing_by_uid[uid] = False
            else:
                if existing_deleted is not None and not existing_deleted:
                    live -= 1
                conn.execute(
                    "INSERT INTO field_features (map_id, uid, feature_json, deleted, rev) VALUES (?,?,NULL,1,?)"
                    " ON CONFLICT(map_id, uid) DO UPDATE SET feature_json=NULL, deleted=1, rev=excluded.rev",
                    (map_id, uid, rev),
                )
                existing_by_uid[uid] = True
            conn.execute(
                "INSERT INTO field_ops (map_id, rev, uid, op, feature_json, member_id, ts) VALUES (?,?,?,?,?,?,?)",
                (map_id, rev, uid, kind, text, member["id"], now),
            )
        conn.execute("UPDATE field_maps SET rev=?, updated_at=? WHERE id=?", (rev, now, map_id))
        conn.execute(
            "DELETE FROM field_ops WHERE map_id=? AND rev <= ?", (map_id, rev - OPS_KEEP)
        )
        accounts.bump_usage(conn, member["team_id"], ops=len(prepared))
        conn.commit()
        return {"rev": rev, "applied": len(prepared)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ------------------------------------------------------------------ events and positions

def _position_rows(conn, team_id: str, since_seq: int) -> List[Dict[str, Any]]:
    cutoff = _now() - POSITION_MAX_AGE_SECONDS
    rows = conn.execute(
        "SELECT p.member_id, m.display_name, p.lat, p.lng, p.accuracy, p.speed, p.course,"
        " p.alt, p.cleared, p.seq, p.ts FROM field_positions p JOIN field_members m ON m.id = p.member_id"
        " WHERE p.team_id=? AND p.seq>? AND p.ts>? ORDER BY p.seq",
        (team_id, since_seq, cutoff),
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["cleared"] = bool(d["cleared"])
        d["ts"] = int(d["ts"])
        out.append(d)
    return out


def _bump_seq(conn: sqlite3.Connection, team_id: str) -> int:
    conn.execute("UPDATE field_teams SET pos_seq = pos_seq + 1 WHERE id=?", (team_id,))
    return conn.execute("SELECT pos_seq FROM field_teams WHERE id=?", (team_id,)).fetchone()["pos_seq"]


def events_since(
    member: Dict[str, Any], map_id: Optional[str], since_rev: int, since_pos: int
) -> Dict[str, Any]:
    """Everything new for this client: map operations after ``since_rev`` and positions after
    ``since_pos``. ``reset`` tells the client its copy is too old and must be reloaded."""
    conn = _connect()
    try:
        out: Dict[str, Any] = {"rev": since_rev, "ops": [], "reset": False}
        if map_id:
            m = _map_row(conn, member, map_id)
            out["rev"] = m["rev"]
            if m["rev"] < since_rev:
                # The client is ahead of the server (map recreated): start over.
                out["reset"] = True
            elif m["rev"] > since_rev:
                oldest = conn.execute(
                    "SELECT MIN(rev) FROM field_ops WHERE map_id=?", (map_id,)
                ).fetchone()[0]
                if oldest is None or oldest > since_rev + 1:
                    out["reset"] = True
                else:
                    rows = conn.execute(
                        "SELECT o.rev, o.uid, o.op, o.feature_json, o.member_id FROM field_ops o"
                        " WHERE o.map_id=? AND o.rev>? ORDER BY o.rev LIMIT 1000",
                        (map_id, since_rev),
                    ).fetchall()
                    out["ops"] = [
                        {
                            "rev": r["rev"],
                            "uid": r["uid"],
                            "op": r["op"],
                            "feature": json.loads(r["feature_json"]) if r["feature_json"] else None,
                            "by": r["member_id"],
                        }
                        for r in rows
                    ]
        positions = _position_rows(conn, member["team_id"], since_pos)
        out["positions"] = positions
        out["pos_seq"] = max([since_pos] + [p["seq"] for p in positions])
        return out
    finally:
        conn.close()


def has_news(member: Dict[str, Any], map_id: Optional[str], since_rev: int, since_pos: int) -> bool:
    """Cheap check used by the long-poll loop."""
    conn = _connect()
    try:
        if map_id:
            m = _map_row(conn, member, map_id)
            if m["rev"] != since_rev:
                return True
        seq = conn.execute(
            "SELECT pos_seq FROM field_teams WHERE id=?", (member["team_id"],)
        ).fetchone()
        return bool(seq and seq["pos_seq"] > since_pos)
    finally:
        conn.close()


def report_position(member: Dict[str, Any], pos: Dict[str, Any]) -> Dict[str, Any]:
    # Sub-second precision so the one-per-second limit cannot be dodged at a second boundary.
    now = time.time()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        accounts.require_writable(accounts.team_entitlement(conn, member["team_id"]))
        last = conn.execute(
            "SELECT ts FROM field_positions WHERE member_id=?", (member["id"],)
        ).fetchone()
        if last and now - last["ts"] < 1:
            raise FieldError("Positions can be sent at most once per second.", 429)
        seq = _bump_seq(conn, member["team_id"])
        conn.execute(
            "INSERT INTO field_positions (member_id, team_id, lat, lng, accuracy, speed, course, alt, cleared, seq, ts)"
            " VALUES (?,?,?,?,?,?,?,?,0,?,?) ON CONFLICT(member_id) DO UPDATE SET lat=excluded.lat,"
            " lng=excluded.lng, accuracy=excluded.accuracy, speed=excluded.speed,"
            " course=excluded.course, alt=excluded.alt, cleared=0, seq=excluded.seq, ts=excluded.ts",
            (
                member["id"],
                member["team_id"],
                pos["lat"],
                pos["lng"],
                pos.get("accuracy"),
                pos.get("speed"),
                pos.get("course"),
                pos.get("alt"),
                seq,
                now,
            ),
        )
        accounts.bump_usage(conn, member["team_id"], positions=1)
        conn.commit()
        return {"seq": seq}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_position(member: Dict[str, Any]) -> None:
    """Stop sharing: teammates are told to drop this member's marker."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        seq = _bump_seq(conn, member["team_id"])
        conn.execute(
            "UPDATE field_positions SET cleared=1, seq=?, ts=? WHERE member_id=?",
            (seq, _now(), member["id"]),
        )
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------------ plan information

def team_plan(member: Dict[str, Any]) -> Dict[str, Any]:
    """Plan, quotas and current usage for the member's team (shown in the app)."""
    conn = _connect()
    try:
        ent = accounts.team_entitlement(conn, member["team_id"])
        counts = {
            "members": conn.execute(
                "SELECT COUNT(*) FROM field_members WHERE team_id=? AND revoked=0", (member["team_id"],)
            ).fetchone()[0],
            "maps": conn.execute(
                "SELECT COUNT(*) FROM field_maps WHERE team_id=?", (member["team_id"],)
            ).fetchone()[0],
        }
        return {
            "plan": ent["plan"],
            "status": ent["status"],
            "expires_at": ent["expires_at"],
            "message": ent["reason"],
            "limits": ent["limits"],
            "usage": counts,
        }
    finally:
        conn.close()


# ------------------------------------------------------------------ account administration

def _with_conn(fn):
    ensure_field_schema()
    conn = _connect()
    try:
        return fn(conn)
    finally:
        conn.close()


def admin_create_account(name, contact, plan, expires_at, notes, limits):
    return _with_conn(
        lambda c: accounts.create_account(c, name, contact, plan, expires_at, notes, limits)
    )


def admin_list_accounts():
    def go(conn):
        rows = conn.execute("SELECT * FROM field_accounts ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            v = accounts.public_view(r)
            v["teams"] = conn.execute(
                "SELECT COUNT(*) FROM field_teams WHERE account_id=?", (r["id"],)
            ).fetchone()[0]
            out.append(v)
        return out

    return _with_conn(go)


def admin_update_account(account_id, fields):
    return _with_conn(lambda c: accounts.update_account(c, account_id, fields))


def admin_rotate_account_token(account_id):
    return _with_conn(lambda c: accounts.rotate_account_token(c, account_id))


def admin_usage(month=None):
    return _with_conn(lambda c: accounts.usage_report(c, month))


def account_overview(token):
    """What an account holder sees about their own plan and teams."""

    def go(conn):
        row = accounts.account_for_token(conn, token)
        v = accounts.public_view(row)
        teams = conn.execute(
            "SELECT id, name FROM field_teams WHERE account_id=? ORDER BY created_at", (row["id"],)
        ).fetchall()
        v["teams"] = [
            {
                "id": t["id"],
                "name": t["name"],
                "members": conn.execute(
                    "SELECT COUNT(*) FROM field_members WHERE team_id=? AND revoked=0", (t["id"],)
                ).fetchone()[0],
                "maps": conn.execute(
                    "SELECT COUNT(*) FROM field_maps WHERE team_id=?", (t["id"],)
                ).fetchone()[0],
            }
            for t in teams
        ]
        return v

    return _with_conn(go)

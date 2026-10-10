# Copyright (c) 2026 Show Me Fire. PolyForm Internal Use License 1.0.0; see LICENSE-FIELD.md.
"""
Accounts, plans, quotas and usage metering for the ShowMeFire Field team service.

Only the *server-backed* Field features are gated. The desktop and web apps are free to use
and everything local (maps, drawing, GPS, printing, public Show Me Fire data) never touches
this module. A **team owner** needs an account on a paid plan; **teammates** join the owner's
team with a join code and share the owner's plan quotas.

Accounts are granted by a Show Me Fire administrator (admin JWT) and authenticate with an
account token (``fldacct_...``; only its SHA-256 is stored). A payment processor can later be
wired to ``set_plan``/``extend`` without changing anything else.

Plans are defined here, and an account can carry per-account overrides in ``limits_json``.
"""
import hashlib
import json
import secrets
import sqlite3
import time
from typing import Any, Dict, Optional

ACCOUNT_TOKEN_PREFIX = "fldacct_"

# Defaults per plan. "operator" is used for teams created with the operator key (no account).
PLAN_LIMITS: Dict[str, Dict[str, int]] = {
    "team": {"max_teams": 1, "max_members": 10, "max_maps": 20, "max_features": 5000},
    "agency": {"max_teams": 5, "max_members": 50, "max_maps": 50, "max_features": 20000},
    "operator": {"max_teams": 1000, "max_members": 50, "max_maps": 50, "max_features": 20000},
}
LIMIT_KEYS = ("max_teams", "max_members", "max_maps", "max_features")
# Absolute ceilings no override can exceed (protects the server, not a pricing choice).
HARD_CEILINGS = {"max_teams": 1000, "max_members": 200, "max_maps": 200, "max_features": 50000}
PLAN_NAMES = tuple(p for p in PLAN_LIMITS if p != "operator")


class AccountError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def new_account_token() -> str:
    return ACCOUNT_TOKEN_PREFIX + secrets.token_urlsafe(32)


def _now() -> int:
    return int(time.time())


def resolve_limits(plan: str, overrides_json: Optional[str]) -> Dict[str, int]:
    base = dict(PLAN_LIMITS.get(plan, PLAN_LIMITS["team"]))
    if overrides_json:
        try:
            overrides = json.loads(overrides_json)
        except ValueError:
            overrides = {}
        for k in LIMIT_KEYS:
            v = overrides.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                base[k] = min(v, HARD_CEILINGS[k])
    return base


def clean_limit_overrides(limits: Optional[Dict[str, Any]]) -> Optional[str]:
    if not limits:
        return None
    out = {}
    for k, v in limits.items():
        if k not in LIMIT_KEYS:
            raise AccountError(f"Unknown limit '{k}'.", 422)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > HARD_CEILINGS[k]:
            raise AccountError(f"Limit '{k}' must be a whole number from 0 to {HARD_CEILINGS[k]}.", 422)
        out[k] = v
    return json.dumps(out, sort_keys=True)


def status_of(row: sqlite3.Row, now: Optional[int] = None) -> str:
    """'active', 'suspended' or 'expired'."""
    if row["status"] != "active":
        return "suspended"
    exp = row["expires_at"]
    if exp is not None and exp <= (now if now is not None else _now()):
        return "expired"
    return "active"


def public_view(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "contact": row["contact"],
        "plan": row["plan"],
        "status": status_of(row),
        "expires_at": row["expires_at"],
        "limits": resolve_limits(row["plan"], row["limits_json"]),
        "notes": row["notes"],
        "created_at": row["created_at"],
    }


def create_account(
    conn: sqlite3.Connection,
    name: str,
    contact: str,
    plan: str,
    expires_at: Optional[int],
    notes: str,
    limits: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if plan not in PLAN_NAMES:
        raise AccountError(f"Unknown plan. Choose one of: {', '.join(PLAN_NAMES)}.", 422)
    overrides = clean_limit_overrides(limits)
    account_id = secrets.token_hex(8)
    token = new_account_token()
    conn.execute(
        "INSERT INTO field_accounts (id, name, contact, plan, status, expires_at, token_hash,"
        " limits_json, notes, created_at) VALUES (?,?,?,?, 'active', ?,?,?,?,?)",
        (account_id, name, contact, plan, expires_at, _hash(token), overrides, notes, _now()),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM field_accounts WHERE id=?", (account_id,)).fetchone()
    out = public_view(row)
    out["token"] = token  # shown exactly once
    return out


def account_for_token(conn: sqlite3.Connection, token: Optional[str]) -> sqlite3.Row:
    if not token or not token.startswith(ACCOUNT_TOKEN_PREFIX) or len(token) > 128:
        raise AccountError("Missing or invalid account token.", 401)
    row = conn.execute("SELECT * FROM field_accounts WHERE token_hash=?", (_hash(token),)).fetchone()
    if not row:
        raise AccountError("Missing or invalid account token.", 401)
    return row


def require_active(row: sqlite3.Row) -> None:
    s = status_of(row)
    if s == "suspended":
        raise AccountError("This account is suspended. Contact Show Me Fire.", 402)
    if s == "expired":
        raise AccountError("This account's plan has expired. Contact Show Me Fire to renew.", 402)


def update_account(conn: sqlite3.Connection, account_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
    row = conn.execute("SELECT * FROM field_accounts WHERE id=?", (account_id,)).fetchone()
    if not row:
        raise AccountError("No such account.", 404)
    sets, args = [], []
    if "plan" in fields and fields["plan"] is not None:
        if fields["plan"] not in PLAN_NAMES:
            raise AccountError(f"Unknown plan. Choose one of: {', '.join(PLAN_NAMES)}.", 422)
        sets.append("plan=?")
        args.append(fields["plan"])
    if "status" in fields and fields["status"] is not None:
        if fields["status"] not in ("active", "suspended"):
            raise AccountError("Status must be 'active' or 'suspended'.", 422)
        sets.append("status=?")
        args.append(fields["status"])
    if "expires_at" in fields:
        sets.append("expires_at=?")
        args.append(fields["expires_at"])
    if "limits" in fields:
        sets.append("limits_json=?")
        args.append(clean_limit_overrides(fields["limits"]))
    for k in ("name", "contact", "notes"):
        if k in fields and fields[k] is not None:
            sets.append(f"{k}=?")
            args.append(fields[k])
    if sets:
        conn.execute(f"UPDATE field_accounts SET {', '.join(sets)} WHERE id=?", (*args, account_id))
        conn.commit()
    return public_view(conn.execute("SELECT * FROM field_accounts WHERE id=?", (account_id,)).fetchone())


def rotate_account_token(conn: sqlite3.Connection, account_id: str) -> str:
    token = new_account_token()
    cur = conn.execute("UPDATE field_accounts SET token_hash=? WHERE id=?", (_hash(token), account_id))
    conn.commit()
    if cur.rowcount == 0:
        raise AccountError("No such account.", 404)
    return token


# ------------------------------------------------------------------ team entitlements

def team_entitlement(conn: sqlite3.Connection, team_id: str) -> Dict[str, Any]:
    """What a team may do right now: plan, quotas, and whether writes are allowed.

    Teams made with the operator key (no account) are always active with the operator plan.
    An expired or suspended account leaves its teams readable but blocks writes, so nobody
    loses their data to a lapsed payment.
    """
    team = conn.execute("SELECT account_id FROM field_teams WHERE id=?", (team_id,)).fetchone()
    if not team or not team["account_id"]:
        return {"plan": "operator", "status": "active", "active": True, "reason": "",
                "limits": dict(PLAN_LIMITS["operator"]), "expires_at": None}
    acct = conn.execute("SELECT * FROM field_accounts WHERE id=?", (team["account_id"],)).fetchone()
    if not acct:
        return {"plan": "none", "status": "suspended", "active": False,
                "reason": "This team's account no longer exists.",
                "limits": dict(PLAN_LIMITS["team"]), "expires_at": None}
    s = status_of(acct)
    reasons = {
        "active": "",
        "expired": "This team's plan has expired. The team owner can contact Show Me Fire to renew.",
        "suspended": "This team's account is suspended. The team owner can contact Show Me Fire.",
    }
    return {
        "plan": acct["plan"],
        "status": s,
        "active": s == "active",
        "reason": reasons[s],
        "limits": resolve_limits(acct["plan"], acct["limits_json"]),
        "expires_at": acct["expires_at"],
    }


def require_writable(ent: Dict[str, Any]) -> None:
    if not ent["active"]:
        raise AccountError(ent["reason"] or "This team's plan is not active.", 402)


# ------------------------------------------------------------------ usage

def month_key(ts: Optional[int] = None) -> str:
    return time.strftime("%Y-%m", time.gmtime(ts if ts is not None else _now()))


def bump_usage(conn: sqlite3.Connection, team_id: str, ops: int = 0, positions: int = 0) -> None:
    conn.execute(
        "INSERT INTO field_usage (team_id, month, ops, positions) VALUES (?,?,?,?)"
        " ON CONFLICT(team_id, month) DO UPDATE SET ops = ops + excluded.ops,"
        " positions = positions + excluded.positions",
        (team_id, month_key(), ops, positions),
    )


def usage_report(conn: sqlite3.Connection, month: Optional[str] = None) -> Dict[str, Any]:
    month = month or month_key()
    rows = conn.execute(
        "SELECT t.id AS team_id, t.name AS team_name, t.account_id, a.name AS account_name,"
        " a.plan AS plan,"
        " (SELECT COUNT(*) FROM field_members m WHERE m.team_id=t.id AND m.revoked=0) AS members,"
        " (SELECT COUNT(*) FROM field_maps mp WHERE mp.team_id=t.id) AS maps,"
        " COALESCE(u.ops, 0) AS ops, COALESCE(u.positions, 0) AS positions"
        " FROM field_teams t LEFT JOIN field_accounts a ON a.id = t.account_id"
        " LEFT JOIN field_usage u ON u.team_id = t.id AND u.month = ? ORDER BY ops DESC, t.name",
        (month,),
    ).fetchall()
    return {"month": month, "teams": [dict(r) for r in rows]}

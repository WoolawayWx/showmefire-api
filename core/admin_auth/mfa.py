"""Second factors: authenticator-app codes (TOTP), one-time recovery codes and passkeys (WebAuthn).

A passkey is stored as a public key only. TOTP secrets must be readable to check a code, so they are encrypted at rest with
ADMIN_MFA_KEY (a Fernet key), or a key derived from JWT_SECRET when that is not set (rotating JWT_SECRET then invalidates
enrolled authenticator apps; passkeys, recovery codes and the break-glass CLI still work)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import secrets
import time
from datetime import timedelta
from typing import Optional

import pyotp
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException

from core import security

from . import db

ISSUER = "Show Me Fire"
RECOVERY_CODE_COUNT = 10
_RECOVERY_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
CHALLENGE_SECONDS = 300
_LOCAL_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://localhost:8000"]


# ----------------------------------------------------------------------------- state

def passkeys_for(admin_id: int) -> list[dict]:
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT id, name, transports, created_at, last_used_at FROM admin_passkeys WHERE admin_id = ? ORDER BY id", (admin_id,)
        ).fetchall()
    return [{"id": r["id"], "name": r["name"], "created_at": r["created_at"], "last_used_at": r["last_used_at"]} for r in rows]


def totp_enabled(admin_id: int) -> bool:
    with db.tx() as conn:
        return conn.execute("SELECT 1 FROM admin_totp WHERE admin_id = ? AND confirmed_at IS NOT NULL", (admin_id,)).fetchone() is not None


def recovery_remaining(admin_id: int) -> int:
    with db.tx() as conn:
        return conn.execute("SELECT COUNT(*) FROM admin_recovery_codes WHERE admin_id = ? AND used_at IS NULL", (admin_id,)).fetchone()[0]


def has_factor(admin_id: int) -> bool:
    """A passkey or a confirmed authenticator app. Recovery codes alone are a fallback, not a factor."""
    return bool(passkeys_for(admin_id)) or totp_enabled(admin_id)


def methods(admin_id: int) -> list[str]:
    found = []
    if passkeys_for(admin_id):
        found.append("passkey")
    if totp_enabled(admin_id):
        found.append("totp")
    if found and recovery_remaining(admin_id):
        found.append("recovery")
    return found


def status(admin_id: int) -> dict:
    return {
        "passkeys": passkeys_for(admin_id),
        "totp": totp_enabled(admin_id),
        "recovery_remaining": recovery_remaining(admin_id),
        "enforcement_active": enforcement_active(),
        "enforcement_configured": enforcement_configured(),
    }


def enforcement_configured() -> bool:
    return os.getenv("ADMIN_MFA_ENFORCE", "false").strip().lower() in {"1", "true", "yes"}


def enforcement_active() -> bool:
    """ADMIN_MFA_ENFORCE only bites once an active owner has a factor and saved recovery codes, so switching it on early can never lock
    the owner out. Until then password login keeps working."""
    if not enforcement_configured():
        return False
    with db.tx() as conn:
        owners = [r["id"] for r in conn.execute("SELECT id FROM admin_users WHERE role = 'owner' AND disabled_at IS NULL").fetchall()]
    return any(has_factor(owner) and recovery_remaining(owner) > 0 for owner in owners)


def clear_all(admin_id: int) -> None:
    with db.tx() as conn:
        for table in ("admin_passkeys", "admin_totp", "admin_recovery_codes"):
            conn.execute(f"DELETE FROM {table} WHERE admin_id = ?", (admin_id,))


# ----------------------------------------------------------------------------- tokens between login steps

def pending_token(admin_id: int) -> str:
    return security.create_access_token({"sub": str(admin_id), "type": "mfa_pending"}, timedelta(minutes=5))


def admin_id_from_pending(token: str) -> Optional[int]:
    return _id(security.verify_token(token, expected_type="mfa_pending"))


def enroll_token(admin_id: int) -> str:
    return security.create_access_token({"sub": str(admin_id), "type": "mfa_enroll"}, timedelta(minutes=15))


def admin_id_from_enroll(token: str) -> Optional[int]:
    return _id(security.verify_token(token, expected_type="mfa_enroll"))


def _id(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------- TOTP

def _fernet() -> Fernet:
    configured = os.getenv("ADMIN_MFA_KEY", "").strip()
    key = configured.encode() if configured else base64.urlsafe_b64encode(hashlib.sha256(f"admin-mfa:{security.SECRET_KEY}".encode()).digest())
    return Fernet(key)


def totp_begin(admin_id: int, account_name: str) -> dict:
    """Start (or restart) enrolment: a fresh secret held unconfirmed until a code from the app proves it works."""
    if totp_enabled(admin_id):
        raise HTTPException(status_code=409, detail="An authenticator app is already set up. Remove it first to set up a new one.")
    secret = pyotp.random_base32()
    with db.tx() as conn:
        conn.execute("DELETE FROM admin_totp WHERE admin_id = ?", (admin_id,))
        conn.execute("INSERT INTO admin_totp (admin_id, secret_enc) VALUES (?, ?)", (admin_id, _fernet().encrypt(secret.encode()).decode()))
    uri = pyotp.TOTP(secret).provisioning_uri(name=account_name, issuer_name=ISSUER)
    return {"secret": secret, "uri": uri, "qr_svg": qr_svg(uri)}


def qr_svg(text: str) -> str:
    import qrcode
    import qrcode.image.svg

    image = qrcode.make(text, image_factory=qrcode.image.svg.SvgPathImage, box_size=8)
    buffer = io.BytesIO()
    image.save(buffer)
    return buffer.getvalue().decode("utf-8")


def _totp_secret(admin_id: int, confirmed: bool) -> Optional[tuple[str, int]]:
    with db.tx() as conn:
        row = conn.execute("SELECT secret_enc, confirmed_at, last_step FROM admin_totp WHERE admin_id = ?", (admin_id,)).fetchone()
    if row is None or (row["confirmed_at"] is not None) != confirmed:
        return None
    try:
        return _fernet().decrypt(row["secret_enc"].encode()).decode(), row["last_step"]
    except InvalidToken:
        return None


def _now() -> float:
    return time.time()


def _match_step(secret: str, code: str, last_step: int) -> Optional[int]:
    code = (code or "").strip().replace(" ", "")
    if not (code.isdigit() and len(code) == 6):
        return None
    totp = pyotp.TOTP(secret)
    now_step = int(_now()) // 30
    for step in (now_step, now_step - 1, now_step + 1):
        if step > last_step and hmac.compare_digest(totp.at(step * 30), code):
            return step
    return None


def totp_confirm(admin_id: int, code: str) -> bool:
    found = _totp_secret(admin_id, confirmed=False)
    if not found:
        return False
    step = _match_step(found[0], code, found[1])
    if step is None:
        return False
    with db.tx() as conn:
        conn.execute("UPDATE admin_totp SET confirmed_at = ?, last_step = ? WHERE admin_id = ?", (time.time(), step, admin_id))
    return True


def totp_verify(admin_id: int, code: str) -> bool:
    """Check a code from the app. A code that was already accepted cannot be reused."""
    found = _totp_secret(admin_id, confirmed=True)
    if not found:
        return False
    step = _match_step(found[0], code, found[1])
    if step is None:
        return False
    with db.tx() as conn:
        conn.execute("UPDATE admin_totp SET last_step = ? WHERE admin_id = ?", (step, admin_id))
    return True


def totp_remove(admin_id: int) -> None:
    with db.tx() as conn:
        conn.execute("DELETE FROM admin_totp WHERE admin_id = ?", (admin_id,))


# ----------------------------------------------------------------------------- recovery codes

def _normalize_recovery(code: str) -> str:
    return "".join(ch for ch in (code or "").lower() if ch.isalnum())


def _recovery_hash(code: str) -> str:
    return hashlib.sha256(_normalize_recovery(code).encode()).hexdigest()


def recovery_regenerate(admin_id: int) -> list[str]:
    """Replace all recovery codes with a fresh set. The plain codes are returned here once and never stored."""
    codes = ["".join(secrets.choice(_RECOVERY_ALPHABET) for _ in range(10)) for _ in range(RECOVERY_CODE_COUNT)]
    with db.tx() as conn:
        conn.execute("DELETE FROM admin_recovery_codes WHERE admin_id = ?", (admin_id,))
        conn.executemany("INSERT INTO admin_recovery_codes (admin_id, code_hash) VALUES (?, ?)", [(admin_id, _recovery_hash(c)) for c in codes])
    return [f"{c[:5]}-{c[5:]}" for c in codes]


def recovery_use(admin_id: int, code: str) -> bool:
    if len(_normalize_recovery(code)) != 10:
        return False
    with db.tx() as conn:
        return conn.execute(
            "UPDATE admin_recovery_codes SET used_at = ? WHERE id = (SELECT id FROM admin_recovery_codes WHERE admin_id = ? AND code_hash = ? AND used_at IS NULL LIMIT 1)",
            (time.time(), admin_id, _recovery_hash(code)),
        ).rowcount == 1


# ----------------------------------------------------------------------------- passkeys (WebAuthn)

def webauthn_settings() -> tuple[str, list[str]]:
    rp_id = os.getenv("WEBAUTHN_RP_ID", "").strip()
    origins = [o.strip() for o in os.getenv("WEBAUTHN_ORIGINS", "").split(",") if o.strip()]
    if rp_id and not origins:
        # The production site and its sibling sites (www, preview, dev) all sit under the same domain, so one passkey works on each.
        origins = [f"https://{host}" for host in (rp_id, f"www.{rp_id}", f"preview.{rp_id}", f"dev.{rp_id}")]
    if rp_id and origins:
        return rp_id, origins
    if os.getenv("ENVIRONMENT", "development").lower() != "production":
        return rp_id or "localhost", origins or _LOCAL_ORIGINS
    raise HTTPException(status_code=503, detail="Passkeys are not configured on this server (set WEBAUTHN_RP_ID and WEBAUTHN_ORIGINS).")


def _save_challenge(kind: str, admin_id: Optional[int], challenge: bytes) -> str:
    challenge_id = secrets.token_urlsafe(16)
    now = time.time()
    with db.tx() as conn:
        conn.execute("DELETE FROM admin_challenges WHERE expires_at < ?", (now,))
        conn.execute(
            "INSERT INTO admin_challenges (id, kind, admin_id, challenge, expires_at) VALUES (?, ?, ?, ?, ?)",
            (challenge_id, kind, admin_id, challenge, now + CHALLENGE_SECONDS),
        )
    return challenge_id


def _take_challenge(challenge_id: str, kind: str, admin_id: Optional[int] = None) -> Optional[tuple[bytes, Optional[int]]]:
    """Single use: the challenge is deleted as it is read, so a captured response cannot be replayed."""
    with db.tx() as conn:
        row = conn.execute("SELECT * FROM admin_challenges WHERE id = ? AND kind = ?", (challenge_id, kind)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM admin_challenges WHERE id = ?", (challenge_id,))
    if row["expires_at"] < time.time():
        return None
    if admin_id is not None and row["admin_id"] != admin_id:
        return None
    return bytes(row["challenge"]), row["admin_id"]


def passkey_register_options(user: dict) -> dict:
    from webauthn import generate_registration_options, options_to_json
    from webauthn.helpers.structs import (
        AuthenticatorSelectionCriteria,
        PublicKeyCredentialDescriptor,
        ResidentKeyRequirement,
        UserVerificationRequirement,
    )

    rp_id, _ = webauthn_settings()
    with db.tx() as conn:
        existing = [bytes(r["credential_id"]) for r in conn.execute("SELECT credential_id FROM admin_passkeys WHERE admin_id = ?", (user["id"],)).fetchall()]
    options = generate_registration_options(
        rp_id=rp_id,
        rp_name=ISSUER,
        user_id=f"smf-admin-{user['id']}".encode(),
        user_name=user["email"],
        user_display_name=user["display_name"] or user["email"],
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.REQUIRED, user_verification=UserVerificationRequirement.REQUIRED
        ),
        exclude_credentials=[PublicKeyCredentialDescriptor(id=c) for c in existing],
    )
    return {"challenge_id": _save_challenge("register", user["id"], options.challenge), "options": json.loads(options_to_json(options))}


def passkey_register_verify(user: dict, challenge_id: str, credential: dict, name: str) -> dict:
    from webauthn import verify_registration_response

    rp_id, origins = webauthn_settings()
    taken = _take_challenge(challenge_id, "register", user["id"])
    if taken is None:
        raise HTTPException(status_code=400, detail="That passkey request expired. Start again.")
    try:
        verified = verify_registration_response(
            credential=credential, expected_challenge=taken[0], expected_rp_id=rp_id, expected_origin=origins, require_user_verification=True
        )
    except Exception as exc:  # the library raises several types for a bad response
        raise HTTPException(status_code=400, detail="That passkey could not be verified.") from exc
    transports = (credential.get("response") or {}).get("transports") or []
    with db.tx() as conn:
        try:
            conn.execute(
                "INSERT INTO admin_passkeys (admin_id, credential_id, public_key, sign_count, name, transports, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user["id"], verified.credential_id, verified.credential_public_key, verified.sign_count,
                 (name or "Passkey").strip()[:60] or "Passkey", json.dumps(transports), time.time()),
            )
        except Exception as exc:
            raise HTTPException(status_code=409, detail="That passkey is already registered.") from exc
    return {"passkeys": passkeys_for(user["id"])}


def passkey_remove(admin_id: int, passkey_id: int) -> bool:
    with db.tx() as conn:
        return conn.execute("DELETE FROM admin_passkeys WHERE id = ? AND admin_id = ?", (passkey_id, admin_id)).rowcount == 1


def passkey_auth_options(admin_id: Optional[int] = None) -> dict:
    """Options for signing in (any passkey, discoverable) or, with admin_id, for confirming as that admin."""
    from webauthn import generate_authentication_options, options_to_json
    from webauthn.helpers.structs import PublicKeyCredentialDescriptor, UserVerificationRequirement

    rp_id, _ = webauthn_settings()
    allow = None
    if admin_id is not None:
        with db.tx() as conn:
            allow = [PublicKeyCredentialDescriptor(id=bytes(r["credential_id"])) for r in
                     conn.execute("SELECT credential_id FROM admin_passkeys WHERE admin_id = ?", (admin_id,)).fetchall()]
    options = generate_authentication_options(rp_id=rp_id, allow_credentials=allow, user_verification=UserVerificationRequirement.REQUIRED)
    return {"challenge_id": _save_challenge("authenticate", admin_id, options.challenge), "options": json.loads(options_to_json(options))}


def passkey_auth_verify(challenge_id: str, credential: dict, expected_admin_id: Optional[int] = None) -> Optional[int]:
    """The admin id the passkey belongs to, if the assertion checks out; None otherwise."""
    from webauthn import verify_authentication_response
    from webauthn.helpers import base64url_to_bytes

    rp_id, origins = webauthn_settings()
    taken = _take_challenge(challenge_id, "authenticate", expected_admin_id)
    if taken is None:
        return None
    try:
        raw_id = base64url_to_bytes(credential.get("rawId") or credential.get("id") or "")
    except Exception:
        return None
    with db.tx() as conn:
        row = conn.execute("SELECT * FROM admin_passkeys WHERE credential_id = ?", (raw_id,)).fetchone()
    if row is None or (expected_admin_id is not None and row["admin_id"] != expected_admin_id):
        return None
    try:
        verified = verify_authentication_response(
            credential=credential, expected_challenge=taken[0], expected_rp_id=rp_id, expected_origin=origins,
            credential_public_key=bytes(row["public_key"]), credential_current_sign_count=row["sign_count"], require_user_verification=True,
        )
    except Exception:
        return None
    with db.tx() as conn:
        conn.execute("UPDATE admin_passkeys SET sign_count = ?, last_used_at = ? WHERE id = ?", (verified.new_sign_count, time.time(), row["id"]))
    return row["admin_id"]

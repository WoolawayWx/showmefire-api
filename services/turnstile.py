"""
Cloudflare Turnstile server-side verification for the public fire-report
endpoint. This is the only defense between an anonymous write endpoint and
a spam flood, so failures are handled fail-closed: any verification error
returns 503, never a silent pass, except in development when the secret
key is deliberately unset.
"""
import logging
import os
from typing import Tuple

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"


def verify_turnstile(token: str, remote_ip: str) -> Tuple[bool, str]:
    """
    Verify a Turnstile response token with Cloudflare.

    Returns (success, verdict) where verdict is a short diagnostic string
    stored in fire_events.captcha_verdict for abuse forensics - never
    returned to the client.
    """
    secret = os.getenv("TURNSTILE_SECRET_KEY", "").strip()
    if not secret:
        if IS_PRODUCTION:
            raise HTTPException(status_code=503, detail="Captcha is not configured")
        logger.warning("TURNSTILE_SECRET_KEY unset; captcha bypassed (development only)")
        return True, "bypassed-dev"

    payload = {"secret": secret, "response": token}
    if remote_ip and remote_ip != "unknown":
        payload["remoteip"] = remote_ip

    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.post(SITEVERIFY_URL, data=payload)
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Turnstile verification unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Captcha verification unavailable") from exc

    if body.get("success"):
        return True, "success"

    codes = ",".join(str(code) for code in (body.get("error-codes") or []))[:200]
    if any(code in ("invalid-input-secret", "bad-request") for code in (body.get("error-codes") or [])):
        logger.error("Turnstile misconfiguration: %s", codes or "unknown-error")
        raise HTTPException(status_code=503, detail="Captcha verification unavailable")
    return False, codes or "unknown-error"

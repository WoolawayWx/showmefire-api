"""Role checks for admin endpoints. `require_admin` (core.security) answers "is anyone signed in"; this answers "is the signed-in account
one of these roles", looking the role up live so a role change applies at once.

Not used anywhere yet: every role currently has the same access. To restrict an endpoint later:

    @router.post("/something", dependencies=[Depends(require_role("owner", "admin"))])
"""
from __future__ import annotations

from typing import Callable, Optional

from fastapi import Depends, HTTPException

from core import security

from . import users


def _session_email(token: Optional[str] = None) -> Optional[str]:
    return security.verify_token(token)


def require_role(*allowed: str) -> Callable[..., dict]:
    unknown = set(allowed) - set(users.ROLES)
    if unknown:
        raise ValueError(f"Unknown role(s): {', '.join(sorted(unknown))}")

    def dependency(email: Optional[str] = Depends(_session_email)) -> dict:
        user = users.get_by_email(email) if email else None
        if user is None or user["disabled_at"]:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if user["role"] not in allowed:
            raise HTTPException(status_code=403, detail="Your account does not have access to that.")
        return user

    return dependency

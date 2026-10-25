"""Break-glass tool for admin accounts. Run it on the server (or in the API container) with access to the database; it needs no
login, so it is the way back in after a lost password, phone or recovery codes.

    python scripts/admin_user.py list
    python scripts/admin_user.py create-owner you@example.org
    python scripts/admin_user.py set-password you@example.org
    python scripts/admin_user.py reset-mfa you@example.org        # clears passkeys, authenticator app and recovery codes
    python scripts/admin_user.py revoke-sessions you@example.org  # signs that account out everywhere
    python scripts/admin_user.py revoke-sessions --all
    python scripts/admin_user.py enable you@example.org           # undo a disable

Passwords are prompted for, never passed as arguments. Honors DATA_DIR like the API does.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import security  # noqa: E402
from core.admin_auth import events, mfa, sessions, users  # noqa: E402


def _prompt_password() -> str:
    while True:
        first = getpass.getpass("New password: ")
        problem = security.password_strength_error(first)
        if problem:
            print(problem)
            continue
        if getpass.getpass("Repeat it: ") != first:
            print("They did not match.")
            continue
        return first


def _find(email: str) -> dict:
    user = users.get_by_email(email)
    if user is None:
        sys.exit(f"No account for {email}.")
    return user


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    for name in ("create-owner", "set-password", "reset-mfa", "enable"):
        commands.add_parser(name).add_argument("email")
    revoke = commands.add_parser("revoke-sessions")
    revoke.add_argument("email", nargs="?")
    revoke.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        for user in users.list_users():
            factors = f"{len(mfa.passkeys_for(user['id']))} passkey(s), authenticator {'on' if mfa.totp_enabled(user['id']) else 'off'}"
            print(f"{user['id']:>3}  {user['email']:<36} {user['role']:<6} {'DISABLED ' if user['disabled_at'] else ''}{factors}")
    elif args.command == "create-owner":
        if users.get_by_email(args.email):
            sys.exit("That account already exists; use set-password.")
        created = users.create_user(args.email, "", "owner", _prompt_password(), must_change=False)
        events.record("cli_create_owner", created["id"], "cli")
        print(f"Created owner {created['email']}.")
    elif args.command == "set-password":
        user = _find(args.email)
        users.set_password(user["id"], _prompt_password())
        sessions.revoke_all(user["id"])
        events.record("cli_set_password", user["id"], "cli")
        print("Password changed and every session ended.")
    elif args.command == "reset-mfa":
        user = _find(args.email)
        mfa.clear_all(user["id"])
        events.record("cli_reset_mfa", user["id"], "cli")
        print("Passkeys, authenticator app and recovery codes cleared. They can sign in with the password alone.")
    elif args.command == "enable":
        user = _find(args.email)
        users.set_disabled(user["id"], False)
        print("Account enabled.")
    elif args.command == "revoke-sessions":
        if args.all:
            total = sum(sessions.revoke_all(u["id"]) for u in users.list_users())
            print(f"Ended {total} session(s).")
        elif args.email:
            print(f"Ended {sessions.revoke_all(_find(args.email)['id'])} session(s).")
        else:
            sys.exit("Give an email or --all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

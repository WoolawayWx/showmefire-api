# ShowMeFire Field team service (accounts, plans, collaboration)

Backs the Team features of the ShowMeFire Field desktop and web apps: shared maps that a small
team edits live, and optional sharing of members' GPS positions. Code: `routers/field.py` (HTTP),
`core/field_store.py` (storage), `core/field_accounts.py` (accounts, plans, quotas, usage).
Tests: `tests/test_field_collab.py`, `tests/test_field_accounts.py`.

Licence: the Field files are **not** under this repository's MIT licence; see `LICENSE-FIELD.md`.

## What is free and what is not

The Field app itself is free. Everything local (maps, drawing, GPS, printing, offline, public Show
Me Fire data) never calls this service. Only the **team features** are served here and need an
account:

* A **team owner** needs an account on a paid plan, granted by a Show Me Fire administrator.
* **Teammates** join the owner's team with a join code and share the owner's plan quotas.
* If a plan lapses (expires or is suspended) the team becomes **read-only**: members can still read
  maps and see positions, but cannot edit, add maps, share positions, or admit new members. Renewing
  restores everything, so nothing is lost to a late payment.

## Plans

Defined in `core/field_accounts.py` (`PLAN_LIMITS`); an account may carry per-account overrides.

| Plan | Teams | Members per team | Maps | Features per map |
|---|---|---|---|---|
| `team` | 1 | 10 | 20 | 5,000 |
| `agency` | 5 | 50 | 50 | 20,000 |

Absolute ceilings (`HARD_CEILINGS`) apply to any override. Teams created directly with the
operator key (below) use the `operator` plan and have no account.

## Granting access (administrators)

Admin routes use the same admin JWT as the rest of the API (`?token=` or the admin cookie).

| Call | Purpose |
|---|---|
| `POST /api/field/v1/admin/accounts` | body `{name, contact?, plan, expires_at?, notes?, limits?}`; the reply contains the account **token (`fldacct_...`) shown once**; give it to the customer |
| `GET /api/field/v1/admin/accounts` | list accounts (never includes tokens) |
| `PATCH /api/field/v1/admin/accounts/{id}` | change `plan`, `status` (`active`/`suspended`), `expires_at`, `limits`, `name`, `contact`, `notes`; this is also where a payment processor would extend `expires_at` |
| `POST /api/field/v1/admin/accounts/{id}/token` | issue a new token, invalidating the old one |
| `GET /api/field/v1/admin/usage?month=YYYY-MM` | per-team operations and position reports, members and maps, for judging server cost |

`expires_at` is unix seconds (UTC); omit it for no expiry. Tokens are stored only as SHA-256.

## Using an account (customers)

* `GET /api/field/v1/account` with `Authorization: Bearer fldacct_...` shows the plan, status,
  limits and the account's teams with their member and map counts.
* `POST /api/field/v1/teams` with the same header creates a team (counts against `max_teams`).

## Operator key (optional)

`FIELD_TEAM_CREATE_KEY` lets an operator create a team without an account via the
`X-Field-Create-Key` header. Leave it unset to require an account for every team.

## Model

* **Member token**: `Authorization: Bearer fld_...`, returned when creating or joining a team
  (hashed in the database). Join codes are 10 characters without look-alike letters, stored hashed;
  the owner can rotate the code and remove members.
* **Shared map**: features keyed by a client-chosen `uid` (8-64 chars: letters, digits, `-`, `_`).
  Every change is an operation (`upsert`/`delete`) taking the next integer `rev`, so all clients apply
  changes in the same order (last writer wins per feature). Deleted features keep a tombstone; the
  operation log keeps the latest 5000 operations per map, and a client that falls behind gets
  `reset: true` and reloads the whole map.
* **Positions**: the newest report per member, shared only within the team, hidden after 24 hours;
  a member can stop sharing, and removing a member clears their marker.

## Team endpoints (prefix `/api/field/v1`)

| Method and path | Who | Notes |
|---|---|---|
| `POST /teams` | account token (or operator key) | body `{team_name, display_name}`; returns `{team_id, join_code, member_id, token}` |
| `POST /teams/join` | anyone with a join code | body `{join_code, display_name}` |
| `GET /me` | member | profile, team, members, and the team's `plan` with limits and usage |
| `POST /teams/join-code` | owner | rotates the join code |
| `DELETE /members/{id}` | owner | revokes a member's token |
| `GET /maps`, `POST /maps` | member | create with `{name, features: {uid: feature}}` |
| `GET /maps/{id}`, `DELETE /maps/{id}` | member / owner | |
| `POST /maps/{id}/ops` | member | up to 500 operations, atomic; returns `{rev}` |
| `POST /positions`, `DELETE /positions` | member | at most one report per second |
| `GET /events?map_id&since&pos_since&wait` | member | long poll (max 25 s): returns as soon as there is news |

Another team's map answers `404`, exactly like a missing one. Writes on a lapsed plan answer `402`
with a message the app shows to the user.

## Limits and abuse protection

Per-plan quotas above, plus global ceilings: feature size 64 KB, request size 2 MB, 500 operations
per request. `POST /teams` and `POST /teams/join` allow 10 attempts per client address per 10
minutes (in-memory, per process; the address comes from `cf-connecting-ip` when present).

## Deployment notes

* Set no extra env var to run it; set `FIELD_TEAM_CREATE_KEY` only if you want operator-made teams.
  Tables are created at startup (`ensure_field_schema()` in the lifespan handler) and on first use;
  existing databases are migrated in place.
* **Long polling**: the reverse proxy must allow responses to take up to ~25 s (nginx:
  `proxy_read_timeout 60s` or more for `/api/field/`). The poll loop checks the database every
  half second, so it works with several server processes.
* **Browsers**: the web app calls this API from `https://app.showmefire.org` (or wherever it is
  hosted); that origin must be in the CORS list in `main.py`.
* **Secrets in the database**: tokens and join codes are hashed; display names, map content,
  positions and account contact details are not. Treat the database file accordingly.
* **Payments**: none are processed here. Access is granted by an administrator; a payment
  processor can later call the same `PATCH` logic (`field_accounts.update_account`) to extend
  `expires_at` when a payment succeeds.

## Tests

```
python -m pytest tests/test_field_collab.py tests/test_field_accounts.py
```

The Field app repo has an end-to-end test (two clients against a running server) in
`crates/smf-sync/tests/e2e.rs`.

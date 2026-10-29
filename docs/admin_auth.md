# Admin accounts and sign-in

Admins have their own accounts (table `admin_users`). Sign-in is a password, then, once the account has one, a second step: a
passkey, an authenticator-app code, or a recovery code. A passkey can also sign in by itself. Sessions are stored server-side and can
be ended from **Admin → Account Security → Devices**.

## Roles
Accounts have a role: `owner`, `admin`, `staff` or `partner`. All roles sign in the same way (same password, passkey and code
options) and **currently have the same access to the admin area**; only owners can manage accounts. There is no signup: an owner creates
accounts on **Admin Users**, which issues a one-time temporary password. The role comes back from sign-in, `/api/admin/me` and
`/api/admin/verify`. To restrict an endpoint later, add `dependencies=[Depends(require_role("owner", "admin"))]`
(`core/admin_auth/access.py`); it checks the live role, so a role change applies immediately. Note that until endpoints are restricted,
a `partner` account can reach everything an `admin` can.

## First start / migration
On first use the existing `ADMIN_EMAIL` + `ADMIN_PASSWORD_HASH` are copied into an `owner` account, so the same email and password keep
working. After that the database is the source of truth: **changing `ADMIN_PASSWORD_HASH` in `.env` no longer changes the password**.
Change it on the Security page, or with the command-line tool below. The two variables stay required in production (the startup check
still looks for them) but are only a seed.

## Settings (environment)
| Variable | Default | Meaning |
|---|---|---|
| `WEBAUTHN_RP_ID` | `localhost` outside production, **required in production** | Domain passkeys are bound to, e.g. `showmefire.org` (covers `api.` and the site). Changing it invalidates every enrolled passkey. |
| `WEBAUTHN_ORIGINS` | with `WEBAUTHN_RP_ID` set: `https://<rp id>`, plus `www.`, `preview.` and `dev.` of it | Comma list of origins allowed to use passkeys: the site's `https://` addresses and, for the Android app, `android:apk-key-hash:<…>` entries (see "Passkeys in the mobile app"). **Setting it REPLACES the default list, it does not add to it**, so list every website origin you still want (an explicit `https://showmefire.org` alone stops passkeys working on `dev.` and `www.`). |
| `ADMIN_MFA_ENFORCE` | `false` | Require the second step for everyone. Only takes effect once an active owner has a passkey or authenticator app **and** unused recovery codes, so switching it on early cannot lock the owner out. |
| `ADMIN_ALLOW_LEGACY_TOKENS` | `true` | Accept access tokens issued before sessions existed (no session id). Set `false` once everyone has signed in again. |
| `ADMIN_MFA_KEY` | derived from `JWT_SECRET` | Fernet key that encrypts authenticator-app secrets. If unset, rotating `JWT_SECRET` invalidates enrolled authenticator apps (passkeys and recovery codes are unaffected). |
| `ADMIN_LOGIN_MAX_PW` / `_TOTP` / `_RECOVERY` / `_IP` | 8 / 6 / 10 / 80 | Failed attempts per 15 minutes before that step locks. A lock only covers the factor being guessed (never the passkey path) and clears itself. `_IP` is a wide brake; behind a proxy that hides client addresses all visitors share it. |
| `ADMIN_SESSION_IDLE_DAYS` | `14` | A session unused this long cannot be renewed. Sessions also end after `REFRESH_TOKEN_EXPIRE_DAYS` (30). |

## Dev and preview sites
`dev.showmefire.org` and `preview.showmefire.org` call the same API (same accounts, same database) and are on the same domain as
production, so with `WEBAUTHN_RP_ID=showmefire.org` **one passkey works on all three sites**, and the session cookies work there as
they already did. Nothing extra to configure on the API.

What does not work: `*.pages.dev` preview deployments and `dev.woolawaywx.com` are different domains. Cookies from the API are not
sent to the pages.dev hosts at all (this was already true), and a passkey made for `showmefire.org` cannot be used on another
domain. Use the password and an authenticator app or recovery code there if the cookies allow it.

**Order matters:** the new sign-in page (website `dev` branch) must be promoted to `preview` and `master` *before* any account turns on a
passkey or authenticator app. An older sign-in page cannot do the second step and would appear to do nothing after the password.
Until then password sign-in works on every site.

## Getting back in
Run on the server (or inside the API container); it needs database access, not a login:

    python scripts/admin_user.py list
    python scripts/admin_user.py set-password you@example.org     # prompts; ends all sessions
    python scripts/admin_user.py reset-mfa you@example.org        # clears passkeys, authenticator app, recovery codes
    python scripts/admin_user.py revoke-sessions --all
    python scripts/admin_user.py create-owner you@example.org

Setting `ADMIN_MFA_ENFORCE=false` and restarting also drops back to password-only sign-in.

## Rolling out
1. Deploy the API and the website together (the new sign-in page understands the second step; an old one would not).
2. Sign in, open Account Security, add a passkey and/or authenticator app, and save the recovery codes.
3. Sign out and back in to test each method.
4. When ready: set `ADMIN_MFA_ENFORCE=true` and later `ADMIN_ALLOW_LEGACY_TOKENS=false`.

## Passkeys in the mobile app
The app (`mobile/`, staff section) signs in with a passkey through `POST /api/admin/mobile/login/passkey/options` and `/verify`
(`routers/admin_mobile.py`, off unless `STAFF_API_ENABLED`). It is sign-in only: passkeys are still created on the website's Account
Security page, and a passkey synced to the phone (iCloud Keychain / Google Password Manager) works in the app because the relying
party is the same domain. A passkey replaces the password and the second step, as on the website. The mobile endpoint also applies
the rules a password sign-in does (account enabled and not expired, reviewer accounts need `SANDBOX_ENABLED`, a temporary password
must be changed on the website first); the website's passkey sign-in now refuses expired and reviewer accounts too.

What the server must accept:
- **iOS:** the origin is `https://<WEBAUTHN_RP_ID>` (already allowed). The app needs `webcredentials:showmefire.org` in its entitlements and
  the website must serve `https://showmefire.org/.well-known/apple-app-site-association` (`webcredentials.apps` listing the app id). Apple's
  CDN caches that file, so deploy it well before testing.
- **Android:** the origin is `android:apk-key-hash:<base64url of the SHA-256 of the certificate that signed the installed build>`, which
  is NOT an https address. List one for every key a build may carry: the Play App Signing key (store installs), the upload key (EAS
  internal builds) and any debug/preview key. Turn the fingerprints (as shown in Play Console, colon-separated hex) into the setting with
  `python scripts/android_origin.py --line "<fp>" "<fp>"`, which prints the whole `WEBAUTHN_ORIGINS` line including the website origins.
  The same fingerprints go into the website's `NUXT_ANDROID_ASSET_LINKS_SHA256` (`/.well-known/assetlinks.json`).
- A wrong Android hash fails as "That passkey could not be verified"; the reason (and the origins the server expected) is in the API log:
  `docker logs show-me-fire-api 2>&1 | grep "Passkey"`.
- Recreate the container after changing the setting (a plain restart does not re-read the env file).

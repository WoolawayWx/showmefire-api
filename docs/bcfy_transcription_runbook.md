# Broadcastify wildland-fire transcription

The API can poll one Broadcastify Calls channel, download call audio, run a
local `faster-whisper` model, and create an admin-reviewable fire signal when
the transcript contains likely wildland/brush-fire language.

## Runtime configuration

The credentials are environment-only:

- `BCFY_KEYID` — the API Key ID from your Broadcastify developer app (used as the JWT `kid` header)
- `BCFY_KEYSECRET` — the API Key itself (used as the HMAC-SHA256 signing secret, never sent over the wire)
- `BCFY_ISSUER` — your Application ID from bcfy.io/dev/profile/ (used as the JWT `iss` claim)

Authentication is a self-signed JWT per Broadcastify's Calls Client API
contract — there is no OAuth token-exchange call. `services/transcription.py`
mints a short-lived (1 hour) HS256 JWT locally for every request.

- `BCFY_CALLS_URL` (defaults to `https://api.bcfy.io/calls/v1/live/`, the Live Calls endpoint)
- `BCFY_API_BASE_URL` (defaults to `https://api.bcfy.io`)
- `BCFY_WHISPER_DEVICE` (`auto` by default)
- `BCFY_WHISPER_COMPUTE_TYPE` (`int8` by default)

Polling uses the Live Calls endpoint's `pos` cursor (stored as `last_pos` in
`bcfy_transcription_config`), not a time window — the Live Calls endpoint is
documented as the correct one for near-real-time polling; the separate Group
Archives endpoint is for backfill/history only and lags live by ~15 minutes.

The admin page stores the initial department (Central Crossing Fire Protection
District - Shell Knob), source feed `30217`, Calls group `6847-24421`, polling
interval, model, threshold, and audio-retention period in SQLite. It never stores or displays
the credentials.

## Lifecycle

1. The scheduler polls only when the admin configuration is enabled.
2. External call IDs are deduplicated in `bcfy_calls`.
3. Audio is stored under `BCFY_DATA_DIR` and processed outside the event loop.
4. A transcript is classified as `wildland_fire` or `other`.
5. Accepted calls create a pending row in `bcfy_fire_signals`.
6. An administrator confirms or rejects the signal. Confirmation does not
   create or publish a public fire event in this first release.

The API requeues interrupted jobs after restart and removes raw audio older
than the configured retention period. Do not expose the BCFY secret in logs or
commit the runtime `.env` file. Rotate credentials if the file has been shared
or committed.

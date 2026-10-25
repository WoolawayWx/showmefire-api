# Fire report moderation runbook

`POST /api/fires/reports` is the first unauthenticated write path in this
API, the first rate limiter, the first append-only audit trail
(`fire_event_moderation`), and the first PII purge job outside mobile push
delivery records. This runbook covers the operational parts the code can't
enforce on its own.

## Pre-launch checklist

- [ ] `TURNSTILE_SECRET_KEY` is set in the API environment. Production
  refuses to boot without it (`api/main.py` lifespan check).
- [ ] `NUXT_PUBLIC_TURNSTILE_SITE_KEY` is set in the website environment and
  matches the same Turnstile widget as the secret key above.
- [ ] `TRUST_PROXY_HEADERS=true` in production **only if** the origin is
  reachable exclusively through Cloudflare (Tunnel or an IP allowlist). If
  the origin is directly reachable, `CF-Connecting-IP` is attacker-supplied
  and the rate limiter becomes cosmetic.
- [ ] `FIRE_REPORT_IP_SALT` is set to a value distinct from `JWT_SECRET`
  (it falls back to `JWT_SECRET` if unset, which works but mixes purposes).
- [ ] `website/nuxt.config.ts` CSP includes `challenges.cloudflare.com` in
  `script-src`, `connect-src`, and `frame-src`.
- [ ] The privacy policy and `PRIVACY_OPERATIONS.md` fire-report sections
  are live before the public form is.

## Daily moderation loop

Everything is reviewed in one place: **Fire Review** (`/admin/fires/workspace`, in the admin menu under Content;
the old `/admin/fires` and `/admin/fires/incidents/{id}` addresses redirect there). The list on the left has
**Needs review** (incidents with pending detections or feedback), **Single reports** (pending reports and
detections not in any incident yet, with a pending / approved / rejected switch), and the All, Has feedback and
Confirmed views. Pick an incident to approve, reject, delete or reanalyze it, review its public feedback
and see every detection with the feed it came in through. Pick a single report to approve or reject it.
The **Label** control next to every Approve button is the fire / controlled burn label described below; it clears
after each approval. Fixed non-fire heat sources (mills, flares) are on **Non-fire sources**, and the full report
page (`/admin/fires/{id}`: media, edits, moderation history, where it came from) and the area review page
(`/admin/fires/area/{id}`: same fire / different fire) open from the workspace.

For each report:

1. Check the "N other reports within 2 km in the last 6 hours" hint on the
   detail page (`GET /api/admin/fires/reports/{id}`, `nearby_reports`).
   Reject obvious duplicates with a reason citing the original report's id.
2. Judge plausibility: does the description and acreage make sense for the
   location and time of year? An implausible report should be rejected, not
   silently approved at a low tier.
3. Approve with a tier:
   - **`admin_reviewed`** - your default for a report you believe is real
     but have not cross-checked against an outside source. This tier is an
     auxiliary training signal, never primary.
   - **`official_source_confirmed`** - only when you can cite a specific
     external record (NFIRS entry, MDC incident number, fire department
     report, news report with an official quote). The `official_source_ref`
     field is required by the schema for this tier - use it to record the
     citation, not a restatement of the report.
4. Optionally **label** the approval: send `"label": "fire"` (an unplanned wildfire) or
   `"label": "controlled_burn"` (a planned burn) with the approval. Approving means it is a real fire of
   some kind; the label says which kind. They set `cause_category` to `wildfire` / `prescribed` (the same
   categories an approved public classification writes), bump `label_revision`, and appear in the audit row
   (`changed_fields_json`). Leave it out when you do not know: the report is approved with no label, which is
   better than a guess in the training labels. Rejecting never labels.
5. Reject with a specific, one-line reason. Rejected reports are retained
   (soft state, not deleted) so the audit trail explains why.

`POST /api/admin/fires/incidents/{id}/approve-all` takes the same `label`. It labels every member approved in
that action and is stored on the incident (`review_label`), so a detection that joins the incident later is
approved and labelled the same way (audit reason "inherited the label from the incident's approval"). An
explicit label on the incident wins over a label carried forward from public feedback. Approving the incident
again without a label clears the standing label. Members you already moderated individually are left alone.

**What counts as an official source:** a record from a fire department,
MDC, MO DNR, USFS Mark Twain National Forest, or a NFIRS-linked incident
number. A news article alone is not sufficient for `official_source_confirmed`
unless it quotes an official incident number or department statement.

## OSINT/social-monitoring ingested reports (`source='muse'` etc.)

`POST /fires/ingest/reports` is a Bearer-API-key-authenticated write path for
automated social-media monitoring tools (e.g. Muse scanning department
Facebook pages) to submit fire leads without a human re-typing them. These
rows land in the exact same pending queue as public submissions and use the
same approve/reject UI - there is no separate review surface.

What's different about them:

- **`source` is the tool's slug** (e.g. `muse`), not `user_submission`. Use
  this to filter the queue (`GET /api/admin/fires/reports?status=pending`
  and check `source`) if you want to triage OSINT leads separately.
- **Coordinates are frequently geocoded from a county/city string**, not a
  precise pin - the submitting tool rarely has exact lat/lng from a social
  post. Treat the pin as approximate and correct it via
  `PUT /api/admin/fires/events/{id}` if you can pin down a better location.
- **The original post link is folded into `description`** (a trailing
  `Source: <url>` line) rather than stored in `official_source_ref` - that
  field is reserved for an official citation you add yourself and is
  cleared to empty on a normal `admin_reviewed` approval, so it would lose
  the Facebook link if reused for it.
- **Default to `admin_reviewed`**, same as any other unverified report.
  Only use `official_source_confirmed` if you can independently verify the
  post against NFIRS, MDC, an official department statement, or similar -
  the social post itself is never sufficient grounds for that tier.
- Ingest sources and their API keys are managed via
  `POST /api/admin/fires/ingest-sources` (create, returns the raw key once)
  and `POST /api/admin/fires/ingest-sources/{id}/revoke`.

## Where everything came from (source log)

Every event that arrives gets one row in `fire_event_origins`, written the first time it is seen and kept
(later arrivals of the same event only update `last_received_at` and `receive_count`, so the original source
is never overwritten). It is returned as `origin` on `GET /api/admin/fires/reports/{id}`, and `origin_feed` is
on the incident detail and pending-queue members. The first audit row (`ingested` / `submitted`) also says
"ingested from <feed>" and carries the same facts.

| `kind` | `feed` | what else is recorded |
|---|---|---|
| `satellite_feed` | NASA FIRMS (VIIRS/MODIS), NOAA NESDIS NGFS | the file it was read from (`channel`), the provider's own id (`provider_ref`), the ingest run, the provider's raw record (`raw`) |
| `official_import` | USFS FPA-FOD | the service, the FPA-FOD id, the import run, the raw record |
| `osint_api` | the registered integration (e.g. Muse) | `POST /fires/ingest/reports`, the post link, the page it was posted on, which API key and ingest source sent it, the lead as received |
| `public_form` | Public fire report form | `POST /api/fires/reports`, browser/app string, the site that sent it, and country/region when the proxy headers are trusted (never an IP address; the IP hash stays on the event) |

Batch ingests also write a row to `fire_ingest_runs` per feed file or import: the file name and its
modified time, how many records it held, how many were inserted, updated and skipped, **why** each skipped
record was skipped (`outside_missouri`, `no_coordinates`, `no_acquisition_time`, ...) and any errors.
`GET /api/admin/fires/origins?days=7&runs=20` summarises both: events and arrivals per kind and feed, newest
arrival, and the latest runs. A feed that has gone quiet shows up as an old `last_received_at`; a feed file
that stopped refreshing shows up as an old `input_modified_at`.

Events that arrived before this log existed have no origin until they are seen again.

## Abuse-spike playbook

If the pending queue fills with obvious spam or the throttle table shows
concentrated hits:

1. Read `fire_submission_throttle` for the offending window to confirm the
   pattern (single bucket vs. spread across many - the latter suggests a
   botnet and the global cap, not the per-IP cap, is what will stop it).
2. Add the offending `submitter_ip_hash` values to the blocklist via
   `POST /api/admin/fires/blocklist` (`{ip_hash, reason}`). You only have
   the hash, not the raw IP - that is intentional.
3. If the spike is broad, temporarily lower `FIRE_REPORT_GLOBAL_LIMIT_PER_DAY`
   or set `FIRE_REPORT_LIMIT_PER_HOUR=0` as a full kill switch on new
   submissions while you triage. Both are environment variables - restart
   picks them up.
4. Reject the spam batch with reason `"spam"` rather than deleting - keep
   the audit trail.

## Turnstile outage behavior

Turnstile verification fails **closed**: a Cloudflare outage or network
timeout returns `503` to the submitter, and no row is written. This is
deliberate - Turnstile is the only defense against a spam flood, so a
fail-open branch would defeat it exactly when it matters most. **Do not**
patch around a `503` spike by bypassing verification; wait for Cloudflare,
or as a last resort unset `TURNSTILE_SECRET_KEY` in a non-production
environment only, never in production.

## PII purge verification

The `purge_fire_report_pii` job runs daily at 02:45 Central and:

1. Auto-rejects reports still `pending` after `FIRE_REPORT_MAX_AGE_DAYS`
   equivalent (30 days) with reason `expired-unmoderated`.
2. Clears `reporter_contact` and `submitter_ip_hash` on any report
   `moderated_at` more than 90 days ago, stamping `pii_purged_at`.
3. Deletes throttle rows older than 48 hours.

To verify it ran, check counts only - never dump the PII columns to confirm
they're empty:

```sql
SELECT COUNT(*) FROM fire_events
WHERE moderated_at <= datetime('now', '-90 days')
  AND (reporter_contact != '' OR submitter_ip_hash != '');
-- should be 0 after the job runs
```

If you need to redact PII from a report body (a reporter volunteered a
neighbor's name in the description), use `PUT /api/admin/fires/events/{id}`
with a populated `edit_reason` - never edit the database directly. The
`edit_reason` is what makes the audit trail meaningful.

## Label export and manifest interpretation

`python scripts/export_fire_labels.py --min-tier official_source_confirmed`
writes `data/fire_labels/fire_labels_<date>.csv` and
`fire_labels_manifest.json`. The manifest's `rows_by_tier` and
`csv_sha256` are what a training run cites to prove which label population
it was fit on - if you're asked to explain a model's data lineage, this
manifest is the artifact. Bump the export cadence to align with the
`end_of_day_archive` job when the training team is running the next batch.

## The deliberate audit orphan on delete

`DELETE /api/admin/fires/events/{id}` is a soft delete (`status='deleted'`).
`fire_event_moderation` rows for that event are **retained**, not deleted -
this is intentional, since the moderation table is the audit trail and an
event's history should survive its own deletion. There is no FK enforcement
in this SQLite database, so nothing breaks; this is a design choice, not an
oversight.

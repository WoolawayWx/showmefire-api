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

Open `/admin/fires` and work the pending queue (`GET /api/admin/fires/reports?status=pending`).
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
4. Reject with a specific, one-line reason. Rejected reports are retained
   (soft state, not deleted) so the audit trail explains why.

**What counts as an official source:** a record from a fire department,
MDC, MO DNR, USFS Mark Twain National Forest, or a NFIRS-linked incident
number. A news article alone is not sufficient for `official_source_confirmed`
unless it quotes an official incident number or department statement.

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

# Contact tickets

The website contact page posts to `POST /v1/contact`. The API stores the ticket and two email outbox entries together in the existing `showmefire.db`. Ticket IDs look like `SMF-20261119-A1B2C3D4E5F6`.

## Configuration and release

Deploy the API before the website. New tables are created automatically. Keep the API scheduler enabled (`run_sch=true`, with the `api` job group) so `maintain_contact_tickets` runs every five minutes.

Email uses the existing Resend account. Set `RESEND_API_KEY` and optionally `CONTACT_EMAIL_FROM` to a sender on a verified domain. The sender falls back to `API_KEY_EMAIL_FROM`, `GRAPHICS_EMAIL_FROM`, then `BULLETIN_EMAIL_FROM`. The staff recipient and the confirmation reply address are always `contact@showmefire.org`; replies to the staff notification go to the submitter. New submissions return 503 before creating a ticket when email configuration is missing.

Use the existing `TURNSTILE_SECRET_KEY` on the API and matching `NUXT_PUBLIC_TURNSTILE_SITE_KEY` on the website. Turnstile verification, a honeypot, a per-minute limiter, and an atomic five-ticket-per-IP daily limit protect submissions. Only a keyed hash of the IP is stored.

The client generates a UUID `request_id` and keeps it for retries of unchanged content during the current page session. Repeating it with the same normalized fields returns the original ticket, including after a consumed CAPTCHA token or a lost response. Changed content with that UUID returns 409. There is no anonymous ticket listing or lookup endpoint.

## Delivery and recovery

Each recipient gets a separate email containing the same ticket ID. A successful API response means the ticket is saved. `email_status: sent` means Resend accepted both messages, not that they reached the inbox; `pending` means at least one is still queued. Email failure never deletes a ticket or reports that it was not submitted.

Outbox entries contain the exact email payload and delivery timestamps. Each email has an atomic two-minute delivery lease and a stable Resend idempotency key. Only unsent emails retry; ordinary retries run every five minutes. Automatic retries stop 23 hours after the first attempt, within [Resend's 24-hour idempotency window](https://resend.com/changelog/idempotency-keys). Unsent entries remain stored and the scheduler logs their count for staff review. After fixing delivery configuration, check the receiving mailboxes or Resend delivery records before manually recovering an expired entry to avoid duplicate email. A bounce after provider acceptance requires mailbox/provider follow-up.

Inspect counts without exposing messages:

```sql
SELECT kind, COUNT(*) FROM contact_email_outbox WHERE sent_at IS NULL GROUP BY kind;
SELECT COUNT(*) FROM contact_tickets WHERE resolved_at < unixepoch() - 90 * 86400;
```

Delivery warnings include a fixed `reason` and `http_status`, without logging addresses, API keys, messages, or raw provider errors. For `sender_domain_unverified`, verify the exact sending domain in Resend. For `sender_in_testing_mode`, configure a verified sender. For 401/403, also check API key validity and sender-domain permissions; for 429, check rate limits and quotas. `provider_timeout` and `provider_connection_error` indicate network failures. Other provider errors can be inspected in the Resend dashboard logs.

The sender is saved inside each outbox payload when the ticket is created. Changing `CONTACT_EMAIL_FROM` changes new tickets only. Verifying the queued sender's domain allows the existing payload to retry unchanged. If a queued payload must be corrected, first confirm from provider logs that it was rejected and never accepted; changing a previously accepted payload under the same idempotency key can cause a 409 conflict. Do not reset timestamps or idempotency keys blindly.

## Resolution and retention

Ticket correspondence happens by email; keep the ticket ID in reply subjects. This change does not add an admin ticket inbox. Mark a ticket resolved using a parameterized statement from a trusted database maintenance session:

```sql
UPDATE contact_tickets SET resolved_at = unixepoch() WHERE ticket_id = ?;
```

The scheduled maintenance job removes tickets and their outbox copies 90 days after resolution. Staff must also remove copies from receiving mailboxes and the delivery provider according to `website/PRIVACY_OPERATIONS.md`. Document any legally required or security-related retention exception and leave `resolved_at` unset while that exception applies. Review open tickets and exceptions regularly.

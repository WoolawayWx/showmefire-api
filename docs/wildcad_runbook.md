# WildCAD-E dispatch feed (MOMOC)

Dispatch incidents and resources for the Missouri-Iowa Interagency Dispatch Center, polled from the JSON endpoints behind
WildWeb and folded into the normal fire incident system (`services/wildcad.py`). Fixtures and the field notes from the first
capture are in `tests/fixtures/wildcad/NOTES.md`.

## Rules (all enforced in code)

- At most **one request per endpoint per 5 minutes**. The guard is in the database (`wildcad_feed_state.next_allowed_at`, claimed
  atomically before the request), so restarts, a second worker or a manual call can't exceed it. Incidents run at :x0:10 and
  resources at :x2:40, 150 s apart.
- User agent `ShowMeFire (contact@showmefire.org)`. No redirects, 20 s timeout, 2 MB cap.
- Failure back-off 5 -> 10 -> 20 -> 30 min. **HTTP 403/429 (or Retry-After) pauses that endpoint for an hour and alerts staff.**
  If this ever happens, stop and contact MOMOC / the Mark Twain National Forest before resuming.
- Off unless `WILDCAD_ENABLED=1`. Unsetting it stops all requests immediately (nothing is queued).
- Everything reads the **local cache** (`wildcad_snapshots`, `wildcad_incidents`, `wildcad_resources`); only `services/wildcad.py::fetch_json`
  touches the network.

## What it does with the data

- Dedupe key `MOMOC:<inc_num>:<year>` (the dispatcher's number restarts every year). Edits to acreage/name update the same record.
- Only `Wildfire` (and prescribed-burn) records inside Missouri, reported in the last 7 days, become fire incidents: they join the
  satellite incident they belong to (2 km / 48 h) or start one. Everything else (false alarms, flights, training, Iowa) is kept for staff only.
- Incident and detection each get a permanent SMF number (`SMF-2026-00123`, `SMFD-2026-0004567`) from triggers on `fire_incidents` / `fire_events`.
- Resources link to an incident through `incident_identifiers {unit, incident_num}` = the incident's `wfdssunit` + `inc_num`.
- Timeline entries (`kind = dispatch`): reported, name/size changes, contained/out, and "On scene: ..." when the on-scene mix changes.
- A new dispatch fire is pushed through the normal fire-detection channel once per incident (`notify_official_listing`); NIFC listing the same
  incident later does not push again. The first poll after enabling only records what is already there (no pushes).
- **Privacy**: the incident commander (`ic_name`) and individual responders are staff-only. Public output has equipment names and a head-count
  of personnel (`public_resources`). Anything the classifier can't recognise is treated as a person.

## Turning it on

1. Deploy the API. `WILDCAD_ENABLED=1` in the API's environment. Optionally `STAFF_PUSH_ENABLED=1` for staff phone alerts.
2. Watch the first two cycles: `GET /api/v2/dispatch/status`, and the `wildcad_*` rows in `data/showmefire.db`.

## When it goes stale

Staff get a Discord system-issue message (and a staff push) when a feed has had no good fetch for 20 minutes, a reminder every 6 hours,
and a "recovered" message. A 403/429 or an unreadable payload (format change) alerts at once. Check `wildcad_feed_state.last_error`.
If the payload format changed, update `parse_incident` / `parse_resources` and the fixtures.

## Reading the cache

```bash
sqlite3 data/showmefire.db "SELECT kind, last_success_at, next_allowed_at, consecutive_failures, last_error FROM wildcad_feed_state"
sqlite3 data/showmefire.db "SELECT incident_key, name, incident_type, acres, incident_id FROM wildcad_incidents ORDER BY reported_at DESC LIMIT 20"
```

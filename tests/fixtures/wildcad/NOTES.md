# WildCAD-E (MOMOC) feed fixtures

Captured 2026-11-13 03:31 UTC with one request per endpoint (user agent `ShowMeFire (contact@showmefire.org)`):
`GET https://snknmqmon6.execute-api.us-west-2.amazonaws.com/centers/MOMOC/{incidents,resources}`.
Names of people were replaced with made-up ones; fiscal comments and the phone line were removed.

Observed facts the parser relies on (response is `[ {data: [...], retrieved, webComment} ]`, a one-element list):
- Incident: `uuid` (stable id), `inc_num` (string, per-year sequence), `date` (local Central time, no zone), `name`, `type`
  (Wildfire, False Alarm, Preparedness/Preposition, Aircraft, Law Enforcement, Miscellaneous, Classroom Training),
  `acres` (string or null), `latitude`/`longitude` as strings and the **longitude is missing its minus sign**, `ic`
  (a person: never public), `resources` (list of resource codes, or `[null]`/`["*******"]`), `webComment` (often the county),
  `fire_status` and `fiscal_data` (JSON strings; `fiscal_data.wfdssunit` is the dispatch unit, e.g. MOMOS/MOMTF).
- Resource: `id`, `code`, `description`, `status` (Out of Svc / On Scene / Returning), `location`, `is_public`,
  `incident_name`, `incident_identifiers {unit, incident_num}` (unit + inc_num = the incident's wfdssunit + inc_num).
- The incident list is a rolling ~20 recent incidents, not only open ones. `retrieved` is the server's own clock.
- Response headers have no ETag / Last-Modified / Cache-Control / rate-limit headers.

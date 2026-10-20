# Burn Check API (v1) and embed, for agencies and developers

A free, public, read-only screening aid for Missouri. For a location it answers **Do not burn**, **Use extreme caution** or
**Conditions look okay**, with the reasons. It is not a burn permit and not a guarantee, and Show Me Fire is not responsible for what
anyone does with it. **Using the API or the embed means you accept the [terms](api_terms.md)** (shown at
https://showmefire.org/api-terms; Burn Check has its own section). The most important rule: always show the disclaimer from the response next to the answer.

## Quick start

    curl "https://api.showmefire.org/v1/burn-check?lat=38.9523&lon=-92.3738"
    curl "https://api.showmefire.org/v1/burn-check?address=205+Loch+Ln+Columbia+MO"

No key is needed. Be polite: identify your application in the `User-Agent` header.

## Endpoints

| Method | Path | What it does | Limit (no key) |
| --- | --- | --- | --- |
| GET | `/v1/burn-check?lat=&lon=` or `?address=` | The burn screening for one point. | 20/min per IP |
| GET | `/v1/burn-check/geocode?q=` | Missouri matches for an address, town, zip or "lat, lon". | 12/min per IP |
| GET | `/v1/burn-check/disclaimer` | The disclaimer text, its version and the terms URL. | 20/min per IP |

A key (`Authorization: Bearer <key>`) raises the limits (ten times by default). Apply on the key request page, https://showmefire.org/tools/api-key-request: staff review each request and email the key when it is approved. Responses are
CORS-enabled for browsers. `/v1/burn-check` answers are never cacheable (`Cache-Control: no-store`): they describe right now, so do not
reuse one for more than a few minutes. Every response carries `Link: <https://showmefire.org/api-terms>; rel="terms-of-service"`. API keys are general: one key works on the whole public API (apply at https://showmefire.org/tools/api-key-request).

## The answer

| Field | Meaning |
| --- | --- |
| `verdict.state` | `not_safe`, `caution` or `conditions_ok` (stable values for your code). |
| `verdict.label` | The exact text to show: "Do not burn", "Use extreme caution", "Conditions look okay". |
| `verdict.headline` | One plain sentence for the state. |
| `reasons[]` | `{code, level, text}`: why. `level` is `not_safe` or `caution`; most serious first. Codes include `burn_ban`, `fire_weather_alert`, `fire_danger`, `wind`, `gust`, `humidity`, `heat`, `worsening`, `data_missing`, `unchecked_*`. |
| `burn_ban` | `{checked, active, details}`: a county burn ban reported to Show Me Fire. `checked: false` means it could not be checked. |
| `alerts` | `{checked, active[]}`: NWS Red Flag Warning / Fire Weather Watch for the county. |
| `current` | Latest analysis at the point: `temp_f`, `rh_pct`, `wind_mph`, `fire_danger` (0-4) and its label, `observed_at`. May be `null`. |
| `today` | `{available, hours[], best_window}`: each remaining hour of today with `state`, wind, humidity, temperature, and the longest run of "okay" hours. |
| `location` | The point, county, and `matched_address` when an address was used. `alternatives[]` lists other address matches. |
| `notes[]` | Things that limit this answer (for example the hourly forecast is missing). Show them. |
| `rules` | The version and thresholds used (Show Me Fire's own conservative screening limits, not official guidance). |
| `disclaimer` | `{version, short, full, terms_url}`. **Show `short` next to every answer and make `full` reachable.** |
| `generated_at` | When the answer was made (UTC). |

It fails closed: a burn ban or Red Flag Warning / Fire Weather Watch always gives "Do not burn", and anything that could not be checked
(old or missing data) keeps the answer from being "Conditions look okay".

### Errors

| Status | Meaning |
| --- | --- |
| 404 | The address was not found in Missouri (`detail` says so). |
| 422 | Bad input, or the point is outside Missouri. |
| 429 | Rate limited; wait a minute. |
| 503 | Address lookup is unavailable; try coordinates. |

Treat any failure as "unknown", never as "okay".

## Embed it on your website

Paste this where you want it. `allow="geolocation"` lets the "Use my location" button work inside the frame.

    <iframe src="https://showmefire.org/embed/burn-check" title="Burn Check"
            width="100%" height="720" style="border:0;max-width:640px"
            allow="geolocation" loading="lazy"></iframe>

Options (query string): `theme=light|dark|auto` (default auto) and `compact=1` (smaller header). The frame tells your page how tall it
needs to be with `postMessage`, so you can size it automatically:

    window.addEventListener('message', function (e) {
      if (e.origin === 'https://showmefire.org' && e.data && e.data.type === 'smf-burn-check-height') {
        document.querySelector('iframe[title="Burn Check"]').style.height = e.data.height + 'px';
      }
    });

The embed shows the required disclaimer, asks the visitor to acknowledge it before any lookup, and keeps the "Powered by Show Me Fire"
credit. Do not hide or alter them (see the terms).

## Rules in one list

Show the disclaimer. Show the label as given. Do not call it official, a permit or "safe". Keep the credit. Stay within the limits. Do not
use it to issue or automate permits or for emergencies. Do not keep visitors' addresses beyond what you need. Full text: [terms](api_terms.md).

# Burn Check: terms of use for agencies, developers and websites that embed it

Version 2026-10-20. **Draft: have it reviewed by someone qualified to give legal advice before you rely on it.** The website page
`/tools/burn-check-api#terms` shows this text, and the API returns its address in every answer (`terms_url`, and a `Link` header
with `rel="terms-of-service"`). Bump the version here and on the page when the wording changes.

## 1. What this is

Burn Check is a free screening aid operated by Show Me Fire. For a Missouri location it reports "Do not burn", "Use extreme
caution" or "Conditions look okay" from public weather data, county-reported burn bans, National Weather Service fire weather
alerts and a few simple rules. It is public information and a convenience. It is not a burn permit, an approval, an official
determination, or advice from any fire department, agency or Show Me Fire.

## 2. By using it you agree

Using the Burn Check API, or showing Burn Check on your site (an iframe or any other way), means you have read and accept these
terms and the disclaimer. If you use it for an organization, you confirm you can accept them for that organization.

## 3. No guarantee, no responsibility

* Show Me Fire does not guarantee that any answer is correct, complete, current or available. Data can be late, wrong or missing,
  and conditions change quickly and locally.
* "Conditions look okay" never means it is safe or legal to burn. Burn bans, permits, open-burning rules and restrictions are set by
  counties, cities, fire districts and state and federal agencies and may apply when Burn Check shows none.
* **Show Me Fire, its operators, volunteers, contributors and data providers are not responsible or liable for any fire, damage,
  injury, death, fine, penalty, loss or other consequence that results, directly or indirectly, from Burn Check or from your use of
  it, to the fullest extent the law allows.** This applies to you and to everyone who sees Burn Check on your site or in your app.
* The service is provided "as is", without warranty of any kind. It may change, slow down or stop at any time, with or without notice.

## 4. Rules for using the API or the embed

You must:

1. **Show the disclaimer.** Whenever you show an answer, show the `disclaimer.short` text from the response next to it, unaltered, and
   make `disclaimer.full` reachable from it (a link or an expandable section). The iframe embed does this for you.
2. **Show the answer as given.** Use the label text exactly (Do not burn / Use extreme caution / Conditions look okay) and do not change
   its meaning, soften it, or show it without its reasons. Never describe an answer as "safe", "approved" or "permitted".
3. **Do not present it as official.** Do not say or imply that Burn Check, or your use of it, is a permit, an approval, or a determination
   by your agency, unless your own agency has separately made that decision. Make clear that it is a general screening aid.
4. **Credit Show Me Fire.** Keep the "Powered by Show Me Fire" credit and link (the embed includes it).
5. **Keep to the limits.** Without a key the API allows 20 checks and 12 address lookups per minute per IP address. Use caching sensibly
   (an answer describes right now, so do not keep one for more than a few minutes) and do not scrape, loop or bulk-process locations.
   Identify your application in the `User-Agent` header. Higher limits are available with a key (see below).
6. **Protect people's information.** Burn Check does not store the location or address it is given. Do not log, store or share the
   addresses or locations of the people using your page beyond what you need, and tell them what you do with them.
7. **Follow the law.** You are responsible for how you use Burn Check, for what you say around it, and for complying with the laws and
   rules that apply to you.

You must not:

* use Burn Check to issue, deny or automate burn permits or any legal, safety or emergency decision;
* use it for emergency alerting or as the only source for a safety-critical decision (call 911 for fires and emergencies);
* run it for any purpose that is unlawful, misleading or harmful, or try to disrupt or overload the service, or get around the limits;
* remove, hide or alter the disclaimer, the labels or the credit.

## 5. Who may use it, and keys

Use is free for fire departments, fire districts, emergency management, public safety, government, schools, and non-commercial
community and non-profit organizations. Commercial use, or use that needs higher limits, needs written permission. Apply for a key on the
key request page (https://showmefire.org/tools/api-key-request), where staff review each request. Approved users are emailed an API key (sent as
`Authorization: Bearer <key>`), which raises the per-minute limits. Keep your key private; we can turn it off at any time.

## 6. Changes and ending access

The API is versioned (`/v1/...`). We aim to give notice of breaking changes to people who have a key, but we may change or remove
anything, or limit or end any use (including by turning off a key or blocking an address), at any time, for any reason, especially to
protect the service or people's safety.

## 7. Your responsibility

You are responsible for your own use and for any claim that arises from it. To the fullest extent the law allows, you agree to hold
Show Me Fire, its operators, volunteers, contributors and data providers harmless from claims arising from your use of Burn Check or
from content you present with it.

## 8. Contact

Questions, keys and permission requests: use the contact page at https://showmefire.org/contact.

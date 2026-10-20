# Show Me Fire API and data: terms of use

Version 2026-10-20. **Draft: have it reviewed by someone qualified to give legal advice before you rely on it.** The website page
`/api-terms` shows this text, and the API returns its address in `Link` headers (`rel="terms-of-service"`) and in the Burn Check
answers (`disclaimer.terms_url`). Bump the version here and on the page when the wording changes. (This replaces
`burn_check_terms.md`, which covered only Burn Check.)

## 1. What this covers

These terms cover the public Show Me Fire API (api.showmefire.org), the data and maps it serves, any API key issued to you, and the
tools Show Me Fire publishes for others to use or embed, including **Burn Check**. Show Me Fire provides weather, fire weather,
fire danger, fire-detection and related information for general awareness. It is public information and a convenience. It is not
an official forecast, warning, order or determination, and it is not advice from any fire department, agency or Show Me Fire.

## 2. By using it you agree

Using the API (with or without a key), or showing Show Me Fire data or tools on your site or in your app, means you have read and
accept these terms and the disclaimers that come with the data. If you use it for an organization, you confirm you can accept them
for that organization.

## 3. No guarantee, no responsibility

* Show Me Fire does not guarantee that any data, forecast, map or answer is correct, complete, current or available. Data can be late,
  wrong or missing, and conditions change quickly and locally.
* Nothing here is a permit, an approval, or a determination that anything is safe or legal. Burn bans, permits, open-burning rules
  and restrictions are set by counties, cities, fire districts and state and federal agencies and may apply when Show Me Fire shows none.
* **Show Me Fire, its operators, volunteers, contributors and data providers are not responsible or liable for any fire, damage,
  injury, death, fine, penalty, loss or other consequence that results, directly or indirectly, from the data or tools or from your
  use of them, to the fullest extent the law allows.** This applies to you and to everyone who sees the data or tools on your site or
  in your app.
* The service is provided "as is", without warranty of any kind. It may change, slow down or stop at any time, with or without notice.

## 4. Rules for everyone

You must:

1. **Keep disclaimers with the data.** Where Show Me Fire provides a disclaimer with a product (Burn Check does, in every answer), show it
   next to the product, unaltered, and keep its full text reachable.
2. **Not present it as official.** Do not say or imply that Show Me Fire data or tools, or your use of them, are a permit, an approval, or
   a determination by your agency or any agency, unless that agency has separately made that decision. Make clear they are general
   information.
3. **Credit Show Me Fire** where you show our data, maps or answers, with a link to https://showmefire.org (the embeds include it).
4. **Keep to the limits.** Without a key the limits are low and per IP address (each route's limit is in the API index). Use caching
   sensibly, do not scrape, loop or bulk-download beyond what the API offers for that purpose, and identify your application in the
   `User-Agent` header.
5. **Protect people's information.** Where you send us a person's location or address (as Burn Check does), we use it only to answer that
   request and do not store it. Do not log, store or share the locations or addresses of the people using your page beyond what you
   need, and tell them what you do with them.
6. **Follow the law.** You are responsible for how you use the data, for what you say around it, and for complying with the laws and rules
   that apply to you.

You must not:

* use the data or tools to issue, deny or automate permits or any legal, safety or emergency decision;
* use them for emergency alerting or as the only source for a safety-critical decision (call 911 for fires and emergencies);
* use them for any purpose that is unlawful, misleading or harmful, or try to disrupt or overload the service, or get around the limits;
* remove, hide or alter a disclaimer, an answer label or the credit.

## 5. Burn Check: additional rules

Burn Check (the `/v1/burn-check` API and the embeddable page) answers "Do not burn", "Use extreme caution" or "Conditions look okay". In
addition to section 4 you must:

1. Show `disclaimer.short` from the response next to every answer and make `disclaimer.full` reachable from it. The iframe embed does this
   for you.
2. Show the answer label exactly as given and not change its meaning, soften it, or show it without its reasons. Never describe an answer
   as "safe", "approved" or "permitted". "Conditions look okay" is never a guarantee that burning is safe or legal.
3. Not use an answer older than a few minutes: it describes right now.

## 6. Who may use it, and keys

Use is free for fire departments, fire districts, emergency management, public safety, government, schools, and non-commercial
community and non-profit organizations, and for personal and hobby projects within the public limits. Commercial use, or use that needs
higher limits, needs written permission. Apply for a key on the key request page (https://showmefire.org/tools/api-key-request), where
staff review each request. A key is general: it raises your per-minute limits on the public Show Me Fire API, not only on one tool. Approved
users are emailed a key (sent as `Authorization: Bearer <key>`). Keep your key private and out of public web pages; we can turn it off at any
time, and may replace it if it is exposed.

## 7. Changes and ending access

The API is versioned where it has versions (for example `/v1/...`). We aim to give notice of breaking changes to people who have a key,
but we may change or remove anything, or limit or end any use (including by turning off a key or blocking an address), at any time,
for any reason, especially to protect the service or people's safety.

## 8. Your responsibility

You are responsible for your own use and for any claim that arises from it. To the fullest extent the law allows, you agree to hold
Show Me Fire, its operators, volunteers, contributors and data providers harmless from claims arising from your use of the data or tools
or from content you present with them.

## 9. Contact

Questions and permission requests: use the contact page at https://showmefire.org/contact.

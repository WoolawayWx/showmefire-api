"""The disclaimer shown with every burn-check answer (the API returns it; the website shows the same words).

Plain language on purpose, and kept in one place. It should be reviewed by someone qualified to give legal advice before the
tool is promoted widely; bump VERSION whenever the wording changes so a client can ask people to read it again.
"""
from __future__ import annotations

VERSION = "2026-10-20"
TERMS_URL = "https://showmefire.org/tools/burn-check-api#terms"   # terms for agencies, developers and embeds: docs/burn_check_terms.md

SHORT = (
    "For general awareness only. This is not a burn permit, not official fire-agency guidance, and not a promise that "
    "burning is safe. Show Me Fire is not responsible for any fire, damage, injury, loss or legal consequence that follows "
    "from using it. You alone decide whether to burn."
)

FULL = (
    "Show Me Fire provides this tool for general information only. It estimates conditions from public weather data and "
    "simple screening rules, and the data can be late, wrong or missing, and conditions can change quickly and locally.\n\n"
    "An answer of \"Conditions look okay\" is never a guarantee that burning is safe or legal. It is not a burn permit, not "
    "an approval, and not advice from any fire department or government agency. Burn bans, permits, open-burning rules and "
    "restrictions are set by counties, cities, fire districts and state and federal agencies, and may apply even when this "
    "tool shows no ban. Check with your local fire department, the Missouri Department of Conservation, and local "
    "ordinances before you burn, and follow what they say over anything shown here.\n\n"
    "Show Me Fire, its operators, volunteers, contributors and data providers do not accept responsibility or liability for "
    "any fire, property damage, injury, death, fine, penalty, or other loss or consequence that results, directly or "
    "indirectly, from using this tool or relying on it, to the fullest extent the law allows. You are solely responsible "
    "for your decision to burn, for the fire you start, and for keeping it under control. Never leave a fire unattended, "
    "and call 911 if a fire gets out of control.\n\n"
    "Your location or address is used only to look up conditions for this request and is not stored by this tool."
)


def payload() -> dict:
    return {"version": VERSION, "short": SHORT, "full": FULL, "terms_url": TERMS_URL}

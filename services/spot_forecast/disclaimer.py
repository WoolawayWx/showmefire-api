"""The words shown with every unofficial spot-style forecast. One place, plain language; bump VERSION when the wording changes.
Like the burn check disclaimer, it should get legal review before the tool is promoted widely."""
from __future__ import annotations

VERSION = "2026-10-24"
TERMS_URL = "https://showmefire.org/api-terms"

TITLE = "Unofficial forecast. Not a National Weather Service spot forecast."

SHORT = (
    "This is an automated, unofficial estimate from public model data. It is not a National Weather Service (NWS) spot "
    "forecast, was not reviewed by a forecaster, and must not be used to plan or manage a fire. For a burn or incident, "
    "have an eligible agency request an official spot forecast. Show Me Fire is not responsible for any fire, damage, "
    "injury or loss that follows from using it."
)

FULL = (
    "Show Me Fire builds this from raw HRRR model output and simple rules. A real NWS spot forecast is written by a "
    "forecaster who checks the model against current observations, terrain and the site you give them, and who can talk "
    "to you about it. This page cannot do any of that, and model output at a single grid point can be wrong about wind, "
    "humidity, timing and wind shifts, especially near fronts, storms and rough terrain.\n\n"
    "Do not use this as a substitute for an official spot forecast, a burn plan, a permit, or the judgment of your fire "
    "department or land manager. Burn bans, permits and open-burning rules may apply even when this page shows nothing. "
    "Show Me Fire, its operators, volunteers, contributors and data providers do not accept responsibility or liability "
    "for any fire, property damage, injury, death, fine or other loss that results, directly or indirectly, from using "
    "this page or relying on it, to the fullest extent the law allows.\n\n"
    "Your location or address is used only to build this forecast and is not stored."
)


def payload() -> dict:
    return {"version": VERSION, "title": TITLE, "short": SHORT, "full": FULL, "terms_url": TERMS_URL}

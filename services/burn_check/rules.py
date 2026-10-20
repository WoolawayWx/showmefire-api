"""The screening rules: turn one set of conditions into "do not burn", "use extreme caution" or "conditions look okay".

Pure functions, no I/O: the thresholds come from core/burn_rules.json, so they can be read, tested and changed in one place.
The tool fails closed: anything it cannot check (missing or old data) is never treated as fine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from core.fire_danger import CATEGORY_LABELS

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "core" / "burn_rules.json"
RULES: dict[str, Any] = json.loads(RULES_PATH.read_text(encoding="utf-8"))

NOT_SAFE, CAUTION, OK, UNKNOWN = "not_safe", "caution", "conditions_ok", "unknown"
ORDER = {OK: 0, CAUTION: 1, NOT_SAFE: 2}
LABELS = {
    NOT_SAFE: "Do not burn",
    CAUTION: "Use extreme caution",
    OK: "Conditions look okay",
    UNKNOWN: "Can't tell right now",
}


def reason(code: str, level: str, text: str) -> dict:
    return {"code": code, "level": level, "text": text}


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def assess_conditions(values: dict, rules: Optional[dict] = None, *, label: str = "") -> tuple[str, list[dict]]:
    """(state, reasons) for one moment. `values` may hold temp_f, rh_pct, wind_mph, gust_mph and fire_danger (0-4).

    Wind, humidity or fire danger missing means UNKNOWN: a screening that skipped them would be guessing. Gust and temperature are
    optional (the current-conditions source has no gust)."""
    rules = rules or RULES
    wind, gust = _num(values.get("wind_mph")), _num(values.get("gust_mph"))
    rh, temp = _num(values.get("rh_pct")), _num(values.get("temp_f"))
    danger = values.get("fire_danger")
    danger = int(danger) if _num(danger) is not None else None
    when = f" {label}" if label else ""
    if wind is None or rh is None or danger is None:
        return UNKNOWN, [reason("data_missing", CAUTION, f"Wind, humidity or fire danger is not available{when}.")]

    no, ca = rules["not_safe"], rules["caution"]
    reasons: list[dict] = []
    if danger >= no["fire_danger_at_least"]:
        reasons.append(reason("fire_danger", NOT_SAFE, f"Fire danger is {CATEGORY_LABELS[danger]}{when}."))
    elif danger >= ca["fire_danger_at_least"]:
        reasons.append(reason("fire_danger", CAUTION, f"Fire danger is {CATEGORY_LABELS[danger]}{when}."))
    if wind >= no["wind_mph_at_least"]:
        reasons.append(reason("wind", NOT_SAFE, f"Wind is {wind:.0f} mph{when}, too strong to hold a fire."))
    elif wind >= ca["wind_mph_at_least"]:
        reasons.append(reason("wind", CAUTION, f"Wind is {wind:.0f} mph{when}; embers can blow out of a burn."))
    if gust is not None:
        if gust >= no["gust_mph_at_least"]:
            reasons.append(reason("gust", NOT_SAFE, f"Gusts reach {gust:.0f} mph{when}."))
        elif gust >= ca["gust_mph_at_least"]:
            reasons.append(reason("gust", CAUTION, f"Gusts reach {gust:.0f} mph{when}."))
    if rh < no["rh_pct_below"]:
        reasons.append(reason("humidity", NOT_SAFE, f"Relative humidity is {rh:.0f}%{when}, very dry."))
    elif rh < ca["rh_pct_below"]:
        reasons.append(reason("humidity", CAUTION, f"Relative humidity is {rh:.0f}%{when}, dry."))
    if temp is not None and temp >= ca["temp_f_at_least"]:
        reasons.append(reason("heat", CAUTION, f"It is hot ({temp:.0f}°F){when}, which dries fuels."))

    state = max((r["level"] for r in reasons), key=ORDER.get, default=OK)
    return state, reasons


def combine(now_state: str, now_reasons: list[dict], *, ban: Optional[dict], alerts: list[dict], later_not_safe: Optional[dict],
            checks_failed: list[str]) -> tuple[str, list[dict]]:
    """The overall answer: the current screening, raised by a burn ban, a fire-weather alert, a worsening hour soon, or by
    something that could not be checked. Anything that could not be checked keeps the answer from being "okay"."""
    reasons = list(now_reasons)
    state = now_state if now_state != UNKNOWN else CAUTION
    if ban:
        reasons.insert(0, reason("burn_ban", NOT_SAFE, f"A burn ban is in effect for {ban['county_name']} County."))
        state = NOT_SAFE
    for alert in alerts:
        reasons.insert(0, reason("fire_weather_alert", NOT_SAFE, f"{alert['event']} is in effect for this area."))
        state = NOT_SAFE
    if later_not_safe and state != NOT_SAFE:
        reasons.append(reason("worsening", CAUTION, f"Conditions are expected to turn unsafe around {later_not_safe['hour_label']}."))
        if state == OK:
            state = CAUTION
    for name in checks_failed:
        reasons.append(reason(f"unchecked_{name}", CAUTION, f"{_CHECK_TEXT[name]} could not be checked, so this answer may be incomplete."))
        if state == OK:
            state = CAUTION
    return state, reasons


_CHECK_TEXT = {
    "burn_ban": "Burn bans",
    "alerts": "Red Flag and Fire Weather alerts",
    "county": "The county for this location",
    "current": "Current conditions",
}


def headline(state: str) -> str:
    return {
        NOT_SAFE: "Do not burn.",
        CAUTION: "Burning is risky right now. Use extreme caution, or wait.",
        OK: "Conditions look okay right now, but that is not a guarantee.",
    }[state]


def best_window(hours: list[dict]) -> Optional[dict]:
    """The longest run of consecutive hours whose state is OK (a tie goes to the earlier one)."""
    best: Optional[tuple[int, int]] = None
    start: Optional[int] = None
    for index, hour in enumerate([*hours, {"state": None}]):
        if hour["state"] == OK:
            start = index if start is None else start
            continue
        if start is not None and (best is None or index - start > best[1] - best[0]):
            best = (start, index)
        start = None
    if best is None:
        return None
    first, last = hours[best[0]], hours[best[1] - 1]
    return {"from_local": first["hour_label"], "to_local": last["hour_label"], "hours": best[1] - best[0]}

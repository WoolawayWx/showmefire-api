"""Burn check: for a Missouri location, is it a reasonable time to burn?

Pulls the current conditions, the county burn ban, any Red Flag Warning or Fire Weather Watch and today's hourly forecast for
the point, screens them with the rules in core/burn_rules.json (rules.py), and returns one of "Do not burn", "Use extreme
caution" or "Conditions look okay" with the reasons. It is a screening aid and never a clearance: every answer carries the
disclaimer (disclaimer.py), and anything that cannot be checked keeps the answer from being "okay".
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from services.burn_check import disclaimer, rules, sources

logger = logging.getLogger(__name__)

LOCAL_TZ = ZoneInfo("America/Chicago")


class OutsideMissouri(ValueError):
    """The point is not in a Missouri county."""


def hour_label(moment: datetime) -> str:
    local = moment.astimezone(LOCAL_TZ)
    return f"{local.hour % 12 or 12} {'AM' if local.hour < 12 else 'PM'}"


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _number(value, digits: int = 0):
    return None if value is None or value != value else round(float(value), digits)


def _hours_today(outlook: Optional[dict], now: datetime, rule_set: dict) -> list[dict]:
    """Each remaining hour of today (this hour until the end of the local day set in the rules), screened."""
    if not outlook:
        return []
    start = now.replace(minute=0, second=0, microsecond=0)
    end_local = now.astimezone(LOCAL_TZ).replace(hour=rule_set["day_ends_local_hour"], minute=0, second=0, microsecond=0)
    hours = []
    for row in outlook["rows"]:
        if not (start <= row["time_utc"] <= end_local.astimezone(timezone.utc)):
            continue
        state, _reasons = rules.assess_conditions(row, rule_set)
        hours.append({"time_utc": row["time_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"), "hour_label": hour_label(row["time_utc"]),
                      "temp_f": _number(row["temp_f"]), "rh_pct": _number(row["rh_pct"]), "wind_mph": _number(row["wind_mph"]),
                      "gust_mph": _number(row["gust_mph"]), "state": state})
    return hours


def assess(lat: float, lon: float, *, now: Optional[datetime] = None, matched_address: Optional[str] = None,
           src=sources, rule_set: Optional[dict] = None) -> dict:
    """The full burn-check answer for a point. Raises OutsideMissouri if the point is not in a Missouri county."""
    rule_set = rule_set or rules.RULES
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    fips, county_name = src.county(lat, lon)
    if not fips:
        raise OutsideMissouri()
    ages = rule_set["max_age_hours"]
    failed: list[str] = []
    notes: list[str] = []

    current = None
    try:
        current = src.current_conditions(lat, lon)
        observed = _parse_time(current["observed_at"]) if current else None
        if current is None or observed is None or now - observed > timedelta(hours=ages["current"]):
            current = None
    except Exception:  # noqa: BLE001 - reported to the caller as "could not be checked", never as fine
        logger.exception("burn-check current conditions failed")
        current = None
    if current is None:
        failed.append("current")
        now_state, now_reasons = rules.UNKNOWN, [rules.reason("data_missing", rules.CAUTION, "Current conditions are not available.")]
    else:
        now_state, now_reasons = rules.assess_conditions(current, rule_set)

    ban = None
    try:
        ban = src.active_burn_ban(fips, now)
    except Exception:  # noqa: BLE001
        logger.exception("burn-check burn ban lookup failed")
        failed.append("burn_ban")

    alerts: list[dict] = []
    try:
        alerts, fresh = src.fire_weather_alerts(fips, now, ages["alerts_minutes"])
        if not fresh:
            failed.append("alerts")
    except Exception:  # noqa: BLE001
        logger.exception("burn-check alert lookup failed")
        failed.append("alerts")

    outlook = None
    try:
        outlook = src.hourly_outlook(lat, lon, now, ages["outlook"])
    except Exception:  # noqa: BLE001
        logger.exception("burn-check outlook failed")
    hours = _hours_today(outlook, now, rule_set)
    if not hours:
        notes.append("The hourly forecast for today is not available, so this answer only covers conditions right now.")
    soon = now + timedelta(hours=rule_set["lookahead_not_safe_hours"])
    # The first hour soon that is worse than right now: okay turning risky counts, not only turning unsafe.
    # When right now could not be checked the answer is already "caution", so only an unsafe hour adds anything.
    baseline = rules.ORDER.get(now_state, rules.ORDER[rules.CAUTION])
    later = next((h for h in hours if rules.ORDER.get(h["state"], 0) > baseline
                  and _parse_time(h["time_utc"]) > now and _parse_time(h["time_utc"]) <= soon), None)

    state, reasons = rules.combine(now_state, now_reasons, ban=ban, alerts=alerts, later_not_safe=later, checks_failed=failed)
    reasons.sort(key=lambda r: -rules.ORDER[r["level"]])
    return {
        "success": True,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "location": {"lat": round(lat, 4), "lon": round(lon, 4), "county": county_name, "county_fips": fips,
                     "matched_address": matched_address},
        "verdict": {"state": state, "label": rules.LABELS[state], "headline": rules.headline(state)},
        "reasons": reasons,
        "notes": notes,
        "burn_ban": {"checked": "burn_ban" not in failed, "active": bool(ban), "details": ban},
        "alerts": {"checked": "alerts" not in failed, "active": alerts},
        "current": None if current is None else {
            "observed_at": current["observed_at"], "temp_f": _number(current["temp_f"]), "rh_pct": _number(current["rh_pct"]),
            "wind_mph": _number(current["wind_mph"]), "fire_danger": current["fire_danger"],
            "fire_danger_label": None if current["fire_danger"] is None else rules.CATEGORY_LABELS[current["fire_danger"]],
            "fuel_moisture_pct": _number(current["fuel_moisture_pct"], 1), "precip_mm": _number(current["precip_mm"], 1)},
        "today": {"available": bool(hours), "cycle": outlook["cycle"] if outlook else None, "hours": hours,
                  "best_window": rules.best_window(hours) if hours else None,
                  "forecast_note": "Hours after now come from the HRRR forecast model; its fire danger is an estimate."},
        "rules": {"version": rule_set["version"], "not_safe": rule_set["not_safe"], "caution": rule_set["caution"],
                  "note": rule_set["note"]},
        "disclaimer": disclaimer.payload(),
    }

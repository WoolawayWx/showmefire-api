"""Unofficial spot-style forecast for a Missouri point: hourly HRRR values summarised the way a National Weather Service spot
forecast is laid out (temperature, humidity, wind and gusts, mixing height, transport wind, ventilation, Haines, Hot-Dry-Windy,
wind shifts). It reports model values and does not judge them. It is not a spot forecast, so every answer carries the push to
request an official one (official.py) and the disclaimer (disclaimer.py).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.burn_check import OutsideMissouri
from services.spot_forecast import disclaimer, narrative, official, sources
from services.spot_forecast.narrative import LOCAL_TZ, RULES

logger = logging.getLogger(__name__)

WINDOWS = ("today", "tomorrow")


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(value, digits: int = 0):
    return None if value is None or value != value else round(float(value), digits)


def window_rows(rows: list[dict], window: str, now: datetime, rules: Optional[dict] = None) -> list[dict]:
    """The hours of today (this hour to the end of the local day set in the rules) or of tomorrow (the whole local day)."""
    cfg = (rules or RULES)["windows"]
    local = now.astimezone(LOCAL_TZ)
    day = local.date() + timedelta(days=1 if window == "tomorrow" else 0)
    start_local = datetime(day.year, day.month, day.day, cfg["day_starts_local_hour"], tzinfo=LOCAL_TZ)
    end_local = datetime(day.year, day.month, day.day, cfg["day_ends_local_hour"], tzinfo=LOCAL_TZ)
    start = max(start_local, local.replace(minute=0, second=0, microsecond=0)) if window == "today" else start_local
    return [r for r in rows if start <= r["time_utc"].astimezone(LOCAL_TZ) <= end_local]


def assess(lat: float, lon: float, *, window: str = "today", now: Optional[datetime] = None,
           matched_address: Optional[str] = None, src=sources, office_lookup=official.office_for_point,
           rule_set: Optional[dict] = None) -> dict:
    """The spot-style forecast for a point. Raises OutsideMissouri if the point is not in a Missouri county."""
    rule_set = rule_set or RULES
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    fips, county_name = src.county(lat, lon)
    if not fips:
        raise OutsideMissouri()
    ages = rule_set["max_age_hours"]
    notes: list[str] = []

    ban = None
    try:
        ban = src.active_burn_ban(fips, now)
    except Exception:  # noqa: BLE001
        logger.exception("spot-forecast burn ban lookup failed")
        notes.append("Burn bans could not be checked. Ask your county or fire department.")
    alerts: list[dict] = []
    try:
        alerts, fresh = src.fire_weather_alerts(fips, now, ages["alerts_minutes"])
        if not fresh:
            notes.append("Red Flag and Fire Weather alerts could not be checked.")
    except Exception:  # noqa: BLE001
        logger.exception("spot-forecast alert lookup failed")
        notes.append("Red Flag and Fire Weather alerts could not be checked.")

    outlook = None
    try:
        outlook = src.hourly(lat, lon, now, ages["outlook"])
    except Exception:  # noqa: BLE001
        logger.exception("spot-forecast hourly failed")
    rows = window_rows(outlook["rows"], window, now, rule_set) if outlook else []
    if not rows:
        notes.append("The hourly forecast for this window is not available, so no forecast is shown." if outlook is None
                     else "The model forecast does not cover this window yet.")

    hourly_rows = [{
        "time_utc": _iso(r["time_utc"]), "hour_label": narrative.hour_label(r["time_utc"]),
        "temp_f": _num(r["temp_f"]), "rh_pct": _num(r["rh_pct"]), "wind_mph": _num(r["wind_mph"]), "gust_mph": _num(r["gust_mph"]),
        "wind_dir": narrative.compass(r["wind_dir_deg"]), "wind_arrow": narrative.arrow(r["wind_dir_deg"]),
        "wind_dir_deg": _num(r["wind_dir_deg"]),
        "mixing_height_ft": _num(r["mixing_height_ft"], -2), "transport_mph": _num(r["transport_mph"]),
        "ventilation_kt_ft": _num(r["ventilation_kt_ft"], -2), "ventilation": narrative.ventilation_category(r["ventilation_kt_ft"], rule_set),
        "haines": _num(r["haines"]), "haines_label": narrative.haines_label(r["haines"]), "hdw": _num(r["hdw"]),
        "inversion_strength_f": _num(r["inversion_strength_f"], 1), "inversion_top_ft": _num(r["inversion_top_ft"], -2),
        "smoke_ugm3": _num(r["smoke_ugm3"]),
    } for r in rows]
    return {
        "success": True,
        "generated_at": _iso(now),
        "window": window,
        "location": {"lat": round(lat, 4), "lon": round(lon, 4), "county": county_name, "county_fips": fips, "matched_address": matched_address},
        "official_spot": official.block(office_lookup(lat, lon)),
        "unofficial": {"label": disclaimer.TITLE, "available": bool(rows), "model": "HRRR",
                       "cycle": outlook["cycle"] if outlook else None,
                       "periods": narrative.build_periods(rows, rule_set) if rows else [],
                       "hours": hourly_rows},
        "alerts": alerts,
        "burn_ban": ban,
        "notes": notes,
        "settings": {"version": rule_set["version"], "note": rule_set["note"]},
        "disclaimer": disclaimer.payload(),
    }

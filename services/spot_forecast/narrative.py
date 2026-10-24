"""Turn hourly model values into NWS-spot-style periods and plain text. Pure functions, no I/O.

The layout follows what a spot forecast covers (sky and weather are left out: the model gives no reliable sky): temperature,
relative humidity, 10 m wind with gusts, mixing height, transport wind and ventilation, plus wind shifts. It is deliberately
labelled as an estimate everywhere it is shown (see disclaimer.py).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from services.fire_weather.indices import VENTILATION_CATEGORIES, ventilation_category as _vent_index

RULES_PATH = Path(__file__).resolve().parent.parent.parent / "core" / "spot_rules.json"
RULES: dict[str, Any] = json.loads(RULES_PATH.read_text(encoding="utf-8"))
LOCAL_TZ = ZoneInfo("America/Chicago")
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def compass(degrees: Optional[float]) -> Optional[str]:
    if degrees is None or degrees != degrees:
        return None
    return COMPASS[int((degrees % 360) / 22.5 + 0.5) % 16]


ARROWS = ("↑", "↗", "→", "↘", "↓", "↙", "←", "↖")


def arrow(from_degrees: Optional[float]) -> Optional[str]:
    """An arrow pointing where the wind is going, for a direction given as where it blows FROM (a north wind is a down arrow)."""
    if from_degrees is None or from_degrees != from_degrees:
        return None
    return ARROWS[int((((from_degrees + 180) % 360) / 45) + 0.5) % 8]


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def hour_label(moment: datetime) -> str:
    local = moment.astimezone(LOCAL_TZ)
    return f"{local.hour % 12 or 12} {'AM' if local.hour < 12 else 'PM'}"


HAINES_LABELS = {2: "very low", 3: "very low", 4: "low", 5: "moderate", 6: "high"}   # as the data viewer labels them


def ventilation_category(value: Optional[float], rules: Optional[dict] = None) -> Optional[str]:
    """The NWS Springfield smoke-dispersal category for a ventilation rate in kt ft (Poor, Fair, Good, Very Good, Excellent), from
    the same table the HRRR hourly product and data viewer use (services/fire_weather/indices.py)."""
    if value is None or value != value:
        return None
    index = int(_vent_index(value))
    return VENTILATION_CATEGORIES[index][0] if index >= 0 else None


def haines_label(value: Optional[float]) -> Optional[str]:
    return None if value is None or value != value else HAINES_LABELS.get(int(round(value)))


def _vals(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None and r[key] == r[key]]


def _range(values: list[float], unit: str = "") -> Optional[str]:
    if not values:
        return None
    low, high = round(min(values)), round(max(values))
    return f"{low}{unit}" if low == high else f"{low} to {high}{unit}"


def wind_shifts(rows: list[dict], rules: Optional[dict] = None) -> list[dict]:
    """Hours where the 10 m wind direction turns by the configured angle or more from the previous hour. Light winds are
    ignored: their direction is noise."""
    limit = (rules or RULES)["wind_shift_degrees_at_least"]
    out = []
    for before, after in zip(rows, rows[1:]):
        a, b = before.get("wind_dir_deg"), after.get("wind_dir_deg")
        if a is None or b is None or (after.get("wind_mph") or 0) < 5 or (before.get("wind_mph") or 0) < 5:
            continue
        if angle_diff(a, b) >= limit:
            out.append({"time_utc": after["time_utc"], "hour_label": hour_label(after["time_utc"]),
                        "from": compass(a), "to": compass(b), "from_arrow": arrow(a), "to_arrow": arrow(b)})
    return out


def period_summary(name: str, rows: list[dict], rules: Optional[dict] = None) -> dict:
    """One period (Morning/Afternoon/Evening) summarised in the NWS element order."""
    rules = rules or RULES
    winds, gusts, humidity = _vals(rows, "wind_mph"), _vals(rows, "gust_mph"), _vals(rows, "rh_pct")
    directions = [r["wind_dir_deg"] for r in rows if r.get("wind_dir_deg") is not None and (r.get("wind_mph") or 0) >= 3]
    mixing, transport, vent = _vals(rows, "mixing_height_ft"), _vals(rows, "transport_mph"), _vals(rows, "ventilation_kt_ft")
    mid = directions[len(directions) // 2] if directions else None
    haines, hdw, smoke = _vals(rows, "haines"), _vals(rows, "hdw"), _vals(rows, "smoke_ugm3")
    summary = {
        "name": name,
        "from_local": hour_label(rows[0]["time_utc"]), "to_local": hour_label(rows[-1]["time_utc"]),
        "temp_f": _range(_vals(rows, "temp_f")), "rh_pct": _range(humidity, "%"),
        "rh_min_pct": round(min(humidity)) if humidity else None,
        "wind_mph": _range(winds), "wind_dir": compass(mid), "wind_arrow": arrow(mid), "gust_max_mph": round(max(gusts)) if gusts else None,
        "mixing_height_ft_max": int(round(max(mixing), -2)) if mixing else None,
        "transport_mph": _range(transport),
        "ventilation_kt_ft_max": int(round(max(vent), -2)) if vent else None,
        "ventilation": ventilation_category(max(vent), rules) if vent else None,
        "haines_max": int(round(max(haines))) if haines else None,
        "haines_label": haines_label(max(haines)) if haines else None,
        "hdw_max": int(round(max(hdw))) if hdw else None,
        "smoke_max_ugm3": round(max(smoke)) if smoke else None,
        "shifts": [{k: v for k, v in s.items() if k != "time_utc"} for s in wind_shifts(rows, rules)],
    }
    summary["text"] = period_text(summary)
    return summary


def period_text(p: dict) -> str:
    parts = [f"{p['name']} ({p['from_local']} to {p['to_local']}):"]
    if p["temp_f"]:
        parts.append(f"Temperature {p['temp_f']}°F.")
    if p["rh_pct"]:
        parts.append(f"Humidity {p['rh_pct']}.")
    if p["wind_mph"]:
        wind = f"Wind {p['wind_arrow']} {p['wind_dir']} " if p["wind_dir"] else "Wind "
        gust = f", gusts to {p['gust_max_mph']} mph" if p["gust_max_mph"] else ""
        parts.append(f"{wind}{p['wind_mph']} mph{gust}.")
    for shift in p["shifts"]:
        parts.append(f"Wind shifts {shift['from_arrow']} {shift['from']} to {shift['to_arrow']} {shift['to']} around {shift['hour_label']}.")
    if p["mixing_height_ft_max"] is not None:
        parts.append(f"Mixing height up to {p['mixing_height_ft_max']:,} ft.")
    if p["ventilation"]:
        parts.append(f"Ventilation {p['ventilation']} (peak {p['ventilation_kt_ft_max']:,} kt-ft).")
    if p["haines_max"] is not None:
        parts.append(f"Haines {p['haines_max']} ({p['haines_label']}).")
    if p["hdw_max"] is not None:
        parts.append(f"Hot-Dry-Windy index up to {p['hdw_max']}.")
    return " ".join(parts)


def build_periods(rows: list[dict], rules: Optional[dict] = None) -> list[dict]:
    rules = rules or RULES
    out = []
    for name, start, end in rules["periods"]:
        chunk = [r for r in rows if start <= r["time_utc"].astimezone(LOCAL_TZ).hour < end]
        if chunk:
            out.append(period_summary(name, chunk, rules))
    return out

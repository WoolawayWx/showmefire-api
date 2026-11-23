"""Freshness of the automated products (maps, model and external data feeds).

Each product is judged by a file its job writes when it runs: a status JSON,
an hourly "checked" stamp, a ledger, etc. A product is "stale" when that file
is older than the product's expected refresh interval plus slack, "missing"
when no file exists yet. Used by GET /health/products and by
scripts/check_product_health.py (Discord alerts on stale/recovered).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1]
STATUS_DIR = PROJECT_DIR / "data" / "current_conditions" / "status"
REMINDER_HOURS = 6  # a still-stale product is mentioned again this often
ROUTINE_SOUNDING_MAX_HOURS = 30  # twice-daily sites, with slack for once-a-day staffing schedules


@dataclass
class Product:
    key: str
    label: str
    files: list[str]                 # paths relative to the project; the newest one counts
    max_age_minutes: float
    source: str = ""
    extra: Optional[Callable[[], list[str]]] = field(default=None, repr=False)  # extra problems, if any
    # Extra minutes past max_age_minutes before STAFF are alerted. The product still reports "stale" (health endpoint,
    # status) the moment it passes max_age_minutes; only the alert waits, so a one-cycle blip never pages anyone.
    grace_minutes: float = 0


def _fire_weather_problems() -> list[str]:
    try:
        status = json.loads((STATUS_DIR / "fire_weather_maps.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    behind = status.get("hours_behind", status.get("cycles_behind", 0)) or 0
    return [f"{behind} HRRR hours behind"] if behind > 2 else []


def _sounding_problems() -> list[str]:
    """NWS sites whose newest kept launch is too old (missed or late launches)."""
    try:
        manifest = json.loads((PROJECT_DIR / "images" / "soundings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    now = datetime.now(timezone.utc)
    problems = []
    for station, site in manifest.get("stations", {}).items():
        if not station.startswith("K"):  # university sites launch only for field campaigns
            continue
        launches = site.get("soundings") or []
        if not launches:
            problems.append(f"{station}: no recent launch")
            continue
        newest = datetime.strptime(launches[0]["valid_time"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
        hours = (now - newest).total_seconds() / 3600
        if hours > ROUTINE_SOUNDING_MAX_HOURS:
            problems.append(f"{station}: newest launch {hours:.0f} h old")
    return problems


def _status(name: str) -> str:
    return f"data/current_conditions/status/{name}"


PRODUCTS = [
    Product("maps_cycle", "15-minute map cycle", ["public/rss.xml"], 45, "scripts/maps.sh"),
    Product("station_maps", "Station maps", [_status("dewpointmap.json"), _status("station_plot_current.json")], 45,
            "Synoptic stations"),
    Product("radar", "Radar maps", [_status("radar_missouri.json")], 45, "NOAA RIDGE mosaic"),
    Product("satellite", "Satellite maps", [_status("satellite_ir.json")], 45, "GOES-19 on AWS"),
    Product("satellite_fires", "Satellite fire detections", [_status("satellite_fire_detections.json")], 45,
            "GOES-19 FDC on AWS"),
    Product("fire_weather", "HRRR fire-weather maps", [_status("fire_weather_maps.json")], 120, "HRRR on AWS",
            extra=_fire_weather_problems),
    Product("soundings", "Balloon soundings", ["data/soundings/processed.json"], 45, "IEM RAOB, CW3E",
            extra=_sounding_problems, grace_minutes=30),
    Product("eddi", "EDDI", [_status("drought_eddi.checked")] + [_status(f"eddi_{s}.signature")
                                                               for s in ("1wk", "2wk", "1mo", "3mo")],
            180, "NOAA PSL"),
    Product("vegdri", "VegDRI", [_status("drought_vegdri.checked"), _status("vegdri.signature")], 180, "NCEI / NDMC"),
    Product("quickdri", "QuickDRI", [_status("drought_quickdri.checked"), _status("quickdri.signature")], 180,
            "NCEI / NDMC"),
    Product("fordri", "ForDRI", [_status("drought_fordri.checked"), _status("fordri.signature")], 180, "NDMC"),
    Product("drought_monitor", "U.S. Drought Monitor", [_status("drought_monitor.json"),
                                                         _status("drought_monitor.signature")], 8 * 24 * 60,
            "droughtmonitor.unl.edu"),
]


def _newest_mtime(paths: list[str]) -> Optional[float]:
    times = []
    for relative in paths:
        try:
            times.append((PROJECT_DIR / relative).stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def check_product(product: Product, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    newest = _newest_mtime(product.files)
    result = {"key": product.key, "label": product.label, "source": product.source,
              "max_age_minutes": product.max_age_minutes}
    if newest is None:
        return {**result, "status": "missing", "last_update": None, "age_minutes": None, "problems": ["no output yet"]}
    age = (now.timestamp() - newest) / 60
    problems = []
    if age > product.max_age_minutes:
        problems.append(f"not updated for {age:.0f} min (expected every {product.max_age_minutes:.0f})")
    age_only = len(problems) == 1
    if product.extra:
        try:
            problems += product.extra()
        except Exception as exc:  # noqa: BLE001 - a broken check must not break the endpoint
            problems.append(f"check failed: {exc}")
    # Only a lateness-only problem can sit inside the grace window; anything else (a failed extra check) alerts at once.
    within_grace = age_only and len(problems) == 1 and age <= product.max_age_minutes + product.grace_minutes
    return {**result, "status": "stale" if problems else "ok", "within_grace": within_grace,
            "last_update": datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "age_minutes": round(age, 1), "problems": problems}


def active_products() -> list[Product]:
    """The fixed products plus the WildCAD feeds, which only count once the feed is switched on."""
    products = list(PRODUCTS)
    try:
        from services import wildcad, wildcad_health

        if wildcad.enabled():
            products += wildcad_health.products()
        from services import scheduler_heartbeat

        products += scheduler_heartbeat.products()   # one per worker that has ever reported
    except Exception:  # noqa: BLE001 - a broken optional check must not break the endpoint
        pass
    return products


def plan_alerts(report: dict, state: dict, now: datetime) -> tuple[list[tuple[str, str]], dict]:
    """(alerts to send as (message, detail), new state).

    Two-strike rule: a product that turns stale is only noted ("pending") on the first check that sees it; staff are told
    when the NEXT check still finds it stale, so a one-off blip that clears by itself never alerts (and never sends a
    "recovered" message either). Products inside their `grace_minutes` window are not noted at all.
    """
    alerts, new_state = [], {}
    for product in report["products"]:
        key, label = product["key"], product["label"]
        previous = state.get(key)
        alerted = bool(previous and previous.get("alerted"))
        if product["status"] == "ok":
            if alerted:
                alerts.append((f"{label} recovered", f"{label} is updating normally again."))
            continue
        if product.get("within_grace") and not alerted:
            continue   # late, but inside the grace window: say nothing yet
        if previous is None:
            new_state[key] = {"pending": now.isoformat()}   # first strike
            continue
        detail = f"{label} ({product['source']}): " + "; ".join(product["problems"])
        if not alerted:   # second strike: still stale on the next check
            alerts.append((f"{label} is stale", detail))
            new_state[key] = {"alerted": now.isoformat(), "since": previous.get("pending", now.isoformat())}
            continue
        last_sent = datetime.fromisoformat(previous["alerted"])
        if (now - last_sent).total_seconds() >= REMINDER_HOURS * 3600:
            alerts.append((f"{label} still stale", detail))
            new_state[key] = {"alerted": now.isoformat(), "since": previous.get("since", now.isoformat())}
        else:
            new_state[key] = previous
    return alerts, new_state


def check_all(now: Optional[datetime] = None) -> dict:
    now = now or datetime.now(timezone.utc)
    products = [check_product(product, now) for product in active_products()]
    bad = [p for p in products if p["status"] != "ok"]
    return {
        "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok" if not bad else "degraded",
        "stale_count": len(bad),
        "products": products,
    }

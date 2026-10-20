"""Burn check: rules, the overall answer, geocoding and the public route. The tool fails closed, so most tests are about what
must NOT come out as "okay"."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.rate_limit import RateLimiter
from routers import burn_check as router_module
from services import burn_check
from services.burn_check import disclaimer, geocode, rules, sources

NOW = datetime(2026, 10, 20, 17, 30, tzinfo=timezone.utc)        # 12:30 PM Central
CALM = {"temp_f": 65, "rh_pct": 55, "wind_mph": 5, "gust_mph": 8, "fire_danger": 0}


def state_of(**changes):
    return rules.assess_conditions({**CALM, **changes})[0]


# --- the rules -----------------------------------------------------------------------------------

def test_calm_humid_low_danger_is_okay():
    assert state_of() == rules.OK


@pytest.mark.parametrize("changes, expected", [
    ({"wind_mph": 9}, rules.OK), ({"wind_mph": 10}, rules.CAUTION), ({"wind_mph": 19}, rules.CAUTION), ({"wind_mph": 20}, rules.NOT_SAFE),
    ({"gust_mph": 14}, rules.OK), ({"gust_mph": 15}, rules.CAUTION), ({"gust_mph": 25}, rules.NOT_SAFE),
    ({"rh_pct": 36}, rules.OK), ({"rh_pct": 34}, rules.CAUTION), ({"rh_pct": 25}, rules.CAUTION), ({"rh_pct": 24}, rules.NOT_SAFE),
    ({"fire_danger": 1}, rules.CAUTION), ({"fire_danger": 2}, rules.NOT_SAFE), ({"fire_danger": 4}, rules.NOT_SAFE),
    ({"temp_f": 89}, rules.OK), ({"temp_f": 90}, rules.CAUTION),
    ({"wind_mph": 22, "rh_pct": 60}, rules.NOT_SAFE),
])
def test_thresholds(changes, expected):
    assert state_of(**changes) == expected


@pytest.mark.parametrize("missing", ["wind_mph", "rh_pct", "fire_danger"])
def test_missing_essential_input_is_unknown_never_okay(missing):
    values = {**CALM, missing: None}
    assert rules.assess_conditions(values)[0] == rules.UNKNOWN
    assert rules.assess_conditions({**CALM, missing: float("nan")})[0] == rules.UNKNOWN


def test_gust_and_temperature_are_optional():
    assert rules.assess_conditions({"rh_pct": 55, "wind_mph": 5, "fire_danger": 0})[0] == rules.OK


def combined(state=rules.OK, ban=None, alerts=(), later=None, failed=()):
    return rules.combine(state, [], ban=ban, alerts=list(alerts), later_not_safe=later, checks_failed=list(failed))


def test_a_burn_ban_or_fire_weather_alert_overrides_perfect_weather():
    assert combined(ban={"county_name": "Boone"})[0] == rules.NOT_SAFE
    assert "Boone County" in combined(ban={"county_name": "Boone"})[1][0]["text"]
    assert combined(alerts=[{"event": "Red Flag Warning"}])[0] == rules.NOT_SAFE
    assert combined(alerts=[{"event": "Fire Weather Watch"}])[0] == rules.NOT_SAFE


def test_anything_unchecked_or_unknown_keeps_the_answer_from_being_okay():
    for failed in ("burn_ban", "alerts", "current"):
        state, reasons = combined(failed=[failed])
        assert state == rules.CAUTION and any(r["code"] == f"unchecked_{failed}" for r in reasons)
    assert combined(state=rules.UNKNOWN)[0] == rules.CAUTION
    assert combined(state=rules.NOT_SAFE, failed=["alerts"])[0] == rules.NOT_SAFE      # never softened


def test_a_worsening_hour_soon_raises_okay_to_caution_but_never_lowers_anything():
    later = {"hour_label": "3 PM"}
    state, reasons = combined(later=later)
    assert state == rules.CAUTION and "3 PM" in reasons[-1]["text"]
    assert combined(state=rules.NOT_SAFE, later=later)[0] == rules.NOT_SAFE


def test_best_window_is_the_longest_okay_run():
    def hour(label, state):
        return {"hour_label": label, "state": state}
    hours = [hour("1 PM", rules.OK), hour("2 PM", rules.CAUTION), hour("3 PM", rules.OK), hour("4 PM", rules.OK), hour("5 PM", rules.NOT_SAFE)]
    assert rules.best_window(hours) == {"from_local": "3 PM", "to_local": "4 PM", "hours": 2}
    assert rules.best_window([hour("1 PM", rules.CAUTION)]) is None
    assert rules.best_window([hour("1 PM", rules.OK)]) == {"from_local": "1 PM", "to_local": "1 PM", "hours": 1}


def test_rules_file_is_ordered_sensibly():
    no, ca = rules.RULES["not_safe"], rules.RULES["caution"]
    assert ca["wind_mph_at_least"] < no["wind_mph_at_least"] and ca["gust_mph_at_least"] < no["gust_mph_at_least"]
    assert ca["rh_pct_below"] > no["rh_pct_below"] and ca["fire_danger_at_least"] < no["fire_danger_at_least"]


# --- the overall answer -----------------------------------------------------------------------------

def make_src(current=None, ban=None, alerts=((), True), rows=None, fips="29019", name="Boone", current_error=None):
    observed = (NOW - timedelta(minutes=40)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def current_conditions(lat, lon):
        if current_error:
            raise current_error
        return None if current is False else {"observed_at": observed, "fuel_moisture_pct": 12.0, "precip_mm": 0.0, **(current or CALM)}

    def outlook(lat, lon, now, max_age):
        return None if rows is None else {"cycle": "20261020_12z", "cycle_time_utc": "2026-10-20T12:00:00Z", "rows": rows}

    return SimpleNamespace(county=lambda lat, lon: (fips, name), current_conditions=current_conditions,
                           active_burn_ban=lambda f, now: ban, fire_weather_alerts=lambda f, now, minutes: alerts, hourly_outlook=outlook)


def hourly(*pairs):
    """(hours after 12:00 Central, overrides) -> outlook rows."""
    base = datetime(2026, 10, 20, 17, tzinfo=timezone.utc)
    return [{"time_utc": base + timedelta(hours=h), **{**CALM, "fire_danger": 0}, **changes} for h, changes in pairs]


def test_a_calm_day_with_everything_checked_is_okay_and_carries_the_disclaimer():
    result = burn_check.assess(38.95, -92.33, now=NOW, src=make_src(rows=hourly((0, {}), (1, {}), (2, {}), (3, {}))))
    assert result["verdict"]["state"] == rules.OK and result["verdict"]["label"] == "Conditions look okay"
    assert result["burn_ban"] == {"checked": True, "active": False, "details": None}
    assert result["alerts"]["checked"] is True and result["location"]["county"] == "Boone"
    assert result["today"]["available"] and result["today"]["best_window"]["hours"] >= 3
    assert result["disclaimer"]["version"] == disclaimer.VERSION and "not responsible" in result["disclaimer"]["short"]
    assert result["rules"]["version"] == rules.RULES["version"]


def test_okay_now_but_unsafe_within_three_hours_is_caution():
    rows = hourly((0, {}), (1, {}), (2, {"wind_mph": 25, "gust_mph": 35}))
    result = burn_check.assess(38.95, -92.33, now=NOW, src=make_src(rows=rows))
    assert result["verdict"]["state"] == rules.CAUTION and any(r["code"] == "worsening" for r in result["reasons"])
    assert [h["state"] for h in result["today"]["hours"]][:3] == [rules.OK, rules.OK, rules.NOT_SAFE]


def test_an_unsafe_hour_beyond_the_lookahead_does_not_change_the_answer():
    rows = hourly((0, {}), (1, {}), (2, {}), (3, {}), (4, {"wind_mph": 25}))
    assert burn_check.assess(38.95, -92.33, now=NOW, src=make_src(rows=rows))["verdict"]["state"] == rules.OK


def test_ban_and_alert_make_it_not_safe_and_are_listed_first():
    src = make_src(ban={"county_name": "Boone", "effective_at": "x", "expires_at": None, "note": None, "proof_url": None},
                   alerts=([{"event": "Red Flag Warning", "headline": "h", "expires": None, "area": "Boone"}], True), rows=hourly((0, {})))
    result = burn_check.assess(38.95, -92.33, now=NOW, src=src)
    assert result["verdict"]["state"] == rules.NOT_SAFE
    assert {r["code"] for r in result["reasons"][:2]} == {"burn_ban", "fire_weather_alert"}
    assert result["burn_ban"]["active"] and result["alerts"]["active"][0]["event"] == "Red Flag Warning"


def test_old_current_conditions_are_not_used_and_cannot_give_okay():
    stale = make_src(rows=hourly((0, {})))
    original = stale.current_conditions
    stale.current_conditions = lambda lat, lon: {**original(lat, lon), "observed_at": (NOW - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    result = burn_check.assess(38.95, -92.33, now=NOW, src=stale)
    assert result["current"] is None and result["verdict"]["state"] == rules.CAUTION


@pytest.mark.parametrize("kind", ["none", "error", "missing_time"])
def test_failed_current_conditions_fail_closed(kind):
    if kind == "none":
        src = make_src(current=False)
    elif kind == "error":
        src = make_src(current_error=RuntimeError("boom"))
    else:
        src = make_src()
        src.current_conditions = lambda lat, lon: {**CALM, "observed_at": None, "fuel_moisture_pct": None, "precip_mm": None}
    result = burn_check.assess(38.95, -92.33, now=NOW, src=src)
    assert result["verdict"]["state"] == rules.CAUTION and result["current"] is None


def test_unreadable_ban_list_or_stale_alerts_fail_closed():
    def boom(f, now):
        raise RuntimeError("db locked")
    src = make_src(rows=hourly((0, {})))
    src.active_burn_ban = boom
    result = burn_check.assess(38.95, -92.33, now=NOW, src=src)
    assert result["verdict"]["state"] == rules.CAUTION and result["burn_ban"]["checked"] is False
    result = burn_check.assess(38.95, -92.33, now=NOW, src=make_src(alerts=((), False), rows=hourly((0, {}))))
    assert result["verdict"]["state"] == rules.CAUTION and result["alerts"]["checked"] is False


def test_missing_forecast_is_noted_but_does_not_block_a_current_answer():
    result = burn_check.assess(38.95, -92.33, now=NOW, src=make_src(rows=None))
    assert result["today"] == {**result["today"], "available": False, "hours": [], "best_window": None}
    assert result["notes"] and result["verdict"]["state"] == rules.OK


def test_a_point_outside_every_county_is_rejected():
    with pytest.raises(burn_check.OutsideMissouri):
        burn_check.assess(38.95, -92.33, now=NOW, src=make_src(fips=None, name=None))


def test_hours_stop_at_the_end_of_the_local_day_and_start_this_hour():
    rows = [{"time_utc": datetime(2026, 10, 20, 16, tzinfo=timezone.utc) + timedelta(hours=h), **CALM} for h in range(14)]
    labels = [h["hour_label"] for h in burn_check.assess(38.95, -92.33, now=NOW, src=make_src(rows=rows))["today"]["hours"]]
    assert labels[0] == "12 PM" and labels[-1] == "9 PM"


# --- geocoding ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("38.5767, -92.1735", (38.5767, -92.1735)), ("38.5767 -92.1735", (38.5767, -92.1735)), ("38.5767,-92.1735", (38.5767, -92.1735)),
    ("38.5767 N 92.1735 W", (38.5767, -92.1735)), ("38.5767°N, 92.1735°W", (38.5767, -92.1735)),
    ("38.5767, 92.1735", (38.5767, -92.1735)),            # a Missouri longitude typed without its minus sign
    ("123 Main St, Jefferson City", None), ("65101", None), ("", None), ("95, -92", None),
])
def test_parse_coordinates(text, expected):
    assert geocode.parse_coordinates(text) == expected


def test_coordinates_outside_missouri_find_nothing():
    assert geocode.geocode("40.0, -100.0") == [] and geocode.geocode("95, -92") == []


def census_hit(address, lat, lon):
    return {"result": {"addressMatches": [{"matchedAddress": address, "coordinates": {"x": lon, "y": lat}}]}}


def test_geocode_uses_census_first_then_nominatim_and_filters_to_missouri():
    geocode.clear_cache()
    calls = []

    def get(url, params):
        calls.append(url)
        if url == geocode.CENSUS_URL:
            return census_hit("100 MAIN ST, JEFFERSON CITY, MO, 65101", 38.5767, -92.1735)
        raise AssertionError("nominatim should not be asked")
    found = geocode.geocode("100 Main St Jefferson City", get=get)
    assert found == [{"label": "100 Main St, Jefferson City, Mo, 65101", "lat": 38.5767, "lon": -92.1735, "source": "census"}]
    assert geocode.geocode("100 Main St Jefferson City", get=get) == found and len(calls) == 1           # cached

    geocode.clear_cache()

    def fallback(url, params):
        if url == geocode.CENSUS_URL:
            return {"result": {"addressMatches": []}}
        return [{"display_name": "Rolla, Phelps County, Missouri", "lat": "37.9487", "lon": "-91.7715"},
                {"display_name": "Rolla, North Dakota", "lat": "48.85", "lon": "-99.62"}]
    found = geocode.geocode("Rolla", get=fallback)
    assert [c["source"] for c in found] == ["openstreetmap"] and found[0]["lat"] == 37.9487


def test_geocode_failure_is_different_from_nothing_found():
    geocode.clear_cache()

    def down(url, params):
        raise RuntimeError("timeout")
    with pytest.raises(geocode.GeocodeUnavailable):
        geocode.geocode("somewhere", get=down)

    def census_down_osm_empty(url, params):
        if url == geocode.CENSUS_URL:
            raise RuntimeError("timeout")
        return []
    assert geocode.geocode("nowhere at all", get=census_down_osm_empty) == []
    geocode.clear_cache()


# --- the public route ---------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    geocode.clear_cache()
    src = make_src(rows=hourly((0, {}), (1, {}), (2, {}), (3, {})))
    for name in ("county", "current_conditions", "active_burn_ban", "fire_weather_alerts", "hourly_outlook"):
        monkeypatch.setattr(sources, name, getattr(src, name))
    monkeypatch.setattr(burn_check, "datetime", SimpleNamespace(now=lambda tz=None: NOW, strptime=datetime.strptime, fromisoformat=datetime.fromisoformat))
    for name in ("check_limiter", "geocode_limiter"):
        monkeypatch.setattr(router_module, name, RateLimiter(f"t_{name}", 10_000))
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app)


def test_route_by_coordinates_is_never_cached_and_includes_the_disclaimer(client):
    response = client.get("/v1/burn-check", params={"lat": 38.95, "lon": -92.33})
    body = response.json()
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert disclaimer.TERMS_URL in response.headers["link"] and 'rel="terms-of-service"' in response.headers["link"]
    assert body["disclaimer"]["terms_url"] == disclaimer.TERMS_URL
    assert body["verdict"]["state"] == "conditions_ok" and body["disclaimer"]["full"] and body["location"]["matched_address"] is None


def test_route_by_address_geocodes_and_offers_the_other_matches(client, monkeypatch):
    monkeypatch.setattr(geocode, "geocode", lambda q: [{"label": "A", "lat": 38.95, "lon": -92.33, "source": "census"},
                                                       {"label": "B", "lat": 38.0, "lon": -92.0, "source": "census"}])
    body = client.get("/v1/burn-check", params={"address": "1 Main St"}).json()
    assert body["location"]["matched_address"] == "A" and [a["label"] for a in body["alternatives"]] == ["B"]


def test_route_errors(client, monkeypatch):
    assert client.get("/v1/burn-check").status_code == 422
    assert client.get("/v1/burn-check", params={"lat": 38.95}).status_code == 422
    assert client.get("/v1/burn-check", params={"lat": 120, "lon": -92}).status_code == 422
    monkeypatch.setattr(geocode, "geocode", lambda q: [])
    assert client.get("/v1/burn-check", params={"address": "nowhere"}).status_code == 404

    def down(q):
        raise geocode.GeocodeUnavailable()
    monkeypatch.setattr(geocode, "geocode", down)
    assert client.get("/v1/burn-check", params={"address": "x street"}).status_code == 503
    monkeypatch.setattr(sources, "county", lambda lat, lon: (None, None))
    outside = client.get("/v1/burn-check", params={"lat": 38.95, "lon": -92.33})
    assert outside.status_code == 422 and "Missouri" in outside.json()["detail"]


def test_geocode_and_disclaimer_routes(client, monkeypatch):
    monkeypatch.setattr(geocode, "geocode", lambda q: [{"label": "A", "lat": 38.95, "lon": -92.33, "source": "census"}])
    assert client.get("/v1/burn-check/geocode", params={"q": "1 Main St"}).json()["candidates"][0]["label"] == "A"
    assert client.get("/v1/burn-check/geocode", params={"q": "ab"}).status_code == 422
    body = client.get("/v1/burn-check/disclaimer").json()
    assert body["version"] == disclaimer.VERSION and "sole" in body["full"].lower() and "not responsible" in body["short"].lower()


def test_the_disclaimer_says_what_it_must():
    text = (disclaimer.SHORT + disclaimer.FULL).lower()
    for phrase in ("not responsible", "not a burn permit", "never a guarantee", "local fire department", "911", "solely responsible", "not stored"):
        assert phrase in text

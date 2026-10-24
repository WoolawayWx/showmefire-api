"""Spot-style forecast: the narrative (values only, no judgments), the official-spot block and the public route."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.rate_limit import RateLimiter
from routers import spot_forecast as router_module
from services import burn_check, spot_forecast
from services.spot_forecast import disclaimer, narrative, official

NOW = datetime(2026, 10, 24, 14, 20, tzinfo=timezone.utc)        # 9:20 AM Central (CDT)


def make_rows(start=NOW.replace(minute=0), hours=30, **over):
    base = {"temp_f": 70, "rh_pct": 50, "wind_mph": 6, "gust_mph": 10, "wind_dir_deg": 200, "mixing_height_ft": 3000,
            "transport_mph": 10, "ventilation_kt_ft": 80000, "haines": 3, "hdw": 20,
            "inversion_strength_f": 0, "inversion_top_ft": 0, "smoke_ugm3": 2}
    return [{"time_utc": start + timedelta(hours=i), **base, **over} for i in range(hours)]


def fake_sources(rows=None, ban=None, alerts=(), fresh=True, fips="29019", outlook_none=False):
    outlook = None if outlook_none else {"cycle": "c1", "cycle_time_utc": "2026-10-24T12:00:00Z", "rows": make_rows() if rows is None else rows}
    return SimpleNamespace(
        county=lambda lat, lon: (fips, "Boone") if fips else (None, None),
        active_burn_ban=lambda f, n: ban, fire_weather_alerts=lambda f, n, m: (list(alerts), fresh),
        hourly=lambda lat, lon, n, m: outlook)


def run(src=None, **kw):
    return spot_forecast.assess(38.95, -92.33, now=NOW, src=src or fake_sources(), office_lookup=lambda a, b: "LSX", **kw)


# --- narrative -----------------------------------------------------------------------------------

def test_compass_and_ventilation():
    assert narrative.compass(0) == "N" and narrative.compass(200) == "SSW" and narrative.compass(359) == "N"
    # The NWS Springfield categories shared with the HRRR hourly product (services/fire_weather/indices.py).
    cases = {39999: "Poor", 40000: "Fair", 59999: "Fair", 60000: "Good", 100000: "Very Good", 150000: "Excellent"}
    assert {k: narrative.ventilation_category(k) for k in cases} == cases
    assert narrative.ventilation_category(None) is None and narrative.ventilation_category(float("nan")) is None
    assert narrative.haines_label(5) == "moderate" and narrative.haines_label(6) == "high" and narrative.haines_label(None) is None


def test_wind_shift_found_and_light_wind_ignored():
    rows = make_rows(hours=4)
    rows[2]["wind_dir_deg"] = rows[3]["wind_dir_deg"] = 320
    assert [s["to"] for s in narrative.wind_shifts(rows)] == ["NW"]
    for r in rows:
        r["wind_mph"] = 3
    assert narrative.wind_shifts(rows) == []


def test_periods_follow_local_time_and_read_like_a_spot():
    result = run()
    names = [p["name"] for p in result["unofficial"]["periods"]]
    assert names == ["Morning", "Afternoon", "Evening"]
    assert "Wind ↑ SSW 6 mph, gusts to 10 mph." in result["unofficial"]["periods"][1]["text"]


def test_arrow_points_where_the_wind_is_going():
    # Directions are where the wind blows FROM, so a north wind is a down arrow and a west wind points east.
    assert [narrative.arrow(d) for d in (0, 90, 180, 270, 45, 225)] == ["↓", "←", "↑", "→", "↙", "↗"]
    assert narrative.arrow(None) is None and narrative.arrow(float("nan")) is None


def test_wind_arrow_in_hours_and_period_text():
    result = run()
    assert result["unofficial"]["hours"][0]["wind_arrow"] == "↑"          # from 200 deg (SSW) blows toward 20 deg (NNE), nearest of the 8 arrows is north
    assert "Wind ↑ SSW 6 mph" in result["unofficial"]["periods"][1]["text"]


def test_wind_shift_text_has_arrows():
    rows = make_rows(hours=30)
    for r in rows:
        if r["time_utc"].astimezone(narrative.LOCAL_TZ).hour >= 15:
            r["wind_dir_deg"] = 320
    text = " ".join(p["text"] for p in run(src=fake_sources(rows=rows))["unofficial"]["periods"])
    assert "Wind shifts ↑ SSW to ↘ NW around 3 PM." in text


def test_period_text_includes_haines_and_hdw():
    text = run(src=fake_sources(rows=make_rows(haines=5, hdw=120)))["unofficial"]["periods"][1]["text"]
    assert "Haines 5 (moderate)." in text and "Hot-Dry-Windy index up to 120." in text


def test_tomorrow_window_is_next_local_day():
    hours = run(window="tomorrow")["unofficial"]["hours"]
    assert hours and hours[0]["hour_label"] == "6 AM"
    assert all(datetime.fromisoformat(h["time_utc"].replace("Z", "+00:00")).astimezone(narrative.LOCAL_TZ).day == 25 for h in hours)


# --- the official-spot push ----------------------------------------------------------------------

def test_official_block_is_always_the_same_and_says_who_can_request():
    for src in (fake_sources(), fake_sources(alerts=[{"event": "Red Flag Warning"}]), fake_sources(rows=make_rows(wind_mph=30, hdw=300))):
        block = run(src=src)["official_spot"]
        assert "priority" not in block and "why" not in block
        assert block["office"] == {"code": "LSX", "name": "St. Louis", "request_page": "https://www.weather.gov/lsx/fire"}
        assert "government agencies" in block["who_can_request"] and "private individuals" in block["who_can_request"]
        assert block["if_you_are_not_an_agency"]


def test_no_judgments_in_the_output():
    result = run(src=fake_sources(rows=make_rows(wind_mph=30, gust_mph=45, rh_pct=10, hdw=300, haines=6)))
    assert "hints" not in result["unofficial"] and "fire_danger" not in result["unofficial"]["hours"][0]


def test_no_forecast_is_reported():
    result = run(src=fake_sources(outlook_none=True))
    assert result["unofficial"]["available"] is False and result["unofficial"]["periods"] == []
    assert any("not available" in n for n in result["notes"])


def test_unknown_office_lists_all_offices_and_never_invents_a_link():
    result = spot_forecast.assess(38.95, -92.33, now=NOW, src=fake_sources(), office_lookup=lambda a, b: None)
    block = result["official_spot"]
    assert block["office"] is None and block["office_note"]
    assert {o["code"] for o in block["all_offices"]} == {"EAX", "SGF", "LSX", "PAH", "DVN"}
    out_of_area = spot_forecast.assess(38.95, -92.33, now=NOW, src=fake_sources(), office_lookup=lambda a, b: "TSA")
    assert out_of_area["official_spot"]["office"]["request_page"] is None


def test_always_labelled_unofficial_with_disclaimer():
    result = run()
    assert result["unofficial"]["label"] == disclaimer.TITLE and "Not a National Weather Service spot forecast" in disclaimer.TITLE
    assert result["disclaimer"]["version"] == disclaimer.VERSION


def test_outside_missouri_raises():
    with pytest.raises(burn_check.OutsideMissouri):
        run(src=fake_sources(fips=None))


def test_failed_alert_and_ban_lookups_are_reported_not_hidden():
    def boom(*a):
        raise RuntimeError("down")
    src = fake_sources()
    src.active_burn_ban, src.fire_weather_alerts = boom, boom
    notes = " ".join(run(src=src)["notes"])
    assert "Burn bans could not be checked" in notes and "alerts could not be checked" in notes


# --- office lookup -------------------------------------------------------------------------------

def test_office_lookup_caches_and_survives_failure():
    official._cache.clear()
    calls = []
    def get(url):
        calls.append(url)
        return {"properties": {"cwa": "eax"}}
    assert official.office_for_point(39.1, -94.5, get, now=1000.0) == "EAX"
    assert official.office_for_point(39.1, -94.5, get, now=2000.0) == "EAX" and len(calls) == 1
    official._cache.clear()
    def bad(url):
        raise RuntimeError("down")
    assert official.office_for_point(39.1, -94.5, bad) is None and not official._cache


# --- route ---------------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(router_module, "forecast_limiter", RateLimiter("spot_forecast_test", 1000))
    monkeypatch.setattr(router_module.spot_forecast, "assess", lambda lat, lon, **kw: {"success": True, "kw": kw, "lat": lat})
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app)


def test_route_validates_and_passes_options(client):
    assert client.get("/v1/spot-forecast").status_code == 422
    assert client.get("/v1/spot-forecast?lat=38").status_code == 422
    assert client.get("/v1/spot-forecast?lat=38&lon=-92&window=nextweek").status_code == 422
    ok = client.get("/v1/spot-forecast?lat=38&lon=-92&window=tomorrow")
    assert ok.status_code == 200 and ok.json()["kw"]["window"] == "tomorrow"
    assert ok.headers["cache-control"] == "no-store" and "terms-of-service" in ok.headers["link"]
    assert client.get("/v1/spot-forecast/disclaimer").json()["title"] == disclaimer.TITLE

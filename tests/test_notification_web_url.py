import pytest
from pydantic import ValidationError

from routers.admin_notifications import ManualNotification


def make(**extra):
    return ManualNotification(channel="general", title="Newsletter", body="Fall issue is out", **extra)


def test_web_url_accepts_own_site_pages():
    assert make(web_url="https://showmefire.org/comms/newsletters/7").web_url == "https://showmefire.org/comms/newsletters/7"
    assert make(web_url="https://www.showmefire.org/help").web_url == "https://www.showmefire.org/help"


def test_web_url_blank_or_missing_is_none():
    assert make().web_url is None
    assert make(web_url="  ").web_url is None


@pytest.mark.parametrize("bad", [
    "http://showmefire.org/x",
    "https://showmefire.org.evil.com/x",
    "https://evil.com/?u=showmefire.org",
    "https://showmefire.org@evil.com/x",
    "https://user:pw@showmefire.org/x",
    "https://showmefire.org:8443/x",
    "javascript:alert(1)",
    "//showmefire.org/x",
    "/forecasts",
    "https://[::1/x",
])
def test_web_url_rejects_everything_else(bad):
    with pytest.raises(ValidationError):
        make(web_url=bad)


def test_audience_preview_needs_no_message_text():
    from routers.admin_notifications import AudienceRequest
    request = AudienceRequest(channel="forecast", title="", body="", url="/", web_url="", countyFips=[])
    assert request.channel == "forecast" and request.countyFips == []
    with pytest.raises(ValidationError):
        AudienceRequest(channel="nope")


def test_testers_only_audience(tmp_path, monkeypatch):
    from core import database
    from services import mobile_push

    monkeypatch.setattr(database, "get_db_path", lambda: tmp_path / "t.db")
    monkeypatch.setattr(mobile_push, "get_db_path", lambda: tmp_path / "t.db")
    database.init_database()

    def register(n, channel):
        mobile_push.upsert_subscription(
            installation_id=f"00000000-0000-0000-0000-00000000000{n}", expo_push_token=f"ExponentPushToken[tok{n}aaaaaaaaaaaa]",
            platform="ios", app_version="1.2.0", forecast=True, sitrep=False, fire_weather=False, fire_detection=False,
            county_fips=[], update_channel=channel,
        )

    register(1, "preview")
    register(2, "production")
    assert len(mobile_push._eligible_subscriptions("forecast", [], all_counties=True)) == 2
    only = mobile_push._eligible_subscriptions("forecast", [], all_counties=True, testers_only=True)
    assert [item["installation_id"][-1] for item in only] == ["1"]
    register(1, "production")  # switch turned off
    assert mobile_push._eligible_subscriptions("forecast", [], all_counties=True, testers_only=True) == []

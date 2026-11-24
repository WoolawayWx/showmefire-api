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

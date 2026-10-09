from datetime import datetime, timedelta, timezone

from services.synoptic import flatten_station_data, observation_is_usable

NOW = datetime(2026, 10, 9, 21, 0, tzinfo=timezone.utc)


def _iso(delta):
    return (NOW - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_old_reading_rejected():
    assert not observation_is_usable("fuel_moisture", 22.3, "2015-03-11T17:22:00Z", now=NOW)
    assert observation_is_usable("fuel_moisture", 6.1, _iso(timedelta(minutes=40)), now=NOW)


def test_implausible_values_rejected():
    assert not observation_is_usable("relative_humidity", 240, _iso(timedelta(minutes=5)), now=NOW)
    assert not observation_is_usable("air_temp", -9999, _iso(timedelta(minutes=5)), now=NOW)
    assert not observation_is_usable("fuel_moisture", None, _iso(timedelta(minutes=5)), now=NOW)


def test_flatten_drops_stale_sensor_but_keeps_station():
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    station = flatten_station_data({
        "STID": "BGSM7", "NAME": "BIG SPRING", "LATITUDE": "36.97", "LONGITUDE": "-91.0",
        "OBSERVATIONS": {
            "air_temp_value_1": {"value": 90.0, "date_time": fresh},
            "fuel_moisture_value_1": {"value": 22.3, "date_time": "2015-03-11T17:22:00Z"},
        },
    }, {})
    assert station["observations"]["air_temp"]["value"] == 90.0
    assert "fuel_moisture" not in station["observations"]

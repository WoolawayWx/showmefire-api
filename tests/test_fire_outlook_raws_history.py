"""The outlook's RAWS anchor reads the station history the API records, causally."""
from datetime import datetime, timedelta, timezone

import numpy as np

from maps import station_history
from services.fire_outlook import conditioning, raws_history

CYCLE = datetime(2026, 10, 31, 0, tzinfo=timezone.utc)


def _station(stid, lat, lon, fm):
    return {"stid": stid, "latitude": lat, "longitude": lon,
            "observations": {"fuel_moisture": {"value": fm, "time": "2026-10-30T23:00:00Z"}}}


def _record(db, stations, when):
    station_history.record_observations(
        stations, {"fuel_moisture_pct": lambda s: s["observations"]["fuel_moisture"]["value"]},
        run_time_utc=when, db_path=db)
    station_history.record_station_locations(stations, run_time_utc=when, db_path=db)


def test_returns_newest_reading_at_or_before_the_cycle(tmp_path):
    db = tmp_path / "h.sqlite"
    _record(db, [_station("A", 38.0, -92.0, 10.0)], CYCLE - timedelta(minutes=50))
    _record(db, [_station("A", 38.0, -92.0, 12.0)], CYCLE - timedelta(minutes=5))
    payload = raws_history.fuel_moisture_payload(CYCLE, db_path=db)
    [station] = payload["stations"]
    assert station["stid"] == "A" and station["state"] == ""
    assert station["observations"]["fuel_moisture"]["value"] == 12.0
    assert (station["latitude"], station["longitude"]) == (38.0, -92.0)


def test_nothing_after_the_cycle_is_used(tmp_path):
    db = tmp_path / "h.sqlite"
    _record(db, [_station("A", 38.0, -92.0, 10.0)], CYCLE - timedelta(minutes=20))
    _record(db, [_station("A", 38.0, -92.0, 99.0)], CYCLE + timedelta(minutes=2))   # a later reading must not leak in
    _record(db, [_station("B", 39.0, -93.0, 8.0)], CYCLE + timedelta(minutes=2))    # a station seen only after the cycle
    payload = raws_history.fuel_moisture_payload(CYCLE, db_path=db)
    assert [(s["stid"], s["observations"]["fuel_moisture"]["value"]) for s in payload["stations"]] == [("A", 10.0)]


def test_readings_older_than_the_window_are_dropped(tmp_path):
    db = tmp_path / "h.sqlite"
    _record(db, [_station("OLD", 38.0, -92.0, 10.0)], CYCLE - timedelta(hours=5))
    assert raws_history.fuel_moisture_payload(CYCLE, db_path=db) == {"stations": []}


def test_stations_without_a_recorded_location_are_skipped(tmp_path):
    db = tmp_path / "h.sqlite"
    station_history.record_observations(
        [_station("A", 38.0, -92.0, 10.0)], {"fuel_moisture_pct": lambda s: 10.0},
        run_time_utc=CYCLE - timedelta(minutes=10), db_path=db)
    assert raws_history.fuel_moisture_payload(CYCLE, db_path=db) == {"stations": []}


def test_whole_second_timestamps_are_ordered_correctly(tmp_path):
    db = tmp_path / "h.sqlite"
    _record(db, [_station("A", 38.0, -92.0, 10.0)], CYCLE - timedelta(minutes=30))                  # no fraction
    _record(db, [_station("A", 38.0, -92.0, 11.0)], (CYCLE - timedelta(minutes=10)).replace(microsecond=250000))
    assert raws_history.fuel_moisture_payload(CYCLE, db_path=db)["stations"][0]["observations"]["fuel_moisture"]["value"] == 11.0


def test_missing_database_means_no_anchor_not_an_error(tmp_path):
    assert raws_history.fuel_moisture_payload(CYCLE, db_path=tmp_path / "nope.sqlite") == {"stations": []}


def test_payload_anchors_conditioning_with_three_stations(tmp_path):
    db = tmp_path / "h.sqlite"
    stations = [_station("A", 37.5, -93.0, 12.0), _station("B", 38.0, -92.0, 14.0), _station("C", 38.5, -91.0, 13.0)]
    _record(db, stations, CYCLE - timedelta(minutes=15))
    payload = raws_history.fuel_moisture_payload(CYCLE, db_path=db)
    points = conditioning.causal_raws_points(payload, CYCLE, states=None)
    assert len(points) == 3 and all(p["age_minutes"] <= conditioning.MAX_RAW_AGE_MINUTES for p in points)

    lat, lon = np.meshgrid(np.linspace(37.0, 39.0, 5), np.linspace(-94.0, -90.0, 6), indexing="ij")
    result = {"fm1_pct": np.full(lat.shape, 8.0), "fm10_pct": np.full(lat.shape, 6.0), "fm100_pct": np.full(lat.shape, 10.0)}
    state, quality = conditioning.condition_initial_state(
        CYCLE, lat, lon, payload, raws_states=None, condition=lambda *a, **k: result)
    assert quality["level"] == conditioning.LEVEL_ANCHORED and quality["raws_station_count"] == 3
    assert float(state["fm10"].mean()) > 6.0   # RAWS (12-14) pulls the too-dry replay up

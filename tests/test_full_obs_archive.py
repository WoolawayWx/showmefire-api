import json
from datetime import datetime

from services import full_obs_archive as fo


def test_central_day_bounds_are_utc_with_dst():
    start, end = fo.central_day_bounds_utc("2026-10-10")        # CDT, UTC-5
    assert start == datetime(2026, 10, 10, 5) and end == datetime(2026, 10, 11, 5)
    start, end = fo.central_day_bounds_utc("2026-12-10")        # CST, UTC-6
    assert start == datetime(2026, 12, 10, 6)


def test_capture_writes_file_with_all_networks_and_is_idempotent(tmp_path, monkeypatch):
    seen = {}

    async def fake_fetch(states, networks, start_time, end_time):
        seen.update(states=states, networks=networks, start=start_time, end=end_time)
        return {"STATION": [{"STID": "A"}, {"STID": "B"}]}

    path = fo.capture_full_observations("2026-10-10", directory=tmp_path, fetch=fake_fetch)
    assert path.name == "raw_data_full_20261010.json"
    assert seen["networks"] == [1, 2, 156, 65] and seen["states"] == ["MO"]
    data = json.loads(path.read_text())
    assert data["_archive"]["stations"] == 2 and data["_archive"]["day_central"] == "2026-10-10"
    seen.clear()
    assert fo.capture_full_observations("2026-10-10", directory=tmp_path, fetch=fake_fetch) == path
    assert seen == {}          # not fetched again


def test_empty_response_saves_nothing(tmp_path):
    async def empty(**kwargs):
        return {"STATION": []}

    assert fo.capture_full_observations("2026-10-10", directory=tmp_path, fetch=empty) is None
    assert not list(tmp_path.glob("*.json"))

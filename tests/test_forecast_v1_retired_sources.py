"""FV3-HIRES (HiResW FV3) is retired by NOAA: it must never be acquired or weighted, whatever the DB row says."""
import json
import sqlite3

import pytest

from core import database
from forecast_v1 import registry


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "forecast_sources.db"
    connection = sqlite3.connect(path)
    database._ensure_graphics_tables(connection.cursor())
    connection.commit()
    connection.close()
    monkeypatch.setattr(registry, "get_db_path", lambda: str(path))
    return path


def _reseed(path):
    connection = sqlite3.connect(path)
    database._ensure_graphics_tables(connection.cursor())
    connection.commit()
    connection.close()


def _force_row(path, status, weight_profile=None):
    connection = sqlite3.connect(path)
    connection.execute("UPDATE forecast_source_models SET status=?, weight_profile_json=? WHERE key='fv3hires'",
                       (status, weight_profile))
    connection.commit()
    connection.close()


def test_fresh_database_seeds_fv3hires_disabled(db_path):
    model = registry.get_model("fv3hires")
    assert model["status"] == "disabled"
    assert "RETIRED" in model["notes"]


def test_existing_shadow_row_is_migrated_to_disabled_once_and_logged(db_path):
    _force_row(db_path, "shadow")
    _reseed(db_path)
    assert registry.get_model("fv3hires")["status"] == "disabled"
    _reseed(db_path)  # idempotent: no second event
    events = sqlite3.connect(db_path).execute(
        "SELECT action, detail FROM forecast_source_model_events WHERE model_key='fv3hires'").fetchall()
    assert len(events) == 1 and events[0][0] == "disabled" and "retired" in events[0][1]


def test_nothing_is_acquired_even_if_the_row_is_shadow_or_active(db_path):
    for status in ("shadow", "active"):
        _force_row(db_path, status, json.dumps({"weight": 0.2}))
        assert registry.extra_specs() == ()


def test_retired_adapter_cannot_be_re_enabled(db_path):
    with pytest.raises(ValueError, match="retired"):
        registry.set_status("fv3hires", "shadow")
    with pytest.raises(ValueError, match="retired"):
        registry.set_status("fv3hires", "active", weight=0.1)
    assert registry.set_status("fv3hires", "disabled")["status"] == "disabled"


def test_retired_adapter_cannot_be_added(db_path):
    with pytest.raises(ValueError, match="retired"):
        registry.add_model("fv3hires_2", "FV3 again", "fv3hires")


def test_retired_model_carries_no_blend_weight(db_path):
    _force_row(db_path, "active", json.dumps({"weight": 0.3}))
    base = {"hrrr": 0.5, "rrfs": 0.5}
    assert registry.weight_overlay(base, 6) == base

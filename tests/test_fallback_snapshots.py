"""Fallback snapshots: an outage must never overwrite good data (services/fallback_snapshots.py)."""
import json
from datetime import datetime, timezone

from services import fallback_snapshots as fs

NOW = datetime(2026, 11, 16, 20, 0, tzinfo=timezone.utc)


def _run(fetch, previous=None):
    store = {}
    meta = fs.publish(lambda key, body: store.__setitem__(key, body), fetch, now=NOW, previous_meta=previous)
    return store, meta


def test_every_source_is_saved_with_a_timestamp():
    store, meta = _run(lambda path: {"path": path})
    assert set(store) == {f"fallback/{n}.json" for n in fs.SOURCES} | {"fallback/meta.json"}
    saved = json.loads(store["fallback/opsbrief.json"])
    assert saved == {"at": "2026-11-16T20:00:00Z", "data": {"path": "/api/opsbrief"}}
    assert all(s["ok"] for s in meta["sources"].values()) and meta["generated_at"] == "2026-11-16T20:00:00Z"


def test_a_failing_source_keeps_its_previous_copy_and_reports_it():
    def fetch(path):
        if path == "/api/burn-bans/active":
            raise TimeoutError("api hung")
        return {"ok": 1}

    previous = {"sources": {"burn-bans-active": {"ok": True, "at": "2026-11-16T19:00:00Z", "error": None}}}
    store, meta = _run(fetch, previous)
    assert "fallback/burn-bans-active.json" not in store              # nothing written: the old object stays in R2
    bad = meta["sources"]["burn-bans-active"]
    assert bad["ok"] is False and bad["at"] == "2026-11-16T19:00:00Z" and "TimeoutError" in bad["error"]
    assert meta["sources"]["opsbrief"]["ok"] is True                  # the others still refresh


def test_empty_and_non_json_bodies_are_not_saved():
    for bad in ({}, [], None, "<html>502</html>", 0):
        store, meta = _run(lambda path, b=bad: b)
        assert set(store) == {"fallback/meta.json"}, bad
        assert meta["generated_at"] is None


def test_the_job_is_in_the_heavy_group():
    from core.job_groups import HEAVY, JOB_GROUPS
    assert JOB_GROUPS["publish_fallback_snapshots"] == HEAVY

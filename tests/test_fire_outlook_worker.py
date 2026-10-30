"""Tests for services/fire_outlook/worker.py with fake adapters on a tiny grid."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from services.fire_outlook import assemble, driver, moisture, rules, worker
from tests.test_fire_outlook_driver import GRID, member

CYCLE = datetime(2026, 10, 29, 0, tzinfo=timezone.utc)   # member() datasets are valid from 2026-10-29T00
NOW = CYCLE + timedelta(hours=6)
GRID_DEF = {"id": "t", "crs": "EPSG:32615", "width": 4, "height": 3, "resolution_m": 3000.0,
            "west": 500_000.0, "north": 4_300_000.0}


def make_adapters(members=None, ready=True, n=4, quality=None):
    members = members if members is not None else [member()] * n
    state = moisture.initial_state(np.full(GRID, 20.0), np.full(GRID, 40.0))

    def inputs_ready(cycle):
        return ready, {"gefs_members": len(members), "available_utc": CYCLE + timedelta(hours=5, minutes=30)}

    return worker.Adapters(
        inputs_ready=inputs_ready, members=lambda cycle: iter(members), n_members=n,
        conditioning=lambda cycle: (state, quality or {"level": "conditioned"}),
        rules=lambda: rules.rule_functions(gdd_accum=None), grid=GRID_DEF,
        source_cycles=lambda cycle: {"gefs": "2026-10-29T00Z"})


def cfg(tmp_path, **kw):
    return worker.Config(root=tmp_path / "outlook", **kw)


class Recorder:
    def __init__(self, fail=0):
        self.calls, self.fail = [], fail

    def __call__(self, dataset, run_id, run_dir, summary, **kwargs):
        self.calls.append((dataset, run_id, summary, kwargs))
        if len(self.calls) <= self.fail:
            raise ConnectionError("R2 unreachable")
        return {"uploaded": 3, "pruned": []}


def run(tmp_path, adapters=None, publisher=None, now=NOW, **cfg_kw):
    return worker.run_cycle(CYCLE, adapters or make_adapters(), cfg(tmp_path, **cfg_kw),
                            publish_fn=publisher, now_fn=lambda: now)


def status(tmp_path):
    return json.loads((tmp_path / "outlook" / "status.json").read_text())


def test_latest_cycle_targets_00z_and_12z():
    assert worker.latest_cycle(datetime(2026, 10, 29, 11, 59, tzinfo=timezone.utc)).hour == 0
    assert worker.latest_cycle(datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)).hour == 12
    assert worker.latest_cycle(datetime(2026, 10, 29, 23, 30, tzinfo=timezone.utc)) == datetime(2026, 10, 29, 12, tzinfo=timezone.utc)


def test_happy_path_writes_verifies_publishes_and_reports(tmp_path):
    rec = Recorder()
    out = run(tmp_path, publisher=rec)
    assert out["ok"] and out["state"] == "complete" and out["published"]
    runs = tmp_path / "outlook" / "runs"
    assert assemble.verify_checksums(runs / "20261029T00Z")
    assert json.loads((tmp_path / "outlook" / "latest.json").read_text())["run_id"] == "20261029T00Z"
    dataset, run_id, summary, kwargs = rec.calls[0]
    assert dataset == "fire-outlook" and run_id == "20261029T00Z" and summary["id"] == run_id and summary["time_utc"]
    st = status(tmp_path)
    assert st["state"] == "complete" and st["within_4h_budget"] is True and st["source_coverage"]["gefs_members"] == 4
    assert st["cycle_age_hours"] == 6.0 and st["last_complete"]["run_id"] == "20261029T00Z"
    assert st["threads"] == 1 and st["memory_limit_gb"] == 8.0 and "runtime_s" in st
    manifest = json.loads((runs / "20261029T00Z" / "manifest.json").read_text())
    assert manifest["member_count"] == 4 and manifest["runtime"]["threads"] == 1
    assert (runs / "20261029T00Z" / "contours" / "elevated" / "w27.geojson").exists()


def test_second_check_is_idempotent_and_does_not_republish(tmp_path):
    rec = Recorder()
    run(tmp_path, publisher=rec)
    again = run(tmp_path, publisher=rec)
    assert again["state"] == "already_complete" and len(rec.calls) == 1


def test_waits_when_inputs_incomplete(tmp_path):
    rec = Recorder()
    out = run(tmp_path, make_adapters(ready=False), rec)
    assert out["ok"] and out["state"] == "waiting_for_inputs" and not rec.calls
    assert not (tmp_path / "outlook" / "runs").exists() and status(tmp_path)["state"] == "waiting"


def test_incomplete_member_fails_and_previous_run_stays_current(tmp_path):
    run(tmp_path, publisher=Recorder())
    later = CYCLE + timedelta(hours=12)
    bad = make_adapters([member()] * 3 + [member(n=50)])
    out = worker.run_cycle(later, bad, cfg(tmp_path), publish_fn=Recorder(), now_fn=lambda: later + timedelta(hours=1))
    assert not out["ok"] and out["stage"] == "calculating"
    runs = tmp_path / "outlook" / "runs"
    assert sorted(p.name for p in runs.iterdir()) == ["20261029T00Z"]          # nothing partial, no temp dir
    assert json.loads((tmp_path / "outlook" / "latest.json").read_text())["run_id"] == "20261029T00Z"
    st = status(tmp_path)
    assert st["state"] == "failed" and st["errors"][0]["type"] == "IncompleteMember"
    assert st["last_complete"]["run_id"] == "20261029T00Z"
    assert assemble.verify_checksums(runs / "20261029T00Z")


def test_too_few_members_refuses_to_publish(tmp_path):
    rec = Recorder()
    out = run(tmp_path, make_adapters([member()] * 3, n=4), rec)
    assert not out["ok"] and out["stage"] == "calculating" and "3 of 4 members" in out["error"]
    assert not rec.calls and not (tmp_path / "outlook" / "latest.json").exists()


def test_publish_failure_keeps_local_run_and_retries(tmp_path):
    rec = Recorder(fail=1)
    out = run(tmp_path, publisher=rec)
    assert not out["ok"] and out["stage"] == "publishing"
    assert status(tmp_path)["errors"][0]["type"] == "ConnectionError"
    assert assemble.verify_checksums(tmp_path / "outlook" / "runs" / "20261029T00Z")
    retry = run(tmp_path, publisher=rec)                       # next cron tick: no recompute, upload succeeds
    assert retry["ok"] and retry["published"] and len(rec.calls) == 2
    assert run(tmp_path, publisher=rec)["state"] == "already_complete"


def test_publish_disabled_still_completes_locally(tmp_path):
    out = run(tmp_path, publish=False)
    assert out["ok"] and out["published"] is False and run(tmp_path, publish=False)["state"] == "already_complete"


def test_emc_seed_quality_is_recorded_in_manifest(tmp_path):
    run(tmp_path, make_adapters(quality={"level": "emc_seed", "reason": "no history"}), Recorder())
    manifest = json.loads((tmp_path / "outlook" / "runs" / "20261029T00Z" / "manifest.json").read_text())
    assert manifest["conditioning"]["level"] == "emc_seed"


def test_budget_flag_false_when_late(tmp_path):
    run(tmp_path, publisher=Recorder(), now=CYCLE + timedelta(hours=5, minutes=30) + timedelta(hours=4, minutes=1))
    assert status(tmp_path)["within_4h_budget"] is False


def test_prune_keeps_latest_and_young_runs_and_clears_old_cache(tmp_path):
    c = cfg(tmp_path)
    for name in ("20261001T00Z", "20261020T12Z", "20261028T12Z"):
        (c.runs_dir / name).mkdir(parents=True)
    (c.runs_dir / ".tmp-partial").mkdir()
    c.root.mkdir(exist_ok=True)
    (c.root / "latest.json").write_text(json.dumps({"run_id": "20261001T00Z"}))   # old but current: kept
    c.cache_dir.mkdir()
    old, new = c.cache_dir / "old", c.cache_dir / "new"
    old.mkdir(), new.mkdir()
    ts = (NOW - timedelta(days=31)).timestamp()
    import os
    os.utime(old, (ts, ts))
    removed = worker.prune(c, NOW)
    assert sorted(removed["runs"]) == ["20261020T12Z"] and removed["cache"] == 1
    assert (c.runs_dir / "20261001T00Z").exists() and (c.runs_dir / "20261028T12Z").exists()
    assert new.exists() and not old.exists() and (c.runs_dir / ".tmp-partial").exists()


def test_config_defaults_and_env(monkeypatch, tmp_path):
    c = worker.Config.from_env(tmp_path)
    assert (c.memory_limit_gb, c.threads, c.public_keep_days, c.cache_keep_days) == (8.0, 1, 7, 30)
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_PUBLIC_KEEP_DAYS", "3")
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_MEMORY_LIMIT_GB", "4")
    c = worker.Config.from_env(tmp_path)
    assert c.public_keep_days == 3 and c.memory_limit_gb == 4.0


def test_cdn_publishing_is_opt_in_from_the_environment(monkeypatch, tmp_path):
    """The CDN dataset is public, so the staff-only pilot must never upload unless asked."""
    monkeypatch.delenv("SMF_FIRE_OUTLOOK_PUBLISH", raising=False)
    assert worker.Config.from_env(tmp_path).publish is False
    for value in ("false", "0", "no", ""):
        monkeypatch.setenv("SMF_FIRE_OUTLOOK_PUBLISH", value)
        assert worker.Config.from_env(tmp_path).publish is False
    for value in ("true", "TRUE", "1", "yes", "on"):
        monkeypatch.setenv("SMF_FIRE_OUTLOOK_PUBLISH", value)
        assert worker.Config.from_env(tmp_path).publish is True

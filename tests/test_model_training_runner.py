"""Test runner authentication, bounded snapshots, and the job lifecycle."""
import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routers import model_training as routes

ACTOR = "owner@example.com"
MODEL = "fire_escape"


def request_sync(body, token=None):
    async def stream():
        yield body

    headers = {"authorization": f"Bearer {token}"} if token else {}
    request = SimpleNamespace(headers=headers, stream=stream)
    with pytest.raises(StopIteration) as completed:
        routes.sync(request).send(None)
    return completed.value.value


def test_runner_lifecycle(monkeypatch, tmp_path):
    async def in_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(routes.store, "get_db_path", lambda: tmp_path / "training.db")
    monkeypatch.setattr(routes, "run_in_threadpool", in_thread)
    monkeypatch.setattr(routes, "verify_confirm_token", lambda token, action: ACTOR if token == action else None)
    monkeypatch.setenv("SMF_MODEL_TRAINING_REMOTE", "1")

    with pytest.raises(HTTPException) as refused:
        request_sync(b"")
    assert refused.value.status_code == 401
    registration = routes.register(routes.Register(name="Test workstation", confirm_token="training_runner_create"), ACTOR)
    runner_id, token = registration["id"], registration["token"]
    assert token not in json.dumps(routes.overview(ACTOR))
    snapshot = {
        "remote": True, "free_bytes": 1, "sources": [], "jobs": [],
        "models": [{"key": MODEL, "label": "Fire Escape", "stages": ["evaluate"], "remote": True}],
    }

    def sync():
        return request_sync(json.dumps(snapshot).encode(), token)

    with pytest.raises(HTTPException) as oversized:
        request_sync(b"x" * 512_001, token)
    assert oversized.value.status_code == 413
    assert sync()["assignment"] is None

    request = routes.Start(runner=runner_id, model=MODEL, stage="evaluate", confirm_token="wrong")
    with pytest.raises(HTTPException) as unconfirmed:
        routes.start(request, ACTOR)
    assert unconfirmed.value.status_code == 401
    request.confirm_token = f"training_start:{runner_id}:{MODEL}:evaluate"
    job_id = routes.start(request, ACTOR)["id"]

    monkeypatch.setenv("SMF_MODEL_TRAINING_REMOTE", "0")
    assert sync()["assignment"] is None
    monkeypatch.setenv("SMF_MODEL_TRAINING_REMOTE", "1")
    assert sync()["assignment"] == {"id": job_id, "model": MODEL, "stage": "evaluate"}
    assert sync()["assignment"] is None

    snapshot["jobs"] = [{"id": job_id, "target": MODEL, "started_at": time.time(),
                         "status": "running", "log": "password=example-secret"}]
    sync()
    assert "example-secret" not in json.dumps(routes.overview(ACTOR))
    routes.cancel(job_id, routes.Confirmation(confirm_token=f"training_cancel:{job_id}"), ACTOR)
    assert sync()["cancel"] == [job_id]
    snapshot["jobs"][0]["status"] = "cancelled"
    assert sync()["cancel"] == []
    assert routes.overview(ACTOR)["requests"][0]["status"] == "cancelled"

    pending = routes.start(request, ACTOR)["id"]
    routes.set_remote(routes.RemoteControl(paused=True, confirm_token="training_remote_pause"), ACTOR)
    assert sync()["assignment"] is None
    history = {row["id"]: row["status"] for row in routes.overview(ACTOR)["requests"]}
    assert history[pending] == "cancelled"

    routes.revoke(runner_id, routes.Confirmation(confirm_token=f"training_runner_revoke:{runner_id}"), ACTOR)
    with pytest.raises(HTTPException) as revoked:
        sync()
    assert revoked.value.status_code == 401

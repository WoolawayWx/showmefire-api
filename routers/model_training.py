"""Runner sync and staff controls for the model training workstation."""
import json
import os
import re
import secrets
import time
from datetime import date, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError, model_validator

from core.admin_auth.permissions import MODEL_TRAINING_RUN, MODEL_TRAINING_VIEW, require_permission
from core.rate_limit import RateLimiter
from core.logging_setup import redact
from core.security import verify_confirm_token
from services import model_training_store as store
from services import day_archive

admin = APIRouter(prefix="/api/admin/model-training", tags=["model-training"])
runner = APIRouter(prefix="/api/runner/v1", tags=["model-training-runner"])
view = require_permission(MODEL_TRAINING_VIEW)
run = require_permission(MODEL_TRAINING_RUN)


class Job(BaseModel):
    id: str = Field(max_length=32, pattern=r"^[a-f0-9]+$")
    target: str = Field(max_length=80)
    status: Literal["running", "ok", "failed", "cancelled", "lost"]
    started_at: float
    log: str = Field(max_length=4000)


class Model(BaseModel):
    key: str = Field(max_length=80)
    label: str = Field(max_length=160)
    stages: list[str] = Field(max_length=6)
    remote: bool
    scorecard: dict | None = None


class Snapshot(BaseModel):
    remote: bool
    free_bytes: int = Field(ge=0)
    models: list[Model] = Field(max_length=50)
    jobs: list[Job] = Field(max_length=40)
    sources: list[dict] = Field(max_length=50)
    transfers: list[Literal["r2", "server"]] = Field(default_factory=list, max_length=2)


class Confirmation(BaseModel):
    confirm_token: str


class Register(Confirmation):
    name: str = Field(min_length=1, max_length=80)


class RemoteControl(Confirmation):
    paused: bool


class Start(Confirmation):
    runner: str
    model: str
    stage: str | None = None
    from_stage: bool = False


Component = Literal["hrrr", "rrfs", "rtma", "ensemble", "observations", "forecasts"]


class Transfer(BaseModel):
    runner: str = Field(pattern=r"^[a-f0-9]{24}$")
    source: Literal["r2", "server"]
    start: date
    end: date
    components: list[Component] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def validate_selection(self):
        if not 0 <= (self.end - self.start).days < 31:
            raise ValueError("Choose an ordered range of up to 31 days")
        if self.source == "r2" and (not self.components or len(set(self.components)) != len(self.components)):
            raise ValueError("Select one or more distinct R2 components")
        if self.source == "server" and self.components:
            raise ValueError("Server archives are transferred as whole ZIP files")
        return self


class StartTransfer(Transfer, Confirmation):
    pass


def confirm(payload, action, actor):
    if verify_confirm_token(payload.confirm_token, action) != actor:
        raise HTTPException(401, "Confirmation required or expired")


sync_limit = RateLimiter("model_training_runner", 60)


def authenticate_runner(token):
    with store.connect() as conn:
        row = conn.execute("SELECT id FROM mt_runners WHERE digest=? AND revoked=0",
                           (store.digest(token),)).fetchone()
    if row is None:
        raise HTTPException(401, "Invalid runner credential")
    return row["id"]


def save_snapshot(runner_id, snapshot):
    with store.connect() as conn:
        return store.sync(conn, runner_id, snapshot)


@runner.post("/sync", dependencies=[Depends(sync_limit)])
async def sync(request: Request):
    auth = request.headers.get("authorization", "")
    if not re.fullmatch(r"Bearer smfrun_[A-Za-z0-9_-]{43}", auth):
        raise HTTPException(401, "Invalid runner credential")
    runner_id = await run_in_threadpool(authenticate_runner, auth[7:])
    body = bytearray()
    async for part in request.stream():
        if len(body) + len(part) > 512_000:
            raise HTTPException(413, "Runner snapshot too large")
        body.extend(part)
    try:
        snapshot = Snapshot.model_validate_json(body).model_dump()
        json.dumps(snapshot, allow_nan=False)
        for job in snapshot["jobs"]:
            job["log"] = redact(job["log"])
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, "Invalid runner snapshot") from exc
    if not remote_enabled():
        snapshot["remote"] = False
        snapshot["transfers"] = []
    result = await run_in_threadpool(save_snapshot, runner_id, snapshot)
    if result is None:
        raise HTTPException(401, "Runner revoked")
    return result


@admin.get("")
def overview(_actor: str = Depends(view)):
    with store.connect() as conn:
        return {**store.overview(conn), "remote": remote_enabled(), "paused": store.paused(conn)}


@admin.post("/runners", status_code=201)
def register(payload: Register, actor: str = Depends(run)):
    confirm(payload, "training_runner_create", actor)
    with store.connect() as conn:
        return store.create(conn, payload.name, actor)


@admin.post("/runners/{runner_id}/revoke")
def revoke(runner_id: str, payload: Confirmation, _actor: str = Depends(run)):
    confirm(payload, f"training_runner_revoke:{runner_id}", _actor)
    with store.connect() as conn:
        conn.execute("UPDATE mt_runners SET revoked=1 WHERE id=?", (runner_id,))
        conn.execute("UPDATE mt_requests SET status='cancelled' WHERE runner=? AND status='queued'", (runner_id,))
    return {"revoked": True}


def remote_enabled():
    return os.getenv("SMF_MODEL_TRAINING_REMOTE") == "1"


@admin.post("/remote")
def set_remote(payload: RemoteControl, actor: str = Depends(run)):
    confirm(payload, "training_remote_pause" if payload.paused else "training_remote_resume", actor)
    if not remote_enabled():
        raise HTTPException(409, "Remote training is disabled on the API")
    with store.connect() as conn:
        conn.execute(
            """INSERT INTO mt_control VALUES (1,?,?)
               ON CONFLICT(id) DO UPDATE SET paused=excluded.paused, actor=excluded.actor""",
            (payload.paused, actor),
        )
        if payload.paused:
            conn.execute("UPDATE mt_requests SET status='cancelled' WHERE status='queued'")
    return {"paused": payload.paused}


@admin.post("/jobs", status_code=201)
def start(payload: Start, actor: str = Depends(run)):
    if payload.from_stage and not payload.stage:
        raise HTTPException(422, "Select a stage to resume from")
    selection = f"from:{payload.stage}" if payload.from_stage else payload.stage
    confirm(payload, f"training_start:{payload.runner}:{payload.model}:{selection or 'all'}", actor)
    if not remote_enabled():
        raise HTTPException(409, "Remote training is disabled on the API")
    return queue_job(payload.runner, payload.model, selection, actor,
                     lambda snapshot: any(model["key"] == payload.model and model["remote"] and
                                          (payload.stage is None or payload.stage in model["stages"])
                                          for model in snapshot["models"]) and snapshot["remote"])


@admin.post("/transfers/estimate")
def estimate_transfer(payload: Transfer, _actor: str = Depends(view)):
    if payload.source == "server":
        return {"bytes": None, "available_days": None, "missing": []}
    from routers.day_archive_admin import _s3

    s3 = _s3()
    total = available = 0
    missing = []
    for offset in range((payload.end - payload.start).days + 1):
        day = (payload.start + timedelta(days=offset)).isoformat()
        manifest = day_archive.read_manifest(s3, day) or {}
        count = 0
        for name in payload.components:
            part = manifest.get("components", {}).get(name, {})
            if part.get("status") == "present" and part.get("key") and part.get("sha256"):
                total += part["size"]
                count += 1
            else:
                missing.append(f"{day} {name}")
        available += bool(count)
    return {"bytes": total, "available_days": available, "missing": missing}


@admin.post("/transfers", status_code=201)
def start_transfer(payload: StartTransfer, actor: str = Depends(run)):
    components = sorted(payload.components)
    selection = f"{payload.start}:{payload.end}:{','.join(components)}"
    confirm(payload, f"training_transfer:{payload.runner}:{payload.source}:{selection}", actor)
    if not remote_enabled():
        raise HTTPException(409, "Remote starts are disabled on the API")
    return queue_job(payload.runner, f"transfer:{payload.source}", selection, actor,
                     lambda snapshot: payload.source in snapshot.get("transfers", []))


def queue_job(runner_id, target, selection, actor, available):
    now = time.time()
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.expire(conn, now)
        if store.paused(conn):
            raise HTTPException(409, "Remote starts are paused")
        row = conn.execute("SELECT seen,snapshot FROM mt_runners WHERE id=? AND revoked=0", (runner_id,)).fetchone()
        if row is None or row["seen"] is None:
            raise HTTPException(409, "Runner has not reported its capabilities")
        snapshot = json.loads(row["snapshot"])
        if not available(snapshot):
            raise HTTPException(422, "Operation unavailable on this runner")
        if conn.execute(
            "SELECT 1 FROM mt_requests WHERE runner=? AND model=? AND status IN ('queued','claimed','running')",
            (runner_id, target),
        ).fetchone():
            raise HTTPException(409, "Operation already queued or running")
        job_id = secrets.token_hex(12)
        conn.execute("INSERT INTO mt_requests(id,runner,model,stage,actor,created,updated) VALUES (?,?,?,?,?,?,?)",
                     (job_id, runner_id, target, selection, actor, now, now))
    return {"id": job_id}


@admin.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, payload: Confirmation, _actor: str = Depends(run)):
    confirm(payload, f"training_cancel:{job_id}", _actor)
    with store.connect() as conn:
        updated = conn.execute(
            """UPDATE mt_requests SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END
               WHERE id=? AND status IN ('queued','claimed','running')""", (job_id,),
        )
        if not updated.rowcount:
            raise HTTPException(409, "Job is no longer active")
    return {"cancel_requested": True}

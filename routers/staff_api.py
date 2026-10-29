"""The staff API: what the app's staff section talks to (/api/staff/v1). Separate from the website's /api/admin routes on
purpose: the shipped app is a binary that lives for months, so it gets a small, versioned contract of its own, with a permission
on every route and nothing it does not need (no reporter names, contact details or network hashes).

The moderation itself is never reimplemented here. Each action calls the same function the website's route calls, so an approval
from the app and one from the website go through identical code (including the audit log, which records the actor as "mobile" or
"sandbox"). What this layer adds is the permission check, a required reason where staff must explain themselves (reject, deny,
not-a-fire), and the narrower response shapes.

Off (404) unless STAFF_API_ENABLED is set. A reviewer's requests are run against the sandbox database by the lockdown
middleware before they get here (core/admin_auth/lockdown.py), so the same routes serve both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core import database, sandbox
from core.admin_auth import permissions, users
from core.admin_auth.permissions import require_permission
from routers import admin_mobile
from services import audit, staff_push

router = APIRouter(prefix="/api/staff/v1", tags=["staff"], dependencies=[Depends(admin_mobile._require_enabled)])

VIEW = require_permission(permissions.STAFF_ACCESS)
REPORTS_VIEW = require_permission(permissions.REPORTS_VIEW)
REPORTS_MODERATE = require_permission(permissions.REPORTS_MODERATE)
BURNBANS_VIEW = require_permission(permissions.BURNBANS_VIEW)
BURNBANS_MODERATE = require_permission(permissions.BURNBANS_MODERATE)
AUDIT_VIEW = require_permission(permissions.AUDIT_VIEW)
PUSH_RECEIVE = require_permission(permissions.PUSH_RECEIVE)

HISTORY_KINDS = {"reports": "fire_event", "incidents": "fire_incident", "feedback": "incident_feedback", "burn-bans": "burn_ban"}


class Reason(BaseModel):
    """Staff must say why when they reject, deny or call something not a fire."""
    reason: str = Field(..., min_length=3, max_length=1000)


class Approval(BaseModel):
    label: Optional[Literal["fire", "controlled_burn"]] = None
    note: str = Field("", max_length=2000)


class BanConfirmation(BaseModel):
    note: str = Field("", max_length=2000)
    effective_at: Optional[str] = None
    expires_at: Optional[str] = None


class FeedbackApproval(BaseModel):
    note: str = Field("", max_length=1000)


class PushDevice(BaseModel):
    installation_id: str = Field(..., min_length=8, max_length=100)
    expo_token: str = Field(..., max_length=300)
    platform: str = Field("", max_length=20)
    app_version: str = Field("", max_length=20)


def _page(limit: int, offset: int = 0) -> tuple:
    return max(1, min(limit, 100)), max(0, offset)


def _rows(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    import sqlite3

    with sqlite3.connect(database.get_db_path(), timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def _scalar(sql: str, params: tuple = ()) -> int:
    rows = _rows(sql, params)
    return int(list(rows[0].values())[0]) if rows else 0


def _short(text: Optional[str], length: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


def _clef_line(clef: Optional[Dict[str, Any]]) -> Optional[str]:
    from ai.clef_tasks import summarize_report

    return summarize_report(clef)


# ----------------------------------------------------------------------------------------------- who am I, counts

@router.get("/me")
def me(email: str = Depends(VIEW)):
    user = users.get_by_email(email)
    return {"success": True, "user": admin_mobile.account_view(user), "sandbox": sandbox.is_sandbox() or users.is_sandbox_role(user["role"])}


@router.get("/summary")
def summary(response: Response, _email: str = Depends(VIEW)):
    """What is waiting for review, for the home screen and the app badge."""
    response.headers["Cache-Control"] = "no-store"
    return {"success": True, "pending": {
        "reports": _scalar("SELECT COUNT(*) FROM fire_events WHERE source = 'user_submission' AND status = 'pending'"),
        "incidents": _scalar("""SELECT COUNT(*) FROM fire_incidents WHERE status NOT IN ('merged', 'deleted')
                                AND EXISTS (SELECT 1 FROM fire_events e WHERE e.incident_id = fire_incidents.id AND e.status = 'pending')"""),
        "feedback": _scalar("""SELECT COUNT(*) FROM fire_incident_feedback f JOIN fire_incidents i ON i.id = f.incident_id
                               WHERE f.status = 'pending' AND i.status NOT IN ('deleted', 'merged')"""),
        "burn_bans": database.count_burn_ban_submissions(status="pending"),
    }}


# ----------------------------------------------------------------------------------------------- reports

def _report_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": event["id"], "county_name": event.get("county_name"), "occurred_at": event.get("occurred_at"),
        "created_at": event.get("created_at"), "acres": event.get("acres"), "incident_id": event.get("incident_id"),
        "description": _short(event.get("description"), 280), "photo_count": event.get("photo_count", 0),
        "status": event.get("status"),
    }


@router.get("/reports/pending")
def pending_reports(response: Response, limit: int = 50, offset: int = 0, _email: str = Depends(REPORTS_VIEW)):
    response.headers["Cache-Control"] = "no-store"
    limit, offset = _page(limit, offset)
    events = database.list_fire_events(status="pending", source="user_submission", limit=limit, offset=offset, admin=True)
    return {"success": True, "reports": [_report_summary(event) for event in events], "count": len(events)}


@router.get("/reports/{event_id}")
def report(event_id: int, response: Response, _email: str = Depends(REPORTS_VIEW)):
    response.headers["Cache-Control"] = "no-store"
    event = database.get_fire_event(event_id, admin=True)
    if not event or event.get("source") != "user_submission":
        raise HTTPException(status_code=404, detail="Report not found")
    features = (event.get("extracted_features") or {}).get("features") or None
    return {"success": True, "report": {
        **_report_summary(event),
        "description": event.get("description") or "", "anything_unusual": event.get("out_of_ordinary") or "",
        "latitude": event.get("latitude"), "longitude": event.get("longitude"), "address_text": event.get("address_text") or "",
        "reporter_relationship": event.get("reporter_relationship"), "reported_fire_status": event.get("reported_fire_status"),
        "reported_structures_threatened": event.get("reported_structures_threatened"),
        "reported_responders_on_scene": event.get("reported_responders_on_scene"),
        "cause_category": event.get("cause_category"), "verification_tier": event.get("verification_tier"),
        "media": [{"id": m["id"], "content_type": m["content_type"], "kind": m.get("kind", "photo")}
                  for m in event.get("media", []) if m.get("kind", "photo") == "photo"],
        "text_features": features,
        "clef": event.get("clef"), "clef_summary": _clef_line(event.get("clef")),
        "history": audit.history("fire_event", event_id),
    }}


def _media_dir() -> Path:
    from routers import fires

    return sandbox.sandbox_media_dir() if sandbox.is_sandbox() else fires.MEDIA_DIR


@router.get("/reports/{event_id}/media/{media_id}")
def report_media(event_id: int, media_id: int, _email: str = Depends(REPORTS_VIEW)):
    event = database.get_fire_event(event_id, admin=True)
    item = next((m for m in (event or {}).get("media", []) if m["id"] == media_id and m.get("kind", "photo") == "photo"), None)
    if not event or event.get("source") != "user_submission" or not item:
        raise HTTPException(status_code=404, detail="Media not found")
    target = _media_dir() / Path(item["stored_filename"]).name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Media file unavailable")
    return FileResponse(target, media_type=item["content_type"], headers={"Cache-Control": "private, no-store"})


@router.post("/reports/{event_id}/approve")
def approve_report(event_id: int, body: Approval, email: str = Depends(REPORTS_MODERATE)):
    from routers import fires

    return fires.admin_approve_fire_report(event_id, fires.FireReportModeration(label=body.label, moderator_note=body.note), actor=email)


@router.post("/reports/{event_id}/reject")
def reject_report(event_id: int, body: Reason, email: str = Depends(REPORTS_MODERATE)):
    from routers import fires

    return fires.admin_reject_fire_report(event_id, fires.FireReportRejection(reason=body.reason), actor=email)


# ----------------------------------------------------------------------------------------------- incidents

@router.get("/incidents/pending")
def pending_incidents(response: Response, limit: int = 50, _email: str = Depends(REPORTS_VIEW)):
    response.headers["Cache-Control"] = "no-store"
    limit, _ = _page(limit)
    incidents = database.list_incidents_with_pending_members(limit=limit)
    return {"success": True, "count": len(incidents), "incidents": [{
        "id": i["id"], "public_slug": i.get("public_slug"), "county_name": i.get("county_name"),
        "first_detected_at": i.get("first_detected_at"), "last_detected_at": i.get("last_detected_at"),
        "pending_count": i.get("pending_count"), "status_counts": i.get("status_counts"),
        "has_additional_info": bool(i.get("has_additional_info")),
        "sources": sorted({m["source"] for m in i.get("members", [])}),
        "feedback": {key: value for key, value in i.items() if key.startswith("feedback")},
    } for i in incidents]}


@router.get("/incidents/{incident_id}")
def incident(incident_id: int, response: Response, _email: str = Depends(REPORTS_VIEW)):
    response.headers["Cache-Control"] = "no-store"
    row = database.get_fire_incident(incident_id)
    if not row or row.get("status") in ("deleted", "merged"):
        raise HTTPException(status_code=404, detail="Incident not found")
    members = database.list_fire_incident_members(incident_id)
    from ai.clef import get_assessment
    from ai.clef_tasks import summarize_incident

    clef = get_assessment("fire_incident", incident_id, "incident_type")
    return {"success": True, "incident": {
        "id": incident_id, "public_slug": row.get("public_slug"), "county_name": row.get("county_name"),
        "first_detected_at": row.get("first_detected_at"), "last_detected_at": row.get("last_detected_at"),
        "review_decision": row.get("review_decision"), "review_label": row.get("review_label"),
        "detections": [{k: m.get(k) for k in ("id", "source", "status", "latitude", "longitude", "occurred_at", "confidence", "frp")}
                       for m in members],
        "reports": [_report_summary(e) for e in (database.get_fire_event(m["id"], admin=True) for m in members
                                                  if m.get("source") == "user_submission") if e],
        "feedback": [{k: f.get(k) for k in ("id", "classification", "note", "status", "created_at", "submission_source")}
                     for f in database.list_fire_incident_feedback(incident_id)],
        "clef": clef, "clef_summary": summarize_incident(clef),
        "history": audit.history("fire_incident", incident_id),
    }}


@router.post("/incidents/{incident_id}/approve")
def approve_incident(incident_id: int, body: Approval, email: str = Depends(REPORTS_MODERATE)):
    from routers import fires

    return fires.admin_approve_all_incident_reports(incident_id, fires.FireReportModeration(label=body.label, moderator_note=body.note), actor=email)


@router.post("/incidents/{incident_id}/reject")
def reject_incident(incident_id: int, body: Reason, email: str = Depends(REPORTS_MODERATE)):
    from routers import fires

    return fires.admin_reject_all_incident_reports(incident_id, fires.FireReportRejection(reason=body.reason), actor=email)


@router.post("/incidents/{incident_id}/not-a-fire")
def not_a_fire(incident_id: int, body: Reason, email: str = Depends(REPORTS_MODERATE)):
    """The website's one-click 'not a fire' (label, reject, delete after seven days), with the reason kept in the audit log."""
    from services.fire_labeling import apply_feedback_label

    if not database.get_fire_incident(incident_id):
        raise HTTPException(status_code=404, detail="Incident not found")
    apply_feedback_label(incident_id, "not_a_fire", reviewed_by=email)
    decision = database.set_incident_review_decision(incident_id, "rejected", email, reason="not_a_fire", note=body.reason)
    due = database.schedule_incident_deletion(incident_id, days=7, actor=email, reason=body.reason)
    return {"success": True, "incident_decision": decision, "delete_after": due}


# ----------------------------------------------------------------------------------------------- incident feedback

@router.get("/feedback")
def pending_feedback(response: Response, limit: int = 50, _email: str = Depends(REPORTS_VIEW)):
    response.headers["Cache-Control"] = "no-store"
    limit, _ = _page(limit)
    items = database.list_pending_fire_incident_feedback(limit=limit)
    return {"success": True, "count": len(items), "feedback": [{
        "id": f["id"], "incident_id": f["incident_id"], "classification": f["classification"], "note": f.get("note") or "",
        "created_at": f.get("created_at"), "submission_source": f.get("submission_source"), "county_name": f.get("county_name"),
        "public_slug": f.get("public_slug"), "sources": f.get("sources"),
    } for f in items]}


@router.post("/feedback/{feedback_id}/approve")
def approve_feedback(feedback_id: int, body: FeedbackApproval, email: str = Depends(REPORTS_MODERATE)):
    """Approve a public classification: the website's flow (label the incident, decide it), with the reviewer's note recorded."""
    from services.fire_labeling import apply_feedback_label

    feedback = database.set_fire_incident_feedback_status(feedback_id, "approved", reviewed_by=email, note=body.note)
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    labeled = apply_feedback_label(feedback["incident_id"], feedback["classification"], reviewed_by=email)
    decision = database.set_incident_review_decision(feedback["incident_id"], "approved", email, reason=f"incident_feedback:{feedback_id}",
                                                     note=body.note)
    return {"success": True, "labeled_event_count": len(labeled), "incident_decision": decision}


@router.post("/feedback/{feedback_id}/reject")
def reject_feedback(feedback_id: int, body: Reason, email: str = Depends(REPORTS_MODERATE)):
    """Reject a public classification. As on the website, this also rejects the incident it was about."""
    feedback = database.set_fire_incident_feedback_status(feedback_id, "rejected", reviewed_by=email, note=body.reason)
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    decision = database.set_incident_review_decision(feedback["incident_id"], "rejected", email, reason=f"incident_feedback:{feedback_id}",
                                                     note=body.reason)
    return {"success": True, "incident_decision": decision}


# ----------------------------------------------------------------------------------------------- burn bans

def _ban_summary(ban: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": ban["id"], "status": ban.get("status"), "county_name": ban.get("county_name"), "request_type": ban.get("request_type"),
        "effective_at": ban.get("effective_at"), "expires_at": ban.get("expires_at"), "proof_url": ban.get("proof_url") or "",
        "has_proof_file": bool(ban.get("proof_stored_filename")), "submitter_comment": _short(ban.get("submitter_comment"), 400),
        "public_note": ban.get("public_note") or "", "created_at": ban.get("created_at"),
    }


@router.get("/burn-bans")
def burn_bans(response: Response, status: str = "pending", limit: int = 50, offset: int = 0, _email: str = Depends(BURNBANS_VIEW)):
    response.headers["Cache-Control"] = "no-store"
    if status not in ("pending", "confirmed", "denied", "expired"):
        raise HTTPException(status_code=400, detail="Unknown status")
    limit, offset = _page(limit, offset)
    items = database.list_burn_ban_submissions(status=status, limit=limit, offset=offset, admin=True)
    return {"success": True, "count": len(items), "submissions": [_ban_summary(i) for i in items]}


@router.get("/burn-bans/{submission_id}")
def burn_ban(submission_id: int, response: Response, _email: str = Depends(BURNBANS_VIEW)):
    response.headers["Cache-Control"] = "no-store"
    ban = database.get_burn_ban_submission(submission_id, admin=True)
    if not ban:
        raise HTTPException(status_code=404, detail="Submission not found")
    from ai.clef import get_assessment

    return {"success": True, "submission": {
        **_ban_summary(ban), "proof_content_type": ban.get("proof_content_type") or "",
        "clef": get_assessment("burn_ban", submission_id, "burn_ban"),
        "history": audit.history("burn_ban", submission_id),
    }}


@router.get("/burn-bans/{submission_id}/proof")
def burn_ban_proof(submission_id: int, _email: str = Depends(BURNBANS_VIEW)):
    from routers import burn_bans as bans

    ban = database.get_burn_ban_submission(submission_id, admin=True)
    stored = Path((ban or {}).get("proof_stored_filename") or "").name
    if not ban or not stored:
        raise HTTPException(status_code=404, detail="No proof file")
    folder = sandbox.sandbox_root() / "proofs" if sandbox.is_sandbox() else bans.PROOF_DIR
    target = folder / stored
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Proof file unavailable")
    return FileResponse(target, media_type=ban.get("proof_content_type") or "application/octet-stream",
                        headers={"Cache-Control": "private, no-store"})


@router.post("/burn-bans/{submission_id}/confirm")
def confirm_burn_ban(submission_id: int, body: BanConfirmation, _email: str = Depends(BURNBANS_MODERATE)):
    from routers import burn_bans as bans

    return bans.admin_confirm_burn_ban(
        submission_id, bans.BurnBanModeration(moderator_note=body.note, effective_at=body.effective_at, expires_at=body.expires_at))


@router.post("/burn-bans/{submission_id}/deny")
def deny_burn_ban(submission_id: int, body: Reason, _email: str = Depends(BURNBANS_MODERATE)):
    from routers import burn_bans as bans

    return bans.admin_deny_burn_ban(submission_id, bans.BurnBanRejection(reason=body.reason))


# ----------------------------------------------------------------------------------------------- audit log

@router.get("/audit")
def audit_log(subject_type: Optional[str] = None, subject_id: Optional[str] = None, actor: Optional[str] = None,
              incident_id: Optional[int] = None, since: Optional[float] = None, until: Optional[float] = None,
              before_id: Optional[int] = None, limit: int = 50, _email: str = Depends(AUDIT_VIEW)):
    if subject_type is not None and subject_type not in audit.SUBJECT_TYPES:
        raise HTTPException(status_code=400, detail="Unknown subject type")
    entries = audit.list_entries(subject_type=subject_type, subject_id=subject_id, actor=actor, incident_id=incident_id,
                                 since=since, until=until, before_id=before_id, limit=limit)
    return {"success": True, "entries": entries, "next_before_id": entries[-1]["id"] if entries else None}


@router.get("/history/{kind}/{item_id}")
def history(kind: str, item_id: str, _email: str = Depends(AUDIT_VIEW)):
    if kind not in HISTORY_KINDS:
        raise HTTPException(status_code=404, detail="Unknown kind")
    return {"success": True, "history": audit.history(HISTORY_KINDS[kind], item_id)}


# ----------------------------------------------------------------------------------------------- staff push

@router.put("/push-device")
def register_push_device(body: PushDevice, email: str = Depends(PUSH_RECEIVE)):
    """Bind this phone's push token to the signed-in session. It stops receiving the moment the session ends or the account
    loses access (services/staff_push.py decides at send time)."""
    actor = permissions.resolve_actor()
    if actor is None or not actor.admin_id or not actor.session_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not staff_push.register_device(actor.admin_id, actor.session_id, body.installation_id, body.expo_token, body.platform, body.app_version):
        raise HTTPException(status_code=422, detail="That is not a valid push token.")
    return {"success": True}


@router.delete("/push-device")
def unregister_push_device(_email: str = Depends(VIEW)):
    actor = permissions.resolve_actor()
    if actor is not None and actor.session_id:
        staff_push.unregister_session(actor.session_id)
    return {"success": True}


@router.post("/push-test")
def push_test(_email: str = Depends(VIEW)):
    """For app-store reviewers, who receive no real pushes: the content of a sample alert, for the app to show as a local
    notification. Nothing is sent from the server. Only exists in the sandbox."""
    if not sandbox.is_sandbox():
        raise HTTPException(status_code=404, detail="Not found")
    return {"success": True, "notification": {"title": "New fire report", "body": "Phelps County", "url": "/staff/reports"}}

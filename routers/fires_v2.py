"""v2 fires API: a consolidated, clearly-named read model over the same
fire_events/fire_incidents data v1 (routers/fires.py) serves, plus the new
AI-summary admin endpoints. Additive - v1 is untouched so the current
frontend keeps working; this is a read/orchestration layer, not a new data
model.

Today's v1 spreads one incident's confidence picture across four separate
endpoints (incidents.geojson, incidents-confidence.geojson,
incident-shapes.geojson, plus per-member detection_confidence_pct) and uses
"confidence" ambiguously in three different places (detection_confidence.py,
fire_confidence.py, feedback_confidence.py). v2's /fires/incidents endpoint
returns one object per incident with clearly-named fields instead:
  detection_confidence  - per-event ML score (services/detection_confidence.py)
  incident_confidence   - cluster-level score (services/fire_confidence.py)
  feedback_priority_score - per-pending-feedback triage score
                            (services/feedback_confidence.py, renamed here
                            only - no logic change)
  weather_danger         - today's published county fire-danger context
                            (services/weather_context.py)
  land_cover_context      - fuel model / canopy cover / land-cover fractions
"""
from __future__ import annotations

import statistics
from typing import Optional

from fastapi import APIRouter, Response

from core.database import (
    get_public_fire_incident,
    list_fire_events,
    list_fire_incident_members,
    list_fire_incidents,
    list_pending_fire_incident_feedback,
)
from routers.fires import PUBLIC_API_BASE_URL, _require_admin

router = APIRouter(prefix="/api/v2", tags=["fires-v2"])


def _land_cover_context(members: list) -> dict:
    fuel_models = [m["fuel_model_fbfm40"] for m in members if m.get("fuel_model_fbfm40") is not None]
    canopy_values = [m["canopy_cover_pct"] for m in members if m.get("canopy_cover_pct") is not None]
    land_cover_strings = [m["land_cover"] for m in members if m.get("land_cover")]
    return {
        "fuel_model_fbfm40": statistics.mode(fuel_models) if fuel_models else None,
        "avg_canopy_cover_pct": round(sum(canopy_values) / len(canopy_values), 1) if canopy_values else None,
        "land_cover": land_cover_strings[-1] if land_cover_strings else None,
    }


def _incident_confidence(incident: dict, members: list) -> dict:
    from services.fire_confidence import _features, _score

    score, label, method = _score(_features(incident, members))
    return {"score": score, "label": label, "method": method}


def _weather_danger(incident: dict) -> dict:
    from services.weather_context import county_fire_danger_today

    category, probability = county_fire_danger_today(incident.get("county_fips"))
    return {"category": category, "probability": probability}


def _detection_confidence(members: list) -> Optional[int]:
    scored = [m["detection_confidence_pct"] for m in members if m.get("detection_confidence_pct") is not None]
    return round(max(scored)) if scored else None


def _incident_v2(incident: dict) -> dict:
    members = list_fire_incident_members(incident["id"])
    graphic_url = (
        f"{PUBLIC_API_BASE_URL}/images/fire-incidents/{incident['public_slug']}.png"
        if incident.get("public_slug") and incident.get("graphic_filename") else None
    )
    feedback_url = f"/fires/incident/{incident['public_slug']}" if incident.get("public_slug") else None
    return {
        "incident_id": incident["id"],
        "incident_slug": incident.get("public_slug"),
        "county_name": incident.get("county_name"),
        "first_detected_at": incident.get("first_detected_at"),
        "last_detected_at": incident.get("last_detected_at"),
        "detection_count": incident.get("detection_count"),
        "sources": incident.get("sources"),
        "shape_geojson": incident.get("shape_geojson"),
        "centroid": {"latitude": incident.get("centroid_latitude"), "longitude": incident.get("centroid_longitude")},
        "detection_confidence": _detection_confidence(members),
        "incident_confidence": _incident_confidence(incident, members),
        "weather_danger": _weather_danger(incident),
        "land_cover_context": _land_cover_context(members),
        "feedback_summary": {
            "count": incident.get("feedback_count", 0),
            "pending_count": incident.get("pending_feedback_count", 0),
            "confirmed": bool(incident.get("confirmed")),
        },
        "graphic_url": graphic_url,
        "feedback_url": feedback_url,
    }


@router.get("/fires/incidents")
def list_incidents_v2(
    response: Response,
    since: Optional[str] = None,
    until: Optional[str] = None,
    source: Optional[str] = None,
    has_feedback: Optional[bool] = None,
    confirmed_only: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
):
    """Consolidated incident read model - see module docstring. Public,
    same cache policy as v1's incident endpoints."""
    incidents = list_fire_incidents(
        since=since, until=until, source=source, has_feedback=has_feedback,
        confirmed_only=confirmed_only, limit=limit, offset=offset,
    )
    response.headers["Cache-Control"] = "public, max-age=60"
    return {"success": True, "incidents": [_incident_v2(incident) for incident in incidents], "count": len(incidents)}


@router.get("/fires/incidents/{slug}")
def get_incident_v2(slug: str):
    incident = get_public_fire_incident(slug)
    if not incident:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"success": True, "incident": _incident_v2(incident)}


@router.get("/admin/fires/queue")
def admin_queue_v2(token: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Combined moderation queue: pending reports/detections annotated with
    a renamed (not re-derived) feedback_priority_score where applicable, and
    an ai_summary read from cache (never generated here - see the dedicated
    summary endpoints below, so listing the queue never triggers billed
    Cloudflare calls on its own)."""
    from ai.report_summary import report_summary_status
    from services.feedback_confidence import score_feedback

    _require_admin(token)
    events = list_fire_events(status="pending", admin=True, limit=limit, offset=offset)
    pending_feedback = list_pending_fire_incident_feedback(limit=limit)
    for item in pending_feedback:
        item["feedback_priority_score"] = score_feedback(item["id"])
    for event in events:
        event["ai_summary"] = report_summary_status(event)
    return {
        "success": True,
        "reports": events,
        "pending_incident_feedback": pending_feedback,
        "count": len(events),
    }


@router.get("/admin/fires/reports/{event_id}/summary")
def get_report_summary_v2(event_id: int, token: Optional[str] = None):
    from ai.report_summary import report_summary_status
    from core.database import get_fire_event
    from fastapi import HTTPException

    _require_admin(token)
    event = get_fire_event(event_id, admin=True)
    if not event:
        raise HTTPException(status_code=404, detail="Fire event not found")
    return {"success": True, "summary": report_summary_status(event)}


@router.post("/admin/fires/reports/{event_id}/summary")
def generate_report_summary_v2(event_id: int, token: Optional[str] = None):
    """Generates (and caches) the AI summary for one report/detection - a
    separate, explicit call from the queue listing above so viewing the
    queue never itself triggers a billed Cloudflare call."""
    from ai.report_summary import generate_report_summary
    from core.database import get_fire_event
    from fastapi import HTTPException

    _require_admin(token)
    event = get_fire_event(event_id, admin=True)
    if not event:
        raise HTTPException(status_code=404, detail="Fire event not found")
    return {"success": True, "summary": generate_report_summary(event)}


@router.post("/admin/fires/overview")
def generate_queue_overview_v2(token: Optional[str] = None, limit: int = 20):
    """Generates a short AI overview of the current pending queue - explicit
    POST (not GET) for the same reason as the per-report summary above."""
    from ai.report_summary import generate_queue_overview

    _require_admin(token)
    events = list_fire_events(status="pending", admin=True, limit=limit)
    return {"success": True, "overview": generate_queue_overview(events)}

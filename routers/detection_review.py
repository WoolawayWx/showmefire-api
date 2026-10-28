"""Admin endpoints for reviewing satellite detections as vegetation fire / not / cannot determine.

Reviewers never receive confidence scores. Outcomes only feed training and the shadow comparison; nothing here changes
what the public sees. See services/detection_review.py.
"""
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from core.security import require_admin
from services import detection_review as review

router = APIRouter(tags=["detection-review"])


class DetectionOutcome(BaseModel):
    outcome: Literal["vegetation_fire", "not_vegetation_fire", "cannot_determine"]
    reason: str = ""
    evidence: str = Field(default="", max_length=2000)


@router.get("/api/admin/fires/detection-review/next")
def next_detection(response: Response, stream: Literal["random", "enriched"] = "random",
                   second: bool = Query(False, description="double-check one another reviewer already judged"),
                   actor: str = Depends(require_admin)):
    response.headers["Cache-Control"] = "no-store"
    item = review.second_review(actor) if second else review.next_detection(stream, actor)
    return {"success": True, "item": item, "stream": stream, "second_review": second}


@router.post("/api/admin/fires/detection-review/{event_id}")
def submit_detection_outcome(event_id: int, payload: DetectionOutcome, actor: str = Depends(require_admin)):
    try:
        result = review.submit_outcome(event_id, payload.outcome, actor, payload.reason, payload.evidence)
    except review.ReviewError as exc:
        raise HTTPException(status_code=422 if "found" not in str(exc) else 404, detail=str(exc))
    return {"success": True, **result}


@router.get("/api/admin/fires/detection-review/stats")
def detection_review_stats(response: Response, _admin: str = Depends(require_admin)):
    response.headers["Cache-Control"] = "no-store"
    return {"success": True, **review.stats()}


@router.get("/api/admin/fires/detection-review/shadow-summary")
def detection_shadow_summary(response: Response, _admin: str = Depends(require_admin)):
    """How v1 and the candidate score against reviewed detections. Informational only."""
    response.headers["Cache-Control"] = "no-store"
    return {"success": True, **review.shadow_summary()}

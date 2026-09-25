"""Optional Cloudflare Workers AI summaries for the admin fire-report/
detection moderation queue - same fingerprint-cache-regenerate shape as
ai/verification_summary.py + services/verification_feedback.py, applied to
a fire report/incident instead of a forecast verification report.

Two summaries:
  generate_report_summary(event)   - a 2-3 sentence summary of one pending
                                      report/incident, for the admin queue.
  generate_queue_overview(items)   - a short paragraph flagging which
                                      pending items look most worth
                                      prioritizing.

Both degrade to a `status` of "not_generated" (Cloudflare configured but no
cached summary yet) or "not_configured" (no Cloudflare credentials) rather
than failing the request - an admin queue must render with or without AI.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from ai.cloudflare import CloudflareAIClient
from core.config import DATA_DIR

logger = logging.getLogger(__name__)

SUMMARY_DIR = Path(os.getenv("SMF_FIRE_AI_SUMMARY_DIR", str(DATA_DIR / "fire_report_ai_summaries")))

_EVIDENCE_KEYS = (
    "id", "source", "status", "verification_tier", "cause_category", "description",
    "out_of_ordinary", "acres", "fuel_types", "frp", "confidence", "land_cover",
    "detection_confidence_pct", "county_name", "occurred_at", "nearby_reports",
)


def _evidence(event: Dict[str, Any]) -> Dict[str, Any]:
    return {key: event.get(key) for key in _EVIDENCE_KEYS if key in event}


def _fingerprint(evidence: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()


def _cache_path(event_id: int) -> Path:
    return SUMMARY_DIR / f"{event_id}.json"


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def report_summary_status(event: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only: returns the cached summary if still fresh (evidence
    fingerprint unchanged), else a status telling the caller whether it's
    worth calling generate_report_summary()."""
    evidence = _evidence(event)
    fingerprint = _fingerprint(evidence)
    path = _cache_path(event["id"])
    try:
        cached = json.loads(path.read_text())
        if cached.get("evidence_fingerprint") == fingerprint and cached.get("summary"):
            return {**cached, "status": "ready"}
    except (OSError, ValueError):
        pass
    configured = CloudflareAIClient().configured
    return {
        "event_id": event["id"],
        "summary": None,
        "status": "not_generated" if configured else "not_configured",
    }


def generate_report_summary(event: Dict[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    """Generate (and cache) a 2-3 sentence plain-language summary of a single
    pending report/incident for the admin queue."""
    client = CloudflareAIClient()
    evidence = _evidence(event)
    fingerprint = _fingerprint(evidence)
    if not client.configured:
        if strict:
            raise RuntimeError("Cloudflare Workers AI credentials are not configured")
        return {"event_id": event["id"], "summary": None, "status": "not_configured"}

    prompt = (
        "You are summarizing one pending fire report/detection for a wildland fire department's admin "
        "moderation queue. Use only the supplied fields - do not invent details. In 2-3 plain-language "
        "sentences, describe what was reported/detected, anything that stands out (high confidence, "
        "corroborating nearby reports, unusual land cover), and whether it looks worth prioritizing for "
        "review or public feedback.\n\n"
        f"Report/detection data:\n{json.dumps(evidence, default=str)}"
    )
    try:
        text = client.generate_text(prompt, max_tokens=220)
    except Exception:
        logger.exception("report_summary: Cloudflare generation failed for event %s", event.get("id"))
        if strict:
            raise
        return {"event_id": event["id"], "summary": None, "status": "not_generated"}

    result = {
        "event_id": event["id"],
        "summary": text or None,
        "evidence_fingerprint": fingerprint,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(_cache_path(event["id"]), result)
    return {**result, "status": "ready" if text else "not_generated"}


def generate_queue_overview(items: Iterable[Dict[str, Any]], *, strict: bool = False) -> Dict[str, Any]:
    """A short paragraph over the whole pending queue, flagging which items
    look most worth prioritizing (multiple corroborating reports, high FRP +
    high confidence, near a confirmed incident). Not cached per-item - the
    caller decides how often to regenerate (e.g. once per queue page load)."""
    client = CloudflareAIClient()
    evidence = [_evidence(item) for item in items]
    if not client.configured:
        if strict:
            raise RuntimeError("Cloudflare Workers AI credentials are not configured")
        return {"overview": None, "status": "not_configured"}
    if not evidence:
        return {"overview": None, "status": "not_generated", "reason": "queue is empty"}

    prompt = (
        "You are triaging a wildland fire department's pending moderation queue of fire reports and "
        "satellite detections. Use only the supplied fields - do not invent details. In one short "
        "paragraph, flag which items look most worth prioritizing for review or for asking the public "
        "for more feedback (e.g. multiple corroborating reports, high FRP plus high detection "
        "confidence, proximity to an already-confirmed incident), referencing them by their id.\n\n"
        f"Pending queue:\n{json.dumps(evidence, default=str)}"
    )
    try:
        text = client.generate_text(prompt, max_tokens=320)
    except Exception:
        logger.exception("report_summary: Cloudflare queue overview generation failed")
        if strict:
            raise
        return {"overview": None, "status": "not_generated"}
    return {"overview": text or None, "status": "ready" if text else "not_generated"}



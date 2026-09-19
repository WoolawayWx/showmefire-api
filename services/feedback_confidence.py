"""Confidence (0-100%) that a public incident-feedback submission's
classification is likely correct, to help admins triage the review queue
before approving/rejecting (api/routers/fires.py:
admin_list_pending_incident_feedback). Same graceful-degradation shape as
detection_confidence.py and fire_confidence.py: a calibrated heuristic for
now, upgradeable to a trained model once enough admin decisions exist -
core.database.set_fire_incident_feedback_status's approve/reject calls are
exactly the labels that would need (mirrors fire_confidence.py's
admin_reviewed/official_source_confirmed training signal). Not persisted -
recomputed on every queue fetch, since it's a triage aid, not a stored
classification.

Beyond the feedback submission itself (contact/note/corroboration/spam),
this also weighs the actual satellite-detection evidence already attached to
the incident being reported on - FRP, land cover, whether any member
detection has already been officially/admin confirmed, and whether the
incident is a single blip or a sustained multi-pass signature - not just the
single pre-aggregated ML confidence percentage.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from typing import Optional

from core.database import get_db_path
from services.detection_confidence import _land_cover_fraction

OFFICIAL_TIERS = {"official_source_confirmed", "admin_reviewed"}
SUSTAINED_MIN_HOURS = 3.0
SUSTAINED_MIN_DETECTIONS = 3


def _incident_evidence(cursor: sqlite3.Cursor, incident_id: int) -> dict:
    """Pulls the actual detection data attached to this incident - not just
    the pre-computed per-detection ML confidence, but the raw FRP/land-cover/
    verification-tier readings behind it, plus how the incident's fire_incidents
    row already frames it (detection count, first/last seen)."""
    detection_rows = cursor.execute(
        "SELECT frp, land_cover, verification_tier, detection_confidence_pct FROM fire_events WHERE incident_id = ?",
        (incident_id,),
    ).fetchall()

    confidences = [row["detection_confidence_pct"] for row in detection_rows if row["detection_confidence_pct"] is not None]
    frps = [row["frp"] for row in detection_rows if row["frp"] is not None]
    cropland_fracs = [_land_cover_fraction(row["land_cover"], "Cropland", "Agricult") for row in detection_rows]
    water_fracs = [_land_cover_fraction(row["land_cover"], "Water") for row in detection_rows]
    has_official_or_reviewed = any(row["verification_tier"] in OFFICIAL_TIERS for row in detection_rows)

    incident = cursor.execute(
        "SELECT detection_count, first_detected_at, last_detected_at FROM fire_incidents WHERE id = ?",
        (incident_id,),
    ).fetchone()

    span_hours = 0.0
    if incident and incident["first_detected_at"] and incident["last_detected_at"]:
        try:
            first = datetime.fromisoformat(str(incident["first_detected_at"]).replace("Z", "+00:00"))
            last = datetime.fromisoformat(str(incident["last_detected_at"]).replace("Z", "+00:00"))
            span_hours = max(0.0, (last - first).total_seconds() / 3600.0)
        except ValueError:
            span_hours = 0.0

    return {
        "avg_ml_confidence": (sum(confidences) / len(confidences) / 100.0) if confidences else None,
        "max_frp": max(frps) if frps else None,
        "avg_cropland_frac": (sum(cropland_fracs) / len(cropland_fracs)) if cropland_fracs else 0.0,
        "avg_water_frac": (sum(water_fracs) / len(water_fracs)) if water_fracs else 0.0,
        "has_official_or_reviewed": has_official_or_reviewed,
        "sustained": span_hours >= SUSTAINED_MIN_HOURS and (incident["detection_count"] or 0) >= SUSTAINED_MIN_DETECTIONS if incident else False,
    }


def _corroboration_count(cursor: sqlite3.Cursor, incident_id: int, classification: str, exclude_feedback_id: int) -> int:
    """Distinct submitter IPs that independently reported the same
    classification for this incident - agreement between strangers is a much
    stronger signal than one anonymous submission on its own."""
    row = cursor.execute(
        """SELECT COUNT(DISTINCT submitter_ip_hash) FROM fire_incident_feedback
           WHERE incident_id = ? AND classification = ? AND id != ? AND submitter_ip_hash != ''""",
        (incident_id, classification, exclude_feedback_id),
    ).fetchone()
    return row[0] if row else 0


def _recent_submission_count(cursor: sqlite3.Cursor, ip_hash: str, exclude_feedback_id: int) -> int:
    """Other feedback submissions from this same IP hash in the last 24h,
    across any incident - a burst from one source is a spam/vote-stuffing
    signal, not corroboration."""
    if not ip_hash:
        return 0
    row = cursor.execute(
        """SELECT COUNT(*) FROM fire_incident_feedback
           WHERE submitter_ip_hash = ? AND id != ? AND created_at >= datetime('now', '-1 day')""",
        (ip_hash, exclude_feedback_id),
    ).fetchone()
    return row[0] if row else 0


def _directional(classification: str, value_for_confirmed: float, value_for_not_a_fire: float) -> float:
    """Most evidence signals only clearly favor one end of confirmed_fire vs
    not_a_fire; controlled_burn/unsure stay neutral since a real fire can
    legitimately be a deliberate burn, and 'unsure' isn't a claim either way."""
    if classification == "confirmed_fire":
        return value_for_confirmed
    if classification == "not_a_fire":
        return value_for_not_a_fire
    return 0.5


def score_feedback(feedback_id: int) -> Optional[dict]:
    """Returns {"score": 0-100, "factors": {...}} for one feedback row, or
    None if it no longer exists."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        feedback = cursor.execute(
            "SELECT id, incident_id, classification, note, contact, submitter_ip_hash FROM fire_incident_feedback WHERE id = ?",
            (feedback_id,),
        ).fetchone()
        if not feedback:
            return None

        classification = feedback["classification"]
        evidence = _incident_evidence(cursor, feedback["incident_id"])
        corroboration = _corroboration_count(cursor, feedback["incident_id"], classification, feedback_id)
        repeat_submissions = _recent_submission_count(cursor, feedback["submitter_ip_hash"], feedback_id)

        ml_alignment = (
            _directional(classification, evidence["avg_ml_confidence"], 1.0 - evidence["avg_ml_confidence"])
            if evidence["avg_ml_confidence"] is not None else 0.5
        )
        frp_signal = min(1.0, math.log1p(evidence["max_frp"]) / 6.0) if evidence["max_frp"] else 0.0
        frp_component = _directional(classification, frp_signal, 1.0 - frp_signal)
        # High water fraction is a classic false-positive pattern (sun glint) -
        # a strong signal against confirmed_fire specifically, independent of
        # whatever the averaged ML score says. Cropland doesn't get the same
        # penalty (agricultural areas have plenty of real fires); it only
        # mildly supports controlled_burn.
        water_component = _directional(classification, 1.0 - evidence["avg_water_frac"], evidence["avg_water_frac"])
        cropland_component = 0.5 + (0.3 * evidence["avg_cropland_frac"] if classification == "controlled_burn" else 0.0)
        official_component = _directional(classification, 0.85 if evidence["has_official_or_reviewed"] else 0.5,
                                           0.15 if evidence["has_official_or_reviewed"] else 0.5)
        sustained_component = _directional(classification, 0.75 if evidence["sustained"] else 0.5,
                                            0.3 if evidence["sustained"] else 0.5)

        corroboration_boost = min(0.15, corroboration * 0.08)
        contact_boost = 0.03 if (feedback["contact"] or "").strip() else 0.0
        detail_boost = 0.03 if len((feedback["note"] or "").strip()) >= 20 else 0.0
        spam_penalty = min(0.4, repeat_submissions * 0.15)

        score = (
            0.30 * ml_alignment
            + 0.15 * frp_component
            + 0.10 * water_component
            + 0.05 * cropland_component
            + 0.20 * official_component
            + 0.10 * sustained_component
            + corroboration_boost + contact_boost + detail_boost - spam_penalty
        )
        score = max(0.0, min(1.0, score))

        return {
            "score": round(score * 100),
            "factors": {
                "satellite_ml_confidence_pct": round(evidence["avg_ml_confidence"] * 100) if evidence["avg_ml_confidence"] is not None else None,
                "peak_frp_signal_pct": round(frp_signal * 100),
                "water_fraction_pct": round(evidence["avg_water_frac"] * 100),
                "cropland_fraction_pct": round(evidence["avg_cropland_frac"] * 100),
                "officially_confirmed_detection": evidence["has_official_or_reviewed"],
                "sustained_multi_pass_incident": evidence["sustained"],
                "corroborating_reports": corroboration,
                "contact_provided": bool((feedback["contact"] or "").strip()),
                "detailed_note": len((feedback["note"] or "").strip()) >= 20,
                "recent_submissions_from_source": repeat_submissions,
            },
        }
    finally:
        conn.close()

"""What gets asked of Clef (ai/clef.py) for each kind of record. Advisory only: see that module's docstring.

Only the facts a reviewer would weigh are sent: report text and photos, request dates and proof. Never a name, contact
detail, IP hash or upload token.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai import clef
from core.database import get_burn_ban_submission, get_db_path

logger = logging.getLogger(__name__)

# Same locations as routers/fires.py and routers/burn_bans.py (which own the uploads).
MEDIA_DIR = Path(os.getenv("FIRE_MEDIA_DIR", str(Path(os.getenv("DATA_DIR", ".")) / "fire-media")))
PROOF_DIR = Path(os.getenv("BURN_BAN_PROOF_DIR", str(Path(os.getenv("DATA_DIR", ".")) / "burn-ban-proofs")))

FIRE_REPORT = "fire_report"
BURN_BAN = "burn_ban"

_YES_NO = {"true": "Yes", "false": "No"}

FIRE_REPORT_QUESTIONS: Dict[str, Any] = {
    "shows_fire_or_smoke": {
        "type": "noul",
        "instructions": "Do the attached photos, or failing that the written description, show or describe an actual fire or a smoke plume?",
        "criteria": {"true": "A real fire or smoke plume", "false": "Nothing burning: unrelated photo, joke, or no fire described"},
    },
    "burn_type": {
        "type": "choice",
        "instructions": "What kind of fire is this most likely to be?",
        "criteria": {
            "wildfire": "Unplanned vegetation fire spreading on its own",
            "prescribed_or_agricultural": "Planned, attended burn: prescribed burn, field or pasture burning",
            "debris_or_brush_pile": "Small pile or debris burn",
            "structure_or_vehicle": "A building or vehicle on fire",
            "not_a_fire": "No fire",
            "unclear": "Cannot tell from what was sent",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this for a fire-weather staff reviewer to look at?",
        "criteria": [
            "Not a threat or not a fire",
            "Small or contained, unlikely to spread",
            "Spreading, or could spread with the weather",
            "People or structures threatened",
        ],
    },
    "likely_spam_or_prank": {
        "type": "noul",
        "instructions": "Does this submission look like spam, a prank or a test rather than a sincere report?",
        "criteria": {"true": "Spam, prank or test", "false": "A sincere report"},
    },
}


def _burn_ban_questions(has_proof: bool) -> Dict[str, Any]:
    questions: Dict[str, Any] = {
        "dates_consistent": {
            "type": "noul",
            "instructions": "Are the effective and expiration dates sensible for this type of request (an expiration after the "
                            "start, a removal that does not set a future expiration, nothing years away)?",
            "criteria": _YES_NO,
        },
        "likely_spam": {
            "type": "noul",
            "instructions": "Does this submission look like spam, a prank or a test rather than a sincere request about a county burn ban?",
            "criteria": {"true": "Spam, prank or test", "false": "A sincere request"},
        },
    }
    if has_proof:
        questions["proof_supports_request"] = {
            "type": "noul",
            "instructions": "Does the attached proof (an image, if one is attached) support this request: an official order, "
                            "announcement or notice from the named county about a burn ban? A link in the request cannot be opened; "
                            "judge only what is attached.",
            "criteria": {"true": "Proof is an official county burn-ban notice matching the request",
                         "false": "Proof is missing, unrelated, for another county, or not official"},
        }
    return questions


def _read_images(rows: List[sqlite3.Row], folder: Path) -> List[Dict[str, Any]]:
    files = []
    for row in rows:
        try:
            files.append({"content_type": row["content_type"], "data": (folder / Path(row["stored_filename"]).name).read_bytes()})
        except OSError:
            logger.warning("Clef: could not read stored file %s", row["stored_filename"])
    return clef.prepare_images(files)


def assess_fire_report(event_id: int, *, client: Optional[clef.ClefClient] = None, manual: bool = False) -> str:
    """Assess one public fire report with its photos. Run after submit and again after each photo upload: the
    stored answers are only replaced when the text or the photos changed. `manual` (an admin asked) skips the
    CLEF_ENABLED switch and always asks again."""
    if not manual and not clef.enabled() and client is None:
        return "skipped"
    try:
        with sqlite3.connect(get_db_path(), timeout=15) as conn:
            conn.row_factory = sqlite3.Row
            event = conn.execute(
                """SELECT source, description, out_of_ordinary, acres, acres_is_estimate, occurred_at, county_name,
                          reporter_relationship, reported_fire_status, reported_structures_threatened,
                          reported_responders_on_scene, cause_category FROM fire_events WHERE id = ?""", (event_id,)).fetchone()
            if not event or event["source"] != "user_submission":
                return "skipped"
            photos = conn.execute(
                "SELECT stored_filename, content_type FROM fire_event_media WHERE event_id = ? AND kind = 'photo' ORDER BY id",
                (event_id,)).fetchall()
        images = _read_images(photos, MEDIA_DIR)
        state = {
            "description": event["description"],
            "anything_unusual": event["out_of_ordinary"],
            "county": event["county_name"],
            "occurred_at": event["occurred_at"],
            "acres": event["acres"],
            "acres_is_estimate": bool(event["acres_is_estimate"]),
            "reporter_is": event["reporter_relationship"],
            "reporter_says_status": event["reported_fire_status"],
            "reporter_says_structures_threatened": event["reported_structures_threatened"],
            "reporter_says_responders_on_scene": event["reported_responders_on_scene"],
            "photos_attached": len(images),
        }
        return clef.assess("fire_event", event_id, FIRE_REPORT, state, FIRE_REPORT_QUESTIONS, images, client=client,
                            force=manual, manual=manual)
    except Exception:  # noqa: BLE001
        logger.warning("Clef fire report assessment setup failed for %s", event_id, exc_info=True)
        return "failed"


def assess_burn_ban(submission_id: int, *, client: Optional[clef.ClefClient] = None, manual: bool = False) -> str:
    """Assess one burn-ban request with its proof. Run after submit and after the proof upload. A PDF proof is
    noted but not read (Clef takes PNG, JPEG and WebP). `manual` as for assess_fire_report."""
    if not manual and not clef.enabled() and client is None:
        return "skipped"
    try:
        submission = get_burn_ban_submission(submission_id, admin=True)
        if not submission:
            return "skipped"
        images: List[Dict[str, Any]] = []
        stored = submission.get("proof_stored_filename") or ""
        content_type = submission.get("proof_content_type") or ""
        if stored and content_type in clef.IMAGE_TYPES:
            images = _read_images([{"stored_filename": stored, "content_type": content_type}], PROOF_DIR)  # type: ignore[list-item]
        proof_link = (submission.get("proof_url") or "").strip()
        has_proof = bool(images or proof_link or stored)
        state = {
            "request_type": submission.get("request_type"),
            "county": submission.get("county_name"),
            "effective_at": submission.get("effective_at"),
            "expires_at": submission.get("expires_at"),
            "submitter_comment": submission.get("submitter_comment"),
            "public_note": submission.get("public_note"),
            "proof_link_given": bool(proof_link),
            "proof_link": proof_link,
            "proof_file": ("image attached" if images else "PDF attached (not readable here)" if stored else "none"),
        }
        return clef.assess("burn_ban", submission_id, BURN_BAN, state, _burn_ban_questions(has_proof), images, client=client,
                            force=manual, manual=manual)
    except Exception:  # noqa: BLE001
        logger.warning("Clef burn-ban assessment setup failed for %s", submission_id, exc_info=True)
        return "failed"

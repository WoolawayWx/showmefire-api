"""Turns an approved public incident-feedback classification into a real
label on the underlying satellite/user-submission detections, so the
detection-confidence model (api/detection-confidence-model/) eventually has
genuine labeled examples to train on instead of only historical
`source='official'` import rows.

Classification -> cause_category mapping used at write time:
  confirmed_fire  -> "wildfire"      (a real fire)
  controlled_burn -> "prescribed"    (also a real fire: the detection-confidence model only learns
                                       "fire vs not a fire", so a planned burn is a positive, not a miss)
  not_a_fire      -> "not_a_fire"    (the only negative label; features.label_for in
                                       api/detection-confidence-model/)
  unsure          -> no label written; an "unsure" classification carries no
                     reliable signal and would just inject noise.
"""
from typing import Dict, List

from core.database import list_fire_incident_members, update_fire_event

CLASSIFICATION_TO_CAUSE = {
    "confirmed_fire": "wildfire",
    "controlled_burn": "prescribed",
    "not_a_fire": "not_a_fire",
}

# What a reviewer can label a detection or report as when approving it (approval means it is a real fire of some
# kind): "fire" is an unplanned wildfire, "controlled_burn" a planned one. Same cause categories as above.
REVIEW_LABEL_TO_CAUSE = {
    "fire": CLASSIFICATION_TO_CAUSE["confirmed_fire"],
    "controlled_burn": CLASSIFICATION_TO_CAUSE["controlled_burn"],
}


def apply_feedback_label(incident_id: int, classification: str, reviewed_by: str) -> List[Dict]:
    """Label every member detection of `incident_id` from an approved
    incident-feedback classification. Returns the list of updated events
    (empty if the classification carries no label, e.g. "unsure")."""
    cause_category = CLASSIFICATION_TO_CAUSE.get(classification)
    if cause_category is None:
        return []

    updated = []
    for member in list_fire_incident_members(incident_id):
        event = update_fire_event(
            member["id"],
            actor=reviewed_by,
            edit_reason=f"incident feedback approved: {classification}",
            verification_tier="admin_reviewed",
            cause_category=cause_category,
        )
        if event:
            updated.append(event)
    return updated

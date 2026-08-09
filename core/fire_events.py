"""
Shared enums and constants for the unified fire-event store.

Imported by the DB layer, the fires router, the ingest service, and the
label export so there is exactly one definition of each vocabulary.
"""

FIRE_SOURCES = ("user_submission", "viirs", "modis", "ngfs", "goes", "official")

FIRE_STATUSES = ("pending", "approved", "rejected", "deleted")

VERIFICATION_TIERS = ("unverified", "admin_reviewed", "official_source_confirmed")

TIER_RANK = {
    "unverified": 0,
    "admin_reviewed": 1,
    "official_source_confirmed": 2,
}

CAUSE_CATEGORIES = (
    "wildfire",
    "prescribed",
    "agricultural",
    "debris_burn",
    "equipment",
    "incendiary",
    "lightning",
    "unknown",
)

FUEL_TYPES = (
    "grass",
    "brush",
    "timber_litter",
    "timber_understory",
    "slash_blowdown",
    "crop_residue",
    "crp_native_prairie",
    "hardwood_leaf_litter",
    "structure_adjacent",
    "other",
)

MODERATION_ACTIONS = (
    "submitted",
    "ingested",
    "approved",
    "rejected",
    "edited",
    "tier_changed",
    "deleted",
    "pii_purged",
)

# Missouri bounding box used to validate submissions and detections alike.
# Mirrors api/tools/firedetections.py:45 (MISSOURI_BBOX_PARAM = "-95.8,35.9,-89.0,40.7").
MO_LAT_MIN = 35.9
MO_LAT_MAX = 40.7
MO_LON_MIN = -95.8
MO_LON_MAX = -89.0

"""Public, read-only Testbed product endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from services.beta_products import load_manifest, refresh_observation_products
from services.synoptic import get_station_data
from core.scheduler import raws_station_data


router = APIRouter(prefix="/api/testbed", tags=["testbed"])


@router.get("/products")
def get_testbed_products():
    """Return the current beta manifest and refreshable product metadata."""
    return load_manifest()


@router.post("/observations/refresh")
def refresh_testbed_observations():
    """Re-score the latest already-fetched observations for beta products."""
    result = refresh_observation_products(get_station_data(), raws_station_data)
    return result["manifest"]


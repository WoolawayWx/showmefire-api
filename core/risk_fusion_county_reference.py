"""
Loaders for the vendored county reference data used by the risk-fusion
shadow hook.

county_reference.json is copied from model-training/risk_fusion/ (built
there by scripts/build_county_reference.py - grid-independent, just
per-county area/region metadata, so no crop concerns).

county_cells.json is copied from
model-training/risk_fusion/county_cells_api.json, built there by
`python scripts/build_county_cells.py --source hrrr_api`. That source
reproduces DailyForecast.py's OWN two-step HRRR crop (core.domain.crop's
buffered MO_BUFFERED_BBOX, then DailyForecast.py's own tighter
`mo_bounds`) rather than the plain "hrrr" source's single buffered crop
- the plain source's grid is (273, 267) and does NOT match what
DailyForecast.py hands the shadow hooks at runtime, (196, 205). Do not
regenerate this file from the plain "hrrr" source/output
(risk_fusion/county_cells.json) - that was the cause of a real incident
(2026-09-21) where every shadow hook silently zero-scored on a grid-shape
mismatch for months.

Anything that consumes county_cells()["cell_to_fips"] MUST first check
county_cells()["grid_shape"] against the actual grid it has in hand and
refuse to proceed on a mismatch - see services/risk_fusion_hook.py for
the guard that does this. Silently indexing a foreign grid with this
cell map would misassign every county's weather.
"""
import json
from functools import lru_cache
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent / "risk_fusion_reference"
COUNTY_CELLS_PATH = REFERENCE_DIR / "county_cells.json"
COUNTY_REFERENCE_PATH = REFERENCE_DIR / "county_reference.json"


@lru_cache(maxsize=1)
def county_cells() -> dict:
    return json.loads(COUNTY_CELLS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def county_reference() -> dict:
    """Returns {fips: {area_km2, burnable_area_km2, region_id, ...}}."""
    data = json.loads(COUNTY_REFERENCE_PATH.read_text(encoding="utf-8"))
    return {row["fips"]: row for row in data["counties"]}

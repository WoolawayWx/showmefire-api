"""
Loaders for the vendored county reference data used by the risk-fusion
shadow hook.

county_cells.json and county_reference.json are copied from
model-training/risk_fusion/ (built there by scripts/build_county_cells.py
and scripts/build_county_reference.py against a TRAINING-repo HRRR grid
crop - see those scripts for provenance). They are NOT regenerated here
and must NOT be assumed to align with whatever grid DailyForecast.py
happens to be using at runtime - the two repos crop HRRR to different
bounding boxes (MO_BUFFERED_BBOX in model-training vs
DailyForecast.py's own mo_bounds), so their grids are different shapes.

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

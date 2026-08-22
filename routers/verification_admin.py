"""Admin-triggered verification actions (rerun for a past date).

Kept separate from the public, read-only routers/verification.py to keep
the trust boundary explicit: that router only ever reads
reports/validation_history.json and reports/{date}/validation_summary.json;
this one can trigger regeneration of those files.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import verify_token
from forecast.endOfDayReport import run_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/verification", tags=["verification-admin"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


class RerunRequest(BaseModel):
    date: str
    suffix: Optional[str] = ""
    forecast_glob: Optional[str] = "station_forecasts_*.json"


@router.post("/rerun")
async def rerun_verification(payload: RerunRequest, token: Optional[str] = None):
    email = _require_admin(token)
    try:
        report = run_report(
            date=payload.date,
            forecast_glob=payload.forecast_glob or "station_forecasts_*.json",
            report_suffix=payload.suffix or "",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Verification rerun failed for %s (requested by %s)", payload.date, email)
        raise HTTPException(status_code=500, detail=f"Rerun failed: {exc}")

    logger.info("Verification rerun for %s (suffix=%r) requested by %s", payload.date, payload.suffix, email)
    return {"success": True, "report": report}

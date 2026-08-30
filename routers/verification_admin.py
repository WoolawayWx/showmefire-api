"""Admin-triggered verification actions (rerun for a past date).

Kept separate from the public, read-only routers/verification.py to keep
the trust boundary explicit: that router only ever reads
reports/validation_history.json and reports/{date}/validation_summary.json;
this one can trigger regeneration of those files.
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import verify_token
from forecast.endOfDayReport import run_report
from core.config import REPORTS_DIR

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
        from maps.observed_peak_history import snapshot_observed_peak_for_date
        from services.rtma_peak import generate_rtma_peak_for_verification
        snapshot_observed_peak_for_date(payload.date)
        generate_rtma_peak_for_verification(payload.date)
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


@router.get("/summary/{date}")
async def get_verification_ai_summary(date: str, token: Optional[str] = None):
    """Return the generated verification narrative to authenticated admins."""
    _require_admin(token)
    try:
        report_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc

    summary_path = Path(REPORTS_DIR) / report_date / "validation_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"No validation report available for {report_date}")
    try:
        with summary_path.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Failed to read validation report") from exc
    return {"date": report_date, "ai_summary": report.get("ai_summary")}

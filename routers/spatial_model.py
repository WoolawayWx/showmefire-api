import math

from fastapi import APIRouter, HTTPException, Query

from services.spatial_fm import diagnostics
from services.model_shadow import diagnostics as shadow_diagnostics
from services.v4_shadow import diagnostics as v4_shadow_diagnostics
from services.v5_shadow import diagnostics as v5_shadow_diagnostics
from services.risk_fusion_shadow import diagnostics as risk_fusion_shadow_diagnostics
from services.risk_fusion_glm_shadow import diagnostics as risk_fusion_glm_shadow_diagnostics
from services import fm_explain
from services.drift_monitor import diagnostics as drift_diagnostics
from core.fire_danger import missing_input_diagnostics

router = APIRouter(prefix="/api/model/spatial", tags=["model-diagnostics"])


@router.get("/diagnostics")
def spatial_diagnostics():
    return diagnostics()


@router.get("/shadow-diagnostics")
def model_shadow_diagnostics():
    return shadow_diagnostics()


@router.get("/v4-shadow-diagnostics")
def guarded_v4_shadow_diagnostics():
    return v4_shadow_diagnostics()


@router.get("/v5-shadow-diagnostics")
def summer_guarded_v5_shadow_diagnostics():
    return v5_shadow_diagnostics()


@router.get("/fire-danger-diagnostics")
def fire_danger_diagnostics():
    return {"missing_inputs": missing_input_diagnostics()}


@router.get("/risk-fusion-shadow-diagnostics")
def risk_fusion_shadow_diagnostics_endpoint():
    return risk_fusion_shadow_diagnostics()


@router.get("/risk-fusion-glm-shadow-diagnostics")
def risk_fusion_glm_shadow_diagnostics_endpoint():
    return risk_fusion_glm_shadow_diagnostics()


@router.get("/drift-diagnostics")
def drift_diagnostics_endpoint():
    return drift_diagnostics()


@router.get("/fm-explain")
def fm_explain_endpoint(
    temp_c: float = Query(..., description="Temperature in Celsius"),
    rel_humidity: float = Query(..., ge=0, le=100, description="Relative humidity percentage"),
    wind_speed_ms: float = Query(..., ge=0, description="Wind speed in meters per second"),
    hour: int = Query(..., ge=0, le=23, description="Hour of day (0-23)"),
    month: int = Query(..., ge=1, le=12, description="Month of year (1-12)"),
    temp_mean_3h: float = Query(None, description="3-hour mean temperature (defaults to temp_c)"),
    rh_mean_3h: float = Query(None, description="3-hour mean relative humidity (defaults to rel_humidity)"),
    temp_mean_6h: float = Query(None, description="6-hour mean temperature (defaults to temp_c)"),
    rh_mean_6h: float = Query(None, description="6-hour mean relative humidity (defaults to rel_humidity)"),
    precip_1h: float = Query(0.0, ge=0), precip_3h: float = Query(0.0, ge=0),
    precip_6h: float = Query(0.0, ge=0), precip_24h: float = Query(0.0, ge=0),
    hours_since_rain: float = Query(24.0, ge=0),
):
    """
    Per-prediction fuel-moisture explanation for an arbitrary point-in-time
    reading - the concrete building block for a "why is this rated Extreme"
    UI panel (frontend wiring is out of scope here).

    Callers supply the same feature values predict_fm_grid would build for
    one grid cell rather than a lat/lon (this endpoint does not look up
    live weather itself - see forecast/DailyForecast.py::predict_fm_grid
    for how those features are assembled from RTMA/HRRR grids in the
    actual forecast pipeline).
    """
    day_of_year = int((month - 1) * 365.25 / 12 + 15)
    row = {
        "temp_c": temp_c,
        "rel_humidity": rel_humidity,
        "wind_speed_ms": wind_speed_ms,
        "hour": hour,
        "month": month,
        "emc_baseline": rel_humidity / 5.0,
        "temp_mean_3h": temp_mean_3h if temp_mean_3h is not None else temp_c,
        "rh_mean_3h": rh_mean_3h if rh_mean_3h is not None else rel_humidity,
        "temp_mean_6h": temp_mean_6h if temp_mean_6h is not None else temp_c,
        "rh_mean_6h": rh_mean_6h if rh_mean_6h is not None else rel_humidity,
        "precip_1h": precip_1h, "precip_3h": precip_3h, "precip_6h": precip_6h, "precip_24h": precip_24h,
        "hours_since_rain": hours_since_rain,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "day_of_year_sin": math.sin(2 * math.pi * day_of_year / 365.25),
        "day_of_year_cos": math.cos(2 * math.pi * day_of_year / 365.25),
    }
    try:
        return fm_explain.explain_prediction(row)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))

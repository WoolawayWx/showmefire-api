from fastapi import APIRouter

from services.spatial_fm import diagnostics

router = APIRouter(prefix="/api/model/spatial", tags=["model-diagnostics"])


@router.get("/diagnostics")
def spatial_diagnostics():
    return diagnostics()

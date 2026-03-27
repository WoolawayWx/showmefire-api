from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException

from core.database import get_afds_by_office
from services.afds import get_allowed_afd_offices

router = APIRouter(tags=["afds"])


@router.get("/afds/{office}")
async def get_afds_for_office(
    office: str,
    limit: int = 10,
    since: Optional[str] = None,
):
    office_code = office.strip().upper()
    allowed_offices = get_allowed_afd_offices()

    if office_code not in allowed_offices:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported office code",
                "office": office_code,
                "allowed_offices": allowed_offices,
            },
        )

    if since:
        try:
            datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid since datetime format") from exc

    items = get_afds_by_office(office=office_code, limit=limit, since=since)

    return {
        "office": office_code,
        "count": len(items),
        "last_updated": items[0]["issued_at"] if items else None,
        "items": items,
    }

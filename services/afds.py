import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from core.config import AFD_OFFICES
from core.database import get_known_afd_product_ids, insert_afd_records
from nws.get_afds import fetch_all_mo_afds

logger = logging.getLogger(__name__)


afd_ingest_state = {
    "last_run": None,
    "last_success": None,
    "error": None,
    "stats": None,
}


def get_allowed_afd_offices() -> List[str]:
    return list(AFD_OFFICES)


async def ingest_latest_afds(offices: Optional[Iterable[str]] = None) -> Dict:
    office_list = [o.strip().upper() for o in (offices or AFD_OFFICES) if o and o.strip()]
    started_at = datetime.now(timezone.utc).isoformat()

    if not office_list:
        result = {
            "offices": [],
            "started_at": started_at,
            "fetched_new_count": 0,
            "inserted_count": 0,
            "per_office_new": {},
            "status": "skipped",
            "message": "No offices configured",
        }
        afd_ingest_state["last_run"] = started_at
        afd_ingest_state["stats"] = result
        return result

    known_ids = get_known_afd_product_ids()

    try:
        new_records = await fetch_all_mo_afds(known_ids=known_ids, offices=office_list)
        inserted_count = insert_afd_records(new_records)

        per_office_new = {office: 0 for office in office_list}
        for record in new_records:
            office = (record.get("office") or "").upper()
            if office in per_office_new:
                per_office_new[office] += 1

        result = {
            "offices": office_list,
            "started_at": started_at,
            "fetched_new_count": len(new_records),
            "inserted_count": inserted_count,
            "per_office_new": per_office_new,
            "status": "ok",
        }

        afd_ingest_state["last_run"] = started_at
        afd_ingest_state["last_success"] = datetime.now(timezone.utc).isoformat()
        afd_ingest_state["error"] = None
        afd_ingest_state["stats"] = result

        logger.info("AFD ingest complete: offices=%s fetched_new=%s inserted=%s", office_list, len(new_records), inserted_count)
        return result
    except Exception as exc:
        afd_ingest_state["last_run"] = started_at
        afd_ingest_state["error"] = str(exc)
        logger.error("AFD ingest failed: %s", exc, exc_info=True)
        raise

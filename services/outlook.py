import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException
from shapely.geometry import shape, mapping
from shapely.validation import make_valid
from services.discord_notifier import notify_outlook_published

from core.config import DATA_DIR, GIS_DIR

logger = logging.getLogger(__name__)

OUTLOOK_INDEX_FILE = Path(DATA_DIR) / "outlook_index.json"
OUTLOOK_GRAPHIC_SCRIPT = Path(__file__).resolve().parents[1] / "maps" / "outlookgraphic.py"
SUPPORTED_OUTLOOK_DAYS = {2, 3}

FIXED_OUTLOOK_RISK_LEVEL = "Elevated"
FIXED_OUTLOOK_LABEL = "15% Risk (Elevated/High Fire Weather Conditions)"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_day(day: Any) -> int:
    try:
        normalized_day = int(day)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="day must be an integer") from exc

    if normalized_day not in SUPPORTED_OUTLOOK_DAYS:
        raise HTTPException(status_code=422, detail="day must be 2 or 3")

    return normalized_day


def _normalize_valid_date(valid_date: Any) -> Optional[str]:
    if valid_date is None:
        return None
    value = str(valid_date).strip()
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="valid_date must be in YYYY-MM-DD format") from exc


def _normalize_issue_time(issue_time: Any) -> Optional[str]:
    if issue_time is None:
        return None

    value = str(issue_time).strip()
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="issue_time must be a valid ISO datetime") from exc

    return parsed.isoformat()


def _draft_file(day: int) -> Path:
    return Path(GIS_DIR) / f"outlook_day{day}_draft.geojson"


def _published_file(day: int) -> Path:
    return Path(GIS_DIR) / f"outlook_day{day}_published.geojson"


def _read_json(path: Path, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return deepcopy(fallback) if fallback is not None else {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to read JSON from %s: %s", path, exc)
        return deepcopy(fallback) if fallback is not None else {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, indent=2)
            tmp_file.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _empty_feature_collection() -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": [], "outlook_text": ""}


def _normalize_outlook_text(raw_text: Any) -> str:
    text = str(raw_text or "").strip()
    return text[:2000]


def _validate_and_normalize_feature(feature: Dict[str, Any], fallback_issue_time: str, expected_day: int) -> Dict[str, Any]:
    geometry = feature.get("geometry")
    if not geometry:
        raise HTTPException(status_code=422, detail="Each feature must include geometry")

    geom_type = geometry.get("type")
    if geom_type not in {"Polygon", "MultiPolygon"}:
        raise HTTPException(status_code=422, detail="Only Polygon and MultiPolygon geometries are supported")

    try:
        shp = shape(geometry)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid geometry: {exc}") from exc

    if shp.is_empty:
        raise HTTPException(status_code=422, detail="Geometry cannot be empty")

    if not shp.is_valid:
        shp = make_valid(shp)
        if shp.is_empty or not shp.is_valid:
            raise HTTPException(status_code=422, detail="Geometry is invalid and could not be repaired")

    if shp.geom_type not in {"Polygon", "MultiPolygon"}:
        raise HTTPException(status_code=422, detail="Geometry repair produced unsupported geometry type")

    props = feature.get("properties") or {}

    valid_day = expected_day

    issue_time = str(props.get("issue_time") or fallback_issue_time)

    normalized_props = {
        "risk_level": FIXED_OUTLOOK_RISK_LEVEL,
        "valid_day": valid_day,
        "issue_time": issue_time,
        "label": FIXED_OUTLOOK_LABEL,
        "notes": "",
    }

    feature_id = feature.get("id") or str(uuid.uuid4())

    return {
        "type": "Feature",
        "id": feature_id,
        "properties": normalized_props,
        "geometry": mapping(shp),
    }


def validate_and_normalize_feature_collection(collection: Dict[str, Any], day: int) -> Dict[str, Any]:
    if not isinstance(collection, dict):
        raise HTTPException(status_code=422, detail="GeoJSON payload must be an object")

    if collection.get("type") != "FeatureCollection":
        raise HTTPException(status_code=422, detail="GeoJSON type must be FeatureCollection")

    features = collection.get("features")
    if not isinstance(features, list):
        raise HTTPException(status_code=422, detail="FeatureCollection features must be a list")

    now_iso = _utc_now_iso()
    normalized = [_validate_and_normalize_feature(f, now_iso, day) for f in features]
    outlook_text = _normalize_outlook_text(collection.get("outlook_text"))

    return {
        "type": "FeatureCollection",
        "features": normalized,
        "outlook_text": outlook_text,
    }


def _default_index() -> Dict[str, Any]:
    return {
        "days": {
            "2": {"draft": None, "published": None},
            "3": {"draft": None, "published": None},
        },
        "updated_at": None,
        "history": [],
    }


def _save_index(index_payload: Dict[str, Any]) -> None:
    _atomic_write_json(OUTLOOK_INDEX_FILE, index_payload)


def _load_index() -> Dict[str, Any]:
    idx = _read_json(OUTLOOK_INDEX_FILE, fallback=_default_index())
    idx.setdefault("days", {})
    idx["days"].setdefault("2", {"draft": None, "published": None})
    idx["days"].setdefault("3", {"draft": None, "published": None})
    idx.setdefault("history", [])
    return idx


def save_draft(collection: Dict[str, Any], editor_email: str, day: int) -> Dict[str, Any]:
    day = _normalize_day(day)
    normalized = validate_and_normalize_feature_collection(collection, day)
    draft_file = _draft_file(day)
    _atomic_write_json(draft_file, normalized)

    idx = _load_index()
    now_iso = _utc_now_iso()
    idx["days"][str(day)]["draft"] = {
        "file": str(draft_file),
        "feature_count": len(normalized["features"]),
        "updated_at": now_iso,
        "updated_by": editor_email,
        "day": day,
    }
    idx["updated_at"] = now_iso
    idx["history"].append({
        "action": "draft_saved",
        "day": day,
        "at": now_iso,
        "by": editor_email,
        "feature_count": len(normalized["features"]),
    })
    idx["history"] = idx["history"][-50:]
    _save_index(idx)

    return {
        "message": "Draft outlook saved",
        "day": day,
        "feature_count": len(normalized["features"]),
        "updated_at": now_iso,
        "outlook_text": normalized.get("outlook_text", ""),
    }


def get_draft(day: int) -> Dict[str, Any]:
    day = _normalize_day(day)
    draft_file = _draft_file(day)
    if not draft_file.exists():
        return _empty_feature_collection()
    return _read_json(draft_file, fallback=_empty_feature_collection())


def publish_draft(editor_email: str, day: int) -> Dict[str, Any]:
    day = _normalize_day(day)
    draft_file = _draft_file(day)
    published_file = _published_file(day)

    if not draft_file.exists():
        raise HTTPException(status_code=404, detail=f"No day {day} draft outlook exists to publish")

    draft = _read_json(draft_file, fallback=_empty_feature_collection())

    _atomic_write_json(published_file, draft)

    idx = _load_index()
    now_iso = _utc_now_iso()
    idx["days"][str(day)]["published"] = {
        "file": str(published_file),
        "feature_count": len(draft["features"]),
        "published_at": now_iso,
        "published_by": editor_email,
        "day": day,
    }
    idx["updated_at"] = now_iso
    idx["history"].append({
        "action": "draft_published",
        "day": day,
        "at": now_iso,
        "by": editor_email,
        "feature_count": len(draft["features"]),
    })
    idx["history"] = idx["history"][-50:]
    _save_index(idx)

    return {
        "message": "Draft outlook published",
        "day": day,
        "feature_count": len(draft["features"]),
        "published_at": now_iso,
        "outlook_text": _normalize_outlook_text(draft.get("outlook_text")),
    }


def clear_draft(editor_email: str, day: int) -> Dict[str, Any]:
    day = _normalize_day(day)
    draft_file = _draft_file(day)
    if draft_file.exists():
        draft_file.unlink()

    idx = _load_index()
    now_iso = _utc_now_iso()
    idx["days"][str(day)]["draft"] = None
    idx["updated_at"] = now_iso
    idx["history"].append({"action": "draft_deleted", "day": day, "at": now_iso, "by": editor_email})
    idx["history"] = idx["history"][-50:]
    _save_index(idx)

    return {"message": "Draft outlook deleted", "day": day, "updated_at": now_iso}


def get_published(day: int) -> Dict[str, Any]:
    day = _normalize_day(day)
    published_file = _published_file(day)
    if not published_file.exists():
        raise HTTPException(status_code=404, detail=f"No published day {day} outlook available")
    return _read_json(published_file, fallback=_empty_feature_collection())


def get_status(day: int) -> Dict[str, Any]:
    day = _normalize_day(day)
    draft_file = _draft_file(day)
    published_file = _published_file(day)
    idx = _load_index()
    day_meta = idx["days"].get(str(day), {"draft": None, "published": None})
    return {
        "day": day,
        "draft_exists": draft_file.exists(),
        "published_exists": published_file.exists(),
        "draft": day_meta.get("draft"),
        "published": day_meta.get("published"),
        "updated_at": idx.get("updated_at"),
    }


def generate_graphic(editor_email: str, day: int, valid_date: Optional[str] = None, issue_time: Optional[str] = None) -> Dict[str, Any]:
    day = _normalize_day(day)
    normalized_valid_date = _normalize_valid_date(valid_date)
    normalized_issue_time = _normalize_issue_time(issue_time)
    if not OUTLOOK_GRAPHIC_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Outlook graphic script not found")

    command = [sys.executable, str(OUTLOOK_GRAPHIC_SCRIPT), "--day", str(day)]
    command.extend(["--source", "auto"])
    if normalized_valid_date:
        command.extend(["--valid-date", normalized_valid_date])
    if normalized_issue_time:
        command.extend(["--issue-time", normalized_issue_time])

    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Outlook graphic generation timed out") from exc
    except Exception as exc:
        logger.error("Failed to run outlook graphic script: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to run outlook graphic script") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        logger.error("Outlook graphic script failed (%s): %s", completed.returncode, stderr or stdout)
        raise HTTPException(status_code=500, detail=stderr or stdout or "Outlook graphic generation failed")

    now_iso = _utc_now_iso()
    idx = _load_index()
    idx["updated_at"] = now_iso
    idx["history"].append({
        "action": "outlook_graphic_generated",
        "day": day,
        "at": now_iso,
        "by": editor_email,
    })
    idx["history"] = idx["history"][-50:]
    _save_index(idx)

    source = _read_json(_published_file(day), fallback=None)
    if not source:
        source = _read_json(_draft_file(day), fallback=_empty_feature_collection())

    feature_count = len(source.get("features", [])) if isinstance(source, dict) else 0
    normalized_outlook_text = _normalize_outlook_text((source or {}).get("outlook_text"))

    effective_issue_time = normalized_issue_time
    if not effective_issue_time and feature_count > 0:
        first_props = ((source.get("features") or [{}])[0].get("properties") or {})
        raw_issue_time = str(first_props.get("issue_time") or "").strip()
        if raw_issue_time:
            try:
                effective_issue_time = datetime.fromisoformat(raw_issue_time.replace("Z", "+00:00")).isoformat()
            except ValueError:
                effective_issue_time = None

    effective_valid_date = normalized_valid_date
    if not effective_valid_date and effective_issue_time:
        try:
            effective_valid_date = datetime.fromisoformat(
                effective_issue_time.replace("Z", "+00:00")
            ).date().isoformat()
        except ValueError:
            effective_valid_date = None

    try:
        notify_outlook_published(
            day=day,
            feature_count=feature_count,
            published_at=now_iso,
            issue_time=effective_issue_time,
            valid_date=effective_valid_date,
            outlook_text=normalized_outlook_text,
            image_version=str(int(datetime.now(timezone.utc).timestamp())),
        )
    except Exception as exc:
        logger.warning("Failed to emit Discord outlook graphic event: %s", exc)

    return {
        "message": "Outlook graphic generated",
        "day": day,
        "valid_date": normalized_valid_date,
        "issue_time": normalized_issue_time,
        "updated_at": now_iso,
        "output": stdout,
    }

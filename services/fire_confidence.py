"""Small, explainable confidence model for grouped satellite fire detections."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from core.config import GIS_DIR
from core.database import get_db_path, list_fire_incidents, list_fire_incident_members

OUTPUT_PATH = Path(GIS_DIR) / "fire_incident_confidence.geojson"


def _raw_confidence(value) -> float:
    text = str(value or "").lower()
    if text in {"high", "90", "nominal"}: return 0.82 if text == "nominal" else 0.92
    if text in {"medium", "probable"}: return 0.68
    if text in {"low", "0", "false"}: return 0.42
    try: return max(0.0, min(1.0, float(value) / 100.0))
    except (TypeError, ValueError): return 0.50


def _features(incident: dict, members: list[dict]) -> list[float]:
    confidences = [_raw_confidence(row.get("confidence")) for row in members]
    frps = [float(row["frp"]) for row in members if row.get("frp") is not None]
    return [
        sum(confidences) / max(1, len(confidences)),
        math.log1p(max(frps) if frps else 0.0) / 6.0,
        min(1.0, len(members) / 10.0),
        1.0 if len({row.get("source") for row in members}) > 1 else 0.0,
        1.0 if any(str(row.get("source")).lower() == "ngfs" for row in members) else 0.0,
    ]


def _score(features: list[float]) -> tuple[float, str, str]:
    """Use reviewed labels when enough exist; otherwise use a calibrated prior."""
    model = None
    try:
        import sqlite3
        from sklearn.linear_model import LogisticRegression
        with sqlite3.connect(get_db_path()) as conn:
            rows = conn.execute(
                """SELECT e.cause_category, e.confidence, e.frp, e.incident_id
                   FROM fire_events e WHERE e.incident_id IS NOT NULL
                   AND e.verification_tier IN ('admin_reviewed','official_source_confirmed')
                   AND e.cause_category != 'unknown'"""
            ).fetchall()
        samples, labels = [], []
        for cause, confidence, frp, incident_id in rows:
            labels.append(0 if cause in {"prescribed", "agricultural", "debris_burn"} else 1)
            samples.append([_raw_confidence(confidence), math.log1p(float(frp or 0)) / 6.0, 0.5, 0.0, 0.0])
        if len(samples) >= 12 and len(set(labels)) == 2:
            model = LogisticRegression(max_iter=200, class_weight="balanced").fit(samples, labels)
    except Exception:
        model = None

    if model is not None:
        score = float(model.predict_proba([features])[0][1])
        method = "reviewed logistic model"
    else:
        score = 0.45 * features[0] + 0.20 * min(1.0, features[1]) + 0.25 * features[2] + 0.10 * features[3]
        method = "calibrated detection prior"
    label = "high" if score >= 0.75 else "moderate" if score >= 0.50 else "low"
    return round(score, 3), label, method


def _circle(lon: float, lat: float, radius_km: float, steps: int = 24) -> list[list[float]]:
    points = []
    for index in range(steps + 1):
        angle = 2 * math.pi * index / steps
        points.append([lon + radius_km * math.cos(angle) / (111.32 * max(0.2, math.cos(math.radians(lat)))), lat + radius_km * math.sin(angle) / 110.574])
    return points


def build_confidence_geojson() -> dict:
    features = []
    for incident in list_fire_incidents(limit=200):
        members = list_fire_incident_members(incident["id"])
        score, label, method = _score(_features(incident, members))
        radius = min(3.0, max(0.5, 0.45 + len(members) * 0.08))
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [_circle(float(incident["centroid_longitude"]), float(incident["centroid_latitude"]), radius)]},
            "properties": {"incident_id": incident["id"], "incident_slug": incident.get("public_slug"), "confidence_score": score, "confidence_label": label, "model": method, "radius_km": round(radius, 2), "detection_count": len(members)},
        })
    return {"type": "FeatureCollection", "features": features, "metadata": {"generated_at": datetime.now(timezone.utc).isoformat(), "model": "fire-confidence-v1"}}


def refresh_confidence_shapes() -> dict:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_confidence_geojson()
    temporary = OUTPUT_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH)
    return {"feature_count": len(payload["features"]), "path": str(OUTPUT_PATH)}

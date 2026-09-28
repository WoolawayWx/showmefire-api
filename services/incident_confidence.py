"""One explainable confidence score for a fire incident (0-100), with reasons.

Replaces showing two numbers that could disagree (the best per-detection ML
score and a separate incident label). The label is derived from the score, so
they cannot diverge, and every score comes with the plain-language reasons
that produced it.

Rules-based on purpose: the labeled data a trained model would need doesn't
exist yet (see fire_labeling.py), and a hand-set score built from the signals
analysts actually use is easier to explain and to audit than a model fit on
near-empty labels. Signals (max points):

  persistence   30  distinct satellite scans that saw it, and for how long
  intensity     25  peak fire radiative power, nudged up if rising
  cross-sensor  20  more than one instrument (e.g. GOES + VIIRS) agrees
  sensor grade  15  the sensors' own confidence flags
  land cover   -25  cropland / water pixels are common false-positive sources
  public review +/- admin-approved feedback on the incident

Enabled with FIRE_INCIDENT_CONFIDENCE_VERSION=v2 (default v1 = legacy scorer).
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone

SCAN_BUCKET_MINUTES = 10
HIGH_THRESHOLD = 65
MODERATE_THRESHOLD = 40


def use_v2() -> bool:
    return os.getenv("FIRE_INCIDENT_CONFIDENCE_VERSION", "v1").strip().lower() == "v2"


def _raw_confidence_prior(value) -> float:
    text = str(value or "").lower()
    if text == "nominal":
        return 0.82
    if text in {"high", "90"}:
        return 0.92
    if text in {"medium", "probable"}:
        return 0.68
    if text in {"low", "0", "false"}:
        return 0.42
    try:
        return max(0.0, min(1.0, float(value) / 100.0))
    except (TypeError, ValueError):
        return 0.50


def _land_cover_fraction(land_cover, *keywords) -> float:
    """land_cover is stored as 'Name:pct, Name:pct'."""
    if not land_cover:
        return 0.0
    total = 0.0
    for part in str(land_cover).split(","):
        name, _, pct = part.partition(":")
        if any(keyword.lower() in name.strip().lower() for keyword in keywords):
            try:
                total += float(pct) / 100.0
            except ValueError:
                continue
    return min(1.0, total)


def _parse_time(value) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _label(score: int) -> str:
    return "high" if score >= HIGH_THRESHOLD else "moderate" if score >= MODERATE_THRESHOLD else "low"


def _factor(factors: list, label: str, score: float, maximum: float) -> None:
    """Record a component's effect relative to its midpoint: above half of its
    maximum raises confidence, below lowers it, near the middle is omitted."""
    effect = int(round(score - maximum * 0.5))
    if effect >= 2:
        factors.append({"label": label, "effect": "raises", "points": effect})
    elif effect <= -2:
        factors.append({"label": label, "effect": "lowers", "points": effect})


def score_incident(incident: dict, members: list[dict]) -> dict:
    """Returns {score: 0-100, label, factors: [{label, effect, points}],
    reasons: [str], method}. factors say what raised or lowered confidence and
    by roughly how many points; reasons is the same list as plain strings.
    Never raises on missing fields - absent evidence simply contributes nothing."""
    factors: list[dict] = []

    # --- persistence -------------------------------------------------------
    stamped = [(m, _parse_time(m.get("occurred_at"))) for m in members]
    times = [t for _, t in stamped if t is not None]
    bucket = SCAN_BUCKET_MINUTES * 60
    scans = len({int(t.timestamp() // bucket) for t in times})
    duration_h = (max(times) - min(times)).total_seconds() / 3600 if times else 0.0
    persistence = 30 * (0.7 * min(1.0, scans / 8) + 0.3 * min(1.0, duration_h / 3))
    if scans >= 2:
        _factor(factors, f"Seen in {scans} satellite scans over {duration_h:.1f} h", persistence, 30)
    else:
        _factor(factors, "Seen in only one scan so far" if scans == 1 else "No timed detections", persistence, 30)

    # --- intensity & trend -------------------------------------------------
    frp_by_scan: dict[int, float] = {}
    for member, t in stamped:
        if t is not None and member.get("frp") is not None:
            key = int(t.timestamp() // bucket)
            frp_by_scan[key] = max(frp_by_scan.get(key, 0.0), float(member["frp"]))
    peak_frp = max(frp_by_scan.values(), default=0.0)
    intensity_unit = min(1.0, math.log1p(peak_frp) / math.log1p(60))
    ordered = [frp_by_scan[k] for k in sorted(frp_by_scan)]
    rising = False
    # Only a rising trend earns anything: a fire that is dying down is still a
    # real fire, so fading heat must not lower confidence.
    if len(ordered) >= 4:
        early = sum(ordered[:2]) / 2
        late = sum(ordered[-2:]) / 2
        if early > 0 and late / early >= 1.2:
            intensity_unit = min(1.0, intensity_unit + 0.1)
            rising = True
    intensity = 25 * intensity_unit
    if peak_frp > 0:
        strength = "High" if intensity_unit >= 0.7 else "Low" if intensity_unit < 0.4 else "Moderate"
        _factor(factors, f"{strength} heat output ({peak_frp:.0f} MW peak)", intensity, 25)
    else:
        _factor(factors, "No heat output reading", intensity, 25)
    if rising:
        factors.append({"label": "Heat output is rising", "effect": "raises", "points": 2})

    # --- cross-sensor agreement -------------------------------------------
    sources = {str(m.get("source") or "").lower() for m in members} - {""}
    if "ngfs" in sources and sources & {"viirs", "modis"}:
        cross_unit, cross_label = 1.0, "Confirmed by more than one satellite (GOES + polar orbiter)"
    elif len(sources) > 1:
        cross_unit, cross_label = 0.8, "Confirmed by more than one satellite"
    else:
        cross_unit, cross_label = 0.4, "Only one satellite source has seen it"
    cross = 20 * cross_unit
    _factor(factors, cross_label, cross, 20)

    # --- sensor grade ------------------------------------------------------
    priors = [_raw_confidence_prior(m.get("confidence")) for m in members]
    mean_prior = sum(priors) / len(priors) if priors else 0.5
    grade = 15 * mean_prior
    _factor(factors, "Satellite's own confidence flag is high" if mean_prior >= 0.75 else "Satellite's own confidence flag is low",
            grade, 15)

    # --- land cover penalty ------------------------------------------------
    lc = [m.get("land_cover") for m in members if m.get("land_cover")]
    cropland = sum(_land_cover_fraction(v, "Cropland", "Agricult") for v in lc) / len(lc) if lc else 0.0
    water = sum(_land_cover_fraction(v, "Water") for v in lc) / len(lc) if lc else 0.0
    land_penalty = 15 * cropland + 10 * water
    if cropland >= 0.25:
        factors.append({"label": f"Mostly cropland ({cropland:.0%}) - often agricultural burning or false alarms",
                        "effect": "lowers", "points": -int(round(15 * cropland))})
    if water >= 0.25:
        factors.append({"label": f"Partly over water ({water:.0%})", "effect": "lowers", "points": -int(round(10 * water))})

    # --- public review (admin-approved only) -------------------------------
    approved = incident.get("approved_feedback_counts") or {}
    review = 0.0
    if approved.get("confirmed_fire"):
        review += 10
        factors.append({"label": "Confirmed by a public report", "effect": "raises", "points": 10})
    if approved.get("not_a_fire"):
        review -= 30
        factors.append({"label": "Reported as not a fire", "effect": "lowers", "points": -30})
    if approved.get("controlled_burn"):
        review -= 10
        factors.append({"label": "Reported as a controlled burn", "effect": "lowers", "points": -10})

    total = persistence + intensity + cross + grade - land_penalty + review
    score = int(round(max(0.0, min(100.0, total))))
    factors.sort(key=lambda f: -abs(f["points"]))
    return {"score": score, "label": _label(score), "factors": factors,
            "reasons": [f["label"] for f in factors], "method": "rules-v2"}

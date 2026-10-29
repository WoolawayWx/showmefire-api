"""Clef (Workers AI decision model): a recorded second opinion for staff, never an input to anything.

Clef takes a state (text/JSON plus up to 4 images) and a set of typed questions (noul = yes/no, choice, score) and
returns a probability for every option. Here it only annotates: the answers are stored in `clef_assessments` and
shown to staff, and nothing in the app reads them to publish, hide, approve, alert or rank anything. Keep it that way
until its answers have been compared with what staff actually decided.

Off unless CLEF_ENABLED=true. Uses the same credentials as the rest of ai/ (CLOUDFLARE_AI_API_KEY plus
CLOUDFLARE_ACCOUNT_ID or R2_ACCOUNT_ID). CLEF_MODEL picks "clef" (default) or "clef-flash".

`assess` never raises, so it is safe in a background task after a record is already saved.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

from ai.cloudflare import CloudflareAIError
from core.database import get_db_path

logger = logging.getLogger(__name__)

WORKERS_AI_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/{model}"
MODELS = ("clef", "clef-flash")
IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024


def enabled() -> bool:
    return os.getenv("CLEF_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def model_name() -> str:
    value = os.getenv("CLEF_MODEL", "clef").strip().lower()
    return value if value in MODELS else "clef"


class ClefClient:
    def __init__(self, api_key: Optional[str] = None, account_id: Optional[str] = None, *,
                 model: Optional[str] = None, timeout: float = 60.0, http_client: Any = None):
        self.api_key = (api_key or os.getenv("CLOUDFLARE_AI_API_KEY", "")).strip()
        self.account_id = (account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID") or os.getenv("R2_ACCOUNT_ID", "")).strip()
        self.model = model or model_name()
        self.timeout = timeout
        self.http_client = http_client

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.account_id)

    def run(self, state: Any, questions: Mapping[str, Any], images: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
        """The raw reply: {"model", "answers", "usage"}. Raises CloudflareAIError. `images` are
        {"content_type", "base64"} dicts, already checked by `prepare_images`."""
        if not self.configured:
            raise CloudflareAIError("Clef is not configured (CLOUDFLARE_AI_API_KEY and an account id are needed)")
        body: Dict[str, Any] = {"model": self.model, "state": state, "questions": dict(questions)}
        if images:
            body["images"] = [dict(image) for image in images]
        url = WORKERS_AI_URL.format(account_id=self.account_id, model=self.model)
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            if self.http_client is not None:
                response = self.http_client.post(url, headers=headers, json=body, timeout=self.timeout)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise CloudflareAIError(f"Clef request failed with HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise CloudflareAIError("Clef request could not be completed") from exc
        if isinstance(payload, Mapping) and payload.get("success") is False:
            raise CloudflareAIError("; ".join(str(e.get("message", e)) if isinstance(e, Mapping) else str(e)
                                              for e in (payload.get("errors") or [])) or "Clef returned an unsuccessful response")
        # The REST API wraps the model's reply in {"result": ...}; the Workers binding does not.
        reply = payload.get("result") if isinstance(payload, Mapping) and "result" in payload else payload
        if not isinstance(reply, Mapping) or not isinstance(reply.get("answers"), Mapping):
            raise CloudflareAIError("Clef reply had no answers")
        return dict(reply)


def prepare_images(files: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """{"content_type", "data": bytes} items -> the request's {"content_type", "base64"} items, dropping anything
    Clef would refuse (wrong type, over 4 MiB each, more than 4, over 8 MiB in all)."""
    chosen: List[Dict[str, str]] = []
    total = 0
    for item in files:
        data = item.get("data") or b""
        if item.get("content_type") not in IMAGE_TYPES or not data or len(data) > MAX_IMAGE_BYTES:
            continue
        if len(chosen) >= MAX_IMAGES or total + len(data) > MAX_TOTAL_IMAGE_BYTES:
            break
        total += len(data)
        chosen.append({"content_type": item["content_type"], "base64": base64.b64encode(data).decode("ascii")})
    return chosen


def _clamp(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(min(1.0, max(0.0, number)), 4)


def clean_answers(answers: Mapping[str, Any], questions: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only what staff need from each answer, checked against the question that was asked: yes-probability for
    noul, the chosen option + every option's probability + confidence for choice, the weighted level + confidence
    for score. A malformed or unexpected answer is dropped rather than guessed at."""
    cleaned: Dict[str, Any] = {}
    for qid, question in questions.items():
        answer = answers.get(qid)
        if not isinstance(answer, Mapping):
            continue
        kind = question.get("type")
        if kind == "noul" and (p := _clamp(answer.get("noul"))) is not None:
            cleaned[qid] = {"type": "noul", "noul": p}
        elif kind == "choice":
            options = set(question.get("criteria") or {})
            probabilities = answer.get("probabilities")
            if answer.get("choice") in options and isinstance(probabilities, Mapping):
                cleaned[qid] = {"type": "choice", "choice": answer["choice"], "confidence": _clamp(answer.get("confidence")),
                                "probabilities": {k: _clamp(v) for k, v in probabilities.items() if k in options}}
        elif kind == "score" and isinstance(answer.get("score"), (int, float)):
            cleaned[qid] = {"type": "score", "score": round(float(answer["score"]), 3),
                            "confidence": _clamp(answer.get("confidence")), "levels": len(question.get("criteria") or [])}
    return cleaned


def input_hash(state: Any, questions: Mapping[str, Any], images: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(json.dumps([state, questions], sort_keys=True, default=str).encode("utf-8"))
    for image in images:
        digest.update(hashlib.sha256(image["base64"].encode("ascii")).digest())
    return digest.hexdigest()


def assess(subject_type: str, subject_id: int, task: str, state: Any, questions: Mapping[str, Any],
           images: Sequence[Mapping[str, Any]] = (), *, client: Optional[ClefClient] = None, force: bool = False,
           manual: bool = False) -> str:
    """Ask Clef and store the answers for (subject_type, subject_id, task), replacing the earlier ones. Returns
    "stored", "unchanged" (same input as the stored answers), "skipped" (off or not configured) or "failed". `manual`
    means an admin asked: it runs even while CLEF_ENABLED is off (credentials are still needed)."""
    if not manual and not enabled() and client is None:
        return "skipped"
    try:
        client = client or ClefClient()
        if not client.configured:
            return "skipped"
        digest = input_hash(state, questions, images)
        with sqlite3.connect(get_db_path(), timeout=15) as conn:
            current = conn.execute(
                "SELECT input_hash FROM clef_assessments WHERE subject_type = ? AND subject_id = ? AND task = ?",
                (subject_type, subject_id, task)).fetchone()
        if current and current[0] == digest and not force:
            return "unchanged"
        images_used = len(images)
        try:
            reply = client.run(state, questions, images)
        except CloudflareAIError as exc:
            if not images:
                raise
            # An image Clef refuses (say, over its 16 megapixel limit) fails the whole request; the text alone is
            # still worth assessing, and images_used records that the photos were not read.
            logger.warning("Clef %s %s/%s failed with images (%s); retrying on the text alone", task, subject_type, subject_id, exc)
            images_used = 0
            reply = client.run(state, questions, ())
        answers = clean_answers(reply["answers"], questions)
        if not answers:
            raise CloudflareAIError("Clef returned no usable answers")
        usage = reply.get("usage") if isinstance(reply.get("usage"), Mapping) else {}
        with sqlite3.connect(get_db_path(), timeout=15) as conn:
            conn.execute(
                """INSERT INTO clef_assessments (subject_type, subject_id, task, input_hash, answers_json, model,
                                                 images_used, input_tokens, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(subject_type, subject_id, task) DO UPDATE SET input_hash = excluded.input_hash,
                       answers_json = excluded.answers_json, model = excluded.model, images_used = excluded.images_used,
                       input_tokens = excluded.input_tokens, created_at = excluded.created_at""",
                (subject_type, subject_id, task, digest, json.dumps(answers), str(reply.get("model") or client.model),
                 images_used, int(usage.get("input_tokens") or 0), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
        return "stored"
    except Exception:  # noqa: BLE001 - advisory only; never break the request that triggered this
        logger.warning("Clef assessment %s for %s %s failed", task, subject_type, subject_id, exc_info=True)
        return "failed"


def get_assessment(subject_type: str, subject_id: int, task: str) -> Optional[Dict[str, Any]]:
    """The stored assessment for staff screens: {"answers", "model", "images_used", "assessed_at"}, or None."""
    try:
        with sqlite3.connect(get_db_path(), timeout=5) as conn:
            row = conn.execute(
                """SELECT answers_json, model, images_used, created_at FROM clef_assessments
                   WHERE subject_type = ? AND subject_id = ? AND task = ?""", (subject_type, subject_id, task)).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    return {"answers": json.loads(row[0]), "model": row[1], "images_used": row[2], "assessed_at": row[3]}

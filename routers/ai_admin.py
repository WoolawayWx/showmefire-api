"""Admin: which Workers AI model each AI task uses (the page at /admin/ai-models). Settings saved here beat the
environment variables and take effect within seconds - see ai/models.py."""
import logging
import time
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai import models
from ai.cloudflare import CloudflareAIClient
from core.security import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai-admin"])

TASKS = [
    {"key": "report_features", "label": "Report extraction",
     "description": "Reads a fire report's text and pulls out fire type, status, acres, locations and key points."},
    {"key": "feedback_analysis", "label": "Feedback read",
     "description": "Summarizes an incident's public feedback notes when an admin clicks Analyze with AI."},
    {"key": "report_summary", "label": "Admin report summaries",
     "description": "The short summary and queue overview shown to reviewers on pending reports."},
]

# Workers AI text models worth considering, with USD per million tokens (input / output) from Cloudflare's pricing
# page on 2026-10-27. Any @cf/ model id can still be typed in; prices change, so check the page before relying on them.
SUGGESTED = [
    {"id": "@cf/google/gemma-4-26b-a4b-it", "label": "Gemma 4 26B", "price": "0.10 / 0.30", "note": "Default: cheap, good at following a field list"},
    {"id": "@cf/openai/gpt-oss-20b", "label": "GPT-OSS 20B", "price": "0.20 / 0.30", "note": "A bit more capable, for judging feedback"},
    {"id": "@cf/meta/llama-3.1-8b-instruct-fp8-fast", "label": "Llama 3.1 8B (fast)", "price": "0.045 / 0.384", "note": "Cheapest sensible fallback"},
    {"id": "@cf/mistralai/mistral-small-3.1-24b-instruct", "label": "Mistral Small 3.1 24B", "price": "0.351 / 0.555", "note": ""},
    {"id": "@cf/meta/llama-3.3-70b-instruct-fp8-fast", "label": "Llama 3.3 70B", "price": "0.293 / 2.253", "note": "Large; the usual escalation model"},
    {"id": "@cf/openai/gpt-oss-120b", "label": "GPT-OSS 120B", "price": "0.350 / 0.750", "note": "Large"},
]


def _payload() -> Dict:
    saved = models.overrides()
    tasks = []
    for task in TASKS:
        model, source = models.resolve_task(task["key"])
        tasks.append({**task, "model": model, "source": source, "override": saved.get(task["key"], "")})
    escalation, source = models.resolve_escalation()
    return {
        "success": True,
        "configured": CloudflareAIClient().configured,
        "default_model": models.DEFAULT_TEXT_MODEL,
        "tasks": tasks,
        "escalation": {"model": escalation, "source": source, "override": saved.get(models.ESCALATION_KEY, ""),
                       "description": "Retried once when a task's model errors or returns something unusable. "
                                      "Set it equal to a task's model to turn the retry off for that task."},
        "suggested": SUGGESTED,
    }


class ModelSettingsUpdate(BaseModel):
    # key -> model id; an empty string or null clears the admin setting (the environment/default applies again)
    settings: Dict[str, Optional[str]] = Field(default_factory=dict)


class ModelTest(BaseModel):
    model: str = Field(min_length=1, max_length=200)


@router.get("/api/admin/ai/models")
def get_ai_models(_admin: str = Depends(require_admin)):
    return _payload()


@router.put("/api/admin/ai/models")
def update_ai_models(payload: ModelSettingsUpdate, admin: str = Depends(require_admin)):
    try:
        models.save_overrides(payload.settings, admin)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _payload()


@router.post("/api/admin/ai/models/test")
def test_ai_model(payload: ModelTest, _admin: str = Depends(require_admin)):
    """Send one tiny request to a model, so a typo or an unavailable model is caught before it is saved."""
    model = payload.model.strip()
    if not models.MODEL_ID.fullmatch(model):
        raise HTTPException(status_code=422, detail="Not a Workers AI model id (expected something like @cf/google/gemma-4-26b-a4b-it)")
    client = CloudflareAIClient()
    if not client.configured:
        raise HTTPException(status_code=409, detail="Cloudflare AI credentials are not configured on the server")
    started = time.monotonic()
    try:
        reply = client.generate_text('Reply with exactly this JSON and nothing else: {"ok": true}', model=model, max_tokens=40)
    except Exception as exc:  # noqa: BLE001 - the point of the test is to report why it failed
        return {"success": True, "ok": False, "model": model, "error": str(exc)[:300]}
    return {"success": True, "ok": True, "model": model, "ms": int((time.monotonic() - started) * 1000),
            "valid_json": _is_json(reply), "reply": reply[:200]}


def _is_json(reply: str) -> bool:
    try:
        models.parse_reply(reply)
        return True
    except ValueError:
        return False

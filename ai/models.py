"""Which Workers AI model each task uses, and how a task falls back to a stronger one.

Short, structured jobs (reading report text, summarizing feedback) do not need a large model, so each task has its
own setting and they default to a small, cheap one. If that model errors or returns something unusable, the task is
retried once on the escalation model, so a bad reply costs one larger-model call instead of a missing result.

  task                 env var (first set wins)                                  default
  report_features      CLOUDFLARE_AI_REPORT_FEATURES_MODEL, CLOUDFLARE_AI_FEATURE_MODEL   CLOUDFLARE_AI_TEXT_MODEL or Gemma
  feedback_analysis    CLOUDFLARE_AI_FEEDBACK_MODEL                                  CLOUDFLARE_AI_TEXT_MODEL or Gemma
  report_summary       CLOUDFLARE_AI_REPORT_SUMMARY_MODEL                            CLOUDFLARE_AI_TEXT_MODEL or Gemma

CLOUDFLARE_AI_ESCALATION_MODEL (default: the 70B Llama) is the retry model; set it equal to a task's model to turn
escalation off for that task. Anything not listed here (the public forecast discussion, verification summaries)
still uses CLOUDFLARE_AI_MODEL / the client default.

The admin site (Admin > AI Models) can override any of these without touching the environment: those settings are
stored in `ai_model_settings` and win over the environment, which wins over the defaults above. They are read at
most every CACHE_SECONDS, so a change takes effect within seconds and without a restart.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from ai.cloudflare import DEFAULT_MODEL as ESCALATION_DEFAULT, CloudflareAIClient
from core.database import get_db_path

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "@cf/google/gemma-4-26b-a4b-it"

TASK_ENV: Dict[str, Tuple[str, ...]] = {
    "report_features": ("CLOUDFLARE_AI_REPORT_FEATURES_MODEL", "CLOUDFLARE_AI_FEATURE_MODEL"),
    "feedback_analysis": ("CLOUDFLARE_AI_FEEDBACK_MODEL",),
    "report_summary": ("CLOUDFLARE_AI_REPORT_SUMMARY_MODEL",),
}


ESCALATION_KEY = "escalation"
SETTING_KEYS = (*TASK_ENV, ESCALATION_KEY)
MODEL_ID = re.compile(r"^@cf/[A-Za-z0-9._-]{1,60}/[A-Za-z0-9._-]{1,80}$")
CACHE_SECONDS = 15

_cache: Dict[str, Any] = {"at": 0.0, "values": {}}
_cache_lock = threading.Lock()


def overrides() -> Dict[str, str]:
    """The model settings saved from the admin site, {key: model id}. Cached for CACHE_SECONDS; a database
    problem means "no overrides" (the environment and defaults still apply), never an error."""
    with _cache_lock:
        if time.monotonic() - _cache["at"] < CACHE_SECONDS:
            return dict(_cache["values"])
    try:
        with sqlite3.connect(get_db_path(), timeout=5) as conn:
            values = {key: value for key, value in conn.execute("SELECT key, value FROM ai_model_settings")
                      if key in SETTING_KEYS and value}
    except sqlite3.Error:
        values = {}
    with _cache_lock:
        _cache.update(at=time.monotonic(), values=values)
    return dict(values)


def save_overrides(changes: Dict[str, Optional[str]], actor: str) -> None:
    """Set (a model id) or clear (None / "") settings. Raises ValueError for an unknown key or a malformed id."""
    for key, value in changes.items():
        if key not in SETTING_KEYS:
            raise ValueError(f"unknown setting: {key}")
        if value and not MODEL_ID.fullmatch(value.strip()):
            raise ValueError(f"{value!r} is not a Workers AI model id (expected something like @cf/google/gemma-4-26b-a4b-it)")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with sqlite3.connect(get_db_path(), timeout=15) as conn:
        for key, value in changes.items():
            if value and value.strip():
                conn.execute(
                    """INSERT INTO ai_model_settings (key, value, updated_by, updated_at) VALUES (?, ?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_by = excluded.updated_by,
                           updated_at = excluded.updated_at""", (key, value.strip(), actor, now))
            else:
                conn.execute("DELETE FROM ai_model_settings WHERE key = ?", (key,))
    with _cache_lock:
        _cache["at"] = 0.0


def _environment_model(task: str) -> Optional[str]:
    for name in TASK_ENV[task]:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return os.getenv("CLOUDFLARE_AI_TEXT_MODEL", "").strip() or None


def resolve_task(task: str) -> Tuple[str, str]:
    """(model, where it came from: "admin", "env" or "default") for a task."""
    saved = overrides().get(task)
    if saved:
        return saved, "admin"
    from_env = _environment_model(task)
    return (from_env, "env") if from_env else (DEFAULT_TEXT_MODEL, "default")


def resolve_escalation() -> Tuple[str, str]:
    saved = overrides().get(ESCALATION_KEY)
    if saved:
        return saved, "admin"
    from_env = os.getenv("CLOUDFLARE_AI_ESCALATION_MODEL", "").strip()
    return (from_env, "env") if from_env else (ESCALATION_DEFAULT, "default")


def model_for(task: str) -> str:
    return resolve_task(task)[0]


def escalation_model() -> str:
    return resolve_escalation()[0]


def parse_reply(reply: str) -> Dict[str, Any]:
    """The first JSON object in a model reply (tolerates code fences and chatter around it)."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the model reply")
    return json.loads(reply[start:end + 1])


def _attempt(task: str, model: Optional[str], run: Callable[[str], Any]) -> Tuple[Any, str]:
    """run(model) on the task's model, then once on the escalation model if that raises. Returns (result, model
    actually used); raises the last error if both fail."""
    models = [model or model_for(task)]
    fallback = escalation_model()
    if fallback != models[0]:
        models.append(fallback)
    error: Optional[Exception] = None
    for candidate in models:
        try:
            return run(candidate), candidate
        except Exception as exc:  # noqa: BLE001 - a bad reply and an API error both deserve the retry
            error = exc
            logger.warning("AI task %s failed on %s: %s", task, candidate, exc)
    assert error is not None
    raise error


def generate_json(client: CloudflareAIClient, task: str, prompt: str, system: Optional[str] = None, *,
                  max_tokens: int = 600, model: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """(parsed JSON object, model used) for a task, escalating once on an error or an unparseable reply."""
    return _attempt(task, model, lambda m: parse_reply(client.generate_text(prompt, system, model=m, max_tokens=max_tokens)))


def generate_prose(client: CloudflareAIClient, task: str, prompt: str, system: Optional[str] = None, *,
                   max_tokens: int = 300, model: Optional[str] = None) -> Tuple[str, str]:
    """(text, model used) for a task, escalating once on an error."""
    return _attempt(task, model, lambda m: client.generate_text(prompt, system, model=m, max_tokens=max_tokens))

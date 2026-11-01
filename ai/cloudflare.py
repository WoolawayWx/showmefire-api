"""Small REST client for Cloudflare Workers AI text-generation models."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping

import httpx

from core.sandbox import is_sandbox

DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
WORKERS_AI_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


logger = logging.getLogger(__name__)

SESSION_AFFINITY_HEADER = "x-session-affinity"


def _cached_tokens(usage: Mapping[str, Any]) -> int | None:
    """Cached prompt tokens from a Workers AI `usage` object, whichever shape it uses
    (OpenAI-style prompt_tokens_details.cached_tokens, or a flat cached_tokens)."""
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping) and isinstance(details.get("cached_tokens"), int):
        return details["cached_tokens"]
    if isinstance(usage.get("cached_tokens"), int):
        return usage["cached_tokens"]
    return None


class CloudflareAIError(RuntimeError):
    """Raised when Workers AI cannot produce a text response."""


class CloudflareAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        http_client: Any | None = None,
    ):
        self.api_key = (api_key or os.getenv("CLOUDFLARE_AI_API_KEY", "")).strip()
        self.account_id = (
            account_id
            or os.getenv("CLOUDFLARE_ACCOUNT_ID")
            or os.getenv("R2_ACCOUNT_ID", "")
        ).strip()
        self.model = (model or os.getenv("CLOUDFLARE_AI_MODEL", DEFAULT_MODEL)).strip()
        self.timeout = timeout
        self.http_client = http_client
        # Usage of the most recent call (None when the API reported none), for cache-hit logging.
        self.last_usage: dict[str, Any] | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.account_id and self.model)

    def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        *,
        model: str | None = None,
        max_tokens: int = 320,
        session_id: str | None = None,
    ) -> str:
        """`session_id` turns on Workers AI prefix caching: requests with the same id are routed
        to the same model instance, so a static system prompt is reused instead of recomputed.
        Keep everything that never changes in `system`, and everything dynamic in `prompt`."""
        if is_sandbox():  # reviewer sandbox: no outside calls
            raise CloudflareAIError("AI is switched off in the reviewer sandbox")
        if not self.api_key:
            raise CloudflareAIError("CLOUDFLARE_AI_API_KEY is not configured")
        if not self.account_id:
            raise CloudflareAIError("CLOUDFLARE_ACCOUNT_ID or R2_ACCOUNT_ID is not configured")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        selected_model = (model or self.model).strip()
        url = WORKERS_AI_URL.format(
            account_id=self.account_id,
            model=selected_model,
        )
        request = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        session_id = (session_id or os.getenv("CLOUDFLARE_AI_SESSION_AFFINITY") or "").strip()
        if session_id:
            headers[SESSION_AFFINITY_HEADER] = session_id
        self.last_usage = None

        try:
            if self.http_client is not None:
                response = self.http_client.post(
                    url,
                    headers=headers,
                    json=request,
                    timeout=self.timeout,
                )
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=request)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise CloudflareAIError(
                f"Workers AI request failed with HTTP {exc.response.status_code}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise CloudflareAIError("Workers AI request could not be completed") from exc

        if not isinstance(payload, Mapping):
            raise CloudflareAIError("Workers AI returned an invalid JSON response")

        if payload.get("success") is False:
            errors = payload.get("errors") or []
            detail = "; ".join(
                str(error.get("message", error))
                if isinstance(error, Mapping)
                else str(error)
                for error in errors
            )
            raise CloudflareAIError(detail or "Workers AI returned an unsuccessful response")

        result = payload.get("result")
        usage = result.get("usage") if isinstance(result, Mapping) else None
        if isinstance(usage, Mapping):
            self.last_usage = dict(usage)
            if session_id:
                logger.info(
                    "Workers AI %s: prompt_tokens=%s cached_tokens=%s",
                    selected_model, usage.get("prompt_tokens"), _cached_tokens(usage),
                )
        text = result.get("response") if isinstance(result, Mapping) else None
        if text is None and isinstance(result, Mapping) and isinstance(result.get("choices"), list):
            # Chat-completions shape (e.g. @cf/moonshotai/kimi-k2.5). Reasoning models spend max_tokens on
            # their thinking first, so a short budget can leave `content` empty with finish_reason "length".
            choice = (result["choices"] or [{}])[0]
            message = choice.get("message") if isinstance(choice, Mapping) else None
            text = message.get("content") if isinstance(message, Mapping) else None
            if not (isinstance(text, str) and text.strip()) and isinstance(choice, Mapping) and choice.get("finish_reason") == "length":
                raise CloudflareAIError(
                    f"Workers AI ran out of max_tokens ({max_tokens}) before producing an answer "
                    "(a reasoning model spends tokens thinking); raise the token limit"
                )
        # When the model's whole reply is valid JSON, Workers AI can return it already parsed (an object/array)
        # instead of a string. Hand callers the JSON text either way.
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
        if not isinstance(text, str) or not text.strip():
            keys = sorted(result) if isinstance(result, Mapping) else type(result).__name__
            raise CloudflareAIError(
                f"Workers AI response did not contain result.response (response is {type(text).__name__}; "
                f"result: {keys})"
            )
        return text.strip()


def generate_text(
    prompt: str,
    system: str | None = None,
    *,
    model: str | None = None,
    client: CloudflareAIClient | None = None,
    session_id: str | None = None,
) -> str:
    """Generate text using configured Workers AI credentials."""
    extra = {"session_id": session_id} if session_id else {}  # fakes without the argument keep working
    return (client or CloudflareAIClient()).generate_text(
        prompt,
        system=system,
        model=model,
        **extra,
    )

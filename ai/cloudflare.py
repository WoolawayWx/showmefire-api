"""Small REST client for Cloudflare Workers AI text-generation models."""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
WORKERS_AI_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"


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

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.account_id and self.model)

    def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        *,
        model: str | None = None,
    ) -> str:
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
            "max_tokens": 320,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
        text = result.get("response") if isinstance(result, Mapping) else None
        if not isinstance(text, str) or not text.strip():
            raise CloudflareAIError("Workers AI response did not contain result.response")
        return text.strip()


def generate_text(
    prompt: str,
    system: str | None = None,
    *,
    model: str | None = None,
    client: CloudflareAIClient | None = None,
) -> str:
    """Generate text using configured Workers AI credentials."""
    return (client or CloudflareAIClient()).generate_text(
        prompt,
        system=system,
        model=model,
    )

"""Optional Gemini summary generation for completed verification reports."""

import json
import logging
import os
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def generate_verification_summary(report: Dict[str, Any], comparison_rows: Iterable[Dict[str, Any]]) -> Optional[str]:
    """Summarize a report and representative raw forecast/observation pairs.

    Gemini is deliberately imported and initialized lazily so verification still
    succeeds when the optional API key or package is unavailable.
    """
    api_key = os.getenv("genai_key", "").strip()
    if not api_key:
        return None

    try:
        from google import genai

        rows = list(comparison_rows)
        prompt = (
            "You are summarizing a completed Missouri fire-weather forecast verification. "
            "Use only the supplied data. Explain what was forecast, what was observed, "
            "the most meaningful differences, and any limitations. Do not invent causes "
            "or claim more certainty than the metrics support. Return 2-4 concise paragraphs "
            "in plain text with no markdown headings.\n\n"
            f"Aggregate report:\n{json.dumps(report, default=str)}\n\n"
            f"Representative forecast/observed rows:\n{json.dumps(rows[:40], default=str)}"
        )
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        text = (response.text or "").strip()
        return text or None
    except Exception:
        logger.exception("Gemini verification summary generation failed")
        return None

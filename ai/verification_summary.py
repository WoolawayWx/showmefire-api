"""Optional Cloudflare Workers AI summaries for completed verification reports."""

import json
import logging
from typing import Any, Dict, Iterable, Optional

from ai.cloudflare import CloudflareAIClient

logger = logging.getLogger(__name__)


def generate_verification_summary(report: Dict[str, Any], comparison_rows: Iterable[Dict[str, Any]]) -> Optional[str]:
    """Summarize a report and representative raw forecast/observation pairs.

    Workers AI is initialized lazily so verification still succeeds when the
    optional Cloudflare credentials or package is unavailable.
    """
    client = CloudflareAIClient()
    if not client.configured:
        return None

    try:
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
        text = client.generate_text(prompt)
        return text or None
    except Exception:
        logger.exception("Cloudflare verification summary generation failed")
        return None

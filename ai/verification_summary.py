"""Optional Cloudflare Workers AI summaries for completed verification reports."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ai.cloudflare import CloudflareAIClient

logger = logging.getLogger(__name__)


def _metric_snapshot(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics", {})
    return {
        name: {key: value for key, value in values.items()
               if key in {"mae", "rmse", "bias", "count", "correlation",
                          "exact_match_rate", "within_one_category_rate"}}
        for name, values in metrics.items()
        if isinstance(values, dict)
    }


def build_verification_ai_packet(report, recent_history, comparison_rows):
    """Build a safe JSON payload for the Cloudflare AI worker."""
    recent = []
    for entry in list(recent_history)[-30:]:
        recent.append({
            "date": entry.get("date"),
            "record_count": entry.get("record_count", 0),
            "metrics": _metric_snapshot(entry),
            "confusion_matrix": entry.get("confusion_matrix"),
            "neighborhood_verification": entry.get("neighborhood_verification"),
        })
    return {
        "schema_version": "verification-ai-packet.v1",
        "instructions": {
            "today": "Write one concise paragraph explaining today's verification.",
            "recent_trends": "Write at least one paragraph covering on-par performance, outliers, bias, underforecasting, and overforecasting when supported.",
            "improvement_ideas": "Suggest practical improvements as a paragraph or concise bullet list. Do not invent causes.",
            "output": "Return exactly three sections: Today, Recent trends, Improvement ideas.",
        },
        "today": {
            "date": report.get("date"),
            "generated_at": report.get("generated_at"),
            "record_count": report.get("record_count", 0),
            "stations_count": report.get("stations_count"),
            "metrics": _metric_snapshot(report),
            "confusion_matrix": report.get("confusion_matrix"),
            "wind_confusion_matrix": report.get("wind_confusion_matrix"),
            "neighborhood_verification": report.get("neighborhood_verification"),
            "qc_exclusions": report.get("qc_exclusions", []),
        },
        "recent_history": recent,
        "representative_comparisons": list(comparison_rows)[:40],
    }


def write_verification_ai_packet(report, recent_history, comparison_rows, output_path: Path):
    """Write the worker input beside the dated validation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        build_verification_ai_packet(report, recent_history, comparison_rows),
        indent=2, default=str,
    ), encoding="utf-8")
    return output_path


def generate_verification_summary(
    report: Dict[str, Any],
    comparison_rows: Iterable[Dict[str, Any]],
    recent_history: Iterable[Dict[str, Any]] = (),
) -> Optional[str]:
    """Summarize a report and representative raw forecast/observation pairs.

    Workers AI is initialized lazily so verification still succeeds when the
    optional Cloudflare credentials or package is unavailable.
    """
    client = CloudflareAIClient()
    if not client.configured:
        return None

    try:
        packet = build_verification_ai_packet(report, recent_history, comparison_rows)
        prompt = (
            "You are summarizing a completed Missouri fire-weather forecast verification. "
            "Use only the supplied data. Do not invent causes or claim more certainty "
            "than the metrics support. Return exactly three labeled sections: Today "
            "(one paragraph), Recent trends (at least one paragraph), and Improvement "
            "ideas (a paragraph or concise bullet list), in plain text.\n\n"
            f"Verification packet:\n{json.dumps(packet, default=str)}"
        )
        text = client.generate_text(prompt)
        return text or None
    except Exception:
        logger.exception("Cloudflare verification summary generation failed")
        return None

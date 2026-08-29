"""Generate the optional RSS summary from numeric forecast data."""

import os
from pathlib import Path

from dotenv import load_dotenv
from ai.briefing import briefing_json, build_briefing, validate_briefing_text
from ai.cloudflare import CloudflareAIClient

load_dotenv()

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parent.parent))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", BASE_DIR / "archive" / "forecasts"))


def _fallback(briefing: dict) -> str:
    state = briefing["statewide"]
    danger = state["highest_fire_danger"] or "Low"
    precipitation = state["precipitation"]
    if precipitation == "trace":
        rain = "trace precipitation"
    elif state["precip_in"]["max"] is not None:
        rain = f"precipitation up to {state['precip_in']['max']:.3f} inches"
    else:
        rain = "precipitation unavailable"
    return (
        f"{danger} is the highest forecast fire-danger class represented by the available station and county data. "
        f"Minimum relative humidity ranges from {state['rh']['min']}% to {state['rh']['max']}%, "
        f"fuel moisture ranges from {state['fuel_moisture']['min']}% to "
        f"{state['fuel_moisture']['max']}%, with peak winds up to "
        f"{state['wind_mph']['max']} mph where available and {rain}."
    )


def _valid_summary(text: str, briefing: dict) -> bool:
    return validate_briefing_text(text, briefing)


def generate_summary() -> str | None:
    """Return a factual RSS summary, or None when optional AI is unavailable."""
    try:
        briefing = build_briefing(ARCHIVE_DIR)
    except (OSError, ValueError) as exc:
        print(f"Unable to build RSS numeric briefing: {exc}")
        return None

    fallback = _fallback(briefing)
    cloudflare_client = CloudflareAIClient()
    if not cloudflare_client.configured:
        return fallback

    prompt = (
        "Write one concise paragraph summarizing this Missouri fire-weather "
        "briefing for an RSS feed. Use only the supplied JSON. Use mph and "
        "percentages. Report precipitation only in inches and never above "
        "statewide.precip_in.max. Never mention a danger class absent from "
        "statewide.fire_danger_present. Do not give advice or use headings.\n\n"
        + briefing_json(briefing)
    )
    try:
        text = cloudflare_client.generate_text(prompt)
        if text and _valid_summary(text, briefing):
            return text
        print("Gemini RSS summary failed numeric validation; omitting summary item.")
    except Exception as exc:
        print(f"Gemini RSS summary failed; omitting summary item: {exc}")
    return None


if __name__ == "__main__":
    print(generate_summary() or "No summary available.")

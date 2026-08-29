"""Generate the optional RSS summary from numeric forecast data."""

import os
from pathlib import Path

from dotenv import load_dotenv
from ai.briefing import (
    briefing_json,
    build_briefing,
    contains_core_weather_facts,
    validate_briefing_text,
    validate_operational_style,
)
from ai.cloudflare import CloudflareAIClient

load_dotenv()

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parent.parent))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", BASE_DIR / "archive" / "forecasts"))


def _fallback(briefing: dict) -> str:
    state = briefing["statewide"]
    danger = state["highest_fire_danger"] or "Low"
    rain = (
        f", with precipitation up to {state['precip_in']['max']:.2f} inches"
        if state["precipitation"] == "measurable" and state["precip_in"]["max"] is not None
        else ""
    )
    return (
        f"{danger} is the highest forecast fire-danger class represented by the available station and county data. "
        f"Minimum relative humidity ranges from {round(state['rh']['min'])}% to {round(state['rh']['max'])}%, "
        f"fuel moisture ranges from {round(state['fuel_moisture']['min'])}% to "
        f"{round(state['fuel_moisture']['max'])}%, with peak winds up to "
        f"{round(state['wind_mph']['max'])} mph where available{rain}."
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
        "briefing for an RSS feed, using no more than 6 sentences and fewer "
        "when sufficient. Use only the supplied JSON. Use mph and "
        "percentages. Report precipitation only in inches and never above "
        "statewide.precip_in.max. Never mention a danger class absent from "
        "statewide.fire_danger_present. Do not give advice or use headings.\n\n"
        + briefing_json(briefing)
    )
    try:
        text = cloudflare_client.generate_text(prompt)
        if (
            text
            and _valid_summary(text, briefing)
            and contains_core_weather_facts(text, briefing)
            and validate_operational_style(text, briefing)
        ):
            return text
        print("Cloudflare RSS summary failed operational validation; omitting summary item.")
    except Exception as exc:
        print(f"Cloudflare RSS summary failed; omitting summary item: {exc}")
    return None


if __name__ == "__main__":
    print(generate_summary() or "No summary available.")

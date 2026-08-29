"""Generate forecast text from numeric forecast artifacts.

Images are published for people, but are intentionally not sent to an AI model.
Use ``--test`` to preview a generated forecast without writing to the database.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parent.parent))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", BASE_DIR / "archive" / "forecasts"))

sys.path.append(str(BASE_DIR))
from ai.briefing import (
    briefing_json,
    build_briefing,
    contains_core_weather_facts,
    validate_briefing_text,
)
from ai.cloudflare import CloudflareAIClient
from core.database import insert_forecast


def valid_text(text: str, briefing: dict) -> bool:
    """Reject prose that adds unsupported danger classes or rainfall totals."""
    return validate_briefing_text(text, briefing)


def fallback_text(briefing: dict) -> tuple[str, str]:
    state = briefing["statewide"]
    highest = state["highest_fire_danger"] or "Low"
    precip = state["precip_in"]
    if state["precipitation"] == "trace":
        rain_text = "forecast precipitation is trace statewide"
    elif precip["max"] is not None:
        rain_text = f"forecast precipitation ranges up to {precip['max']:.3f} inches"
    else:
        rain_text = "forecast precipitation is unavailable"

    headline = f"{highest} Fire Danger Across Missouri"
    discussion = (
        f"{highest} is the highest forecast fire-danger class represented by the available station and county data. "
        f"Statewide minimum relative humidity ranges from {state['rh']['min']}% to {state['rh']['max']}%, "
        f"fuel moisture ranges from {state['fuel_moisture']['min']}% to {state['fuel_moisture']['max']}%, "
        f"and peak winds range from {state['wind_mph']['min']} to {state['wind_mph']['max']} mph where available. "
        f"{rain_text}."
    )
    return headline, discussion


def generate_text(
    client: CloudflareAIClient,
    prompt: str,
    briefing: dict,
    require_core_facts: bool = False,
) -> str | None:
    """Generate validated text, retrying once with stricter instructions."""
    for attempt in range(2):
        try:
            text = client.generate_text(prompt)
            if text and valid_text(text, briefing) and (
                not require_core_facts or contains_core_weather_facts(text, briefing)
            ):
                return text
            prompt = (
                "Rewrite using only the supplied JSON. Do not mention a danger class "
                "absent from fire_danger_present or precipitation above precip_in.max. "
                "The discussion must include numeric relative humidity, fuel moisture, "
                "and wind values from statewide. Return only the requested answer.\n\n"
                + prompt
            )
        except Exception as exc:
            print(f"Cloudflare AI attempt {attempt + 1} failed: {exc}")
    return None


def generate_forecast_text(
    briefing: dict,
    client: CloudflareAIClient | None = None,
) -> tuple[str, str]:
    """Return a headline and discussion without persisting anything."""
    headline, discussion = fallback_text(briefing)
    client = client or CloudflareAIClient()
    if not client.configured:
        return headline, discussion

    data = briefing_json(briefing)
    headline_prompt = (
        "Write a factual 5-8 word headline for a Missouri fire-weather forecast. "
        "Use only this JSON and mention only classes in statewide.fire_danger_present. "
        "Return plain text only.\n\n" + data
    )
    summary_prompt = (
        "Write a concise 3-4 sentence Missouri fire-weather forecast discussion. "
        "Use only this JSON. Report precipitation only in inches, never above "
        "statewide.precip_in.max, and never mention a danger class absent from "
        "statewide.fire_danger_present. Use mph and percentages. State facts only; "
        "no recommendations, headings, or markdown.\n\n" + data
    )
    return (
        generate_text(client, headline_prompt, briefing) or headline,
        generate_text(client, summary_prompt, briefing, require_core_facts=True) or discussion,
    )


def main(
    test_mode: bool = False,
    forecast_path: str | None = None,
    client: CloudflareAIClient | None = None,
) -> tuple[str, str]:
    briefing = build_briefing(ARCHIVE_DIR, forecast_path=forecast_path)
    print(f"Built numeric briefing from {briefing['source_file']}")
    headline, discussion = generate_forecast_text(briefing, client=client)

    current_date = datetime.now(timezone.utc)
    print("\n" + "=" * 60)
    print(f"{headline} - {current_date.strftime('%B %d, %Y')}")
    print("=" * 60)
    print(discussion)

    if not test_mode:
        valid_time = current_date.replace(hour=12, minute=0, second=0, microsecond=0)
        forecast_id = insert_forecast(
            valid_time=valid_time,
            title=headline,
            discussion=discussion,
        )
        print(f"\nForecast saved to database with ID: {forecast_id}")
    else:
        print("\nTEST MODE: forecast was not saved to the database.")
    return headline, discussion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the daily forecast text.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Preview the forecast and never write to the database.",
    )
    parser.add_argument(
        "--forecast-file",
        help="Use a specific station forecast JSON instead of the newest archive file.",
    )
    args = parser.parse_args()
    main(test_mode=args.test, forecast_path=args.forecast_file)

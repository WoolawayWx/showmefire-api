"""Generate forecast text from numeric forecast artifacts.

Images are published for people, but are intentionally not sent to an AI model.
Use ``--test`` to preview a generated forecast without writing to the database.
"""

import argparse
import os
import re
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
    REGION_NAMES,
    validate_briefing_text,
    validate_operational_style,
)
from ai.cloudflare import CloudflareAIClient
from core.database import insert_forecast


def valid_text(text: str, briefing: dict) -> bool:
    """Reject prose that adds unsupported danger classes or rainfall totals."""
    return validate_briefing_text(text, briefing)


def valid_headline(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:Low|Moderate|Elevated|Critical|Extreme)"
            r"(?: to (?:Low|Moderate|Elevated|Critical|Extreme))?"
            r" Fire Danger (?:Across|in) Missouri",
            text.strip(),
            re.IGNORECASE,
        )
    )


def fallback_text(briefing: dict) -> tuple[str, str]:
    state = briefing["statewide"]
    highest = state["highest_fire_danger"] or "Low"
    present = state.get("fire_danger_present") or [highest]
    lowest = present[0]
    precip = state["precip_in"]
    rain_text = (
        f" Measurable precipitation is possible, up to {precip['max']:.2f} inches."
        if state["precipitation"] == "measurable" and precip["max"] is not None
        else ""
    )
    headline = (
        f"{lowest} to {highest} Fire Danger Across Missouri"
        if lowest != highest
        else f"{highest} Fire Danger Across Missouri"
    )
    regional_classes = {}
    for region_name, region in briefing["regions"].items():
        region_highest = region.get("highest_fire_danger")
        if region.get("station_count") and region_highest:
            regional_classes.setdefault(region_highest, []).append(
                REGION_NAMES.get(region_name, region_name)
            )
    driest_region = min(
        (
            region for region in briefing["regions"].values()
            if region.get("station_count") and region["rh"]["min"] is not None
        ),
        key=lambda region: region["rh"]["min"],
        default=None,
    )
    strongest_region = max(
        (
            region for region in briefing["regions"].values()
            if region.get("station_count") and region["wind_mph"]["max"] is not None
        ),
        key=lambda region: region["wind_mph"]["max"],
        default=None,
    )
    driest_name = next(
        (
            REGION_NAMES.get(name, name) for name, region in briefing["regions"].items()
            if region is driest_region
        ),
        "the driest areas",
    )
    strongest_name = next(
        (
            REGION_NAMES.get(name, name) for name, region in briefing["regions"].items()
            if region is strongest_region
        ),
        "the strongest-wind areas",
    )
    lower_classes = [
        f"{danger.lower()} conditions are expected in "
        f"{', '.join(regions[:-1]) + ', and ' if len(regions) > 1 else ''}{regions[-1]} Missouri"
        for danger, regions in regional_classes.items()
        if danger != highest
    ]
    highest_regions = ", ".join(regional_classes.get(highest, []))
    regional_sentence = (
        f"The highest fire danger is expected in {highest_regions} Missouri, while "
        f"{' and '.join(lower_classes)}."
        if highest_regions and lower_classes
        else f"{highest} fire danger is expected across Missouri."
    )
    peak_class = state.get("highest_station_fire_danger") or highest
    peak_regions = regional_classes.get(peak_class, [])
    peak_start = state.get("fire_danger_peak_start")
    peak_end = state.get("fire_danger_peak_end")
    if peak_start and peak_end and peak_start != peak_end:
        peak_time = f"between {peak_start} and {peak_end}"
    elif peak_start:
        peak_time = f"around {peak_start}"
    else:
        peak_time = None
    peak_location = ", ".join(peak_regions)
    if not peak_location:
        county_locations = (
            briefing.get("county_danger", {})
            .get("counties_by_fire_danger", {})
            .get(highest, [])
        )
        peak_location = ", ".join(county_locations[:5])
    peak_sentence = (
        f"Peak {peak_class.lower()} fire danger in {peak_location} Missouri is "
        f"expected {peak_time}."
        if peak_time and peak_location
        else ""
    )
    wind_direction = state.get("predominant_wind_direction")
    wind_start = state.get("wind_direction_start")
    wind_end = state.get("wind_direction_end")
    if wind_start and wind_end and wind_start != wind_end:
        wind_direction_text = (
            f" Winds turn from the {wind_start} early in the period toward the "
            f"{wind_end} direction later."
        )
    elif wind_direction:
        wind_direction_text = f" Winds are predominantly from the {wind_direction}."
    else:
        wind_direction_text = ""
    discussion = (
        f"Fire danger will range from {lowest.lower()} to {highest.lower()} across "
        f"Missouri, with maximum temperatures ranging from {round(state['temp_f']['min'])}°F "
        f"to {round(state['temp_f']['max'])}°F. "
        + regional_sentence
        + " "
        + peak_sentence
        + (" " if peak_sentence else "")
        + f"The driest air will be in {driest_name} Missouri, where humidity falls to "
        f"{round(driest_region['rh']['min']) if driest_region else round(state['rh']['min'])}%; "
        f"fuel moisture ranges from {round(state['fuel_moisture']['min'])}% to "
        f"{round(state['fuel_moisture']['max'])}%, and winds remain generally light, "
        f"with the strongest values near {round(state['wind_mph']['max'])} mph in "
        f"{strongest_name} Missouri.{wind_direction_text}"
        + rain_text
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
                not require_core_facts
                or (
                    contains_core_weather_facts(text, briefing)
                    and validate_operational_style(text, briefing)
                )
            ):
                return text
            prompt = (
                "Rewrite using only the supplied JSON. Do not mention a danger class "
                "absent from fire_danger_present or precipitation above precip_in.max. "
                "The discussion must include rounded whole-number relative humidity, "
                "fuel moisture, and wind values from statewide. Omit precipitation "
                "when it is trace or unavailable. Combine regions with similar "
                "conditions instead of repeating the same sentence for each region. "
                "Return only the requested answer.\n\n"
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
        "Do not include a date. Return plain text only.\n\n" + data
    )
    summary_prompt = (
        "Act as an NWS meteorologist writing a concise fire-weather forecast "
        "discussion in AFD style. Use 4-6 well-formed sentences when regional conditions differ, "
        "using fewer only when the data is uniform. Start with the overall statewide "
        "pattern, then synthesize the meaningful regional differences and fire-weather "
        "drivers into natural human meteorologist prose. Combine regions with similar "
        "conditions and emphasize the primary contrast; do not write one repetitive "
        "sentence per region. Do not mechanically recite every JSON field, use field "
        "labels such as 'RH', or write a list of regions. "
        "Prioritize smooth grammar, complete sentences, and natural transitions over "
        "mentioning every available number. "
        "Use full region names such as Northwest and Southwest. If wind direction "
        "data is present, describe the predominant direction or a meaningful shift "
        "through the forecast period; do not invent a direction when it is absent. "
        "State when the peak danger is expected and identify the most affected regions "
        "or counties when the briefing provides them. Mention other locations only "
        "when they add meaningful contrast. Do not invent exact timing beyond the "
        "supplied peak window. "
        "Use only this JSON. Report precipitation only in inches, never above "
        "statewide.precip_in.max, and never mention a danger class absent from "
        "statewide.fire_danger_present. Round temperature, RH, wind, and fuel "
        "moisture to whole numbers. Omit precipitation when it is trace or "
        "unavailable. Do not invent fronts, timing, cloud cover, confidence, "
        "causes, or impacts that are not in the JSON. Do not give recommendations, "
        "use headings, report averages, or use markdown.\n\n" + data
    )
    generated_headline = generate_text(client, headline_prompt, briefing)
    generated_discussion = generate_text(
        client,
        summary_prompt,
        briefing,
        require_core_facts=True,
    )
    return (
        generated_headline if generated_headline and valid_headline(generated_headline) else headline,
        generated_discussion or discussion,
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
    print(headline)
    print(f"Forecast date: {current_date.strftime('%B %d, %Y')}")
    print("=" * 60)
    print(discussion)

    if not test_mode:
        cycle_hour = int(os.getenv("FORECAST_CYCLE_HOUR", "12"))
        valid_time = current_date.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        forecast_id = insert_forecast(
            valid_time=valid_time,
            title=headline,
            discussion=discussion,
            cycle=cycle_hour,
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

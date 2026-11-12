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
    regions = {name: region for name, region in briefing["regions"].items() if region["station_count"]}
    priority = [
        name for name, region in regions.items()
        if region["highest_fire_danger"] == highest
    ]
    peak_start = state.get("fire_danger_peak_start")
    peak_end = state.get("fire_danger_peak_end")
    timing = ""
    if peak_start and state.get("highest_station_fire_danger") == highest:
        timing = f" around {peak_start}" if peak_start == peak_end else f" between {peak_start} and {peak_end}"
    if priority:
        overview = f"Highest concern is {' and '.join(priority)} Missouri, where fire danger reaches {highest}{timing}."
    else:
        overview = f"The forecast reaches {highest} fire danger{timing}."

    groups = {}
    for name, region in sorted(regions.items(), key=lambda item: item[0] not in priority):
        facts = []
        for key, bound, wording in (
            ("rh", "min", "relative humidity down to {}%"),
            ("fuel_moisture", "min", "fuel moisture down to {}%"),
            ("wind_mph", "max", "sustained winds up to {} mph"),
            ("gust_mph", "max", "gusts up to {} mph"),
        ):
            value = region.get(key, {}).get(bound)
            if value is not None:
                facts.append(wording.format(round(value)))
        groups.setdefault((region["highest_fire_danger"], tuple(facts)), []).append(name)

    priority_details = []
    regional_details = []
    for (danger, facts), names in groups.items():
        location = " and ".join(names) + " Missouri"
        detail = ", ".join(facts[:-1]) + f", and {facts[-1]}" if len(facts) > 2 else " and ".join(facts)
        if danger == highest:
            if detail:
                priority_details.append(f"In {location}, expect {detail}.")
        elif danger:
            detail = f" with {detail}" if detail else ""
            regional_details.append(f"Fire danger reaches {danger} in {location}{detail}.")

    regional_timing = []
    for name, region in regions.items():
        start = region.get("fire_danger_peak_start")
        end = region.get("fire_danger_peak_end")
        if name in priority or not start:
            continue
        if start == end:
            regional_timing.append(f"{name} Missouri stations reach their peak around {start}.")
        elif end:
            regional_timing.append(f"In {name} Missouri, stations reach their peak at times from {start} to {end}.")

    shifts = {}
    for name, region in regions.items():
        start = region.get("wind_direction_start")
        end = region.get("wind_direction_end")
        if start and end and start != end:
            shifts.setdefault((start, end), []).append(name)
    wind = [
        f"In {' and '.join(names)} Missouri, winds shift from {start} to {end} over the forecast period."
        for (start, end), names in shifts.items()
    ]

    smoke = ""
    if dispersion := briefing.get("forecast_dispersion"):
        hours = dispersion["hours"]
        peak_index = max(range(len(hours)), key=lambda index: hours[index]["ventilation_kt_ft"])
        last_category = hours[-1]["smoke_dispersal"]
        late_index = len(hours) - 1
        while late_index > peak_index + 1 and hours[late_index - 1]["smoke_dispersal"] == last_category:
            late_index -= 1
        first, peak, late = hours[0], hours[peak_index], hours[late_index]
        smoke = (
            f"HRRR starts the period with {first['smoke_dispersal'].lower()} dispersal at {first['time']} "
            f"and mixing heights near {round(first['mixing_height_ft'], -2):,} ft above ground."
        )
        if peak_index:
            smoke += (
                f" Ventilation peaks at {peak['time']}, with {peak['smoke_dispersal'].lower()} dispersal, "
                f"mixing heights near {round(peak['mixing_height_ft'], -2):,} ft, "
                f"and transport winds near {peak['transport_speed_mph']} mph."
            )
        if late_index != peak_index:
            smoke += (
                f" By {late['time']}, HRRR forecasts {late['smoke_dispersal'].lower()} dispersal "
                f"with mixing heights near {round(late['mixing_height_ft'], -2):,} ft."
            )

    discussion = "\n\n".join(filter(None, (
        " ".join([overview, *priority_details]),
        " ".join([*regional_details, *regional_timing, rain_text.strip()]).strip(),
        " ".join(wind),
        smoke,
    )))
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
            text = client.generate_text(prompt, max_tokens=950 if require_core_facts else 64)
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

    headline_data = briefing_json({"statewide": {"fire_danger_present": briefing["statewide"]["fire_danger_present"]}})
    data = briefing_json(briefing)
    headline_prompt = (
        "Write a factual 5-8 word headline for a Missouri fire-weather forecast. "
        "Use only this JSON and mention only classes in statewide.fire_danger_present. "
        "Do not include a date. Return plain text only.\n\n" + headline_data
    )
    summary_prompt = (
        "Write a Missouri fire-weather memo that opens like an incident command briefing and develops "
        "into a plain-language forecast discussion in the style of an NWS Area Forecast Discussion. "
        "Aim for 250-400 words across 4-5 plain-text paragraphs when the data supports that detail. "
        "Use a brief opening for the main concern, then develop the forecast evolution, regional "
        "differences, and smoke and mixing guidance. Explain what changes through the operational period "
        "and connect the timing to fire-weather drivers only where the supplied data supports it. "
        "Lead with the highest concern, where it is, and when crews can expect it. Follow with the "
        "wind shifts and regional contrasts that matter during the operational period. Group areas "
        "with similar conditions and keep sentences easy to read aloud. Mention minimum relative "
        "humidity, minimum fuel moisture, and peak sustained winds where they explain the concern. "
        "The ranges describe daily station extrema, which may occur at different hours. "
        "Use gust_mph.max where available and meaningful, identifying it as the forecast peak gust "
        "and keeping it distinct from sustained wind. Omit missing values. Avoid a repetitive list "
        "of statistics or field labels such as 'RH'. "
        "Prioritize smooth grammar, complete sentences, and natural transitions over "
        "mentioning every available number. "
        "Use full region names such as Northwest and Southwest. If wind direction "
        "data is present, describe the predominant direction or a meaningful shift "
        "through the forecast period; do not invent a direction when it is absent. "
        "State when the peak danger is expected and identify the most affected regions "
        "or counties when the briefing provides them. Mention other locations only "
        "when they add meaningful contrast. Treat peak windows as the earliest and latest "
        "times at which stations reach the area's highest class, not a continuous period "
        "at that class across the area. Do not infer exact wind-shift timing. "
        "Use only this JSON. Report precipitation only in inches, never above "
        "statewide.precip_in.max, and never mention a danger class absent from "
        "statewide.fire_danger_present. Round temperature, RH, wind, and fuel "
        "moisture to whole numbers. Omit precipitation when it is trace or "
        "unavailable. Do not invent fronts, timing, cloud cover, confidence, "
        "causes, or impacts that are not in the JSON. Do not give recommendations, "
        "use headings, report averages, or use markdown. "
        "When forecast_dispersion is available, weave smoke dispersal and mixing heights into "
        "the forecast narrative alongside the other weather changes. Describe when dispersal is "
        "limited, when it improves or is strongest, and when it weakens if the hourly values show "
        "those changes. Include representative mixing heights in feet above ground and transport "
        "wind speeds when they help explain the trend. These are statewide grid medians and the "
        "dispersal category corresponds to the median ventilation, so avoid applying these values "
        "to individual regions or extending them outside the supplied hours. Omit dispersion if "
        "forecast_dispersion is absent.\n\n" + data
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

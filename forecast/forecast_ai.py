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
    DANGER_LABELS,
    build_briefing,
    contains_core_weather_facts,
    REGION_NAMES,
    briefing_text_problem,
    validate_briefing_text,
    validate_operational_style,
)
from ai.cloudflare import CloudflareAIClient
from core.database import insert_forecast


def valid_text(text: str, briefing: dict) -> bool:
    """Reject prose that adds unsupported danger classes or rainfall totals."""
    return validate_briefing_text(text, briefing)


def valid_headline(text: str) -> bool:
    classes = [label for label in DANGER_LABELS if re.search(rf"\b{label}\b", text, re.IGNORECASE)]
    positions = [text.lower().index(label.lower()) for label in classes]
    if len(classes) == 2 and positions[0] > positions[1]:
        return False  # "Elevated to Moderate": classes must run low to high
    return bool(
        re.fullmatch(
            r"(?:Low|Moderate|Elevated|Critical|Extreme)"
            r"(?: to (?:Low|Moderate|Elevated|Critical|Extreme))?"
            r" Fire Danger (?:Across|in) Missouri",
            text.strip(),
            re.IGNORECASE,
        )
    )


# Everything that never changes lives here and is sent as the system message, so Workers AI prefix
# caching can reuse it (it is identical for the headline and the discussion). Keep dates and any other
# per-run text OUT of it; bump FORECAST_SESSION_ID when this text changes.
FORECAST_SESSION_ID = "smf-forecast-discussion-v1"
# Reasoning models (e.g. Kimi K2.5) spend this budget on thinking before the answer, so set
# CLOUDFLARE_AI_FORECAST_MAX_TOKENS (e.g. 4000) when CLOUDFLARE_AI_FORECAST_MODEL is one of them.
DISCUSSION_MAX_TOKENS = int(os.getenv("CLOUDFLARE_AI_FORECAST_MAX_TOKENS", "1100") or 1100)
HEADLINE_MAX_TOKENS = max(60, DISCUSSION_MAX_TOKENS // 4 if DISCUSSION_MAX_TOKENS > 1100 else 60)

DISCUSSION_SYSTEM_PROMPT = """You write the daily Missouri fire-weather forecast discussion for Show Me Fire, in the style of a National Weather Service forecast discussion (AFD), from one block of processed forecast values (JSON). The audience is fire departments, land managers and the public.

SOURCE RULES
- Use only the supplied values. Never add fronts, cloud cover, causes, confidence, impacts, observations or any value that is not in it.
- This is a forecast for the day ahead. Never describe current conditions, smoke dispersal, mixing height or transport winds.
- Danger classes, in order: Low, Moderate, Elevated, Critical, Extreme. Mention only classes listed in statewide.fire_danger_present.
- Round temperature, relative humidity, wind and fuel moisture to whole numbers. Give precipitation in inches, never above the data's maximum for the area you are describing.
- Times are Central Time and come only from the data's time fields (peak windows, min_rh_time, max_wind_time, precip_window, danger_trend). Do not invent other times.

STRUCTURE AND LENGTH
Write 11 to 14 complete sentences in three short paragraphs separated by a blank line (never fewer than 11 unless the data is uniform), no headings, no lists, no markdown.
1. Statewide pattern: the range of fire danger, the temperature range, and the main drivers (humidity, fuel moisture, wind).
2. Spatial detail: go region by region (Northwest, Northeast, Central, Southwest, Southeast) where conditions differ. For each region give its highest danger class, its humidity, fuel moisture and wind values, and when danger peaks or whether it is building or easing (danger_trend). For each region that has counties_above_low in county_danger.by_region, name those listed with their class, using a separate sentence for each class (at most five counties in any one sentence, so never combine two classes' lists), as "Boone County" or "Boone and Callaway Counties"; name only counties that are in the data. Put each county in the paragraph or sentence for the region it is listed under in by_region, and say the region's name in that same sentence (for example "In the Central region, Boone and Callaway Counties reach Moderate"); never name a county under a different region. Group regions that share the same conditions instead of repeating yourself.
3. Wind and precipitation: wind direction (predominant, or a shift from wind_direction_start to wind_direction_end) and where it is strongest. Mention precipitation only for regions whose precipitation is "measurable", with that region's amount and its precip_window, described by region only (the data has no county rainfall, so never name a county in a sentence about rain); leave it out entirely when it is trace or unavailable and never say there is none.
4. If the data has red_flag, say which warning or watch is active, for which counties, and until when. If it has no red_flag key, say nothing about warnings or watches.

UNITS AND WORDING
Wind values are sustained speeds: say "winds", never "gusts". Always write units: "30%" or "30 percent", "9 mph", "0.09 inches", "88°F". Say each fact once; do not repeat a region's conditions in a later paragraph. Never mention what the data lacks, never refer to "the data" or "JSON", and never say a warning, watch or county list is unavailable: if something is not in the data, leave it out silently.

STYLE
Plain, precise meteorologist prose with natural transitions. Do not recite every field, use field names or abbreviations such as RH, give averages, make recommendations, or mention burn bans. Return only the requested text."""

HEADLINE_TASK = (
    "Task: ignore the length and structure rules above and write only a factual 5-8 word headline of the form 'Elevated Fire Danger Across Missouri' "
    "(or 'Moderate to Critical Fire Danger in Missouri'), using only classes in "
    "statewide.fire_danger_present, no date, plain text only.\n\nData:\n"
)
DISCUSSION_TASK = "Task: write today's forecast discussion following the rules above.\n\nData:\n"
REWRITE_NOTE = (
    "\n\nYour previous answer was rejected. Rewrite it using only the supplied values (never use the word 'briefing'): no danger class outside "
    "statewide.fire_danger_present, no precipitation above the area's precip_in.max, no county that is not "
    "listed, no smoke dispersal or current conditions, and include whole-number relative humidity, fuel "
    "moisture and wind values from statewide. Omit precipitation when it is trace or unavailable. "
    "Return only the requested text."
)


def _and(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _counties(names: list[str], limit: int | None = None, total: int | None = None) -> str:
    """"Boone and Callaway Counties", or "A, B, C, D, E and 10 other counties" when more exist than are shown
    (`total` is the real count when `names` was already capped)."""
    shown = names if limit is None or len(names) <= limit else names[:limit]
    count = max(total or 0, len(names))
    if len(shown) < count:
        return f"{', '.join(shown)} and {count - len(shown)} other counties"
    return f"{_and(shown)} {'County' if len(names) == 1 else 'Counties'}"


def _span(window: dict | None) -> str:
    if not window or not window.get("from"):
        return ""
    if window.get("to") and window["to"] != window["from"]:
        return f"between {window['from']} and {window['to']}"
    return f"around {window['from']}"


def fallback_text(briefing: dict) -> tuple[str, str]:
    """Deterministic headline + discussion from the briefing, used when the model is
    unavailable or its output fails validation. Same structure as the model's."""
    state = briefing["statewide"]
    highest = state["highest_fire_danger"] or "Low"
    present = state.get("fire_danger_present") or [highest]
    lowest = present[0]
    headline = (
        f"{lowest} to {highest} Fire Danger Across Missouri"
        if lowest != highest
        else f"{highest} Fire Danger Across Missouri"
    )
    populated = {
        REGION_NAMES.get(name, name): region
        for name, region in briefing["regions"].items()
        if region.get("station_count")
    }
    county_regions_detail = (briefing.get("county_danger") or {}).get("by_region", {})

    def whole(value) -> str:
        return str(round(value))

    # Paragraph 1: statewide pattern.
    opening = (
        f"Fire danger will range from {lowest.lower()} to {highest.lower()} across Missouri, "
        f"with maximum temperatures from {whole(state['temp_f']['min'])}°F to {whole(state['temp_f']['max'])}°F. "
        if lowest != highest
        else f"{highest} fire danger is forecast across Missouri, with maximum temperatures from "
        f"{whole(state['temp_f']['min'])}°F to {whole(state['temp_f']['max'])}°F. "
    )
    opening += (
        f"Minimum relative humidity ranges from {whole(state['rh']['min'])}% to {whole(state['rh']['max'])}%, "
        f"fuel moisture bottoms out between {whole(state['fuel_moisture']['min'])}% and "
        f"{whole(state['fuel_moisture']['max'])}%, and winds peak near {whole(state['wind_mph']['max'])} mph."
    )

    # Paragraph 2: region by region, naming counties above Low.
    sentences = []
    for region_name, region in populated.items():
        danger = region.get("highest_fire_danger") or "Low"
        sentence = (
            f"In the {region_name} region, fire danger reaches {danger}"
        )
        trend = region.get("danger_trend") or {}
        if trend.get("direction") in ("building", "easing"):
            sentence += f" and is {trend['direction']} through the period"
        sentence += (
            f", with humidity near {whole(region['rh']['min'])}%"
            f"{' by ' + region['min_rh_time']['from'] if region.get('min_rh_time') else ''}, "
            f"fuel moisture near {whole(region['fuel_moisture']['min'])}%, "
            f"and winds to {whole(region['wind_mph']['max'])} mph."
        )
        sentences.append(sentence)
        detail = county_regions_detail.get(region_name, {})
        names = (detail.get("counties_above_low") or {}).get(danger)
        if names:
            total = (detail.get("class_counts") or {}).get(danger) or len(names)
            sentences.append(
                f"In the {region_name} region, {_counties(names, 5, total)} reach {danger}."
            )
    peak_class = state.get("highest_station_fire_danger") or highest
    peak_time = _span(
        {"from": state.get("fire_danger_peak_start"), "to": state.get("fire_danger_peak_end")}
        if state.get("fire_danger_peak_start") else None
    )
    if peak_time:
        sentences.append(f"Peak {peak_class.lower()} fire danger is expected {peak_time}.")
    regional = " ".join(sentences)

    # Paragraph 3: wind and precipitation.
    wind_bits = []
    strongest = max(
        populated.items(), key=lambda item: item[1]["wind_mph"]["max"] or 0, default=None
    )
    if strongest and strongest[1]["wind_mph"]["max"] is not None:
        when = _span(strongest[1].get("max_wind_time"))
        wind_bits.append(
            f"The strongest winds, near {whole(strongest[1]['wind_mph']['max'])} mph, are expected in the "
            f"{strongest[0]}{' ' + when if when else ''}."
        )
    start, end, predominant = (
        state.get("wind_direction_start"), state.get("wind_direction_end"),
        state.get("predominant_wind_direction"),
    )
    if start and end and start != end:
        wind_bits.append(f"Winds turn from the {start} early in the period toward the {end} later.")
    elif predominant:
        wind_bits.append(f"Winds are predominantly from the {predominant}.")
    rain = [
        (name, region) for name, region in populated.items()
        if region.get("precipitation") == "measurable" and region["precip_in"]["max"] is not None
    ]
    if rain:
        parts = []
        for name, region in rain:
            window = _span(region.get("precip_window"))
            parts.append(f"the {name} (up to {region['precip_in']['max']:.2f} inches{', ' + window if window else ''})")
        wind_bits.append(f"Measurable rainfall is forecast in {_and(parts)}.")
    elif state.get("precipitation") == "measurable" and state["precip_in"]["max"] is not None:
        wind_bits.append(f"Measurable rainfall is possible, up to {state['precip_in']['max']:.2f} inches.")
    paragraphs = [opening, regional, " ".join(wind_bits)]

    # Paragraph 4: Red Flag Warnings / Fire Weather Watches, only when active.
    alerts = []
    for alert in briefing.get("red_flag") or []:
        until = f" until {alert['ends']}" if alert.get("ends") else ""
        alerts.append(f"A {alert['event']} is in effect for {_counties(alert['counties'])}{until}.")
    if alerts:
        paragraphs.append(" ".join(alerts))
    return headline, "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())


def generate_text(
    client: CloudflareAIClient,
    prompt: str,
    briefing: dict,
    require_core_facts: bool = False,
    max_tokens: int = 320,
) -> str | None:
    """Generate validated text, retrying once with a corrective note.

    The static instructions go in the system message (cacheable prefix, shared by the headline and
    the discussion); `prompt` is the dynamic part. The retry note is appended to the END of the
    prompt so the cached prefix is never disturbed."""
    for attempt in range(2):
        try:
            # CLOUDFLARE_AI_FORECAST_MODEL picks a model for the discussion only (e.g. one with formal
            # prompt caching such as @cf/moonshotai/kimi-k2.5); unset keeps the client default.
            model = os.getenv("CLOUDFLARE_AI_FORECAST_MODEL", "").strip()
            text = client.generate_text(
                prompt,
                system=DISCUSSION_SYSTEM_PROMPT,
                max_tokens=max_tokens,
                session_id=FORECAST_SESSION_ID,
                **({"model": model} if model else {}),
            )
            if text:  # tidy blank lines/trailing spaces so paragraphs are separated by exactly one blank line
                text = "\n\n".join(
                    " ".join(part.split()) for part in re.split(r"\n\s*\n", text.strip()) if part.strip()
                )
            failed = []
            if not text:
                failed.append("empty")
            else:
                problem = None if valid_text(text, briefing) else (briefing_text_problem(text, briefing) or "validate_briefing_text")
                if problem:
                    failed.append(f"validate_briefing_text: {problem}")
                if require_core_facts:
                    if not contains_core_weather_facts(text, briefing):
                        failed.append("contains_core_weather_facts")
                    if not validate_operational_style(text, briefing):
                        failed.append("validate_operational_style")
            if text and not failed:
                return text
            print(f"Cloudflare AI attempt {attempt + 1} rejected by: {', '.join(failed)}")
            if REWRITE_NOTE not in prompt:
                prompt += REWRITE_NOTE
        except Exception as exc:
            print(f"Cloudflare AI attempt {attempt + 1} failed: {exc}")
    return None


def generate_forecast_text(
    briefing: dict,
    client: CloudflareAIClient | None = None,
) -> tuple[str, str]:
    """Return a headline and discussion without persisting anything."""
    headline, discussion = fallback_text(briefing)
    # This runs from a cron script, so a slow reasoning model may take longer than the client's 60 s default.
    client = client or CloudflareAIClient(timeout=float(os.getenv("CLOUDFLARE_AI_FORECAST_TIMEOUT", "180") or 180))
    if not client.configured:
        return headline, discussion

    data = briefing_json(briefing)
    generated_headline = generate_text(
        client, HEADLINE_TASK + data, briefing, max_tokens=HEADLINE_MAX_TOKENS
    )
    generated_discussion = generate_text(
        client,
        DISCUSSION_TASK + data,
        briefing,
        require_core_facts=True,
        max_tokens=DISCUSSION_MAX_TOKENS,
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

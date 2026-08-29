"""Create a forecast title from the numeric briefing using Workers AI."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", PROJECT_ROOT / "archive" / "forecasts"))
sys.path.append(str(PROJECT_ROOT))

from ai.briefing import briefing_json, build_briefing, validate_briefing_text
from ai.cloudflare import CloudflareAIClient

briefing = build_briefing(ARCHIVE_DIR)
highest = briefing["statewide"].get("highest_fire_danger") or "Low"
fallback = f"{highest} Fire Danger Across Missouri"
client = CloudflareAIClient()

if client.configured:
    prompt = (
        "Write one factual, informative 5-8 word headline for a Missouri "
        "fire-weather RSS feed. Use only the supplied numeric briefing JSON. "
        "Mention only fire-danger classes listed in statewide.fire_danger_present. "
        "Do not include a date. Return plain text only.\n\n" + briefing_json(briefing)
    )
    try:
        generated_title = client.generate_text(prompt)
        title = (
            generated_title
            if validate_briefing_text(generated_title, briefing)
            else fallback
        )
    except Exception as exc:
        print(f"Cloudflare title generation failed; using fallback: {exc}")
        title = fallback
else:
    title = fallback

print(title)

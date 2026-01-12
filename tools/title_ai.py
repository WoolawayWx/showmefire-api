from google import genai
import PIL.Image
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# Determine project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

genai_key = os.getenv('genai_key')
client = genai.Client(api_key=genai_key)

img_path = PROJECT_ROOT / "images" / "mo-realtimefiredanger.png"
img = PIL.Image.open(img_path)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        "Look at this graphic, and create a short informative headline",
        "Needs to be useable for a RSS feed that will display in fire departments.",
        "Need to be regionally correct in Missouri, so like NE, SW, Central, etc.",
        "Just respond with one singular headline.",
        "Use the legend on the side of the image to correctly address the level of fire danger.",
        "green is low, yellow is moderate, orange is elevated, red is critical, and maroon is extreme.",
        "get teh correct risk to color",
        img
    ]
)

print(response.text)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.core.database import insert_forecast
from datetime import timezone
from google import genai
import PIL.Image
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

genai_key = os.getenv('genai_key')
client = genai.Client(api_key=genai_key)

images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
archive_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'forecasts'))

# Get the most recent forecast JSON
forecast_date = datetime.now().strftime('%Y%m%d')
forecast_hour = '12'
forecast_json_path = os.path.join(archive_dir, f"forecast_{forecast_date}_{forecast_hour}.json")

# Load forecast metadata if it exists
forecast_metadata = {}
if os.path.exists(forecast_json_path):
    with open(forecast_json_path, 'r') as f:
        forecast_metadata = json.load(f)
    print(f"Loaded forecast metadata from: {forecast_json_path}\n")
else:
    print(f"Warning: Forecast JSON not found at {forecast_json_path}\n")

# List of image filenames to analyze with specific parameter info
image_configs = {
    "mo-forecastfiredanger": {
        "param": "Fire Danger Index",
        "instruction": """Match colors EXACTLY to the legend:
        - Light green = Low
        - Yellow = Moderate  
        - Orange = Elevated
        - Red = Critical
        - Dark red/maroon = Extreme
        
        Look at what color dominates each region. If most of the state is yellow, say it's Moderate. Don't overstate the danger."""
    },
    "mo-forecastfuelmoisture": {
        "param": "Fuel Moisture (%)",
        "instruction": "Read the EXACT numbers on the left legend (likely 0, 10, 20, 30, 40). Match map colors to these specific values. Brown/tan = low values (dry), green = high values (moist)."
    },
    "mo-forecastmaxtemp": {
        "param": "Maximum Temperature (°F)",
        "instruction": "Read the EXACT temperature numbers on the left legend. Blue = cold, red = warm. Report the actual numeric ranges shown."
    },
    "mo-forecastmaxwind": {
        "param": "Maximum Wind Speed (knots)",
        "instruction": "Read the EXACT wind speed numbers on the left legend. Cool colors = light winds, warm colors = stronger winds. Report actual numeric ranges."
    },
    "mo-forecastminrh": {
        "param": "Minimum Relative Humidity (%)",
        "instruction": "Read the EXACT percentage numbers on the left legend (likely ranges from 10-100%). Cool colors = high RH (moist), warm colors = low RH (dry). RH below 20% is VERY rare."
    }
}

# Collect all analyses
analyses = {}

for image_filename, config in image_configs.items():
    image_path = os.path.join(images_dir, f"{image_filename}.png")
    img = PIL.Image.open(image_path)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            f"You are analyzing a {config['param']} map for Missouri.",
            "",
            "STEP 1: Look at the vertical color scale on the LEFT side of the image.",
            "STEP 2: Read the NUMBERS written next to the color bar. These are the actual values.",
            "STEP 3: Look at the map and identify which colors appear in each region.",
            "STEP 4: Match those colors to the numbers you read in the legend.",
            "",
            config['instruction'],
            "",
            "Create ONE concise headline describing what you actually see on the map.",
            "Be accurate - don't exaggerate. If most of the map is one color, that's the dominant condition.",
            "Format: '[Region] sees [actual values from legend], while [Region] experiences [actual values]'",
            "",
            img
        ]
    )
    
    analyses[image_filename] = response.text.strip()
    print(f"{image_filename}: {response.text}\n")

# Generate headline/title
current_date = datetime.now()
date_string = current_date.strftime('%B %d, %Y')

headline_prompt = f"""Based on these fire weather conditions for Missouri:

Fire Danger: {analyses['mo-forecastfiredanger']}
Fuel Moisture: {analyses['mo-forecastfuelmoisture']}
Max Temperature: {analyses['mo-forecastmaxtemp']}
Max Wind: {analyses['mo-forecastmaxwind']}
Min Relative Humidity: {analyses['mo-forecastminrh']}

Create a short, catchy headline (5-8 words max) that summarizes the overall fire danger for Missouri.

Examples:
- "Low to Moderate Fire Danger Across Missouri"
- "Moderate Fire Danger for Central and Southern MO"
- "Elevated Fire Risk in Southern Missouri"
- "Low Fire Danger Statewide"

Be accurate to the actual conditions - don't overstate. Respond with ONLY the headline, no extra text."""

headline_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=headline_prompt
)

headline = headline_response.text.strip()

# Now create a comprehensive summary paragraph
summary_prompt = f"""Based on these fire weather conditions for Missouri:

Fire Danger: {analyses['mo-forecastfiredanger']}
Fuel Moisture: {analyses['mo-forecastfuelmoisture']}
Max Temperature: {analyses['mo-forecastmaxtemp']}
Max Wind: {analyses['mo-forecastmaxwind']}
Min Relative Humidity: {analyses['mo-forecastminrh']}

Write a concise 3-4 sentence paragraph summarizing the overall fire danger outlook for Missouri fire departments. 

IMPORTANT ACCURACY RULES:
- If the fire danger analysis says "Low" or "Moderate", DO NOT say conditions are "Critical" or "Extreme"
- Match the severity level to what was actually analyzed in the maps
- Be specific about regions but don't overstate the danger
- Only mention elevated/critical/extreme conditions if they were explicitly stated in the analyses above

Keep it professional and actionable for emergency responders.

Don't need input to first responders/fire departments. They know what they are doing. Focus on the facts to give a summary of the fire risk for the day.

Use common units for the united states even if the data plots might be using others. etc use mph converted from the knots data"""

summary_response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=summary_prompt
)

# Output
print("\n" + "="*60)
print(f"{headline} - {date_string}")
print("="*60)
print(summary_response.text)

# Store forecast in database
valid_time = current_date.replace(hour=12, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
try:
    forecast_id = insert_forecast(
        valid_time=valid_time,
        title=headline,
        discussion=summary_response.text.strip()
    )
    print(f"\nForecast saved to database with ID: {forecast_id}")
except Exception as e:
    print(f"\nError saving forecast to database: {e}")

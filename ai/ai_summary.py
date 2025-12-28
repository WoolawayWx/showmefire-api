from google import genai
import PIL.Image
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

DANGER_LEVELS = {
    "Low": hex_to_rgb("#90EE90"),       
    "Moderate": hex_to_rgb("#FFED4E"),  
    "Elevated": hex_to_rgb("#FFA500"),  
    "Critical": hex_to_rgb("#FF0000"),  
    "Extreme": hex_to_rgb("#8B0000"),  
}

def get_present_danger_levels(image_path, legend_colors=DANGER_LEVELS, tolerance=30):
    img = PIL.Image.open(image_path).convert('RGB')
    arr = np.array(img)
    present = []
    for level, color in legend_colors.items():
        mask = np.all(np.abs(arr - color) <= tolerance, axis=-1)
        if np.any(mask):
            present.append(level)
    return present

def generate_summary():
    genai_key = os.getenv('genai_key')
    client = genai.Client(api_key=genai_key)

    images_folder = "images"
    image_files = [
        "mo-realtimefiredanger.png",
        "mo-rh.png",
        "mo-fuelmoisture.png",
        "mo-windfilmap.png"
    ]

    images = []
    for img_file in image_files:
        img_path = os.path.join(images_folder, img_file)
        if os.path.exists(img_path):
            images.append(PIL.Image.open(img_path))
        else:
            print(f"Warning: {img_path} not found")

    present_levels = get_present_danger_levels(os.path.join(images_folder, "mo-realtimefiredanger.png"))
    levels_str = ", ".join(present_levels)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            f"You are analyzing Missouri fire weather graphics. The only fire danger levels present on the map are: {levels_str}.",
            f"Explicitly mention each of these levels and the regions where they appear. Do NOT mention any other danger levels.",
            "Write a very short summary (1-2 sentences) of the current fire weather conditions for Missouri.",
            "Describe which geographic regions (e.g., north, south, central, southwest, etc.) are experiencing each present fire danger level, and include any notable weather patterns (such as humidity, fuel moisture, or wind) that are visible in the graphics.",
            "Do not include advice or recommendations for fire departments.",
            "Respond with one concise paragraph."
        ] + images
    )

    return response.text

if __name__ == "__main__":
    print(generate_summary())
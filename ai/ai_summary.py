from google import genai
import PIL.Image
from dotenv import load_dotenv
import os

load_dotenv()

def generate_summary():
    genai_key = os.getenv('genai_key')
    client = genai.Client(api_key=genai_key)

    # Define the images folder
    images_folder = "images"

    # Load all relevant images
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

    # Generate content with all images
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            "You are analyzing Missouri fire weather graphics. Your summary must ONLY mention colors and danger levels that are visually present in the provided images.",
            "IGNORE any colors or danger levels listed in the legend if they do not actually appear on the map.",
            "For example, if only green and yellow are visible, ONLY report green (Low) and yellow (Moderate) fire danger. Do NOT mention orange, red, or maroon unless you see those colors on the map.",
            "Describe the spatial distribution of each color present (e.g., 'yellow in southwest Missouri').",
            "For other graphics (relative humidity, fuel moisture, winds), ONLY describe the color gradients and values that are present in the images.",
            "Do NOT invent or extrapolate colors or danger levels not shown.",
            "Keep the summary concise and suitable for an RSS feed for fire departments.",
            "Respond with one cohesive summary paragraph."
        ] + images
    )

    return response.text

if __name__ == "__main__":
    print(generate_summary())
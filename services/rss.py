from feedgen.feed import FeedGenerator
from datetime import datetime, timezone
import pytz  # Add this import for timezone handling
from services.synoptic import get_station_data  # Import your data function
import argparse  # Add for command-line flags
from ai.ai_summary import generate_summary  # Import AI summary function

def generate_rss_feed(add_summary=False):
    fg = FeedGenerator()
    fg.title('Show Me Fire | Missouri Weather & Danger Maps')
    fg.description('Real-time fire weather analysis for Missouri')
    fg.link(href='https://api.showmefire.org/rss.xml', rel='self')
    
    now_utc = datetime.now(timezone.utc)
    central_tz = pytz.timezone('US/Central')
    now_central = now_utc.astimezone(central_tz)
    valid_time = now_central.strftime('%H:%M CT')
    fg.lastBuildDate(now_utc)

    # Define your 4 images
    maps = [
        {"title": "Fire Danger Assessment", "url": "https://api.showmefire.org/images/mo-realtimefiredanger.png", "id": "danger"},
        {"title": "Relative Humidity", "url": "https://api.showmefire.org/images/mo-rh.png", "id": "rh"},
        {"title": "Fuel Moisture", "url": "https://api.showmefire.org/images/mo-fuelmoisture.png", "id": "fuel"},
        {"title": "Sustained Winds", "url": "https://api.showmefire.org/images/mo-windfilmap.png", "id": "wind"}
    ]

    for m in maps:
        fe = fg.add_entry()
        fe.title(f"{m['title']} - {valid_time}")
        
        # Add a cache-buster (?t=timestamp) to ensure the CDN serves the fresh version
        image_with_cache_buster = f"{m['url']}?t={int(now_utc.timestamp())}"
        
        fe.description(f'<img src="{image_with_cache_buster}" alt="{m["title"]}">')
        fe.link(href=m['url'])
        
        # Use a static ID + type so it updates the same "post" in the reader
        fe.guid(f"mo-map-{m['id']}", permalink=False)
        fe.pubDate(now_utc)
        fe.enclosure(image_with_cache_buster, 0, 'image/png')

    # Add AI summary as an additional item if requested
    if add_summary:
        summary_text = generate_summary()
        fe = fg.add_entry()
        fe.title(f"Current Fire Weather Summary - {valid_time}")
        fe.description(summary_text)
        fe.link(href='https://api.showmefire.org/rss.xml')  # Link to the feed itself
        fe.guid("mo-summary", permalink=False)
        fe.pubDate(now_utc)

    return fg.rss_str(pretty=True).decode('utf-8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RSS feed with optional AI summary.")
    parser.add_argument('--add-summary', action='store_true', help="Include AI-generated fire weather summary in the feed.")
    args = parser.parse_args()

    # 1. Generate the XML string
    xml_output = generate_rss_feed(add_summary=args.add_summary)
    
    # 2. Define where you want to save the file
    # Change 'rss.xml' to the full path where your web server looks for files
    output_file = "public/rss.xml" 
    
    # 3. Write to the file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_output)
        
    print(f"Successfully created {output_file} at {datetime.now().strftime('%H:%M:%S')}")
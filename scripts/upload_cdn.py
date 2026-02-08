import os
import boto3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from botocore.config import Config

# --- CONFIGURATION ---
load_dotenv()

R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
R2_ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID')
BUCKET_NAME = 'cdn-showmefire'

# Directories to scan
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FOLDERS_TO_UPLOAD = [
    PROJECT_ROOT / "images"
]

def upload_to_cdn(files, dest_keys, content_types=None, cache_controls=None):
    """
    Uploads files to the CDN with custom destination keys.
    Args:
        files (list of Path or str): List of file paths to upload.
        dest_keys (list of str): List of destination keys (paths in the bucket) for each file.
        content_types (list of str, optional): List of content types for each file. Defaults to 'application/octet-stream'.
        cache_controls (list of str, optional): List of cache control headers for each file. Defaults to 'max-age=3600'.
    """
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID]):
        print("Error: Missing R2 credentials in .env file.")
        return

    if len(files) != len(dest_keys):
        raise ValueError("files and dest_keys must have the same length.")

    if content_types is None:
        content_types = ['application/octet-stream'] * len(files)
    if cache_controls is None:
        cache_controls = ['max-age=3600'] * len(files)

    if not (len(content_types) == len(files) == len(cache_controls)):
        raise ValueError("files, dest_keys, content_types, and cache_controls must have the same length.")

    s3 = get_r2_client()
    for file_path, key, ctype, cache in zip(files, dest_keys, content_types, cache_controls):
        file_path = Path(file_path)
        if not file_path.exists():
            print(f" [SKIP] {file_path} does not exist.")
            continue
        try:
            s3.upload_file(
                str(file_path), BUCKET_NAME, key,
                ExtraArgs={'ContentType': ctype, 'CacheControl': cache}
            )
            print(f" [OK] {file_path.name} -> {key}")
        except Exception as e:
            print(f" [FAILED] {file_path.name} to {key}: {e}")

def get_quarter_hour_path():
    """Returns a path like '2025/12/25/2330' rounded to the nearest 15 mins."""
    now = datetime.now()
    minute = (now.minute // 15) * 15
    rounded_time = now.replace(minute=minute, second=0, microsecond=0)
    return rounded_time.strftime('%Y/%m/%d/%H%M')

def get_forecast_path():
    """Returns a path like '2025/12/25/forecast'."""
    now = datetime.now()
    return now.strftime('%Y/%m/%d/forecast')

def generate_image_timeline(hours=12, cdn_base_url='https://cdn.showmefire.org', path_prefix=None):
    """
    Generate a JSON timeline of image URLs for the last N hours, grouped by image type.
    
    Args:
        hours: Number of hours to go back (default: 12)
        cdn_base_url: Base URL for the CDN
        path_prefix: Optional prefix (e.g., 'test' for testing)
    
    Returns:
        dict: Timeline data with timestamps and URLs grouped by image type
    """
    from datetime import timedelta
    import json
    
    now = datetime.now()
    intervals = hours * 4  # 4 intervals per hour (every 15 minutes)
    
    # Image types to track
    image_files = {
        'realtimefiredanger': 'mo-realtimefiredanger.png',
        'fuelmoisture': 'mo-fuelmoisture.png',
        'rh': 'mo-rh.png',
        'windfilmap': 'mo-windfilmap.png'
    }
    
    timeline = {
        'generated_at': now.strftime('%Y-%m-%dT%H:%M:%S'),
        'hours': hours,
        'count': intervals,
        'cdn_base_url': cdn_base_url,
        'timestamps': []
    }
    
    # Initialize image arrays
    for image_key in image_files.keys():
        timeline[image_key] = []
    
    # Generate entries from oldest to newest
    for i in range(intervals - 1, -1, -1):
        # Go back in 15-minute increments
        timestamp = now - timedelta(minutes=i * 15)
        
        # Round to nearest 15 minutes
        minute = (timestamp.minute // 15) * 15
        rounded_time = timestamp.replace(minute=minute, second=0, microsecond=0)
        
        # Generate path
        date_path = rounded_time.strftime('%Y/%m/%d/%H%M')
        if path_prefix:
            date_path = f"{path_prefix.rstrip('/')}/{date_path}"
        
        # Add timestamp info
        timeline['timestamps'].append({
            'timestamp': rounded_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'timestamp_readable': rounded_time.strftime('%Y-%m-%d %H:%M'),
            'path': date_path
        })
        
        # Add URLs for each image type
        for image_key, image_file in image_files.items():
            timeline[image_key].append({
                'url': f"{cdn_base_url}/{date_path}/{image_file}",
                'timestamp': rounded_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'path': f"{date_path}/{image_file}"
            })
    
    return timeline

def save_image_timeline(output_path=None, hours=12, cdn_base_url='https://cdn.showmefire.org', path_prefix=None):
    """
    Generate and save image timeline JSON file.
    
    Args:
        output_path: Path to save JSON file (default: PROJECT_ROOT/public/image-timeline.json)
        hours: Number of hours to go back
        cdn_base_url: Base URL for the CDN
        path_prefix: Optional prefix for testing
    
    Returns:
        Path: Path to saved JSON file
    """
    import json
    
    if output_path is None:
        output_path = PROJECT_ROOT / 'public' / 'image-timeline.json'
    else:
        output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate timeline
    timeline = generate_image_timeline(hours=hours, cdn_base_url=cdn_base_url, path_prefix=path_prefix)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(timeline, f, indent=2)
    
    print(f"Image timeline saved to {output_path}")
    print(f"  - Generated {timeline['count']} timeline entries per image type")
    print(f"  - Covers last {hours} hours")
    
    return output_path

def get_r2_client():
    """Initializes the S3-compatible client for Cloudflare R2."""
    return boto3.client(
        service_name='s3',
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4'),
        region_name='auto'
    )

def upload_quarter_hour_files(files_to_upload=None, path_prefix=None):
    """
    Uploads files that update on the quarter hour with path structure yyyy/mm/dd/hhmm.
    Args:
        files_to_upload (list of Path or str, optional): List of file paths to upload.
        path_prefix (str, optional): Prefix to prepend to all paths (e.g., 'test/' for testing).
    """
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID]):
        print("Error: Missing R2 credentials in .env file.")
        return

    s3 = get_r2_client()
    timestamp_path = get_quarter_hour_path()
    
    # Apply prefix if provided
    if path_prefix:
        timestamp_path = f"{path_prefix.rstrip('/')}/{timestamp_path}"
    
    print(f"--- Uploading Quarter-Hour Files to: {timestamp_path} ---")

    if files_to_upload:
        files = [Path(f) for f in files_to_upload]
    else:
        files = []
        for folder in FOLDERS_TO_UPLOAD:
            if not folder.exists():
                print(f"Skipping {folder.name}: Folder does not exist.")
                continue
            files.extend([f for f in folder.glob('*') if f.is_file()])

    for file_path in files:
        filename = file_path.name
        content_type = 'application/json' if file_path.suffix == '.geojson' else 'image/png'
        
        # Archive path: yyyy/mm/dd/hhmm/filename.png
        archive_key = f"{timestamp_path}/{filename}"
        
        # Latest path: latest/filename.png (or test/latest/filename.png)
        latest_key = f"{path_prefix.rstrip('/')}/latest/{filename}" if path_prefix else f"latest/{filename}"

        try:
            # Upload to Archive
            s3.upload_file(
                str(file_path), BUCKET_NAME, archive_key,
                ExtraArgs={'ContentType': content_type, 'CacheControl': 'max-age=3600'}
            )
            
            # Upload to Latest
            s3.upload_file(
                str(file_path), BUCKET_NAME, latest_key,
                ExtraArgs={'ContentType': content_type, 'CacheControl': 'max-age=300'}
            )
            
            print(f" [OK] {filename} -> R2")
        except Exception as e:
            print(f" [FAILED] {filename}: {e}")
    
    # Generate and upload timeline JSON
    try:
        cdn_base_url = os.getenv('CDN_BASE_URL', 'https://cdn.showmefire.org')
        timeline_hours = int(os.getenv('TIMELINE_HOURS', '12'))
        
        timeline_path = save_image_timeline(
            hours=timeline_hours, 
            cdn_base_url=cdn_base_url, 
            path_prefix=path_prefix
        )
        
        # Upload timeline JSON to CDN
        timeline_key = f"{path_prefix.rstrip('/')}/image-timeline.json" if path_prefix else "image-timeline.json"
        s3.upload_file(
            str(timeline_path), BUCKET_NAME, timeline_key,
            ExtraArgs={'ContentType': 'application/json', 'CacheControl': 'max-age=300'}
        )
        print(f" [OK] image-timeline.json -> {timeline_key}")
    except Exception as e:
        print(f" [INFO] Timeline generation skipped or failed: {e}")

def upload_forecast_files(files_to_upload=None, path_prefix=None):
    """
    Uploads forecast files with path structure yyyy/mm/dd/forecast.
    Args:
        files_to_upload (list of Path or str, optional): List of file paths to upload.
        path_prefix (str, optional): Prefix to prepend to all paths (e.g., 'test/' for testing).
    """
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID]):
        print("Error: Missing R2 credentials in .env file.")
        return

    s3 = get_r2_client()
    forecast_path = get_forecast_path()
    
    # Apply prefix if provided
    if path_prefix:
        forecast_path = f"{path_prefix.rstrip('/')}/{forecast_path}"
    
    print(f"--- Uploading Forecast Files to: {forecast_path} ---")

    if files_to_upload:
        files = [Path(f) for f in files_to_upload]
    else:
        files = []
        for folder in FOLDERS_TO_UPLOAD:
            if not folder.exists():
                print(f"Skipping {folder.name}: Folder does not exist.")
                continue
            files.extend([f for f in folder.glob('*') if f.is_file()])

    for file_path in files:
        filename = file_path.name
        content_type = 'application/json' if file_path.suffix == '.geojson' else 'image/png'
        
        # Archive path: yyyy/mm/dd/forecast/filename.png
        archive_key = f"{forecast_path}/{filename}"
        
        # Latest path: latest/filename.png (or test/latest/filename.png)
        latest_key = f"{path_prefix.rstrip('/')}/latest/{filename}" if path_prefix else f"latest/{filename}"

        try:
            # Upload to Archive
            s3.upload_file(
                str(file_path), BUCKET_NAME, archive_key,
                ExtraArgs={'ContentType': content_type, 'CacheControl': 'max-age=3600'}
            )
            
            # Upload to Latest
            s3.upload_file(
                str(file_path), BUCKET_NAME, latest_key,
                ExtraArgs={'ContentType': content_type, 'CacheControl': 'max-age=300'}
            )
            
            print(f" [OK] {filename} -> R2")
        except Exception as e:
            print(f" [FAILED] {filename}: {e}")

# Backwards compatibility - keep original function name
def run_upload(files_to_upload=None):
    """
    Legacy function. Use upload_quarter_hour_files() or upload_forecast_files() instead.
    """
    upload_quarter_hour_files(files_to_upload)

if __name__ == "__main__":
    # Get optional test prefix from environment
    test_prefix = os.getenv('CDN_TEST_PREFIX', None)
    
    # Example: Quarter-hour files
    quarter_hour_files = [
        PROJECT_ROOT / "images" / "mo-windfilmap.png",
        PROJECT_ROOT / "images" / "mo-rh.png",
        PROJECT_ROOT / "images" / "mo-realtimefiredanger.png",
        PROJECT_ROOT / "images" / "mo-fuelmoisture.png",
    ]
    upload_quarter_hour_files(files_to_upload=quarter_hour_files, path_prefix=test_prefix)
    
    # Example: Forecast files
    # forecast_files = [
    #     PROJECT_ROOT / "images" / "forecast-map.png",
    # ]
    # upload_forecast_files(files_to_upload=forecast_files, path_prefix=test_prefix)
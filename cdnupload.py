
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
FOLDERS_TO_UPLOAD = [
    SCRIPT_DIR / "images"
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

def get_rounded_timestamp():
    """Returns a string like '20251225_2330' rounded to the nearest 15 mins."""
    now = datetime.now()
    minute = (now.minute // 15) * 15
    rounded_time = now.replace(minute=minute, second=0, microsecond=0)
    return rounded_time.strftime('%Y%m%d_%H%M')

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

def run_upload(files_to_upload=None):
    """
    Uploads only the specified files if files_to_upload is provided,
    otherwise uploads all files in FOLDERS_TO_UPLOAD.
    Args:
        files_to_upload (list of Path or str, optional): List of file paths to upload.
    """
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID]):
        print("Error: Missing R2 credentials in .env file.")
        return

    s3 = get_r2_client()
    timestamp_folder = get_rounded_timestamp()
    
    print(f"--- Starting Sync for interval: {timestamp_folder} ---")

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
        # Set content type based on extension
        content_type = 'application/json' if file_path.suffix == '.geojson' else 'image/png'
        
        # 1. Path for the archive (e.g., 20251225_2330/mo-fuelmoisture.png)
        archive_key = f"{timestamp_folder}/{filename}"
        
        # 2. Path for the website (e.g., latest/mo-fuelmoisture.png)
        latest_key = f"latest/{filename}"

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

if __name__ == "__main__":
    # Example: specify files to upload by their paths
    files_to_upload = [
        SCRIPT_DIR / "images" / "mo-windfilmap.png",
        SCRIPT_DIR / "images" / "mo-rh.png",
        SCRIPT_DIR / "images" / "mo-realtimefiredanger.png",
        SCRIPT_DIR / "images" / "mo-fuelmoisture.png",
    ]
    run_upload(files_to_upload=files_to_upload)
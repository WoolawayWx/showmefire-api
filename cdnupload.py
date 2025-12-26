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

def run_upload():
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID]):
        print("Error: Missing R2 credentials in .env file.")
        return

    s3 = get_r2_client()
    timestamp_folder = get_rounded_timestamp()
    
    print(f"--- Starting Sync for interval: {timestamp_folder} ---")

    for folder in FOLDERS_TO_UPLOAD:
        if not folder.exists():
            print(f"Skipping {folder.name}: Folder does not exist.")
            continue

        for file_path in folder.glob('*'):
            if file_path.is_dir(): continue # Skip subfolders
            
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
    run_upload()
"""
archive_day.py

Downloads all R2 files for a given day and saves them as a zip archive.

Usage:
    python archive_day.py              # Archives yesterday
    python archive_day.py 2026-01-15   # Archives a specific date (YYYY-MM-DD)
"""

import os
import sys
import boto3
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from botocore.config import Config

# --- CONFIGURATION ---
load_dotenv()

R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID')
R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY')
R2_ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID')
BUCKET_NAME = 'cdn-showmefire'

# Where to save the zip files — change this to your desired folder
OUTPUT_DIR = Path('/app/data_archive')


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


def list_objects_for_day(s3, date: datetime) -> list:
    """
    Lists all objects in R2 for a given day using the yyyy/mm/dd/ prefix.
    Handles pagination so it won't miss anything if there are many files.
    """
    prefix = date.strftime('%Y/%m/%d/')
    print(f"Fetching file list for prefix: {prefix}")

    objects = []
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

    for page in pages:
        if 'Contents' in page:
            objects.extend(page['Contents'])

    print(f"Found {len(objects)} files for {date.strftime('%Y-%m-%d')}")
    return objects


def download_and_zip(s3, objects: list, date: datetime, output_dir: Path) -> Path:
    """
    Downloads all objects and writes them into a zip file.
    Preserves the original yyyy/mm/dd/hhmm/filename.png folder structure inside the zip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = date.strftime('%Y-%m-%d')
    zip_filename = output_dir / f"showmefire-{date_str}.zip"

    if zip_filename.exists():
        print(f"Warning: {zip_filename} already exists — overwriting.")

    print(f"Creating zip: {zip_filename}")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, obj in enumerate(objects):
            key = obj['Key']
            size_kb = obj['Size'] / 1024

            try:
                print(f"  [{i + 1}/{len(objects)}] Downloading {key} ({size_kb:.1f} KB)...")
                response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                file_data = response['Body'].read()

                # Write into zip preserving the full path structure
                zf.writestr(key, file_data)

            except Exception as e:
                print(f"  [FAILED] {key}: {e}")

    zip_size_mb = zip_filename.stat().st_size / (1024 * 1024)
    print(f"\nDone! Zip saved to: {zip_filename} ({zip_size_mb:.2f} MB)")
    return zip_filename


def archive_day(date: datetime):
    """Main function — lists, downloads, and zips all files for a given day."""
    if not all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID]):
        print("Error: Missing R2 credentials in .env file.")
        sys.exit(1)

    print(f"\n=== Archiving {date.strftime('%Y-%m-%d')} ===\n")

    s3 = get_r2_client()
    objects = list_objects_for_day(s3, date)

    if not objects:
        print("No files found for this date. Nothing to archive.")
        return

    zip_path = download_and_zip(s3, objects, date, OUTPUT_DIR)
    print(f"\nArchive complete: {zip_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # A date was passed as an argument
        try:
            target_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
        except ValueError:
            print("Error: Date must be in YYYY-MM-DD format (e.g. 2026-01-15)")
            sys.exit(1)
    else:
        # Default to yesterday
        target_date = datetime.now() - timedelta(days=1)

    archive_day(target_date)
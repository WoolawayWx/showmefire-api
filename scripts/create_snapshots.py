#!/usr/bin/env python3
"""
Create snapshots for new HRRR files in the cache directory.
This allows adding new data without deleting existing processed snapshots.
"""

import sqlite3
import re
from pathlib import Path
from datetime import datetime

# Database path
DB_PATH = Path(__file__).resolve().parent.parent / 'data' / 'showmefire.db'

def get_date_from_filename(filename):
    """Extract YYYY-MM-DD from HRRR filename like 'hrrr_20260127_12z_f04-15.nc'"""
    match = re.search(r'hrrr_(\d{8})_', filename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return None

def create_snapshots_from_hrrr():
    """Create snapshots for HRRR files that don't exist in database yet."""

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure snapshots table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL UNIQUE,
            hrrr_filename TEXT,
            is_processed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Get existing snapshot dates
    cursor.execute("SELECT snapshot_date FROM snapshots")
    existing_dates = {row[0] for row in cursor.fetchall()}

    # Scan HRRR cache directory
    hrrr_dir = Path("cache/hrrr")
    if not hrrr_dir.exists():
        print("❌ HRRR cache directory not found")
        return

    new_snapshots = []
    for nc_file in hrrr_dir.glob("*.nc"):
        date_str = get_date_from_filename(nc_file.name)
        if date_str and date_str not in existing_dates:
            new_snapshots.append((date_str, nc_file.name))

    if not new_snapshots:
        print("✅ No new HRRR files found to create snapshots for")
        conn.close()
        return

    # Insert new snapshots
    cursor.executemany(
        "INSERT INTO snapshots (snapshot_date, hrrr_filename) VALUES (?, ?)",
        new_snapshots
    )

    conn.commit()
    conn.close()

    print(f"✅ Created {len(new_snapshots)} new snapshots:")
    for date_str, filename in new_snapshots:
        print(f"   - {date_str}: {filename}")

if __name__ == "__main__":
    create_snapshots_from_hrrr()
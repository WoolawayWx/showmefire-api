#!/usr/bin/env python3
"""
Reset all snapshots to unprocessed state and clear weather features.
This allows re-running the extract_hrrr.py pipeline with updated extraction logic.
"""
import sqlite3
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db_path

def reset_snapshots():
    db_path = get_db_path()
    print(f"Using database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get counts before reset
    cursor.execute("SELECT COUNT(*) FROM snapshots WHERE is_processed = 1")
    processed_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM weather_features")
    features_count = cursor.fetchone()[0]
    
    print(f"\nCurrent state:")
    print(f"  - Processed snapshots: {processed_count}")
    print(f"  - Weather features records: {features_count}")
    
    # Ask for confirmation
    response = input(f"\nReset all snapshots and delete {features_count} weather features? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        conn.close()
        return
    
    # Delete all weather features (extracted HRRR data)
    cursor.execute("DELETE FROM weather_features")
    deleted_features = cursor.rowcount
    
    # Reset all snapshots to unprocessed
    cursor.execute("UPDATE snapshots SET is_processed = 0")
    reset_count = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Reset complete:")
    print(f"  - Deleted {deleted_features} weather features")
    print(f"  - Reset {reset_count} snapshots to unprocessed")
    print(f"\nYou can now run: python3 pipelines/extract_hrrr.py")

if __name__ == "__main__":
    reset_snapshots()

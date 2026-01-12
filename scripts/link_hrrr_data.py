import sqlite3
from pathlib import Path

# Internal Docker paths
HRRR_STORAGE = Path("/app/cache/hrrr") 
DB_PATH = Path("/app/data/showmefire.db")

def link_hrrr():
    if not HRRR_STORAGE.exists():
        print(f"Error: Folder {HRRR_STORAGE} not found inside container.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find snapshots that need an HRRR file
    cursor.execute("SELECT snapshot_date FROM snapshots WHERE hrrr_filename IS NULL")
    missing_dates = cursor.fetchall()

    if not missing_dates:
        print("All snapshots already have HRRR data linked.")
        return

    print(f"Scanning {HRRR_STORAGE} for {len(missing_dates)} dates...")

    for (date_str,) in missing_dates:
        # Converts YYYY-MM-DD to YYYYMMDD
        date_clean = date_str.replace("-", "") 
        
        # Look for the file matching hrrr_YYYYMMDD
        # The '12z_f04-15' part is ignored so it matches any variant of that date
        match = list(HRRR_STORAGE.glob(f"hrrr_{date_clean}_*"))

        if match:
            # We save the path RELATIVE to /app so it works on any machine
            relative_path = str(match[0].relative_to("/app"))
            cursor.execute("""
                UPDATE snapshots 
                SET hrrr_filename = ? 
                WHERE snapshot_date = ?
            """, (relative_path, date_str))
            print(f"✅ Linked {date_str} -> {match[0].name}")
        else:
            print(f"❌ No HRRR file found for {date_str}")

    conn.commit()
    conn.close()
    print("HRRR linking complete.")

if __name__ == "__main__":
    link_hrrr()
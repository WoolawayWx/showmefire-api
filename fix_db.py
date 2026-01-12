import sqlite3
from core.database import get_db_path

db_path = get_db_path()
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

columns_to_add = [
    ('snapshots', 'is_processed', 'INTEGER DEFAULT 0'),
    ('snapshots', 'hrrr_filename', 'TEXT')
]

for table, column, definition in columns_to_add:
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        print(f"✅ Added {column} to {table}")
    except sqlite3.OperationalError:
        print(f"ℹ️ {column} already exists in {table}")

conn.commit()
conn.close()
print("Migration complete.")


import sqlite3
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())
from api.core.database import get_db_path

db_path = get_db_path()
print(f"DB Path: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT COUNT(*) as count FROM observations", conn)
    print("Observations count:", df)
    df2 = pd.read_sql("SELECT COUNT(*) as count FROM station_grid_indices", conn)
    print("Indices count:", df2)
    conn.close()
except Exception as e:
    print(e)

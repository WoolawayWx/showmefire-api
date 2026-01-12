import os
from pathlib import Path

# Directory Paths
IMAGES_DIR = Path("images")
GIS_DIR = Path("gis")
PUBLIC_DIR = Path("public")
REPORTS_DIR = Path("reports")
LOGS_DIR = Path("logs")
ARCHIVE_DIR = Path("archive")
ARCHIVE_RAW_DATA_DIR = ARCHIVE_DIR / "raw_data"
DATA_DIR = Path("data")

# File Paths
BANNER_CONFIG_FILE = DATA_DIR / "banner_config.json"
MISSOURI_FIRES_JSON = DATA_DIR / "missouri_fires_coords.json"
MISSOURI_FIRES_GEOJSON = DATA_DIR / "missouri_fires.geojson"

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
MISSOURI_FIRES_JSON = DATA_DIR / "missouri_fires_coords.json"
MISSOURI_FIRES_GEOJSON = DATA_DIR / "missouri_fires.geojson"


def _parse_office_codes(raw: str):
	offices = [code.strip().upper() for code in raw.split(",") if code.strip()]
	return offices or ["EAX", "SGF", "LSX"]


AFD_OFFICES = _parse_office_codes(os.getenv("AFD_OFFICES", "EAX,SGF,LSX"))

try:
	AFD_POLL_MINUTES = max(1, int(os.getenv("AFD_POLL_MINUTES", "60")))
except ValueError:
	AFD_POLL_MINUTES = 60

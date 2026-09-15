import os
from pathlib import Path

# Directory Paths
IMAGES_DIR = Path("images")
# When configured in production this is the same host directory mounted
# read-only at /data by MapServer. The API remains able to serve the static
# catalog/GeoJSON contracts from /gis.
GIS_DIR = Path(os.getenv("SMF_GIS_PUBLISH_DIR", "gis"))
FORECAST_V1_DIR = Path(os.getenv("SMF_FORECAST_V1_ROOT", "data/forecast-v1"))
PUBLIC_DIR = Path("public")
REPORTS_DIR = Path(os.getenv("SMF_REPORTS_DIR", str(Path(__file__).resolve().parents[1] / "reports")))
LOGS_DIR = Path("logs")
ARCHIVE_DIR = Path("archive")
ARCHIVE_RAW_DATA_DIR = ARCHIVE_DIR / "raw_data"
DATA_DIR = Path("data")
BCFY_DATA_DIR = Path(os.getenv("BCFY_DATA_DIR", str(DATA_DIR / "bcfy")))
BCFY_AUDIO_DIR = BCFY_DATA_DIR / "audio"
BCFY_TRANSCRIPT_DIR = BCFY_DATA_DIR / "transcripts"

# BCFY_KEY_ID is the API Key ID (JWT "kid"), BCFY_KEY_SECRET is the API Key
# itself (the HMAC-SHA256 signing secret), and BCFY_ISSUER is the Application
# ID (JWT "iss"). Broadcastify's Calls API authenticates with a self-signed
# JWT minted locally per request -- there is no token-exchange HTTP call.
BCFY_KEY_ID = os.getenv("BCFY_KEYID", "").strip()
BCFY_KEY_SECRET = os.getenv("BCFY_KEYSECRET", "").strip()
BCFY_ISSUER = os.getenv("BCFY_ISSUER", "").strip()
BCFY_API_BASE_URL = os.getenv("BCFY_API_BASE_URL", "https://api.bcfy.io").rstrip("/")
BCFY_CALLS_URL = os.getenv("BCFY_CALLS_URL", "").strip()

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

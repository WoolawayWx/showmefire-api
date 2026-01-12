import json
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from core.config import BANNER_CONFIG_FILE

logger = logging.getLogger(__name__)

class BannerData(BaseModel):
    enabled: bool = False
    type: str = "info"  # info, warning, danger, success
    message: str = ""
    link: Optional[str] = None

def load_banner_config() -> BannerData:
    if BANNER_CONFIG_FILE.exists():
        try:
            with open(BANNER_CONFIG_FILE, "r") as f:
                data = json.load(f)
            return BannerData(**data)
        except Exception as e:
            logger.error(f"Failed to load banner config: {e}")
    return BannerData()

def save_banner_config(banner: BannerData):
    try:
        with open(BANNER_CONFIG_FILE, "w") as f:
            json.dump(banner.dict(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save banner config: {e}")

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add api directory to path to import services
# Script is in api/scripts, so we want to add api/
project_root = Path(__file__).resolve().parent.parent # api/
sys.path.append(str(project_root))

from dotenv import load_dotenv
# Explicitly load .env from api directory
load_dotenv(project_root / ".env")

from services.synoptic import save_raw_data_to_archive

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting RAWS data archive process...")
    
    # Calculate absolute path to archive directory
    archive_dir = project_root / "archive" / "raw_data"
    
    try:
        # Fetch last 24 hours of data
        filepath = await save_raw_data_to_archive(days_back=1, archive_dir=str(archive_dir))
        
        if filepath:
            logger.info(f"Successfully archived RAWS data to {filepath}")
        else:
            logger.warning("No data archived.")
            
    except Exception as e:
        logger.error(f"Failed to archive RAWS data: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

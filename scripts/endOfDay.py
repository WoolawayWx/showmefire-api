import asyncio
import sys
import os
import logging

# Add project root to path (api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from services.synoptic import save_raw_data_to_archive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting end-of-day RAWS data pull for MO...")
    try:
        # Define archive directory absolute path
        archive_dir = os.path.join(BASE_DIR, "archive", "raw_data")

        # Define the time window: 10am to 9pm US/Central (16:00 to 03:00 UTC next day)
        from datetime import datetime, timedelta, timezone
        import pytz
        central = pytz.timezone('US/Central')
        today_central = datetime.now(central).replace(hour=0, minute=0, second=0, microsecond=0)
        start_central = today_central.replace(hour=10)
        end_central = today_central.replace(hour=21)
        # Convert to UTC
        start_utc = start_central.astimezone(timezone.utc)
        end_utc = end_central.astimezone(timezone.utc)

        # Fetch data for the last 1 day (today) for MO only, Network 2 (RAWS), but only keep obs in this window
        filepath = save_raw_data_to_archive(
            days_back=1,
            states=['MO'],
            networks=[2],
            archive_dir=archive_dir,
            obs_start_utc=start_utc,
            obs_end_utc=end_utc
        )

        if filepath:
            logger.info(f"Successfully archived RAWS data to: {filepath}")
        else:
            logger.warning("No data was archived.")

    except Exception as e:
        logger.error(f"Failed to run end-of-day script: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

import os
import sys

import hopsworks
import pandas as pd

from shared.file_logger import setup_logger
import shared.features as sf
import weather.pipeline as wp

log_file_path = os.path.join(os.path.dirname(__file__), 'weather_backfill.log')
logger = setup_logger('weather_backfill', log_file_path)


def backfill_days(start_date: str, end_date: str, fg=None, dry_run=False) -> int:
    print(f"Fetching weather data for {start_date} - {end_date}")
    df = wp.get_historical_weather(start_date, end_date)

    if dry_run:
        df.to_csv("weather_backfill.csv", index=False)
        return -1

    if fg is None:
        logger.warning("No Hopsworks connection. Skipping upload.")
        return 2
    try:
        fg.insert(df)
        sf.weather_update_feature_descriptions(fg)
    except Exception as e:
        logger.warning(f"Failed to connect to Hopsworks and skipping upload. {e}")
        return 2

    return 0


if __name__ == "__main__":
    START_DATE = os.environ.get("START_DATE", "2023-01-01")
    END_DATE = os.environ.get("END_DATE", "2024-12-15")
    DRY_RUN = os.environ.get("DRY_RUN", "True").lower() == "true"

    try:
        dates = pd.date_range(START_DATE, END_DATE)
    except (ValueError, TypeError):
        logger.error("Invalid date range. Exiting.")
        sys.exit(1)
    total_dates = len(dates)

    weather_fg = None
    if not DRY_RUN:
        if os.environ.get("HOPSWORKS_API_KEY") is None:
            os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()

        try:
            project = hopsworks.login()
            fs = project.get_feature_store()

            weather_fg = fs.get_or_create_feature_group(
                name='weather',
                description='Hourly weather data for Gävle',
                version=3,
                primary_key=['date'],
                event_time="date",
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Hopsworks. Continuing without uploads {e}")

    logger.info("Starting backfill process for dates: %s - %s (%s days)", START_DATE, END_DATE, total_dates)
    exit_code = backfill_days(START_DATE, END_DATE, dry_run=DRY_RUN)
    logger.info("Backfill process completed for dates: %s - %s with exit code: %s", START_DATE, END_DATE, exit_code)

    sys.exit(exit_code)

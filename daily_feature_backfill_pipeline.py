import os
import time

import hopsworks
import pandas as pd

from koda_backfill_feature_pipeline import backfill_date as backfill_date_koda
from shared.file_logger import setup_logger
import shared.features as sf
import weather.pipeline as wp

log_file_path = os.path.join(os.path.dirname(__file__), 'daily_backfill.log')
logger = setup_logger('daily_backfill', log_file_path)

# Enable copy-on-write for pandas to avoid SettingWithCopyWarning
pd.options.mode.copy_on_write = True

def backfill_recent_days(start_date: str, end_date: str, fg=None, dry_run=False) -> int:
    print(f"Fetching weather data for {start_date} - {end_date}")
    df = wp.get_recent_weather(start_date, end_date)

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
    DRY_RUN = os.environ.get("DRY_RUN", "True").lower() == "true"
    FG_VERSION = int(os.environ.get("FG_VERSION", 8))

    today = pd.Timestamp.now().normalize()
    yesterday = today - pd.Timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    start_time = time.time()

    delays_fg = None
    weather_fg = None
    if not DRY_RUN:
        if os.environ.get("HOPSWORKS_API_KEY") is None:
            os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()
        try:
            project = hopsworks.login()
            fs = project.get_feature_store()
            delays_fg = fs.get_or_create_feature_group(
                name='delays',
                description='Aggregated delay metrics per hour per day',
                version=FG_VERSION,
                primary_key=['arrival_time_bin'],
                event_time='arrival_time_bin'
            )
            weather_fg = fs.get_or_create_feature_group(
                name='weather',
                description='Hourly weather data for Gävle',
                version=3,
                primary_key=['date'],
                event_time="date",
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Hopsworks. Continuing without uploads {e}")

    logger.info("Starting backfill process for yesterday: %s", yesterday_str)

    delays_exit_code, job = backfill_date_koda(yesterday_str, fg=delays_fg, dry_run=DRY_RUN)
    if delays_exit_code == -1:
        logger.info("Delay dry run completed for date: %s", yesterday_str)
    if delays_exit_code == 0 and job is not None:
        logger.info("Running offline materialization jobs")
        try:
            job.run(await_termination=False)
        except Exception as e:
            logger.error(f"Failed to run offline materialization job. Skipping. {e}")


    weather_exit_code = backfill_recent_days(yesterday_str, yesterday_str, fg=weather_fg, dry_run=DRY_RUN)
    if weather_exit_code == -1:
        logger.info("Weather dry run completed for date: %s", yesterday_str)

    logger.info("Completed processing for date: %s with exit codes: %d, %d", yesterday_str, delays_exit_code, weather_exit_code)

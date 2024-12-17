import os
import sys
from datetime import datetime

import hopsworks

from koda.koda_constants import OperatorsWithRT
from shared.constants import GAEVLE_LONGITUDE, GAEVLE_LATITUDE
from shared.file_logger import setup_logger
import weather.fetch as wf
import weather.parse as wp
import gtfs_regional.pipeline as gp

ON_TIME_MIN_SECONDS = -180
ON_TIME_MAX_SECONDS = 300
OPERATOR = OperatorsWithRT.X_TRAFIK
RUN_HW_MATERIALIZATION_EVERY = 10

log_file_path = os.path.join(os.path.dirname(__file__), 'live_feature.log')
logger = setup_logger('live_feature', log_file_path)


def get_live_weather_data(today: str, fg = None, dry_run=False) -> int:
    logger.info(f"Fetching weather data for {today}")
    weather_response = wf.fetch_forecast_weather(GAEVLE_LONGITUDE, GAEVLE_LATITUDE)
    weather_df = wp.parse_weather_response(weather_response)
    weather_df['hour'] = weather_df['date'].dt.hour

    if dry_run:
        weather_df.to_csv("live_weather.csv", index=False)
        return 0

    if fg is None:
        logger.warning("No Hopsworks connection. Skipping upload.")
        return 1

    try:
        fg.insert(weather_df)
    except Exception as e:
        logger.error(f"Failed to insert weather data. {e}")
        return 1

    return 0


def get_live_delays_data(today: str, fg=None, dry_run=False) -> int:
    logger.info(f"Fetching delays data for {today}")
    df, map_df = gp.get_gtfr_data_for_day(today, OPERATOR)

    if df.empty:
        logger.warning(f"No data available for {today}. Pipeline exiting.")
        return 1

    # TODO: Similar/Same as koda_backfill_feature_pipeline.py

    return 0


if __name__ == "__main__":
    DRY_RUN = os.environ.get("DRY_RUN", "True").lower() == "true"

    today = datetime.now().strftime("%Y-%m-%d")

    delays_fg = None
    weather_fg = None
    if not DRY_RUN:
        if os.environ.get("HOPSWORKS_API_KEY") is None:
            os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()
        try:
            project = hopsworks.login()
            fs = project.get_feature_store()
            delays_fg = fs.get_feature_group(
                name='delays',
                version=6
            )
            weather_fg = fs.get_feature_group(
                name='weather',
                version=3
            )
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks. Exiting. {e}")
            sys.exit(1)

    logger.info(f"Starting live feature pipeline for {today}")
    weather_exit_code = get_live_weather_data(today, weather_fg, DRY_RUN)
    delays_exit_code = get_live_delays_data(today, delays_fg, DRY_RUN)

    logger.info(f"Completed live feature pipeline for {today} with weather exit code {weather_exit_code} and delays exit code {delays_exit_code}")

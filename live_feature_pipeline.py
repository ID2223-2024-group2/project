import os
import sys
from datetime import datetime

import hopsworks
import pandas as pd

from shared.constants import OperatorsWithRT
from shared.file_logger import setup_logger
import weather.pipeline as wp
import gtfs_regional.pipeline as gp
import shared.features as sf

OPERATOR = OperatorsWithRT.X_TRAFIK
LIVE_MIN_TRIP_UPDATES_PER_TIMESLOT = 5

log_file_path = os.path.join(os.path.dirname(__file__), 'live_feature.log')
logger = setup_logger('live_feature', log_file_path)

# Enable copy-on-write for pandas to avoid SettingWithCopyWarning
pd.options.mode.copy_on_write = True


def get_live_weather_data(today: str, fg = None, dry_run=False) -> int:
    logger.info(f"Fetching weather data for {today}")
    weather_df = wp.get_forecast_weather()

    if dry_run:
        weather_df.to_csv("live_feature_weather.csv", index=False)
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
    rt_df, route_types_map_df, stop_count_df, stop_location_map_df = gp.get_gtfr_data_for_day(today, OPERATOR, force_rt=True)

    if rt_df.empty:
        logger.warning(f"No data available for {today}. get_live_delays_data exiting.")
        return 1

    if route_types_map_df.empty:
        logger.warning(f"No routy type data available for {today}. get_live_delays_data exiting.")
        return 1

    if stop_count_df.empty:
        logger.warning(f"No stop count data available for {today}. get_live_delays_data exiting.")
        return 1

    if stop_location_map_df.empty:
        logger.warning(f"No stop location data available for {today}. get_live_delays_data exiting.")
        return 1

    # Write rt_df to csv for debugging
    stop_location_map_df.to_csv("stop_location_map_df.csv", index=False)

    final_metrics = sf.build_feature_group(rt_df, route_types_map_df, stop_count_df=stop_count_df, min_trip_updates_per_slot=LIVE_MIN_TRIP_UPDATES_PER_TIMESLOT)

    if dry_run:
        final_metrics.to_csv("live_feature_delays.csv", index=False)
        return -1

    if fg is None:
        logger.warning("No Hopsworks connection. Skipping upload.")
        return 2
    try:
        j, _ = fg.insert(final_metrics, write_options={"start_offline_materialization": False})
        j.run()
    except Exception as e:
        logger.warning(f"Failed to connect to Hopsworks and skipping upload. {e}")
        return 2

    return 0


if __name__ == "__main__":
    DRY_RUN = os.environ.get("DRY_RUN", "True").lower() == "true"
    WEATHER_FG_VERSION = int(os.environ.get("WEATHER_FG_VERSION", 3))
    DELAYS_FG_VERSION = int(os.environ.get("DELAYS_FG_VERSION", 10))
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
                version=DELAYS_FG_VERSION
            )
            weather_fg = fs.get_feature_group(
                name='weather',
                version=WEATHER_FG_VERSION
            )
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks. Exiting. {e}")
            sys.exit(1)

    logger.info(f"Starting live feature pipeline for {today}")
    weather_exit_code = get_live_weather_data(today, weather_fg, DRY_RUN)
    delays_exit_code = get_live_delays_data(today, delays_fg, DRY_RUN)

    logger.info(f"Completed live feature pipeline for {today} with weather exit code {weather_exit_code} and delays exit code {delays_exit_code}")

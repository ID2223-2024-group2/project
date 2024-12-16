import logging.config
import os
import sys
import time

import hopsworks
import pandas as pd

from koda.koda_constants import OperatorsWithRT
import koda.koda_pipeline as kp
import koda.koda_transform as kt

ON_TIME_MIN_SECONDS = -180
ON_TIME_MAX_SECONDS = 300
OPERATOR = OperatorsWithRT.X_TRAFIK

SAVE_TO_HW = True

# Set up logging with an absolute path
log_file_path = os.path.join(os.path.dirname(__file__), 'progress.log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def backfill_date(date: str):
    df = kp.get_trip_updates_for_day(date, OPERATOR)

    if df.empty:
        logger.warning(f"No data available for {date}. Pipeline exiting.")
        sys.exit(0)

    columns_to_keep = [
        "trip_id", "start_date", "timestamp",
        "vehicle_id", "stop_sequence", "stop_id", "arrival_delay",
        "arrival_time", "departure_delay", "departure_time"
    ]
    df = df[columns_to_keep]
    df = kt.keep_only_latest_stop_updates(df)

    # Set up arrival_time as our index and main datetime column
    df = df.dropna(subset=['arrival_time']) # Drop rows with missing arrival_time
    df['arrival_time'] = df['arrival_time'].astype(int)
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], unit='s')
    df = df.sort_values(by='arrival_time')
    df.set_index('arrival_time', inplace=True)

    # Make df with how many rows (stops) are in each hour
    hour_df = df.resample('h').size().reset_index()
    hour_df.sort_values(by='arrival_time', inplace=True)
    hour_df.columns = ['arrival_time', 'stop_count']

    # Prepare on_time column
    df["on_time"] = df["arrival_delay"].between(ON_TIME_MIN_SECONDS, ON_TIME_MAX_SECONDS)

    # Perform rolling metrics to capture trends
    WINDOW_SIZE = '3h'
    rolling_metrics = df.rolling(WINDOW_SIZE).agg({
        'arrival_delay': ['mean', 'max'],
        'departure_delay': ['mean', 'max'],
        'on_time': 'mean',
    }).reset_index()

    # Rename nested columns
    rolling_metrics.columns = ['arrival_time',
                               'mean_arrival_delay', 'max_arrival_delay',
                               'mean_departure_delay', 'max_departure_delay',
                               'on_time_mean']
    # Convert 'on_time_mean' to percentage
    rolling_metrics['on_time_mean'] *= 100

    # Resample rolling metrics to fixed hourly intervals to summarize day
    final_metrics = rolling_metrics.resample('h', on='arrival_time').agg({
        'mean_arrival_delay': 'mean',
        'max_arrival_delay': 'max',
        'mean_departure_delay': 'mean',
        'max_departure_delay': 'max',
        'on_time_mean': 'mean',
    }).reset_index()

    # Rename columns
    final_metrics.columns = ['arrival_time_bin',
                             'mean_arrival_delay_seconds', 'max_arrival_delay_seconds',
                             'mean_departure_delay_seconds', 'max_departure_delay_seconds',
                             'on_time_mean_percent']

    final_metrics['stop_count'] = hour_df['stop_count']

    # TODO: Change based on model?
    final_metrics.fillna(0,
                         inplace=True)  # Fill NaNs with 0 - During night time (00:00-02:00), no data is generally available

    # TODO: Data expectations?

    if not SAVE_TO_HW:
        final_metrics.to_csv("final_metrics.csv", index=False)
        sys.exit(0)

    if os.environ.get("HOPSWORKS_API_KEY") is None:
        os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()

    project = hopsworks.login()
    fs = project.get_feature_store()

    delays_fg = fs.get_or_create_feature_group(
        name='delays',
        description='Aggregated delay metrics per hour per day',
        version=1,
        primary_key=['arrival_time_bin'],
        event_time='arrival_time_bin'
    )
    delays_fg.insert(final_metrics)
    delays_fg.update_feature_description("arrival_time_bin", "Hourly time bin by stop arrival time")
    delays_fg.update_feature_description("mean_arrival_delay_seconds", "Mean stop arrival delay in seconds")
    delays_fg.update_feature_description("max_arrival_delay_seconds", "Max stop arrival delay in seconds")
    delays_fg.update_feature_description("mean_departure_delay_seconds", "Mean stop departure delay in seconds")
    delays_fg.update_feature_description("max_departure_delay_seconds", "Max stop departure delay in seconds")
    delays_fg.update_feature_description("on_time_mean_percent", "Percentage of stops on time (-2 to 5 minutes)")
    delays_fg.update_feature_description("stop_count", "Number of stops in the hour")


if __name__ == "__main__":
    # START_DATE = "2022-02-01"
    START_DATE = "2024-10-01"
    END_DATE = "2024-12-01"
    STRIDE = pd.DateOffset(days=1)

    dates = pd.date_range(START_DATE, END_DATE, freq=STRIDE)
    total_dates = len(dates)
    start_time = time.time()

    for i, datetime in enumerate(dates):
        logger.info("Starting processing for date: %s", datetime)
        date = datetime.strftime("%Y-%m-%d")
        backfill_date(date)
        elapsed_time = time.time() - start_time
        avg_time_per_date = elapsed_time / (i + 1)
        remaining_time = avg_time_per_date * (total_dates - (i + 1))
        logger.info("Completed processing for date: %s", date)
        logger.info("Progress: %d/%d (%.2f%%) - Estimated time remaining: %.2f seconds",
                     i + 1, total_dates, (i + 1) / total_dates * 100, remaining_time)
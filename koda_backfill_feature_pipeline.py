import os
import sys
import time

import hopsworks
import pandas as pd

from koda.koda_constants import OperatorsWithRT
import koda.koda_pipeline as kp
import koda.koda_transform as kt
from shared.file_logger import setup_logger

ON_TIME_MIN_SECONDS = -180
ON_TIME_MAX_SECONDS = 300
OPERATOR = OperatorsWithRT.X_TRAFIK
RUN_HW_MATERIALIZATION_EVERY = 10

log_file_path = os.path.join(os.path.dirname(__file__), 'koda_backfill.log')
logger = setup_logger('koda_backfill', log_file_path)


def backfill_date(date: str, fg=None, dry_run=True) -> (int, None | object):
    df, map_df = kp.get_koda_data_for_day(date, OPERATOR)

    if df.empty:
        logger.warning(f"No data available for {date}. Pipeline exiting.")
        return 1, None

    columns_to_keep = [
        "trip_id", "start_date", "timestamp",
        "vehicle_id", "stop_sequence", "stop_id", "arrival_delay",
        "arrival_time", "departure_delay", "departure_time"
    ]
    df = df[columns_to_keep]
    df = kt.keep_only_latest_stop_updates(df)
    # print(f"Unique trip_ids: {df['trip_id'].nunique()}")

    # Merge with map_df to get route_type
    df = df.merge(map_df, on='trip_id', how='inner')

    # Set up arrival_time as our index and main datetime column
    df = df.dropna(subset=['arrival_time'])  # Drop rows with missing arrival_time
    df['arrival_time'] = df['arrival_time'].astype(int)
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], unit='s')
    df.sort_values(by='arrival_time', inplace=True)
    df.set_index('arrival_time', inplace=True)

    # Group by route_type and resample to get stop count for each hour
    hour_df = df.groupby('route_type').resample('h').size().reset_index()
    hour_df.sort_values(by=['route_type', 'arrival_time'], inplace=True)
    hour_df.columns = ['route_type', 'arrival_time', 'stop_count']

    # Prepare on_time column
    df["on_time"] = df["arrival_delay"].between(ON_TIME_MIN_SECONDS, ON_TIME_MAX_SECONDS)

    # Identify the final stop for each trip
    final_stops = df.groupby('trip_id').tail(1)
    # Create a dictionary of final stop delays
    final_stop_delays_dict = final_stops.set_index('trip_id')['arrival_delay'].to_dict()
    # Map the final stop delays to the main DataFrame
    df['final_stop_delay'] = df['trip_id'].map(final_stop_delays_dict)

    # Sort the DataFrame by trip_id and stop_sequence
    df = df.sort_values(by=['trip_id', 'stop_sequence'])
    # Calculate the difference in delays between consecutive stops
    df['delay_change'] = df.groupby('trip_id')['arrival_delay'].diff()

    # Re-sort the DataFrame by arrival_time
    df.sort_values(by='arrival_time', inplace=True)

    # Perform rolling metrics to capture trends
    WINDOW_SIZE = '3h'
    rolling_metrics = df.groupby('route_type').rolling(WINDOW_SIZE).agg({
        'delay_change': ['mean', 'max', 'min', 'var'],
        'arrival_delay': ['mean', 'max', 'min', 'var'],
        'departure_delay': ['mean', 'max', 'min', 'var'],
        'on_time': 'mean',
        'final_stop_delay': 'mean'
    }).reset_index()

    # Rename nested columns
    rolling_metrics.columns = ['route_type', 'arrival_time',
                               'mean_delay_change', 'max_delay_change', 'min_delay_change', 'var_delay_change',
                               'mean_arrival_delay', 'max_arrival_delay', 'min_arrival_delay', 'var_arrival_delay',
                               'mean_departure_delay', 'max_departure_delay', 'min_departure_delay',
                               'var_departure_delay',
                               'mean_on_time', 'mean_final_stop_delay']

    # Convert 'on_time_mean' to percentage
    rolling_metrics['mean_on_time'] *= 100

    # Resample rolling metrics to fixed hourly intervals to summarize day
    final_metrics = rolling_metrics.groupby('route_type').resample('h', on='arrival_time').agg({
        'mean_delay_change': 'mean',
        'max_delay_change': 'max',
        'min_delay_change': 'min',
        'var_delay_change': 'mean',
        'mean_arrival_delay': 'mean',
        'max_arrival_delay': 'max',
        'min_arrival_delay': 'min',
        'var_arrival_delay': 'mean',
        'mean_departure_delay': 'mean',
        'max_departure_delay': 'max',
        'min_departure_delay': 'min',
        'var_departure_delay': 'mean',
        'mean_on_time': 'mean',
        'mean_final_stop_delay': 'mean'
    }).reset_index()

    # Rename columns
    final_metrics.columns = ['route_type', 'arrival_time_bin',
                             'mean_delay_change_seconds', 'max_delay_change_seconds', 'min_delay_change_seconds',
                             'var_delay_change_seconds',
                             'mean_arrival_delay_seconds', 'max_arrival_delay_seconds', 'min_arrival_delay_seconds',
                             'var_arrival_delay',
                             'mean_departure_delay_seconds', 'max_departure_delay_seconds',
                             'min_departure_delay_seconds', 'var_departure_delay',
                             'mean_on_time_percent', 'mean_final_stop_delay_seconds']

    # Merge the stop count information into the final metrics DataFrame
    final_metrics = final_metrics.merge(hour_df, left_on=['route_type', 'arrival_time_bin'],
                                        right_on=['route_type', 'arrival_time'], how='left')
    final_metrics.drop(columns=['arrival_time'], inplace=True)

    # TODO: Change based on model?
    final_metrics.fillna(0,
                         inplace=True)  # Fill NaNs with 0 - During night time (00:00-02:00), no data is generally available

    if dry_run:
        final_metrics.to_csv("koda_backfill.csv", index=False)
        return -1, None

    if fg is None:
        logger.warning("No Hopsworks connection. Skipping upload.")
        return 2, None
    try:
        j, _ = fg.insert(final_metrics, write_options={"start_offline_materialization": False})
        delays_fg.update_feature_description("arrival_time_bin", "Hourly time bin by stop arrival time")
        delays_fg.update_feature_description("mean_delay_change_seconds",
                                             "Mean change in delay between consecutive stops")
        delays_fg.update_feature_description("max_delay_change_seconds",
                                             "Max change in delay between consecutive stops")
        delays_fg.update_feature_description("min_delay_change_seconds",
                                             "Min change in delay between consecutive stops")
        delays_fg.update_feature_description("var_delay_change_seconds",
                                             "Variance of change in delay between consecutive stops")
        delays_fg.update_feature_description("mean_arrival_delay_seconds", "Mean stop arrival delay in seconds")
        delays_fg.update_feature_description("max_arrival_delay_seconds", "Max stop arrival delay in seconds")
        delays_fg.update_feature_description("min_arrival_delay_seconds", "Min stop arrival delay in seconds")
        delays_fg.update_feature_description("var_arrival_delay", "Variance of stop arrival delay in seconds")
        delays_fg.update_feature_description("mean_departure_delay_seconds", "Mean stop departure delay in seconds")
        delays_fg.update_feature_description("max_departure_delay_seconds", "Max stop departure delay in seconds")
        delays_fg.update_feature_description("min_departure_delay_seconds", "Min stop departure delay in seconds")
        delays_fg.update_feature_description("var_departure_delay", "Variance of stop departure delay in seconds")
        delays_fg.update_feature_description("mean_on_time_percent", "Percentage of stops on time (-3 to 5 minutes)")
        delays_fg.update_feature_description("stop_count", "Number of stops in the hour")
        delays_fg.update_feature_description("route_type",
                                             "Type of route (see https://www.trafiklab.se/api/gtfs-datasets/overview/extensions/#gtfs-regional-gtfs-sweden-3)")
        delays_fg.update_feature_description("mean_final_stop_delay_seconds",
                                             "Average delay at the final stop of each trip")
    except Exception as e:
        logger.warning(f"Failed to connect to Hopsworks and skipping upload. {e}")
        return 2, None

    return 0, j


if __name__ == "__main__":
    START_DATE = os.environ.get("START_DATE", "2023-12-10")
    END_DATE = os.environ.get("END_DATE", "2023-12-11")
    DRY_RUN = os.environ.get("DRY_RUN", "False").lower() == "true"
    STRIDE = pd.DateOffset(days=int(os.environ.get("STRIDE", 1)))

    try:
        dates = pd.date_range(START_DATE, END_DATE, freq=STRIDE)
    except (ValueError, TypeError):
        logger.error("Invalid date range. Exiting.")
        sys.exit(1)
    total_dates = len(dates)
    start_time = time.time()

    delays_fg = None
    if not DRY_RUN:
        if os.environ.get("HOPSWORKS_API_KEY") is None:
            os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()
        try:
            project = hopsworks.login()
            fs = project.get_feature_store()
            # TODO: Data expectations?
            delays_fg = fs.get_or_create_feature_group(
                name='delays',
                description='Aggregated delay metrics per hour per day',
                version=6,
                primary_key=['arrival_time_bin'],
                event_time='arrival_time_bin'
            )
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks. Exiting. {e}")
            sys.exit(1)

    logger.info("Starting backfill process for dates: %s - %s and stride: %s (%s days)", START_DATE, END_DATE, STRIDE,
                total_dates)

    exit_codes = []

    for i, datetime in enumerate(dates):
        logger.info("Starting processing for date: %s", datetime)
        date = datetime.strftime("%Y-%m-%d")
        exit_code, job = backfill_date(date, fg=delays_fg, dry_run=DRY_RUN)
        if exit_code == -1:
            logger.info("Dry run completed for date: %s", date)
            sys.exit(0)
        if exit_code == 0 and job is not None and i % RUN_HW_MATERIALIZATION_EVERY == 0:
            logger.info("Running offline materialization jobs")
            job.run()
        exit_codes.append(exit_code)
        elapsed_time = time.time() - start_time
        avg_time_per_date = elapsed_time / (i + 1)
        remaining_time = avg_time_per_date * (total_dates - (i + 1))
        logger.info("Completed processing for date: %s with exit code: %d", date, exit_code)
        logger.info("Progress: %d/%d (%.2f%%) - Estimated time remaining: %.2f seconds",
                    i + 1, total_dates, (i + 1) / total_dates * 100, remaining_time)

    logger.info("Backfill process completed for dates: %s - %s", START_DATE, END_DATE)
    successfully_uploaded = exit_codes.count(0)
    not_uploaded = exit_codes.count(2)
    missing_data = exit_codes.count(1)
    logger.info("Summary: Successfully uploaded: %d, Not uploaded: %d, Missing data: %d", successfully_uploaded,
                not_uploaded, missing_data)

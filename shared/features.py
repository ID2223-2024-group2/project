import pandas as pd
from hsfs.feature_group import FeatureGroup

import koda.koda_transform as kt
import shared.transform as st

ON_TIME_MIN_SECONDS = -180
ON_TIME_MAX_SECONDS = 300

MIN_TRIP_UPDATES_PER_TIMESLOT = 1


def on_time(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the 'on_time' feature for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing an 'arrival_delay' column.

    Returns:
        pd.Series: Series with the 'on_time' feature.
    """
    return df["arrival_delay"].between(ON_TIME_MIN_SECONDS, ON_TIME_MAX_SECONDS)


def final_stop_delay(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the 'final_stop_delay' feature for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'trip_id', 'stop_sequence', and 'arrival_delay' columns.

    Returns:
        pd.Series: Series with the 'final_stop_delay' feature.
    """
    # Ensure the DataFrame is sorted by 'trip_id' and 'stop_sequence'
    df = df.sort_values(by=['arrival_time'])

    # Identify the final stop for each trip
    final_stops = df.groupby('trip_id').tail(1)

    # Create a dictionary of final stop delays
    final_stop_delays_dict = final_stops.set_index('trip_id')['arrival_delay'].to_dict()

    # Map the final stop delays to the main DataFrame
    return final_stop_delays_dict


def windowed_lagged_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    if 'arrival_time' not in df.index.names:
        raise ValueError("arrival_time must be the index of the DataFrame")

    if not set(columns).issubset(df.columns):
        raise ValueError("Columns to lag are not present in the DataFrame")

    df = df.sort_values(by=['trip_id', 'stop_sequence'])
    window_size = 5 # Number of stops to consider for lagging

    # Create lagged features using windowing within each trip
    lagged_features = df.groupby(['route_type', 'trip_id']).rolling(window=window_size, closed='left').agg({
        'arrival_delay': 'mean',
        'departure_delay': 'mean',
        'delay_change': 'mean'
    }).reset_index()

    lagged_features.columns = ['route_type', 'trip_id', 'arrival_time', 'arrival_delay_lag_5stops',
                               'departure_delay_lag_5stops', 'delay_change_lag_5stops']

    return lagged_features


def build_feature_group(rt_df: pd.DataFrame, route_types_map_df: pd.DataFrame,
                        stop_count_df=pd.DataFrame(), min_trip_updates_per_slot=MIN_TRIP_UPDATES_PER_TIMESLOT) -> pd.DataFrame:
    columns_to_keep = [
        "trip_id", "start_date", "timestamp",
        "vehicle_id", "stop_sequence", "stop_id", "arrival_delay",
        "arrival_time", "departure_delay", "departure_time"
    ]
    rt_df = rt_df[columns_to_keep]
    rt_df = kt.keep_only_latest_stop_updates(rt_df)

    # Merge with map_df to get route_type
    rt_df['trip_id'] = rt_df['trip_id'].astype(str)
    route_types_map_df['trip_id'] = route_types_map_df['trip_id'].astype(str)
    rt_df = rt_df.merge(route_types_map_df, on='trip_id', how='inner')

    # Set up arrival_time as our index and main datetime column
    rt_df = rt_df.dropna(subset=['arrival_time'])  # Drop rows with missing arrival_time
    rt_df['arrival_time'] = rt_df['arrival_time'].astype(int)
    rt_df['arrival_time'] = pd.to_datetime(rt_df['arrival_time'], unit='s')

    rt_df.sort_values(by='arrival_time', inplace=True)
    rt_df.set_index('arrival_time', inplace=True)

    # Hopsworks expects the stop count to be a double for some reason.
    stop_count_df['stop_count'] = stop_count_df['stop_count'].astype('float64')

    # Group by route_type and resample to get trip update counts per hour
    trip_update_count_df = rt_df.groupby('route_type').resample('h').size().reset_index()
    trip_update_count_df.sort_values(by=['route_type', 'arrival_time'], inplace=True)
    trip_update_count_df.columns = ['route_type', 'arrival_time_bin', 'trip_update_count']

    rt_df['on_time'] = on_time(rt_df)
    final_stop_delays_dict = final_stop_delay(rt_df)
    rt_df['final_stop_delay'] = rt_df['trip_id'].map(final_stop_delays_dict)

    # Sort the DataFrame by trip_id and stop_sequence
    rt_df = rt_df.sort_values(by=['trip_id', 'stop_sequence'])
    # Calculate the difference in delays between consecutive stops
    rt_df['delay_change'] = rt_df.groupby('trip_id')['arrival_delay'].diff()

    lagged_columns = ['arrival_delay', 'departure_delay', 'delay_change']
    lagged_rt_df = windowed_lagged_features(rt_df, lagged_columns)

    # Re-sort the DataFrame by arrival_time
    rt_df.sort_values(by='arrival_time', inplace=True)

    # Perform rolling metrics to capture trends
    WINDOW_SIZE = '20min'
    rolling_metrics = rt_df.groupby('route_type').rolling(WINDOW_SIZE).agg({
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
    rolling_resampled_df = rolling_metrics.groupby('route_type').resample('h', on='arrival_time').agg({
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

    lagged_resampled_df = lagged_rt_df.groupby('route_type').resample('h', on='arrival_time').agg({
        'arrival_delay_lag_5stops': 'mean',
        'departure_delay_lag_5stops': 'mean',
        'delay_change_lag_5stops': 'mean'
    }).reset_index()

    final_metrics = rolling_resampled_df.merge(lagged_resampled_df, on=['route_type', 'arrival_time'], how='inner')

    # Merge the stop count information into the final metrics DataFrame
    final_metrics = final_metrics.merge(stop_count_df, left_on=['route_type', 'arrival_time'],
                                        right_on=['route_type', 'arrival_time'], how='left')

    # Rename columns
    final_metrics.columns = ['route_type', 'arrival_time_bin',
                             'mean_delay_change_seconds', 'max_delay_change_seconds', 'min_delay_change_seconds',
                             'var_delay_change_seconds',
                             'mean_arrival_delay_seconds', 'max_arrival_delay_seconds', 'min_arrival_delay_seconds',
                             'var_arrival_delay',
                             'mean_departure_delay_seconds', 'max_departure_delay_seconds',
                             'min_departure_delay_seconds', 'var_departure_delay',
                             'mean_on_time_percent', 'mean_final_stop_delay_seconds',
                             'mean_arrival_delay_seconds_lag_5stops', 'mean_departure_delay_seconds_lag_5stops', 'mean_delay_change_seconds_lag_5stops',
                             'stop_count']

    final_metrics = st.drop_rows_with_not_enough_updates(final_metrics, trip_update_count_df, min_trip_updates_per_slot)

    final_metrics.fillna(0,
                         inplace=True)  # Fill NaNs with 0 - During night time (00:00-02:00), no data is generally available
    return final_metrics


def delays_update_feature_descriptions(delays_fg: FeatureGroup) -> None:
    delays_fg.update_feature_description("route_type",
                                         "Type of route (see https://www.trafiklab.se/api/gtfs-datasets/overview/extensions/#gtfs-regional-gtfs-sweden-3)")
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
    delays_fg.update_feature_description("mean_final_stop_delay_seconds",
                                         "Average delay at the final stop of each trip")
    delays_fg.update_feature_description("mean_arrival_delay_seconds_lag_5stops", "Mean arrival delay lagged (windowed) 5 stops")
    delays_fg.update_feature_description("mean_departure_delay_seconds_lag_5stops", "Mean departure delay lagged 5 (windowed) stops")
    delays_fg.update_feature_description("mean_delay_change_seconds_lag_5stops", "Mean delay change lagged 5 (windowed) stops")
    delays_fg.update_feature_description("stop_count", "Number of scheduled stops in the hour")
    delays_fg.update_feature_description("trip_update_count", "Number of received trip updates in the hour")


def weather_update_feature_descriptions(weather_fg: FeatureGroup) -> None:
    weather_fg.update_feature_description("apparent_temperature", "Apparent temperature in Celsius")
    weather_fg.update_feature_description("cloud_cover", "Cloud cover in percentage")
    weather_fg.update_feature_description("date", "Timestamp of the weather data")
    weather_fg.update_feature_description("precipitation", "Precipitation in mm")
    weather_fg.update_feature_description("rain", "Rainfall in mm")
    weather_fg.update_feature_description("snow_depth", "Snow depth in m")
    weather_fg.update_feature_description("snowfall", "Snowfall in cm")
    weather_fg.update_feature_description("temperature_2m", "Temperature at 2m in Celsius")
    weather_fg.update_feature_description("wind_gusts_10m", "Wind gusts at 10m in km/h")
    weather_fg.update_feature_description("wind_speed_100m", "Wind speed at 100m in km/h")
    weather_fg.update_feature_description("wind_speed_10m", "Wind speed at 10m in km/h")
    weather_fg.update_feature_description("hour", "Hour of the day")
import shutil
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tqdm

import koda.koda_parse as kp
from koda.koda_constants import FeedType, OperatorsWithRT, route_types


def get_rt_feather_path(operator: str, feed_type: str, date: str, hour: str):
    rt_folder_path = kp.get_rt_dir_path(operator, date)
    return f"{rt_folder_path}/{operator}-{feed_type.lower()}-{date}T{hour}.feather"


# For context, see getdata in pykoda project

def normalize_keys(df: pd.DataFrame) -> None:
    """Reformat the name of the keys to a consistent format, according to GTFS"""
    renames = {'tripUpdate_trip_tripId': 'trip_id', 'tripUpdate_trip_startDate': 'start_date',
               'tripUpdate_trip_directionId': 'direction_id', 'tripUpdate_trip_routeId': 'route_id',
               'tripUpdate_trip_scheduleRelationship': 'schedule_relationship',
               'tripUpdate_trip_startTime': 'start_time',
               'tripUpdate_timestamp': 'timestamp', 'tripUpdate_vehicle_id': 'vehicle_id',
               'stopSequence': 'stop_sequence', 'stopId': 'stop_id',
               'scheduleRelationship': 'schedule_relationship2',
               'vehicle_trip_tripId': 'trip_id', 'vehicle_trip_scheduleRelationship': 'schedule_relationship',
               'vehicle_timestamp': 'timestamp', 'vehicle_vehicle_id': 'vehicle_id',
               'vehicle_trip_startTime': 'start_time', 'vehicle_trip_startDate': 'start_date',
               'vehicle_trip_routeId': 'route_id', 'vehicle_trip_directionId': 'direction_id',
               'tripUpdate_stopTimeUpdate_stopSequence': 'stop_sequence',
               'tripUpdate_stopTimeUpdate_stopId': 'stop_id',
               'tripUpdate_stopTimeUpdate_arrival_delay': 'arrival_delay',
               'tripUpdate_stopTimeUpdate_arrival_time': 'arrival_time',
               'tripUpdate_stopTimeUpdate_departure_delay': 'departure_delay',
               'tripUpdate_stopTimeUpdate_departure_time': 'departure_time',
               'tripUpdate_stopTimeUpdate_arrival_uncertainty': 'arrival_uncertainty',
               'tripUpdate_stopTimeUpdate_departure_uncertainty': 'departure_uncertainty',
               'alert_activePeriod_start': 'period_start', 'alert_activePeriod_end': 'period_end',
               'alert_informedEntity_routeId': 'route_id', 'alert_informedEntity_stopId': 'stop_id',
               'alert_informedEntity_trip_tripId': 'trip_id',
               'alert_informedEntity_trip_scheduleRelationship': 'schedule_relationship',
               'alert_headerText_translation_text': 'header_text',
               'alert_descriptionText_translation_text': 'description_text',
               }
    df.rename(columns=renames, inplace=True)


def sanitise_array(df: pd.DataFrame) -> None:
    normalize_keys(df)

    # Remove columns and rows with all NaNs
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    # Remove old indexes
    df.drop(columns='level_0', inplace=True, errors='ignore')

    # Remove duplicated entries, ignoring timpestamps and index
    keys = list(df.keys())
    with contextlib.suppress(ValueError):
        keys.remove('timestamp')
        keys.remove('index')

        # These may be updated in the database, so ignore as well
        keys.remove('arrival_delay')
        keys.remove('arrival_time')
        keys.remove('departure_delay')
        keys.remove('departure_time')
        keys.remove('arrival_uncertainty')
        keys.remove('departure_uncertainty')

    df.drop_duplicates(subset=keys, inplace=True, keep='last')


def _read_pb_file_helper(file_path):
    try:
        return kp.read_pb_to_dataframe(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found")
        return pd.DataFrame()

def read_rt_hour_to_df(operator: OperatorsWithRT, feed_type: FeedType, date: str, hour: int, executor: ProcessPoolExecutor = None) -> (pd.DataFrame, str):
    hour_filled = str(hour).zfill(2)
    feather_path = get_rt_feather_path(operator.value, feed_type.value, date, hour_filled)
    if os.path.exists(feather_path):
        # print(f"Reading from {feather_path}")
        return pd.read_feather(feather_path), feather_path

    search_path = kp.get_rt_hour_dir_path(operator.value, feed_type.value, date, hour)
    file_list = []
    for root, _, files in os.walk(search_path):
        for file in files:
            if file.endswith(".pb"):
                file_list.append(os.path.join(root, file))

    # print(f"Reading {len(file_list)} files with {os.cpu_count() - 2} processes")

    if executor is None:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            future_list = [executor.submit(_read_pb_file_helper, file_path) for file_path in file_list]
            df_list = [future.result() for future in as_completed(future_list)]
    else:
        future_list = [executor.submit(_read_pb_file_helper, file_path) for file_path in file_list]
        df_list = [future.result() for future in as_completed(future_list)]

    df_list = [df for df in df_list if not df.empty]
    if df_list:
        merged_df = pd.concat(df_list, axis=0)
    else:
        print(f"No data found in {search_path}")
        merged_df = pd.DataFrame()
    # Force casts:
    castings = {}
    for k in merged_df.keys():
        if 'timestamp' in k:  # Timestamps should be ints, not strings
            castings[k] = np.int64
        elif k == 'id':
            castings[k] = np.int64
    merged_df.dropna(how='all', inplace=True)  # Remove rows of only NaNs
    merged_df = merged_df.astype(castings)

    # Remove dots from column names
    rename = dict((k, k.replace('.', '_')) for k in merged_df.keys() if '.' in k)
    merged_df.rename(columns=rename, inplace=True)

    # Clean up duplicates, fix keys, etc
    sanitise_array(merged_df)

    if merged_df.empty:  # Feather does not support a DF without columns, so add a dummy one
        merged_df['_'] = np.zeros(len(merged_df), dtype=np.bool_)

    merged_df.reset_index(inplace=True)
    merged_df.to_feather(feather_path, compression='zstd', compression_level=9)
    return merged_df, feather_path


def read_rt_day_to_df(operator: OperatorsWithRT, feed_type: FeedType, date: str, remove_folder_after=False, executor: ProcessPoolExecutor = None) -> (pd.DataFrame, list[str]):
    frames = []
    feather_paths = []
    for hour in tqdm.tqdm(range(24), desc=f"Reading {operator.value} {feed_type.value} {date}"):
        df, path = read_rt_hour_to_df(operator, feed_type, date, hour, executor=executor)
        feather_paths.append(path)
        df.drop(columns='index', errors='ignore', inplace=True)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), feather_paths
    if remove_folder_after:
        rt_folder_path = kp.get_rt_dir_path(operator.value, date)
        operator_folder = os.path.join(rt_folder_path, operator.value)
        if os.path.exists(operator_folder):
            print(f"Removing {operator_folder}")
            shutil.rmtree(operator_folder)
    return pd.concat(frames, axis=0), feather_paths


# From pykoda datautils
def drop_tripupdates_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df.sort_values(by='timestamp', ascending=True)

    # Not all feeds provide exactly the same fields, so this filters for it:
    keys = list({'trip_id', 'direction_id', 'stop_sequence', 'stop_id'}.intersection(df.keys()))
    return df.drop_duplicates(subset=keys, keep='last')


def keep_only_latest_stop_updates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Ensure the DataFrame is sorted by 'timestamp'
    df.sort_values(by='timestamp', ascending=True, inplace=True)

    # Drop duplicates, keeping the last occurrence (latest timestamp)
    df = df.drop_duplicates(subset=['trip_id', 'stop_id'], keep='last')

    # Reset the index
    return df.reset_index(drop=True)


def create_route_types_map_df(rt_df: pd.DataFrame, trips_df: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
    if rt_df.empty or trips_df.empty or routes_df.empty:
        raise ValueError("One or more DataFrames are empty")
    # Make sure trip_id is a string for both DataFrames
    rt_df['trip_id'] = rt_df['trip_id'].astype(str)
    trips_df['trip_id'] = trips_df['trip_id'].astype(str)

    # Drop unnecessary columns
    map_df = rt_df.drop(columns=['id', 'start_date', 'schedule_relationship', 'timestamp',
                                'vehicle_id', 'stop_sequence', 'stop_id', 'arrival_delay',
                                'arrival_time', 'arrival_uncertainty', 'departure_delay',
                                'departure_time', 'departure_uncertainty'])
    trips_df = trips_df.drop(columns=['service_id', 'trip_headsign', 'direction_id',
                                      'shape_id'])
    routes_df = routes_df.drop(columns=['agency_id', 'route_short_name', 'route_long_name', 'route_desc'])

    # Drop route_id from trips_df if it exists as it is likely incomplete/incorrect
    if 'route_id' in map_df.keys():
        map_df.drop(columns=['route_id'], errors='ignore', inplace=True)
    map_df = map_df.merge(trips_df, on='trip_id', how='inner')
    # Drop route_type from routes_df if it exists
    if 'route_type' in map_df.keys():
        map_df.drop(columns=['route_type'], errors='ignore', inplace=True)
    map_df = map_df.merge(routes_df, on='route_id', how='inner')
    # Remove duplicate trip_ids
    map_df = map_df.drop_duplicates(subset=['trip_id'])
    # Map route_type to strings with route_types dict
    map_df['route_type_description'] = map_df['route_type'].map(route_types)

    if len(map_df) < 10:
        print(f"Warning: map_df only has {len(map_df)} rows, likely missing data")

    return map_df

def create_stop_count_df(date: str, stop_times_df: pd.DataFrame, route_types_map_df: pd.DataFrame) -> pd.DataFrame:
    if stop_times_df.empty or route_types_map_df.empty:
        raise ValueError("One or more DataFrames are empty")

    # Drop rows with missing arrival_time
    stop_times_df = stop_times_df.dropna(subset=['arrival_time'])

    # Ensure trip_ids are integers for both DataFrames and merge them
    stop_times_df['trip_id'] = stop_times_df['trip_id'].astype(str)
    route_types_map_df['trip_id'] = route_types_map_df['trip_id'].astype(str)
    stop_times_df = stop_times_df.merge(route_types_map_df, on='trip_id', how='inner')

    # Remove arrival_times > 24:00:00 (valid in GTFS but not in pandas and not useful for our purposes)
    stop_times_df = stop_times_df[stop_times_df['arrival_time'] < '24:00:00']

    # Set up arrival_time as our index and main datetime column
    stop_times_df['arrival_time'] = pd.to_datetime(date + ' ' + stop_times_df['arrival_time'])
    stop_times_df.sort_values(by='arrival_time', inplace=True)
    stop_times_df.set_index('arrival_time', inplace=True)

    # Group by route_type and resample to get stop count for each hour
    stop_count_df = stop_times_df.groupby('route_type').resample('h').size().reset_index()
    stop_count_df.sort_values(by=['route_type', 'arrival_time'], inplace=True)
    stop_count_df.columns = ['route_type', 'arrival_time', 'stop_count']
    return stop_count_df
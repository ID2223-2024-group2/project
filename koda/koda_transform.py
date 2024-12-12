from concurrent import futures
import contextlib
import os

import numpy as np
import pandas as pd
import tqdm

import koda.koda_parse as kp


def get_rt_feather_path(operator, feed_type, date, hour):
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
        return pd.DataFrame()


def read_rt_hour_to_df(operator: str, feed_type: str, date: str, hour: int) -> pd.DataFrame:
    feather_path = get_rt_feather_path(operator, feed_type, date, hour)
    if os.path.exists(feather_path):
        print(f"Reading from {feather_path}")
        return pd.read_feather(feather_path)

    search_path = kp.get_rt_hour_dir_path(operator, feed_type, date, hour)
    file_list = []
    for root, _, files in os.walk(search_path):
        for file in files:
            if file.endswith(".pb"):
                file_list.append(os.path.join(root, file))

    with futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        print(f"Reading {len(file_list)} files with {os.cpu_count() - 2} processes")
        future_list = [executor.submit(_read_pb_file_helper, file_path) for file_path in file_list]
        df_list = [future.result() for future in futures.as_completed(future_list)]

    df_list = [df for df in df_list if not df.empty]
    if not df_list:
        print(f"No files found in {search_path}")
        return pd.DataFrame()

    merged_df = pd.concat(df_list, axis=0)
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
    return merged_df


def read_rt_day_to_df(operator: str, feed_type: str, date: str) -> pd.DataFrame:
    frames = []
    for hour in tqdm.tqdm(range(24), desc=f"Reading {operator} {feed_type} {date}"):
        df = read_rt_hour_to_df(operator, feed_type, date, hour)
        df.drop(columns='index', errors='ignore', inplace=True)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0)


# From pykoda datautils
def drop_tripupdates_duplicates(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df.sort_values(by='timestamp', ascending=True)

    # Not all feeds provide exactly the same fields, so this filters for it:
    keys = list({'trip_id', 'direction_id', 'stop_sequence', 'stop_id'}.intersection(df.keys()))
    df.drop_duplicates(subset=keys, keep='last', inplace=True)
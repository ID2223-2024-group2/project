import contextlib
import json
import warnings
import zipfile

import numpy as np
import py7zr
import os
import pandas as pd
from google.protobuf import json_format # Use protobuf==3.20.3
from protobuf_defs.gtfs_realtime_pb2 import FeedMessage

DATA_DIR = "./dev_data/koda_data"

def get_compression_type(file_path: str) -> str:
    with open(file_path, "rb") as f:
        header = f.read(5)
    if header[:4] == b"7z\xbc\xaf":
        return "7z"
    elif header[:4] == b"PK\x03\x04":
        return "zip"
    elif header[:3] == b"Rar":
        return "rar"
    elif header[:2] == b"\x1f\x8b":
        return "gzip"
    elif header[:4] == b"\x42\x5a\x68":
        return "bzip2"
    elif header[:6] == b"\x75\x73\x74\x61\x72":
        return "tar"
    else:
        print(f"Unknown compression type: {header}")
        return "unknown"

def unzip_data(input_path: str, data_dir = DATA_DIR):
    print(f"Unzipping {input_path}")
    input_file = os.path.basename(input_path)
    output_path = os.path.join(data_dir, input_file.replace(".7z", ""))
    # Check if file is already unzipped
    if os.path.exists(output_path):
        print("File already unzipped.")
        return output_path
    compression_type = get_compression_type(input_path)
    if compression_type == "7z":
        with py7zr.SevenZipFile(input_path, mode="r") as z:
            z.extractall(output_path)
    elif compression_type == "zip":
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
    else:
        raise ValueError(f"Unsupported compression type: {compression_type}")
    return output_path


def _is_json(series: pd.Series) -> bool:
    try:
        return isinstance(series[0][0], dict)
    except TypeError:
        return False


def unpack_jsons(df: pd.DataFrame) -> pd.DataFrame:
    keys_to_sanitise = []
    for k in list(df.keys()):
        # If the content is a json, unpack and remove
        if df[k].dtype == np.dtype('O') and _is_json(df[k]):
            keys_to_sanitise.append(k)

    if keys_to_sanitise:
        indexes = []
        unpacked = {k: [] for k in keys_to_sanitise}
        for ix in df.index:
            for k in keys_to_sanitise:
                this_unpack = pd.json_normalize(df[k][ix])
                unpacked[k].append(this_unpack)
                indexes.extend(ix for _ in range(len(this_unpack)))

        df.drop(keys_to_sanitise, axis='columns', inplace=True)

        unpacked_series = []
        for k in keys_to_sanitise:
            this_df = pd.concat(unpacked[k], axis='index').reset_index(drop=True)
            this_df.rename(columns={curr_name: '_'.join((k, curr_name)) for curr_name in this_df.keys()},
                           inplace=True)
            unpacked_series.append(this_df)

        repeated = df.iloc[indexes].reset_index(drop=True)
        df = pd.concat([repeated] + unpacked_series, axis='columns')

    for k in df.keys():
        if df[k].dtype == np.dtype('O') and _is_json(df[k]):
            warnings.warn(RuntimeWarning(f'There are extra json in column {k}'))
    return df


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

# For context, see _pasrse_gtfs in pykoda project
def read_pb_to_dataframe(input_path:str, operator, feed_type, year, month, day, hour, minute, second):
    filename = f"{operator}-{feed_type.lower()}-{year}-{month}-{day}T{hour}-{minute}-{second}Z.pb"
    file_path = os.path.join(input_path, operator, feed_type, year, month, day, hour, filename)

    with open(file_path, 'rb') as file:
        proto_message = FeedMessage()
        # Not sure if decompression is necessary
        # pbfile = gzip.decompress(gtfsrt)
        proto_message.ParseFromString(file.read())

    msg_json = json_format.MessageToJson(proto_message)
    msg_dict = json.loads(msg_json)

    df = pd.json_normalize(msg_dict.get('entity', dict()), sep='_')
    df = unpack_jsons(df)
    df.reset_index(drop=True, inplace=True)
    return df


# For context, see merge_files in pykoda project
def read_rt_folder_to_df(input_path:str, operator, feed_type, year, month, day, hour):
    df_list = []
    for minute in range(60):
        for second in range(60):
            try:
                df = read_pb_to_dataframe(input_path, operator, feed_type, year, month, day, hour, str(minute), str(second))
                df_list.append(df)
            except FileNotFoundError:
                pass
    merged_df = pd.concat(df_list, axis=0)
    # Force casts:
    castings = dict()
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
    return merged_df
import json
import warnings
import zipfile
import os

import numpy as np
import py7zr
import pandas as pd
from google.protobuf import json_format  # Use protobuf==3.20.3
from protobuf_defs.gtfs_realtime_pb2 import FeedMessage

DATA_DIR = "./dev_data/koda_data"


def get_static_dir_path(operator: str, date: str, data_dir=DATA_DIR) -> str:
    return f'{data_dir}/{operator}_static_{date.replace("-", "_")}'


def get_rt_dir_path(operator: str, date: str, data_dir=DATA_DIR) -> str:
    return f'{data_dir}/{operator}_rt_{date.replace("-", "_")}'


def get_rt_dir_info(rt_dir_path: str):
    parts = rt_dir_path.split("/")
    operator = parts[-1].split("_")[0]
    date = "_".join(parts[-1].split("_")[2:])
    return operator, date


def get_rt_hour_dir_path(operator: str, feed_type: str, date: str, hour: int, data_dir=DATA_DIR) -> str:
    year, month, day = date.split("-")
    hour_filled = str(hour).zfill(2)
    rt_dir_path = get_rt_dir_path(operator, date, data_dir)
    subfolder_path = os.path.join(operator, feed_type, year, month, day, hour_filled)
    hour_dir_path = os.path.join(rt_dir_path, subfolder_path)
    return hour_dir_path


def get_pb_file_path(operator: str, feed_type: str, date: str, hour: int, minute: int, second: int, data_dir=DATA_DIR):
    year, month, day = date.split("-")
    hour_filled = str(hour).zfill(2)
    minute_filled = str(minute).zfill(2)
    second_filled = str(second).zfill(2)
    file_name = f"{operator}-{feed_type.lower()}-{year}-{month}-{day}T{hour_filled}-{minute_filled}-{second_filled}Z.pb"
    hour_dir_path = get_rt_hour_dir_path(operator, feed_type, date, hour, data_dir)
    file_path = os.path.join(hour_dir_path, file_name)
    return file_path


def get_pb_file_info(file_path: str):
    parts = file_path.split("/")
    operator = parts[-7]
    feed_type = parts[-6]
    date = "-".join(parts[-5:-2])
    hour, minute, second = parts[-2].split("-")
    return operator, feed_type, date, int(hour), int(minute), int(second)


def get_compression_type(file_path: str) -> str:
    with open(file_path, "rb") as f:
        header = f.read(5)
    if header[:4] == b"7z\xbc\xaf":
        return "7z"
    if header[:4] == b"PK\x03\x04":
        return "zip"
    if header[:3] == b"Rar":
        return "rar"
    if header[:2] == b"\x1f\x8b":
        return "gzip"
    if header[:4] == b"\x42\x5a\x68":
        return "bzip2"
    if header[:6] == b"\x75\x73\x74\x61\x72":
        return "tar"
    warnings.warn(f"Unknown compression type: {header}")
    return "unknown"


def unzip_gtfs_archive(input_path: str, data_dir=DATA_DIR, remove_archive_after=False) -> str:
    print(f"Unzipping {input_path}")
    input_file = os.path.basename(input_path)
    output_path = os.path.join(data_dir, input_file.replace(".7z", ""))
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
    if remove_archive_after:
        print(f"Removing {input_path}")
        os.remove(input_path)
    return output_path


# For context, see getdata in pykoda project

def is_json(series: pd.Series) -> bool:
    try:
        return isinstance(series[0][0], dict)
    except TypeError:
        return False


def unpack_jsons(df: pd.DataFrame) -> pd.DataFrame:
    keys_to_sanitise = []
    for k in list(df.keys()):
        # If the content is a json, unpack and remove
        if df[k].dtype == np.dtype('O') and is_json(df[k]):
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
        if df[k].dtype == np.dtype('O') and is_json(df[k]):
            warnings.warn(RuntimeWarning(f'There are extra json in column {k}'))
    return df


def read_pb_to_dataframe(file_path: str) -> pd.DataFrame:
    with open(file_path, 'rb') as file:
        proto_message = FeedMessage()
        proto_message.ParseFromString(file.read())

    msg_json = json_format.MessageToJson(proto_message)
    msg_dict = json.loads(msg_json)

    df = pd.json_normalize(msg_dict.get('entity', {}), sep='_')
    df = unpack_jsons(df)
    df.reset_index(drop=True, inplace=True)
    return df

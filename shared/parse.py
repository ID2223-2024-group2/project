import json
import os
import warnings
import zipfile

import numpy as np
import pandas as pd
import py7zr
from google.protobuf import json_format

from protobuf_defs.gtfs_realtime_pb2 import FeedMessage
from shared.constants import OperatorsWithRT, StaticDataTypes


def get_static_file_path(operator: str, date: str, static_data_type: str, data_dir: str):
    dir_path = get_static_dir_path(operator, date, data_dir)
    file_name = f"{static_data_type}.txt"
    return os.path.join(dir_path, file_name)


def get_static_dir_path(operator: str, date: str, data_dir: str) -> str:
    return f'{data_dir}/{operator}_static_{date.replace("-", "_")}'


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


def read_static_data_to_dataframe(operator: OperatorsWithRT, static_data_type: StaticDataTypes, date:str, data_dir: str) -> pd.DataFrame:
    file_path = get_static_file_path(operator.value, date, static_data_type.value, data_dir=data_dir)
    return pd.read_csv(file_path, sep=",")


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


def unzip_gtfs_archive(input_path: str, data_dir: str, remove_archive_after=False) -> str:
    print(f"Unzipping {input_path}")
    input_file = os.path.basename(input_path)
    output_path = os.path.join(data_dir, input_file.replace(".7z", ""))
    if os.path.exists(output_path):
        print(f"File already unzipped to {output_path}.")
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

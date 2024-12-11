import json
import zipfile

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

    return df
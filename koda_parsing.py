import json

import py7zr
import os
import pandas as pd
from google.protobuf import json_format # Use protobuf==3.20.3
from protobuf_defs.gtfs_realtime_pb2 import FeedMessage

DATA_FOLDER = "dev_data"
INPUT_FILE = "gtfs-rt-xt-TripUpdates_date_2023-01-05"

# Unzip file with 7z
with py7zr.SevenZipFile(f"{DATA_FOLDER}/{INPUT_FILE}", mode="r") as z:
    z.extractall(DATA_FOLDER)


def read_pb_to_dataframe(operator, feed_type, year, month, day, hour, minute, second):
    filename = f"{operator}-{feed_type.lower()}-{year}-{month}-{day}T{hour}-{minute}-{second}Z.pb"
    file_path = os.path.join(DATA_FOLDER, operator, feed_type, year, month, day, hour, filename)

    with open(file_path, 'rb') as file:
        proto_message = FeedMessage()
        # Not sure if decompression is necessary
        # pbfile = gzip.decompress(gtfsrt)
        proto_message.ParseFromString(file.read())

    msg_json = json_format.MessageToJson(proto_message)
    msg_dict = json.loads(msg_json)

    df = pd.json_normalize(msg_dict.get('entity', dict()), sep='_')

    return df


# Example usage
operator = "xt"
feed_type = "TripUpdates"
year = "2023"
month = "01"
day = "05"
hour = "12"
minute = "05"
second = "38"

# TODO: Figure out how to read cumulative data from multiple files without duplicates
df = read_pb_to_dataframe(operator, feed_type, year, month, day, hour, minute, second)

print(f"Data for {operator} {feed_type} {year}-{month}-{day}T{hour}:{minute}:{second}Z")
print(df.head())
print(df.iloc[0])
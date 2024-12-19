import os

import numpy as np
import pandas as pd

from koda.koda_constants import OperatorsWithRT
from koda.koda_transform import sanitise_array

DATA_DIR = "./dev_data/gtfsr_data"

def get_last_updated_path(operator: str, data_dir=DATA_DIR) -> str:
    return f'{data_dir}/.{operator}_last_updated'

# NOTE: We are using the same file for every date to keep only the latest data
def get_rt_feather_path(operator: str, data_dir=DATA_DIR):
    return f'{data_dir}/{operator}_rt.feather'


def get_route_types_map_df_feather_path(operator: str, data_dir=DATA_DIR):
    return f"{data_dir}/{operator}_route_types_map.feather"


def get_stop_count_df_feather_path(operator: str, data_dir=DATA_DIR):
    return f"{data_dir}/{operator}_stop_count.feather"


def get_trips_df_feather_path(operator: str, data_dir=DATA_DIR):
    return f"{data_dir}/{operator}_trips.feather"


def get_routes_df_feather_path(operator: str, data_dir=DATA_DIR):
    return f"{data_dir}/{operator}_routes.feather"


def get_stop_times_df_feather_path(operator: str, data_dir=DATA_DIR):
    return f"{data_dir}/{operator}_stop_times.feather"


def get_stop_location_map_feather_path(operator: str, data_dir=DATA_DIR):
    return f"{data_dir}/{operator}_stop_location_map.feather"


def write_last_updated(operator: OperatorsWithRT, last_updated: str):
    path = get_last_updated_path(operator.value)
    with open(path, 'w') as f:
        f.write(last_updated)


def read_last_updated(operator: OperatorsWithRT) -> str:
    path = get_last_updated_path(operator.value)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    else:
        return ''


def parse_live_pb(operator: OperatorsWithRT, raw_df: pd.DataFrame, force=False) -> pd.DataFrame:
    feather_path = get_rt_feather_path(operator.value)
    if os.path.exists(feather_path) and not force:
        return pd.read_feather(feather_path)

    # Force casts:
    castings = {}
    for k in raw_df.keys():
        if 'timestamp' in k:  # Timestamps should be ints, not strings
            castings[k] = np.int64
        elif k == 'id':
            castings[k] = np.int64
    raw_df.dropna(how='all', inplace=True)  # Remove rows of only NaNs
    raw_df = raw_df.astype(castings)

    # Remove dots from column names
    rename = dict((k, k.replace('.', '_')) for k in raw_df.keys() if '.' in k)
    raw_df.rename(columns=rename, inplace=True)

    # Clean up duplicates, fix keys, etc
    sanitise_array(raw_df)

    if raw_df.empty:  # Feather does not support a DF without columns, so add a dummy one
        raw_df['_'] = np.zeros(len(raw_df), dtype=np.bool_)

    raw_df.reset_index(inplace=True)
    raw_df.to_feather(feather_path, compression='zstd', compression_level=9)
    return raw_df
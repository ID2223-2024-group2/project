import os

import numpy as np
import pandas as pd

from koda.koda_constants import OperatorsWithRT
from koda.koda_transform import sanitise_array

DATA_DIR = "./dev_data/gtfsr_data"

def get_rt_feather_path(operator: str, date: str, data_dir=DATA_DIR):
    return f'{data_dir}/{operator}_rt_{date.replace("-", "_")}.feather'


def parse_live_pb(operator: OperatorsWithRT, date: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    feather_path = get_rt_feather_path(operator.value, date)
    if os.path.exists(feather_path):
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
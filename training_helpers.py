import pandas as pd
import numpy as np
import os

LATEST_FV = 4
LATEST_TD = 17
NUM_FEATURES = 13
TO_PREDICT = ["mean_arrival_delay_seconds", "max_arrival_delay_seconds", "mean_departure_delay_seconds", "max_departure_delay_seconds"]

_split_cycle_2 = [
    "train", "train", "train", "test",
    "train", "train", "train", "train",
    "train", "train", "train", "test",
    "train", "train", "train", "test",
    "train", "train", "train", "train"
]

_split_cycle_3 = [
    "train", "train", "validate", "test",
    "train", "train", "train", "test",
    "train", "train", "train", "validate",
    "train", "train", "train", "test",
    "train", "train", "train", "validate"
]


def strip_dates(df):
    return df.drop(["date", "arrival_time_bin"], axis=1)


def train_test_split(df):
    total_length = len(df)
    assignments = pd.Series([_split_cycle_2[i % len(_split_cycle_2)] for i in range(total_length)])
    train = df.loc[assignments == "train"]
    test = df.loc[assignments == "test"]
    return train, test


# Splits into train, test and validate sets.
def train_validate_test_split(df):
    total_length = len(df)
    assignments = pd.Series([_split_cycle_3[i % len(_split_cycle_3)] for i in range(total_length)])
    train = df.loc[assignments == "train"]
    validate = df.loc[assignments == "validate"]
    test = df.loc[assignments == "test"]
    return train, validate, test


def get_model_dir(name):
    model_dir = "best_models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return os.path.join(model_dir, name)


if __name__ == "__main__":
    columns = ["foo", "bar"]
    to_test = pd.DataFrame(np.random.rand(50, len(columns)), columns=columns)
    print(train_validate_test_split(to_test))

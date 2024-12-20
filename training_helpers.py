import pandas as pd
import numpy as np
import os

LATEST_FV = 4
LATEST_TD = 20
DATASET = 7517
TO_USE = ["route_type", "stop_count", "temperature_2m", "snowfall", "snow_depth", "wind_gusts_10m", "hour"]
#TO_USE = ["route_type", "stop_count", "temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth", "cloud_cover", "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m", "hour"]
TO_PREDICT = ["mean_on_time_percent"]

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

_route_types = [100, 101, 102, 103, 105, 106, 401, 700, 714, 900, 1000, 1501]


def strip_dates(df):
    return df.drop(["date", "arrival_time_bin"], axis=1)


def one_hot(df):
    if "route_type" in df:
        encoded = pd.get_dummies(df["route_type"], prefix="route_type", dtype="int64")
        for rt in _route_types:
            if f"route_type_{rt}" not in encoded.columns:
                encoded[f"route_type_{rt}"] = 0
        df = pd.concat([df, encoded], axis=1).drop(columns=["route_type"])
    return df


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


def load_dataset():
    dataset_dir = "training_datasets"
    x_name = os.path.join(dataset_dir, f"features_{DATASET}.pickle")
    y_name = os.path.join(dataset_dir, f"labels_{DATASET}.pickle")
    x_all = strip_dates(pd.read_pickle(x_name))
    x_all = one_hot(x_all[TO_USE])
    y_all = pd.read_pickle(y_name)[TO_PREDICT]
    return x_all, y_all


def save_dataset(x_all, y_all):
    dataset_dir = "training_datasets"
    number_of_rows = x_all.shape[0]
    x_name = os.path.join(dataset_dir, f"features_{number_of_rows}.pickle")
    y_name = os.path.join(dataset_dir, f"labels_{number_of_rows}.pickle")
    x_all.to_pickle(x_name)
    y_all.to_pickle(y_name)


def get_model_dir(name):
    model_dir = "best_models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return os.path.join(model_dir, name)


if __name__ == "__main__":
    columns = ["foo", "bar"]
    to_test = pd.DataFrame(np.random.rand(50, len(columns)), columns=columns)
    print(train_validate_test_split(to_test))

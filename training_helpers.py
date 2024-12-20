import pandas as pd
import numpy as np
import os

LATEST_FV = 5
LATEST_TD = 1
DATASET = 6055
TO_USE = ["mean_arrival_delay_seconds_lag_5stops", "mean_departure_delay_seconds_lag_5stops", "mean_delay_change_seconds_lag_5stops", "route_type", "stop_count", "temperature_2m", "snowfall", "snow_depth", "wind_gusts_10m", "hour"]
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


def train_test_split_cyclical(df):
    total_length = len(df)
    assignments = pd.Series([_split_cycle_2[i % len(_split_cycle_2)] for i in range(total_length)])
    train = df.loc[assignments == "train"]
    test = df.loc[assignments == "test"]
    return train, test


def train_test_split_time(df, test_start):
    train = df.loc[df["arrival_time_bin"] < test_start]
    test = df.loc[df["arrival_time_bin"] >= test_start]
    return train, test


# Splits into train, test and validate sets.
def train_validate_test_split(df):
    total_length = len(df)
    assignments = pd.Series([_split_cycle_3[i % len(_split_cycle_3)] for i in range(total_length)])
    train = df.loc[assignments == "train"]
    validate = df.loc[assignments == "validate"]
    test = df.loc[assignments == "test"]
    return train, validate, test


def load_dataset(strip=True):
    dataset_dir = "training_datasets"
    x_name = os.path.join(dataset_dir, f"features_{DATASET}.pickle")
    y_name = os.path.join(dataset_dir, f"labels_{DATASET}.pickle")
    if strip:
        x_all = strip_dates(pd.read_pickle(x_name))
    else:
        x_all = pd.read_pickle(x_name)
    x_all = one_hot(x_all[TO_USE if strip else TO_USE + ["arrival_time_bin", "date"]])
    y_all = pd.read_pickle(y_name)[TO_PREDICT]
    return x_all, y_all


def load_xy_time(stamp):
    x_all, y_all = load_dataset(strip=False)
    y_all["arrival_time_bin"] = x_all["arrival_time_bin"]
    x_train, x_test = train_test_split_time(x_all, stamp)
    y_train, y_test = train_test_split_time(y_all, stamp)
    y_train.drop("arrival_time_bin", axis=1, inplace=True)
    y_test.drop("arrival_time_bin", axis=1, inplace=True)
    return strip_dates(x_train), y_train, strip_dates(x_test), y_test


def load_xy_cyclical():
    x_all, y_all = load_dataset(strip=True)
    x_train, x_test = train_test_split_cyclical(x_all)
    y_train, y_test = train_test_split_cyclical(y_all)
    return x_train, y_train, x_test, y_test


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

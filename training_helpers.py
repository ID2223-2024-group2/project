import pandas as pd
import numpy as np

LATEST_FV = 4
LATEST_TD = 15
NUM_FEATURES = 13
TO_PREDICT = ["mean_arrival_delay_seconds", "max_arrival_delay_seconds", "mean_departure_delay_seconds", "max_departure_delay_seconds"]


_split_cycle = [
    "train", "train", "validate", "test",
    "train", "train", "train", "test",
    "train", "train", "train", "validate",
    "train", "train", "train", "test",
    "train", "train", "train", "validate"
]


def strip_dates(df):
    return df.drop(["date", "arrival_time_bin"], axis=1)


# Splits into train, test and validate sets.
def train_validate_test_split(df):
    total_length = len(df)
    assignments = pd.Series([_split_cycle[i % len(_split_cycle)] for i in range(total_length)])
    train = df.loc[assignments == "train"]
    validate = df.loc[assignments == "validate"]
    test = df.loc[assignments == "test"]
    return train, validate, test


if __name__ == "__main__":
    columns = ["foo", "bar"]
    to_test = pd.DataFrame(np.random.rand(50, len(columns)), columns=columns)
    print(train_validate_test_split(to_test))

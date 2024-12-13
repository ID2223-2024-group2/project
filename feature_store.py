import os
import pandas as pd

loc_weather = os.path.join("dev_data", "openmeteo_data", "weather.db.pickle")


def load_weather():
    return pd.read_pickle(loc_weather)


def upsert_weather(other_df):
    if not os.path.exists(loc_weather):
        new_df = other_df.copy()
        new_df.set_index("date")
    else:
        current_df = pd.read_pickle(loc_weather)
        new_df = current_df.combine_first(other_df)
    new_df.to_pickle(loc_weather)



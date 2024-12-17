import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import koda.koda_transform as kt
import koda.koda_fetch as kf
import koda.koda_parse as kp
from koda.koda_constants import FeedType, OperatorsWithRT, StaticDataTypes

FEATHER_DF_VERSION = 2
FEATHER_COMPLETE_FILE = ".feathers_complete"

USE_PROCESSES = os.environ.get("USE_PROCESSES")
if USE_PROCESSES is None:
    USE_PROCESSES = os.cpu_count() - 2
else:
    USE_PROCESSES = int(USE_PROCESSES)


def get_feather_version(operator: OperatorsWithRT, date: str) -> int:
    rt_folder_path = kp.get_rt_dir_path(operator.value, date)
    feather_path = f"{rt_folder_path}/{FEATHER_COMPLETE_FILE}"
    if not os.path.exists(feather_path):
        return -1
    with open(feather_path, "r") as f:
        content = f.read()
        if content.isdigit():
            return int(content)
        elif content == "get_trip_updates_for_day completed":
            return 1
        else:
            return -1


def get_rt_data(operator: OperatorsWithRT, date: str) -> str:
    rt_archive_path = kf.fetch_gtfs_realtime_archive(operator, FeedType.TRIP_UPDATES, date)
    if rt_archive_path is None:
        raise ValueError(f"Failed to fetch realtime data for {operator.value} on {date}")
    rt_unzipped_path = kp.unzip_gtfs_archive(rt_archive_path, remove_archive_after=True)
    print(f"Unzipped realtime data to {rt_unzipped_path}")
    return rt_unzipped_path

def get_static_data(date: str, operator: OperatorsWithRT) -> str:
    static_archive_path = kf.fetch_gtfs_static_archive(operator, date)
    if static_archive_path is None:
        raise ValueError(f"Failed to fetch static data for {operator.value} on {date}")
    static_unzipped_path = kp.unzip_gtfs_archive(static_archive_path, remove_archive_after=True)
    print(f"Unzipped static data to {static_unzipped_path}")
    return static_unzipped_path


def get_koda_data_for_day(date: str, operator: OperatorsWithRT) -> (pd.DataFrame, pd.DataFrame):
    rt_folder_path = kp.get_rt_dir_path(operator.value, date)
    static_folder_path = kp.get_static_dir_path(operator.value, date)
    day_feather_path = f"{rt_folder_path}/{date}.feather"
    map_df_feather_path = f"{rt_folder_path}/route_types_map.feather"

    if get_feather_version(operator, date) == 2:
        print(f"get_trip_updates_for_day already completed for {operator.value} on {date} with version 2")
        return pd.read_feather(day_feather_path), pd.read_feather(map_df_feather_path)
    elif get_feather_version(operator, date) == 1:
        # Version 1 is missing route types and did not download static data
        print(f"get_trip_updates_for_day already completed for {operator.value} on {date} with version 1")
        print("Updating to version 2")
        try:
            _ = get_static_data(date, operator)
        except ValueError as e:
            print(e)
            return pd.DataFrame(), pd.DataFrame()
    else:
        print(f"Getting trip updates and static data for {operator.value} on {date}")
        # Clean up existing faulty data if it exists
        if os.path.exists(rt_folder_path):
            print(f"Cleaning {rt_folder_path}")
            shutil.rmtree(rt_folder_path)
        if os.path.exists(static_folder_path):
            print(f"Cleaning {static_folder_path}")
            shutil.rmtree(static_folder_path)
        try:
            _ = get_rt_data(operator, date)
            _ = get_static_data(date, operator)
        except ValueError as e:
            print(e)
            return pd.DataFrame(), pd.DataFrame()

    with ProcessPoolExecutor(max_workers=USE_PROCESSES) as executor:
        df, read_feather_paths = kt.read_rt_day_to_df(operator, FeedType.TRIP_UPDATES, date, remove_folder_after=True, executor=executor)

    df.to_feather(day_feather_path, compression='zstd', compression_level=9)
    # Remove individual hour feather files
    for path in read_feather_paths:
        if os.path.exists(path):
            os.remove(path)

    trips_df = kp.read_static_data_to_dataframe(operator, StaticDataTypes.TRIPS, date)
    routes_df = kp.read_static_data_to_dataframe(operator, StaticDataTypes.ROUTES, date)
    map_df = kt.create_route_types_map_df(df, trips_df, routes_df)
    map_df.to_feather(map_df_feather_path, compression='zstd', compression_level=9)

    print(f"get_trip_updates_for_day completed for {operator.value} on {date} with version 2")

    # Remove static data folder
    static_folder_path = kp.get_static_dir_path(operator.value, date)
    if os.path.exists(static_folder_path):
        print(f"Removing {static_folder_path}")
        shutil.rmtree(static_folder_path)

    with open(f"{rt_folder_path}/{FEATHER_COMPLETE_FILE}", "w") as f:
        f.write(str(FEATHER_DF_VERSION))

    return df, map_df

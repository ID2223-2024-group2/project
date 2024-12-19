import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import koda.koda_transform as kt
import koda.koda_fetch as kf
import koda.koda_parse as kp
import shared.parse as sp
import shared.transform as st
from shared.constants import FeedType, OperatorsWithRT, StaticDataTypes

FEATHER_DF_VERSION = 3

USE_PROCESSES = os.environ.get("USE_PROCESSES")
if USE_PROCESSES is None:
    USE_PROCESSES = os.cpu_count() - 2
else:
    USE_PROCESSES = int(USE_PROCESSES)


def get_feather_version(operator: OperatorsWithRT, date: str) -> int:
    version_path = kt.get_feather_version_path(operator.value, date)
    if not os.path.exists(version_path):
        return -1
    with open(version_path, "r") as f:
        content = f.read()
        if content.isdigit():
            return int(content)
        elif content == "get_trip_updates_for_day completed":
            return 1
        else:
            return -1


def set_feather_version(operator: OperatorsWithRT, date: str, version: int):
    version_path = kt.get_feather_version_path(operator.value, date)
    with open(version_path, "w") as f:
        f.write(str(version))


def get_rt_data(operator: OperatorsWithRT, date: str) -> str:
    rt_archive_path = kf.fetch_gtfs_realtime_archive(operator, FeedType.TRIP_UPDATES, date)
    if rt_archive_path is None:
        raise ValueError(f"Failed to fetch realtime data for {operator.value} on {date}")
    rt_unzipped_path = sp.unzip_gtfs_archive(rt_archive_path, kp.DATA_DIR, remove_archive_after=True)
    print(f"Unzipped realtime data to {rt_unzipped_path}")
    return rt_unzipped_path

def get_static_data(date: str, operator: OperatorsWithRT) -> str:
    static_archive_path = kf.fetch_gtfs_static_archive(operator, date)
    if static_archive_path is None:
        raise ValueError(f"Failed to fetch static data for {operator.value} on {date}")
    static_unzipped_path = sp.unzip_gtfs_archive(static_archive_path, kp.DATA_DIR, remove_archive_after=True)
    print(f"Unzipped static data to {static_unzipped_path}")
    return static_unzipped_path


def get_koda_data_for_day(date: str, operator: OperatorsWithRT) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    rt_folder_path = kp.get_rt_dir_path(operator.value, date)
    day_feather_path = kt.get_day_feather_path(operator.value, date)
    route_types_map_df_feather_path = kt.get_route_types_map_df_feather_path(operator.value, date)
    stop_count_df_feather_path = kt.get_stop_count_df_feather_path(operator.value, date)

    static_folder_path = sp.get_static_dir_path(operator.value, date, kp.DATA_DIR)

    # NOTE: Trips, routes and stop times do not need to be kept around as they are only the basis for the features,
    # but we keep them to speed up future new feature calculations
    trips_df_feather_path = kt.get_trips_df_feather_path(operator.value, date)
    routes_df_feather_path = kt.get_routes_df_feather_path(operator.value,date)
    stop_times_df_feather_path = kt.get_stop_times_df_feather_path(operator.value, date)

    stop_location_map_feather_path = kt.get_stop_location_map_feather_path(operator.value, date)

    if get_feather_version(operator, date) < 2:
        print(f"Old feather version found for {operator.value} on {date}")
        # Clean up existing malformatted data if it exists
        if os.path.exists(rt_folder_path):
            print(f"Cleaning {rt_folder_path}")
            shutil.rmtree(rt_folder_path)
        if os.path.exists(static_folder_path):
            print(f"Cleaning {static_folder_path}")
            shutil.rmtree(static_folder_path)

    if os.path.exists(day_feather_path):
        print(f"Reading existing data for {date} {day_feather_path}")
        rt_df = pd.read_feather(day_feather_path)
    else:
        print(f"Fetching realtime data for {operator.value} on {date}")
        _ = get_rt_data(operator, date)
        with ProcessPoolExecutor(max_workers=USE_PROCESSES) as executor:
            rt_df, read_feather_paths = kt.read_rt_day_to_df(operator, FeedType.TRIP_UPDATES, date,
                                                          remove_folder_after=True, executor=executor)
        rt_df.to_feather(day_feather_path, compression='zstd', compression_level=9)
        # Remove individual hour feather files
        for path in read_feather_paths:
            if os.path.exists(path):
                os.remove(path)

    if os.path.exists(route_types_map_df_feather_path):
        print(f"Reading existing data for {date} {route_types_map_df_feather_path}")
        route_types_map_df = pd.read_feather(route_types_map_df_feather_path)
    else:
        print(f"Fetching static data for {operator.value} on {date}")
        get_static_data(date, operator)
        trips_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.TRIPS, date, kp.DATA_DIR)
        routes_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.ROUTES, date, kp.DATA_DIR)
        trips_df.to_feather(trips_df_feather_path, compression='zstd', compression_level=9)
        routes_df.to_feather(routes_df_feather_path, compression='zstd', compression_level=9)
        route_types_map_df = st.create_route_types_map_df(trips_df, routes_df)
        route_types_map_df.to_feather(route_types_map_df_feather_path, compression='zstd', compression_level=9)

    if os.path.exists(stop_count_df_feather_path):
        print(f"Reading existing data for {date} {stop_count_df_feather_path}")
        stop_count_df = pd.read_feather(stop_count_df_feather_path)
    else:
        print(f"Fetching static data for {operator.value} on {date}")
        get_static_data(date, operator)
        stop_times_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.STOP_TIMES, date, kp.DATA_DIR)
        stop_times_df.to_feather(stop_times_df_feather_path, compression='zstd', compression_level=9)
        stop_count_df = st.create_stop_count_df(date, stop_times_df, route_types_map_df)
        stop_count_df.to_feather(stop_count_df_feather_path, compression='zstd', compression_level=9)

    if os.path.exists(stop_location_map_feather_path):
        print(f"Reading existing data for {date} {stop_location_map_feather_path}")
        stop_location_map_df = pd.read_feather(stop_location_map_feather_path)
    else:
        print(f"Fetching static data for {operator.value} on {date} for stops")
        get_static_data(date, operator)
        stops_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.STOPS, date, kp.DATA_DIR)
        stop_location_map_df = st.create_stop_location_map_df(stops_df)
        stop_location_map_df.to_feather(stop_location_map_feather_path, compression='zstd', compression_level=9)


    set_feather_version(operator, date, FEATHER_DF_VERSION)

    if os.path.exists(static_folder_path):
        print(f"Removing {static_folder_path}")
        shutil.rmtree(static_folder_path)

    return rt_df, route_types_map_df, stop_count_df, stop_location_map_df

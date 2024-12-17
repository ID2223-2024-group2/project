import os
import shutil

import pandas as pd

from koda.koda_constants import OperatorsWithRT, FeedType, StaticDataTypes
import gtfs_regional.fetch as gf
import gtfs_regional.transform as gt
import gtfs_regional.parse as gpa
import koda.koda_parse as kpa
import koda.koda_transform as kt


def get_rt_data(operator: OperatorsWithRT, date: str) -> pd.DataFrame:
    pb_path = gf.fetch_gtfs_realtime_pb(operator, FeedType.TRIP_UPDATES, date)
    if pb_path is None:
        raise ValueError(f"Failed to fetch realtime data for {operator.value} on {date}")

    raw_rt_df = kpa.read_pb_to_dataframe(pb_path)
    rt_df = gt.parse_live_pb(operator, date, raw_rt_df)
    return rt_df


def get_static_data(date: str, operator: OperatorsWithRT) -> str:
    static_archive_path = gf.fetch_gtfs_static_archive(operator, date)
    if static_archive_path is None:
        raise ValueError(f"Failed to fetch static data for {operator.value} on {date}")
    static_unzipped_path = kpa.unzip_gtfs_archive(static_archive_path, data_dir=gpa.DATA_DIR, remove_archive_after=True)
    print(f"Unzipped static data to {static_unzipped_path}")
    return static_unzipped_path


def get_gtfr_data_for_day(date: str, operator: OperatorsWithRT) -> (pd.DataFrame, pd.DataFrame):
    rt_feather_path = gt.get_rt_feather_path(operator.value, date)
    map_df_feather_path = f"{gpa.DATA_DIR}/route_types_map.feather"
    static_folder_path = gpa.get_static_dir_path(operator.value, date)

    if os.path.exists(rt_feather_path):
        print(f"Reading {rt_feather_path}")
        rt_df = pd.read_feather(rt_feather_path)
    else:
        print(f"Fetching realtime data for {operator.value} on {date}")
        rt_df = get_rt_data(operator, date)

    if os.path.exists(map_df_feather_path):
        print(f"Reading {map_df_feather_path}")
        map_df = pd.read_feather(map_df_feather_path)
        return rt_df, map_df

    print(f"Fetching static data for {operator.value} on {date}")
    get_static_data(date, operator)
    trips_df = kpa.read_static_data_to_dataframe(operator, StaticDataTypes.TRIPS, date, data_dir=gpa.DATA_DIR)
    routes_df = kpa.read_static_data_to_dataframe(operator, StaticDataTypes.ROUTES, date, data_dir=gpa.DATA_DIR)
    map_df = kt.create_route_types_map_df(rt_df, trips_df, routes_df)
    map_df.to_feather(map_df_feather_path, compression='zstd', compression_level=9)

    if os.path.exists(static_folder_path):
        print(f"Removing {static_folder_path}")
        shutil.rmtree(static_folder_path)

    return rt_df, map_df

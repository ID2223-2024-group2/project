import os
import shutil

import pandas as pd

import gtfs_regional.fetch as gf
import gtfs_regional.parse as gpa
import gtfs_regional.transform as gt
import shared.parse as sp
import shared.transform as st
from shared.constants import FeedType, OperatorsWithRT, StaticDataTypes


def get_rt_data(operator: OperatorsWithRT, date: str, force=False) -> pd.DataFrame:
    pb_path = gf.fetch_gtfs_realtime_pb(operator, FeedType.TRIP_UPDATES, date, force=force)
    if pb_path is None:
        raise ValueError(f"Failed to fetch realtime data for {operator.value} on {date}")

    raw_rt_df = sp.read_pb_to_dataframe(pb_path)
    rt_df = gt.parse_live_pb(operator, raw_rt_df, force=force)
    return rt_df


def get_static_data(date: str, operator: OperatorsWithRT, remove_archive_after=True) -> str:
    static_archive_path = gf.fetch_gtfs_static_archive(operator, date)
    if static_archive_path is None:
        raise ValueError(f"Failed to fetch static data for {operator.value} on {date}")
    static_unzipped_path = sp.unzip_gtfs_archive(static_archive_path, gpa.DATA_DIR,
                                                 remove_archive_after=remove_archive_after)
    print(f"Unzipped static data to {static_unzipped_path}")
    return static_unzipped_path


def get_gtfr_data_for_day(date: str, operator: OperatorsWithRT, force_rt=False) -> (
pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    last_updated = gt.read_last_updated(operator)
    print(f"Getting data for {date} {operator.value} last updated: {last_updated} (force_rt={force_rt})")

    rt_feather_path = gt.get_rt_feather_path(operator.value)
    route_types_map_df_feather_path = gt.get_route_types_map_df_feather_path(operator.value)
    stop_count_df_feather_path = gt.get_stop_count_df_feather_path(operator.value)

    static_folder_path = gpa.get_static_dir_path(operator.value, date)

    # NOTE: Trips, routes and stop times do not need to be kept around as they are only the basis for the features,
    # but we keep them to speed up future new feature calculations
    trips_df_feather_path = gt.get_trips_df_feather_path(operator.value)
    routes_df_feather_path = gt.get_routes_df_feather_path(operator.value)
    stop_times_df_feather_path = gt.get_stop_times_df_feather_path(operator.value)

    stop_location_map_feather_path = gt.get_stop_location_map_feather_path(operator.value)

    if os.path.exists(rt_feather_path) and last_updated == date and not force_rt:
        print(f"Reading existing data for {date} {rt_feather_path}")
        rt_df = pd.read_feather(rt_feather_path)
    else:
        print(f"Fetching realtime data for {operator.value} on {date}")
        rt_df = get_rt_data(operator, date, force=force_rt)

    if os.path.exists(route_types_map_df_feather_path) and last_updated == date:
        print(f"Reading existing data for {date} {route_types_map_df_feather_path}")
        route_types_map_df = pd.read_feather(route_types_map_df_feather_path)
    else:
        # NOTE: We only get 50 API hits per month, so we're keeping the archive for now
        # TODO: Remove archive in production or else it will accumulate every day
        print(f"Fetching static data for {operator.value} on {date} for route types map")
        get_static_data(date, operator, remove_archive_after=False)
        trips_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.TRIPS, date, gpa.DATA_DIR)
        routes_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.ROUTES, date, gpa.DATA_DIR)
        trips_df.to_feather(trips_df_feather_path, compression='zstd', compression_level=9)
        routes_df.to_feather(routes_df_feather_path, compression='zstd', compression_level=9)
        route_types_map_df = st.create_route_types_map_df(trips_df, routes_df)
        route_types_map_df.to_feather(route_types_map_df_feather_path, compression='zstd', compression_level=9)

    if os.path.exists(stop_count_df_feather_path) and last_updated == date:
        print(f"Reading existing data for {date} {stop_count_df_feather_path}")
        stop_count_df = pd.read_feather(stop_count_df_feather_path)
    else:
        # NOTE: We only get 50 API hits per month, so we're keeping the archive for now
        # TODO: Remove archive in production or else it will accumulate every day
        print(f"Fetching static data for {operator.value} on {date}  for stop count")
        get_static_data(date, operator, remove_archive_after=False)
        stop_times_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.STOP_TIMES, date, gpa.DATA_DIR)
        stop_times_df.to_feather(stop_times_df_feather_path, compression='zstd', compression_level=9)
        stop_count_df = st.create_stop_count_df(date, stop_times_df, route_types_map_df)
        stop_count_df.to_feather(stop_count_df_feather_path, compression='zstd', compression_level=9)

    if os.path.exists(stop_location_map_feather_path) and last_updated == date:
        print(f"Reading existing data for {date} {stop_location_map_feather_path}")
        stop_location_map_df = pd.read_feather(stop_location_map_feather_path)
    else:
        # NOTE: We only get 50 API hits per month, so we're keeping the archive for now
        # TODO: Remove archive in production or else it will accumulate every day
        print(f"Fetching static data for {operator.value} on {date} for stops")
        get_static_data(date, operator, remove_archive_after=False)
        stops_df = sp.read_static_data_to_dataframe(operator, StaticDataTypes.STOPS, date, gpa.DATA_DIR)
        stop_location_map_df = st.create_stop_location_map_df(stops_df)
        stop_location_map_df.to_feather(stop_location_map_feather_path, compression='zstd', compression_level=9)

    gt.write_last_updated(operator, date)

    if os.path.exists(static_folder_path):
        print(f"Removing {static_folder_path}")
        shutil.rmtree(static_folder_path)

    return rt_df, route_types_map_df, stop_count_df, stop_location_map_df

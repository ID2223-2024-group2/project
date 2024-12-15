import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import koda.koda_transform as kt
import koda.koda_fetch as kf
import koda.koda_parse as kp
from koda.koda_constants import FeedType, OperatorsWithRT


USE_PROCESSES = os.environ.get("USE_PROCESSES")
if USE_PROCESSES is None:
    USE_PROCESSES = os.cpu_count() - 2
else:
    USE_PROCESSES = int(USE_PROCESSES)

def get_trip_updates_for_day(date: str, operator: OperatorsWithRT) -> pd.DataFrame:
    rt_folder_path = kp.get_rt_dir_path(operator.value, date)
    if os.path.exists(f"{rt_folder_path}/.feathers_complete"):
        print(f"get_trip_updates_for_day already completed for {operator.value} on {date}")
        return kt.read_rt_day_to_df(operator, FeedType.TRIP_UPDATES, date, remove_folder_after=True)

    archive_path = kf.fetch_gtfs_realtime_archive(operator, FeedType.TRIP_UPDATES, date)
    if archive_path is None:
        print(f"Failed to fetch realtime data for {operator.value} on {date}")
        return pd.DataFrame()
    folder_path = kp.unzip_gtfs_archive(archive_path, remove_archive_after=True)
    print(f"Unzipped realtime data to {folder_path}")

    with ProcessPoolExecutor(max_workers=USE_PROCESSES) as executor:
        df = kt.read_rt_day_to_df(operator, FeedType.TRIP_UPDATES, date, remove_folder_after=True, executor=executor)

    with open(f"{rt_folder_path}/.feathers_complete", "w") as f:
        f.write("get_trip_updates_for_day completed")

    return df
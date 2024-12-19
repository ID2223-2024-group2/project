import pandas as pd

from shared.constants import route_types


def create_stop_location_map_df(stops_df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
    stops_df = stops_df[keep_columns]
    # Make sure stop_id is str
    stops_df['stop_id'] = stops_df['stop_id'].astype(str)
    return stops_df


def create_stop_count_df(date: str, stop_times_df: pd.DataFrame, route_types_map_df: pd.DataFrame) -> pd.DataFrame:
    if stop_times_df.empty or route_types_map_df.empty:
        raise ValueError("One or more DataFrames are empty")

    # Drop rows with missing arrival_time
    stop_times_df = stop_times_df.dropna(subset=['arrival_time'])

    # Ensure trip_ids are integers for both DataFrames and merge them
    stop_times_df['trip_id'] = stop_times_df['trip_id'].astype(str)
    route_types_map_df['trip_id'] = route_types_map_df['trip_id'].astype(str)
    stop_times_df = stop_times_df.merge(route_types_map_df, on='trip_id', how='inner')

    # Remove arrival_times > 24:00:00 (valid in GTFS but not in pandas and not useful for our purposes)
    stop_times_df = stop_times_df[stop_times_df['arrival_time'] < '24:00:00']

    # Set up arrival_time as our index and main datetime column
    stop_times_df['arrival_time'] = pd.to_datetime(date + ' ' + stop_times_df['arrival_time'])
    stop_times_df.sort_values(by='arrival_time', inplace=True)
    stop_times_df.set_index('arrival_time', inplace=True)

    # Group by route_type and resample to get stop count for each hour
    stop_count_df = stop_times_df.groupby('route_type').resample('h').size().reset_index()
    stop_count_df.sort_values(by=['route_type', 'arrival_time'], inplace=True)
    stop_count_df.columns = ['route_type', 'arrival_time', 'stop_count']
    return stop_count_df


def create_route_types_map_df(trips_df: pd.DataFrame, routes_df: pd.DataFrame) -> pd.DataFrame:
    if trips_df.empty or routes_df.empty:
        raise ValueError("One or more DataFrames are empty")
    # Make sure trip_id is a string for both DataFrames
    trips_df['trip_id'] = trips_df['trip_id'].astype(str)

    # Drop unnecessary columns
    map_df = trips_df.drop(columns=['service_id', 'trip_headsign', 'direction_id',
                                      'shape_id'])
    routes_df = routes_df.drop(columns=['agency_id', 'route_short_name', 'route_long_name', 'route_desc'])

    # Drop route_type from routes_df if it exists
    if 'route_type' in map_df.keys():
        map_df.drop(columns=['route_type'], errors='ignore', inplace=True)
    map_df = map_df.merge(routes_df, on='route_id', how='inner')
    # Remove duplicate trip_ids
    map_df = map_df.drop_duplicates(subset=['trip_id'])
    # Map route_type to strings with route_types dict
    map_df['route_type_description'] = map_df['route_type'].map(route_types)

    if len(map_df) < 10:
        print(f"Warning: map_df only has {len(map_df)} rows, likely missing data")

    return map_df

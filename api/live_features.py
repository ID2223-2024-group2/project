import pandas as pd

from shared.constants import GAEVLE_LONGITUDE, GAEVLE_LATITUDE, OperatorsWithRT
import weather.fetch as wf
import weather.parse as wp
import gtfs_regional.pipeline as gp
import shared.features as sf
import shared.transform as st

OPERATOR = OperatorsWithRT.X_TRAFIK
MIN_TRIP_UPDATES_PER_TIMESLOT = 5


def _get_live_weather_data(today: str) -> pd.DataFrame:
    print(f"Fetching weather data for {today}")
    weather_response = wf.fetch_forecast_weather(GAEVLE_LONGITUDE, GAEVLE_LATITUDE)
    weather_df = wp.parse_weather_response(weather_response)
    weather_df['hour'] = weather_df['date'].dt.hour

    return weather_df


def _get_live_delays_data(today: str) -> pd.DataFrame:
    rt_df, route_types_map_df, stop_count_df, stop_location_map_df = gp.get_gtfr_data_for_day(today, OPERATOR, force_rt=True)

    if rt_df.empty:
        print(f"No data available for {today}. get_live_delays_data exiting.")
        return pd.DataFrame()

    if route_types_map_df.empty:
        print(f"No routy type data available for {today}. get_live_delays_data exiting.")
        return pd.DataFrame()

    if stop_count_df.empty:
        print(f"No stop count data available for {today}. get_live_delays_data exiting.")
        return pd.DataFrame()

    if stop_location_map_df.empty:
        print(f"No stop location data available for {today}. get_live_delays_data exiting.")
        return pd.DataFrame()

    final_metrics, trip_update_count_df = sf.build_feature_group(rt_df, route_types_map_df, stop_count_df=stop_count_df)
    final_metrics = st.drop_rows_with_not_enough_updates(final_metrics, trip_update_count_df, MIN_TRIP_UPDATES_PER_TIMESLOT)

    return final_metrics


def get_live_features(today: str) -> (pd.DataFrame, pd.Series):
    weather_df = _get_live_weather_data(today)
    delays_df = _get_live_delays_data(today)

    # Localize the timestamps before converting to naive
    delays_df['arrival_time_bin'] = delays_df['arrival_time_bin'].dt.tz_localize('UTC').dt.tz_convert(None)
    weather_df['date'] = weather_df['date'].dt.tz_convert(None)

    feature_view = delays_df.merge(
        weather_df,
        left_on='arrival_time_bin',
        right_on='date',
        how='inner'
    )
    feature_view = feature_view[feature_view['stop_count'] > 0]

    # Keep copy of arrival_time_bin for later
    date_col = feature_view['arrival_time_bin']

    x = feature_view
    x['hour'] = x['arrival_time_bin'].dt.hour
    x = x.drop(['arrival_time_bin'], axis=1)

    expected_columns = ['stop_count', 'temperature_2m', 'apparent_temperature', 'precipitation', 'rain',
                        'snowfall', 'snow_depth', 'cloud_cover', 'wind_speed_10m', 'wind_speed_100m',
                        'wind_gusts_10m', 'hour']
    x = x[expected_columns]

    return x, date_col
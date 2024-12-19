import pandas as pd

from shared.constants import GAEVLE_LONGITUDE, GAEVLE_LATITUDE, OperatorsWithRT
import weather.fetch as wf
import weather.parse as wp
import gtfs_regional.pipeline as gp
import shared.features as sf

OPERATOR = OperatorsWithRT.X_TRAFIK

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

    final_metrics = sf.build_feature_group(rt_df, route_types_map_df, stop_count_df=stop_count_df)

    return final_metrics


def get_live_features(today: str) -> pd.DataFrame:
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
    return feature_view
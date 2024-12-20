import pandas as pd

import weather.fetch as wf
import weather.parse as wp
from shared.constants import GAEVLE_LONGITUDE, GAEVLE_LATITUDE


def get_recent_weather(start_date: str, end_date: str) -> pd.DataFrame:
    response = wf.fetch_recent_weather(GAEVLE_LONGITUDE, GAEVLE_LATITUDE, start_date, end_date)
    df = wp.parse_weather_response(response)
    df['hour'] = df['date'].dt.hour
    return df


def get_forecast_weather() -> pd.DataFrame:
    response = wf.fetch_forecast_weather(GAEVLE_LONGITUDE, GAEVLE_LATITUDE)
    df = wp.parse_weather_response(response)
    df['hour'] = df['date'].dt.hour
    return df


def get_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    response = wf.fetch_weather_archive(GAEVLE_LONGITUDE, GAEVLE_LATITUDE, start_date, end_date)
    df = wp.parse_weather_response(response)
    df['hour'] = df['date'].dt.hour
    return df
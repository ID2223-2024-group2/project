import openmeteo_requests
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse

om_client = openmeteo_requests.Client()

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather_archive(longitude, latitude, start_date, end_date, url=ARCHIVE_URL) -> WeatherApiResponse:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth",
                   "cloud_cover", "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m"]
    }
    responses = om_client.weather_api(url, params=params)
    # NOTE: There are only ever more than one response if we send multiple latitude/longitude pairs
    return responses[0]


def fetch_forecast_weather(longitude, latitude, url=FORECAST_URL) -> WeatherApiResponse:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall", "snow_depth",
                   "cloud_cover", "wind_speed_10m", "wind_speed_100m", "wind_gusts_10m"]
    }
    responses = om_client.weather_api(url, params=params)
    return responses[0]
import pandas as pd
from openmeteo_sdk import WeatherApiResponse


def parse_weather_response(response: WeatherApiResponse) -> pd.DataFrame:
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_apparent_temperature = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_rain = hourly.Variables(3).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
    hourly_snow_depth = hourly.Variables(5).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(8).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(9).ValuesAsNumpy()

    hourly_data = {'date': pd.date_range(start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True), freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"), 'temperature_2m': hourly_temperature_2m, 'apparent_temperature': hourly_apparent_temperature,
        'precipitation': hourly_precipitation, 'rain': hourly_rain, 'snowfall': hourly_snowfall,
        'snow_depth': hourly_snow_depth, 'cloud_cover': hourly_cloud_cover, 'wind_speed_10m': hourly_wind_speed_10m,
        'wind_speed_100m': hourly_wind_speed_100m, 'wind_gusts_10m': hourly_wind_gusts_10m}

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe.dropna(inplace=True)
    return hourly_dataframe

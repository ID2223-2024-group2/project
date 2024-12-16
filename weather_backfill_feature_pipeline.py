import os
import sys

import hopsworks
import pandas as pd

from shared.file_logger import setup_logger
import weather.fetch as wf
import weather.parse as wp

SAVE_TO_HW = True

# Coordinates for Gävle
longitude = 60.6749
latitude = 17.1413

log_file_path = os.path.join(os.path.dirname(__file__), 'weather_backfill.log')
logger = setup_logger('weather_backfill', log_file_path)

if __name__ == "__main__":
    START_DATE = os.environ.get("START_DATE", "2023-01-01")
    END_DATE = os.environ.get("END_DATE", "2024-12-15")

    try:
        dates = pd.date_range(START_DATE, END_DATE)
    except (ValueError, TypeError):
        logger.error("Invalid date range. Exiting.")
        sys.exit(1)
    total_dates = len(dates)

    logger.info("Starting backfill process for dates: %s - %s (%s days)", START_DATE, END_DATE, total_dates)
    response = wf.fetch_weather_archive(longitude, latitude, START_DATE, END_DATE)
    df = wp.parse_weather_response(response)
    # Add hour as a separate column
    df['hour'] = df['date'].dt.hour

    # Column list: [ apparent_temperature, cloud_cover, date, precipitation, rain, snow_depth, snowfall, temperature_2m, wind_gusts_10m, wind_speed_100m, wind_speed_10m]

    logger.info("Parsed %s rows of weather data", len(df))
    if not SAVE_TO_HW:
        df.to_csv("weather_backfill.csv", index=False)
        sys.exit(0)

    if os.environ.get("HOPSWORKS_API_KEY") is None:
        os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()

    project = hopsworks.login()
    fs = project.get_feature_store()

    weather_fg = fs.get_or_create_feature_group(
        name='weather',
        description='Hourly weather data for Gävle',
        version=2,
        primary_key=['date'],
        event_time="date",
    )
    weather_fg.insert(df)
    weather_fg.update_feature_description("apparent_temperature", "Apparent temperature in Celsius")
    weather_fg.update_feature_description("cloud_cover", "Cloud cover in percentage")
    weather_fg.update_feature_description("date", "Timestamp of the weather data")
    weather_fg.update_feature_description("precipitation", "Precipitation in mm")
    weather_fg.update_feature_description("rain", "Rainfall in mm")
    weather_fg.update_feature_description("snow_depth", "Snow depth in m")
    weather_fg.update_feature_description("snowfall", "Snowfall in cm")
    weather_fg.update_feature_description("temperature_2m", "Temperature at 2m in Celsius")
    weather_fg.update_feature_description("wind_gusts_10m", "Wind gusts at 10m in km/h")
    weather_fg.update_feature_description("wind_speed_100m", "Wind speed at 100m in km/h")
    weather_fg.update_feature_description("wind_speed_10m", "Wind speed at 10m in km/h")
    weather_fg.update_feature_description("hour", "Hour of the day")
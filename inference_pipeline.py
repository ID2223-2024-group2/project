import os
import hopsworks
import pandas as pd
from xgboost import XGBRegressor

import weather.fetch as wf
import weather.parse as wp
from shared.constants import GAEVLE_LONGITUDE, GAEVLE_LATITUDE

if os.environ.get("HOPSWORKS_API_KEY") is None:
    os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()


if __name__ == "__main__":
    project = hopsworks.login()
    fs = project.get_feature_store(name='tsedmid2223_featurestore')
    print("Connected to Hopsworks Feature Store. ")

    mr = project.get_model_registry()
    print("Connected to Model Registry. ")

    retrieved_model = mr.get_model(
        name="delays_xgboost_model",
        version=9,
    )
    saved_model_dir = retrieved_model.download()
    print(f"Downloaded model to {saved_model_dir}")

    retrieved_xgboost_model = XGBRegressor()
    retrieved_xgboost_model.load_model(saved_model_dir + "/model.json")
    print(f"Retrieved XGBoost model from Model Registry. {retrieved_xgboost_model}")

    weather_response = wf.fetch_forecast_weather(GAEVLE_LONGITUDE, GAEVLE_LATITUDE)
    weather_df = wp.parse_weather_response(weather_response)

    train_features = weather_df
    train_features['hour'] = weather_df['date'].dt.hour
    train_features = weather_df.drop(['date'], axis=1)
    # TODO: Just for testing, need to drop this from X_train or get it from current gtfs data in the future
    # Fill stop_count column with 0
    train_features['stop_count'] = 50

    # TODO: We only need to do this manually because we're not using feature views yet
    # Ensure the columns are in the correct order
    expected_columns = ['stop_count', 'temperature_2m', 'apparent_temperature', 'precipitation', 'rain', 'snowfall', 'snow_depth', 'cloud_cover', 'wind_speed_10m', 'wind_speed_100m', 'wind_gusts_10m', 'hour']
    train_features = train_features[expected_columns]

    predictions = retrieved_xgboost_model.predict(train_features)

    # Define the output schema
    output_schema = [
        {"name": "mean_arrival_delay_seconds", "type": "float64"},
        {"name": "max_arrival_delay_seconds", "type": "float64"},
        {"name": "mean_departure_delay_seconds", "type": "float64"},
        {"name": "max_departure_delay_seconds", "type": "float64"},
        {"name": "on_time_mean_percent", "type": "float64"}
    ]

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions, columns=[col["name"] for col in output_schema])
    # Add the date column
    predictions_df['date'] = weather_df['date']

    predictions_df.to_csv("predictions.csv", index=False)


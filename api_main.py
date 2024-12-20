import os
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from xgboost import XGBRegressor

import api.live_features as sl

pd.options.mode.copy_on_write = True


delay_models = {}


def load_delays_model_from_hw() -> XGBRegressor:
    import hopsworks
    if os.environ.get("HOPSWORKS_API_KEY") is None:
        os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()

    project = hopsworks.login()
    mr = project.get_model_registry()
    retrieved_model = mr.get_model(
        name="delays_xgboost_model",
        version=9,
    )
    saved_model_dir = retrieved_model.download()
    retrieved_xgboost_model = XGBRegressor()
    retrieved_xgboost_model.load_model(saved_model_dir + "/model.json")
    return retrieved_xgboost_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models
    delay_models['delays'] = load_delays_model_from_hw()
    yield
    # Clean up the models and release the resources
    delay_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return "Root for weather transit delays API"


@app.get("/predict")
def get_current_predictions():
    today = datetime.now().strftime("%Y-%m-%d")
    x_df, date_col = sl.get_live_features(today)

    model = delay_models['delays']
    predictions = model.predict(x_df)
    # Convert predictions to DataFrame
    output_schema = [
        {"name": "mean_arrival_delay_seconds", "type": "float64"},
        {"name": "max_arrival_delay_seconds", "type": "float64"},
        {"name": "mean_departure_delay_seconds", "type": "float64"},
        {"name": "max_departure_delay_seconds", "type": "float64"},
        {"name": "on_time_mean_percent", "type": "float64"}
    ]

    predictions_df = pd.DataFrame(predictions, columns=[col["name"] for col in output_schema])
    predictions_df['date'] = date_col

    predictions_json = predictions_df.to_json(orient='records')

    return predictions_json

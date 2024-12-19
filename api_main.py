from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
import api.live_features as sl


delay_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models
    delay_models['delays'] = lambda x: x # TODO: Load actual model
    yield
    # Clean up the models and release the resources
    delay_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return "Root for weather transit delays API"


@app.get("/predict")
def get_current_predictions():
    print("Getting live features")
    today = datetime.now().strftime("%Y-%m-%d")
    feature_view_df = sl.get_live_features(today)
    print(f"Fetched {feature_view_df.shape[0]} rows")

    model = delay_models['delays']
    predictions = model(feature_view_df)
    print(f"Predicted {predictions.shape[0]} rows")

    predictions_json = predictions.to_json(orient='records')
    print(f"Returning {len(predictions_json)} bytes")

    return predictions_json

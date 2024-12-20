# Scalable Machine Learning and Deep Learning Project
**Group 2**: Paul Hübner, Jonas Müller

**Goal:** Predict weather-based public transport delays

## Setup

> **Dependencies** are currently just managed with venv and there are already some conflicts, but working configurations are possible.
`hopsworks` and the old `protobuf` version needed for KoDa appear to be the main causes of issues.
Additionally, XGBoost seems to not like `scikit-learn > 1.5.2`.
`requirements_jonas.txt` should contain a working configuration for Ubuntu 24/Win 11 with python 3.12.3.

## Data Sources

### Public Transport Delays
**Historical Backfill**:
- Source: https://www.trafiklab.se/api/trafiklab-apis/koda/
- Earliest tested successful request for operator `xt`: `2022-02-01`
- Pipeline: `koda_backfill_feature_pipeline.py`
  - Env Vars:
    - `START_DATE`: Start date for backfilling
    - `END_DATE`: End date for backfilling
    - `STRIDE`: Stride for backfilling
    - `DRY_RUN`: If set to `True`, no data will be written to the feature store, only one day processed and written to a csv file
    - `KODA_KEY`: API key for KoDa
    - `USE_PROCESSES`: Number of processes to use for parallel processing
    - `HOPSWORKS_API_KEY`: API key for Hopsworks
    - `FG_VERSION`: Version of the delay feature group to use
    - `RUN_HW_MATERIALIZATION_EVERY`: How often to run Hopsworks materialization jobs in days processed
  - Example: `HOPSWORKS_API_KEY=your_key KODA_KEY=your_key USE_PROCESSES=4 START_DATE=2024-11-01 END_DATE=2024-11-01 STRIDE=4 DRY_RUN=False python3 koda_backfill_feature_pipeline.py`

**Daily Backfill**:
- Pipeline: `daily_feature_backfill_pipeline.py`
- Env Vars:
  - `DRY_RUN`: If set to `True`, no data will be written to the feature store, only output as csvs
  - `WEATHER_FG_VERSION`: Version of the weather feature group to use
  - `DELAYS_FG_VERSION`: Version of the delay feature group to use
  - `KODA_KEY`: API key for KoDa
  - `USE_PROCESSES`: Number of processes to use for parallel processing
  - `HOPSWORKS_API_KEY`: API key for Hopsworks

**Current data**:
- Source https://www.trafiklab.se/api/gtfs-datasets/gtfs-regional/
- Pipeline: `live_feature_pipeline.py`
  - Env Vars:
    - `DRY_RUN`: If set to `True`, no data will be written to the feature store, only one day processed and written to a csv file
    - `GTRFSR_RT_API_KEY`: API key for GTFS Regional Realtime
    - `GTRFSR_STATIC_API_KEY`: API key for GTFS Regional Static
    - `USE_PROCESSES`: Number of processes to use for parallel processing
    - `HOPSWORKS_API_KEY`: API key for Hopsworks
  - Example: `HOPSWORKS_API_KEY=your_key DRY_RUN=False python3 live_feature_pipeline.py`

General information on GTFS data: https://gtfs.org/documentation/overview/

### Weather Data
Source: https://open-meteo.com/en/docs

**Historical Backfill**:
- Pipeline:`weather_backfill_feature_pipeline.py`
- Env Vars:
  - `START_DATE`: Start date for backfilling
  - `END_DATE`: End date for backfilling
  - `HOPSWORKS_API_KEY`: API key for Hopsworks 

**Daily Backfill**:
- Pipeline: `daily_feature_backfill_pipeline.py` (same as for delays)

**Predictions**
- Pipeline: `live_feature_pipeline.py` (same as for delays)

## Features
Ideas:
- Build small set of delay features for routes to attempt predictions per route

## Training
Experimentation: `training_test.ipynb`

## Serving
`api_main.py` is the entry point for an example of how the model could be served with live data.

Prerequisite:
- `pip install fastapi[standard]`

Run it with:
- `fastapi run api_main.py` (for development).
- `uvicorn api_main:app --host 0.0.0.0 --port 8000` (for production).

## Deployment
- Daily backfill pipelines are scheduled with GitHub Actions: `daily-backfill.yml` using `daily_feature_backfill_pipeline.py`
- (Subject to change) Model serving and live data retrieval API is hosted on the [KTH Cloud](http://deploy.cloud.cbh.kth.se:20114/docs)
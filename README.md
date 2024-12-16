# Scalable Machine Learning and Deep Learning Project
**Group 2**: Paul Hübner, Jonas Müller

**Goal:** Predict weather-based public transport delays

## Setup

> **Dependencies** are currently just managed with venv and there are already some conflicts, but working configurations are possible.
`hopsworks` and the old `protobuf` version needed for KoDa appear to be the main causes of issues.
`requirements_jonas.txt` should contain a working configuration for Win 11 python 3.12.3.

## Data Sources

### Public Transport Delays
Historical data:
- https://www.trafiklab.se/api/trafiklab-apis/koda/
- Earliest tested successful request for operator `xt`: `2022-02-01`
- Backfill Pipeline: `koda_backfill_feature_pipeline.py`
  - Env Vars:
    - `DRY_RUN`: If set to `True`, no data will be written to the feature store, only one day processed and written to a csv file
    - `START_DATE`: Start date for backfilling
    - `END_DATE`: End date for backfilling
    - `STRIDE`: Stride for backfilling
    - `KODA_API_KEY`: API key for KoDa
    - `USE_PROCESSES`: Number of processes to use for parallel processing
    - `HOPSWORKS_API_KEY`: API key for Hopsworks

Current data:
- https://www.trafiklab.se/api/gtfs-datasets/gtfs-regional/
- Pipeline: TODO

General information on GTFS:
- https://gtfs.org/documentation/overview/

### Weather Data
- https://open-meteo.com/en/docs
- Backfill pipeline: `weather_backfill_feature_pipeline.py`
  - Env Vars:
    - `START_DATE`: Start date for backfilling
    - `END_DATE`: End date for backfilling
    - `HOPSWORKS_API_KEY`: API key for Hopsworks

## Training
Experimentation: `training_test.ipynb`
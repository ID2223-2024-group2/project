# Scalable Machine Learning and Deep Learning Project
**Group 2**: Paul Hübner, Jonas Müller

**Goal:** Predict weather-based public transport delays

## Data Sources

### Public Transport Delays
Historical data:
- https://www.trafiklab.se/api/trafiklab-apis/koda/
- Earliest tested successful request for operator `xt`: `2022-02-01`
- Backfill Pipeline: `koda_backfill_feature_pipeline.py`
  - Env Vars:
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
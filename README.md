# Scalable Machine Learning and Deep Learning Project
**Group 2**: Paul Hübner, Jonas Müller

**Goal:** Predict weather-based public transport delays

## Data Sources

### Public Transport Delays
Historical data:
- https://www.trafiklab.se/api/trafiklab-apis/koda/
- Earliest tested successful request for operator `xt`: `2022-02-01`
- Pipeline: `koda_backfill_feature_pipeline.py`
  - Env Vars:
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
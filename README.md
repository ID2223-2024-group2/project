# Scalable Machine Learning and Deep Learning Project

![The Dashboard](https://github.com/user-attachments/assets/d16bf90d-5a82-4a06-83bb-0ffee09ee697)
[**🖥️ Click me to see the dashboard ✨**](https://gaevle.streamlit.app)

**Group 2**: Paul Hübner, Jonas Müller

**Topic:** Predict weather-based public transport delays in Gävleborg.

## Quick Introduction

When snowfall catches a city off-guard --  as it did in Munich in 2023 -- there are often drastic consequences for the public. 
Public transportation is shut down, flights are canceled, and big events postponed [1]. 
Gävleborg country sees snow quite frequently, and therefore should in theory be prepared in terms of infrastructure and operations. 
However, delays or cancellations still occur in public transportation [2]. 
Therefore, the **goal of this project** is to investigate to what extent a machine learning model can quantify delays. 
The **purpose of the project** is to aid the citizens of Gävleborg with a free online tool that they can then use in their own planning.

## The Datasets

At the core of the project, there are two data sources.
The traffic data from [Trafikverket](https://www.trafiklab.se) provides real-time and historical public transport information.
The weather API from [Open-Meteo](https://open-meteo.com) is self-explanatory.
Overall, in the end, **there were ~12 000 datapoints available**.
Below is more information on the different datasets used from these data sources, as well as their related scripts.

### Public Transport Delays

There are three data sources used.
General information on the GTFS data can further be found at https://gtfs.org/documentation/overview/.

#### Historical Backfill
- Purpose: to provide a backfill of data that can be used for model training. 
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

#### Daily Backfill:
- Purpose: provides ground-truth data of the previous day.
- Pipeline: `daily_feature_backfill_pipeline.py`
- Env Vars:
  - `DRY_RUN`: If set to `True`, no data will be written to the feature store, only output as csvs
  - `WEATHER_FG_VERSION`: Version of the weather feature group to use
  - `DELAYS_FG_VERSION`: Version of the delay feature group to use
  - `KODA_KEY`: API key for KoDa
  - `USE_PROCESSES`: Number of processes to use for parallel processing
  - `HOPSWORKS_API_KEY`: API key for Hopsworks

#### Real-Time Data:
- Purpose: to provide real-time data that can be used to make real-time estimations.
In reality, this is used in batch inference, as opposed to truly real-time.
- Source https://www.trafiklab.se/api/gtfs-datasets/gtfs-regional/
- Pipeline: `live_feature_pipeline.py`
  - Env Vars:
    - `DRY_RUN`: If set to `True`, no data will be written to the feature store, only one day processed and written to a csv file
    - `GTRFSR_RT_API_KEY`: API key for GTFS Regional Realtime
    - `GTRFSR_STATIC_API_KEY`: API key for GTFS Regional Static
    - `USE_PROCESSES`: Number of processes to use for parallel processing
    - `HOPSWORKS_API_KEY`: API key for Hopsworks
  - Example: `HOPSWORKS_API_KEY=your_key DRY_RUN=False python3 live_feature_pipeline.py`

### Weather Data
More information can be found here: https://open-meteo.com/en/docs.

**Historical Backfill**:
- Purpose: to provide a backfill of data that can be used for model training.
- Pipeline:`weather_backfill_feature_pipeline.py`
- Env Vars:
  - `START_DATE`: Start date for backfilling
  - `END_DATE`: End date for backfilling
  - `HOPSWORKS_API_KEY`: API key for Hopsworks 

**Daily Backfill**:
- Purpose: provides ground-truth data of the previous day. 
- Pipeline: `daily_feature_backfill_pipeline.py` (same as for delays)

**Predictions**
- Purpose: Weather predictions for the future, that can be used to make delay predictions.
- Pipeline: `live_feature_pipeline.py` (same as for delays)

## Feature Engineering

Both data sources provide a plethora of information. 
There are two feature groups, delays (via from the traffic data) and weather.

The table below summarizes all the features. 
Which features to include in the final models was decided through trial and error early on.
It was found that, as expected, too many features caused quite extreme overfitting.
We ended up **predicting** two features: **average arrival delay** and **on time percentage**.

<details>
  <summary>All features (very long table)</summary>

| Group   | Name                                      | Type      | Label | Used in model |
|---------|-------------------------------------------|-----------|-------|---------------|
| Delays  | `route_type`                              | enum      |       | ✅             |
| Delays  | `arrival_time_bin`                        | timestamp |       | ✅             |
| Delays  | `mean_delay_change_seconds`               | double    | ✅     |               |
| Delays  | `max_delay_change_seconds`                | double    | ✅     |               |
| Delays  | `min_delay_change_seconds`                | double    | ✅     |               |
| Delays  | `var_delay_change_seconds`                | double    | ✅     |               |
| Delays  | `mean_arrival_delay_seconds`              | double    | ✅     |               |
| Delays  | `max_arrival_delay_seconds`               | double    | ✅     |               |
| Delays  | `min_arrival_delay_seconds`               | double    | ✅     |               |
| Delays  | `var_arrival_delay`                       | double    | ✅     |               |
| Delays  | `mean_departure_delay_seconds`            | double    | ✅     |               |
| Delays  | `max_departure_delay_seconds`             | double    | ✅     |               |
| Delays  | `min_departure_delay_seconds`             | double    | ✅     |               |
| Delays  | `var_departure_delay`                     | double    | ✅     |               |
| Delays  | `mean_on_time_percent`*                   | double    | ✅     |               |
| Delays  | `mean_final_stop_delay_seconds`           | double    | ✅     |               |
| Delays  | `mean_arrival_delay_seconds_lag_5stops`   | double    |       | ✅             |
| Delays  | `mean_departure_delay_seconds_lag_5stops` | double    |       |               |
| Delays  | `mean_delay_change_seconds_lag_5stops`    | double    |       |               |
| Delays  | `stop_count`**                            | double    |       | ✅             |
| Delays  | `trip_update_count`                       | integer   |       |               |
| Weather | `date`                                    | timestamp |       |               |
| Weather | `temperature_2m`                          | float     |       | ✅             |
| Weather | `apparent_temperature`                    | float     |       |               |
| Weather | `precipitation`                           | float     |       |               |
| Weather | `rain`                                    | float     |       |               |
| Weather | `snowfall`                                | float     |       | ✅             |
| Weather | `snow_depth`                              | float     |       | ✅             |
| Weather | `cloud_cover`                             | float     |       |               |
| Weather | `wind_speed_10m`                          | float     |       |               |
| Weather | `wind_speed_100m`                         | float     |       |               |
| Weather | `wind_gusts_10m`                          | float     |       | ✅             |
| Weather | `hour`                                    | int       |       | ✅             |

*A train is considered on-time if it arrives no earlier than 3 minutes and no later than 5 minutes.  
**An integer, just stored as a double for legacy reasons.
</details>

### Delay Features

The delay features were derived by ...
**TODO: Jonas write a high-level description**
**TODO: Jonas explain why we have our label in our features**

### Weather Features

**TODO: Jonas write a high-level description**

## Methodology

Our methodology is an iterative empirical one. We performed the following:
1. Download a part of the data (using strides, so every X days).
2. Experiment with some models, play around with hyperparameters, try to develop good models.
3. Perform a rigorous evaluation in the form of a grid search to get an understanding of how the models behave overall.
4. Return to step 1.

At the same time, we worked on refining the data that we had.
Additionally, we developed the UI as it was important to be able to see the data visually too.

## Models

We decided to develop two models, so that we could evaluate which one performed better in hyperparameter tuning.
A brief description of the model is given below.

### Decision Tree

### Artificial Neural Network

![Network Diagram](https://github.com/user-attachments/assets/28a63c24-dd02-46c2-88fb-36ddf25a241d)


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

## Running the Code

> ⚠️ We recommend *not to run the code yourself*, as there are many moving parts.
> The data cannot be shared through Git due to their large sizes.
> **Downloading will take DAYS and training will take hours and require ~200 GB disk space!**

Start off by forking the repository.
Ensure to have ~7000 CI minutes available to you, a lot of the tasks happen on CI.

### Dependencies

Dependencies are currently just managed with venv and there are already some conflicts, but working configurations are possible.
`hopsworks` and the old `protobuf` version needed for KoDa appear to be the main causes of issues.
Additionally, XGBoost seems to not like `scikit-learn > 1.5.2`.
`requirements_jonas.txt` should contain a working configuration for Ubuntu 24/Win 11 with Python 3.12.3.
`requirements.txt` should work for Ubuntu 20 Python 3.10.

### Instructions

Below are best-effort installation and setup instructions. 
These give a high-level overview of what needs to be done. 
It could be that for individual tasks there are breakages depending on the environment.

1. Set up a virtual environment and install the requirements.
See the above section for more details.
2. Set up a Hopsworks project and replace all occurrences of `tsedmid2223_featurestore` in the code with this project. 
In the UI, replace the GitHub repository with the correct one.
3. **TODO: Jonas talk about how to get the features online**
4. **TODO: Jonas talk about making the feature view**
5. Create a test data set WITHOUT any splits once the feature view has been created.
The data should all be ingested and materialized before this happens.
6. Change the feature view and training data numbers in `training_helpers.py`.
7. Download using `hopsworks_download.py`.
8. Change the dataset size parameter in `training_helpers.py`.
9. Run the hyperparameter tuning by running `trainer.py`.
Be aware that **this can take hours** and it wil **take up hundreds of GB** of disk space.
This will save both XGBoost and Keras models. Inference is only done on Keras.
10. Run the UI via `streamlit run ui.py`.
Or host it somewhere.

## References

[1] Reuters. “Munich flights, long-distance trains cancelled due to snow”. In: Reuters (Dec. 2023). url: https://www.reuters.com/world/europe/munich-flights-long-distance-trains-cancelled-due-snow-2023-12-02/ (visited on 12/18/2024).

[2] Becky Warterton. “Swedish weather agency upgrades weather warning as snowstorm set to con-
tinue”. In: thelocal.se (Nov. 2024). url: https://www.thelocal.se/20241121/power-cuts-and-cancelled-trains-in-second-day-of-swedish-snowstorm (visited on 12/18/2024).



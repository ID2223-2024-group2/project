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
**TODO: Jonas explain the weather is only for Gävle**

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

A decision tree using XGBoosts's `XGBRegressor`.
The XGBoost model is able to capture nonlinear relationships, and in our experience has performed well.
Therefore, it is worth investigating, especially since it is quite cheap to train.

### Artificial Neural Network

An artificial neural network using `keras` and `tensorflow`.
ANN can learn extremely complex relationships.
While we were unsure if there was enough data, we decided it was definitely worth investigating.
The final (hyperparameter tuned) ANN had one hidden layer of size 16.

<details>
  <summary>The ANN network diagram</summary>

  ![Network Diagram](https://github.com/user-attachments/assets/28a63c24-dd02-46c2-88fb-36ddf25a241d)
</details>

## Training

This section very crudely outlines how the models were trained.
Trained models were saved on Hopsworks in the model registry.

### Preprocessing

In order to start with training, the data had to be preprocessed.
The route type feature had to be one-hot encoded, but other than that the features were mostly already clean.

From all available data, a training dataset was created on Hopsworks.
One of the challenges was to determine how to split this set into training and test sets.
There were two options:
1. Dedicate every X datapoints to the test set. Make the cycle such that the test set sees various weekdays, seasons, etc.
2. Use every day after day X as a test day and train on the days before.

We ended up going with the second option.
While the first seemed to make more sense intuitively, it was not performing as well.
Similarly, once lagged data was added, this was kind of cheating, as some training days had the laggged value of test days.
Since we had almost 3 years worth of data, this would ensure that there was sufficient season data available in the training set.
The test set was everything after Midsommar (2024-06-22).

For the ANN, normalization of the inputs/outputs was required. 
For this, we used `sklearn`.
`StandardScaler` was found to outperform `MinMaxScaler`, `MaxAbsScaler` and (surprisingly) `RobustScaler`.
The scaler mean/variance was attached to the TensorFlow model as a variable and uploaded to Hopsworks.

### Training

The files `training_keras.py` and `training_xgboost.py` provided workbenches to quickly train and iterate over models.
One caveat is that training data must reside locally on the computer, to avoid the Hopsworks delays.

As aforementioned, the features were narrowed down using trial and error. 
For the ANN, 40 epochs of 32 batch sizes seemed to provide the best results.

### Hyperparameter Tuning

We performed a grid search on the models.
The training dataset was first split into a train and test set.
The test set was withheld entirely, and the training set used in a 5-fold cross validation set.
This then randomly produced training and validation features and labels.

The hyperparameters that were trialled were:
```python
xgboost_grid = {
    "learning_rate": [0.1, 0.2, 0.3, 0.4],
    "max_depth": [2, 4, 6, 8]
}

keras_grid = {
    "model__lr": [0.001, 0.005, 0.1],
    "model__hidden": [8, 10, 16, 32],
    "model__activation": ["relu", "sigmoid", "tanh"],
}
```

Hyperparameter tuning took at most a minute for the decision tree.
The rest of the many hours spent training were spent on tuning the ANN.

## Results and Evaluation

In order to evaluate our models numerically, we chose to use R². 
We find this to be a little more interpretable than MSE.

The best models on the final dataset were as follows:

| Model | Hyperparameters                                                | R² |
| --- |----------------------------------------------------------------| -- |
| Decision Tree | learning rate: 0.2, max depth: 4                               | 0.37 |
| ANN | learning rate: 0.005, relu-activated, 16 nodes in hidden layer | 0.60 |

The reason why XGBoost is so low is that the R² is calculated as the mean R² of the two features.
While on-time-percentage sees a decent R² of 0.58, the average delay R² of 0.16 drags it down significantly.
The ANN is able to learn average delay much better, hence the higher overall R².

The most important features as reported by XGBoost are as follows.
It is unsurprising that lagged delay is weighted so highly.
While not depicted here, we noticed that when we had many other features, 
the weather-based features that we picked contributed much more significantly than other weather/delay features. 

![Most important features](https://github.com/user-attachments/assets/4e8fda01-b73f-4725-9d4a-4122d1b09fec)

R² does not tell the whole picture. (What even is a good R²?)
Therefore, predictions are continuously (and were retroactively) generated to graph the performance.
A screenshot of the last 300 days can be found below:

![Predictive accuracy](https://github.com/user-attachments/assets/35df34d7-0953-4eaf-a74d-3e99c87af902)

While there are obviously deviations between the model and actual data, we think that trends are mostly sufficiently followed.
On some occasions, the model under-predicts, on others it over-predicts, but we believe the model generally overestimates the average delay and underestimates the on-time percentage.

## Real-Time Functionality

The system is batch-inference, although the batches are so frequent it is quite close to real time.
In order to facilitate this, data needs to be updated frequently and predictions need to be made.
For this, three pipelines are required:
* One daily backfill pipeline, which writes the ground truth of the delay values of the day before
This ground truth is then later on used to compute the historical performance.
* One hourly feature pipeline, which extracts the latest features for the current hour.
In theory, this could and should be run more frequently.
However, processing times and CI limitations severely limit the ability to run in real time.
* One hourly prediction pipeline, which performs predictions such that accuracy can be monitored for the past.
Only the pipeline inferences log predictions, the one in the UI is not saved anywhere.

## UI

The Streamlit UI provides a way to infer the data and to see historical performance.
Streamlit was chosen as it is very easy, efficient, flexible and elegant to use.

Currently, all data and the whole model is loaded into Streamlit.
As of right now, this does not take too many rows/resources.
The caching mechanism built into Streamlit alleviates performance significantly.
In the future, the model could be served on Hopsworks or Modal.
However, we are still proud that the current solution is serverless.

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



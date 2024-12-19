import training_helpers
import training_keras
import training_xgboost
import hopsworks
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from xgboost import XGBRegressor
import os

xgboost_grid = {
    "learning_rate": [0.1],
    "max_depth": [4]
}

keras_grid = {
    "model__lr": [0.001],
    "model__hidden": [8],
    "model__activation": ["relu"],
}

# xgboost_grid = {
#     "learning_rate": [0.1, 0.2, 0.3, 0.4],
#     "max_depth": [2, 4, 6, 8]
# }
#
# keras_grid = {
#     "model__lr": [0.001, 0.005, 0.1],
#     "model__hidden": [8, 10, 16, 32],
#     "model__activation": ["relu", "sigmoid", "tanh"],
# }

CV_FOLDS = 2


def load_data():
    api_key = open(".hw_key").read().strip()
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", training_helpers.LATEST_FV)
    if os.getenv("REMAKE", "") == "true":
        print("Creating training data")
        fv.create_training_data(description="All datapoints given as training data")
        return
    return fv.get_training_data(training_dataset_version=training_helpers.LATEST_TD)


def grid_search(X_all, Y_all):
    X_all = training_helpers.strip_dates(X_all)
    Y_all = Y_all[training_helpers.TO_PREDICT]
    grid_search_xgboost(X_all, Y_all)
    grid_search_keras(X_all, Y_all)


def grid_search_xgboost(X_all, Y_all):
    model = XGBRegressor()
    search = GridSearchCV(estimator=model, param_grid=xgboost_grid, scoring="r2", cv=CV_FOLDS, n_jobs=-1)
    search.fit(X_all, Y_all)
    print("Best XGBRegressor: ", search.best_params_)
    print("Scored:", search.best_score_)
    training_xgboost.train_best(search.best_params_, X_all, Y_all)


def grid_search_keras(X_all, Y_all):
    model = KerasRegressor(model=training_keras.create_model, epochs=training_keras.EPOCHS, batch_size=training_keras.BATCH_SIZE)
    search = GridSearchCV(estimator=model, param_grid=keras_grid, cv=CV_FOLDS, n_jobs=-1)
    search.fit(X_all, Y_all)
    print("Best DNN:", search.best_params_)
    print("Scored: ", search.best_score_)
    params = {k[len("model__"):]: v for k, v in search.best_params_.items()}
    training_keras.train_best(params, X_all, Y_all)


if __name__ == "__main__":
    features_, labels_ = load_data()
    X_to_use, X_hold_back = training_helpers.train_test_split(features_)
    Y_to_use, Y_hold_back = training_helpers.train_test_split(labels_)
    grid_search(X_to_use, Y_to_use)
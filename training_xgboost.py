import numpy as np
from xgboost import XGBRegressor
import training_helpers
import hopsworks
import os
from sklearn.metrics import r2_score

TO_PREDICT = ["mean_arrival_delay_seconds", "max_arrival_delay_seconds", "mean_departure_delay_seconds", "max_departure_delay_seconds"]


def train_and_evaluate(X_train, Y_train, X_validate, Y_validate, lr, max_depth):
    X_train = training_helpers.strip_dates(X_train)
    X_validate = training_helpers.strip_dates(X_validate)
    model = XGBRegressor(max_depth=max_depth, learning_rate=lr)
    model.fit(X_train, Y_train[TO_PREDICT])
    Y_pred = model.predict(X_validate)
    r2s = {}
    print(f"XGBRegressor[eta={lr} max_depth={max_depth}]")
    for i, label in enumerate(TO_PREDICT):
        r2 = r2_score(Y_validate[label], Y_pred[:, i])
        print("\tFeature", label, "R^2:", r2)
        r2s[label] = r2
    average_score = np.mean(list(r2s.values()))
    print("\tAverage R^2:", average_score)
    return average_score


if __name__ == "__main__":
    api_key = open(".hw_key").read().strip()
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", training_helpers.LATEST_FV)
    if os.getenv("REMAKE", "") == "true":
        print("Creating training data")
        fv.create_training_data(description="All datapoints given as training data")
    x_all, y_all = fv.get_training_data(training_dataset_version=training_helpers.LATEST_TD)
    print("all", x_all, y_all)
    x_train, x_validate, x_test = training_helpers.train_validate_test_split(x_all)
    y_train, y_validate, y_test = training_helpers.train_validate_test_split(y_all)
    print("train", x_train, y_train)
    train_and_evaluate(x_train, y_train, x_validate, y_validate, 0.3, 6)

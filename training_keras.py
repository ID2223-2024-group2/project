import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import training_helpers
import hopsworks
import os


def train_and_evaluate(X_train, Y_train, X_validate, Y_validate, lr, hidden_size, func):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validate = scaler.transform(X_validate)
    model = Sequential([
        Dense(hidden_size, activation=func, input_shape=(X_train.shape[1],)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["r2_score"])
    model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=50, batch_size=32, verbose=1)
    eval_results = model.evaluate(X_validate, Y_validate, verbose=0)
    print(eval_results)


if __name__ == "__main__":
    api_key = open(".hw_key").read().strip()
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", training_helpers.LATEST_FV)
    x_all, y_all = fv.get_training_data(training_dataset_version=training_helpers.LATEST_TD)
    x_train, x_validate, x_test = training_helpers.train_validate_test_split(x_all)
    y_train, y_validate, y_test = training_helpers.train_validate_test_split(y_all)
    train_and_evaluate(x_train, y_train, x_validate, y_validate, 0.3, 16, "relu")

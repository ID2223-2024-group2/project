import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import training_helpers
import hopsworks
import random


EPOCHS = 40
BATCH_SIZE = 24


def create_model(lr=0.001, hidden=16, activation="relu"):
    print("CREATING MODEL WITH", lr, hidden, activation)
    model = Sequential([
        Dense(hidden, activation=activation, input_shape=(training_helpers.NUM_FEATURES,)),
        Dense(len(training_helpers.TO_PREDICT))
    ])
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse", metrics=["r2_score"], run_eagerly=True)
    return model


def train_and_evaluate(X_train, Y_train, X_validate, Y_validate, lr, hidden_size, func, deterministic=False):
    if deterministic:
        random.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    model = create_model(lr, hidden_size, func)
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    eval_results = model.evaluate(X_validate, Y_validate, verbose=0)
    print(f"DNN[lr={lr} hidden_size={hidden_size} func={func}]")
    r2 = eval_results[1]
    print("\tTotal R^2:", r2)
    return r2


def train_best(parameters, X_full, Y_full):
    model = create_model(**parameters)
    model.fit(X_full, Y_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    tf.saved_model.save(model, training_helpers.get_model_dir("keras"))


if __name__ == "__main__":
    api_key = open(".hw_key").read().strip()
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", training_helpers.LATEST_FV)
    x_all, y_all = fv.get_training_data(training_dataset_version=training_helpers.LATEST_TD)
    x_train, x_validate, x_test = training_helpers.train_validate_test_split(x_all)
    y_train, y_validate, y_test = training_helpers.train_validate_test_split(y_all)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(training_helpers.strip_dates(x_train))
    x_validate = scaler.transform(training_helpers.strip_dates(x_validate))
    y_train = y_train[training_helpers.TO_PREDICT]
    y_validate = y_validate[training_helpers.TO_PREDICT]
    train_and_evaluate(x_train, y_train, x_validate, y_validate, 0.001, 16, "relu")

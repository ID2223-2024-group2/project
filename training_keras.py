import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import training_helpers
import random
from datetime import datetime
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


FEATURE_DIM = None
EPOCHS = 40
BATCH_SIZE = 32


def init_feature_dim(X_all):
    global FEATURE_DIM
    FEATURE_DIM = X_all.shape[1]
    print("Feature dim is", FEATURE_DIM)


def create_model(lr=0.001, hidden=16, activation="relu", feature_dim=None):
    if feature_dim is None:
        feature_dim = FEATURE_DIM
        if feature_dim is None:
            raise ValueError("FEATURE_DIM is None, init_feature_dim(X) has not been called")
    print("CREATING MODEL WITH", lr, hidden, activation)
    model = Sequential([
        Dense(hidden, activation=activation, input_shape=(feature_dim,)),
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
    model.fit(X_train, Y_train, validation_data=(X_validate, Y_validate), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    eval_results = model.evaluate(X_validate, Y_validate, verbose=0)
    print(f"DNN[lr={lr} hidden_size={hidden_size} func={func}]")
    r2 = eval_results[1]
    print("\tTotal R^2:", r2)
    return r2, model


def train_best(parameters, X_full, Y_full, scale_features, scale_labels):
    model = create_model(**parameters)
    model.fit(X_full, Y_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    save(model, X_full, Y_full, scale_features, scale_labels)


def save(model, X_all, Y_all, scale_features, scale_labels):
    model.x_scaler = tf.Variable([scale_features.mean_, scale_features.scale_], trainable=False)
    model.y_scaler = tf.Variable([scale_labels.mean_, scale_labels.scale_], trainable=False)
    tf.saved_model.save(model, training_helpers.get_model_dir("keras"))
    api_key = open(".hw_key").read().strip()
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    input_schema = Schema(X_all)
    output_schema = Schema(Y_all)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    hw_model = mr.python.create_model(
        name="keras",
        model_schema=model_schema,
        description="Gävleborg delay predictor",
    )
    hw_model.save(training_helpers.get_model_dir("keras"))


if __name__ == "__main__":
    test_start = pd.to_datetime(datetime.strptime("2024-06-22", "%Y-%m-%d"))
    x_train, y_train, x_test, y_test = training_helpers.load_xy_time(test_start)
    print(x_train.info())
    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(x_train)
    x_test = feature_scaler.transform(x_test)
    label_scaler = StandardScaler()
    y_train = label_scaler.fit_transform(y_train)
    y_test = label_scaler.transform(y_test)
    init_feature_dim(x_train)
    _, trained = train_and_evaluate(x_train, y_train, x_test, y_test, 0.005, 16, "relu")
    #save(trained, x_train, y_train, feature_scaler, label_scaler)

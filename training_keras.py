import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import training_helpers
import random


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
    return r2


def train_best(parameters, X_full, Y_full):
    model = create_model(**parameters)
    model.fit(X_full, Y_full, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    tf.saved_model.save(model, training_helpers.get_model_dir("keras"))


if __name__ == "__main__":
    x_all, y_all = training_helpers.load_dataset()
    x_train, x_test = training_helpers.train_test_split(x_all)
    y_train, y_test = training_helpers.train_test_split(y_all)
    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(x_train)
    x_test = feature_scaler.transform(x_test)
    label_scaler = StandardScaler()
    y_train = label_scaler.fit_transform(y_train)
    y_test = label_scaler.transform(y_test)
    init_feature_dim(x_all)
    train_and_evaluate(x_train, y_train, x_test, y_test, 0.01, 16, "relu")

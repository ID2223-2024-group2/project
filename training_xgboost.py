import numpy as np
from xgboost import XGBRegressor, plot_importance
import training_helpers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


def train_and_evaluate(X_train, Y_train, X_validate, Y_validate, lr, max_depth):
    model = XGBRegressor(max_depth=max_depth, learning_rate=lr)
    model.fit(X_train, Y_train[training_helpers.TO_PREDICT])
    Y_pred = model.predict(X_validate)
    r2s = {}
    print(f"XGBRegressor[eta={lr} max_depth={max_depth}]")
    for i, label in enumerate(training_helpers.TO_PREDICT):
        if Y_pred.ndim == 1:
            prediction = Y_pred
        else:
            prediction = Y_pred[:, i]
        r2 = r2_score(Y_validate[label], prediction)
        print("\tFeature", label, "R^2:", r2)
        r2s[label] = r2
    average_score = np.mean(list(r2s.values()))
    print("\tAverage R^2:", average_score)
    plot_importance(model)
    plt.show()
    return average_score, model


def train_best(params, X_all, Y_all):
    model = XGBRegressor(**params)
    model.fit(X_all, Y_all)
    save(model, X_all, Y_all)
    print("Saved!")


def save(model, X_all, Y_all):
    model.save_model(training_helpers.get_model_dir("xgboost.json"))
    api_key = open(".hw_key").read().strip()
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    input_schema = Schema(X_all)
    output_schema = Schema(Y_all)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    hw_model = mr.python.create_model(
        name="xgboost",
        model_schema=model_schema,
        input_example=X_all.sample().values,
        description="Gävleborg delay predictor",
    )
    hw_model.save(training_helpers.get_model_dir("xgboost.json"))


if __name__ == "__main__":
    test_start = pd.to_datetime(datetime.strptime("2024-06-22", "%Y-%m-%d"))
    x_train, y_train, x_test, y_test = training_helpers.load_xy_time(test_start)
    _, trained = train_and_evaluate(x_train, y_train, x_test, y_test, 0.1, 2)
    #save(trained, x_train, y_train)

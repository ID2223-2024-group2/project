import numpy as np
from xgboost import XGBRegressor, plot_importance
import training_helpers
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


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
    return average_score


def train_best(params, X_all, Y_all):
    model = XGBRegressor(**params)
    model.fit(X_all, Y_all)
    model.save_model(training_helpers.get_model_dir("xgboost.json"))
    print("Saved!")


if __name__ == "__main__":
    x_all, y_all = training_helpers.load_dataset()
    x_train, x_test = training_helpers.train_test_split(x_all)
    y_train, y_test = training_helpers.train_test_split(y_all)
    train_and_evaluate(x_train, y_train, x_test, y_test, 0.1, 2)

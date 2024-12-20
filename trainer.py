import training_helpers
import training_keras
import training_xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor
from xgboost import XGBRegressor

xgboost_grid = {
    "learning_rate": [0.1, 0.2, 0.3, 0.4],
    "max_depth": [2, 4, 6, 8]
}

keras_grid = {
    "model__lr": [0.001, 0.005, 0.1],
    "model__hidden": [8, 10, 16, 32],
    "model__activation": ["relu", "sigmoid", "tanh"],
    "model__feature_dim": [18]  # Needs to be set manually because of how "multithreading" works
}

CV_FOLDS = 5


def load_data():
    return training_helpers.load_dataset()


def grid_search(X_all, Y_all):
    print("Starting XGBoost search")
    grid_search_xgboost(X_all, Y_all)
    print("Starting Keras search")
    #grid_search_keras(X_all, Y_all)


def grid_search_xgboost(X_all, Y_all):
    model = XGBRegressor()
    search = GridSearchCV(estimator=model, param_grid=xgboost_grid, scoring="r2", cv=CV_FOLDS, n_jobs=-1)
    search.fit(X_all, Y_all)
    print("Best XGBRegressor: ", search.best_params_)
    print("Scored:", search.best_score_)
    training_xgboost.train_best(search.best_params_, X_all, Y_all)


def grid_search_keras(X_all, Y_all):
    feature_scaler = StandardScaler()
    X_all = feature_scaler.fit_transform(X_all)
    label_scaler = StandardScaler()
    Y_all = label_scaler.fit_transform(Y_all)
    training_keras.init_feature_dim(X_all)
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
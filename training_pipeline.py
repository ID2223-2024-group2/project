import os
from datetime import datetime

import hopsworks
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

if os.environ.get("HOPSWORKS_API_KEY") is None:
    os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()


def train_model(labels, train_features, y_train):
    xgb_regressor = XGBRegressor()
    xgb_regressor.fit(train_features, y_train[labels])

    return xgb_regressor


def save_model(model: object, model_dir: str) -> None:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.join(model_dir, "model.json")

    input_schema = Schema(X_train)
    output_schema = Schema(y_train)

    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    schema_dict = model_schema.to_dict()

    model.save_model(model_path)
    res_dict = {
        "MSE": mse_scores,
        "R squared": r2_scores,
    }

    mr = project.get_model_registry()

    delays_model = mr.python.create_model(
        name="delays_xgboost_model",
        metrics=res_dict,
        model_schema=model_schema,
        input_example=X_test.sample().values,
        description="Delays predictor",
    )

    delays_model.save(model_path)


if __name__ == "__main__":
    MODEL_DIR = os.environ.get("MODEL_DIR", "models/delays")

    project = hopsworks.login()
    fs = project.get_feature_store(name='tsedmid2223_featurestore')
    print("Connected to Hopsworks Feature Store. ")

    delays_fg = fs.get_feature_group(
        name='delays',
        version=6,
    )
    weather_fg = fs.get_feature_group(
        name='weather',
        version=2,
    )
    # Join delays and weather feature groups on arrival_time_bin and date respectively
    selected_features = delays_fg.select_all().join(
        weather_fg.select_all(),
        left_on=['arrival_time_bin'],
        right_on=['date'],
        join_type='inner'
    )
    selected_features = selected_features.filter(selected_features['stop_count'] > 0)

    labels = ['mean_delay_change_seconds', 'max_delay_change_seconds', 'min_delay_change_seconds',
                             'var_delay_change_seconds',
                             'mean_arrival_delay_seconds', 'max_arrival_delay_seconds', 'min_arrival_delay_seconds',
                             'var_arrival_delay',
                             'mean_departure_delay_seconds', 'max_departure_delay_seconds',
                             'min_departure_delay_seconds', 'var_departure_delay',
                             'mean_on_time_percent', 'mean_final_stop_delay_seconds']

    feature_view = fs.get_or_create_feature_view(
        name='delays_fv',
        description="weather features with delays as the target",
        version=4,
        labels=labels,
        query=selected_features,
    )
    print("Retrieved Feature View. ")

    start_date_test_data = "2024-01-01"
    test_start = datetime.strptime(start_date_test_data, "%Y-%m-%d")
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_start=test_start)
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    train_features = X_train.drop(['date', 'arrival_time_bin'], axis=1)
    test_features = X_test.drop(['date', 'arrival_time_bin'], axis=1)

    print(X_train)
    print(y_train)

    model = train_model(labels, train_features, y_train)
    y_pred = model.predict(test_features)

    mse_scores = {}
    r2_scores = {}
    for i, label in enumerate(labels):
        mse = mean_squared_error(y_test[label], y_pred[:, i])
        mse_scores[label] = mse
        print(f"MSE for {label}: {mse}")

        r2 = r2_score(y_test[label], y_pred[:, i])
        r2_scores[label] = r2
        print(f"R squared for {label}: {r2}")

    plot_importance(model, max_num_features=8)
    plt.savefig("feature_importance.png")

    save_model(model, MODEL_DIR)
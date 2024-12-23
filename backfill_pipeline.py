import os
import hopsworks
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import training_helpers
import pandas as pd


if os.environ.get("HOPSWORKS_API_KEY") is None:
    os.environ["HOPSWORKS_API_KEY"] = open(".hw_key").read()


def make_inference(transport_int):
    fs = project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", FEATURE_FV_VERSION)
    df = fv.get_batch_data()
    df.sort_values(by=["arrival_time_bin"], inplace=True, ascending=False)
    correct_transport = df[df["route_type"] == transport_int]
    last_entries = correct_transport.head(10000)
    stripped = training_helpers.strip_dates(last_entries)
    useful = stripped[training_helpers.TO_USE]
    one_hotted = training_helpers.one_hot(useful)
    feature = tf.dtypes.cast(feature_scaler.transform(one_hotted), tf.float32)
    predictions = infer(feature)["output_0"]
    values = label_scaler.inverse_transform(predictions)
    numpy_array = values.numpy()
    preds = pd.DataFrame(numpy_array)
    preds["date"] = df["arrival_time_bin"]
    preds["transport_type"] = transport_int
    preds.rename(columns={0: "predicted_mean_arrival_delay_seconds", 1: "predicted_mean_on_time_percent"}, inplace=True)
    preds = preds[["date", "transport_type", "predicted_mean_arrival_delay_seconds", "predicted_mean_on_time_percent"]]
    monitor_fg = fs.get_or_create_feature_group(
        name="delays_predictions",
        description="Train delay prediction monitoring",
        version=MONITOR_FV_VERSION,
        primary_key=["date", "transport_type"],
        event_time="date"
    )
    monitor_fg.insert(preds, write_options={"wait_for_job": True})


if __name__ == "__main__":
    MODEL_VERSION = int(os.environ.get("MODEL_VERSION", 3))
    FEATURE_FV_VERSION = int(os.environ.get("FEATURE_FV_VERSION", 7))
    MONITOR_FV_VERSION = int(os.environ.get("MONITOR_FV_VERSION", 1))
    project = hopsworks.login(project="TSEDMID2223")
    mr = project.get_model_registry()
    hw_model = mr.get_model(name="keras", version=MODEL_VERSION)
    where_model = hw_model.download()
    loaded_model = tf.saved_model.load(where_model)
    feature_scaler = StandardScaler()
    feature_scaler.mean_ = loaded_model.x_scaler[0]
    feature_scaler.scale_ = loaded_model.x_scaler[1]
    label_scaler = StandardScaler()
    label_scaler.mean_ = loaded_model.y_scaler[0]
    label_scaler.scale_ = loaded_model.y_scaler[1]
    infer = loaded_model.signatures["serving_default"]
    make_inference(100)
    make_inference(700)

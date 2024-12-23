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
    last_entry = correct_transport.head(1)
    stripped = training_helpers.strip_dates(last_entry)
    useful = stripped[training_helpers.TO_USE]
    one_hotted = training_helpers.one_hot(useful)
    feature = tf.dtypes.cast(feature_scaler.transform(one_hotted), tf.float32)
    predictions = infer(feature)["output_0"]
    values = label_scaler.inverse_transform(predictions)
    delay = tf.squeeze(values[0, 0]).numpy()
    on_time = tf.squeeze(values[0, 1]).numpy()
    write_inference(last_entry["arrival_time_bin"], transport_int, delay, on_time)


def write_inference(date, transport_int, delay, on_time):
    fs = project.get_feature_store("tsedmid2223_featurestore")
    monitor_fg = fs.get_or_create_feature_group(
        name="delays_predictions",
        description="Train delay prediction monitoring",
        version=MONITOR_FV_VERSION,
        primary_key=["date", "transport_type"],
        event_time="date"
    )
    data = {
        "date": date.to_list(),
        "transport_type": [transport_int],
        "predicted_mean_arrival_delay_seconds": [delay],
        "predicted_mean_on_time_percent": [on_time]
    }
    df = pd.DataFrame.from_dict(data)
    # We need to wait since we do multiple inserts.
    monitor_fg.insert(df, write_options={"wait_for_job": True})


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

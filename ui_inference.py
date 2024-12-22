import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import training_helpers

MODEL_VERSION = 3
FV_VERSION = 7
TTL = 5 * 60


@st.cache_resource
def download_model(_project):
    mr = _project.get_model_registry()
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
    return infer, feature_scaler, label_scaler


@st.cache_data(ttl=TTL)
def download_all_data(_project):
    fs = _project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", 7)
    df = fv.get_batch_data()
    df.sort_values(by=["arrival_time_bin"], inplace=True, ascending=False)
    return df


@st.cache_data(ttl=TTL)
def download_last_entry(_project, transport_string):
    if transport_string == "Train":
        transport_int = 100
    elif transport_string == "Bus":
        transport_int = 700
    else:
        raise RuntimeError("unknown transportation type " + transport_string)
    df = download_all_data(_project)
    correct_transport = df[df["route_type"] == transport_int]
    last = correct_transport.head(1)
    return last


def inference(infer, feature_scaler, label_scaler, last_entry):
    stripped = training_helpers.strip_dates(last_entry)
    useful = stripped[training_helpers.TO_USE]
    one_hotted = training_helpers.one_hot(useful)
    feature = tf.dtypes.cast(feature_scaler.transform(one_hotted), tf.float32)
    predictions = infer(feature)["output_0"]
    values = label_scaler.inverse_transform(predictions)
    delay = tf.squeeze(values[0, 0]).numpy()
    on_time = tf.squeeze(values[0, 1]).numpy()
    return delay, on_time

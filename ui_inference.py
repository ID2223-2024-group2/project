import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import requests
import training_helpers
import time

MODEL_VERSION = 3
FV_VERSION = 7
TTL = 5 * 60

GIT_OWNER = "ID2223-2024-group2"
GIT_REPO = "project"
GIT_WORKFLOW = "regular-update.yml"
GIT_REF = "main"
GIT_BASE_URL = f"https://api.github.com/repos/{GIT_OWNER}/{GIT_REPO}"


def inference(project, transport_string):
    infer, feature_scaler, label_scaler = download_model(project)
    last_entry = download_last_entry(project, transport_string)
    delay, on_time = run_inference(infer, feature_scaler, label_scaler, last_entry)
    return delay, on_time, last_entry["arrival_time_bin"].tolist()[0]


@st.cache_resource(show_spinner="Downloading AI model")
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


@st.cache_data(ttl=TTL, show_spinner="Downloading data")
def download_all_data(_project):
    fs = _project.get_feature_store("tsedmid2223_featurestore")
    fv = fs.get_feature_view("delays_fv", 7)
    df = fv.get_batch_data()
    df.sort_values(by=["arrival_time_bin"], inplace=True, ascending=False)
    return df


@st.cache_data(ttl=TTL, show_spinner="Preparing data")
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


def run_inference(infer, feature_scaler, label_scaler, last_entry):
    stripped = training_helpers.strip_dates(last_entry)
    useful = stripped[training_helpers.TO_USE]
    one_hotted = training_helpers.one_hot(useful)
    feature = tf.dtypes.cast(feature_scaler.transform(one_hotted), tf.float32)
    predictions = infer(feature)["output_0"]
    values = label_scaler.inverse_transform(predictions)
    delay = tf.squeeze(values[0, 0]).numpy()
    on_time = tf.squeeze(values[0, 1]).numpy()
    return delay, on_time


def github_headers():
    github_token = os.environ.get("GITHUB_TOKEN", open(".github_token").read().strip())
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github+json"
    }
    return headers


def github_workflow_running_now(headers):
    url = f"{GIT_BASE_URL}/actions/runs"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    runs = response.json()["workflow_runs"]
    for run in runs:
        if run["head_branch"] == GIT_REF:
            return run["conclusion"] is None


def github_workflow_trigger(headers):
    url = f"{GIT_BASE_URL}/actions/workflows/{GIT_WORKFLOW}/dispatches"
    response = requests.post(url, headers=headers, json={"ref": GIT_REF})
    response.raise_for_status()


def github_workflow_wait(headers):
    with st.spinner("Waiting for pipeline re-run eligibility..."):
        while github_workflow_running_now(headers):
            time.sleep(5)
        download_all_data.clear()
        download_last_entry.clear()
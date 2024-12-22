import streamlit as st
import ui_helpers
import datetime
import pandas as pd
import numpy as np
import hopsworks
import os

import ui_inference

st.title("🐐 Gävleborg Train Delay Forecast")
st.write("*A snow-accelerated real-time delay estimator.*")


@st.cache_resource(show_spinner="Connecting to Hopsworks")
def get_project():
    hopsworks_api_key = os.environ.get("HOPSWORKS_API_KEY", open(".hw_key").read().strip())
    return hopsworks.login(api_key_value=hopsworks_api_key)


project = get_project()
tab_predict, tab_evaluate = st.tabs(["Forecast", "Historical Accuracy"])


with tab_predict:
    col1, col2 = st.columns(2)
    with col1:
        dummy_date = datetime.datetime(2024, 12, 21, 13, 0, 0)
        options = ui_helpers.get_forecast_options(dummy_date)
        what_date = st.selectbox("Forecast interval",
                                 options=options,
                                 disabled=True,
                                 format_func=lambda x: ui_helpers.get_forecast_labels(dummy_date, x))
    with col2:
        transport_string = transport_mode = st.selectbox("Mode of transportation", options=["Train", "Bus"], key="foo")
    infer, feature_scaler, label_scaler = ui_inference.download_model(project)
    last_entry = ui_inference.download_last_entry(project, transport_string)
    delay, on_time = ui_inference.inference(infer, feature_scaler, label_scaler, last_entry)
    with col1:
        st.metric("Estimated Avg. Arrival Delay", ui_helpers.seconds_to_minute_string(delay))
    with col2:
        st.metric("Estimated On Time Percentage", f"{on_time:.1f}%")
    st.write("*Transport is on-time if it arrives within 3 minutes before or 5 minutes after its scheduled time.*")
    st.divider()
    st.write("Forecasts are generated every quarter-hour, but may take a few minutes to appear. "
             "Predictions may not reflect reality. All data is given as-is without guarantees.")


with tab_evaluate:
    st.write("Lipsum")
    col1, col2 = st.columns(2)
    with col1:
        what_type = st.selectbox("Estimator", options=["Avg. Arrival Delay", "On Time Percentage"])
    with col2:
        what_mode = st.selectbox("Mode of transportation", options=["Train", "Bus"], key="bar")
    ranges = [f"{x + 1}h" for x in range(0, 6)]
    what_data = st.pills("Forecast range", ranges, selection_mode="multi")
    dates = pd.date_range(start=pd.Timestamp.now().date() - pd.Timedelta(days=12), periods=10)
    delay_schema = {
        "date": dates,
        "1h": np.random.randint(0, 101, size=10),
        "2h": np.random.randint(0, 101, size=10),
        "3h": np.random.randint(0, 101, size=10),
        "4h": np.random.randint(0, 101, size=10),
        "5h": np.random.randint(0, 101, size=10),
        "6h": np.random.randint(0, 101, size=10)
    }
    delay_df = pd.DataFrame(delay_schema, columns=list(delay_schema.keys()))
    st.markdown("####")
    if len(what_data) > 0:
        st.line_chart(delay_df, x="date", y=what_data)

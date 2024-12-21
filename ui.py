import streamlit as st
import ui_helpers
import datetime
import pandas as pd
import numpy as np

st.title("🐐 Gävleborg Train Delay Forecast")
st.write("*A snow-accelerated real-time delay estimator.*")

tab_predict, tab_evaluate = st.tabs(["Forecast", "Historical Accuracy"])


with tab_predict:
    col1, col2 = st.columns(2)
    with col1:
        dummy_date = datetime.datetime(2024, 12, 21, 13, 0, 0)
        options = ui_helpers.get_forecast_options(dummy_date)
        what_date = st.selectbox("Forecast interval",
                                 options=options,
                                 format_func=lambda x: ui_helpers.get_forecast_labels(dummy_date, x))
    with col2:
        transport_mode = st.selectbox("Mode of transportation", options=["Train", "Bus"], key="foo")
    with col1:
        st.metric("Avg. Arrival Delay", "324s", "-23s")
    with col2:
        st.metric("On Time Percentage", "80%", "3%")
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

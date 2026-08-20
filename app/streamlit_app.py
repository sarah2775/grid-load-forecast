"""
Interactive dashboard for regional electricity demand forecasts.
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Grid Load Forecast", layout="wide")

st.title("⚡ Delhi Electricity Demand Forecast")
st.caption("Short-term load forecasting for Delhi (NCT) with uncertainty intervals")

horizon = st.slider("Forecast horizon (hours)", min_value=24, max_value=168, value=24, step=24)
model_choice = st.selectbox("Model", ["Seasonal Naive", "SARIMA", "LightGBM", "LSTM", "TFT / N-BEATS"])

st.info("TODO: load the trained model, generate forecast + prediction interval for the selected horizon, and plot it (plotly) alongside historical demand.")

# Placeholder layout for what's coming:
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Forecast")
    st.write("Forecast plot goes here.")
with col2:
    st.subheader("Top demand drivers")
    st.write("SHAP summary goes here.")

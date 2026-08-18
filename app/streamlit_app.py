"""
Interactive dashboard for regional electricity demand forecasts.
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Grid Load Forecast", layout="wide")

st.title("⚡ India Electricity Demand Forecast")
st.caption("Multi-region short-term load forecasting with uncertainty intervals")

region = st.selectbox("Region", ["Northern", "Western", "Southern", "Eastern", "North-Eastern"])
horizon = st.slider("Forecast horizon (hours)", min_value=24, max_value=168, value=24, step=24)

st.info("TODO: load the trained model, generate forecast + prediction interval for the selected region/horizon, and plot it (plotly) alongside historical demand.")

# Placeholder layout for what's coming:
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Forecast")
    st.write("Forecast plot goes here.")
with col2:
    st.subheader("Top demand drivers")
    st.write("SHAP summary goes here.")

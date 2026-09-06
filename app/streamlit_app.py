"""
Interactive dashboard for Delhi electricity demand forecasting.
Run with: streamlit run app/streamlit_app.py

Reads pre-computed backtest predictions and analysis tables produced by
the notebooks — this dashboard does not retrain models live (SARIMA in
particular is too slow to refit on every interaction), it lets you
explore and present the results that were already validated in
02_baselines.ipynb, 03_ml_models.ipynb, and 04_error_analysis.ipynb.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Delhi Load Forecast", layout="wide")

# --- Dark purple/pink theme for embedded Plotly charts ---
# Streamlit's own theme (.streamlit/config.toml) covers the app chrome, but
# Plotly figures need their colors set explicitly to match.
BG_COLOR = "#0A0110"
PLOT_BG = "#150720"
GRID_COLOR = "#3A1F52"
TEXT_COLOR = "#F2E9FF"
COLOR_ACTUAL = "#F2E9FF"     # near-white, so the ground truth always reads clearly against the models
COLOR_SARIMA = "#9D4EDD"    # purple
COLOR_LGB = "#FF4DA6"       # pink
COLOR_ACCENT = "#C77DFF"    # lighter purple, used for the single-series history chart

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT_COLOR),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
)

# A touch of extra CSS: reduce the default top whitespace, and add card styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #1A0B2E;
        border: 1px solid #3A1F52;
        border-radius: 8px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value, color):
    """Custom metric card with a per-model color (st.metric can't be
    color-customized per-instance, only globally)."""
    st.markdown(
        f"""
        <div style="background-color:#1A0B2E; border:1px solid #3A1F52;
                    border-radius:8px; padding:14px 18px; margin-bottom:8px;">
            <div style="color:#B8A0D9; font-size:0.85rem;">{label}</div>
            <div style="color:{color}; font-size:1.8rem; font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


@st.cache_data
def load_data():
    """Load raw data + all pre-computed predictions/analysis tables. Cached so
    the dashboard doesn't re-read from disk on every widget interaction."""
    raw = pd.read_csv(DATA_DIR / "load_weather_hourly.csv", index_col=0, parse_dates=True)
    raw.index.name = "timestamp"

    sarima_preds = pd.read_csv(REPORTS_DIR / "sarima_predictions.csv", parse_dates=["timestamp"])
    lgb_preds = pd.read_csv(REPORTS_DIR / "lightgbm_predictions.csv", parse_dates=["timestamp"])

    seasonal_mape = pd.read_csv(REPORTS_DIR / "seasonal_mape.csv")
    temp_mape = pd.read_csv(REPORTS_DIR / "temp_mape.csv")
    hourly_mape = pd.read_csv(REPORTS_DIR / "hourly_mape.csv")

    return raw, sarima_preds, lgb_preds, seasonal_mape, temp_mape, hourly_mape


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


try:
    raw, sarima_preds, lgb_preds, seasonal_mape, temp_mape, hourly_mape = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing data file: {e.filename}\n\n"
        "This dashboard needs the processed dataset and saved backtest predictions. "
        "Run the notebooks in order first (01 → 02 → 03 → 04), including the "
        "`.to_csv(...)` cells that save predictions and analysis tables to `reports/`. "
        "See the README's Reproducibility section."
    )
    st.stop()


st.title("⚡ Delhi Electricity Demand Forecast")
st.caption(
    "Backtest results from SARIMA and LightGBM models, validated with walk-forward "
    "cross-validation. Explore historical forecast accuracy and where each model wins."
)

tab1, tab2 = st.tabs(["📈 Backtest Explorer", "🔍 Where Each Model Wins"])

# ---------------------------------------------------------------------------
# TAB 1: Backtest Explorer
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Explore a backtested forecast window")

    model_choice = st.radio("Model", ["SARIMA", "LightGBM", "Both"], horizontal=True)

    available_folds = sorted(sarima_preds["fold"].unique()) if model_choice != "LightGBM" else sorted(lgb_preds["fold"].unique())
    fold_dates = {
        f: sarima_preds[sarima_preds["fold"] == f]["timestamp"].min() if model_choice != "LightGBM"
        else lgb_preds[lgb_preds["fold"] == f]["timestamp"].min()
        for f in available_folds
    }
    fold_labels = {f: d.strftime("%Y-%m-%d") for f, d in fold_dates.items()}

    selected_fold = st.selectbox(
        "Forecast window (each is a 24h backtested fold)",
        options=available_folds,
        format_func=lambda f: fold_labels[f],
    )

    fig = go.Figure()

    if model_choice in ("SARIMA", "Both"):
        s = sarima_preds[sarima_preds["fold"] == selected_fold]
        if len(s) > 0:
            fig.add_trace(go.Scatter(x=s["timestamp"], y=s["y_true"], name="Actual", line=dict(color=COLOR_ACTUAL, width=2)))
            fig.add_trace(go.Scatter(x=s["timestamp"], y=s["y_pred"], name="SARIMA", line=dict(color=COLOR_SARIMA)))

    if model_choice in ("LightGBM", "Both"):
        l = lgb_preds[lgb_preds["fold"] == selected_fold]
        if len(l) > 0:
            if model_choice == "LightGBM":  # avoid plotting "Actual" twice when showing both
                fig.add_trace(go.Scatter(x=l["timestamp"], y=l["y_true"], name="Actual", line=dict(color=COLOR_ACTUAL, width=2)))
            fig.add_trace(go.Scatter(x=l["timestamp"], y=l["y_pred"], name="LightGBM", line=dict(color=COLOR_LGB)))

    fig.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", height=450, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    if model_choice in ("SARIMA", "Both") and len(sarima_preds[sarima_preds["fold"] == selected_fold]) > 0:
        s = sarima_preds[sarima_preds["fold"] == selected_fold]
        with col1:
            metric_card("SARIMA — accuracy for this day", f"{mape(s['y_true'], s['y_pred']):.2f}% MAPE", COLOR_SARIMA)
    if model_choice in ("LightGBM", "Both") and len(lgb_preds[lgb_preds["fold"] == selected_fold]) > 0:
        l = lgb_preds[lgb_preds["fold"] == selected_fold]
        with col2:
            metric_card("LightGBM — accuracy for this day", f"{mape(l['y_true'], l['y_pred']):.2f}% MAPE", COLOR_LGB)

    st.caption(
        "These numbers are the error for the single selected day only, not the "
        "full-backtest average reported in the README (SARIMA 4.50%, LightGBM 4.96% "
        "overall) — pick different days above to see how much accuracy varies day to day."
    )

    st.divider()
    st.subheader("Recent historical load")
    lookback_days = st.slider("Show last N days of raw history", 7, 90, 30)
    recent = raw["load_MW"].iloc[-24 * lookback_days:]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=recent.index, y=recent.values, line=dict(color=COLOR_ACCENT)))
    fig2.update_layout(xaxis_title="Time", yaxis_title="Load (MW)", height=300, **PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2: Error Analysis (from 04_error_analysis.ipynb)
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("When does each model perform better?")
    st.caption(
        "From the project's error analysis notebook — SARIMA wins on the full-year "
        "average, but that hides real structure underneath. Explore it here."
    )

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**By season**")
        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(x=seasonal_mape["season"], y=seasonal_mape["SARIMA_MAPE"], name="SARIMA", marker_color=COLOR_SARIMA))
        fig_season.add_trace(go.Bar(x=seasonal_mape["season"], y=seasonal_mape["LightGBM_MAPE"], name="LightGBM", marker_color=COLOR_LGB))
        fig_season.update_layout(barmode="group", yaxis_title="MAPE (%)", height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_season, use_container_width=True)
        st.dataframe(seasonal_mape, use_container_width=True, hide_index=True)

    with colB:
        st.markdown("**By temperature**")
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Bar(x=temp_mape["temp_bin"], y=temp_mape["SARIMA_MAPE"], name="SARIMA", marker_color=COLOR_SARIMA))
        fig_temp.add_trace(go.Bar(x=temp_mape["temp_bin"], y=temp_mape["LightGBM_MAPE"], name="LightGBM", marker_color=COLOR_LGB))
        fig_temp.update_layout(barmode="group", yaxis_title="MAPE (%)", height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_temp, use_container_width=True)
        st.dataframe(temp_mape, use_container_width=True, hide_index=True)

    st.markdown("**By hour of day**")
    fig_hour = go.Figure()
    fig_hour.add_trace(go.Scatter(x=hourly_mape["hour"], y=hourly_mape["SARIMA_MAPE"], name="SARIMA", mode="lines+markers", line=dict(color=COLOR_SARIMA)))
    fig_hour.add_trace(go.Scatter(x=hourly_mape["hour"], y=hourly_mape["LightGBM_MAPE"], name="LightGBM", mode="lines+markers", line=dict(color=COLOR_LGB)))
    fig_hour.update_layout(xaxis_title="Hour of day", yaxis_title="MAPE (%)", height=350, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_hour, use_container_width=True)

    st.info(
        "**Key takeaway:** SARIMA is the more reliable default in calm, stable "
        "conditions (winter, moderate temperatures, overnight hours). LightGBM's "
        "access to weather data gives it a real, substantial edge specifically "
        "during weather-driven demand extremes (monsoon, summer, and especially "
        "temperatures above 34°C, where its error is less than half of SARIMA's). "
        "See `notebooks/04_error_analysis.ipynb` and the README for the full analysis."
    )

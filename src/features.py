"""
Feature engineering for the Delhi load forecasting models.

All functions operate on a DataFrame with a DatetimeIndex named 'timestamp'
and a 'load_MW' column (the output of src/data_pipeline.py).
"""

import numpy as np
import pandas as pd


def add_lag_features(df: pd.DataFrame, lags: list[int] = (24, 48, 168)) -> pd.DataFrame:
    """
    Add lagged load values as features.

    Default lags: 24h (same hour yesterday), 48h (two days ago), 168h (same
    hour, same day, one week ago). Confirm these against your ACF/PACF plot —
    if a different lag showed a strong spike, add it here too.
    """
    df = df.copy()
    for lag in lags:
        df[f"load_lag_{lag}h"] = df["load_MW"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = (24, 168)) -> pd.DataFrame:
    """
    Add rolling mean/std of load over past windows (in hours).

    IMPORTANT: shift(1) before rolling, so the window only uses data strictly
    before the current timestamp — otherwise you leak the current (unknown-
    at-forecast-time) value into its own feature.

    min_periods is set to 1/4 of the window rather than pandas' default
    (the full window size). With the full-window default, a single real gap
    in load_MW (e.g. a multi-hour SLDC comms failure, ~2.71% of raw
    readings in this dataset) silently propagates into a full week's worth
    of NaN downstream via the 168h window — enough to wipe out an entire
    forecast fold's features and produce a garbage constant prediction.
    Allowing a partial window keeps the feature usable as long as *most*
    of the window has real data, which is a reasonable and statable
    trade-off given the raw data has real gaps.
    """
    df = df.copy()
    for w in windows:
        shifted = df["load_MW"].shift(1)
        min_p = max(1, w // 4)
        df[f"load_roll_mean_{w}h"] = shifted.rolling(w, min_periods=min_p).mean()
        df[f"load_roll_std_{w}h"] = shifted.rolling(w, min_periods=min_p).std()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encodings for hour-of-day and day-of-week, plus raw flags."""
    df = df.copy()
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(int)
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline and drop rows with NaNs from lag/rolling warm-up.

    IMPORTANT: only checks the newly engineered lag/rolling columns for NaN,
    not the whole dataframe. A blanket dropna() across every column is a
    trap here: 'holiday_name' is legitimately NaN on ~95% of days by design
    (only holidays have a name) — a blanket dropna() would silently discard
    nearly the entire dataset for a reason that has nothing to do with lag
    warm-up. This bug wiped out ~99% of rows in practice before being caught.
    """
    out = add_lag_features(df)
    out = add_rolling_features(out)
    out = add_calendar_features(out)

    engineered_cols = [c for c in out.columns if c.startswith("load_lag_") or c.startswith("load_roll_")]
    out = out.dropna(subset=engineered_cols)
    return out

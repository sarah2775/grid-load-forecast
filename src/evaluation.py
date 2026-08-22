"""
Walk-forward (expanding-window) backtesting harness.

Why not a single train/test split? A single split tells you how the model
does on one particular stretch of time — which might happen to be an easy
or hard period by chance. Walk-forward backtesting re-trains (or re-forecasts)
at multiple points in time and averages the error, giving a much more honest
estimate of real-world performance. This is the standard approach for time
series evaluation and is worth explaining explicitly in your report.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    fold_metrics: pd.DataFrame  # one row per fold: mape, rmse
    predictions: pd.DataFrame   # all folds' predictions concatenated, with actuals


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def walk_forward_folds(n_rows: int, initial_train_size: int, horizon: int, step: int):
    """
    Yield (train_idx_end, test_start, test_end) index positions for an
    expanding-window walk-forward split.

    - initial_train_size: rows in the first training window
    - horizon: how many rows ahead to forecast in each fold (e.g. 24 for a 24h-ahead test)
    - step: how many rows to advance between folds (e.g. 168 -> a new fold every week)
    """
    train_end = initial_train_size
    while train_end + horizon <= n_rows:
        yield train_end, train_end, train_end + horizon
        train_end += step


def run_backtest(df: pd.DataFrame, forecast_fn, target_col: str = "load_MW",
                  initial_train_size: int = 24 * 90, horizon: int = 24, step: int = 168) -> BacktestResult:
    """
    Generic walk-forward backtest.

    forecast_fn: a callable (train_df, test_df) -> np.ndarray of predictions
    for test_df's rows. This lets any model (naive, SARIMA, LightGBM, LSTM)
    plug into the same harness — implement forecast_fn once per model type.

    Defaults: 90 days initial training window, 24h forecast horizon, new
    fold every week (168h). Adjust step to trade off backtest thoroughness
    vs runtime — smaller step = more folds = slower but more robust estimate.
    """
    fold_rows = []
    pred_rows = []

    for fold_i, (train_end, test_start, test_end) in enumerate(
        walk_forward_folds(len(df), initial_train_size, horizon, step)
    ):
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]

        y_pred = forecast_fn(train_df, test_df)
        y_true = test_df[target_col].values

        fold_rows.append({
            "fold": fold_i,
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "mape": mape(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
        })

        pred_rows.append(pd.DataFrame({
            "timestamp": test_df.index,
            "y_true": y_true,
            "y_pred": y_pred,
            "fold": fold_i,
        }))

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    return BacktestResult(fold_metrics=fold_metrics, predictions=predictions)


def summarize(result: BacktestResult, model_name: str) -> dict:
    """Aggregate a BacktestResult into a single row for the model comparison table.

    Reports both mean and median MAPE/RMSE: the mean is standard but can be
    dominated by a single bad fold (seen in practice with SARIMA producing
    an occasional numerically unstable forecast); the median is a useful
    sanity check against that.
    """
    return {
        "model": model_name,
        "mean_mape": result.fold_metrics["mape"].mean(),
        "median_mape": result.fold_metrics["mape"].median(),
        "mean_rmse": result.fold_metrics["rmse"].mean(),
        "median_rmse": result.fold_metrics["rmse"].median(),
        "n_folds": len(result.fold_metrics),
    }

# Progress Log

## Week 1 — Data pipeline & EDA
Built the data pipeline merging Delhi SLDC load data (5-min resolution,
Apr 2023 – Jan 2026, sourced from a maintained Kaggle dataset scraped from
delhisldc.org) with Open-Meteo weather data and an Indian holiday calendar.
2.71% of raw readings were invalid/missing (likely SLDC comms link gaps),
handled via short-gap linear interpolation. Resampled to hourly for the v1
scope. Output: 24,432 hourly rows.

**Bug found and fixed:** `pandas.DataFrame.merge()` on a column key silently
discards the existing index and replaces it with a default `RangeIndex` —
this dropped the datetime timestamp entirely from the merged output without
raising any error, which only surfaced later as a confusing `TypeError`
during EDA plotting. Fixed by explicitly preserving the timestamp as a
column across the merge and restoring it as the index afterward. Added an
assertion (`isinstance(final_df.index, pd.DatetimeIndex)`) to the pipeline
so this failure mode can't silently reoccur.

EDA (`01_eda.ipynb`) confirmed:
- Series is stationary per ADF test (p < 0.05) — but this doesn't mean the
  strong daily/weekly seasonality goes away; stationarity and seasonality
  are separate properties, and SARIMA's seasonal order still needs to
  reflect the seasonality regardless.
- ACF shows dominant daily seasonality (repeating peaks at every 24h
  multiple: 24, 48, 72...168, 192). No distinct additional spike at the
  weekly lag (168h) beyond the general daily pattern — PACF confirms this,
  showing near-zero partial correlation at 168h once shorter lags are
  accounted for. This suggests day-of-week effects are better captured via
  calendar encodings (day-of-week, is_weekend) than via a raw weekly lag
  feature.

## Week 2 — Baselines & feature engineering
Built `src/features.py` (lag features at 24h/48h/168h, rolling mean/std with
a `shift(1)` guard against label leakage, cyclical hour/day-of-week/month
encodings) and `src/evaluation.py` (walk-forward/expanding-window backtest
harness shared by every model going forward — reports mean and median
MAPE/RMSE per model).

Implemented and backtested three baselines in `02_baselines.ipynb`:

| Model | Mean MAPE |
|---|---|
| SARIMA | ~4.5% |
| Seasonal Naive (168h) | ~9.5% |
| Naive | ~20% |

**Bug found and fixed:** initial SARIMA run produced a mean MAPE of ~1e95 —
one fold's forecast numerically diverged (a known SARIMAX failure mode when
`enforce_stationarity`/`enforce_invertibility` are relaxed), and since the
summary used a plain mean, that single exploded fold dominated the entire
result. Fixed by re-enabling stationarity/invertibility enforcement,
clipping forecasts to a physically sane range as a safety guard, and adding
median MAPE/RMSE alongside the mean in `summarize()` so a single bad fold
can never again silently hide inside an average.

**Interesting finding:** the seasonal-naive sample fold (last week of June)
showed actual load running consistently ~1,200 MW below its own prediction
from one week earlier, while the daily shape (trough ~6-8am, peak ~3pm)
stayed intact. Likely explanation: Delhi's monsoon onset typically arrives
in this window and causes a sharp AC-load drop within days — a real regime
shift that "look like last week" baselines fundamentally can't anticipate.
_(To confirm: check actual 2024 Delhi monsoon onset date against this drop.)_

Each baseline meaningfully beat the simpler one (SARIMA ~2x better than
seasonal naive, seasonal naive ~2x better than naive), which sets a
legitimate, non-trivial bar for the ML/DL models in Weeks 3-4 to clear.

## Week 3
_(next up)_

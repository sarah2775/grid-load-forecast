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

**Interesting finding:** the seasonal-naive sample fold (last week of June
2023) showed actual load running consistently ~1,200 MW below its own
prediction from one week earlier, while the daily shape (trough ~6-8am,
peak ~3pm) stayed intact. Likely explanation: Delhi's monsoon onset
typically arrives in this window and causes a sharp AC-load drop within
days — a real regime shift that "look like last week" baselines
fundamentally can't anticipate.

Each baseline meaningfully beat the simpler one (SARIMA ~2x better than
seasonal naive, seasonal naive ~2x better than naive), which sets a
legitimate, non-trivial bar for the ML/DL models in Weeks 3-4 to clear.

## Week 3 — Machine learning model (LightGBM) + a major pipeline bug
Built `03_ml_models.ipynb`: LightGBM using the features from `src/features.py`,
backtested on the same walk-forward harness as the baselines.

**Bug found and fixed (the big one):** `build_feature_set()`'s final
`.dropna()` was blanket-applied across every column, including
`holiday_name` — which is legitimately NaN on ~95% of days by design (only
holidays have a name). This silently wiped out ~99% of usable rows in
testing, not just the expected lag warm-up period. Confirmed via a targeted
reproduction test before fixing. Fixed by restricting the dropna to only
the newly engineered lag/rolling columns, leaving raw pass-through columns
alone. This was hiding underneath an earlier, smaller rolling-window fix
(`min_periods`, to tolerate partial windows around real data gaps) that
was real but insufficient on its own — a good reminder to verify a fix
against the actual data shape, not just the specific symptom that surfaced
first.

Also fixed a smaller issue where Open-Meteo's weather archive hadn't yet
finalized ~14 months of recent data at initial fetch time (confirmed via a
direct API query) — resolved by simply re-fetching once the archive had
caught up. And made the backtest harness itself more robust: a single
unforecastable fold (e.g. from a genuine data gap) is now skipped with a
logged warning instead of crashing the entire multi-fold run.

**Results (66/67 folds; 1 skipped at the very end of the data range, no
full forecast horizon remaining):**

| Model | Mean MAPE |
|---|---|
| SARIMA | 4.50% |
| LightGBM | 4.96% |
| Seasonal Naive | 9.50% |
| Naive | 20.00% |

LightGBM did not beat SARIMA despite access to richer features (weather,
holidays, multiple lags) — a legitimate finding, not a failure, suggesting
SARIMA's built-in seasonal structure is well-suited to this data. SHAP
analysis cross-checks cleanly against Week 1's EDA findings: `load_lag_24h`
dominance matches the ACF finding, temperature has a meaningful visible
impact matching the weather-correlation finding, and `is_holiday` sits at
near-zero SHAP impact — consistent with the "holiday effect was
noisy/inconclusive" finding from the EDA notebook.

## Week 4 — Error analysis (no new model, per scoping decision)
Rather than adding a fourth model, built `04_error_analysis.ipynb` to dig
into *when* SARIMA and LightGBM each perform better — a more valuable use
of remaining time than chasing marginal gains from a third architecture
(see README's "Is This Novel?" section for the reasoning behind this
choice).

**A real methodological issue found and fixed along the way:** initial
comparisons showed suspiciously small overlap between the two models'
backtest predictions. Root cause: SARIMA's per-fold data-gap handling used
an unlimited `.interpolate()` call, silently fabricating values through
gaps of *any* size, while LightGBM's feature pipeline correctly declined
to forecast when a gap was too large to responsibly fill. This meant the
two models were being evaluated under inconsistent honesty standards
around missing data — SARIMA looked more "successful" partly because it
was allowed to guess through problems LightGBM correctly refused to guess
through. Fixed by capping SARIMA's interpolation to the same gap-size limit
used elsewhere in the pipeline, and having it explicitly skip a fold rather
than fabricate through an unfillable gap — bringing both models under the
same standard. Also caught and fixed a silent bug where `np.clip()`'s
return value was never assigned back, meaning the numerical-stability guard
added in Week 2 had not actually been applied since it was written.

Also restructured the analysis to avoid comparing on an artificially small
matched-timestamp subset: seasonal/temperature/hour-of-day breakdowns use
each model's own full, independently valid backtest predictions (bigger,
more trustworthy samples), while only the head-to-head win-rate comparison
uses the smaller exact-timestamp-matched subset it actually requires.

**Key findings:**

- The full-year "SARIMA wins" headline (4.50% vs 4.96% MAPE) hides a more
  interesting split: LightGBM actually beats SARIMA in 2 of 3 seasons
  (Monsoon: 3.19% vs 4.26%; Summer: 4.21% vs 5.37%), and only loses in
  Winter/Other (4.75% vs 3.73%) — which, spanning 6 months vs 3, dominates
  the full-year average.
- Temperature-stratified error is the clearest finding: SARIMA's error
  nearly doubles in extreme heat (>34°C: 7.88% vs LightGBM's 3.26%),
  exactly where weather-awareness should matter most. SARIMA wins
  comfortably in calm, moderate conditions instead.
- Head-to-head: LightGBM was closer to the truth in 63.7% of paired hours
  despite having the worse average error overall — mean MAPE and win-rate
  measure genuinely different things (sensitivity to large misses vs.
  frequency of being closer), and both are worth reporting rather than
  picking one.
- **Conclusion**: SARIMA is the better default in calm/stable conditions;
  LightGBM's weather-awareness provides a real advantage during
  weather-driven demand extremes. A regime-switching or ensemble approach
  combining both is a plausible, well-motivated future extension —
  identified from data, not assumed in advance.

## Project status
Core modeling and analysis phase complete: two baselines, SARIMA,
LightGBM, and a seasonal/weather error analysis, all cross-validated
against each other's findings (EDA → SHAP → error analysis all tell a
consistent story). Remaining: Streamlit dashboard (`app/streamlit_app.py`,
currently a stub) and final report writing.

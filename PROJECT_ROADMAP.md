# Project Roadmap

Target: ~6 weeks, fits within a 1–2 month window before applications open.
Each week ends with a commit + a short entry in `reports/progress_log.md` (keeps your GitHub history meaningful, which reviewers do sometimes glance at).

## Week 1 — Data pipeline & EDA
- [ ] Download the Kaggle dataset (`prash4nt/delhi-sldc-load-data-5-min-resolution`) into `data/raw/`
- [ ] Inspect its actual columns/format and finish `load_raw_load_data()` in `src/data_pipeline.py`
- [ ] Decide: model total Delhi load, or break out by discom (BRPL/NDPL/BYPL/NDMC/MES)? Total is simpler and a fine v1 scope.
- [ ] Pull weather data from Open-Meteo for Delhi (28.6139, 77.2090) over the same date range
- [ ] Build Indian holiday/festival calendar feature (`holidays` package + manual Diwali/Holi/etc. additions)
- [ ] Merge into a clean `data/processed/load_weather.csv` (decide on resolution — 5-min may be more than you need; hourly resampling is a reasonable v1 choice)
- [ ] `notebooks/01_eda.ipynb`: seasonal decomposition (daily/weekly/annual), ACF/PACF, stationarity (ADF test), visualize the COVID lockdown demand shock
- [ ] Commit + push. Write 3-4 sentences in `progress_log.md` on what you found.

## Week 2 — Baselines & feature engineering
- [ ] Implement naive + seasonal naive forecasts
- [ ] Implement SARIMA/ETS (statsmodels or `pmdarima`)
- [ ] Build `src/features.py`: lag features, rolling means/stds, calendar encodings (cyclical sin/cos for hour/day-of-week), weather features
- [ ] Set up the walk-forward backtesting harness in `src/evaluation.py` — this is reused by every model from here on

## Week 3 — Machine learning models
- [ ] Train LightGBM/XGBoost with engineered features
- [ ] Hyperparameter tuning (Optuna, small budget is fine)
- [ ] Compare against Week 2 baselines using the same backtest harness
- [ ] SHAP analysis: what drives demand spikes?

## Week 4 — Deep learning models
- [ ] LSTM/GRU sequence model (PyTorch or Keras)
- [ ] N-BEATS or Temporal Fusion Transformer via `neuralforecast` or `darts`
- [ ] Add quantile/conformal prediction intervals
- [ ] Consolidate all model results into one comparison table

## Week 5 — Deployment
- [ ] Build `app/streamlit_app.py`: region selector, forecast horizon slider, plot with prediction intervals, SHAP explanation panel
- [ ] Deploy to Streamlit Community Cloud or Hugging Face Spaces (both free)
- [ ] Add live demo link to README

## Week 6 — Write-up & polish
- [ ] Write `reports/report.pdf` (4-6 pages, short-paper style: abstract, data, methods, results, discussion, limitations)
- [ ] Fill in the Results table in README
- [ ] Record a 30-second demo GIF for the README
- [ ] Clean commit history, tag a `v1.0` release
- [ ] Add a `LICENSE` (MIT is fine) and finish docstrings/type hints in `src/`

## Notes on making this "application-ready"
- **Limitations section matters.** Admissions readers (and interviewers) respect honesty about what didn't work as much as what did — e.g. if the deep learning model doesn't beat LightGBM, say so and hypothesize why. That's a stronger signal of research maturity than pretending everything worked.
- **Keep notebooks numbered and narrated** (markdown cells explaining *why*, not just code) — this is what people actually read when they click into your repo.
- **The report.pdf is not optional** — it's what you'll actually reference in your SOP and can attach to applications/interviews as a writing sample.
